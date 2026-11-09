from __future__ import annotations

import csv
import gc
import json
import time
import warnings
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]

FILTERED_PATH = (
    ROOT / "data" / "processed" /
    "cs_lg_2018_2024.jsonl"
)

PREPROCESSED_PATH = (
    ROOT / "data" / "processed" /
    "cs_lg_2018_2024_preprocessed.jsonl"
)

MATRIX_PATH = ROOT / "data" / "features" / "tfidf_matrix.npz"
VECTORIZER_PATH = ROOT / "data" / "features" / "tfidf_vectorizer.joblib"

OUTPUT_DIR = (
    ROOT / "outputs" / "models" /
    "final_full_corpus_nmf_k10"
)

EXPECTED_DOCUMENTS = 87945
TOPIC_COUNT = 10
RANDOM_SEED = 42
MAX_ITERATIONS = 300
TOP_TERMS = 20


def show_progress(step, total, label, started):
    width = 25
    fraction = step / total
    filled = int(width * fraction)
    bar = "=" * filled + "-" * (width - filled)
    elapsed = (time.perf_counter() - started) / 60

    print(
        f"[{bar}] {step}/{total} ({fraction * 100:.0f}%) "
        f"{label} | elapsed {elapsed:.2f} min",
        flush=True,
    )


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_and_validate_metadata():
    """
    Validate alignment between filtered and preprocessed data.
    Empty cleaned documents are retained in metadata.
    """
    metadata = []
    seen_ids = set()
    empty_token_ids = []

    year_counts = Counter()
    category_counts = Counter()

    print("\n[1/4] Validating dataset rows and metadata...", flush=True)

    with (
        FILTERED_PATH.open("r", encoding="utf-8") as filtered,
        PREPROCESSED_PATH.open("r", encoding="utf-8") as processed,
    ):
        pairs = zip_longest(filtered, processed)

        for row_number, pair in enumerate(pairs, start=1):
            raw_line, processed_line = pair

            if raw_line is None or processed_line is None:
                raise ValueError(
                    "Filtered and preprocessed files have different row counts."
                )

            raw = json.loads(raw_line)
            clean = json.loads(processed_line)

            paper_id = str(raw.get("id", "")).strip()
            processed_id = str(clean.get("id", "")).strip()

            if not paper_id or paper_id != processed_id:
                raise ValueError(
                    f"Row alignment error at record {row_number}: "
                    f"{paper_id!r} != {processed_id!r}"
                )

            if paper_id in seen_ids:
                raise ValueError(f"Duplicate paper ID: {paper_id}")

            seen_ids.add(paper_id)

            year = raw.get("submission_year")
            category = raw.get("primary_category")

            if year is None:
                raise ValueError(f"Missing submission year: {paper_id}")

            year = int(year)

            if year not in range(2018, 2025):
                raise ValueError(f"Unexpected year for {paper_id}: {year}")

            if category != "cs.LG":
                raise ValueError(
                    f"Unexpected primary category for {paper_id}: {category}"
                )

            if clean.get("primary_category") != category:
                raise ValueError(
                    f"Category mismatch between files: {paper_id}"
                )

            tokens = clean.get("tokens")
            has_clean_tokens = bool(tokens)

            if not has_clean_tokens:
                empty_token_ids.append(paper_id)

            metadata.append({
                "row_index": row_number - 1,
                "arxiv_id": paper_id,
                "year": year,
                "submitted_date": raw.get("submitted_date", ""),
                "primary_category": category,
                "has_clean_tokens": has_clean_tokens,
            })

            year_counts[year] += 1
            category_counts[category] += 1

            if row_number % 20000 == 0:
                print(
                    f"  Validated {row_number:,}/{EXPECTED_DOCUMENTS:,} records",
                    flush=True,
                )

    if len(metadata) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"Expected {EXPECTED_DOCUMENTS:,} records; "
            f"found {len(metadata):,}."
        )

    print(f"  Records validated: {len(metadata):,}")
    print(f"  Empty cleaned-token records: {len(empty_token_ids)}")
    print(f"  Their IDs: {empty_token_ids}")
    print(f"  Year distribution: {dict(sorted(year_counts.items()))}")
    print(f"  Primary categories: {dict(category_counts)}")

    return metadata, empty_token_ids


def main():
    started = time.perf_counter()

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {OUTPUT_DIR}"
        )

    for path in (
        FILTERED_PATH,
        PREPROCESSED_PATH,
        MATRIX_PATH,
        VECTORIZER_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    metadata, empty_token_ids = read_and_validate_metadata()
    show_progress(1, 4, "Metadata validated", started)

    print("\n[2/4] Loading full-corpus TF-IDF matrix...", flush=True)

    matrix = load_npz(MATRIX_PATH).tocsr()
    vectorizer = joblib.load(VECTORIZER_PATH)
    feature_names = vectorizer.get_feature_names_out()

    if matrix.shape != (len(metadata), len(feature_names)):
        raise ValueError(
            f"Matrix dimensions {matrix.shape} do not match "
            f"metadata/features {(len(metadata), len(feature_names))}."
        )

    # Identify rows that contain at least one usable vocabulary feature.
    nonzero_per_row = np.asarray(matrix.getnnz(axis=1)).reshape(-1)
    active_rows = nonzero_per_row > 0

    # A paper with no cleaned tokens must not have a nonzero feature row.
    empty_token_mask = np.array(
        [not row["has_clean_tokens"] for row in metadata],
        dtype=bool,
    )

    if np.any(active_rows & empty_token_mask):
        raise ValueError(
            "A paper with an empty token list has a nonzero TF-IDF row. "
            "The source files and feature matrix may be misaligned."
        )

    active_indices = np.flatnonzero(active_rows)
    active_matrix = matrix[active_rows]

    zero_feature_indices = np.flatnonzero(~active_rows)

    print(f"  Complete matrix: {matrix.shape}")
    print(f"  Non-zero feature documents: {len(active_indices):,}")
    print(f"  Zero-feature documents: {len(zero_feature_indices):,}")
    print(f"  Sparse matrix entries: {matrix.nnz:,}")

    if len(active_indices) == 0:
        raise ValueError("No documents have usable TF-IDF features.")

    show_progress(2, 4, "Feature matrix checked", started)

    print("\n[3/4] Fitting NMF K=10 on usable full-corpus documents...", flush=True)
    print(f"  Documents entering NMF: {active_matrix.shape[0]:,}")
    print(f"  Features: {active_matrix.shape[1]:,}")
    print(f"  Seed: {RANDOM_SEED}")
    print(f"  Maximum iterations: {MAX_ITERATIONS}")
    print(
        "  Note: sklearn does not expose a reliable live iteration progress bar; "
        "the next update appears when fitting finishes.",
        flush=True,
    )

    model = NMF(
        n_components=TOPIC_COUNT,
        init="random",
        solver="cd",
        max_iter=MAX_ITERATIONS,
        random_state=RANDOM_SEED,
        tol=1e-4,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        active_weights = model.fit_transform(active_matrix)

    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning)
        for item in captured
    )

    active_weights = np.asarray(active_weights, dtype=np.float32)

    # Some rows may have no usable topic activation. Keep them unassigned.
    activation_sums = active_weights.sum(axis=1)
    local_modelled = activation_sums > 0

    modelled_indices = active_indices[local_modelled]
    modelled_weights = active_weights[local_modelled]
    modelled_sums = activation_sums[local_modelled]

    if len(modelled_indices) == 0:
        raise ValueError("NMF produced no positive document-topic activations.")

    # Full-corpus arrays preserve the original document row order.
    document_weights = np.zeros(
        (len(metadata), TOPIC_COUNT),
        dtype=np.float32,
    )

    document_proportions = np.zeros_like(document_weights)

    modelled_mask = np.zeros(len(metadata), dtype=bool)

    document_weights[modelled_indices] = modelled_weights

    document_proportions[modelled_indices] = (
        modelled_weights / modelled_sums[:, None]
    )

    modelled_mask[modelled_indices] = True

    if not np.isfinite(document_proportions).all():
        raise ValueError("Non-finite document-topic proportions detected.")

    if not np.allclose(
        document_proportions[modelled_mask].sum(axis=1),
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Modelled document-topic proportions do not sum to one."
        )

    dominant_topics = np.zeros(len(metadata), dtype=np.int16)
    dominant_shares = np.full(len(metadata), np.nan, dtype=np.float32)

    dominant_topics[modelled_indices] = (
        np.argmax(document_proportions[modelled_indices], axis=1) + 1
    )

    dominant_shares[modelled_indices] = (
        np.max(document_proportions[modelled_indices], axis=1)
    )

    print(f"  NMF iterations: {model.n_iter_}")
    print(f"  Training reconstruction error: {model.reconstruction_err_:.6f}")
    print(f"  Convergence warning: {convergence_warning}")
    print(f"  Documents assigned to topics: {modelled_mask.sum():,}")
    print(
        f"  Documents retained but unassigned: "
        f"{(~modelled_mask).sum():,}"
    )

    show_progress(3, 4, "Full-corpus model fitted", started)

    print("\n[4/4] Saving model and full-corpus outputs...", flush=True)

    # Create the output folder only after successful model fitting and validation.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    model_path = OUTPUT_DIR / "final_nmf_k10_seed42.joblib"
    joblib.dump(model, model_path, compress=3)

    np.save(
        OUTPUT_DIR / "document_topic_proportions.npy",
        document_proportions,
    )

    np.save(
        OUTPUT_DIR / "document_topic_weights.npy",
        document_weights,
    )

    np.save(
        OUTPUT_DIR / "modelled_document_mask.npy",
        modelled_mask,
    )

    metadata_fields = [
        "row_index",
        "arxiv_id",
        "year",
        "submitted_date",
        "primary_category",
        "has_clean_tokens",
        "is_modelled",
    ]

    metadata_rows = []

    for index, row in enumerate(metadata):
        output_row = dict(row)
        output_row["is_modelled"] = bool(modelled_mask[index])
        metadata_rows.append(output_row)

    write_csv(
        OUTPUT_DIR / "corpus_metadata.csv",
        metadata_rows,
        metadata_fields,
    )

    topic_rows = []

    for topic_index, component in enumerate(model.components_, start=1):
        top_indices = np.argsort(component)[::-1][:TOP_TERMS]

        for rank, feature_index in enumerate(top_indices, start=1):
            topic_rows.append({
                "topic_number": topic_index,
                "rank": rank,
                "term": str(feature_names[feature_index]),
                "weight": float(component[feature_index]),
            })

    topic_terms_path = OUTPUT_DIR / "final_nmf_k10_topic_terms.csv"

    write_csv(
        topic_terms_path,
        topic_rows,
        ["topic_number", "rank", "term", "weight"],
    )

    assignment_rows = []

    for index, row in enumerate(metadata):
        is_modelled = bool(modelled_mask[index])

        assignment_rows.append({
            "row_index": row["row_index"],
            "arxiv_id": row["arxiv_id"],
            "year": row["year"],
            "is_modelled": is_modelled,
            "dominant_topic": (
                int(dominant_topics[index]) if is_modelled else ""
            ),
            "dominant_topic_share": (
                float(dominant_shares[index]) if is_modelled else ""
            ),
        })

    assignment_path = OUTPUT_DIR / "document_dominant_topics.csv"

    write_csv(
        assignment_path,
        assignment_rows,
        [
            "row_index",
            "arxiv_id",
            "year",
            "is_modelled",
            "dominant_topic",
            "dominant_topic_share",
        ],
    )

    year_counts = Counter(row["year"] for row in metadata)

    summary = {
        "phase": "Phase 10A - final full-corpus NMF K=10",
        "status": (
            "completed_with_convergence_warning"
            if convergence_warning
            else "completed"
        ),
        "model": "NMF",
        "topic_count": TOPIC_COUNT,
        "random_seed": RANDOM_SEED,
        "initialisation": "random",
        "maximum_iterations": MAX_ITERATIONS,
        "actual_iterations": int(model.n_iter_),
        "convergence_warning": bool(convergence_warning),
        "training_reconstruction_error": float(model.reconstruction_err_),
        "total_documents_in_source": len(metadata),
        "nonzero_feature_documents": int(len(active_indices)),
        "zero_feature_documents": int(len(zero_feature_indices)),
        "documents_assigned_to_topics": int(modelled_mask.sum()),
        "documents_retained_but_unassigned": int((~modelled_mask).sum()),
        "features": int(matrix.shape[1]),
        "nonzero_matrix_entries": int(matrix.nnz),
        "document_topic_proportions_shape": list(document_proportions.shape),
        "year_counts": {
            str(year): int(count)
            for year, count in sorted(year_counts.items())
        },
        "methodological_note": (
            "The model was fitted on documents with at least one nonzero TF-IDF "
            "feature. Zero-feature documents remain in metadata and in the "
            "full output arrays, with all-zero topic proportions and no "
            "dominant-topic assignment. The model is descriptive full-corpus "
            "training; its reconstruction error is not a held-out validation score."
        ),
        "files": [
            model_path.name,
            "document_topic_proportions.npy",
            "document_topic_weights.npy",
            "modelled_document_mask.npy",
            "corpus_metadata.csv",
            topic_terms_path.name,
            assignment_path.name,
        ],
    }

    summary_path = OUTPUT_DIR / "final_model_summary.json"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    elapsed = (time.perf_counter() - started) / 60.0

    print("\n============================================================")
    print("PHASE 10A COMPLETED")
    print("============================================================")
    print(f"Total corpus documents: {len(metadata):,}")
    print(f"Documents modelled: {modelled_mask.sum():,}")
    print(f"Documents retained but unassigned: {(~modelled_mask).sum():,}")
    print(f"Topics: {TOPIC_COUNT}")
    print(f"Iterations: {model.n_iter_}")
    print(f"Convergence warning: {convergence_warning}")
    print(f"Elapsed time: {elapsed:.2f} minutes")
    print(f"\nOutput folder: {OUTPUT_DIR}")
    print(f"Topic terms: {topic_terms_path}")
    print(f"Assignments: {assignment_path}")

    print("\nFULL-CORPUS TOPIC TERMS:")

    for topic in range(1, TOPIC_COUNT + 1):
        terms = [
            row["term"]
            for row in topic_rows
            if row["topic_number"] == topic
        ]
        print(f"Topic {topic}: {', '.join(terms[:12])}")

    show_progress(4, 4, "Outputs saved and validated", started)

    print("\nReview convergence status before using this model for trend analysis.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
