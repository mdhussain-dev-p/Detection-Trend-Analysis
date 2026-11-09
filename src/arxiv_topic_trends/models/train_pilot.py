"""Train a reproducible NMF and LDA pilot on a shared document sample."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

from scipy.sparse import load_npz
from sklearn.decomposition import NMF, LatentDirichletAllocation
from threadpoolctl import threadpool_limits


SAMPLE_SIZE = 8000
TOPIC_COUNT = 8
RANDOM_STATE = 42
TOP_WORDS = 15


def choose_sample_indices(
    document_count: int,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Choose reproducible, unique document rows."""

    if document_count <= 0:
        raise ValueError("Document count must be positive.")

    selected_size = min(document_count, sample_size)

    generator = np.random.default_rng(random_state)

    indices = generator.choice(
        document_count,
        size=selected_size,
        replace=False,
    )

    return np.sort(indices)


def extract_top_terms(
    components: np.ndarray,
    feature_names: np.ndarray,
    model_name: str,
    top_n: int = TOP_WORDS,
) -> list[dict]:
    """Return the highest-weight terms from each topic."""

    if components.shape[1] != len(feature_names):
        raise ValueError("Component and vocabulary dimensions do not match.")

    results = []

    for topic_number, component in enumerate(components, start=1):
        top_indices = np.argsort(component)[-top_n:][::-1]

        for rank, index in enumerate(top_indices, start=1):
            results.append({
                "model": model_name,
                "topic": topic_number,
                "rank": rank,
                "term": str(feature_names[index]),
                "weight": float(component[index]),
            })

    return results


def load_vocabulary(path: Path) -> np.ndarray:
    """Read vocabulary in the exact saved feature-index order."""

    terms = []

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)

        for expected_index, row in enumerate(reader):
            if int(row["feature_index"]) != expected_index:
                raise ValueError("Vocabulary indexes are not consecutive.")

            terms.append(row["term"])

    if not terms:
        raise ValueError("Vocabulary is empty.")

    return np.asarray(terms)


def run_pilot(feature_dir: Path, output_dir: Path) -> dict:
    """Fit NMF and LDA sequentially to limit memory usage."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Pilot output directory is not empty.")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (feature_dir / "vectorization_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    vocabulary = load_vocabulary(
        feature_dir / "feature_vocabulary.csv"
    )

    expected_documents = manifest["records_processed"]
    expected_features = manifest["parameters"]["maximum_features"]

    if len(vocabulary) != expected_features:
        raise ValueError("Vocabulary size differs from the feature manifest.")

    indices = choose_sample_indices(
        expected_documents,
        SAMPLE_SIZE,
        RANDOM_STATE,
    )

    sample_hash = hashlib.sha256(
        indices.tobytes()
    ).hexdigest()

    np.save(
        output_dir / "sample_row_indices.npy",
        indices,
    )

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": manifest["dataset_version"],
        "total_documents": expected_documents,
        "pilot_documents": len(indices),
        "features": len(vocabulary),
        "topics": TOPIC_COUNT,
        "random_state": RANDOM_STATE,
        "sample_indices_sha256": sample_hash,
        "models": {},
    }

    all_terms = []

    # Train NMF first and release memory before loading the count matrix.
    print("\n[NMF] Loading TF-IDF matrix...", flush=True)

    tfidf_matrix = load_npz(feature_dir / "tfidf_matrix.npz")

    if tfidf_matrix.shape != (expected_documents, len(vocabulary)):
        raise ValueError("TF-IDF matrix shape is incorrect.")

    tfidf_sample = tfidf_matrix[indices]

    del tfidf_matrix
    gc.collect()

    print(
        f"[NMF] Training on {tfidf_sample.shape[0]:,} documents...",
        flush=True,
    )

    nmf = NMF(
        n_components=TOPIC_COUNT,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=60,
        tol=0.001,
        random_state=RANDOM_STATE,
    )

    with threadpool_limits(limits=1):
        nmf.fit_transform(tfidf_sample)

    all_terms.extend(
        extract_top_terms(
            nmf.components_,
            vocabulary,
            "NMF",
        )
    )

    nmf_reconstruction_error = float(nmf.reconstruction_err_)
    nmf_iterations = int(nmf.n_iter_)

    joblib.dump(
        nmf,
        output_dir / "nmf_k8_seed42.joblib",
        compress=3,
    )

    results["models"]["NMF"] = {
        "input": "TF-IDF",
        "topics": TOPIC_COUNT,
        "random_state": RANDOM_STATE,
        "iterations": nmf_iterations,
        "maximum_iterations": 60,
        "reconstruction_error": nmf_reconstruction_error,
        "training_documents": len(indices),
        "converged_within_iteration_limit": nmf_iterations < 60,
    }

    del nmf, tfidf_sample
    gc.collect()

    # Train LDA separately.
    print("\n[LDA] Loading raw-count matrix...", flush=True)

    count_matrix = load_npz(feature_dir / "count_matrix.npz")

    if count_matrix.shape != (expected_documents, len(vocabulary)):
        raise ValueError("Count matrix shape is incorrect.")

    count_sample = count_matrix[indices]

    del count_matrix
    gc.collect()

    print(
        f"[LDA] Training on {count_sample.shape[0]:,} documents...",
        flush=True,
    )

    lda = LatentDirichletAllocation(
        n_components=TOPIC_COUNT,
        doc_topic_prior=None,
        topic_word_prior=None,
        learning_method="online",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=5,
        batch_size=128,
        evaluate_every=-1,
        max_doc_update_iter=100,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )

    with threadpool_limits(limits=1):
        lda.fit(count_sample)
        lda_perplexity = float(lda.perplexity(count_sample))

    all_terms.extend(
        extract_top_terms(
            lda.components_,
            vocabulary,
            "LDA",
        )
    )

    joblib.dump(
        lda,
        output_dir / "lda_k8_seed42.joblib",
        compress=3,
    )

    results["models"]["LDA"] = {
        "input": "Raw document-term counts",
        "topics": TOPIC_COUNT,
        "random_state": RANDOM_STATE,
        "iterations": int(lda.n_iter_),
        "maximum_iterations": 5,
        "perplexity_on_training_sample": lda_perplexity,
        "training_documents": len(indices),
        "learning_method": "online",
        "batch_size": 128,
    }

    # Save interpretable topic terms.
    terms_path = output_dir / "pilot_topic_terms.csv"

    with terms_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:

        writer = csv.DictWriter(
            target,
            fieldnames=["model", "topic", "rank", "term", "weight"],
        )

        writer.writeheader()
        writer.writerows(all_terms)

    results["software"] = {
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

    results["important_note"] = (
        "Pilot experiment using one random sample, eight topics and "
        "one random seed. Results are preliminary and are not "
        "the final model-selection results."
    )

    report_path = output_dir / "pilot_summary.json"

    report_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n========== PILOT MODEL RESULTS ==========")
    print(json.dumps(results, indent=2))

    print("\nTop topic terms:")
    for entry in all_terms:
        if entry["rank"] <= 10:
            print(
                f"{entry['model']} | Topic {entry['topic']} | "
                f"{entry['term']}"
            )

    print("\nPilot models and reports saved successfully.")

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    run_pilot(args.features, args.output)


if __name__ == "__main__":
    main()
