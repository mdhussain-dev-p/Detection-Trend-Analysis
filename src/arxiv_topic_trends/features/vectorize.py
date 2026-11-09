"""Create sparse text representations for NMF and LDA."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


MAX_FEATURES = 8000
MIN_DOCUMENT_FREQUENCY = 5
MAX_DOCUMENT_FREQUENCY = 0.90

DOMAIN_STOPWORD_CANDIDATES = {
    "paper", "propose", "proposed", "show", "shown",
    "result", "method", "based", "approach", "using",
    "work", "however", "framework", "present",
    "presented", "demonstrate", "provide", "provides",
    "study", "described", "discuss", "introduce",
}


def iter_documents(input_path: Path):
    """Read cleaned token lists one document at a time."""

    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                paper = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

            tokens = paper.get("tokens")

            if not isinstance(tokens, list):
                raise ValueError(
                    f"Invalid tokens on line {line_number}"
                )

            if any(not isinstance(token, str) for token in tokens):
                raise ValueError(
                    f"Non-text token on line {line_number}"
                )

            yield " ".join(tokens)


def create_vectorizers(
    *,
    min_df: int = MIN_DOCUMENT_FREQUENCY,
    max_df: float = MAX_DOCUMENT_FREQUENCY,
    max_features: int = MAX_FEATURES,
):
    """Use shared vocabulary rules with configurable test settings."""

    settings = {
        "lowercase": False,
        "tokenizer": str.split,
        "token_pattern": None,
        "preprocessor": None,
        "stop_words": None,
        "ngram_range": (1, 1),
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
    }

    tfidf = TfidfVectorizer(
        **settings,
        dtype=np.float32,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )

    counts = CountVectorizer(
        **settings,
        dtype=np.int32,
    )

    return tfidf, counts


def save_domain_review(top_terms_path: Path, output_path: Path):
    """Flag candidate academic boilerplate for human review."""

    with top_terms_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as source:

        reader = csv.DictReader(source)

        with output_path.open(
            "w", newline="", encoding="utf-8"
        ) as target:

            writer = csv.writer(target)

            writer.writerow([
                "term",
                "frequency",
                "suggested_review",
                "decision",
                "notes",
            ])

            for row in reader:
                term = row["term"]
                candidate = term in DOMAIN_STOPWORD_CANDIDATES

                writer.writerow([
                    term,
                    row["frequency"],
                    "YES" if candidate else "NO",
                    "PENDING",
                    "Human review required before removal."
                    if candidate else "No removal suggested."
                ])


def save_vectorization(input_path: Path, output_dir: Path) -> dict:
    """Generate TF-IDF and raw-count matrices in separate passes."""

    report_path = Path(
        "outputs/reports/nlp_preprocessing_summary.json"
    )

    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    summary = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    expected_documents = summary["records_processed"]

    top_terms_path = Path(
        "outputs/metrics/top_terms_after_cleaning.csv"
    )

    if not top_terms_path.is_file():
        raise FileNotFoundError(top_terms_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    tfidf_path = output_dir / "tfidf_matrix.npz"
    count_path = output_dir / "count_matrix.npz"

    tfidf_model_path = output_dir / "tfidf_vectorizer.joblib"
    count_model_path = output_dir / "count_vectorizer.joblib"

    vocabulary_path = output_dir / "feature_vocabulary.csv"
    manifest_path = output_dir / "vectorization_manifest.json"
    review_path = output_dir / "domain_stopwords_review.csv"

    all_outputs = [
        tfidf_path, count_path,
        tfidf_model_path, count_model_path,
        vocabulary_path, manifest_path, review_path
    ]

    for path in all_outputs:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    tfidf, counts = create_vectorizers()

    print("\n[TF-IDF] Building matrix for NMF...", flush=True)

    tfidf_matrix = tfidf.fit_transform(
        iter_documents(input_path)
    )

    if tfidf_matrix.shape[0] != expected_documents:
        raise ValueError("Incorrect TF-IDF document count.")

    if tfidf_matrix.shape[1] == 0:
        raise ValueError("Empty TF-IDF vocabulary.")

    vocabulary = tfidf.get_feature_names_out().copy()

    tfidf_metrics = {
        "documents": int(tfidf_matrix.shape[0]),
        "features": int(tfidf_matrix.shape[1]),
        "nonzero_values": int(tfidf_matrix.nnz),
        "density": float(
            tfidf_matrix.nnz /
            (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
        ),
        "dtype": str(tfidf_matrix.dtype),
    }

    save_npz(tfidf_path, tfidf_matrix, compressed=True)
    joblib.dump(tfidf, tfidf_model_path, compress=3)

    print("TF-IDF matrix saved.")

    del tfidf_matrix, tfidf
    gc.collect()

    print("\n[COUNT] Building matrix for LDA...", flush=True)

    count_matrix = counts.fit_transform(
        iter_documents(input_path)
    )

    if count_matrix.shape[0] != expected_documents:
        raise ValueError("Incorrect count-matrix document count.")

    count_vocabulary = counts.get_feature_names_out()

    if not np.array_equal(vocabulary, count_vocabulary):
        raise ValueError("The two model vocabularies do not match.")

    count_metrics = {
        "documents": int(count_matrix.shape[0]),
        "features": int(count_matrix.shape[1]),
        "nonzero_values": int(count_matrix.nnz),
        "density": float(
            count_matrix.nnz /
            (count_matrix.shape[0] * count_matrix.shape[1])
        ),
        "dtype": str(count_matrix.dtype),
    }

    save_npz(count_path, count_matrix, compressed=True)
    joblib.dump(counts, count_model_path, compress=3)

    print("Count matrix saved.")

    del count_matrix, counts
    gc.collect()

    # Save the common vocabulary.
    with vocabulary_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:

        writer = csv.writer(target)
        writer.writerow(["feature_index", "term"])

        for index, term in enumerate(vocabulary):
            writer.writerow([index, term])

    # Generate review candidates without removing them.
    save_domain_review(top_terms_path, review_path)

    result = {
        "dataset_version": summary["dataset_version"],
        "records_processed": expected_documents,
        "nmf_input": "TF-IDF",
        "lda_input": "Raw document-term counts",
        "parameters": {
            "maximum_features": MAX_FEATURES,
            "minimum_document_frequency": MIN_DOCUMENT_FREQUENCY,
            "maximum_document_frequency": MAX_DOCUMENT_FREQUENCY,
            "ngram_range": [1, 1],
            "tfidf_dtype": "float32",
            "count_dtype": "int32",
            "sublinear_tf": True,
        },
        "tfidf": tfidf_metrics,
        "counts": count_metrics,
        "vocabularies_match": True,
        "domain_stopwords_automatically_removed": [],
        "software": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "files": [str(path) for path in all_outputs],
    }

    encoded = json.dumps(result, indent=2) + "\n"

    manifest_path.write_text(encoded, encoding="utf-8")

    Path("outputs/reports/vectorization_summary.json").write_text(
        encoded, encoding="utf-8"
    )

    print("\n========== VECTORISATION RESULTS ==========")
    print(json.dumps(result, indent=2))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()

    save_vectorization(args.input, args.output_dir)


if __name__ == "__main__":
    main()
