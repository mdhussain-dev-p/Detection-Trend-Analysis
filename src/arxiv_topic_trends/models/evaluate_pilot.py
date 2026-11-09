"""Evaluate NMF and LDA using a reproducible held-out document sample."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

from scipy.sparse import load_npz
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from arxiv_topic_trends.models.train_pilot import extract_top_terms


def split_sample(sample_rows, validation_fraction=0.20, seed=2026):
    """Split the pilot rows into disjoint training and validation sets."""

    rows = np.asarray(sample_rows)

    if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise ValueError("Sample rows must be unique one-dimensional indexes.")

    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one.")

    if len(rows) < 2:
        raise ValueError("At least two documents are required.")

    generator = np.random.default_rng(seed)
    positions = generator.permutation(len(rows))

    validation_count = max(1, round(len(rows) * validation_fraction))

    validation_rows = np.sort(rows[positions[:validation_count]])
    training_rows = np.sort(rows[positions[validation_count:]])

    return training_rows, validation_rows


def sparse_reconstruction_error(matrix, document_topics, components):
    """Calculate held-out Frobenius reconstruction error without densifying X."""

    projection = np.asarray(matrix @ components.T)

    components_gram = components @ components.T

    reconstructed_norm = np.sum(
        document_topics * (document_topics @ components_gram)
    )

    cross_product = np.sum(document_topics * projection)

    original_norm = float(matrix.multiply(matrix).sum())

    squared_error = (
        original_norm + reconstructed_norm - 2.0 * cross_product
    )

    return float(np.sqrt(max(float(squared_error), 0.0)))


def load_vocabulary(path):
    terms = []

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)

        for index, row in enumerate(reader):
            if int(row["feature_index"]) != index:
                raise ValueError("Vocabulary indexing is inconsistent.")

            terms.append(row["term"])

    return np.asarray(terms)


def run_validation(feature_dir, pilot_dir, output_dir):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Validation output directory is not empty.")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (feature_dir / "vectorization_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    sample_rows = np.load(
        pilot_dir / "sample_row_indices.npy",
        allow_pickle=False,
    )

    total_documents = manifest["records_processed"]
    vocabulary = load_vocabulary(
        feature_dir / "feature_vocabulary.csv"
    )

    if len(sample_rows) != 8000:
        raise ValueError("Unexpected pilot sample size.")

    if np.any(sample_rows < 0) or np.any(sample_rows >= total_documents):
        raise ValueError("Pilot sample contains invalid document indexes.")

    train_rows, validation_rows = split_sample(sample_rows)

    np.save(output_dir / "training_row_indices.npy", train_rows)
    np.save(output_dir / "validation_row_indices.npy", validation_rows)

    results = {
        "dataset_version": manifest["dataset_version"],
        "total_corpus_documents": total_documents,
        "pilot_sample_documents": len(sample_rows),
        "training_documents": len(train_rows),
        "validation_documents": len(validation_rows),
        "validation_fraction": 0.20,
        "split_seed": 2026,
        "topic_count": 8,
        "models": {},
    }

    topic_terms = []

    # NMF: train on training rows and evaluate on held-out rows.
    print("\n[NMF] Loading sparse TF-IDF data...", flush=True)

    tfidf = load_npz(feature_dir / "tfidf_matrix.npz")

    if tfidf.shape != (total_documents, len(vocabulary)):
        raise ValueError("TF-IDF dimensions do not match the manifest.")

    train_x = tfidf[train_rows]
    validation_x = tfidf[validation_rows]

    del tfidf
    gc.collect()

    nmf = NMF(
        n_components=8,
        init="nndsvda",
        solver="cd",
        max_iter=200,
        tol=0.001,
        random_state=42,
    )

    print(
        f"[NMF] Training on {len(train_rows):,} documents...",
        flush=True,
    )

    with warnings.catch_warnings(record=True) as nmf_warnings:
        warnings.simplefilter("always", ConvergenceWarning)

        with threadpool_limits(limits=1):
            train_topics = nmf.fit_transform(train_x)

    nmf_converged = not any(
        issubclass(item.category, ConvergenceWarning)
        for item in nmf_warnings
    )

    validation_topics = nmf.transform(validation_x)

    heldout_error = sparse_reconstruction_error(
        validation_x,
        validation_topics,
        nmf.components_,
    )

    topic_terms.extend(
        extract_top_terms(
            nmf.components_,
            vocabulary,
            "NMF",
        )
    )

    nmf_error = float(nmf.reconstruction_err_)
    nmf_iterations = int(nmf.n_iter_)

    joblib.dump(
        nmf,
        output_dir / "nmf_k8_seed42.joblib",
        compress=3,
    )

    results["models"]["NMF"] = {
        "input": "TF-IDF",
        "iterations": nmf_iterations,
        "iteration_limit": 200,
        "converged": nmf_converged,
        "training_reconstruction_error": nmf_error,
        "validation_reconstruction_error": heldout_error,
        "validation_documents": len(validation_rows),
    }

    print(f"NMF converged: {nmf_converged}")
    print(f"NMF iterations: {nmf_iterations}")
    print(f"Held-out reconstruction error: {heldout_error:.4f}")

    del nmf, train_topics, validation_topics, train_x, validation_x
    gc.collect()

    # LDA: raw counts only.
    print("\n[LDA] Loading raw-count data...", flush=True)

    count_matrix = load_npz(feature_dir / "count_matrix.npz")

    if count_matrix.shape != (total_documents, len(vocabulary)):
        raise ValueError("Count matrix dimensions do not match.")

    train_x = count_matrix[train_rows]
    validation_x = count_matrix[validation_rows]

    del count_matrix
    gc.collect()

    lda = LatentDirichletAllocation(
        n_components=8,
        learning_method="online",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=10,
        batch_size=128,
        evaluate_every=-1,
        n_jobs=1,
        random_state=42,
    )

    print(
        f"[LDA] Training on {len(train_rows):,} documents...",
        flush=True,
    )

    with threadpool_limits(limits=1):
        lda.fit(train_x)

        # Evaluation uses documents not used for fitting.
        heldout_perplexity = float(lda.perplexity(validation_x))

    topic_terms.extend(
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
        "iterations": int(lda.n_iter_),
        "iteration_limit": 10,
        "heldout_perplexity": heldout_perplexity,
        "validation_documents": len(validation_rows),
        "learning_method": "online",
    }

    print(f"LDA held-out perplexity: {heldout_perplexity:.4f}")

    # Persist results.
    with (
        (output_dir / "validation_topic_terms.csv").open(
            "w", newline="", encoding="utf-8"
        )
    ) as target:

        writer = csv.DictWriter(
            target,
            fieldnames=["model", "topic", "rank", "term", "weight"],
        )

        writer.writeheader()
        writer.writerows(topic_terms)

    results["software"] = {
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

    results["interpretation_note"] = (
        "This is a random 80/20 validation split of the 8,000-document "
        "pilot sample. It is not a temporal holdout or final model selection."
    )

    results_path = output_dir / "validation_summary.json"

    results_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n========== HELD-OUT VALIDATION RESULTS ==========")
    print(json.dumps(results, indent=2))
    print("\nValidation experiment completed.")

    del lda, train_x, validation_x
    gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    run_validation(args.features, args.pilot, args.output)


if __name__ == "__main__":
    main()
