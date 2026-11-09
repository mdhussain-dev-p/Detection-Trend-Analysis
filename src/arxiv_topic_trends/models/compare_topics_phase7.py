from __future__ import annotations

import csv
import gc
import json
import math
import warnings
from pathlib import Path
from statistics import mean, stdev

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import train_test_split
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


ROOT = Path(__file__).resolve().parents[3]

TFIDF_PATH = ROOT / "data" / "features" / "tfidf_matrix.npz"
COUNT_PATH = ROOT / "data" / "features" / "count_matrix.npz"
VECTORIZER_PATH = ROOT / "data" / "features" / "tfidf_vectorizer.joblib"
PREPROCESSED_PATH = (
    ROOT / "data" / "processed" /
    "cs_lg_2018_2024_preprocessed.jsonl"
)
SAMPLE_PATH = ROOT / "outputs" / "models" / "pilot" / "sample_row_indices.npy"
OUTPUT_DIR = ROOT / "outputs" / "models" / "topic_search"

TOPIC_COUNTS = (5, 8, 10, 12)
NMF_SEEDS = (42, 13, 21, 34, 55)
TOP_N_TERMS = 15
NMF_MAX_ITER = 120
LDA_MAX_ITER = 8
SPLIT_SEED = 2026


def jaccard_similarity(left, right):
    """Return Jaccard similarity between two collections of terms."""
    left_set = set(left)
    right_set = set(right)

    union = left_set | right_set
    if not union:
        return 1.0

    return len(left_set & right_set) / len(union)


def matched_topic_jaccard(left_topics, right_topics):
    """
    Match topics using the Hungarian algorithm, then return mean
    top-term Jaccard similarity. This is lexical, not semantic, stability.
    """
    if len(left_topics) != len(right_topics):
        raise ValueError("Both topic sets must contain the same number of topics.")

    if not left_topics:
        return 1.0

    similarity = np.zeros(
        (len(left_topics), len(right_topics)),
        dtype=np.float64
    )

    for i, left in enumerate(left_topics):
        for j, right in enumerate(right_topics):
            similarity[i, j] = jaccard_similarity(left, right)

    row_ind, col_ind = linear_sum_assignment(1.0 - similarity)
    return float(similarity[row_ind, col_ind].mean())


def write_csv(path, rows, fieldnames):
    """Write a CSV file from dictionaries."""
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_clean_tokens(record, line_number):
    """
    Locate the cleaned token list produced by the preprocessing phase.
    Several common field names are supported; no raw abstract is silently
    substituted for cleaned text.
    """
    token_keys = (
        "tokens",
        "clean_tokens",
        "cleaned_tokens",
        "abstract_tokens",
        "tokens_cleaned",
        "processed_tokens",
        "cleaned_abstract_tokens",
    )

    for key in token_keys:
        value = record.get(key)
        if isinstance(value, list):
            return [
                str(token).strip()
                for token in value
                if isinstance(token, (str, int, float)) and str(token).strip()
            ]
        if isinstance(value, str):
            return value.split()

    for key, value in record.items():
        if "token" in key.lower():
            if isinstance(value, list):
                return [
                    str(token).strip()
                    for token in value
                    if isinstance(token, (str, int, float)) and str(token).strip()
                ]
            if isinstance(value, str):
                return value.split()

    cleaned_text_keys = (
        "cleaned_abstract",
        "abstract_clean",
        "clean_abstract",
        "processed_abstract",
        "text_clean",
    )

    for key in cleaned_text_keys:
        value = record.get(key)
        if isinstance(value, str):
            return value.split()

    available_keys = ", ".join(sorted(record.keys()))
    raise ValueError(
        f"Could not identify cleaned tokens on JSONL line {line_number}. "
        f"Available fields: {available_keys}"
    )


def load_sample_tokens(sample_indices):
    """
    Read only the pilot sample's cleaned text from the processed JSONL.
    The rest of the dataset is scanned but not retained in memory.
    """
    wanted = {int(index) for index in sample_indices}
    token_by_row = {}

    print("\nReading cleaned text for the existing pilot sample...", flush=True)

    with PREPROCESSED_PATH.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index in wanted:
                record = json.loads(line)
                token_by_row[row_index] = extract_clean_tokens(
                    record,
                    row_index + 1
                )

            if (row_index + 1) % 20000 == 0:
                print(
                    f"  Scanned {row_index + 1:,} processed records; "
                    f"sample records found: {len(token_by_row):,}/{len(wanted):,}",
                    flush=True
                )

            if len(token_by_row) == len(wanted):
                break

    missing = wanted - set(token_by_row)
    if missing:
        raise ValueError(
            f"Could not find {len(missing)} pilot sample row(s) "
            "in the preprocessed JSONL file."
        )

    print(
        f"  Loaded cleaned tokens for {len(token_by_row):,} pilot records.",
        flush=True
    )
    return token_by_row


def get_top_terms(components, feature_names, top_n=TOP_N_TERMS):
    """Return the highest-weight terms for each topic."""
    result = []

    for component in components:
        top_indices = np.argsort(component)[::-1][:top_n]
        result.append([str(feature_names[index]) for index in top_indices])

    return result


def calculate_coherence(topic_terms, reference_texts, dictionary):
    """Calculate c_v coherence on the supplied reference text sample."""
    usable_topics = [
        [term for term in topic if term in dictionary.token2id]
        for topic in topic_terms
    ]
    usable_topics = [topic for topic in usable_topics if len(topic) >= 2]

    if not usable_topics or not reference_texts:
        return None

    try:
        model = CoherenceModel(
            topics=usable_topics,
            texts=reference_texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
            window_size=110,
        )
        score = float(model.get_coherence())
        return score if math.isfinite(score) else None
    except Exception as exc:
        print(f"  Coherence calculation could not be completed: {exc}")
        return None


def validation_reconstruction_error(model, matrix):
    """Calculate the Frobenius reconstruction error in small row batches."""
    activations = model.transform(matrix)
    squared_error = 0.0
    batch_size = 128

    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        actual = matrix[start:end].toarray()
        predicted = activations[start:end] @ model.components_
        difference = actual - predicted
        squared_error += float(np.square(difference).sum())

        del actual, predicted, difference

    del activations
    return float(math.sqrt(squared_error))


def append_topic_term_rows(
    rows,
    model_name,
    topic_count,
    seed,
    topic_terms,
    components=None,
):
    for topic_number, terms in enumerate(topic_terms, start=1):
        for rank, term in enumerate(terms, start=1):
            weight = None
            if components is not None:
                feature_index = None
                # The term order is already ranked; the exact component
                # weight is stored for that ranked feature.
                weight = float(components[topic_number - 1][rank - 1])

            rows.append({
                "model": model_name,
                "topic_count": topic_count,
                "seed": seed,
                "topic": topic_number,
                "rank": rank,
                "term": term,
                "weight": weight,
            })


def main():
    print("\nChecking input files and loading matrices...", flush=True)

    for path in (
        TFIDF_PATH,
        COUNT_PATH,
        VECTORIZER_PATH,
        PREPROCESSED_PATH,
        SAMPLE_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output folder already exists; refusing to overwrite it: {OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    tfidf = __import__("scipy.sparse", fromlist=["load_npz"]).load_npz(TFIDF_PATH)
    counts = __import__("scipy.sparse", fromlist=["load_npz"]).load_npz(COUNT_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    feature_names = vectorizer.get_feature_names_out()

    if tfidf.shape != counts.shape:
        raise ValueError("TF-IDF and count matrices have different shapes.")

    if tfidf.shape[1] != len(feature_names):
        raise ValueError("Feature-name count does not match matrix columns.")

    sample_indices = np.asarray(np.load(SAMPLE_PATH), dtype=np.int64).reshape(-1)

    if len(sample_indices) != len(np.unique(sample_indices)):
        raise ValueError("The pilot sample contains duplicate row indices.")

    if len(sample_indices) < 100:
        raise ValueError("The pilot sample is unexpectedly small.")

    if sample_indices.min() < 0 or sample_indices.max() >= tfidf.shape[0]:
        raise ValueError("Pilot sample row indices are outside the matrix range.")

    train_indices, validation_indices = train_test_split(
        sample_indices,
        test_size=0.20,
        random_state=SPLIT_SEED,
    )

    print(f"  Full matrix: {tfidf.shape[0]:,} documents x {tfidf.shape[1]:,} terms")
    print(f"  Pilot sample: {len(sample_indices):,}")
    print(f"  Training subset: {len(train_indices):,}")
    print(f"  Validation subset: {len(validation_indices):,}")

    token_by_row = load_sample_tokens(sample_indices)
    train_texts = [token_by_row[int(index)] for index in train_indices]

    # Use a bounded reference set for coherence to limit memory use.
    coherence_texts = [tokens for tokens in train_texts[:4000] if tokens]
    if len(coherence_texts) < 100:
        raise ValueError("Too few non-empty training documents for coherence.")

    dictionary = Dictionary(coherence_texts)
    dictionary.filter_extremes(
        no_below=5,
        no_above=0.95,
        keep_n=8000,
    )

    if len(dictionary) == 0:
        raise ValueError("The coherence reference dictionary is empty.")

    X_tfidf_train = tfidf[train_indices]
    X_tfidf_validation = tfidf[validation_indices]
    X_count_train = counts[train_indices]
    X_count_validation = counts[validation_indices]

    run_rows = []
    term_rows = []
    comparison_rows = []
    nmf_topics_by_k = {}
    nmf_coherence_by_k = {}
    lda_coherence_by_k = {}
    lda_perplexity_by_k = {}
    nmf_validation_errors_by_k = {}
    nmf_stability_by_k = {}

    run_csv = OUTPUT_DIR / "topic_search_runs.csv"
    terms_csv = OUTPUT_DIR / "topic_search_topic_terms.csv"

    # --------------------------------------------------------
    # NMF: each candidate topic count, across five seeds.
    # --------------------------------------------------------
    for topic_count in TOPIC_COUNTS:
        print(
            f"\nNMF topic count K={topic_count}: "
            f"running {len(NMF_SEEDS)} seeds...",
            flush=True
        )

        nmf_topics_by_k[topic_count] = {}
        nmf_validation_errors_by_k[topic_count] = []

        for seed in NMF_SEEDS:
            print(
                f"  NMF K={topic_count}, seed={seed}: fitting...",
                flush=True
            )

            nmf_model = NMF(
                n_components=topic_count,
                init="nndsvda",
                solver="cd",
                max_iter=NMF_MAX_ITER,
                random_state=seed,
                tol=1e-4,
            )

            from sklearn.exceptions import ConvergenceWarning

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                nmf_model.fit(X_tfidf_train)

            convergence_warning = any(
                issubclass(item.category, ConvergenceWarning)
                for item in caught
            )

            terms = get_top_terms(nmf_model.components_, feature_names)
            nmf_topics_by_k[topic_count][seed] = terms

            validation_error = validation_reconstruction_error(
                nmf_model,
                X_tfidf_validation,
            )
            nmf_validation_errors_by_k[topic_count].append(validation_error)

            run_rows.append({
                "model": "NMF",
                "topic_count": topic_count,
                "seed": seed,
                "iterations": int(nmf_model.n_iter_),
                "training_reconstruction_error": float(nmf_model.reconstruction_err_),
                "validation_reconstruction_error": validation_error,
                "convergence_warning": bool(convergence_warning),
                "max_iterations": NMF_MAX_ITER,
            })

            append_topic_term_rows(
                term_rows,
                "NMF",
                topic_count,
                seed,
                terms,
            )

            if seed == 42:
                nmf_coherence_by_k[topic_count] = calculate_coherence(
                    terms,
                    coherence_texts,
                    dictionary,
                )
                model_path = OUTPUT_DIR / f"nmf_k{topic_count}_seed42.joblib"
                joblib.dump(nmf_model, model_path, compress=3)

            write_csv(
                run_csv,
                run_rows,
                [
                    "model",
                    "topic_count",
                    "seed",
                    "iterations",
                    "training_reconstruction_error",
                    "validation_reconstruction_error",
                    "convergence_warning",
                    "max_iterations",
                ],
            )
            write_csv(
                terms_csv,
                term_rows,
                ["model", "topic_count", "seed", "topic", "rank", "term", "weight"],
            )

            print(
                f"    Finished: iterations={nmf_model.n_iter_}, "
                f"validation error={validation_error:.4f}, "
                f"convergence warning={convergence_warning}",
                flush=True
            )

            del nmf_model
            gc.collect()

        # Pairwise, optimally matched top-15 lexical stability.
        pair_scores = []
        seeds = list(NMF_SEEDS)

        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                pair_scores.append(
                    matched_topic_jaccard(
                        nmf_topics_by_k[topic_count][seeds[i]],
                        nmf_topics_by_k[topic_count][seeds[j]],
                    )
                )

        nmf_stability_by_k[topic_count] = {
            "mean": float(mean(pair_scores)),
            "std": float(stdev(pair_scores)) if len(pair_scores) > 1 else 0.0,
            "pair_count": len(pair_scores),
            "definition": "Mean pairwise Hungarian-matched Jaccard similarity of top-15 terms",
        }

    # --------------------------------------------------------
    # LDA: one reproducible run per candidate topic count.
    # --------------------------------------------------------
    for topic_count in TOPIC_COUNTS:
        print(f"\nLDA topic count K={topic_count}: fitting...", flush=True)

        lda_model = LatentDirichletAllocation(
            n_components=topic_count,
            learning_method="online",
            max_iter=LDA_MAX_ITER,
            batch_size=128,
            evaluate_every=-1,
            n_jobs=1,
            random_state=42,
        )
        lda_model.fit(X_count_train)

        perplexity = float(lda_model.perplexity(X_count_validation))
        lda_terms = get_top_terms(lda_model.components_, feature_names)

        lda_perplexity_by_k[topic_count] = perplexity
        lda_coherence_by_k[topic_count] = calculate_coherence(
            lda_terms,
            coherence_texts,
            dictionary,
        )

        append_topic_term_rows(
            term_rows,
            "LDA",
            topic_count,
            42,
            lda_terms,
        )

        joblib.dump(
            lda_model,
            OUTPUT_DIR / f"lda_k{topic_count}_seed42.joblib",
            compress=3,
        )

        write_csv(
            terms_csv,
            term_rows,
            ["model", "topic_count", "seed", "topic", "rank", "term", "weight"],
        )

        print(
            f"  LDA K={topic_count}: held-out perplexity={perplexity:.4f}; "
            f"c_v coherence={lda_coherence_by_k[topic_count]}",
            flush=True
        )

        del lda_model
        gc.collect()

    # --------------------------------------------------------
    # Combine candidate-level metrics.
    # --------------------------------------------------------
    for topic_count in TOPIC_COUNTS:
        errors = nmf_validation_errors_by_k[topic_count]
        stability = nmf_stability_by_k[topic_count]

        comparison_rows.append({
            "topic_count": topic_count,
            "nmf_validation_reconstruction_error_mean": float(mean(errors)),
            "nmf_validation_reconstruction_error_std": (
                float(stdev(errors)) if len(errors) > 1 else 0.0
            ),
            "nmf_top15_jaccard_stability_mean": stability["mean"],
            "nmf_top15_jaccard_stability_std": stability["std"],
            "nmf_pairwise_comparisons": stability["pair_count"],
            "nmf_seed42_cv_coherence": nmf_coherence_by_k.get(topic_count),
            "lda_validation_perplexity": lda_perplexity_by_k.get(topic_count),
            "lda_seed42_cv_coherence": lda_coherence_by_k.get(topic_count),
        })

    comparison_csv = OUTPUT_DIR / "topic_search_comparison.csv"
    write_csv(
        comparison_csv,
        comparison_rows,
        [
            "topic_count",
            "nmf_validation_reconstruction_error_mean",
            "nmf_validation_reconstruction_error_std",
            "nmf_top15_jaccard_stability_mean",
            "nmf_top15_jaccard_stability_std",
            "nmf_pairwise_comparisons",
            "nmf_seed42_cv_coherence",
            "lda_validation_perplexity",
            "lda_seed42_cv_coherence",
        ],
    )

    summary = {
        "phase": "Phase 7 - topic-count selection and stability analysis",
        "status": "completed",
        "dataset": {
            "source": "Existing filtered and preprocessed arXiv cs.LG dataset",
            "matrix_documents": int(tfidf.shape[0]),
            "matrix_features": int(tfidf.shape[1]),
            "pilot_sample_documents": int(len(sample_indices)),
        },
        "split": {
            "method": "Random train_test_split on the existing pilot sample",
            "random_state": SPLIT_SEED,
            "training_documents": int(len(train_indices)),
            "validation_documents": int(len(validation_indices)),
        },
        "experiment": {
            "topic_counts": list(TOPIC_COUNTS),
            "nmf_seeds": list(NMF_SEEDS),
            "nmf_max_iterations": NMF_MAX_ITER,
            "lda_seed": 42,
            "lda_max_iterations": LDA_MAX_ITER,
            "top_terms_per_topic": TOP_N_TERMS,
            "coherence": "c_v on up to 4,000 non-empty training documents",
            "stability": "Pairwise Hungarian-matched Jaccard similarity of top-15 NMF terms",
        },
        "candidate_results": comparison_rows,
        "methodological_limitations": [
            "This is pilot-sample model selection, not final full-dataset model training.",
            "The pilot split is random, not chronological; it does not test future-year generalisation.",
            "The existing vectorizers were fitted before this pilot split. Their vocabulary and IDF therefore saw documents outside the training subset. Treat these validation results as exploratory; final evaluation must refit vectorizers on training data only.",
            "NMF stability measures overlap among top terms; it does not establish semantic equivalence.",
            "Coherence is calculated for the seed-42 representative model at each topic count.",
            "LDA perplexity and NMF reconstruction error are different metrics and should not be compared directly.",
            "Topic labels still require human interpretation and review.",
        ],
    }

    summary_path = OUTPUT_DIR / "topic_search_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    report_path = OUTPUT_DIR / "topic_search_readme.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "PHASE 7 — TOPIC-COUNT SELECTION AND STABILITY\n"
            "================================================\n\n"
            "Candidate topic counts: 5, 8, 10, 12\n"
            "NMF seeds: 42, 13, 21, 34, 55\n"
            "LDA seed: 42\n\n"
            "Files:\n"
            "- topic_search_comparison.csv: candidate-level metrics\n"
            "- topic_search_runs.csv: individual NMF seed runs\n"
            "- topic_search_topic_terms.csv: topic terms for NMF and LDA\n"
            "- topic_search_summary.json: configuration, results and limitations\n"
            "- nmf_*.joblib and lda_*.joblib: representative seed-42 models\n\n"
            "Important: these are pilot results. The current vectorizers were\n"
            "fitted before the pilot split, so final evaluation must refit the\n"
            "vectorizers on training data only. The random split is not a\n"
            "chronological future-year evaluation.\n"
        )

    print("\n============================================================")
    print("PHASE 7 COMPLETED")
    print("============================================================")
    print(f"Comparison: {comparison_csv}")
    print(f"NMF runs:   {run_csv}")
    print(f"Topic terms:{terms_csv}")
    print(f"Summary:    {summary_path}")
    print(f"Models:     {OUTPUT_DIR}")
    print("\nCandidate comparison:")
    for row in comparison_rows:
        print(
            f"  K={row['topic_count']}: "
            f"NMF stability={row['nmf_top15_jaccard_stability_mean']:.4f}; "
            f"NMF validation error={row['nmf_validation_reconstruction_error_mean']:.4f}; "
            f"LDA perplexity={row['lda_validation_perplexity']:.4f}"
        )


if __name__ == "__main__":
    main()
