from __future__ import annotations

import csv
import gc
import json
import math
import time
import warnings
from pathlib import Path
from statistics import mean, stdev

import joblib
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]

PREPROCESSED_PATH = (
    ROOT / "data" / "processed" /
    "cs_lg_2018_2024_preprocessed.jsonl"
)
TFIDF_TEMPLATE_PATH = ROOT / "data" / "features" / "tfidf_vectorizer.joblib"
COUNT_TEMPLATE_PATH = ROOT / "data" / "features" / "count_vectorizer.joblib"
SAMPLE_PATH = ROOT / "outputs" / "models" / "pilot" / "sample_row_indices.npy"
OUTPUT_DIR = ROOT / "outputs" / "models" / "leakage_free_validation"

TOPIC_COUNTS = (5, 8, 10, 12)
NMF_SEEDS = (42, 13, 21, 34, 55)
TOP_N_TERMS = 15
NMF_MAX_ITER = 200
LDA_MAX_ITER = 8
SPLIT_SEED = 2026
COHERENCE_DOCUMENT_LIMIT = 4000


def jaccard_similarity(left, right):
    """Calculate Jaccard similarity between two sets of terms."""
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set

    if not union:
        return 1.0

    return len(left_set & right_set) / len(union)


def matched_topic_jaccard(left_topics, right_topics):
    """
    Match topics optimally using the Hungarian algorithm, then calculate
    mean top-term Jaccard similarity. This measures lexical overlap.
    """
    if len(left_topics) != len(right_topics):
        raise ValueError("The topic sets must have the same number of topics.")

    if not left_topics:
        return 1.0

    similarities = np.zeros(
        (len(left_topics), len(right_topics)),
        dtype=np.float64,
    )

    for i, left in enumerate(left_topics):
        for j, right in enumerate(right_topics):
            similarities[i, j] = jaccard_similarity(left, right)

    row_indices, column_indices = linear_sum_assignment(
        1.0 - similarities
    )
    return float(similarities[row_indices, column_indices].mean())


def write_csv(path, rows, fieldnames):
    """Write a CSV checkpoint."""
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_clean_tokens(record, line_number):
    """Read the cleaned token field without substituting raw abstracts."""
    preferred_keys = (
        "tokens",
        "clean_tokens",
        "cleaned_tokens",
        "abstract_tokens",
        "tokens_cleaned",
        "processed_tokens",
        "cleaned_abstract_tokens",
    )

    for key in preferred_keys:
        value = record.get(key)

        if isinstance(value, list):
            return [
                str(token).strip()
                for token in value
                if isinstance(token, (str, int, float))
                and str(token).strip()
            ]

        if isinstance(value, str):
            return value.split()

    for key, value in record.items():
        if "token" in key.lower():
            if isinstance(value, list):
                return [
                    str(token).strip()
                    for token in value
                    if isinstance(token, (str, int, float))
                    and str(token).strip()
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

    raise ValueError(
        f"Could not identify cleaned tokens on JSONL line {line_number}. "
        f"Available fields: {', '.join(sorted(record.keys()))}"
    )


def load_sample_tokens(sample_indices):
    """Load only the existing 8,000-document pilot sample."""
    wanted = {int(index) for index in sample_indices}
    token_by_row = {}

    print("\n[1/5] Reading cleaned text for the pilot sample...", flush=True)

    with PREPROCESSED_PATH.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index in wanted:
                record = json.loads(line)
                token_by_row[row_index] = extract_clean_tokens(
                    record,
                    row_index + 1,
                )

            if (row_index + 1) % 20000 == 0:
                print(
                    f"  Scanned {row_index + 1:,} records; "
                    f"located {len(token_by_row):,}/{len(wanted):,} sample rows.",
                    flush=True,
                )

            if len(token_by_row) == len(wanted):
                break

    missing = wanted - set(token_by_row)
    if missing:
        raise ValueError(
            f"Could not locate {len(missing)} pilot sample records."
        )

    return token_by_row


def get_top_terms(components, feature_names):
    """Return top terms and their corresponding component weights."""
    all_terms = []
    all_weights = []

    for component in components:
        indices = np.argsort(component)[::-1][:TOP_N_TERMS]
        all_terms.append([str(feature_names[i]) for i in indices])
        all_weights.append([float(component[i]) for i in indices])

    return all_terms, all_weights


def calculate_coherence(topic_terms, reference_texts, dictionary):
    """Calculate c_v coherence on a bounded training-only reference sample."""
    usable_topics = [
        [term for term in topic if term in dictionary.token2id]
        for topic in topic_terms
    ]
    usable_topics = [topic for topic in usable_topics if len(topic) >= 2]

    if not usable_topics or not reference_texts:
        return None

    try:
        coherence_model = CoherenceModel(
            topics=usable_topics,
            texts=reference_texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
            window_size=110,
        )
        score = float(coherence_model.get_coherence())
        return score if math.isfinite(score) else None
    except Exception as exc:
        print(f"  Coherence warning: {exc}", flush=True)
        return None


def validation_reconstruction_error(model, matrix):
    """Calculate validation Frobenius error in small row batches."""
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


def progress_update(completed, total, label, started):
    """Print an ASCII progress bar and elapsed time."""
    width = 28
    fraction = completed / total if total else 1.0
    filled = int(width * fraction)
    bar = "=" * filled + "-" * (width - filled)
    elapsed = (time.perf_counter() - started) / 60.0

    print(
        f"  [{bar}] {completed}/{total} model runs "
        f"({fraction * 100:.0f}%) — {label}; "
        f"elapsed {elapsed:.1f} min",
        flush=True,
    )


def main():
    started = time.perf_counter()

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {OUTPUT_DIR}"
        )

    for path in (
        PREPROCESSED_PATH,
        TFIDF_TEMPLATE_PATH,
        COUNT_TEMPLATE_PATH,
        SAMPLE_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input file is missing: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\nPHASE 8 — LEAKAGE-FREE VALIDATION", flush=True)
    print("The original full-corpus vectorisers will not be used for fitting.", flush=True)

    sample_indices = np.asarray(
        np.load(SAMPLE_PATH),
        dtype=np.int64,
    ).reshape(-1)

    if len(sample_indices) < 100:
        raise ValueError("The pilot sample is unexpectedly small.")

    if len(sample_indices) != len(np.unique(sample_indices)):
        raise ValueError("Duplicate indices were found in the pilot sample.")

    train_indices, validation_indices = train_test_split(
        sample_indices,
        test_size=0.20,
        random_state=SPLIT_SEED,
    )

    token_by_row = load_sample_tokens(sample_indices)

    train_tokens = [
        token_by_row[int(index)]
        for index in train_indices
    ]
    validation_tokens = [
        token_by_row[int(index)]
        for index in validation_indices
    ]

    train_documents = [" ".join(tokens) for tokens in train_tokens]
    validation_documents = [" ".join(tokens) for tokens in validation_tokens]

    if not any(train_documents):
        raise ValueError("The training documents contain no usable tokens.")

    print("\n[2/5] Fitting fresh vectorisers on training documents only...", flush=True)

    original_tfidf = joblib.load(TFIDF_TEMPLATE_PATH)
    original_count = joblib.load(COUNT_TEMPLATE_PATH)

    # clone() retains the original configuration but discards fitted vocabulary
    # and IDF values. The new estimators are fitted on training documents only.
    tfidf_vectorizer = clone(original_tfidf)
    count_vectorizer = clone(original_count)

    X_tfidf_train = tfidf_vectorizer.fit_transform(train_documents)
    X_tfidf_validation = tfidf_vectorizer.transform(validation_documents)

    X_count_train = count_vectorizer.fit_transform(train_documents)
    X_count_validation = count_vectorizer.transform(validation_documents)

    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    count_features = count_vectorizer.get_feature_names_out()

    if not np.array_equal(tfidf_features, count_features):
        raise ValueError(
            "The freshly fitted TF-IDF and count vectorisers produced "
            "different vocabularies. Stopping to avoid incorrect comparisons."
        )

    joblib.dump(
        tfidf_vectorizer,
        OUTPUT_DIR / "tfidf_vectorizer_train_only.joblib",
        compress=3,
    )
    joblib.dump(
        count_vectorizer,
        OUTPUT_DIR / "count_vectorizer_train_only.joblib",
        compress=3,
    )

    np.save(OUTPUT_DIR / "training_row_indices.npy", train_indices)
    np.save(OUTPUT_DIR / "validation_row_indices.npy", validation_indices)

    print(f"  Training documents: {len(train_indices):,}")
    print(f"  Validation documents: {len(validation_indices):,}")
    print(f"  Training-only vocabulary: {len(tfidf_features):,} terms")
    print(f"  TF-IDF training matrix: {X_tfidf_train.shape}")
    print(f"  Count training matrix: {X_count_train.shape}")

    # Coherence reference texts come only from the training subset.
    coherence_texts = [
        tokens for tokens in train_tokens[:COHERENCE_DOCUMENT_LIMIT]
        if tokens
    ]

    if len(coherence_texts) < 100:
        raise ValueError("Too few training documents for coherence analysis.")

    dictionary = Dictionary(coherence_texts)
    dictionary.filter_extremes(
        no_below=5,
        no_above=0.95,
        keep_n=8000,
    )

    if len(dictionary) == 0:
        raise ValueError("The training-only coherence dictionary is empty.")

    print(f"  Coherence reference documents: {len(coherence_texts):,}")
    print(f"  Coherence dictionary terms: {len(dictionary):,}")

    # --------------------------------------------------------
    # Model experiment setup.
    # --------------------------------------------------------
    print("\n[3/5] Running NMF across four topic counts and five seeds...", flush=True)

    total_runs = len(TOPIC_COUNTS) * (len(NMF_SEEDS) + 1)
    completed_runs = 0

    nmf_topics_by_k = {}
    nmf_errors_by_k = {}
    nmf_coherence_by_k = {}
    nmf_stability_by_k = {}

    run_rows = []
    term_rows = []

    run_fields = [
        "model",
        "topic_count",
        "seed",
        "initialisation",
        "iterations",
        "training_reconstruction_error",
        "validation_reconstruction_error",
        "validation_perplexity",
        "c_v_coherence",
        "convergence_warning",
        "max_iterations",
    ]

    term_fields = [
        "model",
        "topic_count",
        "seed",
        "topic",
        "rank",
        "term",
        "weight",
    ]

    run_csv = OUTPUT_DIR / "validation_model_runs.csv"
    terms_csv = OUTPUT_DIR / "validation_topic_terms.csv"

    for topic_count in TOPIC_COUNTS:
        nmf_topics_by_k[topic_count] = {}
        nmf_errors_by_k[topic_count] = []

        for seed in NMF_SEEDS:
            print(
                f"\n  Fitting NMF: K={topic_count}, seed={seed}, "
                f"initialisation=random...",
                flush=True,
            )

            nmf_model = NMF(
                n_components=topic_count,
                init="random",
                solver="cd",
                max_iter=NMF_MAX_ITER,
                random_state=seed,
                tol=1e-4,
            )

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                nmf_model.fit(X_tfidf_train)

            convergence_warning = any(
                issubclass(item.category, ConvergenceWarning)
                for item in captured
            )

            terms, weights = get_top_terms(
                nmf_model.components_,
                tfidf_features,
            )

            nmf_topics_by_k[topic_count][seed] = terms

            validation_error = validation_reconstruction_error(
                nmf_model,
                X_tfidf_validation,
            )
            nmf_errors_by_k[topic_count].append(validation_error)

            coherence = None
            if seed == 42:
                print("  Calculating NMF c_v coherence for seed 42...", flush=True)
                coherence = calculate_coherence(
                    terms,
                    coherence_texts,
                    dictionary,
                )
                nmf_coherence_by_k[topic_count] = coherence

                joblib.dump(
                    nmf_model,
                    OUTPUT_DIR / f"nmf_k{topic_count}_seed42.joblib",
                    compress=3,
                )

            run_rows.append({
                "model": "NMF",
                "topic_count": topic_count,
                "seed": seed,
                "initialisation": "random",
                "iterations": int(nmf_model.n_iter_),
                "training_reconstruction_error": float(nmf_model.reconstruction_err_),
                "validation_reconstruction_error": validation_error,
                "validation_perplexity": None,
                "c_v_coherence": coherence,
                "convergence_warning": bool(convergence_warning),
                "max_iterations": NMF_MAX_ITER,
            })

            for topic_number, topic_terms in enumerate(terms, start=1):
                for rank, term in enumerate(topic_terms, start=1):
                    term_rows.append({
                        "model": "NMF",
                        "topic_count": topic_count,
                        "seed": seed,
                        "topic": topic_number,
                        "rank": rank,
                        "term": term,
                        "weight": weights[topic_number - 1][rank - 1],
                    })

            write_csv(run_csv, run_rows, run_fields)
            write_csv(terms_csv, term_rows, term_fields)

            print(
                f"    Iterations: {nmf_model.n_iter_}; "
                f"validation error: {validation_error:.4f}; "
                f"convergence warning: {convergence_warning}",
                flush=True,
            )

            del nmf_model
            gc.collect()

            completed_runs += 1
            progress_update(
                completed_runs,
                total_runs,
                f"NMF K={topic_count}, seed={seed}",
                started,
            )

        # Compare the five independently random-initialised NMF runs.
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
            "pairwise_comparisons": len(pair_scores),
        }

    # --------------------------------------------------------
    # LDA experiments.
    # --------------------------------------------------------
    print("\n[4/5] Running LDA across the four topic counts...", flush=True)

    lda_perplexity_by_k = {}
    lda_coherence_by_k = {}

    for topic_count in TOPIC_COUNTS:
        print(f"\n  Fitting LDA: K={topic_count}...", flush=True)

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

        validation_perplexity = float(
            lda_model.perplexity(X_count_validation)
        )

        lda_terms, lda_weights = get_top_terms(
            lda_model.components_,
            count_features,
        )

        lda_coherence = calculate_coherence(
            lda_terms,
            coherence_texts,
            dictionary,
        )

        lda_perplexity_by_k[topic_count] = validation_perplexity
        lda_coherence_by_k[topic_count] = lda_coherence

        joblib.dump(
            lda_model,
            OUTPUT_DIR / f"lda_k{topic_count}_seed42.joblib",
            compress=3,
        )

        run_rows.append({
            "model": "LDA",
            "topic_count": topic_count,
            "seed": 42,
            "initialisation": "online",
            "iterations": int(lda_model.n_iter_),
            "training_reconstruction_error": None,
            "validation_reconstruction_error": None,
            "validation_perplexity": validation_perplexity,
            "c_v_coherence": lda_coherence,
            "convergence_warning": None,
            "max_iterations": LDA_MAX_ITER,
        })

        for topic_number, topic_terms in enumerate(lda_terms, start=1):
            for rank, term in enumerate(topic_terms, start=1):
                term_rows.append({
                    "model": "LDA",
                    "topic_count": topic_count,
                    "seed": 42,
                    "topic": topic_number,
                    "rank": rank,
                    "term": term,
                    "weight": lda_weights[topic_number - 1][rank - 1],
                })

        write_csv(run_csv, run_rows, run_fields)
        write_csv(terms_csv, term_rows, term_fields)

        print(
            f"    Held-out perplexity: {validation_perplexity:.4f}; "
            f"c_v coherence: {lda_coherence}",
            flush=True,
        )

        del lda_model
        gc.collect()

        completed_runs += 1
        progress_update(
            completed_runs,
            total_runs,
            f"LDA K={topic_count}",
            started,
        )

    # --------------------------------------------------------
    # Candidate-level comparison and reporting.
    # --------------------------------------------------------
    print("\n[5/5] Writing validation comparison and summary...", flush=True)

    comparison_rows = []

    for topic_count in TOPIC_COUNTS:
        errors = nmf_errors_by_k[topic_count]
        stability = nmf_stability_by_k[topic_count]

        comparison_rows.append({
            "topic_count": topic_count,
            "nmf_validation_error_mean": float(mean(errors)),
            "nmf_validation_error_std": (
                float(stdev(errors)) if len(errors) > 1 else 0.0
            ),
            "nmf_top15_jaccard_stability_mean": stability["mean"],
            "nmf_top15_jaccard_stability_std": stability["std"],
            "nmf_pairwise_comparisons": stability["pairwise_comparisons"],
            "nmf_seed42_cv_coherence": nmf_coherence_by_k.get(topic_count),
            "lda_validation_perplexity": lda_perplexity_by_k.get(topic_count),
            "lda_seed42_cv_coherence": lda_coherence_by_k.get(topic_count),
        })

    comparison_fields = [
        "topic_count",
        "nmf_validation_error_mean",
        "nmf_validation_error_std",
        "nmf_top15_jaccard_stability_mean",
        "nmf_top15_jaccard_stability_std",
        "nmf_pairwise_comparisons",
        "nmf_seed42_cv_coherence",
        "lda_validation_perplexity",
        "lda_seed42_cv_coherence",
    ]

    comparison_csv = OUTPUT_DIR / "leakage_free_model_comparison.csv"
    write_csv(comparison_csv, comparison_rows, comparison_fields)

    summary = {
        "phase": "Phase 8 - leakage-free validation",
        "status": "completed",
        "dataset": {
            "source": "Existing filtered and preprocessed arXiv cs.LG dataset",
            "pilot_sample_documents": int(len(sample_indices)),
            "training_documents": int(len(train_indices)),
            "validation_documents": int(len(validation_indices)),
            "training_only_vocabulary_size": int(len(tfidf_features)),
        },
        "split": {
            "method": "Random 80/20 split of the existing pilot sample",
            "random_state": SPLIT_SEED,
        },
        "vectorisation": {
            "method": "Cloned the existing vectoriser configurations, discarded fitted state, then fitted on training documents only",
            "tfidf_training_shape": list(X_tfidf_train.shape),
            "tfidf_validation_shape": list(X_tfidf_validation.shape),
            "count_training_shape": list(X_count_train.shape),
            "count_validation_shape": list(X_count_validation.shape),
            "validation_documents_used_in_vectoriser_fit": False,
        },
        "models": {
            "nmf_topic_counts": list(TOPIC_COUNTS),
            "nmf_seeds": list(NMF_SEEDS),
            "nmf_initialisation": "random",
            "nmf_max_iterations": NMF_MAX_ITER,
            "lda_topic_counts": list(TOPIC_COUNTS),
            "lda_seed": 42,
            "lda_max_iterations": LDA_MAX_ITER,
            "top_terms_per_topic": TOP_N_TERMS,
        },
        "coherence": {
            "metric": "c_v",
            "reference": "Up to 4,000 non-empty training documents only",
        },
        "stability": {
            "metric": "Mean pairwise Hungarian-matched Jaccard similarity of top-15 NMF terms",
            "interpretation": "Lexical stability; not proof of semantic equivalence",
        },
        "candidate_results": comparison_rows,
        "limitations": [
            "This is pilot-sample validation, not final model training on all 87,945 records.",
            "The random pilot split does not test generalisation to future years.",
            "The vectorisers were refitted using training documents only, preventing validation-document vocabulary and IDF leakage in this experiment.",
            "The NMF initialisation is random so that the seed comparison tests genuinely different initialisations.",
            "Convergence warnings must be considered when comparing NMF candidates.",
            "LDA perplexity and NMF reconstruction error are different metrics and cannot be directly compared.",
            "Coherence is measured on a training-only reference sample.",
            "Human interpretation is still required before selecting the final topic model.",
        ],
    }

    summary_path = OUTPUT_DIR / "leakage_free_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    report_path = OUTPUT_DIR / "README.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "PHASE 8 — LEAKAGE-FREE VALIDATION\n"
            "=================================\n\n"
            "Fresh TF-IDF and count vectorisers were fitted on training documents only.\n"
            "The validation documents were transformed, not used for fitting.\n\n"
            "NMF: K=5, 8, 10, 12; five random initialisation seeds per K.\n"
            "LDA: K=5, 8, 10, 12; seed 42.\n\n"
            "See leakage_free_model_comparison.csv for candidate metrics.\n"
            "See validation_model_runs.csv for individual model runs.\n"
            "See validation_topic_terms.csv for extracted topic terms.\n"
            "See leakage_free_validation_summary.json for settings and limitations.\n\n"
            "This remains pilot-sample validation, not final full-dataset training.\n"
        )

    elapsed = (time.perf_counter() - started) / 60.0

    print("\n============================================================")
    print("PHASE 8 COMPLETED")
    print("============================================================")
    print(f"Model comparison: {comparison_csv}")
    print(f"Individual runs:  {run_csv}")
    print(f"Topic terms:      {terms_csv}")
    print(f"Summary:          {summary_path}")
    print(f"Models and vectorisers: {OUTPUT_DIR}")
    print(f"Elapsed time: {elapsed:.1f} minutes")

    print("\nCorrected candidate comparison:")
    for row in comparison_rows:
        print(
            f"  K={row['topic_count']}: "
            f"NMF error={row['nmf_validation_error_mean']:.4f}; "
            f"NMF stability={row['nmf_top15_jaccard_stability_mean']:.4f}; "
            f"LDA perplexity={row['lda_validation_perplexity']:.4f}"
        )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
