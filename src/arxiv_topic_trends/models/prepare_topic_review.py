from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT / "data" / "processed" /
    "cs_lg_2018_2024_preprocessed.jsonl"
)

MODEL_PATH = (
    ROOT / "outputs" / "models" / "leakage_free_validation" /
    "nmf_k10_seed42.joblib"
)

VECTORIZER_PATH = (
    ROOT / "outputs" / "models" / "leakage_free_validation" /
    "tfidf_vectorizer_train_only.joblib"
)

VALIDATION_PATH = (
    ROOT / "outputs" / "models" / "leakage_free_validation" /
    "validation_row_indices.npy"
)

TERMS_PATH = (
    ROOT / "outputs" / "models" / "leakage_free_validation" /
    "validation_topic_terms.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "reports" / "topic_review_k10"

PAPERS_PER_TOPIC = 5
TOPIC_COUNT = 10


def progress_bar(done, total, label, started):
    width = 25
    fraction = done / total if total else 1
    filled = int(width * fraction)
    bar = "=" * filled + "-" * (width - filled)
    elapsed = (time.perf_counter() - started) / 60

    print(
        f"[{bar}] {done:,}/{total:,} "
        f"({fraction * 100:.0f}%) {label} | {elapsed:.1f} min",
        flush=True,
    )


def first_value(record, keys):
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return ""


def clean_tokens(record, line_number):
    preferred = (
        "tokens",
        "clean_tokens",
        "cleaned_tokens",
        "abstract_tokens",
        "tokens_cleaned",
        "processed_tokens",
        "cleaned_abstract_tokens",
    )

    for key in preferred:
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

    for key in (
        "cleaned_abstract",
        "abstract_clean",
        "clean_abstract",
        "processed_abstract",
        "text_clean",
    ):
        value = record.get(key)
        if isinstance(value, str):
            return value.split()

    raise ValueError(
        f"Cleaned token field not identified at JSONL line {line_number}. "
        f"Available keys: {', '.join(sorted(record.keys()))}"
    )


def get_year(record):
    versions = record.get("versions", [])

    if isinstance(versions, list):
        for version in versions:
            if not isinstance(version, dict):
                continue

            if version.get("version") == "v1" or version.get("name") == "v1":
                created = version.get("created", "")
                match = re.search(r"\b(19|20)\d{2}\b", str(created))
                if match:
                    return int(match.group(0))

        if versions and isinstance(versions[0], dict):
            created = versions[0].get("created", "")
            match = re.search(r"\b(19|20)\d{2}\b", str(created))
            if match:
                return int(match.group(0))

    update_date = record.get("update_date", "")
    match = re.search(r"\b(19|20)\d{2}\b", str(update_date))

    return int(match.group(0)) if match else ""


def normalise_categories(value):
    if isinstance(value, list):
        return " ".join(str(item) for item in value)

    return str(value or "")


def load_terms():
    grouped = {}

    with TERMS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if (
                row.get("model") == "NMF"
                and int(row.get("topic_count", 0)) == TOPIC_COUNT
                and int(row.get("seed", 0)) == 42
            ):
                topic = int(row["topic"])
                grouped.setdefault(topic, []).append(row)

    for topic in grouped:
        grouped[topic].sort(key=lambda row: int(row["rank"]))

    if len(grouped) != TOPIC_COUNT:
        raise ValueError(
            f"Expected {TOPIC_COUNT} NMF topics, but found {len(grouped)}."
        )

    return {
        topic: [row["term"] for row in rows]
        for topic, rows in grouped.items()
    }


def main():
    started = time.perf_counter()

    for path in (
        DATA_PATH,
        MODEL_PATH,
        VECTORIZER_PATH,
        VALIDATION_PATH,
        TERMS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_DIR}"
        )

    validation_indices = np.asarray(
        np.load(VALIDATION_PATH),
        dtype=np.int64,
    ).reshape(-1)

    if len(validation_indices) != 1600:
        raise ValueError(
            f"Expected 1,600 validation indices; found {len(validation_indices)}."
        )

    wanted = set(int(i) for i in validation_indices)

    if len(wanted) != len(validation_indices):
        raise ValueError("Duplicate validation row indices were found.")

    print("\n[1/4] Loading model and training-only vectoriser...", flush=True)

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    if model.n_components != TOPIC_COUNT:
        raise ValueError("The saved model is not an NMF K=10 model.")

    if model.components_.shape[1] != len(vectorizer.get_feature_names_out()):
        raise ValueError("Model and vectoriser feature dimensions do not match.")

    print("[2/4] Reading validation metadata and cleaned text...", flush=True)

    records = {}

    with DATA_PATH.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index in wanted:
                record = json.loads(line)
                tokens = clean_tokens(record, row_index + 1)

                title = first_value(record, ("title",))
                abstract = first_value(record, ("abstract", "summary"))

                if not title or not abstract:
                    raise ValueError(
                        f"Title or abstract missing for validation row {row_index}."
                    )

                categories = normalise_categories(
                    first_value(record, ("categories", "category"))
                )

                records[row_index] = {
                    "row_index": row_index,
                    "id": first_value(record, ("id", "paper_id")),
                    "title": str(title).strip().replace("\n", " "),
                    "abstract": str(abstract).strip(),
                    "year": get_year(record),
                    "categories": categories,
                    "primary_category": categories.split()[0] if categories else "",
                    "tokens": tokens,
                }

            if (row_index + 1) % 20000 == 0:
                print(
                    f"  Scanned {row_index + 1:,} dataset rows; "
                    f"found {len(records):,}/{len(wanted):,} validation papers.",
                    flush=True,
                )

            if len(records) == len(wanted):
                break

    missing = wanted - set(records)

    if missing:
        raise ValueError(
            f"Could not locate {len(missing)} validation records."
        )

    ordered_records = [records[int(index)] for index in validation_indices]
    documents = [" ".join(row["tokens"]) for row in ordered_records]

    print("[3/4] Transforming validation papers through the saved model...", flush=True)

    matrix = vectorizer.transform(documents)
    activations = model.transform(matrix)

    if activations.shape != (len(ordered_records), TOPIC_COUNT):
        raise ValueError("Unexpected topic activation matrix shape.")

    topic_terms = load_terms()

    # Candidate labels are provisional and must be checked against abstracts.
    provisional_labels = {
        1: "Supervised learning and classification",
        2: "Federated and privacy-preserving learning",
        3: "Reinforcement learning",
        4: "Optimisation, online learning and bandits",
        5: "Time-series forecasting",
        6: "Graph neural networks",
        7: "Deep neural networks",
        8: "Large language models",
        9: "General machine learning methods",
        10: "Adversarial robustness",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    paper_csv = OUTPUT_DIR / "nmf_k10_validation_representative_papers.csv"

    paper_fields = [
        "topic_number",
        "provisional_topic_label",
        "top_15_topic_terms",
        "rank_within_topic",
        "activation_score",
        "arxiv_id",
        "year",
        "primary_category",
        "title",
        "abstract",
        "reviewer_confirmed_label",
        "review_notes",
    ]

    paper_rows = []

    print("[4/4] Selecting five most strongly associated validation papers per topic...", flush=True)

    for topic_index in range(TOPIC_COUNT):
        topic_number = topic_index + 1
        scores = activations[:, topic_index]
        top_indices = np.argsort(scores)[::-1][:PAPERS_PER_TOPIC]

        terms_text = ", ".join(topic_terms[topic_number])

        for rank, document_index in enumerate(top_indices, start=1):
            record = ordered_records[int(document_index)]

            paper_rows.append({
                "topic_number": topic_number,
                "provisional_topic_label": provisional_labels[topic_number],
                "top_15_topic_terms": terms_text,
                "rank_within_topic": rank,
                "activation_score": float(scores[document_index]),
                "arxiv_id": record["id"],
                "year": record["year"],
                "primary_category": record["primary_category"],
                "title": record["title"],
                "abstract": record["abstract"],
                "reviewer_confirmed_label": "",
                "review_notes": "",
            })

        print(
            f"  Topic {topic_number}/10: selected {len(top_indices)} papers; "
            f"{provisional_labels[topic_number]}",
            flush=True,
        )

    with paper_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=paper_fields)
        writer.writeheader()
        writer.writerows(paper_rows)

    label_csv = OUTPUT_DIR / "topic_label_review_template.csv"

    with label_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "topic_number",
            "top_15_terms",
            "provisional_label",
            "reviewer_confirmed_label",
            "distinct_from_other_topics",
            "review_notes",
        ]

        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for topic_number in range(1, TOPIC_COUNT + 1):
            writer.writerow({
                "topic_number": topic_number,
                "top_15_terms": ", ".join(topic_terms[topic_number]),
                "provisional_label": provisional_labels[topic_number],
                "reviewer_confirmed_label": "",
                "distinct_from_other_topics": "",
                "review_notes": "",
            })

    summary = {
        "phase": "Phase 9 - representative paper review",
        "status": "extraction_completed",
        "model": "NMF K=10 seed 42",
        "source": "Leakage-free validation subset",
        "validation_documents": len(ordered_records),
        "topics": TOPIC_COUNT,
        "papers_per_topic": PAPERS_PER_TOPIC,
        "representative_papers_extracted": len(paper_rows),
        "selection_method": "Top five validation-document NMF topic activations per topic",
        "human_review_completed": False,
        "note": "Provisional labels are suggestions based on top terms. They must be verified by reading the selected paper abstracts.",
        "output_files": [
            paper_csv.name,
            label_csv.name,
        ],
    }

    with (OUTPUT_DIR / "topic_review_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    elapsed = (time.perf_counter() - started) / 60

    print("\n============================================================")
    print("PHASE 9 — PAPER EXTRACTION COMPLETED")
    print("============================================================")
    print(f"Representative papers: {len(paper_rows)}")
    print(f"Topics reviewed: {TOPIC_COUNT}")
    print(f"Elapsed time: {elapsed:.2f} minutes")
    print(f"\nPaper review CSV: {paper_csv}")
    print(f"Topic label template: {label_csv}")
    print(f"Summary: {OUTPUT_DIR / 'topic_review_summary.json'}")
    print("\nHuman interpretation has NOT yet been marked complete.")


if __name__ == "__main__":
    main()
