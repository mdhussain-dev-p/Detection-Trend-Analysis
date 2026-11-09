"""Clean arXiv abstracts while retaining their original text for review."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import re
from collections import Counter
from pathlib import Path

from html import unescape

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def clean_abstract(
    abstract: str,
    stop_words: set[str],
    lemmatizer: WordNetLemmatizer,
) -> list[str]:
    """Tokenise, remove general English stopwords and lemmatise nouns."""

    if not isinstance(abstract, str) or not abstract.strip():
        return []

    text = unescape(abstract).lower()
    tokens = TOKEN_PATTERN.findall(text)

    cleaned = []

    for token in tokens:
        # Remove numeric-only and single-character tokens.
        if len(token) < 2 or not any(character.isalpha() for character in token):
            continue

        if token in stop_words:
            continue

        cleaned.append(lemmatizer.lemmatize(token))

    return cleaned


def preprocess_dataset(input_path: Path, output_path: Path) -> dict:
    """Process one paper at a time and preserve the source abstracts."""

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {output_path}"
        )

    source_manifest_path = input_path.parent / "ingestion_manifest.json"
    manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8-sig")
    )
    expected_records = manifest["counts"]["records_written"]

    english_stopwords = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")

    if temporary_path.exists():
        raise FileExistsError(
            f"Temporary output already exists: {temporary_path}"
        )

    token_frequencies = Counter()
    token_counts = []
    records_processed = 0
    empty_abstracts = 0
    empty_after_cleaning = 0

    try:
        with input_path.open("r", encoding="utf-8") as source, \
             temporary_path.open("w", encoding="utf-8", newline="\n") as target:

            for line_number, line in enumerate(source, start=1):
                try:
                    paper = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at input line {line_number}"
                    ) from error

                abstract = paper.get("abstract", "")

                if not abstract.strip():
                    empty_abstracts += 1

                tokens = clean_abstract(
                    abstract,
                    english_stopwords,
                    lemmatizer,
                )

                if not tokens:
                    empty_after_cleaning += 1

                token_counts.append(len(tokens))
                token_frequencies.update(tokens)
                records_processed += 1

                processed_record = {
                    "id": paper["id"],
                    "title": paper["title"],
                    "abstract": abstract,
                    "submitted_date": paper["submitted_date"],
                    "submission_year": paper["submission_year"],
                    "primary_category": paper["primary_category"],
                    "tokens": tokens,
                    "token_count": len(tokens),
                }

                target.write(
                    json.dumps(processed_record, ensure_ascii=False) + "\n"
                )

                if line_number % 10000 == 0:
                    print(
                        f"Preprocessed {line_number:,} / "
                        f"{expected_records:,} papers...",
                        flush=True,
                    )

        if records_processed != expected_records:
            raise ValueError(
                f"Expected {expected_records:,} records, "
                f"processed {records_processed:,}."
            )

        temporary_path.replace(output_path)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    summary = {
        "dataset_version": manifest["dataset"]["version"],
        "records_processed": records_processed,
        "empty_original_abstracts": empty_abstracts,
        "empty_after_cleaning": empty_after_cleaning,
        "total_tokens": sum(token_counts),
        "mean_tokens_per_abstract": round(
            statistics.mean(token_counts), 2
        ) if token_counts else 0,
        "median_tokens_per_abstract": statistics.median(
            token_counts
        ) if token_counts else 0,
        "vocabulary_size_before_vectorizer_filtering": len(token_frequencies),
        "tokenisation": "Lowercase alphanumeric tokens",
        "lemmatisation": "WordNet noun lemmatisation",
        "stopwords": "NLTK English stopwords",
        "domain_stopwords_removed": [],
        "top_30_terms": [
            {"term": word, "frequency": count}
            for word, count in token_frequencies.most_common(30)
        ],
        "output_file": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
    }

    reports = Path("outputs/reports")
    metrics = Path("outputs/metrics")
    reports.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)

    (reports / "nlp_preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (metrics / "top_terms_after_cleaning.csv").open(
        "w", newline="", encoding="utf-8"
    ) as result:
        writer = csv.writer(result)
        writer.writerow(["term", "frequency"])
        writer.writerows(token_frequencies.most_common(100))

    print("\n========== PREPROCESSING RESULTS ==========")
    print(f"Records processed: {records_processed:,}")
    print(f"Original empty abstracts: {empty_abstracts:,}")
    print(f"Empty after cleaning: {empty_after_cleaning:,}")
    print(f"Total tokens: {sum(token_counts):,}")
    print(f"Mean tokens: {summary['mean_tokens_per_abstract']}")
    print(f"Median tokens: {summary['median_tokens_per_abstract']}")
    print(f"Vocabulary size: {len(token_frequencies):,}")

    print("\nTop 30 terms:")
    for word, count in token_frequencies.most_common(30):
        print(f"  {word}: {count:,}")

    print(f"\nProcessed dataset: {output_path}")
    print(
        "Summary: outputs/reports/nlp_preprocessing_summary.json"
    )
    print("Preprocessing completed successfully.")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    preprocess_dataset(arguments.input, arguments.output)


if __name__ == "__main__":
    main()
