"""Explore the filtered arXiv abstracts without loading the corpus into RAM."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# These are ordinary English function words. Academic terms such as
# "paper", "propose", and "method" are deliberately retained for inspection.
BASIC_STOPWORDS = set("""
a an and are as at be been being by can could did do does doing
for from had has have having he her hers him his how i if in into
is it its itself just may might more most must my no nor not of
on once only or other our ours out over own same she should so
some such than that the their theirs them themselves then there
these they this those through to too under until up very was we
were what when where which while who whom why will with would you
your yours
""".split())

TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9]{2,}")


def percentile(values: list[int], fraction: float) -> float:
    """Return a simple nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def explore_dataset(input_path: Path, project_root: Path) -> dict:
    """Read the filtered corpus once and save reproducible EDA outputs."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Filtered dataset not found: {input_path}")

    manifest_path = input_path.parent / "ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    yearly_counts = Counter()
    word_frequencies = Counter()
    abstract_lengths = []
    title_lengths = []
    seen_ids = set()
    duplicates = 0
    missing_titles = 0
    missing_abstracts = 0
    invalid_records = 0
    total_records = 0

    print("Reading filtered papers one record at a time...", flush=True)

    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                invalid_records += 1
                continue

            if not isinstance(paper, dict):
                invalid_records += 1
                continue

            total_records += 1

            paper_id = paper.get("id")
            if paper_id in seen_ids:
                duplicates += 1
            else:
                seen_ids.add(paper_id)

            title = paper.get("title") or ""
            abstract = paper.get("abstract") or ""

            if not title.strip():
                missing_titles += 1

            if not abstract.strip():
                missing_abstracts += 1

            year = int(paper["submission_year"])
            yearly_counts[year] += 1

            abstract_tokens = TOKEN_PATTERN.findall(abstract.lower())
            title_tokens = TOKEN_PATTERN.findall(title.lower())

            abstract_lengths.append(len(abstract_tokens))
            title_lengths.append(len(title_tokens))

            for token in abstract_tokens:
                if token not in BASIC_STOPWORDS:
                    word_frequencies[token] += 1

            if line_number % 20000 == 0:
                print(
                    f"Examined {line_number:,} papers...",
                    flush=True,
                )

    if not abstract_lengths:
        raise ValueError("No usable abstracts were found.")

    expected = manifest["counts"]["records_written"]

    if total_records != expected:
        raise ValueError(
            f"Record mismatch: read {total_records:,}, manifest says {expected:,}."
        )

    report_directory = project_root / "outputs" / "reports"
    metrics_directory = project_root / "outputs" / "metrics"

    report_directory.mkdir(parents=True, exist_ok=True)
    metrics_directory.mkdir(parents=True, exist_ok=True)

    length_summary = {
        "abstract_words_mean": round(statistics.mean(abstract_lengths), 2),
        "abstract_words_median": statistics.median(abstract_lengths),
        "abstract_words_p90": percentile(abstract_lengths, 0.90),
        "abstract_words_min": min(abstract_lengths),
        "abstract_words_max": max(abstract_lengths),
        "title_words_mean": round(statistics.mean(title_lengths), 2),
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": manifest["dataset"]["version"],
        "source_dataset": manifest["source"]["filename"],
        "selected_category": "cs.LG",
        "selected_years": [2018, 2024],
        "records_examined": total_records,
        "unique_ids": len(seen_ids),
        "duplicate_ids": duplicates,
        "missing_titles": missing_titles,
        "missing_abstracts": missing_abstracts,
        "invalid_records": invalid_records,
        "yearly_paper_counts": {
            str(year): yearly_counts[year]
            for year in range(2018, 2025)
        },
        "text_length_statistics": length_summary,
        "top_30_terms": [
            {"term": word, "frequency": frequency}
            for word, frequency in word_frequencies.most_common(30)
        ],
        "stopword_note": (
            "Only general English function words were removed for this "
            "initial frequency inspection. Domain-specific stopwords "
            "have deliberately not been selected yet."
        ),
    }

    # Save yearly paper counts.
    with (metrics_directory / "yearly_paper_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["year", "paper_count"])
        for year in range(2018, 2025):
            writer.writerow([year, yearly_counts[year]])

    # Save abstract and title length statistics.
    with (metrics_directory / "abstract_length_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["metric", "value"])
        for key, value in length_summary.items():
            writer.writerow([key, value])

    # Save the initial word-frequency table.
    with (metrics_directory / "top_terms_initial.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["term", "frequency"])
        writer.writerows(word_frequencies.most_common(100))

    summary_path = report_directory / "eda_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Exploratory Data Analysis",
        "",
        f"Dataset version: {summary['dataset_version']}",
        f"Records examined: {total_records:,}",
        f"Unique paper IDs: {len(seen_ids):,}",
        f"Duplicate IDs: {duplicates:,}",
        f"Missing titles: {missing_titles:,}",
        f"Missing abstracts: {missing_abstracts:,}",
        f"Invalid records: {invalid_records:,}",
        "",
        "## Papers by submission year",
        "",
        "| Year | Papers |",
        "|---|---:|",
    ]

    for year in range(2018, 2025):
        report_lines.append(f"| {year} | {yearly_counts[year]:,} |")

    report_lines.extend([
        "",
        "## Abstract length",
        "",
        "| Measurement | Words |",
        "|---|---:|",
    ])

    for key, value in length_summary.items():
        report_lines.append(f"| {key.replace('_', ' ').title()} | {value} |")

    report_lines.extend([
        "",
        "## Initial frequent terms",
        "",
        "| Term | Frequency |",
        "|---|---:|",
    ])

    for word, frequency in word_frequencies.most_common(30):
        report_lines.append(f"| {word} | {frequency:,} |")

    report_lines.extend([
        "",
        "These are preliminary frequencies, not final topic labels.",
        "Domain-specific stopwords will be chosen only after reviewing them.",
        "",
    ])

    (report_directory / "eda_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n========== EDA RESULTS ==========")
    print(f"Records examined: {total_records:,}")
    print(f"Unique IDs: {len(seen_ids):,}")
    print(f"Duplicate IDs: {duplicates:,}")
    print(f"Missing titles: {missing_titles:,}")
    print(f"Missing abstracts: {missing_abstracts:,}")
    print(f"Invalid records: {invalid_records:,}")

    print("\nYearly distribution:")
    for year in range(2018, 2025):
        print(f"  {year}: {yearly_counts[year]:,}")

    print("\nAbstract length statistics:")
    for name, value in length_summary.items():
        print(f"  {name}: {value}")

    print("\nTop 30 initial terms:")
    for word, frequency in word_frequencies.most_common(30):
        print(f"  {word}: {frequency:,}")

    print("\nReports saved:")
    print(f"  {summary_path}")
    print(f"  {report_directory / 'eda_report.md'}")
    print(f"  {metrics_directory}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the filtered arXiv corpus.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/cs_lg_2018_2024.jsonl"),
    )
    arguments = parser.parse_args()
    project_root = Path.cwd()

    explore_dataset(arguments.input, project_root)


if __name__ == "__main__":
    main()
