import csv
import json
import time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]

SOURCE = ROOT / "data" / "processed" / "cs_lg_2018_2024.jsonl"
INPUT = ROOT / "outputs" / "reports" / "topic_review_k10" / "nmf_k10_validation_representative_papers.csv"
OUTPUT = ROOT / "outputs" / "reports" / "topic_review_k10" / "nmf_k10_validation_representative_papers_enriched.csv"


def main():
    started = time.perf_counter()

    for path in (SOURCE, INPUT):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")

    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if len(rows) != 50:
        raise ValueError(f"Expected 50 papers, found {len(rows)}.")

    wanted = {
        row["arxiv_id"].strip()
        for row in rows
        if row.get("arxiv_id")
    }

    if len(wanted) != 50:
        raise ValueError("Expected 50 unique arXiv IDs.")

    print("\nPHASE 9 — METADATA RESTORATION", flush=True)
    print("Using submission_year, submitted_date and primary_category.")

    metadata = {}
    scanned = 0

    with SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            scanned += 1
            record = json.loads(line)
            paper_id = str(record.get("id", "")).strip()

            if paper_id in wanted:
                year = record.get("submission_year")
                submitted_date = record.get("submitted_date")
                primary_category = record.get("primary_category")
                categories = record.get("categories")

                if year is None or not str(year).strip():
                    raise ValueError(f"Missing submission_year for {paper_id}.")

                year = int(year)

                if year < 2018 or year > 2024:
                    raise ValueError(f"Unexpected submission year for {paper_id}: {year}")

                if not submitted_date:
                    raise ValueError(f"Missing submitted_date for {paper_id}.")

                if primary_category != "cs.LG":
                    raise ValueError(
                        f"Unexpected primary category for {paper_id}: {primary_category}"
                    )

                metadata[paper_id] = {
                    "year": year,
                    "submitted_date": submitted_date,
                    "primary_category": primary_category,
                    "categories": categories,
                }

                print(
                    f"  Matched {len(metadata)}/50 "
                    f"| scanned {scanned:,} records",
                    flush=True,
                )

                if len(metadata) == 50:
                    break

    missing = wanted - set(metadata)

    if missing:
        raise ValueError(
            "Unmatched paper IDs: " + ", ".join(sorted(missing))
        )

    for column in ("year", "submitted_date", "primary_category", "categories"):
        if column not in fieldnames:
            fieldnames.append(column)

    for row in rows:
        paper_id = row["arxiv_id"].strip()
        info = metadata[paper_id]

        row["year"] = info["year"]
        row["submitted_date"] = info["submitted_date"]
        row["primary_category"] = info["primary_category"]
        row["categories"] = info["categories"]

    year_counts = Counter(str(row["year"]) for row in rows)
    category_counts = Counter(row["primary_category"] for row in rows)

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = (time.perf_counter() - started) / 60

    print("\n================================================")
    print("METADATA RESTORATION COMPLETED")
    print("================================================")
    print(f"Matched papers: {len(metadata)}/50")
    print(f"Missing years: 0")
    print(f"Missing submission dates: 0")
    print(f"Missing primary categories: 0")
    print(f"Category distribution: {dict(category_counts)}")
    print(f"Year distribution: {dict(sorted(year_counts.items()))}")
    print(f"Elapsed time: {elapsed:.2f} minutes")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
