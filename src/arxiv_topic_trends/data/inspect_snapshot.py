"""Inspect a few JSONL records before starting the full dataset scan."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from arxiv_topic_trends.config import DATASET_FILENAME, RAW_DATA_DIRECTORY
from arxiv_topic_trends.data.record_helpers import first_version_date, primary_category


def inspect_first_records(input_path: Path, number_of_records: int = 5) -> dict[str, Any]:
    """Read at most ``number_of_records`` non-empty lines and summarise them."""
    if number_of_records < 1:
        raise ValueError("The number of records to inspect must be at least 1.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Could not find the dataset file: {input_path}")

    inspected: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    malformed_lines = 0

    with input_path.open("rb") as source:
        for raw_line in source:
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                malformed_lines += 1
                continue

            if not isinstance(record, dict):
                malformed_lines += 1
                continue

            category = primary_category(record.get("categories"))
            if category:
                categories[category] += 1

            versions = record.get("versions")
            v1_created = None
            v1_date = None
            if isinstance(versions, list):
                for entry in versions:
                    if isinstance(entry, dict) and entry.get("version") == "v1":
                        v1_created = entry.get("created")
                        try:
                            v1_date = first_version_date(record).isoformat()
                        except ValueError:
                            v1_date = None
                        break

            inspected.append(
                {
                    "id": record.get("id"),
                    "categories": record.get("categories"),
                    "primary_category": category,
                    "versions_type": type(versions).__name__,
                    "versions_count": len(versions) if isinstance(versions, list) else None,
                    "v1_created": v1_created,
                    "v1_submission_date_parsed": v1_date,
                    "top_level_fields": sorted(record.keys()),
                }
            )
            if len(inspected) >= number_of_records:
                break

    if not inspected:
        raise ValueError("No valid JSON object records were found in the sampled lines.")

    return {
        "file": input_path.name,
        "file_size_bytes": input_path.stat().st_size,
        "records_inspected": len(inspected),
        "malformed_lines_skipped": malformed_lines,
        "primary_categories_in_sample": dict(categories),
        "records": inspected,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a few arXiv metadata JSONL records without loading the full file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_DIRECTORY / DATASET_FILENAME,
        help="Path to arxiv-metadata-oai-snapshot.json",
    )
    parser.add_argument("--records", type=int, default=5, help="Number of valid records to inspect")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = inspect_first_records(arguments.input, arguments.records)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
