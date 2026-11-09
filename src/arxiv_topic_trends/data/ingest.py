"""Stream the Kaggle arXiv snapshot and keep the papers for this study.

The source file is read one JSON line at a time. Only papers whose primary
category is ``cs.LG`` and whose v1 submission date is in 2018--2024 are saved.
The original multi-gigabyte file is never loaded into a dataframe or a list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arxiv_topic_trends.config import (
    DATASET_FILENAME,
    DATASET_METADATA_LICENSE,
    DATASET_SLUG,
    DATASET_VERSION,
    FILTERED_FILENAME,
    FIRST_SUBMISSION_YEAR,
    LAST_SUBMISSION_YEAR,
    MANIFEST_FILENAME,
    PRIMARY_CATEGORY,
    PROGRESS_EVERY_LINES,
    RAW_DATA_DIRECTORY,
    PROCESSED_DATA_DIRECTORY,
    IngestionSettings,
)
from arxiv_topic_trends.data.record_helpers import (
    compact_paper_record,
    first_version_date,
    primary_category,
)


def _write_json_line(file_handle: Any, record: dict[str, Any]) -> None:
    file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _first_record_summary(record: dict[str, Any]) -> dict[str, Any]:
    versions = record.get("versions")
    first_created = None
    if isinstance(versions, list):
        for entry in versions:
            if isinstance(entry, dict) and entry.get("version") == "v1":
                first_created = entry.get("created")
                break

    return {
        "id": record.get("id"),
        "categories": record.get("categories"),
        "versions_is_list": isinstance(versions, list),
        "versions_count": len(versions) if isinstance(versions, list) else None,
        "v1_created": first_created,
    }


def _write_json_file(path: Path, contents: dict[str, Any]) -> None:
    """Write a JSON file through a temporary file to avoid half-written manifests."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(contents, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def stream_arxiv_snapshot(
    settings: IngestionSettings,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a filtered JSONL file and a manifest from the full snapshot."""
    source_path = Path(settings.input_path).expanduser().resolve()
    output_path = Path(settings.output_path).expanduser().resolve()
    manifest_path = output_path.parent / MANIFEST_FILENAME

    if not source_path.is_file():
        raise FileNotFoundError(f"The input file does not exist: {source_path}")
    if source_path == output_path:
        raise ValueError("The source and output paths must be different.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"The output already exists: {output_path}. Use --overwrite only if you intend to replace it."
        )
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"The manifest already exists: {manifest_path}. Use --overwrite to replace the previous run."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    source_hash = hashlib.sha256()
    first_record: dict[str, Any] | None = None
    temporary_output: Path | None = None

    try:
        with source_path.open("rb") as source_file:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=output_path.parent,
                prefix=f"{output_path.stem}_",
                suffix=".partial",
                delete=False,
            ) as output_file:
                temporary_output = Path(output_file.name)

                for line_number, raw_line in enumerate(source_file, start=1):
                    source_hash.update(raw_line)
                    counts["lines_read"] += 1

                    if not raw_line.strip():
                        counts["blank_lines"] += 1
                        continue

                    try:
                        record = json.loads(raw_line.decode("utf-8-sig"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        counts["malformed_json_lines"] += 1
                        continue

                    if not isinstance(record, dict):
                        counts["non_object_json_lines"] += 1
                        continue

                    if first_record is None:
                        first_record = _first_record_summary(record)

                    category = primary_category(record.get("categories"))
                    if category is None:
                        counts["missing_categories"] += 1
                        continue
                    if category != settings.primary_category:
                        counts["not_target_primary_category"] += 1
                        continue
                    counts["target_primary_category"] += 1

                    try:
                        submitted_on = first_version_date(record)
                    except ValueError:
                        counts["missing_or_invalid_v1_date"] += 1
                        continue

                    if not settings.start_year <= submitted_on.year <= settings.end_year:
                        counts["outside_submission_year_range"] += 1
                        continue
                    counts["target_category_and_year"] += 1

                    compact_record = compact_paper_record(record, category, submitted_on)
                    if compact_record is None:
                        counts["missing_required_paper_fields"] += 1
                        continue

                    _write_json_line(output_file, compact_record)
                    counts["records_written"] += 1

                    if line_number % settings.progress_every_lines == 0:
                        print(
                            f"Read {line_number:,} lines; kept {counts['records_written']:,} papers...",
                            flush=True,
                        )

        # Only publish the final filename after the complete scan has succeeded.
        os.replace(temporary_output, output_path)
        temporary_output = None

    finally:
        if temporary_output is not None and temporary_output.exists():
            temporary_output.unlink()

    manifest = {
        "run_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "slug": DATASET_SLUG,
            "version": settings.dataset_version,
            "version_date": settings.dataset_version_date,
            "metadata_license": DATASET_METADATA_LICENSE,
        },
        "source": {
            "filename": source_path.name,
            "size_bytes": source_path.stat().st_size,
            "sha256": source_hash.hexdigest(),
        },
        "selection": {
            "primary_category": settings.primary_category,
            "category_rule": "The first whitespace-separated category must match exactly.",
            "start_year": settings.start_year,
            "end_year": settings.end_year,
            "year_rule": "Use created from the version explicitly named v1.",
        },
        "output": {
            "filename": output_path.name,
            "size_bytes": output_path.stat().st_size,
            "format": "JSON Lines (one compact paper record per line)",
        },
        "first_source_record_check": first_record,
        "counts": dict(counts),
    }
    _write_json_file(manifest_path, manifest)

    print("\nFinished filtering the arXiv snapshot.")
    print(f"Papers kept: {counts['records_written']:,}")
    print(f"Filtered data: {output_path}")
    print(f"Run manifest: {manifest_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep primary cs.LG arXiv papers first submitted from 2018 to 2024."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_DIRECTORY / DATASET_FILENAME,
        help="Location of the downloaded Kaggle JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIRECTORY / FILTERED_FILENAME,
        help="Where to write the smaller filtered JSONL file.",
    )
    parser.add_argument("--category", default=PRIMARY_CATEGORY)
    parser.add_argument("--start-year", type=int, default=FIRST_SUBMISSION_YEAR)
    parser.add_argument("--end-year", type=int, default=LAST_SUBMISSION_YEAR)
    parser.add_argument("--dataset-version", type=int, default=DATASET_VERSION)
    parser.add_argument(
        "--dataset-version-date",
        default=None,
        help="Exact Kaggle publication date, after it has been verified.",
    )
    parser.add_argument(
        "--progress-every-lines",
        type=int,
        default=PROGRESS_EVERY_LINES,
        help="Print a progress message after this many source lines.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing filtered file and manifest. Use deliberately.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = IngestionSettings(
            input_path=args.input,
            output_path=args.output,
            primary_category=args.category,
            start_year=args.start_year,
            end_year=args.end_year,
            dataset_version=args.dataset_version,
            dataset_version_date=args.dataset_version_date,
            progress_every_lines=args.progress_every_lines,
        )
        stream_arxiv_snapshot(settings, overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
