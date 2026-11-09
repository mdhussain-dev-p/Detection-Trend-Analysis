"""Project settings in one place.

Values that have not been verified are intentionally left unset. In
particular, add the exact Kaggle Version 311 publication date only when it
can be read from the Kaggle version history.
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
RAW_DATA_DIRECTORY = DATA_DIRECTORY / "raw"
PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"

DATASET_SLUG = "Cornell-University/arxiv"
DATASET_PAGE = "https://www.kaggle.com/datasets/Cornell-University/arxiv"
DATASET_FILENAME = "arxiv-metadata-oai-snapshot.json"
DATASET_VERSION = 311
DATASET_VERSION_DATE: str | None = None  # Add only after checking Kaggle's version history.
DATASET_METADATA_LICENSE = "CC0-1.0 (metadata only)"

PRIMARY_CATEGORY = "cs.LG"
FIRST_SUBMISSION_YEAR = 2018
LAST_SUBMISSION_YEAR = 2024

FILTERED_FILENAME = "cs_lg_2018_2024.jsonl"
MANIFEST_FILENAME = "ingestion_manifest.json"
PROGRESS_EVERY_LINES = 250_000


@dataclass(frozen=True)
class IngestionSettings:
    """Options for one streaming extraction run."""

    input_path: Path
    output_path: Path
    primary_category: str = PRIMARY_CATEGORY
    start_year: int = FIRST_SUBMISSION_YEAR
    end_year: int = LAST_SUBMISSION_YEAR
    dataset_version: int = DATASET_VERSION
    dataset_version_date: str | None = DATASET_VERSION_DATE
    progress_every_lines: int = PROGRESS_EVERY_LINES

    def __post_init__(self) -> None:
        if not self.primary_category.strip():
            raise ValueError("The primary category cannot be empty.")
        if self.start_year > self.end_year:
            raise ValueError("The start year must be less than or equal to the end year.")
        if self.progress_every_lines < 1:
            raise ValueError("progress_every_lines must be at least 1.")
        if self.dataset_version < 1:
            raise ValueError("dataset_version must be a positive integer.")
