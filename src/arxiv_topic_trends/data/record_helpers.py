"""Small functions for interpreting an arXiv metadata record."""

from __future__ import annotations

from datetime import date
from email.utils import parsedate_to_datetime
from typing import Any


def parse_created_date(value: str) -> date:
    """Convert an arXiv version ``created`` value into a calendar date.

    arXiv metadata commonly uses a mail-date format such as
    ``Mon, 2 Apr 2007 19:18:42 GMT``. ISO dates are accepted as well, which
    makes the helper easier to test and more tolerant of equivalent exports.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("The version created date is missing or is not text.")

    cleaned = value.strip()
    try:
        parsed = parsedate_to_datetime(cleaned)
        if parsed is not None:
            return parsed.date()
    except (TypeError, ValueError, OverflowError):
        pass

    try:
        return date.fromisoformat(cleaned[:10])
    except ValueError as error:
        raise ValueError(f"Could not read the arXiv created date: {value!r}") from error


def first_version_date(record: dict[str, Any]) -> date:
    """Return the date from the version explicitly named ``v1``.

    We do not fall back to ``update_date`` or another version. Doing that
    would change the project's definition of a paper's submission year.
    """
    versions = record.get("versions")
    if not isinstance(versions, list):
        raise ValueError("The record does not contain a versions list.")

    for version_entry in versions:
        if not isinstance(version_entry, dict):
            continue
        if version_entry.get("version") == "v1":
            created = version_entry.get("created")
            if not isinstance(created, str):
                raise ValueError("The v1 entry does not contain a text created date.")
            return parse_created_date(created)

    raise ValueError("The record has no version explicitly named v1.")


def primary_category(categories: Any) -> str | None:
    """Read the first category token, which this project treats as primary."""
    if not isinstance(categories, str) or not categories.strip():
        return None
    return categories.split()[0]


def compact_paper_record(
    record: dict[str, Any],
    category: str,
    submitted_on: date,
) -> dict[str, Any] | None:
    """Keep only the fields needed for topic modelling and trend analysis."""
    paper_id = record.get("id")
    title = record.get("title")
    abstract = record.get("abstract")

    if not isinstance(paper_id, str) or not paper_id.strip():
        return None
    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(abstract, str) or not abstract.strip():
        return None

    authors = record.get("authors")
    if not isinstance(authors, str):
        authors = ""

    all_categories = record.get("categories")
    if not isinstance(all_categories, str):
        all_categories = ""

    updated = record.get("update_date")
    if not isinstance(updated, str):
        updated = None

    return {
        "id": paper_id.strip(),
        "title": " ".join(title.split()),
        "abstract": " ".join(abstract.split()),
        "authors": " ".join(authors.split()),
        "categories": " ".join(all_categories.split()),
        "primary_category": category,
        "submitted_date": submitted_on.isoformat(),
        "submission_year": submitted_on.year,
        "update_date": updated,
    }
