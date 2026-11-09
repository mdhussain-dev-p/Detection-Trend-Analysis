from datetime import date

import pytest

from arxiv_topic_trends.data.record_helpers import (
    compact_paper_record,
    first_version_date,
    parse_created_date,
    primary_category,
)


def sample_paper():
    return {
        "id": "1801.00001",
        "title": "  A   Sample Paper  ",
        "abstract": " A short abstract\nwith extra spaces. ",
        "authors": "A. Author and B. Author",
        "categories": "cs.LG cs.AI stat.ML",
        "versions": [
            {"version": "v1", "created": "Mon, 1 Jan 2018 10:00:00 GMT"},
            {"version": "v2", "created": "Tue, 2 Jan 2018 10:00:00 GMT"},
        ],
        "update_date": "2018-01-02",
    }


def test_parse_arxiv_mail_date():
    assert parse_created_date("Mon, 1 Jan 2018 10:00:00 GMT") == date(2018, 1, 1)


def test_parse_iso_date():
    assert parse_created_date("2024-03-12T10:30:00") == date(2024, 3, 12)


def test_invalid_date_is_not_accepted():
    with pytest.raises(ValueError):
        parse_created_date("not a date")


def test_date_is_taken_from_v1_not_the_latest_version():
    assert first_version_date(sample_paper()) == date(2018, 1, 1)


def test_v1_can_be_found_even_when_not_the_first_list_item():
    paper = sample_paper()
    paper["versions"].reverse()
    assert first_version_date(paper) == date(2018, 1, 1)


def test_missing_v1_does_not_fall_back_to_update_date():
    paper = sample_paper()
    paper["versions"] = [{"version": "v2", "created": "Tue, 2 Jan 2018 10:00:00 GMT"}]
    with pytest.raises(ValueError, match="no version explicitly named v1"):
        first_version_date(paper)


def test_missing_or_invalid_versions_is_rejected():
    with pytest.raises(ValueError):
        first_version_date({"versions": None})


def test_first_category_is_primary():
    assert primary_category("cs.LG cs.AI stat.ML") == "cs.LG"
    assert primary_category("cs.AI cs.LG") == "cs.AI"
    assert primary_category("  ") is None
    assert primary_category(None) is None


def test_compact_record_keeps_only_project_fields_and_cleans_whitespace():
    compact = compact_paper_record(sample_paper(), "cs.LG", date(2018, 1, 1))
    assert compact == {
        "id": "1801.00001",
        "title": "A Sample Paper",
        "abstract": "A short abstract with extra spaces.",
        "authors": "A. Author and B. Author",
        "categories": "cs.LG cs.AI stat.ML",
        "primary_category": "cs.LG",
        "submitted_date": "2018-01-01",
        "submission_year": 2018,
        "update_date": "2018-01-02",
    }


def test_compact_record_rejects_missing_required_fields():
    paper = sample_paper()
    paper["abstract"] = "  "
    assert compact_paper_record(paper, "cs.LG", date(2018, 1, 1)) is None
