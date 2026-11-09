import json
from pathlib import Path

import pytest

from arxiv_topic_trends.config import IngestionSettings
from arxiv_topic_trends.data.ingest import stream_arxiv_snapshot


def write_record(file_handle, paper_id, categories, submitted, *, abstract="A short abstract"):
    record = {
        "id": paper_id,
        "title": f"Title for {paper_id}",
        "abstract": abstract,
        "authors": "Example Author",
        "categories": categories,
        "versions": [{"version": "v1", "created": submitted}],
        "update_date": "2024-01-02",
    }
    file_handle.write(json.dumps(record) + "\n")


def test_streaming_keeps_only_primary_cs_lg_papers_from_selected_years(tmp_path: Path):
    source = tmp_path / "arxiv-sample.jsonl"
    output = tmp_path / "processed" / "selected.jsonl"

    with source.open("w", encoding="utf-8") as file_handle:
        write_record(file_handle, "paper-2018", "cs.LG cs.AI", "Mon, 1 Jan 2018 10:00:00 GMT")
        write_record(file_handle, "cross-list", "cs.AI cs.LG", "Tue, 1 Jan 2019 10:00:00 GMT")
        write_record(file_handle, "paper-2024", "cs.LG", "2024-03-12T10:30:00")
        write_record(file_handle, "too-new", "cs.LG", "Wed, 1 Jan 2025 10:00:00 GMT")
        write_record(file_handle, "no-abstract", "cs.LG", "Mon, 1 Jan 2020 10:00:00 GMT", abstract=" ")
        file_handle.write("{this is not valid json}\n")
        file_handle.write("[]\n")

    settings = IngestionSettings(
        input_path=source,
        output_path=output,
        primary_category="cs.LG",
        start_year=2018,
        end_year=2024,
        dataset_version=311,
        progress_every_lines=100,
    )
    manifest = stream_arxiv_snapshot(settings)

    written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [paper["id"] for paper in written] == ["paper-2018", "paper-2024"]
    assert manifest["dataset"]["version"] == 311
    assert manifest["counts"]["records_written"] == 2
    assert manifest["counts"]["not_target_primary_category"] == 1
    assert manifest["counts"]["outside_submission_year_range"] == 1
    assert manifest["counts"]["missing_required_paper_fields"] == 1
    assert manifest["counts"]["malformed_json_lines"] == 1
    assert manifest["counts"]["non_object_json_lines"] == 1
    assert len(manifest["source"]["sha256"]) == 64
    assert (output.parent / "ingestion_manifest.json").is_file()


def test_existing_output_is_not_replaced_without_permission(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "filtered.jsonl"
    output.write_text("keep this file\n", encoding="utf-8")

    settings = IngestionSettings(input_path=source, output_path=output)
    with pytest.raises(FileExistsError):
        stream_arxiv_snapshot(settings)

    assert output.read_text(encoding="utf-8") == "keep this file\n"


def test_source_and_output_must_be_different(tmp_path: Path):
    source = tmp_path / "same.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    settings = IngestionSettings(input_path=source, output_path=source)

    with pytest.raises(ValueError, match="must be different"):
        stream_arxiv_snapshot(settings)


def test_missing_source_is_reported(tmp_path: Path):
    settings = IngestionSettings(
        input_path=tmp_path / "missing.jsonl",
        output_path=tmp_path / "filtered.jsonl",
    )
    with pytest.raises(FileNotFoundError):
        stream_arxiv_snapshot(settings)
