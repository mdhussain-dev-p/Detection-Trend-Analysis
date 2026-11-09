import json
from pathlib import Path

from arxiv_topic_trends.data.inspect_snapshot import inspect_first_records


def test_inspection_reads_only_requested_number_of_records(tmp_path: Path):
    source = tmp_path / "sample.jsonl"
    records = [
        {
            "id": f"paper-{index}",
            "categories": "cs.LG cs.AI",
            "versions": [{"version": "v1", "created": "Mon, 1 Jan 2018 10:00:00 GMT"}],
        }
        for index in range(4)
    ]
    source.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")

    result = inspect_first_records(source, number_of_records=2)

    assert result["records_inspected"] == 2
    assert result["records"][0]["v1_submission_date_parsed"] == "2018-01-01"
    assert result["primary_categories_in_sample"] == {"cs.LG": 2}
