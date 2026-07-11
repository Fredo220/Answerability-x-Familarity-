import json

import pytest

from trajectory_extractor.datasets.real_transfer import (
    build_wikidata_transfer_rows,
    load_documented_triples,
)


def test_real_transfer_requires_documented_rows_and_builds_frozen_splits(tmp_path):
    path = tmp_path / "triples.jsonl"
    rows = [
        {
            "id": f"triple-{index}",
            "subject": f"Subject {index}",
            "relation": "created",
            "object": f"Object {index}",
            "source_url": f"https://example.test/{index}",
            "distractors": [f"Other {index} created Wrong {index}."],
            "distractor_answers": [f"Wrong {index}"],
        }
        for index in range(10)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    triples = load_documented_triples(path, limit=10)
    assert [sum(item.split == split for item in triples) for split in ("train", "val", "test")] == [6, 2, 2]
    assert triples[0].distractor_answers == ("Wrong 0",)


def test_real_transfer_rejects_missing_rows(tmp_path):
    path = tmp_path / "triples.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "one",
                "subject": "S",
                "relation": "made",
                "object": "O",
                "source_url": "https://example.test",
            }
        )
    )
    with pytest.raises(ValueError, match="at least 2"):
        load_documented_triples(path, limit=2)


def test_wikidata_builder_creates_nearest_name_distractors():
    bindings = []
    for index in range(8):
        bindings.append(
            {
                "subject": {"value": f"http://www.wikidata.org/entity/Q{index + 1}"},
                "subjectLabel": {"value": f"Alex Name{index}"},
                "object": {"value": f"http://www.wikidata.org/entity/Q{index + 100}"},
                "objectLabel": {"value": f"Place {index}"},
            }
        )
    rows = build_wikidata_transfer_rows(bindings, limit=3)
    assert len(rows) == 3
    assert rows[0]["relation"] == "place of birth"
    assert len(rows[0]["distractors"]) == 4
    assert all(" | place of birth | " in fact for fact in rows[0]["distractors"])
    assert rows[0]["object"] not in rows[0]["distractor_answers"]
