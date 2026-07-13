import hashlib
import json
from pathlib import Path

import pytest

from trajectory_extractor.rlmf_artifacts import RLMFArtifactStore
from trajectory_extractor.rlmf_data import (
    normalize_popqa_row,
    select_subject_and_answer_disjoint_splits,
    write_popqa_snapshot,
)
from trajectory_extractor.rlmf_types import RLMFConfig


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _row(index, *, aliases=None, subject=None):
    subject = subject or f"https://www.wikidata.org/entity/Q{index}"
    return {
        "id": index,
        "s_uri": subject,
        "subj": f"Subject {index}",
        "question": f"Question {index}?",
        "possible_answers": aliases or [f"Answer {index}"],
        "prop": "occupation",
        "s_pop": index,
    }


def _confirmatory_config():
    return RLMFConfig.from_json(CONFIGS / "rlmf_qwen06b_confirmatory.json")


def test_normalize_popqa_row_canonicalizes_unicode_aliases_and_stable_id():
    row = _row(
        1,
        aliases=["  Cafe\u0301  ", "CAF\u00c9", "Cafe\u0301"],
        subject="https://www.wikidata.org/entity/Q42",
    )

    example = normalize_popqa_row(row)

    assert example.example_id == "popqa-" + hashlib.sha256(
        b"https://www.wikidata.org/entity/Q42"
    ).hexdigest()[:16]
    assert example.subject == "https://www.wikidata.org/entity/Q42"
    assert example.answers == ("caf\u00e9",)


def test_select_splits_deduplicates_subjects_and_answer_components_deterministically():
    rows = [
        _row(1, aliases=["Alpha", "A"], subject="Q-1"),
        _row(2, aliases=["a", "Beta"], subject="Q-2"),
        _row(3, aliases=["Gamma"], subject="Q-3"),
        _row(4, aliases=["Delta"], subject="Q-4"),
        _row(5, aliases=["Epsilon"], subject="Q-5"),
        _row(6, aliases=["Zeta"], subject="Q-6"),
        _row(7, aliases=["Eta"], subject="Q-7"),
        _row(8, aliases=["Theta"], subject="Q-8"),
        _row(9, aliases=["Iota"], subject="Q-9"),
        _row(10, aliases=["Kappa"], subject="Q-10"),
        _row(11, aliases=["Lambda"], subject="Q-11"),
        _row(12, aliases=["Mu"], subject="Q-12"),
        _row(13, aliases=["Nu"], subject="Q-13"),
        _row(14, aliases=["Xi"], subject="Q-14"),
        _row(15, aliases=["Omicron"], subject="Q-15"),
        _row(16, aliases=["Pi"], subject="Q-16"),
        _row(17, aliases=["Rho"], subject="Q-17"),
        _row(18, aliases=["Sigma"], subject="Q-18"),
        _row(19, aliases=["Tau"], subject="Q-19"),
        _row(20, aliases=["Upsilon"], subject="Q-20"),
        _row(21, aliases=["Phi"], subject="Q-21"),
        _row(22, aliases=["Chi"], subject="Q-22"),
        _row(23, aliases=["Psi"], subject="Q-23"),
        _row(24, aliases=["Omega"], subject="Q-24"),
        _row(25, aliases=["Duplicate subject"], subject="Q-3"),
    ]
    examples = [normalize_popqa_row(row) for row in rows]
    counts = {"pre_sft": 8, "rl_train": 8, "validation": 3, "test": 4}

    first = select_subject_and_answer_disjoint_splits(examples, counts, split_seed=17)
    second = select_subject_and_answer_disjoint_splits(
        list(reversed(examples)), counts, split_seed=17
    )

    first_ids = [example.example_id for split in first.values() for example in split]
    second_ids = [example.example_id for split in second.values() for example in split]
    assert first_ids == second_ids
    assert {name: len(rows) for name, rows in first.items()} == counts
    assert len({example.subject for split in first.values() for example in split}) == 23
    aliases_by_split = [
        {alias for example in split for alias in example.answers} for split in first.values()
    ]
    assert all(not left & right for index, left in enumerate(aliases_by_split) for right in aliases_by_split[index + 1 :])
    assert len(set(first_ids) & {
        normalize_popqa_row(_row(1, aliases=["Alpha", "A"], subject="Q-1")).example_id,
        normalize_popqa_row(_row(2, aliases=["a", "Beta"], subject="Q-2")).example_id,
    }) == 1
    assert len([example for split in first.values() for example in split if example.subject == "Q-3"]) == 1


def test_select_splits_uses_aliases_from_discarded_duplicate_subject_rows():
    rows = [
        _row(1, aliases=["Alpha"], subject="Q-A"),
        _row(2, aliases=["Bridge"], subject="Q-A"),
        _row(3, aliases=["Bridge"], subject="Q-B"),
        *[_row(index, subject=f"Q-{index}") for index in range(4, 9)],
    ]
    examples = [normalize_popqa_row(row) for row in rows]
    counts = {"pre_sft": 2, "rl_train": 2, "validation": 1, "test": 1}

    splits = select_subject_and_answer_disjoint_splits(examples, counts, split_seed=17)

    selected_subjects = {
        example.subject for split in splits.values() for example in split
    }
    assert len({"Q-A", "Q-B"} & selected_subjects) == 1


def test_select_splits_uses_hash_order_and_rejects_insufficient_components():
    examples = [normalize_popqa_row(_row(index, subject=f"Q-{index}")) for index in range(1, 9)]
    counts = {"pre_sft": 2, "rl_train": 2, "validation": 1, "test": 2}

    splits = select_subject_and_answer_disjoint_splits(examples, counts, split_seed=23)

    expected = sorted(
        examples,
        key=lambda example: hashlib.sha256(f"23:{example.subject}".encode()).hexdigest(),
    )[:7]
    assert [example.example_id for split in splits.values() for example in split] == [
        example.example_id for example in expected
    ]
    with pytest.raises(ValueError, match="eligible components"):
        select_subject_and_answer_disjoint_splits(examples[:6], counts, split_seed=23)


def test_write_snapshot_is_pinned_auditable_and_marks_completion(tmp_path, monkeypatch):
    import trajectory_extractor.rlmf_data as rlmf_data

    rows = [_row(index, subject=f"Q-{index}") for index in range(1, 901)]
    rows[1] = _row(2, aliases=["Bridge"], subject="Q-2")
    rows.append(_row(901, aliases=["Bridge"], subject="Q-1"))

    def load_dataset(dataset_id, *, revision, split):
        assert dataset_id == "akariasai/PopQA"
        assert revision == "5cf59972d88d4aaaa7781ac91b83d053563d8268"
        assert split == "test"
        return rows

    monkeypatch.setattr(rlmf_data, "load_dataset", load_dataset)
    store = RLMFArtifactStore(tmp_path)
    paths = write_popqa_snapshot(_confirmatory_config(), store)

    assert set(paths) == {
        "source_rows",
        "normalized_rows",
        "aliases",
        "discarded_rows",
        "split_manifest",
        "completion",
    }
    snapshot = [json.loads(line) for line in paths["normalized_rows"].read_text().splitlines()]
    aliases = [json.loads(line) for line in paths["aliases"].read_text().splitlines()]
    source_rows = [json.loads(line) for line in paths["source_rows"].read_text().splitlines()]
    discarded_rows = [json.loads(line) for line in paths["discarded_rows"].read_text().splitlines()]
    manifest = json.loads(paths["split_manifest"].read_text())
    assert len(snapshot) == 896
    assert len(aliases) == 896
    assert manifest["dataset"] == {
        "id": "akariasai/PopQA",
        "revision": "5cf59972d88d4aaaa7781ac91b83d053563d8268",
        "split": "test",
    }
    assert manifest["counts"] == {"pre_sft": 256, "rl_train": 256, "test": 256, "validation": 128}
    assert all("relation_counts" in row and "popularity" in row for row in [manifest])
    assert all(row["alias_component_id"] is not None for row in source_rows)
    assert all(row["alias_component_id"] is not None for row in discarded_rows)
    assert store.verify_endpoint(_confirmatory_config().study_id, "prepare-data")
