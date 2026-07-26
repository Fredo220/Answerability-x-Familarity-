import json
from pathlib import Path

import pytest

from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_confirmatory_source import SourceRecord
from trajectory_extractor.fa_development_r9 import (
    R8_AUDIT_ITEMS_SHA256,
    _balanced_assignment,
    _load_and_verify_inputs,
    _selection_key,
    _valid_correction_surfaces,
    derive_r9_source,
)
from tools.audit_fa_development_r9 import _replay_expected_decisions

R8_ROOT = Path("data/fa/development_source_v6_r8")
CORRECTIONS = Path("data/fa/development_source_v6_r9/alias_corrections_v1.json")


def _record(domain: str, qid: str) -> SourceRecord:
    return SourceRecord(
        qid=qid,
        label=f"{domain} {qid}",
        domain=domain,
        sitelinks=10,
        source_rank=1,
        property_values=(
            ("P1", ("one",)),
            ("P2", ("two",)),
            ("P3", ("three",)),
        ),
    )


def test_r9_frozen_inputs_and_structured_corrections_verify():
    inputs = _load_and_verify_inputs(R8_ROOT, CORRECTIONS)

    assert len(inputs["candidates"]) == 192
    assert len(inputs["questions"]) == 576
    assert len(inputs["corrections"]["items"]) == 118
    assert (
        inputs["corrections"]["source_audit_items_sha256"]
        == R8_AUDIT_ITEMS_SHA256
    )


def test_r9_rejects_tampered_correction_manifest(tmp_path):
    corrections = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    corrections["items"][0]["accepted_surfaces_to_add"].append("tampered")
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps(corrections), encoding="utf-8")

    with pytest.raises(ValueError, match="correction manifest hash"):
        _load_and_verify_inputs(R8_ROOT, path)


def test_r9_derivation_rejects_nonconfirmatory_config(tmp_path):
    config = FAConfig.from_json(
        "configs/familiarity_answerability_qwen06b_smoke.json"
    )

    with pytest.raises(ValueError, match="pinned confirmatory config"):
        derive_r9_source(
            r8_root=R8_ROOT,
            corrections_path=CORRECTIONS,
            output_dir=tmp_path,
            tokenizer=object(),
            config=config,
        )


def test_r9_correction_surfaces_fail_closed():
    assert _valid_correction_surfaces(["USA", "U.S."])
    assert not _valid_correction_surfaces(["UNKNOWN"])
    assert not _valid_correction_surfaces(["USA", "usa"])
    assert not _valid_correction_surfaces([" trailing "])
    assert not _valid_correction_surfaces(["line\nbreak"])
    assert not _valid_correction_surfaces([])
    assert not _valid_correction_surfaces("USA")


def test_r9_selection_and_split_assignment_are_deterministic():
    domains = ("person", "place", "organization", "creative_work")
    selected = {
        domain: tuple(
            _record(domain, f"Q{offset * 100 + index}")
            for index in range(1, 25)
        )
        for offset, domain in enumerate(domains, start=1)
    }
    correction_counts = {
        record.qid: int(index % 2 == 0)
        for records in selected.values()
        for index, record in enumerate(records)
    }

    first = _balanced_assignment(selected, correction_counts)
    second = _balanced_assignment(selected, correction_counts)

    assert first == second
    for domain in domains:
        instrument_count = sum(
            row.domain == domain for row in first["instrument_development"]
        )
        assert instrument_count == 12
        assert (
            sum(row.domain == domain for row in first["construction_validation"])
            == 12
        )
    assert _selection_key("person", "Q1") == (
        "8232f6faae74a4e5ab903f61ad7d358a2f84052c2948a00c07fde4de3721b5ee",
        "Q1",
    )
    assert _selection_key("person", "Q1") != _selection_key("place", "Q1")


def test_independent_r9_auditor_replays_eligibility_selection_and_splits():
    blockers = []

    decisions = _replay_expected_decisions(
        R8_ROOT,
        CORRECTIONS.parent,
        blockers,
    )

    assert blockers == []
    assert len(decisions) == 192
    included = [
        decision
        for decision in decisions.values()
        if decision["decision"] == "included"
    ]
    assert len(included) == 96
    assert {split: sum(row["split"] == split for row in included) for split in (
        "instrument_development",
        "construction_validation",
    )} == {
        "instrument_development": 48,
        "construction_validation": 48,
    }
    assert {
        domain: sum(row["domain"] == domain for row in included)
        for domain in ("creative_work", "organization", "person", "place")
    } == {
        "creative_work": 24,
        "organization": 24,
        "person": 24,
        "place": 24,
    }
