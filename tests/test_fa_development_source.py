from __future__ import annotations

import hashlib
import json
import math

import pytest

from trajectory_extractor.fa_confirmatory_source import (
    REGISTERED_DOMAINS,
    SourceRecord,
)
from trajectory_extractor.fa_development_source import (
    DEVELOPMENT_SPLITS,
    ERROR_TAXONOMY,
    DevelopmentSourceDesign,
    assign_development_pools,
    audit_development_source,
    build_manual_error_audit_packet,
    compile_manual_error_audit,
    filter_development_matchable_records,
    main,
    materialize_development_manifests,
    summarize_screening_yield,
    write_development_source,
)


def _record(domain: str, index: int) -> SourceRecord:
    domain_offset = REGISTERED_DOMAINS.index(domain) * 10_000
    return SourceRecord(
        qid=f"Q{100_000 + domain_offset + index}",
        label=f"{domain.title()} Name {index}",
        domain=domain,
        sitelinks=1_000 - index,
        source_rank=index,
        property_values=(
            ("P1", (f"First {index}",)),
            ("P2", (f"Second {index}",)),
            ("P3", (f"Third {index}",)),
        ),
    )


def _records(count: int = 8):
    return {
        domain: tuple(_record(domain, index) for index in range(1, count + 1))
        for domain in REGISTERED_DOMAINS
    }


def _design(count: int = 2) -> DevelopmentSourceDesign:
    return DevelopmentSourceDesign(
        revision="fa-development-source-v6-test",
        split_seed=20260725,
        candidates_per_domain_per_split=count,
    )


class _WordTokenizer:
    name_or_path = "development-test-tokenizer"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def test_assignment_is_balanced_deterministic_and_excludes_prior_qids():
    records = _records()
    excluded = {
        records["person"][0].qid,
        records["place"][1].qid,
    }

    assigned = assign_development_pools(
        {domain: tuple(reversed(rows)) for domain, rows in records.items()},
        _design(),
        excluded_qids=excluded,
    )
    replay = assign_development_pools(
        records,
        _design(),
        excluded_qids=excluded,
    )

    assert tuple(assigned) == DEVELOPMENT_SPLITS
    assert assigned == replay
    assert not excluded.intersection(
        row.qid for rows in assigned.values() for row in rows
    )
    for split in DEVELOPMENT_SPLITS:
        assert {
            domain: sum(row.domain == domain for row in assigned[split])
            for domain in REGISTERED_DOMAINS
        } == {domain: 2 for domain in REGISTERED_DOMAINS}


def test_assignment_fails_closed_when_a_domain_is_short():
    records = _records()
    records["place"] = records["place"][:3]

    with pytest.raises(ValueError, match="place.*requires 4"):
        assign_development_pools(records, _design())


def test_matchability_filter_precedes_assignment_and_returns_exact_frame():
    selected, audit = filter_development_matchable_records(
        _records(count=5),
        _WordTokenizer(),
        required_per_domain=4,
    )

    assert {domain: len(rows) for domain, rows in selected.items()} == {
        domain: 4 for domain in REGISTERED_DOMAINS
    }
    assert audit["eligible_counts"] == {domain: 5 for domain in REGISTERED_DOMAINS}
    assert audit["complete_matchable_counts"] == {
        domain: 4 for domain in REGISTERED_DOMAINS
    }
    assert audit["global_alias_collision_policy"] == "selected_source_names_and_aliases"
    assert len(audit["policy_sha256"]) == 64


def test_manifests_use_development_identity_and_three_questions():
    design = _design()
    manifests = materialize_development_manifests(
        assign_development_pools(_records(), design),
        design=design,
        retrieval_date="2026-07-25",
        query_hashes={domain: domain * 4 for domain in REGISTERED_DOMAINS},
    )

    for split, (candidates, questions) in manifests.items():
        assert len(candidates) == 8
        assert len(questions) == 24
        assert all(
            candidate.entity_id.startswith(f"development-v6-{split}-")
            for candidate in candidates
        )
        assert all(
            "confirmatory-" not in candidate.entity_id for candidate in candidates
        )
        assert all(
            design.revision in candidate.source_provenance for candidate in candidates
        )
        assert {question.qid for question in questions} == {
            candidate.qid for candidate in candidates
        }


def test_audit_rejects_cross_split_qid_leakage_and_returns_semantic_hash():
    design = _design()
    manifests = materialize_development_manifests(
        assign_development_pools(_records(), design),
        design=design,
        retrieval_date="2026-07-25",
        query_hashes={domain: domain * 4 for domain in REGISTERED_DOMAINS},
    )

    audit = audit_development_source(manifests, design=design)
    assert audit["candidate_count"] == 16
    assert audit["question_count"] == 48
    assert len(audit["semantic_sha256"]) == 64

    first_split, second_split = DEVELOPMENT_SPLITS
    removed_candidate = manifests[second_split][0][-1]
    leaked_candidate = next(
        candidate
        for candidate in manifests[first_split][0]
        if candidate.coarse_type == removed_candidate.coarse_type
    )
    corrupted = dict(manifests)
    corrupted[second_split] = (
        (*manifests[second_split][0][:-1], leaked_candidate),
        manifests[second_split][1],
    )
    with pytest.raises(ValueError, match="cross-split QID leakage"):
        audit_development_source(corrupted, design=design)


def test_yield_summary_reports_score_distribution_domain_rates_and_wilson_bounds():
    design = _design(count=1)
    manifests = materialize_development_manifests(
        assign_development_pools(_records(count=3), design),
        design=design,
        retrieval_date="2026-07-25",
        query_hashes={domain: domain * 4 for domain in REGISTERED_DOMAINS},
    )
    candidates = manifests["instrument_development"][0]
    completions_by_entity = {}
    for index, candidate in enumerate(candidates):
        aliases = [values[0] for values in candidate.screening_aliases]
        if index == 0:
            completions_by_entity[candidate.entity_id] = aliases
        elif index == 1:
            completions_by_entity[candidate.entity_id] = (*aliases[:2], "wrong")
        elif index == 2:
            completions_by_entity[candidate.entity_id] = (aliases[0], "wrong", "wrong")
        else:
            completions_by_entity[candidate.entity_id] = ("wrong",) * 3

    summary = summarize_screening_yield(candidates, completions_by_entity)

    assert summary["score_distribution"] == {
        "0_of_3": 1,
        "1_of_3": 1,
        "2_of_3": 1,
        "3_of_3": 1,
    }
    assert summary["qualified_count"] == 2
    assert math.isclose(summary["qualification_rate"], 0.5)
    assert summary["qualification_interval"]["lower"] < 0.5
    assert summary["qualification_interval"]["upper"] > 0.5
    assert set(summary["by_domain"]) == set(REGISTERED_DOMAINS)
    assert set(summary["by_relation"]) == {
        "P17",
        "P19",
        "P27",
        "P57",
        "P106",
        "P131",
        "P159",
        "P421",
        "P495",
        "P571",
        "P577",
    }
    assert summary["question_position_success"] == [0.75, 0.5, 0.25]


def test_yield_summary_requires_exactly_one_completion_triplet_per_candidate():
    design = _design(count=1)
    manifests = materialize_development_manifests(
        assign_development_pools(_records(count=3), design),
        design=design,
        retrieval_date="2026-07-25",
        query_hashes={domain: domain * 4 for domain in REGISTERED_DOMAINS},
    )
    candidates = manifests["instrument_development"][0]

    with pytest.raises(ValueError, match="completion identities"):
        summarize_screening_yield(candidates, {})


def test_manual_error_packet_is_deterministic_stratified_and_blinded():
    items = []
    for domain in REGISTERED_DOMAINS:
        for index in range(4):
            items.append(
                {
                    "question_id": f"{domain}-{index}",
                    "entity_id": f"{domain}-entity-{index}",
                    "qid": (
                        f"Q{700_000 + REGISTERED_DOMAINS.index(domain) * 10 + index}"
                    ),
                    "domain": domain,
                    "prompt": f"Question {index}?",
                    "completion": "An answer",
                    "accepted_aliases": ["Expected"],
                    "is_correct": index == 0,
                    "qualifies": False,
                }
            )

    packet = build_manual_error_audit_packet(
        tuple(reversed(items)),
        sample_per_domain=2,
        seed=20260725,
    )
    replay = build_manual_error_audit_packet(
        items,
        sample_per_domain=2,
        seed=20260725,
    )

    assert packet == replay
    assert len(packet) == 8
    assert {
        domain: sum(row["domain"] == domain for row in packet)
        for domain in REGISTERED_DOMAINS
    } == {domain: 2 for domain in REGISTERED_DOMAINS}
    assert all("is_correct" not in row and "qualifies" not in row for row in packet)
    assert all(
        set(row["allowed_error_labels"]) == set(ERROR_TAXONOMY) for row in packet
    )


def test_manual_error_audit_requires_two_raters_and_independent_adjudication():
    items = [
        {
            "question_id": f"{domain}-0",
            "entity_id": f"{domain}-entity-0",
            "qid": f"Q{800_000 + index}",
            "domain": domain,
            "prompt": "Question?",
            "completion": "Answer",
            "accepted_aliases": ["Expected"],
            "is_correct": False,
        }
        for index, domain in enumerate(REGISTERED_DOMAINS)
    ]
    packet = build_manual_error_audit_packet(
        items,
        sample_per_domain=1,
        seed=20260725,
    )
    ratings = []
    for row in packet:
        ratings.extend(
            [
                {
                    "audit_id": row["audit_id"],
                    "rater_id": "rater-a",
                    "round": 1,
                    "error_label": "entity_unknown",
                },
                {
                    "audit_id": row["audit_id"],
                    "rater_id": "rater-b",
                    "round": 1,
                    "error_label": "entity_unknown",
                },
            ]
        )
    result = compile_manual_error_audit(packet, ratings)
    assert result["decision_counts"] == {"entity_unknown": 4}

    ratings[-1]["error_label"] = "relation_unknown"
    with pytest.raises(ValueError, match="adjudicator"):
        compile_manual_error_audit(packet, ratings)
    ratings.append(
        {
            "audit_id": packet[-1]["audit_id"],
            "rater_id": "rater-c",
            "round": 2,
            "error_label": "relation_unknown",
        }
    )
    adjudicated = compile_manual_error_audit(packet, ratings)
    assert adjudicated["adjudicated_count"] == 1


def test_write_development_source_seals_manifests_and_exclusion_lineage(tmp_path):
    design = _design()
    excluded = {"Q99", "Q100"}
    manifests = materialize_development_manifests(
        assign_development_pools(_records(), design),
        design=design,
        retrieval_date="2026-07-25",
        query_hashes={domain: domain * 4 for domain in REGISTERED_DOMAINS},
    )

    result = write_development_source(
        tmp_path / "development_source_v6",
        manifests,
        design=design,
        excluded_qids=excluded,
    )

    snapshot = json.loads(result["source_snapshot"].read_text())
    integrity = json.loads(result["source_integrity"].read_text())
    assert snapshot["source_revision"] == design.revision
    assert snapshot["splits"] == list(DEVELOPMENT_SPLITS)
    assert snapshot["excluded_qids"] == sorted(excluded)
    assert integrity["source_snapshot_sha256"] == result["source_snapshot_sha256"]
    assert set(integrity["materialized_files"]) == set(DEVELOPMENT_SPLITS)
    assert all(
        "confirmatory" not in values["candidate_manifest"]
        for values in integrity["materialized_files"].values()
    )

    replay = write_development_source(
        tmp_path / "development_source_v6",
        manifests,
        design=design,
        excluded_qids=excluded,
    )
    assert replay["source_snapshot_sha256"] == result["source_snapshot_sha256"]


def test_write_development_source_refuses_nonidentical_overwrite(tmp_path):
    design = _design()
    manifests = materialize_development_manifests(
        assign_development_pools(_records(), design),
        design=design,
        retrieval_date="2026-07-25",
        query_hashes={domain: domain * 4 for domain in REGISTERED_DOMAINS},
    )
    output_dir = tmp_path / "development_source_v6"
    write_development_source(output_dir, manifests, design=design)
    candidate_path = output_dir / "candidate_entities_instrument_development_v1.json"
    candidate_path.write_text("[]\n")

    with pytest.raises(FileExistsError, match="non-identical"):
        write_development_source(output_dir, manifests, design=design)


def test_standalone_cli_materializes_development_source_from_open_frame(
    tmp_path,
    capsys,
):
    records = _records(count=3)
    frame_path = tmp_path / "source_frame.json"
    frame_path.write_text(
        json.dumps(
            (
                lambda payload: {
                    **payload,
                    "frame_payload_sha256": hashlib.sha256(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode()
                    ).hexdigest(),
                }
            )(
                {
                    "schema_version": 1,
                    "source_revision": "open-development-frame-test",
                    "retrieval_date": "2026-07-25",
                    "required_per_domain": 3,
                    "query_sha256s": {
                        domain: domain * 4 for domain in REGISTERED_DOMAINS
                    },
                    "matchability_audit": {"policy_sha256": "a" * 64},
                    "records_by_domain": {
                        domain: [
                            {
                                **record.__dict__,
                                "property_values": [
                                    [property_id, list(aliases)]
                                    for property_id, aliases in record.property_values
                                ],
                            }
                            for record in domain_records
                        ]
                        for domain, domain_records in records.items()
                    },
                }
            )
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "development_source_v6"

    assert (
        main(
            [
                "materialize",
                "--source-frame",
                str(frame_path),
                "--output-dir",
                str(output_dir),
                "--candidates-per-domain-per-split",
                "1",
                "--split-seed",
                "20260725",
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "materialized"
    assert printed["candidate_count"] == 8
    assert (output_dir / "source_integrity_v1.json").exists()
    snapshot = json.loads((output_dir / "source_snapshot_v1.json").read_text())
    assert len(snapshot["source_frame_sha256"]) == 64
