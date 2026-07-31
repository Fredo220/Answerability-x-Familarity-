import pytest

from trajectory_extractor.fa_r11_instrument import (
    assess_frozen_validation,
    compile_r11_human_audit,
    prepare_r11_human_audit_packet,
    select_relation_triplets,
)

DOMAINS = ("creative_work", "organization", "person", "place")
RELATIONS = {
    domain: (f"{domain}-a", f"{domain}-b", f"{domain}-c", f"{domain}-d")
    for domain in DOMAINS
}
BINDINGS = {
    "config_sha256": "a" * 64,
    "source_manifest_sha256": "b" * 64,
    "git_commit": "c" * 40,
    "development_execution_identity_sha256": "d" * 64,
    "screening_identity": {
        "config_sha256": "e" * 64,
        "model_id": "test/model",
        "model_revision": "f" * 40,
        "tokenizer_revision": "f" * 40,
        "chat_template_sha256": "1" * 64,
        "parser_sha256": "2" * 64,
        "semantic_audit_sha256": "3" * 64,
    },
}
VALIDATION_BINDINGS = {
    "validation_execution_identity_sha256": "4" * 64,
    "validation_items_sha256": "5" * 64,
}


def _rows(split: str, *, entity_offset: int = 0) -> list[dict[str, object]]:
    rows = []
    for domain in DOMAINS:
        for index in range(4):
            entity_id = f"{split}-{domain}-{entity_offset + index}"
            outcomes = {
                RELATIONS[domain][0]: True,
                RELATIONS[domain][1]: True,
                RELATIONS[domain][2]: index < 2,
                RELATIONS[domain][3]: False,
            }
            for relation_id, is_correct in outcomes.items():
                rows.append(
                    {
                        "split": split,
                        "domain": domain,
                        "entity_id": entity_id,
                        "qid": (
                            f"Q{entity_offset + DOMAINS.index(domain) * 10 + index + 1}"
                        ),
                        "relation_id": relation_id,
                        "is_correct": is_correct,
                    }
                )
    return rows


def _auditable_rows() -> list[dict[str, object]]:
    rows = _rows("instrument_development")
    for row in rows:
        row.update(
            {
                "prompt": f"Question for {row['entity_id']}?",
                "completion": "answer",
                "accepted_aliases": ["answer"],
            }
        )
    return rows


def test_select_relation_triplets_maximizes_two_of_three_qualification() -> None:
    result = select_relation_triplets(
        _rows("instrument_development"),
        relation_bank=RELATIONS,
        qualification_threshold=2,
        expected_candidates_per_domain=4,
        minimum_qualified_per_domain_validation=3,
        **BINDINGS,
    )

    for domain in DOMAINS:
        assert result["selected_relations"][domain] == list(RELATIONS[domain][:3])
        assert result["selected_metrics"][domain]["qualified_count"] == 4
    assert result["claim_scope"] == "open_instrument_development_only"


def test_selection_is_invariant_to_input_order() -> None:
    rows = _rows("instrument_development")

    forward = select_relation_triplets(
        rows,
        relation_bank=RELATIONS,
        qualification_threshold=2,
        expected_candidates_per_domain=4,
        minimum_qualified_per_domain_validation=3,
        **BINDINGS,
    )
    reverse = select_relation_triplets(
        list(reversed(rows)),
        relation_bank=RELATIONS,
        qualification_threshold=2,
        expected_candidates_per_domain=4,
        minimum_qualified_per_domain_validation=3,
        **BINDINGS,
    )

    assert forward == reverse


def test_validation_rejects_development_entity_overlap() -> None:
    selection = select_relation_triplets(
        _rows("instrument_development"),
        relation_bank=RELATIONS,
        qualification_threshold=2,
        expected_candidates_per_domain=4,
        minimum_qualified_per_domain_validation=3,
        **BINDINGS,
    )

    with pytest.raises(ValueError, match="entity-disjoint"):
        overlapping = _rows("construction_validation")
        overlapping[0]["qid"] = selection["development_qids"][0]
        assess_frozen_validation(
            overlapping,
            selection=selection,
            **VALIDATION_BINDINGS,
        )


def test_validation_uses_frozen_triplets_and_threshold() -> None:
    selection = select_relation_triplets(
        _rows("instrument_development"),
        relation_bank=RELATIONS,
        qualification_threshold=2,
        expected_candidates_per_domain=4,
        minimum_qualified_per_domain_validation=3,
        **BINDINGS,
    )
    validation = assess_frozen_validation(
        _rows("construction_validation", entity_offset=100),
        selection=selection,
        **VALIDATION_BINDINGS,
    )

    assert validation["gate_passed"] is True
    assert validation["qualification_threshold"] == 2
    assert validation["selected_relations"] == selection["selected_relations"]
    assert all(
        row["qualified_count"] == 4
        for row in validation["by_domain"].values()
    )


def test_duplicate_entity_relation_observations_fail_closed() -> None:
    rows = _rows("instrument_development")
    rows.append(dict(rows[0]))

    with pytest.raises(ValueError, match="duplicate"):
        select_relation_triplets(
            rows,
            relation_bank=RELATIONS,
            qualification_threshold=2,
            expected_candidates_per_domain=4,
            minimum_qualified_per_domain_validation=3,
            **BINDINGS,
        )


def test_relation_bank_order_cannot_change_selection() -> None:
    reversed_bank = {
        domain: tuple(reversed(relations))
        for domain, relations in RELATIONS.items()
    }

    result = select_relation_triplets(
        _rows("instrument_development"),
        relation_bank=reversed_bank,
        qualification_threshold=2,
        expected_candidates_per_domain=4,
        minimum_qualified_per_domain_validation=3,
        **BINDINGS,
    )

    for domain in DOMAINS:
        assert result["selected_relations"][domain] == sorted(RELATIONS[domain][:3])


def test_missing_relation_observation_fails_closed() -> None:
    rows = _rows("instrument_development")
    rows.pop()

    with pytest.raises(ValueError, match="complete relation matrix"):
        select_relation_triplets(
            rows,
            relation_bank=RELATIONS,
            qualification_threshold=2,
            expected_candidates_per_domain=4,
            minimum_qualified_per_domain_validation=3,
            **BINDINGS,
        )


def test_validation_rejects_tampered_selection() -> None:
    selection = select_relation_triplets(
        _rows("instrument_development"),
        relation_bank=RELATIONS,
        qualification_threshold=2,
        expected_candidates_per_domain=4,
        minimum_qualified_per_domain_validation=3,
        **BINDINGS,
    )
    selection["minimum_qualified_per_domain_validation"] = 1

    with pytest.raises(ValueError, match="selection hash"):
        assess_frozen_validation(
            _rows("construction_validation", entity_offset=100),
            selection=selection,
            **VALIDATION_BINDINGS,
        )


def test_r11_human_audit_packet_is_deterministic_and_blinded() -> None:
    rows = _auditable_rows()
    selection = select_relation_triplets(
        rows,
        relation_bank=RELATIONS,
        qualification_threshold=2,
        expected_candidates_per_domain=4,
        minimum_qualified_per_domain_validation=3,
        **BINDINGS,
    )

    packet = prepare_r11_human_audit_packet(
        rows,
        selection=selection,
        sample_per_domain=2,
        success_sample_per_domain=1,
        seed=20260726,
    )
    replay = prepare_r11_human_audit_packet(
        list(reversed(rows)),
        selection=selection,
        sample_per_domain=2,
        success_sample_per_domain=1,
        seed=20260726,
    )

    assert packet == replay
    assert len(packet) == 12
    assert all("is_correct" not in row for row in packet)
    assert all(
        row["relation_id"] in selection["selected_relations"][row["domain"]]
        for row in packet
    )


def test_r11_human_audit_requires_two_independent_raters() -> None:
    rows = _auditable_rows()
    selection = select_relation_triplets(
        rows,
        relation_bank=RELATIONS,
        qualification_threshold=2,
        expected_candidates_per_domain=4,
        minimum_qualified_per_domain_validation=3,
        **BINDINGS,
    )
    packet = prepare_r11_human_audit_packet(
        rows,
        selection=selection,
        sample_per_domain=2,
        success_sample_per_domain=1,
        seed=20260726,
    )
    scored = {
        f"{row['split']}:{row['domain']}:{row['qid']}:{row['relation_id']}": row[
            "is_correct"
        ]
        for row in rows
    }
    ratings = [
        {
            "audit_id": row["audit_id"],
            "rater_id": rater,
            "round": 1,
            "error_label": (
                "no_error" if scored[row["question_id"]] else "relation_unknown"
            ),
        }
        for row in packet
        for rater in ("rater-a", "rater-b")
    ]

    result = compile_r11_human_audit(
        rows,
        selection=selection,
        ratings=ratings,
        sample_per_domain=2,
        success_sample_per_domain=1,
        seed=20260726,
    )
    assert result["gate_passed"] is True
    assert result["compiled"]["decision_counts"] == {
        "no_error": 4,
        "relation_unknown": 8,
    }

    with pytest.raises(ValueError, match="two independent"):
        compile_r11_human_audit(
            rows,
            selection=selection,
            ratings=ratings[::2],
            sample_per_domain=2,
            success_sample_per_domain=1,
            seed=20260726,
        )


def test_r11_human_audit_fails_on_scoring_or_instrument_errors() -> None:
    rows = _auditable_rows()
    selection = select_relation_triplets(
        rows,
        relation_bank=RELATIONS,
        qualification_threshold=2,
        expected_candidates_per_domain=4,
        minimum_qualified_per_domain_validation=3,
        **BINDINGS,
    )
    packet = prepare_r11_human_audit_packet(
        rows,
        selection=selection,
        sample_per_domain=2,
        success_sample_per_domain=1,
        seed=20260726,
    )
    ratings = [
        {
            "audit_id": row["audit_id"],
            "rater_id": rater,
            "round": 1,
            "error_label": "incomplete_alias_set",
        }
        for row in packet
        for rater in ("rater-a", "rater-b")
    ]

    result = compile_r11_human_audit(
        rows,
        selection=selection,
        ratings=ratings,
        sample_per_domain=2,
        success_sample_per_domain=1,
        seed=20260726,
    )

    assert result["gate_passed"] is False
    assert result["disallowed_count"] == len(packet)
