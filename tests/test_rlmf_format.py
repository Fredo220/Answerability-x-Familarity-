from dataclasses import replace

import pytest

from trajectory_extractor.rlmf_format import (
    AuditRow,
    alias_exact_match,
    bound_differential_judge_bias,
    build_judge_audit_sample,
    completion_equivalent,
    estimate_arm_confusion_uncertainty,
    normalized_answer,
    parse_metascore_output,
    parse_rlmf_output,
    score_blinded_judge_audit,
)


def test_strict_answer_and_metascore_schemas_accept_only_the_registered_form():
    answer = parse_rlmf_output(
        "<sentence>William Shakespeare</sentence>\n<confidence>0.8</confidence>"
    )
    metascore = parse_metascore_output("<metascore>0.3</metascore>")

    assert answer.answer == "William Shakespeare"
    assert answer.confidence == 0.8
    assert answer.valid_format is True
    assert metascore.metascore == 0.3
    assert metascore.valid_format is True


@pytest.mark.parametrize(
    "text",
    [
        "<sentence>answer</sentence><confidence>0.8</confidence> trailing",
        "<sentence>answer</sentence><confidence>0.8</confidence><confidence>0.8</confidence>",
        "<sentence>answer</sentence>",
        "<sentence>answer</sentence><confidence>NaN</confidence>",
        "<sentence>answer</sentence><confidence>inf</confidence>",
        "<sentence>answer</sentence><confidence>0.81</confidence>",
        "<sentence>answer</sentence><confidence>1.1</confidence>",
        "<sentence></sentence><confidence>0.8</confidence>",
    ],
)
def test_malformed_answer_outputs_are_invalid_without_guessing(text):
    parsed = parse_rlmf_output(text)

    assert parsed.valid_format is False
    assert parsed.answer == ""
    assert parsed.confidence is None


@pytest.mark.parametrize(
    "text",
    [
        "<metascore>0.4</metascore> trailing",
        "<metascore>0.4</metascore><metascore>0.4</metascore>",
        "<metascore>nan</metascore>",
        "<metascore>1.01</metascore>",
        "<metascore>0.40</metascore>",
    ],
)
def test_malformed_metascores_are_invalid_without_guessing(text):
    parsed = parse_metascore_output(text)

    assert parsed.valid_format is False
    assert parsed.metascore is None


def test_alias_normalization_matches_the_registered_contract_without_symbol_collisions():
    assert (
        normalized_answer(" The\u00a0\uff37\uff49\uff4c\uff4c\uff49\uff41\uff4d, Shakespeare! ")
        == "william shakespeare"
    )
    assert normalized_answer("War of the Worlds") == "war of worlds"
    assert alias_exact_match("The U.S.A.", ("usa",))
    assert not alias_exact_match("AB", ("A+B",))
    assert not alias_exact_match("Shakespeare", ("William Shakespeare",))
    assert completion_equivalent(
        "William Shakespeare", "Shakespeare", ("William Shakespeare", "Shakespeare")
    )
    assert completion_equivalent("A Mercury", "Mercury", ("Mercury",))
    assert not completion_equivalent("Mercury", "Freddie Mercury", ("Mercury",))

    with pytest.raises(ValueError, match="non-string sequence"):
        alias_exact_match("a", "abc")
    with pytest.raises(ValueError, match="strings"):
        alias_exact_match("a", ("a", None))


def test_locked_sampling_is_deterministic_balanced_blinded_and_proxy_bound():
    candidates = _candidates(per_stratum=50, phase="locked")

    first = build_judge_audit_sample(candidates, phase="locked", size=400, seed=20260713)
    second = build_judge_audit_sample(
        list(reversed(candidates)), phase="locked", size=400, seed=20260713
    )

    assert first == second
    assert len(first) == 400
    assert _stratum_counts(first) == {
        (arm, judgment_type, label): 50
        for arm in ("standard_grpo", "rlmf")
        for judgment_type in ("correctness", "equivalence")
        for label in (False, True)
    }
    payload = first[0].rater_payload()
    assert set(payload) == {
        "audit_id",
        "judgment_type",
        "question",
        "answer",
        "comparison_answer",
        "reference_answer",
    }
    assert not {"arm", "split", "aliases", "proxy_label"} & set(payload)
    ledger = first[0].to_ledger_record()
    assert ledger["aliases"] == ["william shakespeare"]
    assert first[0] == AuditRow.from_ledger_record(ledger)

    contradictory = [dict(row) for row in candidates]
    contradictory[0]["proxy_label"] = not contradictory[0]["proxy_label"]
    with pytest.raises(ValueError, match="proxy_label"):
        build_judge_audit_sample(
            contradictory, phase="locked", size=400, seed=20260713
        )


def test_development_uses_shared_pretreatment_split_strata_without_arms():
    rows = build_judge_audit_sample(
        _candidates(per_stratum=25, phase="development"),
        phase="development",
        size=200,
        seed=20260713,
    )

    assert {row.arm for row in rows} == {None}
    assert {
        (row.split, row.judgment_type, row.proxy_label) for row in rows
    } == {
        (split, judgment_type, label)
        for split in ("pre_sft", "rl_train")
        for judgment_type in ("correctness", "equivalence")
        for label in (False, True)
    }
    counts = {
        key: sum((row.split, row.judgment_type, row.proxy_label) == key for row in rows)
        for key in {
            (row.split, row.judgment_type, row.proxy_label) for row in rows
        }
    }
    assert set(counts.values()) == {25}

    invented_arm = _candidates(per_stratum=25, phase="development")
    invented_arm[0]["arm"] = "standard_grpo"
    with pytest.raises(ValueError, match="pre-treatment"):
        build_judge_audit_sample(
            invented_arm, phase="development", size=200, seed=20260713
        )


@pytest.mark.parametrize(
    ("phase", "wrong_split"),
    [("development", "validation"), ("locked", "test"), ("test", "validation")],
)
def test_sampling_enforces_phase_split_eligibility(phase, wrong_split):
    per_stratum = {"development": 25, "locked": 50, "test": 125}[phase]
    size = {"development": 200, "locked": 400, "test": 1000}[phase]
    candidates = _candidates(per_stratum=per_stratum, phase=phase)
    candidates[0]["split"] = wrong_split

    with pytest.raises(ValueError, match="split"):
        build_judge_audit_sample(candidates, phase=phase, size=size, seed=20260713)


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_sampling_rejects_missing_null_and_empty_candidate_ids(bad_id):
    candidates = _candidates(per_stratum=50, phase="locked")
    candidates[0]["candidate_id"] = bad_id

    with pytest.raises(ValueError, match="source ID"):
        build_judge_audit_sample(
            candidates, phase="locked", size=400, seed=20260713
        )


def test_sampling_rejects_duplicate_ids_and_extension_preserves_selected_ids():
    candidates = _candidates(per_stratum=250, phase="test")
    duplicate = [dict(row) for row in candidates]
    duplicate[1]["candidate_id"] = duplicate[0]["candidate_id"]
    with pytest.raises(ValueError, match="unique"):
        build_judge_audit_sample(duplicate, phase="test", size=1000, seed=20260713)

    rows_1000 = build_judge_audit_sample(
        candidates, phase="test", size=1000, seed=20260713
    )
    rows_1250 = build_judge_audit_sample(
        list(reversed(candidates)), phase="test", size=1250, seed=20260713
    )
    ids_1000 = {row.source_id: row.audit_id for row in rows_1000}
    ids_1250 = {row.source_id: row.audit_id for row in rows_1250}

    assert ids_1000.items() <= ids_1250.items()
    assert len(ids_1250) == 1250


def test_locked_and_test_audits_apply_all_registered_gates_fail_closed():
    locked = _rated(
        build_judge_audit_sample(
            _candidates(per_stratum=50, phase="locked"),
            phase="locked",
            size=400,
            seed=20260713,
        )
    )
    assert score_blinded_judge_audit(locked).passed is True

    test_rows = _rated(
        build_judge_audit_sample(
            _candidates(per_stratum=125, phase="test"),
            phase="test",
            size=1000,
            seed=20260713,
        )
    )
    test_decision = score_blinded_judge_audit(test_rows)
    assert test_decision.passed is True
    assert test_decision.status == "passed"

    disagreement = tuple(
        replace(
            row,
            rater_b="incorrect" if row.rater_a == "correct" else "correct",
            adjudicated_label=row.rater_a,
        )
        for row in test_rows
    )
    failed = score_blinded_judge_audit(disagreement)
    assert failed.passed is False
    assert failed.status == "failed"
    assert failed.kappa < 0.80

    ambiguous = tuple(
        replace(row, rater_a="ambiguous", rater_b="ambiguous") if index < 51 else row
        for index, row in enumerate(test_rows)
    )
    assert score_blinded_judge_audit(ambiguous).passed is False


def test_task4_emits_confusion_uncertainty_but_defers_endpoint_bias_propagation():
    rows = _rated(
        build_judge_audit_sample(
            _candidates(per_stratum=125, phase="test"),
            phase="test",
            size=1000,
            seed=20260713,
        )
    )

    uncertainty = estimate_arm_confusion_uncertainty(rows)

    assert uncertainty["schema_version"] == 2
    assert uncertainty["method"] == "deterministic_stratified_bootstrap"
    assert uncertainty["sampling_design"]["schema_version"] == 1
    assert len(uncertainty["joint_draws"]) == uncertainty["replicates"] == 2000
    assert set(uncertainty["estimates"]) == {
        f"{arm}:{judgment_type}"
        for arm in ("standard_grpo", "rlmf")
        for judgment_type in ("correctness", "equivalence")
    }
    for draw in uncertainty["joint_draws"]:
        assert set(draw) == set(uncertainty["estimates"])
        assert all(
            set(confusion) == {"sensitivity", "specificity"}
            for confusion in draw.values()
        )
    for value in uncertainty["estimates"].values():
        assert value["sensitivity"]["estimate"] == 1.0
        assert 0.0 <= value["sensitivity"]["lower"] <= value["sensitivity"]["upper"] <= 1.0
        assert value["specificity"]["estimate"] == 1.0

    with pytest.raises(ValueError, match="Task 5/10"):
        bound_differential_judge_bias(rows, replicates=200)


def test_proxy_stratified_confusion_uses_population_weights_not_raw_audit_ratios():
    low_prevalence = _reviewer_probe_rows(
        proxy_positive_population=125, proxy_negative_population=1125
    )
    high_prevalence = _reviewer_probe_rows(
        proxy_positive_population=1000, proxy_negative_population=250
    )

    low = score_blinded_judge_audit(low_prevalence)
    high = score_blinded_judge_audit(high_prevalence)

    key = "standard_grpo:correctness"
    assert low.sensitivity[key] == pytest.approx(0.5714285714)
    assert high.sensitivity[key] == pytest.approx(0.9795918367)
    assert low.sensitivity[key] != pytest.approx(120 / 130)
    assert high.sensitivity[key] != pytest.approx(120 / 130)

def _candidates(*, per_stratum: int, phase: str):
    if phase == "development":
        groups = ((split, None) for split in ("pre_sft", "rl_train"))
    else:
        split = "validation" if phase == "locked" else "test"
        groups = ((split, arm) for arm in ("standard_grpo", "rlmf"))
    candidates = []
    for split, arm in groups:
        for judgment_type in ("correctness", "equivalence"):
            for proxy_label in (False, True):
                for index in range(per_stratum):
                    prefix = arm or split
                    source_id = f"{prefix}-{judgment_type}-{proxy_label}-{index}"
                    answer = "William Shakespeare" if proxy_label else "Marlowe"
                    comparison = "William Shakespeare" if proxy_label else "Jonson"
                    candidates.append(
                        {
                            "candidate_id": source_id,
                            "example_id": f"example-{source_id}",
                            "split": split,
                            "arm": arm,
                            "judgment_type": judgment_type,
                            "proxy_label": proxy_label,
                            "question": "Who wrote Hamlet?",
                            "answer": answer,
                            "comparison_answer": comparison,
                            "reference_answer": "William Shakespeare",
                            "gold_aliases": ["william shakespeare"],
                        }
                    )
    return candidates


def _rated(rows):
    return tuple(
        replace(
            row,
            rater_a="correct" if row.proxy_label else "incorrect",
            rater_b="correct" if row.proxy_label else "incorrect",
        )
        for row in rows
    )


def _reviewer_probe_rows(*, proxy_positive_population: int, proxy_negative_population: int):
    candidates = _candidates(per_stratum=125, phase="test")
    rows = build_judge_audit_sample(
        candidates, phase="test", size=1000, seed=20260713
    )
    by_stratum = {}
    for row in rows:
        by_stratum.setdefault((row.arm, row.judgment_type, row.proxy_label), []).append(row)

    rated = []
    for (_arm, _judgment_type, proxy_label), stratum_rows in by_stratum.items():
        human_positive = 120 if proxy_label else 10
        population_count = (
            proxy_positive_population if proxy_label else proxy_negative_population
        )
        for index, row in enumerate(stratum_rows):
            label = "correct" if index < human_positive else "incorrect"
            rated.append(
                replace(
                    row,
                    stratum_population_count=population_count,
                    rater_a=label,
                    rater_b=label,
                )
            )
    return tuple(rated)


def _stratum_counts(rows):
    return {
        key: sum((row.arm, row.judgment_type, row.proxy_label) == key for row in rows)
        for key in {
            (row.arm, row.judgment_type, row.proxy_label) for row in rows
        }
    }
