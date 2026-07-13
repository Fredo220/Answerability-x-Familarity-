from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from trajectory_extractor.rlmf_metrics import (
    calibration_metrics,
    cmfg_star,
    cmfg_tie_preserving,
    common_support_sensitivity,
    evaluation_intrinsic_confidence,
    factual_calibration_reward,
    faithful_calibration_reward,
    faithfulness_accuracy,
    gold_faithfulness_level,
    judge_bias_adjusted_delta,
    metacognitive_reward,
    paired_fixed_seed_prompt_bootstrap,
    soft_format_reward,
    strict_format_reward,
    training_leave_one_out_confidence,
)
import trajectory_extractor.rlmf_metrics as rlmf_metrics
from trajectory_extractor.rlmf_types import BehavioralEvaluationRecord, ParsedRLMFOutput


def test_training_confidence_is_leave_one_out_for_unanimous_split_and_unique_groups():
    aliases = {
        "Alpha": ("Alpha", "Alpha Prime"),
        "Alpha Prime": ("Alpha", "Alpha Prime"),
        "Beta": ("Beta",),
        "Gamma": ("Gamma",),
        "Delta": ("Delta",),
    }

    assert np.array_equal(
        training_leave_one_out_confidence(
            ["Alpha", "Alpha Prime", "Alpha", "Alpha Prime"], aliases
        ),
        np.ones(4),
    )
    assert np.allclose(
        training_leave_one_out_confidence(
            ["Alpha", "Alpha Prime", "Beta", "Beta"], aliases
        ),
        np.full(4, 1 / 3),
    )
    assert np.array_equal(
        training_leave_one_out_confidence(
            ["Alpha", "Beta", "Gamma", "Delta"], aliases
        ),
        np.zeros(4),
    )


def test_evaluation_confidence_uses_exactly_twenty_auxiliaries_and_excludes_designated():
    auxiliaries = ["William Shakespeare"] * 13 + ["Christopher Marlowe"] * 7

    intrinsic = evaluation_intrinsic_confidence(
        "Shakespeare",
        auxiliaries,
        ("William Shakespeare", "Shakespeare"),
    )

    assert intrinsic == 0.65
    assert faithfulness_accuracy(0.8, intrinsic) == pytest.approx(0.85)
    with pytest.raises(ValueError, match="exactly 20"):
        evaluation_intrinsic_confidence("Shakespeare", ["Shakespeare"] * 19, ("Shakespeare",))


def test_quadratic_rewards_and_tau_boundary_match_upstream_behavior():
    confidence = np.array([0.0, 0.25, 1.0])
    intrinsic = np.array([0.0, 0.75, 0.0])

    assert np.array_equal(
        faithful_calibration_reward(confidence, intrinsic),
        np.array([1.0, 0.75, 0.0]),
    )
    assert np.array_equal(
        factual_calibration_reward(confidence, np.array([0.0, 1.0, 1.0])),
        np.array([1.0, 0.4375, 1.0]),
    )
    assert np.array_equal(
        gold_faithfulness_level(
            np.array([0.4, 0.4, 0.4]),
            np.array([0.5, 0.50000000001, 0.51]),
            tau=0.10,
        ),
        np.array([1.0, 0.0, 0.0]),
    )
    assert np.array_equal(
        metacognitive_reward(np.array([1.0, 0.25]), np.array([1.0, 0.75])),
        np.array([1.0, 0.75]),
    )


def test_local_strict_schema_reward_rejects_prose_while_upstream_soft_format_fixtures_stay_frozen():
    fixtures = [
        ("<sentence>A</sentence><confidence>0.8</confidence>", True, 0.0),
        ("<sentence>A <sentence>B</sentence></sentence><confidence>0.8</confidence>", False, -0.125),
        ("<sentence>A</sentence><sentence>B</sentence><confidence>0.5</confidence>", False, -0.125),
        ("<sentence>A</sentence><confidence>0.8", False, -0.5),
        ("<sentence></sentence><confidence>0.5</confidence>", False, -0.05),
        ("prefix <sentence>A</sentence><confidence>0.8</confidence> suffix", False, 0.0),
        ("<sentence>A</sentence><confidence>1.1</confidence>", False, -0.25),
        ("<sentence>A</sentence><confidence>0.81</confidence>", False, 0.0),
    ]
    parsed = [
        ParsedRLMFOutput(answer="A", confidence=0.8, valid_format=True)
        if valid
        else ParsedRLMFOutput(answer="")
        for _text, valid, _soft in fixtures
    ]

    assert np.array_equal(
        strict_format_reward(parsed),
        np.asarray([1.0 if valid else -1.0 for _text, valid, _soft in fixtures]),
    )
    assert np.array_equal(
        soft_format_reward([text for text, _valid, _soft in fixtures]),
        np.asarray([soft for _text, _valid, soft in fixtures]),
    )


def test_cmfg_star_uses_nonempty_equal_mass_bins_and_confidence_axis_widths():
    confidence = np.array([0.0, 0.1, 0.2, 0.6, 0.9, 1.0])
    intrinsic = np.array([0.0, 0.1, 0.2, 0.1, 0.4, 0.5])

    assert cmfg_star(confidence, intrinsic, bins=2) == pytest.approx(0.7, abs=1e-10)
    assert cmfg_star(confidence[:2], intrinsic[:2], bins=10) == pytest.approx(1.0, abs=1e-10)


def test_tie_preserving_bins_do_not_split_equal_confidences():
    confidence = np.array([0.0, 0.0, 0.0, 1.0])
    intrinsic = np.array([0.0, 0.0, 1.0, 1.0])

    assert cmfg_star(confidence, intrinsic, bins=2) == pytest.approx(0.5, abs=1e-10)
    assert cmfg_tie_preserving(confidence, intrinsic) == pytest.approx(5 / 6, abs=1e-10)


def test_common_support_reports_both_sensitivities_on_the_shared_axis_only():
    standard = [
        {"confidence": 0.0, "intrinsic": 1.0},
        {"confidence": 0.5, "intrinsic": 0.5},
    ]
    rlmf = [
        {"confidence": 0.5, "intrinsic": 0.0},
        {"confidence": 1.0, "intrinsic": 1.0},
    ]

    sensitivity = common_support_sensitivity(standard, rlmf)

    assert sensitivity == {
        "support_lower": 0.5,
        "support_upper": 0.5,
        "standard_cmfg_star": 1.0,
        "rlmf_cmfg_star": 0.5,
        "delta_cmfg_star": -0.5,
        "standard_cmfg_tie_preserving": 1.0,
        "rlmf_cmfg_tie_preserving": 0.5,
        "delta_cmfg_tie_preserving": -0.5,
    }

    with pytest.raises(ValueError, match="common confidence support"):
        common_support_sensitivity(
            [
                {"confidence": 0.4, "intrinsic": 0.4},
                {"confidence": 0.6, "intrinsic": 0.6},
            ],
            [
                {"confidence": 0.5, "intrinsic": 0.5},
                {"confidence": 0.7, "intrinsic": 0.7},
            ],
        )


def test_calibration_metrics_emit_primary_and_sensitivities_without_dropping_rows():
    records = [
        _behavior_record("standard_grpo", 11, "p1", 0.8, positives=15, correctness=True),
        _behavior_record("standard_grpo", 11, "p2", 0.2, positives=5, correctness=False),
        _behavior_record("standard_grpo", 11, "broken", None, positives=0, correctness=None),
    ]

    result = calibration_metrics(records)

    assert result.status == "evaluable"
    assert result.reason is None
    assert result.total_records == 3
    assert result.complete_case_records == 2
    assert result.retained_record_ids == (
        ("standard_grpo", 11, "p1"),
        ("standard_grpo", 11, "p2"),
        ("standard_grpo", 11, "broken"),
    )
    assert result.complete_case_record_ids == (
        ("standard_grpo", 11, "p1"),
        ("standard_grpo", 11, "p2"),
    )
    assert result.format_validity == pytest.approx(2 / 3)
    assert result.metrics["cmfg_star"] == pytest.approx(0.95)
    assert result.metrics["cmfg_tie_preserving"] == pytest.approx(0.95)
    assert result.metrics["faithfulness_accuracy"] == pytest.approx(0.95)
    assert result.metrics["accuracy"] == 0.5
    assert all(np.isfinite(value) for value in result.metrics.values())


def test_calibration_metrics_returns_machine_readable_not_evaluable_without_complete_cases():
    result = calibration_metrics(
        [_behavior_record("standard_grpo", 11, "broken", None, positives=0, correctness=None)]
    )

    assert result.status == "not_evaluable"
    assert result.reason == "no_valid_complete_cases"
    assert result.total_records == 1
    assert result.complete_case_records == 0
    assert result.retained_record_ids == (("standard_grpo", 11, "broken"),)
    assert result.metrics == {}


def test_fixed_seed_bootstrap_resamples_paired_prompts_within_seed_and_is_reproducible():
    records = []
    for seed in (11, 22, 33):
        for prompt, value in (("p1", 0.2), ("p2", 0.8), ("p3", 0.5)):
            records.append({"arm": "standard_grpo", "seed": seed, "example_id": prompt, "value": value})
            records.append({"arm": "rlmf", "seed": seed, "example_id": prompt, "value": value})

    def paired_delta(rows):
        frame = {(row["arm"], row["seed"], row["example_id"], index): row["value"] for index, row in enumerate(rows)}
        assert frame
        return np.mean([row["value"] for row in rows if row["arm"] == "rlmf"]) - np.mean(
            [row["value"] for row in rows if row["arm"] == "standard_grpo"]
        )

    first = paired_fixed_seed_prompt_bootstrap(
        records, paired_delta, seeds=(11, 22, 33), replicates=200, rng_seed=7
    )
    second = paired_fixed_seed_prompt_bootstrap(
        records, paired_delta, seeds=(11, 22, 33), replicates=200, rng_seed=7
    )

    assert first == second
    assert first.to_record() == {"lower": 0.0, "estimate": 0.0, "upper": 0.0}

    unpaired = [row for row in records if not (row["arm"] == "rlmf" and row["seed"] == 22 and row["example_id"] == "p2")]
    with pytest.raises(ValueError, match="paired"):
        paired_fixed_seed_prompt_bootstrap(
            unpaired, paired_delta, seeds=(11, 22, 33), replicates=20, rng_seed=7
        )

    for seed in (11, 22, 33):
        interval = paired_fixed_seed_prompt_bootstrap(
            [row for row in records if row["seed"] == seed],
            paired_delta,
            seeds=(seed,),
            replicates=20,
            rng_seed=7,
        )
        assert interval.to_record() == {"lower": 0.0, "estimate": 0.0, "upper": 0.0}

    with pytest.raises(ValueError, match="one registered seed or exactly"):
        paired_fixed_seed_prompt_bootstrap(
            [row for row in records if row["seed"] in (11, 22)],
            paired_delta,
            seeds=(11, 22),
            replicates=20,
            rng_seed=7,
        )
    with pytest.raises(ValueError, match="one registered seed or exactly"):
        paired_fixed_seed_prompt_bootstrap(
            records,
            paired_delta,
            seeds=(11, 22, 44),
            replicates=20,
            rng_seed=7,
        )


def test_fixed_seed_bootstrap_preserves_duplicate_cluster_multiplicity():
    records = [
        {"arm": arm, "seed": 11, "example_id": prompt, "value": value}
        for prompt, value in (("p1", 0.0), ("p2", 1.0))
        for arm in ("standard_grpo", "rlmf")
    ]
    sampled_ids = []

    def capture(rows):
        sampled_ids.append(tuple(row["example_id"] for row in rows if row["arm"] == "rlmf"))
        return 0.0

    paired_fixed_seed_prompt_bootstrap(
        records, capture, seeds=(11,), replicates=20, rng_seed=7
    )

    assert any(len(set(ids)) < len(ids) for ids in sampled_ids[1:])


def test_judge_bias_adjustment_consumes_whole_joint_draws_and_returns_registered_bias_quantity():
    records = []
    for seed in (11, 22, 33):
        for prompt in ("p1", "p2", "p3"):
            records.append(_behavior_record("standard_grpo", seed, prompt, 0.5, positives=10, correctness=True))
            records.append(_behavior_record("rlmf", seed, prompt, 0.6, positives=10, correctness=True))
    audit = _confusion_uncertainty()
    audit["estimates"]["rlmf:equivalence"] = {
        "sensitivity": {"lower": 0.8, "estimate": 0.8, "upper": 0.8},
        "specificity": {"lower": 0.95, "estimate": 1.0, "upper": 1.0},
    }
    for draw in audit["joint_draws"]:
        draw["rlmf:equivalence"] = {"sensitivity": 0.8, "specificity": 1.0}

    first = judge_bias_adjusted_delta(records, audit, rng_seed=19)
    second = judge_bias_adjusted_delta(records, audit, rng_seed=19)

    assert first == second
    assert first.status == "evaluable"
    assert first.total_pairs == first.complete_pairs == 9
    assert first.excluded_pair_count == 0
    assert first.proxy_delta_cmfg_star.estimate == pytest.approx(-0.1, abs=1e-10)
    assert first.adjusted_delta_cmfg_star.estimate == pytest.approx(-0.025, abs=1e-10)
    assert first.absolute_differential_bias.estimate == pytest.approx(0.075, abs=1e-10)
    assert first.absolute_differential_bias.upper >= first.absolute_differential_bias.estimate

    legacy = deepcopy(audit)
    legacy["schema_version"] = 1
    with pytest.raises(ValueError, match="schema-v2"):
        judge_bias_adjusted_delta(records, legacy, rng_seed=19)
    with pytest.raises(ValueError, match="confusion uncertainty"):
        judge_bias_adjusted_delta(records, [], rng_seed=19)
    missing_joint = deepcopy(audit)
    del missing_joint["joint_draws"]
    with pytest.raises(ValueError, match="joint draws"):
        judge_bias_adjusted_delta(records, missing_joint, rng_seed=19)
    malformed_design = deepcopy(audit)
    first_stratum = next(iter(malformed_design["sampling_design"]["strata"].values()))
    first_stratum["population_count"] = 0
    with pytest.raises(ValueError, match="sampling design"):
        judge_bias_adjusted_delta(records, malformed_design, rng_seed=19)

    missing_seed = [row for row in records if row.seed != 33]
    with pytest.raises(ValueError, match="exactly the registered fixed seeds"):
        judge_bias_adjusted_delta(missing_seed, audit, rng_seed=19)

    with pytest.raises(ValueError, match="BehavioralEvaluationRecord"):
        judge_bias_adjusted_delta([_judge_row("standard_grpo", 11, "p1", 0.5, positives=10)], audit, rng_seed=19)


def test_judge_bias_adjustment_excludes_only_malformed_complete_pairs_and_reports_not_evaluable():
    records = []
    for seed in (11, 22, 33):
        records.extend(
            (
                _behavior_record("standard_grpo", seed, "good", 0.5, positives=10, correctness=True),
                _behavior_record("rlmf", seed, "good", 0.6, positives=10, correctness=True),
                _behavior_record("standard_grpo", seed, "bad", None, positives=10, correctness=None),
                _behavior_record("rlmf", seed, "bad", 0.6, positives=10, correctness=True),
            )
        )

    result = judge_bias_adjusted_delta(records, _confusion_uncertainty(), rng_seed=19)

    assert result.status == "evaluable"
    assert result.total_pairs == 6
    assert result.complete_pairs == 3
    assert result.excluded_pair_count == 3
    assert result.excluded_pair_ids == ((11, "bad"), (22, "bad"), (33, "bad"))

    no_complete = [
        _behavior_record(row.arm, row.seed, row.example_id, None, positives=10, correctness=None)
        for row in records
    ]
    not_evaluable = judge_bias_adjusted_delta(no_complete, _confusion_uncertainty(), rng_seed=19)
    assert not_evaluable.status == "not_evaluable"
    assert not_evaluable.reason == "no_complete_pairs"
    assert not_evaluable.proxy_delta_cmfg_star is None

    seedless = [
        _behavior_record(
            row.arm,
            row.seed,
            row.example_id,
            None if row.seed == 33 else row.confidence,
            positives=10,
            correctness=None if row.seed == 33 else True,
        )
        for row in records
    ]
    missing_complete_seed = judge_bias_adjusted_delta(
        seedless, _confusion_uncertainty(), rng_seed=19
    )
    assert missing_complete_seed.status == "not_evaluable"
    assert missing_complete_seed.reason == "registered_seed_has_no_complete_pairs"


def test_task5_percentile_interval_does_not_widen_to_include_point_estimate():
    interval = rlmf_metrics._percentile_interval(1.0, (0.0, 0.0, 0.0, 0.0))

    assert interval.to_record() == {"lower": 0.0, "estimate": 1.0, "upper": 0.0}


def test_all_metric_interfaces_fail_closed_on_nonfinite_or_malformed_input():
    with pytest.raises(ValueError, match="at least two"):
        training_leave_one_out_confidence(["only"], {"only": ("only",)})
    with pytest.raises(ValueError, match="finite"):
        faithful_calibration_reward(np.array([np.nan]), np.array([0.5]))
    with pytest.raises(ValueError, match="bins"):
        cmfg_star(np.array([0.5]), np.array([0.5]), bins=0)
    with pytest.raises(ValueError, match="common confidence support"):
        common_support_sensitivity(
            [{"confidence": 0.1, "intrinsic": 0.1}],
            [{"confidence": 0.9, "intrinsic": 0.9}],
        )


def _behavior_record(arm, seed, example_id, confidence, *, positives, correctness):
    parsed = (
        ParsedRLMFOutput(answer="answer", confidence=confidence, valid_format=True)
        if confidence is not None
        else ParsedRLMFOutput(answer="")
    )
    return BehavioralEvaluationRecord(
        arm=arm,
        seed=seed,
        example_id=example_id,
        designated_member_id=f"{example_id}-designated",
        designated_raw_output=(
            f"<sentence>answer</sentence><confidence>{confidence:.1f}</confidence>"
            if confidence is not None
            else "malformed output"
        ),
        designated=parsed,
        auxiliary_member_ids=tuple(f"{example_id}-aux-{index}" for index in range(20)),
        auxiliary_proxy_labels=(True,) * positives + (False,) * (20 - positives),
        correctness=correctness,
        provenance={
            "designated_bundle_hash": "a" * 64,
            "auxiliary_bundle_hash": "b" * 64,
            "alias_evidence_hash": "c" * 64,
            "judge_evidence_hash": "d" * 64,
            "config_hash": "e" * 64,
        },
    )


def _judge_row(arm, seed, example_id, confidence, *, positives):
    return {
        "arm": arm,
        "seed": seed,
        "example_id": example_id,
        "confidence": confidence,
        "auxiliary_proxy_labels": [True] * positives + [False] * (20 - positives),
    }


def _confusion_uncertainty():
    perfect = {
        "sensitivity": {"lower": 1.0, "estimate": 1.0, "upper": 1.0},
        "specificity": {"lower": 1.0, "estimate": 1.0, "upper": 1.0},
    }
    estimates = {
        f"{arm}:{judgment_type}": deepcopy(perfect)
        for arm in ("standard_grpo", "rlmf")
        for judgment_type in ("correctness", "equivalence")
    }
    joint_draw = {
        key: {"sensitivity": 1.0, "specificity": 1.0}
        for key in estimates
    }
    return {
        "schema_version": 2,
        "method": "deterministic_stratified_bootstrap",
        "replicates": 2000,
        "rng_seed": 20260713,
        "sampling_design": _sampling_design(),
        "estimates": estimates,
        "joint_draws": [deepcopy(joint_draw) for _ in range(2000)],
    }


def _sampling_design():
    strata = {}
    for arm in ("standard_grpo", "rlmf"):
        for judgment_type in ("correctness", "equivalence"):
            for proxy_label in (False, True):
                key = f"{arm}:{judgment_type}:proxy_{str(proxy_label).lower()}"
                strata[key] = {
                    "group": arm,
                    "judgment_type": judgment_type,
                    "proxy_label": proxy_label,
                    "population_count": 250,
                    "sample_count": 125,
                    "inclusion_probability": {"numerator": 1, "denominator": 2},
                }
    return {
        "schema_version": 1,
        "stratified_on": ["group", "judgment_type", "proxy_label"],
        "strata": strata,
    }
