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
from trajectory_extractor.rlmf_types import ParsedRLMFOutput


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


def test_strict_and_soft_format_rewards_match_pinned_upstream_fixtures():
    parsed = [
        ParsedRLMFOutput(answer="A", confidence=0.8, valid_format=True),
        ParsedRLMFOutput(answer=""),
    ]
    texts = [
        "<sentence>A</sentence><confidence>0.8</confidence>",
        "plain text",
        "<sentence>A</sentence><sentence>B</sentence><confidence>0.5</confidence>",
        "<sentence></sentence><confidence>0.5</confidence>",
    ]

    assert np.array_equal(strict_format_reward(parsed), np.array([1.0, -1.0]))
    assert np.array_equal(soft_format_reward(texts), np.array([0.0, -1.0, -0.125, -0.05]))


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
        _behavior_row("standard_grpo", 11, "p1", 0.8, 0.75, correctness=1.0),
        _behavior_row("standard_grpo", 11, "p2", 0.2, 0.25, correctness=0.0),
    ]

    metrics = calibration_metrics(records)

    assert metrics["cmfg_star"] == pytest.approx(0.95)
    assert metrics["cmfg_tie_preserving"] == pytest.approx(0.95)
    assert metrics["faithfulness_accuracy"] == pytest.approx(0.95)
    assert metrics["accuracy"] == 0.5
    assert metrics["format_validity"] == 1.0
    assert all(np.isfinite(value) for value in metrics.values())

    malformed = records + [{**records[0], "example_id": "broken", "intrinsic": np.nan}]
    with pytest.raises(ValueError, match="finite"):
        calibration_metrics(malformed)


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


def test_judge_bias_adjustment_consumes_schema_v2_arm_specific_uncertainty():
    records = []
    for seed in (11, 22, 33):
        for prompt in ("p1", "p2", "p3"):
            records.append(_judge_row("standard_grpo", seed, prompt, 0.5, positives=10))
            records.append(_judge_row("rlmf", seed, prompt, 0.6, positives=10))
    audit = _confusion_uncertainty()
    audit["estimates"]["rlmf:equivalence"] = {
        "sensitivity": {"lower": 0.75, "estimate": 0.8, "upper": 0.85},
        "specificity": {"lower": 0.95, "estimate": 1.0, "upper": 1.0},
    }

    first = judge_bias_adjusted_delta(records, audit, replicates=300, rng_seed=19)
    second = judge_bias_adjusted_delta(records, audit, replicates=300, rng_seed=19)

    assert first == second
    assert first.estimate == pytest.approx(-0.025, abs=1e-10)
    assert first.lower <= first.estimate <= first.upper

    legacy = deepcopy(audit)
    legacy["schema_version"] = 1
    with pytest.raises(ValueError, match="schema-v2"):
        judge_bias_adjusted_delta(records, legacy, replicates=20, rng_seed=19)
    with pytest.raises(ValueError, match="confusion uncertainty"):
        judge_bias_adjusted_delta(records, [], replicates=20, rng_seed=19)
    malformed_design = deepcopy(audit)
    first_stratum = next(iter(malformed_design["sampling_design"]["strata"].values()))
    first_stratum["population_count"] = 0
    with pytest.raises(ValueError, match="sampling design"):
        judge_bias_adjusted_delta(records, malformed_design, replicates=20, rng_seed=19)


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


def _behavior_row(arm, seed, example_id, confidence, intrinsic, *, correctness):
    return {
        "arm": arm,
        "seed": seed,
        "example_id": example_id,
        "answer": "answer",
        "confidence": confidence,
        "intrinsic": intrinsic,
        "correctness": correctness,
        "valid_format": True,
    }


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
    return {
        "schema_version": 2,
        "method": "deterministic_stratified_bootstrap",
        "replicates": 2000,
        "rng_seed": 20260713,
        "sampling_design": _sampling_design(),
        "estimates": {
            f"{arm}:{judgment_type}": deepcopy(perfect)
            for arm in ("standard_grpo", "rlmf")
            for judgment_type in ("correctness", "equivalence")
        },
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
