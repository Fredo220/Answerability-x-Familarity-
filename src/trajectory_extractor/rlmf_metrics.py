"""Behavioral metrics for the resource-scaled RLMF reproduction.

The official cMFG* implementation is ``get_cmfg_star`` in
``src/exp0_baseline/utilities/utils.py`` at RLMF commit
``a087e7a1e49f52aaa701add19cd80699b709fdef``. The complete upstream file has
SHA-256 ``b3c6a38e3acf64d8b91c8fba08b71dc171d26458f6e6a5b7aefc1faa1f31c8cf``.
This module preserves its equal-mass bins and confidence-axis-width weighting,
while retaining malformed observations for explicit format reporting. Task 4
seals aligned joint confusion draws; Task 5 consumes each draw exactly once
alongside the corresponding paired prompt-cluster bootstrap replicate.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from trajectory_extractor.rlmf_format import (
    Interval,
    completion_equivalent,
    normalized_answer,
)
from trajectory_extractor.rlmf_types import BehavioralEvaluationRecord, ParsedRLMFOutput


_ARMS = ("standard_grpo", "rlmf")
_JUDGMENT_TYPES = ("correctness", "equivalence")
_CONFIRMATORY_SEEDS = (11, 22, 33)
_PAIR_PATTERN = re.compile(
    r"<sentence>.*?</sentence>\s*"
    r"<confidence>\s*([01](?:\.\d+)?|0?\.\d+)\s*</confidence>",
    re.DOTALL,
)
_CONFIDENCE_PATTERN = re.compile(
    r"<confidence>\s*(0(\.\d*)?|1(\.0*)?)\s*</confidence>"
)


@dataclass(frozen=True)
class CalibrationMetricsResult:
    status: str
    reason: str | None
    total_records: int
    valid_format_records: int
    complete_case_records: int
    retained_record_ids: tuple[tuple[str, int, str], ...]
    complete_case_record_ids: tuple[tuple[str, int, str], ...]
    format_validity: float
    answer_coverage: float
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.status not in {"evaluable", "not_evaluable"}:
            raise ValueError("calibration status is invalid")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "total_records": self.total_records,
            "valid_format_records": self.valid_format_records,
            "complete_case_records": self.complete_case_records,
            "retained_record_ids": [list(value) for value in self.retained_record_ids],
            "complete_case_record_ids": [
                list(value) for value in self.complete_case_record_ids
            ],
            "format_validity": self.format_validity,
            "answer_coverage": self.answer_coverage,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class JudgeBiasAdjustedDeltaResult:
    status: str
    reason: str | None
    total_pairs: int
    complete_pairs: int
    excluded_pair_count: int
    excluded_pair_ids: tuple[tuple[int, str], ...]
    proxy_delta_cmfg_star: Interval | None
    adjusted_delta_cmfg_star: Interval | None
    absolute_differential_bias: Interval | None

    def __post_init__(self) -> None:
        if self.status not in {"evaluable", "not_evaluable"}:
            raise ValueError("judge-bias status is invalid")
        if self.total_pairs < 0 or self.complete_pairs < 0:
            raise ValueError("judge-bias pair counts must be non-negative")
        if self.complete_pairs + self.excluded_pair_count != self.total_pairs:
            raise ValueError("judge-bias pair counts are inconsistent")
        if self.status == "evaluable":
            if self.reason is not None or any(
                interval is None
                for interval in (
                    self.proxy_delta_cmfg_star,
                    self.adjusted_delta_cmfg_star,
                    self.absolute_differential_bias,
                )
            ):
                raise ValueError("evaluable judge-bias result requires all intervals")
        elif self.reason is None or any(
            interval is not None
            for interval in (
                self.proxy_delta_cmfg_star,
                self.adjusted_delta_cmfg_star,
                self.absolute_differential_bias,
            )
        ):
            raise ValueError("not-evaluable judge-bias result must contain only a reason")

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "total_pairs": self.total_pairs,
            "complete_pairs": self.complete_pairs,
            "excluded_pair_count": self.excluded_pair_count,
            "excluded_pair_ids": [list(value) for value in self.excluded_pair_ids],
            "proxy_delta_cmfg_star": (
                None
                if self.proxy_delta_cmfg_star is None
                else self.proxy_delta_cmfg_star.to_record()
            ),
            "adjusted_delta_cmfg_star": (
                None
                if self.adjusted_delta_cmfg_star is None
                else self.adjusted_delta_cmfg_star.to_record()
            ),
            "absolute_differential_bias": (
                None
                if self.absolute_differential_bias is None
                else self.absolute_differential_bias.to_record()
            ),
        }


def training_leave_one_out_confidence(
    answers: Sequence[str], aliases_by_answer: Mapping[str, Sequence[str]]
) -> np.ndarray:
    """Return each group member's agreement rate against the other members."""
    answers = _string_sequence(answers, "answers")
    if len(answers) < 2:
        raise ValueError("leave-one-out confidence requires at least two answers")
    if not isinstance(aliases_by_answer, Mapping):
        raise ValueError("aliases_by_answer must be a mapping")
    alias_sets = [_answer_aliases(answer, aliases_by_answer) for answer in answers]
    denominator = len(answers) - 1
    return np.asarray(
        [
            sum(
                _alias_sets_equivalent(
                    answers[index], aliases, answers[other], alias_sets[other]
                )
                for other in range(len(answers))
                if other != index
            )
            / denominator
            for index, aliases in enumerate(alias_sets)
        ],
        dtype=float,
    )


def evaluation_intrinsic_confidence(
    designated: str, auxiliaries: Sequence[str], aliases: Sequence[str]
) -> float:
    """Compute g_eval from exactly 20 independent auxiliary responses."""
    if not isinstance(designated, str) or not normalized_answer(designated):
        raise ValueError("designated answer must be a non-empty string")
    auxiliaries = _string_sequence(auxiliaries, "auxiliaries")
    if len(auxiliaries) != 20:
        raise ValueError("evaluation confidence requires exactly 20 auxiliaries")
    aliases = _string_sequence(aliases, "aliases")
    if not aliases:
        raise ValueError("aliases must not be empty")
    return (
        sum(completion_equivalent(designated, answer, aliases) for answer in auxiliaries)
        / 20.0
    )


def faithfulness_accuracy(confidence: float, intrinsic: float) -> float:
    confidence = _unit_scalar(confidence, "confidence")
    intrinsic = _unit_scalar(intrinsic, "intrinsic")
    return 1.0 - abs(confidence - intrinsic)


def faithful_calibration_reward(
    confidence: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    confidence, intrinsic = _paired_unit_arrays(
        confidence, intrinsic, "confidence", "intrinsic"
    )
    return 1.0 - np.square(confidence - intrinsic)


def factual_calibration_reward(confidence: np.ndarray, correctness: np.ndarray) -> np.ndarray:
    confidence, correctness = _paired_unit_arrays(
        confidence, correctness, "confidence", "correctness"
    )
    return 1.0 - np.square(confidence - correctness)


def gold_faithfulness_level(
    confidence: Any, intrinsic: Any, tau: float = 0.10
) -> np.ndarray:
    confidence, intrinsic = _paired_unit_arrays(
        confidence, intrinsic, "confidence", "intrinsic"
    )
    tau = _unit_scalar(tau, "tau")
    return (np.abs(confidence - intrinsic) <= tau).astype(float)


def metacognitive_reward(metascore: Any, gold_level: Any) -> np.ndarray:
    metascore, gold_level = _paired_unit_arrays(
        metascore, gold_level, "metascore", "gold_level"
    )
    return 1.0 - np.square(metascore - gold_level)


def strict_format_reward(parsed: Sequence[ParsedRLMFOutput]) -> np.ndarray:
    """Score the project's local exact-two-tag schema, not upstream parity."""
    if isinstance(parsed, (str, bytes)) or not isinstance(parsed, Sequence):
        raise ValueError("parsed outputs must be a sequence")
    values = tuple(parsed)
    if any(not isinstance(value, ParsedRLMFOutput) for value in values):
        raise ValueError("strict format rewards require ParsedRLMFOutput values")
    return np.asarray(
        [1.0 if value.valid_format else -1.0 for value in values], dtype=float
    )


def soft_format_reward(texts: Sequence[str]) -> np.ndarray:
    """Keep upstream-derived approximate-format fixtures frozen locally."""
    texts = _string_sequence(texts, "texts", allow_empty=True)
    return np.asarray([_soft_format_one(text) for text in texts], dtype=float)


def cmfg_star(confidence: Any, intrinsic: Any, *, bins: int = 10) -> float:
    """Compute official equal-mass, confidence-axis-width weighted cMFG*."""
    confidence, intrinsic = _paired_unit_arrays(
        confidence, intrinsic, "confidence", "intrinsic"
    )
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    if confidence.size == 0:
        raise ValueError("cMFG* requires at least one observation")
    faithfulness = 1.0 - np.abs(confidence - intrinsic)
    order = np.argsort(confidence)
    sorted_confidence = confidence[order]
    sorted_faithfulness = faithfulness[order]
    bin_count = min(bins, confidence.size)
    quotient, remainder = divmod(confidence.size, bin_count)
    slices: list[tuple[int, int]] = []
    start = 0
    for index in range(bin_count):
        stop = start + quotient + (1 if index < remainder else 0)
        slices.append((start, stop))
        start = stop
    means = np.asarray(
        [float(np.mean(sorted_faithfulness[start:stop])) for start, stop in slices]
    )
    widths = _slice_widths(sorted_confidence, slices)
    total_width = float(np.sum(widths))
    result = (
        float(np.mean(means))
        if total_width == 0.0
        else float(np.dot(widths, means) / total_width)
    )
    return _finite_result(result, "cMFG*")


def cmfg_tie_preserving(confidence: Any, intrinsic: Any) -> float:
    """Compute cMFG* with one bin per observed confidence value."""
    confidence, intrinsic = _paired_unit_arrays(
        confidence, intrinsic, "confidence", "intrinsic"
    )
    if confidence.size == 0:
        raise ValueError("tie-preserving cMFG requires at least one observation")
    faithfulness = 1.0 - np.abs(confidence - intrinsic)
    levels = np.unique(confidence)
    means = np.asarray(
        [float(np.mean(faithfulness[confidence == level])) for level in levels]
    )
    widths = _level_widths(levels)
    total_width = float(np.sum(widths))
    result = (
        float(np.mean(means))
        if total_width == 0.0
        else float(np.dot(widths, means) / total_width)
    )
    return _finite_result(result, "tie-preserving cMFG")


def common_support_sensitivity(
    standard_records: Sequence[Any], rlmf_records: Sequence[Any]
) -> Mapping[str, float]:
    standard = _metric_rows(standard_records, require_outcomes=False)
    rlmf = _metric_rows(rlmf_records, require_outcomes=False)
    shared_levels = np.intersect1d(
        np.unique(standard["confidence"]), np.unique(rlmf["confidence"])
    )
    if shared_levels.size == 0:
        raise ValueError("arms have no common confidence support")
    lower = float(shared_levels[0])
    upper = float(shared_levels[-1])
    standard_mask = np.isin(standard["confidence"], shared_levels)
    rlmf_mask = np.isin(rlmf["confidence"], shared_levels)
    standard_primary = cmfg_star(
        standard["confidence"][standard_mask], standard["intrinsic"][standard_mask]
    )
    rlmf_primary = cmfg_star(
        rlmf["confidence"][rlmf_mask], rlmf["intrinsic"][rlmf_mask]
    )
    standard_ties = cmfg_tie_preserving(
        standard["confidence"][standard_mask], standard["intrinsic"][standard_mask]
    )
    rlmf_ties = cmfg_tie_preserving(
        rlmf["confidence"][rlmf_mask], rlmf["intrinsic"][rlmf_mask]
    )
    return {
        "support_lower": lower,
        "support_upper": upper,
        "standard_cmfg_star": standard_primary,
        "rlmf_cmfg_star": rlmf_primary,
        "delta_cmfg_star": rlmf_primary - standard_primary,
        "standard_cmfg_tie_preserving": standard_ties,
        "rlmf_cmfg_tie_preserving": rlmf_ties,
        "delta_cmfg_tie_preserving": rlmf_ties - standard_ties,
    }


def calibration_metrics(
    records: Sequence[BehavioralEvaluationRecord],
) -> CalibrationMetricsResult:
    retained = _behavioral_evaluation_records(records)
    complete = tuple(row for row in retained if row.valid_complete_case)
    retained_ids = tuple(row.record_id for row in retained)
    complete_ids = tuple(row.record_id for row in complete)
    valid_format_records = sum(row.valid_format for row in retained)
    format_validity = valid_format_records / len(retained)
    answer_coverage = sum(
        bool(normalized_answer(row.designated.answer)) for row in retained
    ) / len(retained)
    if not complete:
        return CalibrationMetricsResult(
            status="not_evaluable",
            reason="no_valid_complete_cases",
            total_records=len(retained),
            valid_format_records=valid_format_records,
            complete_case_records=0,
            retained_record_ids=retained_ids,
            complete_case_record_ids=(),
            format_validity=format_validity,
            answer_coverage=answer_coverage,
            metrics={},
        )
    confidence = np.asarray([row.confidence for row in complete], dtype=float)
    intrinsic = np.asarray([row.intrinsic for row in complete], dtype=float)
    correctness = np.asarray([float(row.correctness) for row in complete])
    faithfulness = 1.0 - np.abs(confidence - intrinsic)
    metrics = {
        "cmfg_star": cmfg_star(confidence, intrinsic),
        "cmfg_tie_preserving": cmfg_tie_preserving(confidence, intrinsic),
        "faithfulness_accuracy": float(np.mean(faithfulness)),
        "accuracy": float(np.mean(correctness)),
        "intrinsic_brier": float(np.mean(np.square(intrinsic - correctness))),
        "expressed_brier": float(np.mean(np.square(confidence - correctness))),
        "ece": _ece(confidence, correctness),
        "absolute_expression_gap": float(np.mean(np.abs(confidence - intrinsic))),
    }
    for name, value in metrics.items():
        _finite_result(value, name)
    return CalibrationMetricsResult(
        status="evaluable",
        reason=None,
        total_records=len(retained),
        valid_format_records=valid_format_records,
        complete_case_records=len(complete),
        retained_record_ids=retained_ids,
        complete_case_record_ids=complete_ids,
        format_validity=format_validity,
        answer_coverage=answer_coverage,
        metrics=metrics,
    )


def paired_fixed_seed_prompt_bootstrap(
    records: Sequence[Any],
    metric: Callable[[Sequence[Any]], float] | str,
    *,
    seeds: Sequence[int],
    replicates: int,
    rng_seed: int,
) -> Interval:
    """Bootstrap paired prompts within each fixed seed; seeds are never sampled."""
    rows, groups = _paired_prompt_groups(records, seeds)
    replicates = _positive_int(replicates, "replicates")
    rng_seed = _integer(rng_seed, "rng_seed")
    estimate = _metric_value(metric, rows)
    rng = np.random.default_rng(rng_seed)
    values = []
    for _ in range(replicates):
        sampled: list[Any] = []
        for seed in seeds:
            prompt_groups = groups[seed]
            selected = rng.integers(0, len(prompt_groups), size=len(prompt_groups))
            for index in selected:
                sampled.extend(prompt_groups[int(index)])
        values.append(_metric_value(metric, sampled))
    return _percentile_interval(estimate, values)


def judge_bias_adjusted_delta(
    records: Sequence[BehavioralEvaluationRecord],
    audit: Mapping[str, Any],
    *,
    rng_seed: int,
) -> JudgeBiasAdjustedDeltaResult:
    """Propagate every aligned confusion draw through paired delta cMFG*."""
    records = _behavioral_evaluation_records(records)
    confusion = _validate_confusion_uncertainty(audit)
    complete_records, total_pairs, excluded_pair_ids, complete_by_seed = (
        _paired_behavioral_complete_cases(records)
    )
    if not complete_records:
        return _not_evaluable_judge_bias_result(
            "no_complete_pairs", total_pairs, excluded_pair_ids
        )
    if any(not complete_by_seed[seed] for seed in _CONFIRMATORY_SEEDS):
        return _not_evaluable_judge_bias_result(
            "registered_seed_has_no_complete_pairs", total_pairs, excluded_pair_ids
        )
    rows = _judge_rows(complete_records)
    _, groups = _paired_prompt_groups(rows, _CONFIRMATORY_SEEDS)
    rng_seed = _integer(rng_seed, "rng_seed")
    estimates = {
        arm: (
            confusion["estimates"][f"{arm}:equivalence"]["sensitivity"]["estimate"],
            confusion["estimates"][f"{arm}:equivalence"]["specificity"]["estimate"],
        )
        for arm in _ARMS
    }
    proxy_estimate = _proxy_delta(rows)
    adjusted_estimate = _adjusted_delta(rows, estimates)
    rng = np.random.default_rng(rng_seed)
    proxy_values = []
    adjusted_values = []
    absolute_bias_values = []
    for draw in confusion["joint_draws"]:
        sampled: list[Mapping[str, Any]] = []
        for seed in _CONFIRMATORY_SEEDS:
            prompt_groups = groups[seed]
            selected = rng.integers(0, len(prompt_groups), size=len(prompt_groups))
            for index in selected:
                sampled.extend(prompt_groups[int(index)])
        draw_confusion = {
            arm: (
                draw[f"{arm}:equivalence"]["sensitivity"],
                draw[f"{arm}:equivalence"]["specificity"],
            )
            for arm in _ARMS
        }
        proxy_delta = _proxy_delta(sampled)
        adjusted_delta = _adjusted_delta(sampled, draw_confusion)
        proxy_values.append(proxy_delta)
        adjusted_values.append(adjusted_delta)
        absolute_bias_values.append(abs(adjusted_delta - proxy_delta))
    return JudgeBiasAdjustedDeltaResult(
        status="evaluable",
        reason=None,
        total_pairs=total_pairs,
        complete_pairs=total_pairs - len(excluded_pair_ids),
        excluded_pair_count=len(excluded_pair_ids),
        excluded_pair_ids=excluded_pair_ids,
        proxy_delta_cmfg_star=_percentile_interval(proxy_estimate, proxy_values),
        adjusted_delta_cmfg_star=_percentile_interval(adjusted_estimate, adjusted_values),
        absolute_differential_bias=_percentile_interval(
            abs(adjusted_estimate - proxy_estimate), absolute_bias_values
        ),
    )


def _soft_format_one(text: str) -> float:
    sentence_open = text.count("<sentence>")
    sentence_close = text.count("</sentence>")
    confidence_open = text.count("<confidence>")
    confidence_close = text.count("</confidence>")
    valid_confidence = len(_CONFIDENCE_PATTERN.findall(text))
    valid_pairs = len(_PAIR_PATTERN.findall(text))
    confidence_content = re.findall(r"<confidence>(.*?)</confidence>", text, re.DOTALL)
    confidence_with_text = sum(
        bool(content.strip()) and re.fullmatch(r"\s*-?\d+(\.\d*)?\s*", content) is None
        for content in confidence_content
    )
    empty_sentences = _empty_tag_count(text, "sentence")
    empty_confidences = _empty_tag_count(text, "confidence")
    if sentence_open == 0:
        return -1.0
    if sentence_open != sentence_close or confidence_open != confidence_close:
        reward = 0.0
        if sentence_open != sentence_close:
            denominator = max(sentence_open, sentence_close)
            reward -= 0.5 * (1.0 - _valid_sentence_count(text) / denominator)
        if confidence_open != confidence_close:
            denominator = max(confidence_open, confidence_close)
            reward -= 0.5 * (1.0 - valid_confidence / denominator)
        return reward
    if valid_pairs == 0:
        return -1.0
    penalties = []
    if sentence_open != confidence_open:
        penalties.append(
            -0.25
            * abs(sentence_open - confidence_open)
            / max(sentence_open, confidence_open)
        )
    if confidence_open > 0:
        penalties.append(-0.25 * (confidence_open - valid_confidence) / confidence_open)
    if confidence_open > 0 and confidence_with_text > 0:
        penalties.append(-0.2 * confidence_with_text / confidence_open)
    expected_pairs = min(sentence_open, confidence_open)
    if expected_pairs > 0:
        penalties.append(-0.3 * (expected_pairs - valid_pairs) / expected_pairs)
    if sentence_open > 0 and empty_sentences > 0:
        penalties.append(-0.05 * empty_sentences / sentence_open)
    if confidence_open > 0 and empty_confidences > 0:
        penalties.append(-0.05 * empty_confidences / confidence_open)
    return float(sum(penalties))


def _empty_tag_count(text: str, tag: str) -> int:
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    return sum(
        not re.sub(r"[^\w]", "", re.sub(r"<[^>]+>", "", value))
        for value in pattern.findall(text)
    )


def _valid_sentence_count(text: str) -> int:
    return sum(
        "<sentence>" not in value
        and "</sentence>" not in value
        and bool(re.sub(r"[^\w]", "", re.sub(r"<[^>]+>", "", value)))
        for value in re.findall(r"<sentence>(.*?)</sentence>", text, re.DOTALL)
    )


def _slice_widths(
    confidence: np.ndarray, slices: Sequence[tuple[int, int]]
) -> np.ndarray:
    widths = []
    for index, (start, stop) in enumerate(slices):
        lower = (
            confidence[start]
            if index == 0
            else (confidence[slices[index - 1][1] - 1] + confidence[start]) / 2
        )
        upper = (
            confidence[stop - 1]
            if index == len(slices) - 1
            else (confidence[stop - 1] + confidence[slices[index + 1][0]]) / 2
        )
        widths.append(float(upper - lower))
    return np.asarray(widths, dtype=float)


def _level_widths(levels: np.ndarray) -> np.ndarray:
    if levels.size == 1:
        return np.zeros(1, dtype=float)
    boundaries = (levels[:-1] + levels[1:]) / 2
    lower = np.concatenate(([levels[0]], boundaries))
    upper = np.concatenate((boundaries, [levels[-1]]))
    return upper - lower


def _metric_rows(
    records: Sequence[Any], *, require_outcomes: bool
) -> dict[str, np.ndarray]:
    records = _nonempty_records(records)
    confidence = np.asarray([_record_unit(row, "confidence") for row in records])
    intrinsic = np.asarray([_record_unit(row, "intrinsic") for row in records])
    result = {"confidence": confidence, "intrinsic": intrinsic}
    if require_outcomes:
        result["correctness"] = np.asarray(
            [_record_unit(row, "correctness") for row in records]
        )
        result["valid_format"] = np.asarray(
            [float(_record_bool(row, "valid_format")) for row in records]
        )
        result["answer_coverage"] = np.asarray(
            [float(bool(normalized_answer(_record_value(row, "answer")))) for row in records]
        )
    return result


def _behavioral_evaluation_records(
    records: Sequence[BehavioralEvaluationRecord],
) -> tuple[BehavioralEvaluationRecord, ...]:
    values = _nonempty_records(records)
    if any(not isinstance(value, BehavioralEvaluationRecord) for value in values):
        raise ValueError(
            "calibration metrics require BehavioralEvaluationRecord values"
        )
    return values


def _paired_prompt_groups(
    records: Sequence[Any], seeds: Sequence[int]
) -> tuple[tuple[Any, ...], dict[int, tuple[tuple[Any, Any], ...]]]:
    rows = _nonempty_records(records)
    seeds = tuple(seeds)
    if not (
        seeds == _CONFIRMATORY_SEEDS
        or (len(seeds) == 1 and seeds[0] in _CONFIRMATORY_SEEDS)
    ):
        raise ValueError(
            "seeds must contain one registered seed or exactly (11, 22, 33)"
        )
    if {_record_value(row, "seed") for row in rows} != set(seeds):
        raise ValueError("records must contain exactly the requested fixed seeds")
    indexed: dict[tuple[int, str, str], Any] = {}
    for row in rows:
        arm = _record_value(row, "arm")
        seed = _record_value(row, "seed")
        example_id = _record_value(row, "example_id")
        if (
            arm not in _ARMS
            or type(seed) is not int
            or not isinstance(example_id, str)
            or not example_id
        ):
            raise ValueError("bootstrap rows have malformed arm, seed, or example_id")
        key = (seed, example_id, arm)
        if key in indexed:
            raise ValueError(
                "bootstrap rows must be unique by seed, example_id, and arm"
            )
        indexed[key] = row
    groups: dict[int, tuple[tuple[Any, Any], ...]] = {}
    for seed in seeds:
        prompt_ids = sorted({key[1] for key in indexed if key[0] == seed})
        paired = []
        for prompt_id in prompt_ids:
            try:
                paired.append(tuple(indexed[(seed, prompt_id, arm)] for arm in _ARMS))
            except KeyError as error:
                raise ValueError("bootstrap requires paired arm records within every seed") from error
        if not paired:
            raise ValueError("each fixed seed requires at least one paired prompt")
        groups[seed] = tuple(paired)
    if sum(len(value) * 2 for value in groups.values()) != len(rows):
        raise ValueError("bootstrap records contain unpaired rows")
    return rows, groups


def _judge_rows(records: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    result = []
    for row in _behavioral_evaluation_records(records):
        if not row.valid_complete_case:
            raise ValueError("judge adjustment requires valid complete-case records")
        result.append(
            {
                "arm": row.arm,
                "seed": row.seed,
                "example_id": row.example_id,
                "confidence": _unit_scalar(row.confidence, "confidence"),
                "auxiliary_proxy_labels": row.auxiliary_proxy_labels,
            }
        )
    return tuple(result)


def _paired_behavioral_complete_cases(
    records: Sequence[BehavioralEvaluationRecord],
) -> tuple[
    tuple[BehavioralEvaluationRecord, ...],
    int,
    tuple[tuple[int, str], ...],
    dict[int, tuple[tuple[BehavioralEvaluationRecord, BehavioralEvaluationRecord], ...]],
]:
    rows = _behavioral_evaluation_records(records)
    if {row.seed for row in rows} != set(_CONFIRMATORY_SEEDS):
        raise ValueError("records must contain exactly the registered fixed seeds")
    indexed: dict[tuple[int, str, str], BehavioralEvaluationRecord] = {}
    for row in rows:
        key = (row.seed, row.example_id, row.arm)
        if key in indexed:
            raise ValueError("behavioral records must be unique by seed, example_id, and arm")
        indexed[key] = row
    all_pairs: list[tuple[BehavioralEvaluationRecord, BehavioralEvaluationRecord]] = []
    for seed in _CONFIRMATORY_SEEDS:
        prompt_ids = sorted({example_id for current_seed, example_id, _arm in indexed if current_seed == seed})
        for example_id in prompt_ids:
            try:
                all_pairs.append(
                    tuple(indexed[(seed, example_id, arm)] for arm in _ARMS)
                )
            except KeyError as error:
                raise ValueError(
                    "judge adjustment requires raw paired arm records within every seed"
                ) from error
    complete_by_seed: dict[
        int, list[tuple[BehavioralEvaluationRecord, BehavioralEvaluationRecord]]
    ] = {seed: [] for seed in _CONFIRMATORY_SEEDS}
    excluded_pair_ids = []
    for pair in all_pairs:
        seed, example_id = pair[0].seed, pair[0].example_id
        if all(row.valid_complete_case for row in pair):
            complete_by_seed[seed].append(pair)
        else:
            excluded_pair_ids.append((seed, example_id))
    frozen_complete_by_seed = {
        seed: tuple(pairs) for seed, pairs in complete_by_seed.items()
    }
    complete_records = tuple(
        row for seed in _CONFIRMATORY_SEEDS for pair in frozen_complete_by_seed[seed] for row in pair
    )
    return (
        complete_records,
        len(all_pairs),
        tuple(excluded_pair_ids),
        frozen_complete_by_seed,
    )


def _not_evaluable_judge_bias_result(
    reason: str, total_pairs: int, excluded_pair_ids: tuple[tuple[int, str], ...]
) -> JudgeBiasAdjustedDeltaResult:
    return JudgeBiasAdjustedDeltaResult(
        status="not_evaluable",
        reason=reason,
        total_pairs=total_pairs,
        complete_pairs=total_pairs - len(excluded_pair_ids),
        excluded_pair_count=len(excluded_pair_ids),
        excluded_pair_ids=excluded_pair_ids,
        proxy_delta_cmfg_star=None,
        adjusted_delta_cmfg_star=None,
        absolute_differential_bias=None,
    )


def _adjusted_delta(
    rows: Sequence[Mapping[str, Any]], confusion: Mapping[str, tuple[float, float]]
) -> float:
    arm_values: dict[str, tuple[list[float], list[float]]] = {
        arm: ([], []) for arm in _ARMS
    }
    for row in rows:
        arm = row["arm"]
        sensitivity, specificity = confusion[arm]
        denominator = sensitivity + specificity - 1.0
        if denominator <= 0.0:
            raise ValueError(
                "confusion correction is not identifiable when "
                "sensitivity + specificity <= 1"
            )
        proxy_rate = float(np.mean(row["auxiliary_proxy_labels"]))
        intrinsic = float(
            np.clip((proxy_rate + specificity - 1.0) / denominator, 0.0, 1.0)
        )
        arm_values[arm][0].append(row["confidence"])
        arm_values[arm][1].append(intrinsic)
    standard = cmfg_star(*arm_values["standard_grpo"])
    rlmf = cmfg_star(*arm_values["rlmf"])
    return _finite_result(rlmf - standard, "judge-bias-adjusted delta cMFG*")


def _proxy_delta(rows: Sequence[Mapping[str, Any]]) -> float:
    arm_values: dict[str, tuple[list[float], list[float]]] = {
        arm: ([], []) for arm in _ARMS
    }
    for row in rows:
        arm = row["arm"]
        arm_values[arm][0].append(row["confidence"])
        arm_values[arm][1].append(float(np.mean(row["auxiliary_proxy_labels"])))
    standard = cmfg_star(*arm_values["standard_grpo"])
    rlmf = cmfg_star(*arm_values["rlmf"])
    return _finite_result(rlmf - standard, "proxy delta cMFG*")


def _validate_confusion_uncertainty(audit: Any) -> dict[str, Any]:
    if not isinstance(audit, Mapping):
        raise ValueError("judge adjustment requires sealed confusion uncertainty")
    if audit.get("schema_version") != 2:
        raise ValueError("judge adjustment requires schema-v2 confusion uncertainty")
    if audit.get("method") != "deterministic_stratified_bootstrap":
        raise ValueError("confusion uncertainty method is not registered")
    if audit.get("replicates") != 2000 or audit.get("rng_seed") != 20260713:
        raise ValueError("confusion uncertainty bootstrap provenance is not registered")
    design = audit.get("sampling_design")
    if not isinstance(design, Mapping) or design.get("schema_version") != 1:
        raise ValueError("confusion uncertainty is missing its sampling design")
    if design.get("stratified_on") != ["group", "judgment_type", "proxy_label"]:
        raise ValueError("confusion uncertainty sampling design is malformed")
    _validate_sampling_design(design)
    estimates = audit.get("estimates")
    expected = {
        f"{arm}:{judgment_type}"
        for arm in _ARMS
        for judgment_type in _JUDGMENT_TYPES
    }
    if not isinstance(estimates, Mapping) or set(estimates) != expected:
        raise ValueError("confusion uncertainty must be arm-specific for both judgments")
    validated = {}
    for key, value in estimates.items():
        if not isinstance(value, Mapping) or set(value) != {"sensitivity", "specificity"}:
            raise ValueError("confusion uncertainty estimate is malformed")
        validated[key] = {
            name: _interval_record(value[name], f"{key} {name}")
            for name in ("sensitivity", "specificity")
        }
    draws = audit.get("joint_draws")
    if (
        isinstance(draws, (str, bytes))
        or not isinstance(draws, Sequence)
        or len(draws) != audit["replicates"]
    ):
        raise ValueError(
            "confusion uncertainty joint draws must match the registered replicates"
        )
    validated_draws = []
    for draw in draws:
        if not isinstance(draw, Mapping) or set(draw) != expected:
            raise ValueError("confusion uncertainty joint draw is malformed")
        validated_draw = {}
        for key, value in draw.items():
            if not isinstance(value, Mapping) or set(value) != {
                "sensitivity",
                "specificity",
            }:
                raise ValueError("confusion uncertainty joint draw is malformed")
            validated_draw[key] = {
                name: _unit_scalar(value[name], f"{key} joint {name}")
                for name in ("sensitivity", "specificity")
            }
        validated_draws.append(validated_draw)
    return {"estimates": validated, "joint_draws": tuple(validated_draws)}


def _validate_sampling_design(design: Mapping[str, Any]) -> None:
    strata = design.get("strata")
    expected = {
        f"{arm}:{judgment_type}:proxy_{str(proxy_label).lower()}": (
            arm,
            judgment_type,
            proxy_label,
        )
        for arm in _ARMS
        for judgment_type in _JUDGMENT_TYPES
        for proxy_label in (False, True)
    }
    if not isinstance(strata, Mapping) or set(strata) != set(expected):
        raise ValueError("confusion uncertainty sampling design strata are malformed")
    record_fields = {
        "group",
        "judgment_type",
        "proxy_label",
        "population_count",
        "sample_count",
        "inclusion_probability",
    }
    for key, identity in expected.items():
        record = strata[key]
        if not isinstance(record, Mapping) or set(record) != record_fields:
            raise ValueError("confusion uncertainty sampling design row is malformed")
        if (
            (record["group"], record["judgment_type"], record["proxy_label"])
            != identity
        ):
            raise ValueError("confusion uncertainty sampling design identity is malformed")
        population = record["population_count"]
        sample = record["sample_count"]
        if (
            type(population) is not int
            or type(sample) is not int
            or population < 1
            or sample < 1
            or sample > population
        ):
            raise ValueError("confusion uncertainty sampling design counts are malformed")
        probability = record["inclusion_probability"]
        if (
            not isinstance(probability, Mapping)
            or set(probability) != {"numerator", "denominator"}
            or type(probability["numerator"]) is not int
            or type(probability["denominator"]) is not int
            or probability["numerator"] < 1
            or probability["denominator"] < 1
            or probability["numerator"] * population
            != probability["denominator"] * sample
        ):
            raise ValueError(
                "confusion uncertainty sampling design inclusion probability "
                "is malformed"
            )


def _interval_record(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"lower", "estimate", "upper"}:
        raise ValueError(f"{name} interval is malformed")
    result = {key: _unit_scalar(value[key], f"{name} {key}") for key in value}
    if not result["lower"] <= result["upper"]:
        raise ValueError(f"{name} interval bounds are invalid")
    return result


def _metric_value(
    metric: Callable[[Sequence[Any]], float] | str, rows: Sequence[Any]
) -> float:
    if isinstance(metric, str):
        result = calibration_metrics(rows)
        if result.status != "evaluable":
            raise ValueError(f"calibration metric is not evaluable: {result.reason}")
        value = result.metrics.get(metric)
        if value is None:
            raise ValueError(f"unknown calibration metric: {metric}")
    elif callable(metric):
        value = metric(rows)
    else:
        raise ValueError("metric must be a callable or calibration metric name")
    return _finite_result(value, "bootstrap metric")


def _percentile_interval(estimate: float, values: Sequence[float]) -> Interval:
    lower, upper = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return Interval(
        lower=float(lower),
        estimate=estimate,
        upper=float(upper),
    )


def _ece(confidence: np.ndarray, correctness: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(
        np.digitize(confidence, boundaries, right=False) - 1, 0, bins - 1
    )
    result = 0.0
    for index in range(bins):
        mask = indices == index
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(correctness[mask])) - float(np.mean(confidence[mask]))
            )
    return result


def _answer_aliases(
    answer: str, aliases_by_answer: Mapping[str, Sequence[str]]
) -> frozenset[str]:
    aliases = aliases_by_answer.get(answer, (answer,))
    aliases = _string_sequence(aliases, f"aliases for {answer!r}")
    if not aliases:
        raise ValueError("answer aliases must not be empty")
    return frozenset(normalized_answer(value) for value in (*aliases, answer))


def _alias_sets_equivalent(
    left: str, left_aliases: frozenset[str], right: str, right_aliases: frozenset[str]
) -> bool:
    return normalized_answer(left) == normalized_answer(right) or bool(
        left_aliases & right_aliases
    )


def _paired_unit_arrays(
    left: Any, right: Any, left_name: str, right_name: str
) -> tuple[np.ndarray, np.ndarray]:
    left_array = _unit_array(left, left_name)
    right_array = _unit_array(right, right_name)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"{left_name} and {right_name} must have matching one-dimensional shapes"
        )
    return left_array, right_array


def _unit_array(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} values must be in [0, 1]")
    return array


def _unit_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _record_unit(record: Any, name: str) -> float:
    return _unit_scalar(_record_value(record, name), name)


def _record_bool(record: Any, name: str) -> bool:
    value = _record_value(record, name)
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _record_value(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        if name not in record:
            raise ValueError(f"behavioral record is missing {name}")
        return record[name]
    if not hasattr(record, name):
        raise ValueError(f"behavioral record is missing {name}")
    return getattr(record, name)


def _nonempty_records(records: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence")
    rows = tuple(records)
    if not rows:
        raise ValueError("records must not be empty")
    return rows


def _string_sequence(
    values: Sequence[str], name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a non-string sequence")
    result = tuple(values)
    if any(not isinstance(value, str) for value in result):
        raise ValueError(f"{name} must contain strings")
    if not allow_empty and any(not normalized_answer(value) for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


def _finite_result(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must return a finite scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must return a finite scalar")
    return value


def _positive_int(value: Any, name: str) -> int:
    value = _integer(value, name)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value
