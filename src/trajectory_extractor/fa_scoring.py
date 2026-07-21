"""Frozen exact scoring and behavioral estimands for Familiarity-vs-Answerability.

The module deliberately accepts an ``FAExample``-like object rather than importing
the dataset type.  This keeps scoring usable on persisted, provenance-checked rows
without coupling it to generation or artifact I/O.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any
import unicodedata
import re

import numpy as np

from trajectory_extractor.fa_config import CONFIRMATORY_THRESHOLDS


Cell = tuple[str, str, str]
_PRIMARY_TARGETS = ("screened_real", "matched_synthetic")
_PRIMARY_ANSWERABILITY = ("target_bound", "distractor_bound", "code_absent")
_H2B_EXPOSURES = ("high_exposure", "low_exposure")
_H2B_ANSWERABILITY = ("target_bound", "code_absent")
_INFRASTRUCTURE_MARKERS = (
    "[[infrastructure",
    "<|infrastructure",
    "generation backend unavailable",
)
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class OutcomeClass(str, Enum):
    ABSTENTION = "abstention"
    EXACT_TARGET_CODE = "exact_target_code"
    DISTRACTOR_CODE_COPY = "distractor_code_copy"
    NOVEL_CODE_ASSERTION = "novel_code_assertion"
    OTHER_NON_ABSTENTION = "other_non_abstention"
    INVALID_FORMAT = "invalid_format"


@dataclass(frozen=True)
class ScoredResponse:
    """One raw response with scoring inputs retained for reproducible analysis."""

    example_id: str
    entity_unit_id: str
    template_family: str
    target_familiarity: str
    distractor_familiarity: str
    answerability: str
    block: str
    exposure: str | None
    registry_code: str
    raw_output: str | None
    normalized_output: str
    outcome: OutcomeClass
    answer_attempt: int
    valid_format: bool
    completed: bool
    infrastructure_marked: bool
    truncated: bool
    sampling_weight: int = 1

    def __post_init__(self) -> None:
        for field in (
            "example_id",
            "entity_unit_id",
            "template_family",
            "target_familiarity",
            "distractor_familiarity",
            "answerability",
            "block",
            "registry_code",
        ):
            _nonempty_text(getattr(self, field), field)
        if self.target_familiarity not in _PRIMARY_TARGETS:
            raise ValueError("target_familiarity is not registered")
        if self.distractor_familiarity not in _PRIMARY_TARGETS:
            raise ValueError("distractor_familiarity is not registered")
        if self.answerability not in _PRIMARY_ANSWERABILITY:
            raise ValueError("answerability is not registered")
        if self.block not in {"factorial", "same_string"}:
            raise ValueError("block is not registered")
        if self.block == "same_string" and self.exposure not in _H2B_EXPOSURES:
            raise ValueError("same_string rows require registered exposure")
        if self.block == "factorial" and self.exposure is not None:
            raise ValueError("factorial rows cannot have an exposure")
        if self.raw_output is not None and not isinstance(self.raw_output, str):
            raise ValueError("raw_output must be text or None")
        if not isinstance(self.normalized_output, str):
            raise ValueError("normalized_output must be text")
        if type(self.answer_attempt) is not int or self.answer_attempt not in {0, 1}:
            raise ValueError("answer_attempt must be 0 or 1")
        if type(self.valid_format) is not bool or type(self.completed) is not bool:
            raise ValueError("valid_format and completed must be boolean")
        if type(self.infrastructure_marked) is not bool or type(self.truncated) is not bool:
            raise ValueError("response markers must be boolean")
        if type(self.sampling_weight) is not int or self.sampling_weight <= 0:
            raise ValueError("sampling_weight must be positive")
        if self.answer_attempt != int(self.outcome is not OutcomeClass.ABSTENTION):
            raise ValueError("answer_attempt must follow the frozen abstention rule")
        if self.valid_format != (self.outcome is not OutcomeClass.INVALID_FORMAT):
            raise ValueError("valid_format must agree with outcome")

    def to_record(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "entity_unit_id": self.entity_unit_id,
            "template_family": self.template_family,
            "target_familiarity": self.target_familiarity,
            "distractor_familiarity": self.distractor_familiarity,
            "answerability": self.answerability,
            "block": self.block,
            "exposure": self.exposure,
            "registry_code": self.registry_code,
            "raw_output": self.raw_output,
            "normalized_output": self.normalized_output,
            "outcome": self.outcome.value,
            "answer_attempt": self.answer_attempt,
            "valid_format": self.valid_format,
            "completed": self.completed,
            "infrastructure_marked": self.infrastructure_marked,
            "truncated": self.truncated,
            "sampling_weight": self.sampling_weight,
        }


@dataclass(frozen=True)
class PercentileInterval:
    estimate: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in (self.estimate, self.lower, self.upper)):
            raise ValueError("interval values must be finite")
        if self.lower > self.upper:
            raise ValueError("interval lower bound must not exceed upper bound")

    def to_record(self) -> dict[str, float]:
        return {"estimate": self.estimate, "lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class SensitivityResult:
    name: str
    interaction: float | None
    analytic_denominators: Mapping[Cell, int]
    original_denominators: Mapping[Cell, int]
    invalid_format_counts: Mapping[Cell, int]

    def __post_init__(self) -> None:
        _nonempty_text(self.name, "name")
        object.__setattr__(self, "analytic_denominators", _freeze_cells(self.analytic_denominators))
        object.__setattr__(self, "original_denominators", _freeze_cells(self.original_denominators))
        object.__setattr__(self, "invalid_format_counts", _freeze_cells(self.invalid_format_counts))

    def to_record(self) -> dict[str, Any]:
        return {
            "interaction": self.interaction,
            "analytic_denominators": _cell_record(self.analytic_denominators),
            "original_denominators": _cell_record(self.original_denominators),
            "invalid_format_counts": _cell_record(self.invalid_format_counts),
        }


@dataclass(frozen=True)
class BehavioralMetrics:
    status: str
    reasons: tuple[str, ...]
    cell_rates: Mapping[Cell, float]
    completion_by_cell: Mapping[Cell, float]
    format_validity_by_cell: Mapping[Cell, float]
    denominators: Mapping[Cell, int]
    invalid_format_counts: Mapping[Cell, int]
    interaction: float | None
    h2_accuracy_difference: float | None
    h2b_interaction: float | None
    sensitivities: Mapping[str, SensitivityResult]

    def __post_init__(self) -> None:
        if self.status not in {"evaluable", "not_evaluable"}:
            raise ValueError("metrics status is invalid")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "cell_rates", _freeze_cells(self.cell_rates))
        object.__setattr__(self, "completion_by_cell", _freeze_cells(self.completion_by_cell))
        object.__setattr__(self, "format_validity_by_cell", _freeze_cells(self.format_validity_by_cell))
        object.__setattr__(self, "denominators", _freeze_cells(self.denominators))
        object.__setattr__(self, "invalid_format_counts", _freeze_cells(self.invalid_format_counts))
        object.__setattr__(self, "sensitivities", MappingProxyType(dict(self.sensitivities)))

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "cell_rates": _cell_record(self.cell_rates),
            "completion_by_cell": _cell_record(self.completion_by_cell),
            "format_validity_by_cell": _cell_record(self.format_validity_by_cell),
            "denominators": _cell_record(self.denominators),
            "invalid_format_counts": _cell_record(self.invalid_format_counts),
            "interaction": self.interaction,
            "h2_accuracy_difference": self.h2_accuracy_difference,
            "h2b_interaction": self.h2b_interaction,
            "sensitivities": {name: value.to_record() for name, value in self.sensitivities.items()},
        }


@dataclass(frozen=True)
class BootstrapDistribution:
    interaction_samples: tuple[float, ...]
    h2_accuracy_difference_samples: tuple[float, ...]
    h2b_interaction_samples: tuple[float, ...]
    interaction_interval: PercentileInterval | None
    h2_accuracy_difference_interval: PercentileInterval | None
    h2b_interaction_interval: PercentileInterval | None
    weighted_denominators: tuple[int, ...]
    seed: int

    def __post_init__(self) -> None:
        for field in (
            "interaction_samples",
            "h2_accuracy_difference_samples",
            "h2b_interaction_samples",
            "weighted_denominators",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if any(value <= 0 for value in self.weighted_denominators):
            raise ValueError("weighted denominators must be positive")

    def to_record(self) -> dict[str, Any]:
        return {
            "interaction_samples": list(self.interaction_samples),
            "h2_accuracy_difference_samples": list(self.h2_accuracy_difference_samples),
            "h2b_interaction_samples": list(self.h2b_interaction_samples),
            "interaction_interval": None if self.interaction_interval is None else self.interaction_interval.to_record(),
            "h2_accuracy_difference_interval": None
            if self.h2_accuracy_difference_interval is None
            else self.h2_accuracy_difference_interval.to_record(),
            "h2b_interaction_interval": None if self.h2b_interaction_interval is None else self.h2b_interaction_interval.to_record(),
            "weighted_denominators": list(self.weighted_denominators),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class GateDecision:
    status: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {"supported", "not_supported", "not_evaluable"}:
            raise ValueError("gate status is invalid")
        object.__setattr__(self, "reasons", tuple(self.reasons))

    def to_record(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class BehavioralGate:
    status: str
    h1: GateDecision
    h2: GateDecision
    h2b: GateDecision
    thresholds: Mapping[str, float]
    same_string_sealed: bool
    config_hash: str
    manifest_hash: str
    h2b_cannot_rescue_h1: bool = True

    def __post_init__(self) -> None:
        if self.status not in {"supported", "not_supported", "not_evaluable"}:
            raise ValueError("gate status is invalid")
        if self.h2b_cannot_rescue_h1 is not True:
            raise ValueError("H2b must not rescue H1")
        if dict(self.thresholds) != CONFIRMATORY_THRESHOLDS:
            raise ValueError("thresholds must match registered thresholds")
        if type(self.same_string_sealed) is not bool:
            raise ValueError("same_string_sealed must be boolean")
        _lowercase_sha256(self.config_hash, "config_hash")
        _lowercase_sha256(self.manifest_hash, "manifest_hash")
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "h1": self.h1.to_record(),
            "h2": self.h2.to_record(),
            "h2b": self.h2b.to_record(),
            "thresholds": dict(self.thresholds),
            "same_string_sealed": self.same_string_sealed,
            "config_hash": self.config_hash,
            "manifest_hash": self.manifest_hash,
            "h2b_cannot_rescue_h1": True,
        }


def score_response(
    example: Any,
    text: str | None,
    *,
    registered_codes: Collection[str],
    truncated: bool = False,
    infrastructure_marked: bool = False,
) -> ScoredResponse:
    """Score one response with the parser precedence fixed by preregistration."""
    if type(truncated) is not bool or type(infrastructure_marked) is not bool:
        raise ValueError("response markers must be boolean")
    if text is not None and not isinstance(text, str):
        raise ValueError("text must be a string or None")
    fields = _example_fields(example)
    vocabulary = _registered_code_vocabulary(registered_codes)
    if fields["row"]["registry_code"] not in vocabulary:
        raise ValueError("registered_codes must contain the example registry_code")
    normalized = "" if text is None else unicodedata.normalize("NFC", text).strip()
    marked = infrastructure_marked or _contains_infrastructure_marker(normalized)
    completed = text is not None and not marked
    if marked or truncated or text is None:
        outcome = OutcomeClass.INVALID_FORMAT
    elif normalized == "UNKNOWN":
        outcome = OutcomeClass.ABSTENTION
    elif not normalized or "\n" in normalized or "\r" in normalized or not normalized.isprintable():
        outcome = OutcomeClass.INVALID_FORMAT
    elif fields["target_code"] is not None and normalized == fields["target_code"]:
        outcome = OutcomeClass.EXACT_TARGET_CODE
    elif fields["distractor_code"] is not None and normalized == fields["distractor_code"]:
        outcome = OutcomeClass.DISTRACTOR_CODE_COPY
    elif normalized in vocabulary:
        outcome = OutcomeClass.NOVEL_CODE_ASSERTION
    else:
        outcome = OutcomeClass.OTHER_NON_ABSTENTION
    return ScoredResponse(
        **fields["row"],
        raw_output=text,
        normalized_output=normalized,
        outcome=outcome,
        answer_attempt=int(outcome is not OutcomeClass.ABSTENTION),
        valid_format=outcome is not OutcomeClass.INVALID_FORMAT,
        completed=completed,
        infrastructure_marked=marked,
        truncated=truncated,
    )


def estimate_behavior(rows: Sequence[ScoredResponse]) -> BehavioralMetrics:
    """Calculate frozen ITT estimands without excluding invalid outcomes."""
    source = _scored_rows(rows)
    return _estimate(source, include_sensitivities=True)


def crossed_bootstrap(
    rows: Sequence[ScoredResponse], replicates: int, seed: int
) -> BootstrapDistribution:
    """Cross-resample entity units and template families, never individual rows."""
    source = _scored_rows(rows)
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    observed = _estimate(source, include_sensitivities=False, require_completion=False)
    if observed.interaction is None or observed.h2_accuracy_difference is None:
        raise ValueError("bootstrap requires every registered factorial cell")
    rng = np.random.default_rng(seed)
    interaction_samples: list[float] = []
    h2_samples: list[float] = []
    h2b_samples: list[float] = []
    weighted_denominators: tuple[int, ...] | None = None
    for _ in range(replicates):
        sampled = cross_resample(source, rng)
        metrics = _estimate(sampled, include_sensitivities=False, require_completion=False)
        if metrics.interaction is None or metrics.h2_accuracy_difference is None:
            raise ValueError("bootstrap sample lacks a registered factorial cell")
        interaction_samples.append(metrics.interaction)
        h2_samples.append(metrics.h2_accuracy_difference)
        if metrics.h2b_interaction is not None:
            h2b_samples.append(metrics.h2b_interaction)
        if weighted_denominators is None:
            weighted_denominators = tuple(metrics.denominators.values())
    return BootstrapDistribution(
        interaction_samples=tuple(interaction_samples),
        h2_accuracy_difference_samples=tuple(h2_samples),
        h2b_interaction_samples=tuple(h2b_samples),
        interaction_interval=_interval(interaction_samples, observed.interaction),
        h2_accuracy_difference_interval=_interval(h2_samples, observed.h2_accuracy_difference),
        h2b_interaction_interval=(
            _interval(h2b_samples, observed.h2b_interaction)
            if h2b_samples and observed.h2b_interaction is not None
            else None
        ),
        weighted_denominators=weighted_denominators or (),
        seed=seed,
    )


def cross_resample(rows: Sequence[ScoredResponse], rng: np.random.Generator) -> tuple[ScoredResponse, ...]:
    """Apply product multiplicities from independent entity and template draws."""
    source = _scored_rows(rows)
    entities = tuple(sorted({row.entity_unit_id for row in source}))
    templates = tuple(sorted({row.template_family for row in source}))
    entity_counts = Counter(entities[int(index)] for index in rng.integers(0, len(entities), len(entities)))
    template_counts = Counter(templates[int(index)] for index in rng.integers(0, len(templates), len(templates)))
    sampled = []
    for row in source:
        multiplicity = entity_counts[row.entity_unit_id] * template_counts[row.template_family]
        if multiplicity:
            sampled.append(replace(row, sampling_weight=row.sampling_weight * multiplicity))
    return tuple(sampled)


def behavioral_gate(
    metrics: BehavioralMetrics,
    bootstrap: BootstrapDistribution,
    *,
    thresholds: Mapping[str, float],
    same_string_sealed: bool,
    config_hash: str,
    manifest_hash: str,
) -> BehavioralGate:
    """Apply registered H1/H2 gates while reporting H2b independently."""
    if not isinstance(thresholds, Mapping):
        raise ValueError("thresholds must be a mapping")
    if dict(thresholds) != CONFIRMATORY_THRESHOLDS:
        raise ValueError("thresholds must match registered thresholds")
    if type(same_string_sealed) is not bool:
        raise ValueError("same_string_sealed must be boolean")
    _lowercase_sha256(config_hash, "config_hash")
    _lowercase_sha256(manifest_hash, "manifest_hash")
    h1_min = _threshold(thresholds, "h1_min_interaction")
    h2_margin = _threshold(thresholds, "h2_noninferiority_margin")
    format_min = _threshold(thresholds, "format_validity_min")
    h2 = _h2_decision(metrics, bootstrap, h2_margin)
    h1 = _h1_decision(metrics, bootstrap, h1_min, format_min, h2)
    h2b = _h2b_decision(metrics, bootstrap, h1_min, same_string_sealed)
    status = h1.status
    return BehavioralGate(
        status=status,
        h1=h1,
        h2=h2,
        h2b=h2b,
        thresholds=thresholds,
        same_string_sealed=same_string_sealed,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
    )


def _estimate(
    rows: tuple[ScoredResponse, ...],
    *,
    include_sensitivities: bool,
    require_completion: bool = True,
) -> BehavioralMetrics:
    factorial = tuple(row for row in rows if row.block == "factorial")
    primary_cells = tuple(
        (target, distractor, answerability)
        for target in _PRIMARY_TARGETS
        for distractor in _PRIMARY_TARGETS
        for answerability in _PRIMARY_ANSWERABILITY
    )
    rates, completion, format_validity, denominators, invalid = _cell_summaries(factorial, primary_cells)
    missing = [cell for cell in primary_cells if denominators[cell] == 0]
    reasons = [f"missing_cell:{cell[0]}:{cell[1]}:{cell[2]}" for cell in missing]
    if require_completion:
        reasons.extend(
            f"completion<0.95:{target}:{distractor}:{answerability}"
            for target, distractor, answerability in primary_cells
            if denominators[(target, distractor, answerability)]
            and completion[(target, distractor, answerability)] < 0.95
        )
    interaction = None if missing else _h1_interaction(rates)
    # Keep primary attempt rates separately from exact-target accuracy in one internal map.
    attempt_rates = {cell: rates[(*cell, "attempt")] for cell in primary_cells}
    h2_difference = None if missing else (
        _equal_distractor_average(rates, "matched_synthetic", "target_bound", "exact_target")
        - _equal_distractor_average(rates, "screened_real", "target_bound", "exact_target")
    )
    h2b, _ = _same_string_interaction(rows)
    sensitivities: dict[str, SensitivityResult] = {}
    if include_sensitivities:
        sensitivities = _sensitivities(rows, denominators, invalid)
    return BehavioralMetrics(
        status="not_evaluable" if reasons else "evaluable",
        reasons=tuple(reasons),
        cell_rates=attempt_rates,
        completion_by_cell=completion,
        format_validity_by_cell=format_validity,
        denominators=denominators,
        invalid_format_counts=invalid,
        interaction=interaction,
        h2_accuracy_difference=h2_difference,
        h2b_interaction=h2b,
        sensitivities=sensitivities,
    )


def _cell_summaries(
    rows: Sequence[ScoredResponse], cells: Sequence[Cell]
) -> tuple[dict[tuple[str, str, str], float], dict[Cell, float], dict[Cell, float], dict[Cell, int], dict[Cell, int]]:
    totals = defaultdict(int)
    attempts = defaultdict(int)
    exact_targets = defaultdict(int)
    completions = defaultdict(int)
    valid_formats = defaultdict(int)
    invalid = defaultdict(int)
    for row in rows:
        cell = (row.target_familiarity, row.distractor_familiarity, row.answerability)
        if cell not in cells:
            continue
        weight = row.sampling_weight
        totals[cell] += weight
        attempts[cell] += weight * row.answer_attempt
        exact_targets[cell] += weight * int(row.outcome is OutcomeClass.EXACT_TARGET_CODE)
        completions[cell] += weight * int(row.completed)
        valid_formats[cell] += weight * int(row.valid_format)
        invalid[cell] += weight * int(row.outcome is OutcomeClass.INVALID_FORMAT)
    rates = {}
    completion = {}
    format_validity = {}
    denominators = {}
    invalid_counts = {}
    for cell in cells:
        denominator = totals[cell]
        denominators[cell] = denominator
        invalid_counts[cell] = invalid[cell]
        if denominator:
            rates[(*cell, "attempt")] = attempts[cell] / denominator
            rates[(*cell, "exact_target")] = exact_targets[cell] / denominator
            completion[cell] = completions[cell] / denominator
            format_validity[cell] = valid_formats[cell] / denominator
        else:
            rates[(*cell, "attempt")] = float("nan")
            rates[(*cell, "exact_target")] = float("nan")
            completion[cell] = float("nan")
            format_validity[cell] = float("nan")
    return rates, completion, format_validity, denominators, invalid_counts


def _h1_interaction(rates: Mapping[tuple[str, str, str, str], float]) -> float:
    absent_real = 0.5 * (
        _equal_distractor_average(rates, "screened_real", "distractor_bound", "attempt")
        + _equal_distractor_average(rates, "screened_real", "code_absent", "attempt")
    )
    absent_synthetic = 0.5 * (
        _equal_distractor_average(rates, "matched_synthetic", "distractor_bound", "attempt")
        + _equal_distractor_average(rates, "matched_synthetic", "code_absent", "attempt")
    )
    return (absent_real - absent_synthetic) - (
        _equal_distractor_average(rates, "screened_real", "target_bound", "attempt")
        - _equal_distractor_average(rates, "matched_synthetic", "target_bound", "attempt")
    )


def _equal_distractor_average(
    rates: Mapping[tuple[str, str, str, str], float],
    target: str,
    answerability: str,
    outcome: str,
) -> float:
    return 0.5 * sum(rates[(target, distractor, answerability, outcome)] for distractor in _PRIMARY_TARGETS)


def _same_string_interaction(rows: Sequence[ScoredResponse]) -> tuple[float | None, tuple[str, ...]]:
    same_string = tuple(row for row in rows if row.block == "same_string")
    cells = tuple((exposure, answerability) for exposure in _H2B_EXPOSURES for answerability in _H2B_ANSWERABILITY)
    totals = defaultdict(int)
    attempts = defaultdict(int)
    for row in same_string:
        cell = (row.exposure, row.answerability)
        if cell in cells:
            totals[cell] += row.sampling_weight
            attempts[cell] += row.sampling_weight * row.answer_attempt
    if not same_string:
        return None, ("no_same_string_rows",)
    if any(totals[cell] == 0 for cell in cells):
        return None, ("incomplete_same_string_cells",)
    rates = {cell: attempts[cell] / totals[cell] for cell in cells}
    return (
        (rates[("high_exposure", "code_absent")] - rates[("low_exposure", "code_absent")])
        - (rates[("high_exposure", "target_bound")] - rates[("low_exposure", "target_bound")]),
        (),
    )


def _sensitivities(
    rows: tuple[ScoredResponse, ...],
    original_denominators: Mapping[Cell, int],
    invalid: Mapping[Cell, int],
) -> dict[str, SensitivityResult]:
    primary = _estimate(rows, include_sensitivities=False, require_completion=False)
    retained = tuple(row for row in rows if row.outcome is not OutcomeClass.INVALID_FORMAT)
    complete_case = _estimate(retained, include_sensitivities=False, require_completion=False)
    return {
        "intention_to_treat": SensitivityResult(
            "intention_to_treat", primary.interaction, primary.denominators, original_denominators, invalid
        ),
        "complete_case_excluding_invalid": SensitivityResult(
            "complete_case_excluding_invalid",
            complete_case.interaction,
            complete_case.denominators,
            original_denominators,
            invalid,
        ),
    }


def _h1_decision(
    metrics: BehavioralMetrics,
    bootstrap: BootstrapDistribution,
    minimum: float,
    format_minimum: float,
    h2: GateDecision,
) -> GateDecision:
    if metrics.status != "evaluable":
        return GateDecision("not_evaluable", metrics.reasons)
    interval = bootstrap.interaction_interval
    if metrics.interaction is None or interval is None:
        return GateDecision("not_evaluable", ("missing_h1_bootstrap",))
    reasons = []
    if interval.lower <= 0.0:
        reasons.append("h1_lower_bound_not_positive")
    if metrics.interaction < minimum:
        reasons.append("h1_point_estimate_below_minimum")
    if any(value < format_minimum for value in metrics.format_validity_by_cell.values()):
        reasons.append("format_validity_below_minimum")
    if h2.status != "supported":
        reasons.append("h2_not_supported")
    return GateDecision("supported" if not reasons else "not_supported", tuple(reasons))


def _h2_decision(metrics: BehavioralMetrics, bootstrap: BootstrapDistribution, margin: float) -> GateDecision:
    if metrics.status != "evaluable":
        return GateDecision("not_evaluable", metrics.reasons)
    interval = bootstrap.h2_accuracy_difference_interval
    if metrics.h2_accuracy_difference is None or interval is None:
        return GateDecision("not_evaluable", ("missing_h2_bootstrap",))
    if interval.lower > -margin:
        return GateDecision("supported", ())
    return GateDecision("not_supported", ("h2_noninferiority_lower_bound",))


def _h2b_decision(
    metrics: BehavioralMetrics,
    bootstrap: BootstrapDistribution,
    minimum: float,
    same_string_sealed: bool,
) -> GateDecision:
    if not same_string_sealed:
        return GateDecision("not_evaluable", ("same_string_prefix_not_sealed",))
    interval = bootstrap.h2b_interaction_interval
    if metrics.h2b_interaction is None or interval is None:
        return GateDecision("not_evaluable", ("missing_h2b_bootstrap",))
    reasons = []
    if interval.lower <= 0.0:
        reasons.append("h2b_interval_not_predicted_direction")
    if metrics.h2b_interaction < minimum:
        reasons.append("h2b_point_estimate_below_minimum")
    return GateDecision("supported" if not reasons else "not_supported", tuple(reasons))


def _example_fields(example: Any) -> dict[str, Any]:
    required = (
        "example_id",
        "entity_unit_id",
        "template_family",
        "target_familiarity",
        "distractor_familiarity",
        "answerability",
        "registry_code",
    )
    row = {field: _field(example, field) for field in required}
    block = getattr(example, "block", "factorial")
    exposure = getattr(example, "exposure", None)
    answerability = row["answerability"]
    target_code = row["registry_code"] if answerability == "target_bound" else None
    distractor_code = row["registry_code"] if answerability == "distractor_bound" else None
    return {
        "row": {**row, "block": block, "exposure": exposure},
        "target_code": target_code,
        "distractor_code": distractor_code,
    }


def _registered_code_vocabulary(codes: Collection[str]) -> frozenset[str]:
    if isinstance(codes, str) or not isinstance(codes, Collection):
        raise ValueError("registered_codes must be a collection of codes")
    vocabulary = frozenset(_nonempty_text(code, "registered code") for code in codes)
    if not vocabulary:
        raise ValueError("registered_codes must not be empty")
    return vocabulary


def _scored_rows(rows: Sequence[ScoredResponse]) -> tuple[ScoredResponse, ...]:
    values = tuple(rows)
    if not values:
        raise ValueError("at least one scored response is required")
    if any(not isinstance(row, ScoredResponse) for row in values):
        raise ValueError("rows must be ScoredResponse values")
    return values


def _interval(samples: Sequence[float], estimate: float) -> PercentileInterval:
    values = np.asarray(samples, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)) or not np.isfinite(estimate):
        raise ValueError("bootstrap samples must be finite and nonempty")
    lower, upper = np.percentile(values, (2.5, 97.5))
    return PercentileInterval(float(estimate), float(lower), float(upper))


def _threshold(thresholds: Mapping[str, float], name: str) -> float:
    value = thresholds.get(name)
    if type(value) not in {int, float} or not np.isfinite(value):
        raise ValueError(f"threshold {name} must be finite")
    return float(value)


def _field(value: Any, name: str) -> str:
    return _nonempty_text(getattr(value, name, None), name)


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _contains_infrastructure_marker(value: str) -> bool:
    normalized = value.casefold()
    return any(marker in normalized for marker in _INFRASTRUCTURE_MARKERS)


def _lowercase_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _LOWERCASE_SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _freeze_cells(values: Mapping[Cell, Any]) -> Mapping[Cell, Any]:
    frozen = {}
    for cell, value in values.items():
        if not isinstance(cell, tuple) or len(cell) != 3 or not all(isinstance(item, str) for item in cell):
            raise ValueError("cell keys must be three text labels")
        frozen[cell] = value
    return MappingProxyType(frozen)


def _cell_record(values: Mapping[Cell, Any]) -> dict[str, Any]:
    return {
        f"{target}:{distractor}:{answerability}": value
        for (target, distractor, answerability), value in sorted(values.items())
    }
