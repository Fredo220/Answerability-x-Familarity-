"""Sealed, fail-closed analysis for the Same-String causal pilot.

This module accepts typed score rows supplied by a runtime.  It never loads a
model or reads an outcome artifact, keeping the protected endpoint boundary at
the caller that creates those rows.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from trajectory_extractor.fa_answerability_causal import (
    CAUSAL_DIRECTION_ANCHOR,
    CAUSAL_EXPOSURES,
    CAUSAL_VALIDATION_LAYERS,
    CausalExpectedProvenance,
    ValidationSelection,
)


BOOTSTRAP_DRAWS = 10_000
PERMUTATION_DRAWS = 9_999
BOOTSTRAP_SEED = 202608041
PERMUTATION_SEED = 202608042
CAUSAL_TEST_SPLITS = ("causal_entity_test", "causal_template_test")
CAUSAL_CONTROLS = (
    "no_intervention",
    "sign_reversed",
    "label_shuffled_direction",
    "norm_matched_random",
    "wrong_anchor",
    "wrong_layer",
)
_ANSWERABILITY = ("target_unbound", "target_bound")
_RANDOM_CONTROL_MEMBERS = tuple(range(5))
_SHA256_LENGTH = 64


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise ValueError(f"{name} must be a SHA-256 hash")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hash") from error
    return value


def deterministic_farthest_layer(selected_layer: int) -> int:
    """Use distance, then the earlier registered layer, as the fixed tie break."""
    if selected_layer not in CAUSAL_VALIDATION_LAYERS:
        raise ValueError("selected layer is not registered")
    return min(CAUSAL_VALIDATION_LAYERS, key=lambda layer: (-abs(layer - selected_layer), layer))


class GenerationClass(StrEnum):
    CORRECT_CODE = "correct_code"
    UNKNOWN = "unknown"
    OTHER_CODE = "other_code"
    INVALID = "invalid"


@dataclass(frozen=True)
class GenerationResult:
    response_class: GenerationClass
    format_valid: bool
    copied_from_other_unit: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.response_class, GenerationClass):
            raise ValueError("generation response class is invalid")
        if self.response_class == GenerationClass.INVALID and self.format_valid:
            raise ValueError("invalid generations cannot pass the format gate")


@dataclass(frozen=True)
class BaselineScore:
    unit_id: str
    split: str
    exposure: str
    answerability: str
    raw_margin: float
    length_normalized_margin: float
    generation: GenerationResult

    def __post_init__(self) -> None:
        _validate_score_fields(self)


@dataclass(frozen=True)
class ExecutionAuditHashes:
    """Identity that must match the pre-outcome seal for every score row."""

    corpus_sha256: str
    selection_sha256: str
    direction_bundle_sha256: str
    direction_sha256: str
    model_sha256: str
    tokenizer_sha256: str
    runtime_sha256: str
    output_contract_sha256: str
    layer_id: int
    multiplier: float
    anchor: str
    random_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.corpus_sha256, "corpus hash"),
            (self.selection_sha256, "selection hash"),
            (self.direction_bundle_sha256, "direction bundle hash"),
            (self.direction_sha256, "direction hash"),
            (self.model_sha256, "model hash"),
            (self.tokenizer_sha256, "tokenizer hash"),
            (self.runtime_sha256, "runtime hash"),
            (self.output_contract_sha256, "output contract hash"),
        ):
            _require_sha256(value, name)
        if self.layer_id not in CAUSAL_VALIDATION_LAYERS:
            raise ValueError("execution layer is not registered")
        if not np.isfinite(self.multiplier) or self.multiplier <= 0.0:
            raise ValueError("execution multiplier must be positive and finite")
        if not isinstance(self.anchor, str) or not self.anchor:
            raise ValueError("execution anchor must be nonempty")
        seeds = tuple(self.random_seeds)
        if len(seeds) != 5 or any(type(seed) is not int for seed in seeds):
            raise ValueError("execution random seeds must contain five integers")
        object.__setattr__(self, "random_seeds", seeds)

    @classmethod
    def for_control(
        cls,
        seal: "CausalEvaluationSeal",
        control: str,
        *,
        member: int | None = None,
    ) -> "ExecutionAuditHashes":
        if control not in ("primary", *CAUSAL_CONTROLS):
            raise ValueError("control is not registered")
        if control == "norm_matched_random":
            if member not in _RANDOM_CONTROL_MEMBERS:
                raise ValueError("random control member is not registered")
        elif member is not None:
            raise ValueError("only random controls may have a member")
        direction = seal.direction_sha256
        if control == "label_shuffled_direction":
            direction = seal.control_direction_hashes[control]
        elif control == "norm_matched_random":
            direction = seal.control_direction_hashes[f"{control}:{member}"]
        layer = seal.layer_id
        anchor = seal.anchor
        if control == "wrong_layer":
            layer = deterministic_farthest_layer(seal.layer_id)
        elif control == "wrong_anchor":
            anchor = "target_intro_end"
        return cls(
            corpus_sha256=seal.corpus_sha256,
            selection_sha256=seal.selection_sha256,
            direction_bundle_sha256=seal.direction_bundle_sha256,
            direction_sha256=direction,
            model_sha256=seal.model_sha256,
            tokenizer_sha256=seal.tokenizer_sha256,
            runtime_sha256=seal.runtime_sha256,
            output_contract_sha256=seal.output_contract_sha256,
            layer_id=layer,
            multiplier=seal.multiplier,
            anchor=anchor,
            random_seeds=seal.random_seeds,
        )


@dataclass(frozen=True)
class _InterventionScore:
    unit_id: str
    split: str
    exposure: str
    answerability: str
    raw_margin: float
    length_normalized_margin: float
    generation: GenerationResult
    audit: ExecutionAuditHashes
    sign: int
    control_member: int | None = None

    def __post_init__(self) -> None:
        _validate_score_fields(self)
        if not isinstance(self.audit, ExecutionAuditHashes):
            raise ValueError("intervention score requires execution audit hashes")
        if self.sign not in {-1, 0, 1}:
            raise ValueError("intervention sign must be -1, 0, or 1")


@dataclass(frozen=True)
class PrimaryScore(_InterventionScore):
    @property
    def control(self) -> str:
        return "primary"


@dataclass(frozen=True)
class ControlScore(_InterventionScore):
    control: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.control not in CAUSAL_CONTROLS:
            raise ValueError("control score control is not registered")
        if self.control == "norm_matched_random":
            if self.control_member not in _RANDOM_CONTROL_MEMBERS:
                raise ValueError("random control requires a registered member")
        elif self.control_member is not None:
            raise ValueError("only random controls may have a member")


@dataclass(frozen=True)
class ManipulationCheck:
    unit_id: str
    split: str
    primary_projection_delta: float

    def __post_init__(self) -> None:
        if not self.unit_id or self.split not in CAUSAL_TEST_SPLITS:
            raise ValueError("manipulation check identity is invalid")
        if not np.isfinite(self.primary_projection_delta):
            raise ValueError("manipulation check must be finite")


@dataclass(frozen=True)
class PreservationResult:
    split: str
    bound_accuracy_drop: float
    unrelated_task_preserved: bool

    def __post_init__(self) -> None:
        if self.split not in CAUSAL_TEST_SPLITS:
            raise ValueError("preservation split is invalid")
        if not np.isfinite(self.bound_accuracy_drop):
            raise ValueError("preservation drop must be finite")

    @property
    def passed(self) -> bool:
        return self.bound_accuracy_drop <= 0.05 and self.unrelated_task_preserved


@dataclass(frozen=True)
class CausalEvaluationSeal:
    corpus_sha256: str
    selection_sha256: str
    direction_bundle_sha256: str
    direction_sha256: str
    model_sha256: str
    tokenizer_sha256: str
    runtime_sha256: str
    output_contract_sha256: str
    layer_id: int
    multiplier: float
    anchor: str
    random_seeds: tuple[int, ...]
    control_direction_hashes: Mapping[str, str]
    seal_sha256: str

    def __post_init__(self) -> None:
        base = ExecutionAuditHashes(
            corpus_sha256=self.corpus_sha256,
            selection_sha256=self.selection_sha256,
            direction_bundle_sha256=self.direction_bundle_sha256,
            direction_sha256=self.direction_sha256,
            model_sha256=self.model_sha256,
            tokenizer_sha256=self.tokenizer_sha256,
            runtime_sha256=self.runtime_sha256,
            output_contract_sha256=self.output_contract_sha256,
            layer_id=self.layer_id,
            multiplier=self.multiplier,
            anchor=self.anchor,
            random_seeds=self.random_seeds,
        )
        controls = dict(self.control_direction_hashes)
        expected_controls = {"label_shuffled_direction", *(
            f"norm_matched_random:{member}" for member in _RANDOM_CONTROL_MEMBERS
        )}
        if set(controls) != expected_controls:
            raise ValueError("seal must bind every derived control direction")
        for value in controls.values():
            _require_sha256(value, "control direction hash")
        object.__setattr__(self, "random_seeds", base.random_seeds)
        object.__setattr__(self, "control_direction_hashes", MappingProxyType(controls))
        if self.seal_sha256 != _sha256(_seal_payload(self, include_hash=False)):
            raise ValueError("seal hash does not match seal content")


def seal_causal_evaluation(
    *,
    selection: ValidationSelection,
    expected_provenance: CausalExpectedProvenance,
    runtime_sha256: str,
    output_contract_sha256: str,
    random_seeds: Sequence[int],
) -> CausalEvaluationSeal:
    """Bind selection and execution identity before causal-test score rows exist."""
    if not isinstance(selection, ValidationSelection):
        raise ValueError("sealing requires a typed validation selection")
    if not isinstance(expected_provenance, CausalExpectedProvenance):
        raise ValueError("sealing requires typed expected provenance")
    if (
        selection.corpus_sha256 != expected_provenance.corpus_sha256
        or selection.direction_bundle_sha256 != expected_provenance.direction_bundle_sha256
        or selection.model_sha256 != expected_provenance.model_sha256
        or selection.tokenizer_sha256 != expected_provenance.tokenizer_sha256
        or selection.direction_sha256 != expected_provenance.direction_hashes[selection.layer_id]
    ):
        raise ValueError("selection does not match expected provenance")
    _require_sha256(runtime_sha256, "runtime hash")
    _require_sha256(output_contract_sha256, "output contract hash")
    seeds = tuple(random_seeds)
    if len(seeds) != 5 or any(type(seed) is not int for seed in seeds):
        raise ValueError("random seeds must contain five integers")
    base = {
        "selection_sha256": selection.selection_sha256,
        "direction_sha256": selection.direction_sha256,
        "random_seeds": seeds,
    }
    controls = {"label_shuffled_direction": _sha256({**base, "control": "label_shuffled_direction"})}
    controls.update(
        {
            f"norm_matched_random:{member}": _sha256(
                {**base, "control": "norm_matched_random", "member": member, "seed": seeds[member]}
            )
            for member in _RANDOM_CONTROL_MEMBERS
        }
    )
    record = {
        "corpus_sha256": selection.corpus_sha256,
        "selection_sha256": selection.selection_sha256,
        "direction_bundle_sha256": selection.direction_bundle_sha256,
        "direction_sha256": selection.direction_sha256,
        "model_sha256": selection.model_sha256,
        "tokenizer_sha256": selection.tokenizer_sha256,
        "runtime_sha256": runtime_sha256,
        "output_contract_sha256": output_contract_sha256,
        "layer_id": selection.layer_id,
        "multiplier": selection.multiplier,
        "anchor": CAUSAL_DIRECTION_ANCHOR,
        "random_seeds": seeds,
        "control_direction_hashes": controls,
    }
    return CausalEvaluationSeal(seal_sha256=_sha256(record), **record)


@dataclass(frozen=True)
class CausalEvidence:
    split: str
    seal: CausalEvaluationSeal
    baselines: tuple[BaselineScore, ...]
    primary_scores: tuple[PrimaryScore, ...]
    control_scores: tuple[ControlScore, ...]
    manipulation_checks: tuple[ManipulationCheck, ...]
    preservation: PreservationResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "baselines", tuple(self.baselines))
        object.__setattr__(self, "primary_scores", tuple(self.primary_scores))
        object.__setattr__(self, "control_scores", tuple(self.control_scores))
        object.__setattr__(self, "manipulation_checks", tuple(self.manipulation_checks))


@dataclass(frozen=True)
class BootstrapSummary:
    mean_effect: float
    lower_95: float
    upper_95: float
    draws: int


@dataclass(frozen=True)
class SignFlipSummary:
    p_value: float
    draws: int


@dataclass(frozen=True)
class ControlEffect:
    mean_effect: float
    member_count: int


@dataclass(frozen=True)
class CausalDecision:
    status: Literal["causally_supported", "not_supported", "not_evaluable"]
    reasons: tuple[str, ...]
    mean_effect: float | None = None
    unbound_effect: float | None = None
    bound_effect: float | None = None
    bootstrap: BootstrapSummary | None = None
    sign_flip: SignFlipSummary | None = None
    control_effects: Mapping[str, ControlEffect] = field(default_factory=dict)
    strongest_control: str | None = None
    contrast_bootstrap: BootstrapSummary | None = None
    length_normalized_sensitivity: BootstrapSummary | None = None

    def __post_init__(self) -> None:
        if self.status not in {"causally_supported", "not_supported", "not_evaluable"}:
            raise ValueError("causal decision status is invalid")
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))
        object.__setattr__(self, "control_effects", MappingProxyType(dict(self.control_effects)))


@dataclass(frozen=True)
class CausalStudyDecision:
    """Unpooled endpoint decision; support requires both registered test splits."""

    status: Literal["causally_supported", "not_supported", "not_evaluable"]
    reasons: tuple[str, ...]
    split_decisions: Mapping[str, CausalDecision] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"causally_supported", "not_supported", "not_evaluable"}:
            raise ValueError("causal study status is invalid")
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))
        object.__setattr__(self, "split_decisions", MappingProxyType(dict(self.split_decisions)))


class _NotEvaluable(ValueError):
    pass


def analyze_causal_evidence(evidence: CausalEvidence) -> CausalDecision:
    """Apply the frozen support rule to one unpooled causal-test split."""
    try:
        prepared = _validate_evidence(evidence)
    except _NotEvaluable as error:
        return CausalDecision(status="not_evaluable", reasons=(str(error),))

    primary_effects, primary_unbound, primary_bound, normalized = _unit_effects(
        prepared["baselines"], prepared["primary"], prepared["units"]
    )
    bootstrap = _bootstrap(primary_effects, BOOTSTRAP_SEED)
    sign_flip = _sign_flip(primary_effects)
    normalized_summary = _bootstrap(normalized, BOOTSTRAP_SEED)
    control_values = {
        control: _control_unit_effects(prepared, control) for control in CAUSAL_CONTROLS
    }
    control_effects = {
        name: ControlEffect(mean_effect=float(np.mean(values)), member_count=(5 if name == "norm_matched_random" else 1))
        for name, values in control_values.items()
    }
    means = {name: value.mean_effect for name, value in control_effects.items()}
    largest = max(means.values())
    tied = tuple(name for name, value in means.items() if np.isclose(value, largest, rtol=0.0, atol=1e-12))
    strongest_control = tied[0] if len(tied) == 1 else None
    contrast = None
    if strongest_control is not None:
        contrast = _bootstrap(primary_effects - control_values[strongest_control], BOOTSTRAP_SEED)

    reasons: list[str] = []
    if bootstrap.mean_effect <= 0.0:
        reasons.append("primary_mean_not_positive")
    if bootstrap.lower_95 <= 0.0:
        reasons.append("primary_bootstrap_lower_not_positive")
    if primary_unbound <= 0.0 or primary_bound <= 0.0:
        reasons.append("primary_component_not_positive")
    if sign_flip.p_value > 0.05:
        reasons.append("sign_flip_p_above_0_05")
    if strongest_control is None:
        reasons.append("strongest_control_tie")
    elif contrast is None or contrast.lower_95 <= 0.0:
        reasons.append("control_contrast_lower_not_positive")
    generations = (*evidence.baselines, *evidence.primary_scores, *evidence.control_scores)
    if any(row.generation.copied_from_other_unit for row in generations):
        reasons.append("cross_unit_code_copying")
    if any(not row.generation.format_valid for row in generations):
        reasons.append("format_gate_failed")
    if not evidence.preservation.passed:
        reasons.append("preservation_failed")

    return CausalDecision(
        status="causally_supported" if not reasons else "not_supported",
        reasons=tuple(reasons),
        mean_effect=bootstrap.mean_effect,
        unbound_effect=primary_unbound,
        bound_effect=primary_bound,
        bootstrap=bootstrap,
        sign_flip=sign_flip,
        control_effects=control_effects,
        strongest_control=strongest_control,
        contrast_bootstrap=contrast,
        length_normalized_sensitivity=normalized_summary,
    )


def analyze_causal_study(
    evidence_by_split: Mapping[str, CausalEvidence],
) -> CausalStudyDecision:
    """Apply the support rule across both fixed test splits without pooling them."""
    supplied = dict(evidence_by_split)
    if set(supplied) != set(CAUSAL_TEST_SPLITS):
        return CausalStudyDecision(status="not_evaluable", reasons=("both_test_splits_required",))
    if any(evidence.split != split for split, evidence in supplied.items()):
        return CausalStudyDecision(status="not_evaluable", reasons=("split_key_mismatch",))
    decisions = {split: analyze_causal_evidence(supplied[split]) for split in CAUSAL_TEST_SPLITS}
    if any(decision.status == "not_evaluable" for decision in decisions.values()):
        return CausalStudyDecision(
            status="not_evaluable",
            reasons=tuple(
                f"{split}:{reason}"
                for split, decision in decisions.items()
                for reason in decision.reasons
            ),
            split_decisions=decisions,
        )
    if all(decision.status == "causally_supported" for decision in decisions.values()):
        return CausalStudyDecision(
            status="causally_supported", reasons=(), split_decisions=decisions
        )
    return CausalStudyDecision(
        status="not_supported",
        reasons=tuple(
            f"{split}:{reason}"
            for split, decision in decisions.items()
            for reason in decision.reasons
        ),
        split_decisions=decisions,
    )


def _validate_score_fields(row: Any) -> None:
    if not isinstance(row.unit_id, str) or not row.unit_id:
        raise ValueError("score unit ID must be nonempty")
    if row.split not in CAUSAL_TEST_SPLITS:
        raise ValueError("score split is invalid")
    if row.exposure not in CAUSAL_EXPOSURES or row.answerability not in _ANSWERABILITY:
        raise ValueError("score factorial cell is invalid")
    if not np.isfinite(row.raw_margin) or not np.isfinite(row.length_normalized_margin):
        raise ValueError("score margins must be finite")
    if not isinstance(row.generation, GenerationResult):
        raise ValueError("score generation is invalid")


def _validate_evidence(evidence: CausalEvidence) -> dict[str, Any]:
    if not isinstance(evidence, CausalEvidence) or evidence.split not in CAUSAL_TEST_SPLITS:
        raise _NotEvaluable("test_split_required")
    if not isinstance(evidence.seal, CausalEvaluationSeal):
        raise _NotEvaluable("execution_identity_mismatch")
    all_rows = (*evidence.baselines, *evidence.primary_scores, *evidence.control_scores)
    if any(row.split != evidence.split for row in all_rows):
        raise _NotEvaluable("test_split_required")
    if evidence.preservation.split != evidence.split:
        raise _NotEvaluable("test_split_required")
    baselines = _index_rows(evidence.baselines, "incomplete_2x2_units")
    units = sorted({row.unit_id for row in evidence.baselines})
    expected_cells = {(unit, exposure, answerability) for unit in units for exposure in CAUSAL_EXPOSURES for answerability in _ANSWERABILITY}
    if set(baselines) != expected_cells:
        raise _NotEvaluable("incomplete_2x2_units")
    primary = _index_rows(evidence.primary_scores, "incomplete_2x2_units")
    if set(primary) != expected_cells:
        raise _NotEvaluable("incomplete_2x2_units")
    if any(not _audit_matches(row.audit, evidence.seal, "primary", row.control_member) for row in evidence.primary_scores):
        raise _NotEvaluable("execution_identity_mismatch")
    if any(row.sign != _expected_sign("primary", row.answerability) for row in evidence.primary_scores):
        raise _NotEvaluable("intervention_sign_mismatch")
    checks = {row.unit_id for row in evidence.manipulation_checks if row.split == evidence.split}
    if checks != set(units) or len(checks) != len(evidence.manipulation_checks):
        raise _NotEvaluable("incomplete_manipulation_checks")
    by_control: dict[str, list[ControlScore]] = defaultdict(list)
    for row in evidence.control_scores:
        by_control[row.control].append(row)
    if set(by_control) != set(CAUSAL_CONTROLS):
        raise _NotEvaluable("incomplete_control_schedule")
    indexed_controls: dict[str, Any] = {}
    for control, rows in by_control.items():
        if any(not _audit_matches(row.audit, evidence.seal, control, row.control_member) for row in rows):
            raise _NotEvaluable("execution_identity_mismatch")
        if any(row.sign != _expected_sign(control, row.answerability) for row in rows):
            raise _NotEvaluable("intervention_sign_mismatch")
        if control == "norm_matched_random":
            members: dict[int, dict[tuple[str, str, str], ControlScore]] = {}
            for member in _RANDOM_CONTROL_MEMBERS:
                member_rows = [row for row in rows if row.control_member == member]
                index = _index_rows(member_rows, "incomplete_control_schedule")
                if set(index) != expected_cells:
                    raise _NotEvaluable("incomplete_control_schedule")
                members[member] = index
            if len(rows) != len(expected_cells) * len(_RANDOM_CONTROL_MEMBERS):
                raise _NotEvaluable("incomplete_control_schedule")
            indexed_controls[control] = members
        else:
            index = _index_rows(rows, "incomplete_control_schedule")
            if set(index) != expected_cells:
                raise _NotEvaluable("incomplete_control_schedule")
            indexed_controls[control] = index
            if control == "no_intervention" and any(
                not np.isclose(row.raw_margin, baselines[key].raw_margin, rtol=0.0, atol=1e-12)
                or not np.isclose(
                    row.length_normalized_margin,
                    baselines[key].length_normalized_margin,
                    rtol=0.0,
                    atol=1e-12,
                )
                for key, row in index.items()
            ):
                raise _NotEvaluable("no_intervention_mismatch")
    return {"baselines": baselines, "primary": primary, "controls": indexed_controls, "units": units}


def _index_rows(rows: Sequence[Any], reason: str) -> dict[tuple[str, str, str], Any]:
    index = {(row.unit_id, row.exposure, row.answerability): row for row in rows}
    if len(index) != len(rows):
        raise _NotEvaluable(reason)
    return index


def _expected_sign(control: str, answerability: str) -> int:
    if control == "no_intervention":
        return 0
    sign = 1 if answerability == "target_unbound" else -1
    return -sign if control == "sign_reversed" else sign


def _audit_matches(
    audit: ExecutionAuditHashes,
    seal: CausalEvaluationSeal,
    control: str,
    member: int | None,
) -> bool:
    try:
        return audit == ExecutionAuditHashes.for_control(seal, control, member=member)
    except ValueError:
        return False


def _unit_effects(
    baselines: Mapping[tuple[str, str, str], Any],
    interventions: Mapping[tuple[str, str, str], Any],
    units: Sequence[str],
) -> tuple[np.ndarray, float, float, np.ndarray]:
    effects = []
    normalized = []
    unbound_effects = []
    bound_effects = []
    for unit in units:
        unbound = []
        bound = []
        normalized_unbound = []
        normalized_bound = []
        for exposure in CAUSAL_EXPOSURES:
            unbound_key = (unit, exposure, "target_unbound")
            bound_key = (unit, exposure, "target_bound")
            unbound.append(interventions[unbound_key].raw_margin - baselines[unbound_key].raw_margin)
            bound.append(baselines[bound_key].raw_margin - interventions[bound_key].raw_margin)
            normalized_unbound.append(
                interventions[unbound_key].length_normalized_margin
                - baselines[unbound_key].length_normalized_margin
            )
            normalized_bound.append(
                baselines[bound_key].length_normalized_margin
                - interventions[bound_key].length_normalized_margin
            )
        unbound_effects.append(float(np.mean(unbound)))
        bound_effects.append(float(np.mean(bound)))
        effects.append(0.5 * (unbound_effects[-1] + bound_effects[-1]))
        normalized.append(0.5 * (float(np.mean(normalized_unbound)) + float(np.mean(normalized_bound))))
    return np.asarray(effects), float(np.mean(unbound_effects)), float(np.mean(bound_effects)), np.asarray(normalized)


def _control_unit_effects(prepared: Mapping[str, Any], control: str) -> np.ndarray:
    if control == "norm_matched_random":
        family = [
            _unit_effects(prepared["baselines"], values, prepared["units"])[0]
            for values in prepared["controls"][control].values()
        ]
        return np.mean(np.stack(family), axis=0)
    return _unit_effects(prepared["baselines"], prepared["controls"][control], prepared["units"])[0]


def _bootstrap(values: np.ndarray, seed: int) -> BootstrapSummary:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    draws = values[indices].mean(axis=1)
    return BootstrapSummary(
        mean_effect=float(np.mean(values)),
        lower_95=float(np.quantile(draws, 0.025)),
        upper_95=float(np.quantile(draws, 0.975)),
        draws=BOOTSTRAP_DRAWS,
    )


def _sign_flip(values: np.ndarray) -> SignFlipSummary:
    generator = np.random.default_rng(PERMUTATION_SEED)
    signs = generator.choice(np.array([-1.0, 1.0]), size=(PERMUTATION_DRAWS, len(values)))
    observed = float(np.mean(values))
    null_means = (signs * values).mean(axis=1)
    p_value = float((1 + np.count_nonzero(null_means >= observed)) / (PERMUTATION_DRAWS + 1))
    return SignFlipSummary(p_value=p_value, draws=PERMUTATION_DRAWS)


def _seal_payload(seal: CausalEvaluationSeal, *, include_hash: bool) -> dict[str, Any]:
    payload = {
        "corpus_sha256": seal.corpus_sha256,
        "selection_sha256": seal.selection_sha256,
        "direction_bundle_sha256": seal.direction_bundle_sha256,
        "direction_sha256": seal.direction_sha256,
        "model_sha256": seal.model_sha256,
        "tokenizer_sha256": seal.tokenizer_sha256,
        "runtime_sha256": seal.runtime_sha256,
        "output_contract_sha256": seal.output_contract_sha256,
        "layer_id": seal.layer_id,
        "multiplier": seal.multiplier,
        "anchor": seal.anchor,
        "random_seeds": seal.random_seeds,
        "control_direction_hashes": dict(seal.control_direction_hashes),
    }
    if include_hash:
        payload["seal_sha256"] = seal.seal_sha256
    return payload


class CausalAnalysisStore:
    """Hash-bound one-use receipt store; partial same-seal resumes are allowed."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def begin_or_resume(self, seal: CausalEvaluationSeal, split: str) -> Mapping[str, str]:
        if split not in CAUSAL_TEST_SPLITS:
            raise ValueError("causal analysis requires one test split")
        path = self._path(split)
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state["seal_sha256"] != seal.seal_sha256:
                raise ValueError("resume seal hash does not match")
            if state["status"] == "completed":
                raise ValueError("completed endpoint cannot be reopened")
            return MappingProxyType(dict(state))
        state = {"split": split, "seal_sha256": seal.seal_sha256, "status": "in_progress"}
        self._write(path, state)
        return MappingProxyType(state)

    def complete(self, evidence: CausalEvidence) -> CausalDecision:
        receipt = self.begin_or_resume(evidence.seal, evidence.split)
        if receipt["status"] == "completed":
            raise ValueError("completed endpoint cannot be reopened")
        decision = analyze_causal_evidence(evidence)
        self._write(
            self._path(evidence.split),
            {
                "split": evidence.split,
                "seal_sha256": evidence.seal.seal_sha256,
                "status": "completed",
                "decision_status": decision.status,
            },
        )
        return decision

    def _path(self, split: str) -> Path:
        return self.root / f"{split}.json"

    def _write(self, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(_canonical_json(value) + b"\n")
        temporary.replace(path)
