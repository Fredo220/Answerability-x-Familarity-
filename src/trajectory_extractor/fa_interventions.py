"""Leakage-resistant prefill interventions for Familiarity-vs-Answerability."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from .fa_artifacts import FAArtifactStore
from .fa_config import (
    CONFIRMATORY_CHAT_TEMPLATE_SHA256,
    CONFIRMATORY_MODEL_ID,
    CONFIRMATORY_MODEL_REVISION,
    CONFIRMATORY_THRESHOLDS,
)
from .fa_probes import F2AGates, GateCriterion, HypothesisGate
from .fa_scoring import (
    BehavioralMetrics,
    BootstrapDistribution,
    OutcomeClass,
    PercentileInterval,
    SensitivityResult,
    behavioral_gate,
    score_response,
)
from .labels import is_refusal


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_ALPHA_GRID = frozenset({0.25, 0.50, 1.00, 1.50})
_ANCHORS = frozenset({"target_intro_end", "user_prompt_end"})
_DIRECTIONS = frozenset({"high_to_low", "low_to_high"})

REQUIRED_CAUSAL_CONTROLS = (
    "no_intervention",
    "norm_matched_random",
    "orthogonal",
    "shuffled",
    "answerability_norm_residualized",
    "sign_reversed",
    "wrong_layer",
    "wrong_anchor",
    "cross_entity",
    "reverse_direction",
)
CAPABILITY_CONTROLS = ("target_bound", "unrelated")
_H7_BOOTSTRAP_REPLICATES = 10_000
_H7_BOOTSTRAP_SEED = 20260722
_H7_ALPHA = 0.05
_CONTROL_FIT_METHOD = "difference_in_means_with_median_norm_v1"
_CONTROL_PERMUTATION_SEED = 20260722
_CONTROL_DIRECTION_DERIVATION = {
    "shuffled_direction": "mean_activation_by_fixed_permuted_familiarity_label",
    "answerability_direction": "mean_activation_answerable_minus_unanswerable",
    "activation_norm_direction": "mean_activation_above_median_norm_minus_at_or_below",
}
_REGISTERED_DOMAINS = frozenset({"person", "place", "organization", "creative_work"})
_CONFIRMATORY_CONFIG_SHA256 = (
    "76f557db589863ab217f963ce5020b4a57e88774582e1e3d3bd58600143103fa"
)
_SOURCE_PINS_SHA256 = (
    "af9f6a042168c4715958d7d376c1eaedc4e643c56eb6da34ff47574799bc33f9"
)
_GATE_EVIDENCE_TOKEN = object()
_CONTROL_SOURCE_TOKEN = object()


class _PatchSession(Protocol):
    def __enter__(self) -> "_PatchSession": ...

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool | None: ...

    def prefill(self, input_ids: Sequence[int]) -> Any: ...


class PrefillPatchRunner(Protocol):
    model_revision: str
    patch_active: bool
    decode_hook_calls: int

    def prefill_patch(self, *, intervention: "AppliedIntervention") -> _PatchSession: ...

    def decode(self, prefill_state: Any) -> Mapping[str, Any]: ...

    def observed_patch_audit(self) -> Mapping[str, Any]: ...


class ConfirmatoryInterventionExecutor(Protocol):
    """Adapter that executes one preregistered raw intervention trial."""

    def execute(
        self,
        *,
        pair: "InterventionPair",
        intervention: "AppliedIntervention",
        activation_manifest_sha256: str,
        confirmatory_pins: "ConfirmatoryPinBundle",
    ) -> "ExecutedPatchEvidence": ...

    def execute_unrelated(
        self,
        *,
        prompt: "UnrelatedCapabilityPrompt",
        intervention: "AppliedIntervention",
        activation_manifest_sha256: str,
        confirmatory_pins: "ConfirmatoryPinBundle",
    ) -> "ExecutedUnrelatedEvidence": ...


@dataclass(frozen=True)
class ReadoutConstraints:
    """Validation-frozen construct and generic-confidence tolerances."""

    familiarity_min_effect: float
    answerability_max_abs_change: float
    entity_type_max_abs_change: float
    generic_confidence_max_abs_change: float

    def __post_init__(self) -> None:
        values = (
            self.familiarity_min_effect,
            self.answerability_max_abs_change,
            self.entity_type_max_abs_change,
            self.generic_confidence_max_abs_change,
        )
        if any(type(value) not in {int, float} or not np.isfinite(value) for value in values):
            raise ValueError("readout constraints must be finite")
        if self.familiarity_min_effect <= 0 or any(value < 0 for value in values[1:]):
            raise ValueError("readout constraints must be positive or nonnegative as registered")

    def canonical_payload(self) -> Mapping[str, float]:
        return {
            "familiarity_min_effect": float(self.familiarity_min_effect),
            "answerability_max_abs_change": float(self.answerability_max_abs_change),
            "entity_type_max_abs_change": float(self.entity_type_max_abs_change),
            "generic_confidence_max_abs_change": float(
                self.generic_confidence_max_abs_change
            ),
        }


@dataclass(frozen=True)
class ReadoutSnapshot:
    familiarity: float
    answerability: float
    entity_type: float
    generic_confidence: float

    def __post_init__(self) -> None:
        values = (
            self.familiarity,
            self.answerability,
            self.entity_type,
            self.generic_confidence,
        )
        if any(type(value) not in {int, float} or not np.isfinite(value) for value in values):
            raise ValueError("decoded readouts must be finite numeric values")
        if not 0.0 <= float(self.generic_confidence) <= 1.0:
            raise ValueError("generic confidence readout must be in [0, 1]")

    @classmethod
    def from_mapping(cls, value: Any, name: str) -> "ReadoutSnapshot":
        if not isinstance(value, Mapping) or set(value) != {
            "familiarity",
            "answerability",
            "entity_type",
            "generic_confidence",
        }:
            raise ValueError(f"{name} must contain the registered readouts")
        return cls(**{key: value[key] for key in value})

    def canonical_payload(self) -> Mapping[str, float]:
        return {
            "familiarity": float(self.familiarity),
            "answerability": float(self.answerability),
            "entity_type": float(self.entity_type),
            "generic_confidence": float(self.generic_confidence),
        }


@dataclass(frozen=True)
class AppliedIntervention:
    """The exact replacement/control tensor that an adapter must apply and audit."""

    control_name: str
    direction: str
    layer: int
    anchor: str
    position: int
    apply_patch: bool
    source_kind: str
    source_example_id: str | None
    source_entity_unit_id: str | None
    source_evidence_sha256: str
    source_activation_sha256: str | None
    destination_example_id: str
    destination_entity_unit_id: str | None
    destination_evidence_sha256: str
    control_source_split: str | None
    control_source_artifact_sha256: str | None
    control_source_component_sha256: str | None
    activation_manifest_sha256: str
    vector: np.ndarray
    replacement: np.ndarray

    def __post_init__(self) -> None:
        if self.control_name not in {
            "primary",
            "target_bound",
            "unrelated",
            *REQUIRED_CAUSAL_CONTROLS,
        }:
            raise ValueError("applied intervention control is not preregistered")
        if self.direction not in _DIRECTIONS:
            raise ValueError("applied intervention direction is invalid")
        if type(self.layer) is not int or self.layer < 0 or type(self.position) is not int or self.position < 0:
            raise ValueError("applied intervention site is invalid")
        if self.anchor not in _ANCHORS:
            raise ValueError("applied intervention anchor is invalid")
        if type(self.apply_patch) is not bool:
            raise ValueError("apply_patch must be boolean")
        _nonempty(self.source_kind, "source_kind")
        if (self.source_example_id is None) != (self.source_entity_unit_id is None):
            raise ValueError("source identity must be complete or absent")
        _sha256(self.source_evidence_sha256, "source_evidence_sha256")
        if self.source_activation_sha256 is not None:
            _sha256(self.source_activation_sha256, "source_activation_sha256")
        _nonempty(self.destination_example_id, "destination_example_id")
        if self.destination_entity_unit_id is not None:
            _nonempty(self.destination_entity_unit_id, "destination_entity_unit_id")
        _sha256(self.destination_evidence_sha256, "destination_evidence_sha256")
        control_source = (
            self.control_source_split,
            self.control_source_artifact_sha256,
            self.control_source_component_sha256,
        )
        if any(value is not None for value in control_source):
            if any(value is None for value in control_source):
                raise ValueError("control source provenance must be complete or absent")
            if self.control_source_split != "locked_validation":
                raise ValueError("confirmatory control source must use locked_validation")
            _sha256(
                self.control_source_artifact_sha256,
                "control_source_artifact_sha256",
            )
            _sha256(
                self.control_source_component_sha256,
                "control_source_component_sha256",
            )
        _sha256(self.activation_manifest_sha256, "activation_manifest_sha256")
        vector = _finite_vector(self.vector, "applied vector")
        replacement = _finite_vector(
            self.replacement, "applied replacement", shape=vector.shape
        )
        vector.setflags(write=False)
        replacement.setflags(write=False)
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "replacement", replacement)
        if not self.apply_patch and (
            self.control_name != "no_intervention" or np.linalg.norm(vector) > 1e-12
        ):
            raise ValueError("only the no_intervention control may be a zero no-op")

    @property
    def vector_sha256(self) -> str:
        return hashlib.sha256(_array_bytes(self.vector)).hexdigest()

    @property
    def replacement_sha256(self) -> str:
        return hashlib.sha256(_array_bytes(self.replacement)).hexdigest()

    @property
    def sha256(self) -> str:
        return _hash(self.canonical_payload())

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "control_name": self.control_name,
            "direction": self.direction,
            "layer": self.layer,
            "anchor": self.anchor,
            "position": self.position,
            "apply_patch": self.apply_patch,
            "source_kind": self.source_kind,
            "source_example_id": self.source_example_id,
            "source_entity_unit_id": self.source_entity_unit_id,
            "source_evidence_sha256": self.source_evidence_sha256,
            "source_activation_sha256": self.source_activation_sha256,
            "destination_example_id": self.destination_example_id,
            "destination_entity_unit_id": self.destination_entity_unit_id,
            "destination_evidence_sha256": self.destination_evidence_sha256,
            "control_source_split": self.control_source_split,
            "control_source_artifact_sha256": self.control_source_artifact_sha256,
            "control_source_component_sha256": self.control_source_component_sha256,
            "activation_manifest_sha256": self.activation_manifest_sha256,
            "vector_sha256": self.vector_sha256,
            "replacement_sha256": self.replacement_sha256,
            "vector_shape": list(self.vector.shape),
            "replacement_shape": list(self.replacement.shape),
        }


@dataclass(frozen=True)
class ExecutedPatchEvidence:
    """Adapter output before any confirmatory label is derived."""

    applied_intervention_sha256: str
    audit: "PatchAudit"
    provenance: Mapping[str, Any]
    decoded: Mapping[str, Any]

    def __post_init__(self) -> None:
        _sha256(self.applied_intervention_sha256, "applied_intervention_sha256")
        if not isinstance(self.audit, PatchAudit):
            raise ValueError("executed patch evidence requires a typed audit")
        object.__setattr__(self, "provenance", _deep_freeze(self.provenance))
        object.__setattr__(self, "decoded", _deep_freeze(self.decoded))


@dataclass(frozen=True)
class ExecutedUnrelatedEvidence:
    """Adapter output for a capability prompt under the selected intervention."""

    applied_intervention_sha256: str
    audit: "PatchAudit"
    provenance: Mapping[str, Any]
    decoded: Mapping[str, Any]

    def __post_init__(self) -> None:
        _sha256(self.applied_intervention_sha256, "applied_intervention_sha256")
        if not isinstance(self.audit, PatchAudit):
            raise ValueError("unrelated evidence requires a typed patch audit")
        object.__setattr__(self, "provenance", _deep_freeze(self.provenance))
        object.__setattr__(self, "decoded", _deep_freeze(self.decoded))


@dataclass(frozen=True, init=False)
class GateArtifactEvidence:
    """Exact immutable F1/F2A result-artifact binding used by F2B selection."""

    phase: str
    status: str
    preregistration_sha256: str
    config_sha256: str
    result_sha256: str
    artifact_sha256: str
    manifest_sha256: str
    probe_selection_sha256: str | None
    endpoint: str
    run_id: str
    artifact_store_sha256: str
    endpoint_input_sha256: str
    endpoint_input_manifest_sha256: str
    closed_state_sha256: str

    @classmethod
    def _from_verified_artifact(
        cls,
        *,
        _verification_token: object,
        phase: str,
        status: str,
        preregistration_sha256: str,
        config_sha256: str,
        result_sha256: str,
        artifact_sha256: str,
        manifest_sha256: str,
        probe_selection_sha256: str | None,
        endpoint: str = "unverified",
        run_id: str = "unverified",
        artifact_store_sha256: str = "0" * 64,
        endpoint_input_sha256: str = "0" * 64,
        endpoint_input_manifest_sha256: str = "0" * 64,
        closed_state_sha256: str = "0" * 64,
    ) -> "GateArtifactEvidence":
        if _verification_token is not _GATE_EVIDENCE_TOKEN:
            raise ValueError("gate evidence may only come from artifact verification")
        value = object.__new__(cls)
        for name, item in {
            "phase": phase,
            "status": status,
            "preregistration_sha256": preregistration_sha256,
            "config_sha256": config_sha256,
            "result_sha256": result_sha256,
            "artifact_sha256": artifact_sha256,
            "manifest_sha256": manifest_sha256,
            "probe_selection_sha256": probe_selection_sha256,
            "endpoint": endpoint,
            "run_id": run_id,
            "artifact_store_sha256": artifact_store_sha256,
            "endpoint_input_sha256": endpoint_input_sha256,
            "endpoint_input_manifest_sha256": endpoint_input_manifest_sha256,
            "closed_state_sha256": closed_state_sha256,
        }.items():
            object.__setattr__(value, name, item)
        value.__post_init__()
        return value

    def __post_init__(self) -> None:
        if self.phase not in {"F1", "F2A"}:
            raise ValueError("gate evidence phase must be F1 or F2A")
        if self.status not in {"supported", "not_supported", "not_evaluable"}:
            raise ValueError("gate evidence status is invalid")
        for name in (
            "preregistration_sha256",
            "config_sha256",
            "result_sha256",
            "artifact_sha256",
            "manifest_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.phase == "F2A":
            _sha256(self.probe_selection_sha256, "probe_selection_sha256")
        elif self.probe_selection_sha256 is not None:
            raise ValueError("F1 gate evidence cannot bind an F2A probe selection")
        expected_endpoint = "behavior_test" if self.phase == "F1" else "probe_test"
        if self.endpoint != expected_endpoint:
            raise ValueError("gate evidence endpoint does not match its phase")
        _nonempty(self.run_id, "gate evidence run_id")
        for name in (
            "artifact_store_sha256",
            "endpoint_input_sha256",
            "endpoint_input_manifest_sha256",
            "closed_state_sha256",
        ):
            _sha256(getattr(self, name), name)

    @property
    def sha256(self) -> str:
        return _hash(self.canonical_payload())

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "preregistration_sha256": self.preregistration_sha256,
            "config_sha256": self.config_sha256,
            "result_sha256": self.result_sha256,
            "artifact_sha256": self.artifact_sha256,
            "manifest_sha256": self.manifest_sha256,
            "probe_selection_sha256": self.probe_selection_sha256,
            "endpoint": self.endpoint,
            "run_id": self.run_id,
            "artifact_store_sha256": self.artifact_store_sha256,
            "endpoint_input_sha256": self.endpoint_input_sha256,
            "endpoint_input_manifest_sha256": self.endpoint_input_manifest_sha256,
            "closed_state_sha256": self.closed_state_sha256,
        }


@dataclass(frozen=True, init=False)
class ValidationControlSource:
    """Verified locked-validation vectors used by confirmatory F2B controls."""

    source_split: str
    source_artifact_sha256: str
    source_manifest_sha256: str
    preregistration_sha256: str
    probe_selection_sha256: str
    input_artifact_sha256: str
    input_manifest_sha256: str
    activation_artifact_sha256: str
    activation_manifest_sha256: str
    example_ids: tuple[str, ...]
    input_row_ids: tuple[str, ...]
    activation_row_ids: tuple[str, ...]
    input_row_ids_sha256: str
    activation_row_ids_sha256: str
    input_row_sha256s: tuple[str, ...]
    activation_row_sha256s: tuple[str, ...]
    input_rows_sha256: str
    activation_rows_sha256: str
    label_sha256: str
    fit_method: str
    permutation_seed: int
    permutation: tuple[int, ...]
    permutation_sha256: str
    direction_derivation: Mapping[str, str]
    direction_derivation_sha256: str
    shuffled_direction: np.ndarray
    answerability_direction: np.ndarray
    activation_norm_direction: np.ndarray

    @classmethod
    def _from_verified_artifact(
        cls,
        *,
        _verification_token: object,
        source_split: str,
        source_artifact_sha256: str,
        source_manifest_sha256: str,
        preregistration_sha256: str,
        probe_selection_sha256: str,
        input_artifact_sha256: str,
        input_manifest_sha256: str,
        activation_artifact_sha256: str,
        activation_manifest_sha256: str,
        example_ids: tuple[str, ...],
        input_row_ids: tuple[str, ...],
        activation_row_ids: tuple[str, ...],
        input_row_ids_sha256: str,
        activation_row_ids_sha256: str,
        input_row_sha256s: tuple[str, ...],
        activation_row_sha256s: tuple[str, ...],
        input_rows_sha256: str,
        activation_rows_sha256: str,
        label_sha256: str,
        fit_method: str,
        permutation_seed: int,
        permutation: tuple[int, ...],
        permutation_sha256: str,
        direction_derivation: Mapping[str, str],
        direction_derivation_sha256: str,
        shuffled_direction: np.ndarray,
        answerability_direction: np.ndarray,
        activation_norm_direction: np.ndarray,
    ) -> "ValidationControlSource":
        if _verification_token is not _CONTROL_SOURCE_TOKEN:
            raise ValueError("control source may only come from artifact verification")
        value = object.__new__(cls)
        for name, item in {
            "source_split": source_split,
            "source_artifact_sha256": source_artifact_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "preregistration_sha256": preregistration_sha256,
            "probe_selection_sha256": probe_selection_sha256,
            "input_artifact_sha256": input_artifact_sha256,
            "input_manifest_sha256": input_manifest_sha256,
            "activation_artifact_sha256": activation_artifact_sha256,
            "activation_manifest_sha256": activation_manifest_sha256,
            "example_ids": example_ids,
            "input_row_ids": input_row_ids,
            "activation_row_ids": activation_row_ids,
            "input_row_ids_sha256": input_row_ids_sha256,
            "activation_row_ids_sha256": activation_row_ids_sha256,
            "input_row_sha256s": input_row_sha256s,
            "activation_row_sha256s": activation_row_sha256s,
            "input_rows_sha256": input_rows_sha256,
            "activation_rows_sha256": activation_rows_sha256,
            "label_sha256": label_sha256,
            "fit_method": fit_method,
            "permutation_seed": permutation_seed,
            "permutation": permutation,
            "permutation_sha256": permutation_sha256,
            "direction_derivation": direction_derivation,
            "direction_derivation_sha256": direction_derivation_sha256,
            "shuffled_direction": shuffled_direction,
            "answerability_direction": answerability_direction,
            "activation_norm_direction": activation_norm_direction,
        }.items():
            object.__setattr__(value, name, item)
        value.__post_init__()
        return value

    def __post_init__(self) -> None:
        if self.source_split != "locked_validation":
            raise ValueError("F2B control vectors must come from locked_validation")
        for name in (
            "source_artifact_sha256",
            "source_manifest_sha256",
            "preregistration_sha256",
            "probe_selection_sha256",
            "input_artifact_sha256",
            "input_manifest_sha256",
            "activation_artifact_sha256",
            "activation_manifest_sha256",
            "input_row_ids_sha256",
            "activation_row_ids_sha256",
            "input_rows_sha256",
            "activation_rows_sha256",
            "label_sha256",
            "permutation_sha256",
            "direction_derivation_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.fit_method != _CONTROL_FIT_METHOD:
            raise ValueError("F2B control fit method is not registered")
        if self.permutation_seed != _CONTROL_PERMUTATION_SEED:
            raise ValueError("F2B shuffled-control seed is not registered")
        example_ids = tuple(self.example_ids)
        input_row_ids = tuple(self.input_row_ids)
        activation_row_ids = tuple(self.activation_row_ids)
        input_hashes = tuple(self.input_row_sha256s)
        activation_hashes = tuple(self.activation_row_sha256s)
        permutation = tuple(self.permutation)
        if (
            not example_ids
            or tuple(sorted(example_ids)) != example_ids
            or len(set(example_ids)) != len(example_ids)
            or len(input_hashes) != len(example_ids)
            or len(activation_hashes) != len(example_ids)
            or len(input_row_ids) != len(example_ids)
            or len(activation_row_ids) != len(example_ids)
            or len(set(input_row_ids)) != len(example_ids)
            or len(set(activation_row_ids)) != len(example_ids)
            or sorted(permutation) != list(range(len(example_ids)))
        ):
            raise ValueError("F2B control row identities or permutation are invalid")
        if (
            _hash(list(input_row_ids)) != self.input_row_ids_sha256
            or _hash(list(activation_row_ids)) != self.activation_row_ids_sha256
        ):
            raise ValueError("F2B control row ID hashes are invalid")
        for digest in (*input_hashes, *activation_hashes):
            _sha256(digest, "F2B control row hash")
        derivation = dict(self.direction_derivation)
        if derivation != _CONTROL_DIRECTION_DERIVATION:
            raise ValueError("F2B control direction derivation is not registered")
        if _hash(derivation) != self.direction_derivation_sha256:
            raise ValueError("F2B control direction derivation hash is invalid")
        if _hash(list(permutation)) != self.permutation_sha256:
            raise ValueError("F2B shuffled-control permutation hash is invalid")
        object.__setattr__(self, "example_ids", example_ids)
        object.__setattr__(self, "input_row_ids", input_row_ids)
        object.__setattr__(self, "activation_row_ids", activation_row_ids)
        object.__setattr__(self, "input_row_sha256s", input_hashes)
        object.__setattr__(self, "activation_row_sha256s", activation_hashes)
        object.__setattr__(self, "permutation", permutation)
        object.__setattr__(
            self, "direction_derivation", MappingProxyType(derivation)
        )
        shuffled = _finite_vector(self.shuffled_direction, "shuffled direction")
        answerability = _finite_vector(
            self.answerability_direction,
            "answerability direction",
            shape=shuffled.shape,
        )
        activation_norm = _finite_vector(
            self.activation_norm_direction,
            "activation norm direction",
            shape=shuffled.shape,
        )
        for name, vector in (
            ("shuffled_direction", shuffled),
            ("answerability_direction", answerability),
            ("activation_norm_direction", activation_norm),
        ):
            if np.linalg.norm(vector) <= 1e-12:
                raise ValueError(f"{name} must have nonzero norm")
            vector.setflags(write=False)
            object.__setattr__(self, name, vector)

    @property
    def shuffled_direction_sha256(self) -> str:
        return hashlib.sha256(_array_bytes(self.shuffled_direction)).hexdigest()

    @property
    def answerability_direction_sha256(self) -> str:
        return hashlib.sha256(_array_bytes(self.answerability_direction)).hexdigest()

    @property
    def activation_norm_direction_sha256(self) -> str:
        return hashlib.sha256(_array_bytes(self.activation_norm_direction)).hexdigest()

    @property
    def residualizer_sha256(self) -> str:
        return _hash(
            {
                "answerability_direction_sha256": self.answerability_direction_sha256,
                "activation_norm_direction_sha256": self.activation_norm_direction_sha256,
                "source_artifact_sha256": self.source_artifact_sha256,
                "direction_derivation_sha256": self.direction_derivation_sha256,
            }
        )

    @property
    def sha256(self) -> str:
        return _hash(self.canonical_payload())

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "source_split": self.source_split,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "probe_selection_sha256": self.probe_selection_sha256,
            "input_artifact_sha256": self.input_artifact_sha256,
            "input_manifest_sha256": self.input_manifest_sha256,
            "activation_artifact_sha256": self.activation_artifact_sha256,
            "activation_manifest_sha256": self.activation_manifest_sha256,
            "example_ids": list(self.example_ids),
            "input_row_ids": list(self.input_row_ids),
            "activation_row_ids": list(self.activation_row_ids),
            "input_row_ids_sha256": self.input_row_ids_sha256,
            "activation_row_ids_sha256": self.activation_row_ids_sha256,
            "input_row_sha256s": list(self.input_row_sha256s),
            "activation_row_sha256s": list(self.activation_row_sha256s),
            "input_rows_sha256": self.input_rows_sha256,
            "activation_rows_sha256": self.activation_rows_sha256,
            "label_sha256": self.label_sha256,
            "fit_method": self.fit_method,
            "permutation_seed": self.permutation_seed,
            "permutation": list(self.permutation),
            "permutation_sha256": self.permutation_sha256,
            "direction_derivation": dict(self.direction_derivation),
            "direction_derivation_sha256": self.direction_derivation_sha256,
            "shuffled_direction_sha256": self.shuffled_direction_sha256,
            "answerability_direction_sha256": self.answerability_direction_sha256,
            "activation_norm_direction_sha256": self.activation_norm_direction_sha256,
            "vector_shape": list(self.shuffled_direction.shape),
        }


def _control_difference_in_means(
    activations: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    matrix = np.asarray(activations, dtype=np.float64)
    binary = np.asarray(labels, dtype=np.int64)
    if (
        matrix.ndim != 2
        or binary.shape != (matrix.shape[0],)
        or set(binary.tolist()) != {0, 1}
    ):
        raise ValueError("F2B control derivation requires both binary label groups")
    return matrix[binary == 1].mean(axis=0) - matrix[binary == 0].mean(axis=0)


def verify_validation_control_artifact(
    store: FAArtifactStore,
    manifest_path: str | Path,
    *,
    preregistration_sha256: str,
    probe_selection_sha256: str,
) -> ValidationControlSource:
    """Verify and type the preregistered locked-validation F2B control vectors."""

    if not isinstance(store, FAArtifactStore):
        raise ValueError("control verification requires an FAArtifactStore")
    _sha256(preregistration_sha256, "preregistration_sha256")
    _sha256(probe_selection_sha256, "probe_selection_sha256")
    shard = store.verify_shard(manifest_path)
    if (
        shard.namespace != "locked_validation"
        or shard.record_kind != "f2b_control_source"
        or shard.row_count != 1
    ):
        raise ValueError("F2B control source must be a one-row locked_validation artifact")
    row = _read_canonical_rows(shard)[0]
    expected_fields = {
        "kind",
        "source_split",
        "preregistration_sha256",
        "probe_selection_sha256",
        "input_manifest_path",
        "input_artifact_sha256",
        "input_manifest_sha256",
        "activation_manifest_path",
        "activation_artifact_sha256",
        "activation_manifest_sha256",
        "example_ids",
        "input_row_ids",
        "activation_row_ids",
        "input_row_ids_sha256",
        "activation_row_ids_sha256",
        "input_row_sha256s",
        "activation_row_sha256s",
        "input_rows_sha256",
        "activation_rows_sha256",
        "label_sha256",
        "fit_method",
        "permutation_seed",
        "permutation",
        "permutation_sha256",
        "direction_derivation",
        "direction_derivation_sha256",
        "shuffled_direction",
        "shuffled_direction_sha256",
        "answerability_direction",
        "answerability_direction_sha256",
        "activation_norm_direction",
        "activation_norm_direction_sha256",
    }
    if set(row) != expected_fields or row.get("kind") != "f2b_control_source":
        raise ValueError("F2B control source row has an invalid schema")
    if (
        row.get("source_split") != "locked_validation"
        or row.get("preregistration_sha256") != preregistration_sha256
        or row.get("probe_selection_sha256") != probe_selection_sha256
    ):
        raise ValueError("F2B control source lineage is invalid")
    if (
        row["fit_method"] != _CONTROL_FIT_METHOD
        or row["permutation_seed"] != _CONTROL_PERMUTATION_SEED
        or row["direction_derivation"] != _CONTROL_DIRECTION_DERIVATION
        or row["direction_derivation_sha256"]
        != _hash(_CONTROL_DIRECTION_DERIVATION)
    ):
        raise ValueError("F2B control derivation is not the registered derivation")
    input_manifest_path = store.root / str(row["input_manifest_path"])
    activation_manifest_path = store.root / str(row["activation_manifest_path"])
    input_shard = store.verify_shard(input_manifest_path)
    activation_shard = store.verify_shard(activation_manifest_path)
    if (
        input_shard.namespace != "locked_validation"
        or input_shard.record_kind != "f2b_control_input"
        or activation_shard.namespace != "locked_validation"
        or activation_shard.record_kind != "f2b_control_activation"
        or input_shard.row_count == 0
        or activation_shard.row_count != input_shard.row_count
    ):
        raise ValueError("F2B controls require verified locked_validation source manifests")
    input_manifest_sha256 = hashlib.sha256(
        Path(input_shard.manifest_path).read_bytes()
    ).hexdigest()
    activation_manifest_sha256 = hashlib.sha256(
        Path(activation_shard.manifest_path).read_bytes()
    ).hexdigest()
    if (
        row["input_artifact_sha256"] != input_shard.sha256
        or row["input_manifest_sha256"] != input_manifest_sha256
        or row["activation_artifact_sha256"] != activation_shard.sha256
        or row["activation_manifest_sha256"] != activation_manifest_sha256
    ):
        raise ValueError("F2B control source manifest hashes do not verify")
    input_lineage = _verified_manifest_lineage(input_shard)
    activation_lineage = _verified_manifest_lineage(activation_shard)
    if dict(input_lineage) != {
        "preregistration_sha256": preregistration_sha256,
        "probe_selection_sha256": probe_selection_sha256,
    } or dict(activation_lineage) != {
        "preregistration_sha256": preregistration_sha256,
        "probe_selection_sha256": probe_selection_sha256,
        "input_artifact_sha256": input_shard.sha256,
        "input_manifest_sha256": input_manifest_sha256,
    }:
        raise ValueError("F2B locked_validation source lineage is invalid")
    input_rows = tuple(
        sorted(_read_canonical_rows(input_shard), key=lambda item: item.get("example_id", ""))
    )
    activation_rows = tuple(
        sorted(
            _read_canonical_rows(activation_shard),
            key=lambda item: item.get("example_id", ""),
        )
    )
    input_schema = {
        "kind",
        "row_id",
        "example_id",
        "familiarity_label",
        "answerability_label",
        "registered_input",
        "input_sha256",
    }
    activation_schema = {
        "kind",
        "row_id",
        "example_id",
        "input_row_sha256",
        "activation",
        "activation_sha256",
    }
    if any(set(item) != input_schema for item in input_rows) or any(
        set(item) != activation_schema for item in activation_rows
    ):
        raise ValueError("F2B locked_validation source rows have an invalid schema")
    example_ids = tuple(item["example_id"] for item in input_rows)
    if (
        not example_ids
        or tuple(sorted(example_ids)) != example_ids
        or len(set(example_ids)) != len(example_ids)
        or tuple(item["example_id"] for item in activation_rows) != example_ids
        or tuple(row["example_ids"]) != example_ids
    ):
        raise ValueError("F2B locked_validation example identities do not match")
    input_hashes = tuple(_hash(item) for item in input_rows)
    activation_hashes = tuple(_hash(item) for item in activation_rows)
    input_row_ids = tuple(item["row_id"] for item in input_rows)
    activation_row_ids = tuple(item["row_id"] for item in activation_rows)
    if (
        tuple(row["input_row_ids"]) != input_row_ids
        or tuple(row["activation_row_ids"]) != activation_row_ids
        or row["input_row_ids_sha256"] != _hash(list(input_row_ids))
        or row["activation_row_ids_sha256"]
        != _hash(list(activation_row_ids))
        or len(set(input_row_ids)) != len(input_row_ids)
        or len(set(activation_row_ids)) != len(activation_row_ids)
        or tuple(row["input_row_sha256s"]) != input_hashes
        or tuple(row["activation_row_sha256s"]) != activation_hashes
        or row["input_rows_sha256"] != _hash(list(input_hashes))
        or row["activation_rows_sha256"] != _hash(list(activation_hashes))
    ):
        raise ValueError("F2B control row identity hashes do not verify")
    labels = []
    activations = []
    for input_row, activation_row, input_hash in zip(
        input_rows, activation_rows, input_hashes, strict=True
    ):
        input_payload = {
            name: input_row[name]
            for name in (
                "example_id",
                "familiarity_label",
                "answerability_label",
                "registered_input",
            )
        }
        if input_row["input_sha256"] != _hash(input_payload):
            raise ValueError("F2B locked_validation input hash does not verify")
        if input_row["familiarity_label"] not in {0, 1} or input_row[
            "answerability_label"
        ] not in {0, 1}:
            raise ValueError("F2B control labels must be binary")
        vector = _finite_vector(
            np.asarray(activation_row["activation"], dtype=np.float64),
            "locked_validation control activation",
        )
        if (
            activation_row["input_row_sha256"] != input_hash
            or activation_row["activation_sha256"]
            != hashlib.sha256(_array_bytes(vector)).hexdigest()
        ):
            raise ValueError("F2B locked_validation activation provenance is invalid")
        labels.append(
            {
                "example_id": input_row["example_id"],
                "familiarity_label": input_row["familiarity_label"],
                "answerability_label": input_row["answerability_label"],
            }
        )
        activations.append(vector)
    if row["label_sha256"] != _hash(labels):
        raise ValueError("F2B control label hash does not verify")
    permutation = tuple(row["permutation"])
    expected_permutation = tuple(
        int(value)
        for value in np.random.default_rng(_CONTROL_PERMUTATION_SEED).permutation(
            len(example_ids)
        )
    )
    if (
        permutation != expected_permutation
        or row["permutation_sha256"] != _hash(list(expected_permutation))
    ):
        raise ValueError("F2B shuffled direction permutation is not registered")
    matrix = np.stack(activations, axis=0)
    familiarity = np.asarray(
        [item["familiarity_label"] for item in labels], dtype=np.int64
    )
    answerability = np.asarray(
        [item["answerability_label"] for item in labels], dtype=np.int64
    )
    norms = np.linalg.norm(matrix, axis=1)
    norm_labels = (norms > np.median(norms)).astype(np.int64)
    recomputed = {
        "shuffled_direction": _control_difference_in_means(
            matrix, familiarity[np.asarray(expected_permutation)]
        ),
        "answerability_direction": _control_difference_in_means(
            matrix, answerability
        ),
        "activation_norm_direction": _control_difference_in_means(
            matrix, norm_labels
        ),
    }
    vectors = {}
    for name, expected_vector in recomputed.items():
        supplied = _finite_vector(
            np.asarray(row[name], dtype=np.float64), name, shape=expected_vector.shape
        )
        expected_hash = hashlib.sha256(_array_bytes(expected_vector)).hexdigest()
        if row[f"{name}_sha256"] != expected_hash or not np.array_equal(
            supplied, expected_vector
        ):
            raise ValueError(f"{name} does not match its recomputed derivation")
        vectors[name] = expected_vector
    lineage = _verified_manifest_lineage(shard)
    lineage_names = (
        "preregistration_sha256",
        "probe_selection_sha256",
        "input_artifact_sha256",
        "input_manifest_sha256",
        "activation_artifact_sha256",
        "activation_manifest_sha256",
        "input_row_ids_sha256",
        "activation_row_ids_sha256",
        "input_rows_sha256",
        "activation_rows_sha256",
        "label_sha256",
        "fit_method",
        "permutation_seed",
        "permutation_sha256",
        "direction_derivation_sha256",
        "shuffled_direction_sha256",
        "answerability_direction_sha256",
        "activation_norm_direction_sha256",
    )
    if dict(lineage) != {name: row[name] for name in lineage_names}:
        raise ValueError("F2B control source manifest lineage is invalid")
    return ValidationControlSource._from_verified_artifact(
        _verification_token=_CONTROL_SOURCE_TOKEN,
        source_split="locked_validation",
        source_artifact_sha256=shard.sha256,
        source_manifest_sha256=hashlib.sha256(
            Path(shard.manifest_path).read_bytes()
        ).hexdigest(),
        preregistration_sha256=preregistration_sha256,
        probe_selection_sha256=probe_selection_sha256,
        input_artifact_sha256=input_shard.sha256,
        input_manifest_sha256=input_manifest_sha256,
        activation_artifact_sha256=activation_shard.sha256,
        activation_manifest_sha256=activation_manifest_sha256,
        example_ids=example_ids,
        input_row_ids=input_row_ids,
        activation_row_ids=activation_row_ids,
        input_row_ids_sha256=row["input_row_ids_sha256"],
        activation_row_ids_sha256=row["activation_row_ids_sha256"],
        input_row_sha256s=input_hashes,
        activation_row_sha256s=activation_hashes,
        input_rows_sha256=row["input_rows_sha256"],
        activation_rows_sha256=row["activation_rows_sha256"],
        label_sha256=row["label_sha256"],
        fit_method=row["fit_method"],
        permutation_seed=row["permutation_seed"],
        permutation=expected_permutation,
        permutation_sha256=row["permutation_sha256"],
        direction_derivation=row["direction_derivation"],
        direction_derivation_sha256=row["direction_derivation_sha256"],
        shuffled_direction=vectors["shuffled_direction"],
        answerability_direction=vectors["answerability_direction"],
        activation_norm_direction=vectors["activation_norm_direction"],
    )


def verify_gate_result_artifact(
    store: FAArtifactStore,
    manifest_path: str | Path,
    *,
    phase: str,
) -> GateArtifactEvidence:
    """Recompute a canonical gate from one CLOSED protected endpoint."""

    if not isinstance(store, FAArtifactStore):
        raise ValueError("gate verification requires an FAArtifactStore")
    if phase not in {"F1", "F2A"}:
        raise ValueError("gate evidence phase must be F1 or F2A")
    endpoint = "behavior_test" if phase == "F1" else "probe_test"
    closed = store.read_closed_metrics(endpoint, manifest_path)
    shard = closed.metrics_artifact
    if shard.row_count != 1:
        raise ValueError("closed gate endpoint must contain one metrics row")
    rows = _read_canonical_rows(shard)
    row = rows[0]
    metrics_lineage = _verified_manifest_lineage(shard)
    input_lineage = _verified_manifest_lineage(closed.input_artifact)
    if phase == "F1":
        status, config, result_sha256 = _verify_f1_gate_row(
            row,
            metrics_lineage=metrics_lineage,
            endpoint_input_sha256=closed.input_artifact.sha256,
            preregistration_sha256=closed.preregistration_hash,
            selection_manifest_sha256=closed.selection_manifest_hash,
        )
        probe_selection = None
    else:
        status, result_sha256, probe_selection = _verify_f2a_gate_row(
            row,
            metrics_lineage=metrics_lineage,
            endpoint_input_sha256=closed.input_artifact.sha256,
            selection_manifest_sha256=closed.selection_manifest_hash,
        )
        config = input_lineage.get("config_sha256")
        _sha256(config, "closed probe endpoint config_sha256")
    return GateArtifactEvidence._from_verified_artifact(
        _verification_token=_GATE_EVIDENCE_TOKEN,
        phase=phase,
        status=status,
        preregistration_sha256=closed.preregistration_hash,
        config_sha256=config,
        result_sha256=result_sha256,
        artifact_sha256=shard.sha256,
        manifest_sha256=hashlib.sha256(Path(shard.manifest_path).read_bytes()).hexdigest(),
        probe_selection_sha256=probe_selection,
        endpoint=endpoint,
        run_id=closed.run_id,
        artifact_store_sha256=_hash({"root": str(store.root)}),
        endpoint_input_sha256=closed.input_artifact.sha256,
        endpoint_input_manifest_sha256=hashlib.sha256(
            Path(closed.input_artifact.manifest_path).read_bytes()
        ).hexdigest(),
        closed_state_sha256=closed.closed_state_sha256,
    )


def _verified_manifest_lineage(shard: Any) -> Mapping[str, Any]:
    try:
        manifest = json.loads(Path(shard.manifest_path).read_bytes())
        lineage = manifest["lineage"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError("verified artifact manifest lineage is unreadable") from error
    if not isinstance(lineage, Mapping):
        raise ValueError("verified artifact manifest lineage is invalid")
    return lineage


def _cell_mapping(value: Any, name: str) -> Mapping[tuple[str, str, str], Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a cell mapping")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or len(key.split(":")) != 3:
            raise ValueError(f"{name} contains an invalid cell key")
        result[tuple(key.split(":"))] = item
    return result


def _percentile_interval(value: Any, name: str) -> PercentileInterval | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"estimate", "lower", "upper"}:
        raise ValueError(f"{name} has an invalid interval schema")
    return PercentileInterval(value["estimate"], value["lower"], value["upper"])


def _behavioral_metrics(value: Any) -> BehavioralMetrics:
    expected = {
        "status",
        "reasons",
        "cell_rates",
        "completion_by_cell",
        "format_validity_by_cell",
        "denominators",
        "invalid_format_counts",
        "interaction",
        "h2_accuracy_difference",
        "h2b_interaction",
        "sensitivities",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("F1 behavioral metrics have an invalid schema")
    sensitivities = {}
    if not isinstance(value["sensitivities"], Mapping):
        raise ValueError("F1 sensitivities must be a mapping")
    for name, record in value["sensitivities"].items():
        if not isinstance(record, Mapping) or set(record) != {
            "interaction",
            "analytic_denominators",
            "original_denominators",
            "invalid_format_counts",
        }:
            raise ValueError("F1 sensitivity record has an invalid schema")
        sensitivities[name] = SensitivityResult(
            name,
            record["interaction"],
            _cell_mapping(record["analytic_denominators"], "analytic_denominators"),
            _cell_mapping(record["original_denominators"], "original_denominators"),
            _cell_mapping(record["invalid_format_counts"], "invalid_format_counts"),
        )
    return BehavioralMetrics(
        status=value["status"],
        reasons=tuple(value["reasons"]),
        cell_rates=_cell_mapping(value["cell_rates"], "cell_rates"),
        completion_by_cell=_cell_mapping(
            value["completion_by_cell"], "completion_by_cell"
        ),
        format_validity_by_cell=_cell_mapping(
            value["format_validity_by_cell"], "format_validity_by_cell"
        ),
        denominators=_cell_mapping(value["denominators"], "denominators"),
        invalid_format_counts=_cell_mapping(
            value["invalid_format_counts"], "invalid_format_counts"
        ),
        interaction=value["interaction"],
        h2_accuracy_difference=value["h2_accuracy_difference"],
        h2b_interaction=value["h2b_interaction"],
        sensitivities=sensitivities,
    )


def _bootstrap_distribution(value: Any) -> BootstrapDistribution:
    expected = {
        "interaction_samples",
        "h2_accuracy_difference_samples",
        "h2b_interaction_samples",
        "interaction_interval",
        "h2_accuracy_difference_interval",
        "h2b_interaction_interval",
        "weighted_denominators",
        "seed",
        "requested_draws",
        "valid_draws",
        "discarded_draws",
        "resampling_unit",
        "alpha",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("F1 bootstrap record has an invalid schema")
    return BootstrapDistribution(
        interaction_samples=tuple(value["interaction_samples"]),
        h2_accuracy_difference_samples=tuple(
            value["h2_accuracy_difference_samples"]
        ),
        h2b_interaction_samples=tuple(value["h2b_interaction_samples"]),
        interaction_interval=_percentile_interval(
            value["interaction_interval"], "interaction_interval"
        ),
        h2_accuracy_difference_interval=_percentile_interval(
            value["h2_accuracy_difference_interval"],
            "h2_accuracy_difference_interval",
        ),
        h2b_interaction_interval=_percentile_interval(
            value["h2b_interaction_interval"], "h2b_interaction_interval"
        ),
        weighted_denominators=tuple(value["weighted_denominators"]),
        seed=value["seed"],
        requested_draws=value["requested_draws"],
        valid_draws=value["valid_draws"],
        discarded_draws=value["discarded_draws"],
        resampling_unit=tuple(value["resampling_unit"]),
        alpha=value["alpha"],
    )


def _verify_f1_gate_row(
    row: Mapping[str, Any],
    *,
    metrics_lineage: Mapping[str, Any],
    endpoint_input_sha256: str,
    preregistration_sha256: str,
    selection_manifest_sha256: str,
) -> tuple[str, str, str]:
    expected = {
        "kind",
        "phase",
        "metrics",
        "bootstrap",
        "gate",
        "scored_rows",
        "evidence_sha256",
    }
    if set(row) != expected or row.get("kind") != "metrics" or row.get("phase") != "F1":
        raise ValueError("closed F1 metrics row has an invalid schema")
    evidence = {
        name: row[name] for name in ("metrics", "bootstrap", "gate", "scored_rows")
    }
    if row["evidence_sha256"] != _hash(evidence):
        raise ValueError("closed F1 evidence hash does not match its canonical payload")
    metrics = _behavioral_metrics(row["metrics"])
    bootstrap = _bootstrap_distribution(row["bootstrap"])
    gate_record = row["gate"]
    if not isinstance(gate_record, Mapping):
        raise ValueError("closed F1 gate has an invalid schema")
    recomputed = behavioral_gate(
        metrics,
        bootstrap,
        thresholds=gate_record.get("thresholds", {}),
        same_string_sealed=gate_record.get("same_string_sealed"),
        config_hash=gate_record.get("config_hash"),
        manifest_hash=gate_record.get("manifest_hash"),
    )
    if recomputed.to_record() != dict(gate_record):
        raise ValueError("closed F1 gate does not match the canonical recomputed gate")
    expected_lineage = {
        "preregistration_sha256": preregistration_sha256,
        "selection_sha256": selection_manifest_sha256,
        "prompt_manifest_sha256": endpoint_input_sha256,
        "evidence_sha256": row["evidence_sha256"],
        "config_sha256": recomputed.config_hash,
    }
    if any(metrics_lineage.get(name) != value for name, value in expected_lineage.items()):
        raise ValueError("closed F1 metrics lineage does not bind its endpoint and result")
    return recomputed.status, recomputed.config_hash, row["evidence_sha256"]


def _hypothesis_gate_record(value: Any, hypothesis: str) -> HypothesisGate:
    if not isinstance(value, Mapping) or set(value) != {
        "hypothesis",
        "criteria",
        "status",
        "reasons",
    }:
        raise ValueError("F2A hypothesis gate has an invalid schema")
    criteria = []
    for record in value["criteria"]:
        if not isinstance(record, Mapping) or set(record) != {
            "name",
            "observed",
            "threshold",
            "comparison",
            "satisfied",
        }:
            raise ValueError("F2A gate criterion has an invalid schema")
        criterion = GateCriterion(
            record["name"], record["observed"], record["threshold"], record["comparison"]
        )
        if criterion.to_record() != dict(record):
            raise ValueError("F2A gate criterion is not canonical")
        criteria.append(criterion)
    gate = HypothesisGate(hypothesis, tuple(criteria))
    if gate.to_record() != dict(value):
        raise ValueError("F2A gate does not match its recomputed criteria")
    return gate


def _verify_f2a_gate_row(
    row: Mapping[str, Any],
    *,
    metrics_lineage: Mapping[str, Any],
    endpoint_input_sha256: str,
    selection_manifest_sha256: str,
) -> tuple[str, str, str]:
    if set(row) != {"kind", "metric_type", "result"} or (
        row.get("kind"), row.get("metric_type")
    ) != ("metrics", "f2a_bundle"):
        raise ValueError("closed F2A metrics row has an invalid schema")
    result = row["result"]
    expected_bundle = {
        "schema_version",
        "selection_bundle_hash",
        "authorization_sha256",
        "endpoint_input_sha256",
        "endpoint_input_identities_sha256",
        "results",
        "gates",
        "refit_performed",
    }
    if not isinstance(result, Mapping) or set(result) != expected_bundle:
        raise ValueError("closed F2A bundle has an invalid schema")
    if (
        result["schema_version"] != 1
        or result["refit_performed"] is not False
        or result["selection_bundle_hash"] != selection_manifest_sha256
        or result["endpoint_input_sha256"] != endpoint_input_sha256
    ):
        raise ValueError("closed F2A bundle is not bound to its protected endpoint")
    for name in (
        "selection_bundle_hash",
        "authorization_sha256",
        "endpoint_input_sha256",
        "endpoint_input_identities_sha256",
    ):
        _sha256(result[name], f"F2A {name}")
    results = result["results"]
    tasks = ("familiarity", "answerability", "unsupported_answer")
    hypotheses = {"familiarity": "H3", "answerability": "H4", "unsupported_answer": "H5"}
    required_probe_fields = {
        "schema_version",
        "task",
        "selection_hash",
        "authorization_sha256",
        "endpoint_input_sha256",
        "endpoint_input_identities_sha256",
        "test_ids",
        "test_row_sha256s",
        "selected_feature_family",
        "metrics",
        "model_metrics",
        "per_condition",
        "worst_condition",
        "ood_transfer",
        "worst_ood_transfer",
        "cross_condition_transfer",
        "relative_h5_log_loss_improvement",
        "relative_h6_log_loss_improvement",
        "crossed_auroc_95",
        "h5_absolute_log_loss_difference_95",
        "h6_absolute_log_loss_difference_95",
        "primary_gate",
        "null_results",
        "refit_performed",
    }
    if not isinstance(results, Mapping) or set(results) != set(tasks):
        raise ValueError("closed F2A bundle lacks registered probe results")
    for task in tasks:
        probe = results[task]
        if not isinstance(probe, Mapping) or set(probe) != required_probe_fields:
            raise ValueError("closed F2A probe result has an invalid schema")
        if (
            probe["schema_version"] != 2
            or probe["task"] != task
            or probe["refit_performed"] is not False
            or probe["endpoint_input_sha256"] != endpoint_input_sha256
            or not probe["test_ids"]
            or not probe["test_row_sha256s"]
        ):
            raise ValueError("closed F2A probe result is not endpoint-bound")
        for name in (
            "selection_hash",
            "authorization_sha256",
            "endpoint_input_sha256",
            "endpoint_input_identities_sha256",
        ):
            _sha256(probe[name], f"F2A probe {name}")
        for digest in probe["test_row_sha256s"]:
            _sha256(digest, "F2A test row hash")
        _hypothesis_gate_record(probe["primary_gate"], hypotheses[task])
    gates_record = result["gates"]
    if not isinstance(gates_record, Mapping) or set(gates_record) != {
        "familiarity_result_sha256",
        "answerability_result_sha256",
        "unsupported_result_sha256",
        "holm_adjusted_p",
        "h3",
        "h4",
        "h5",
        "h6",
        "h6_secondary",
        "status",
    }:
        raise ValueError("closed F2A joint gate has an invalid schema")
    typed_gates = F2AGates(
        familiarity_result_sha256=gates_record["familiarity_result_sha256"],
        answerability_result_sha256=gates_record["answerability_result_sha256"],
        unsupported_result_sha256=gates_record["unsupported_result_sha256"],
        holm_adjusted_p=gates_record["holm_adjusted_p"],
        h3=_hypothesis_gate_record(gates_record["h3"], "H3"),
        h4=_hypothesis_gate_record(gates_record["h4"], "H4"),
        h5=_hypothesis_gate_record(gates_record["h5"], "H5"),
        h6=_hypothesis_gate_record(gates_record["h6"], "H6"),
    )
    expected_hashes = (
        _hash(results["familiarity"]),
        _hash(results["answerability"]),
        _hash(results["unsupported_answer"]),
    )
    if expected_hashes != (
        typed_gates.familiarity_result_sha256,
        typed_gates.answerability_result_sha256,
        typed_gates.unsupported_result_sha256,
    ) or typed_gates.to_record() != dict(gates_record):
        raise ValueError("closed F2A gates do not match canonical typed probe results")
    expected_lineage = {
        "selection_manifest": selection_manifest_sha256,
        "authorization": result["authorization_sha256"],
        "endpoint_input_sha256": endpoint_input_sha256,
        "endpoint_input_identities_sha256": result[
            "endpoint_input_identities_sha256"
        ],
    }
    if any(metrics_lineage.get(name) != value for name, value in expected_lineage.items()):
        raise ValueError("closed F2A metrics lineage does not bind its endpoint and result")
    return typed_gates.status, _hash(result), result["selection_bundle_hash"]


@dataclass(frozen=True)
class ConfirmatoryPinBundle:
    """Exact model/tokenizer/template pins authorized for confirmatory execution."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    chat_template_sha256: str
    config_sha256: str
    source_pins_sha256: str

    def __post_init__(self) -> None:
        if self.model_id != CONFIRMATORY_MODEL_ID:
            raise ValueError("confirmatory model_id must match the frozen confirmatory config")
        _revision(self.model_revision, "confirmatory model_revision")
        _revision(self.tokenizer_revision, "confirmatory tokenizer_revision")
        _sha256(self.chat_template_sha256, "confirmatory chat_template_sha256")
        _sha256(self.config_sha256, "confirmatory config_sha256")
        _sha256(self.source_pins_sha256, "confirmatory source_pins_sha256")
        if (
            self.model_revision != CONFIRMATORY_MODEL_REVISION
            or self.tokenizer_revision != CONFIRMATORY_MODEL_REVISION
            or self.chat_template_sha256 != CONFIRMATORY_CHAT_TEMPLATE_SHA256
            or self.config_sha256 != _CONFIRMATORY_CONFIG_SHA256
            or self.source_pins_sha256 != _SOURCE_PINS_SHA256
        ):
            raise ValueError("confirmatory pins must match the frozen confirmatory config")

    @property
    def sha256(self) -> str:
        return _hash(self.canonical_payload())

    def canonical_payload(self) -> Mapping[str, str]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_sha256": self.chat_template_sha256,
            "config_sha256": self.config_sha256,
            "source_pins_sha256": self.source_pins_sha256,
        }


@dataclass(frozen=True)
class UnrelatedCapabilityPrompt:
    """A separately sealed H8 prompt, never a relabelled same-string row."""

    prompt_id: str
    task: str
    input_ids: tuple[int, ...]
    assistant_prefix_token_ids: tuple[int, ...]
    rendered_prefix_utf8: bytes
    anchor_positions: Mapping[str, int]
    model_revision: str
    tokenizer_revision: str
    chat_template_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.prompt_id, "unrelated prompt_id")
        if self.task not in {"unrelated_factual", "unrelated_instruction_following"}:
            raise ValueError("unrelated prompt task is not registered")
        input_ids = tuple(self.input_ids)
        prefix_ids = tuple(self.assistant_prefix_token_ids)
        if not input_ids or any(type(value) is not int or value < 0 for value in input_ids):
            raise ValueError("unrelated input_ids must contain nonnegative integers")
        if not prefix_ids or any(type(value) is not int or value < 0 for value in prefix_ids):
            raise ValueError("unrelated assistant prefix token IDs must be nonnegative integers")
        if len(prefix_ids) > len(input_ids) or input_ids[-len(prefix_ids) :] != prefix_ids:
            raise ValueError("unrelated assistant prefix token IDs must be a suffix of input_ids")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "assistant_prefix_token_ids", prefix_ids)
        rendered = bytes(self.rendered_prefix_utf8)
        if not rendered:
            raise ValueError("unrelated rendered prefix bytes must be nonempty")
        object.__setattr__(self, "rendered_prefix_utf8", rendered)
        positions = dict(self.anchor_positions)
        if set(positions) != _ANCHORS or any(
            type(value) is not int or not 0 <= value < len(input_ids)
            for value in positions.values()
        ):
            raise ValueError("unrelated anchor positions are invalid")
        object.__setattr__(self, "anchor_positions", MappingProxyType(positions))
        _revision(self.model_revision, "unrelated model_revision")
        _revision(self.tokenizer_revision, "unrelated tokenizer_revision")
        _sha256(self.chat_template_sha256, "unrelated chat_template_sha256")

    @property
    def input_evidence_sha256(self) -> str:
        return _hash(
            {
                "input_ids": list(self.input_ids),
                "assistant_prefix_token_ids": list(self.assistant_prefix_token_ids),
                "rendered_prefix_utf8_sha256": self.rendered_prefix_utf8_sha256,
                "anchor_positions": dict(self.anchor_positions),
                "model_revision": self.model_revision,
                "tokenizer_revision": self.tokenizer_revision,
                "chat_template_sha256": self.chat_template_sha256,
            }
        )

    @property
    def rendered_prefix_utf8_sha256(self) -> str:
        return hashlib.sha256(self.rendered_prefix_utf8).hexdigest()

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "task": self.task,
            "input_ids": list(self.input_ids),
            "assistant_prefix_token_ids": list(self.assistant_prefix_token_ids),
            "rendered_prefix_utf8_hex": self.rendered_prefix_utf8.hex(),
            "rendered_prefix_utf8_sha256": self.rendered_prefix_utf8_sha256,
            "anchor_positions": dict(self.anchor_positions),
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_sha256": self.chat_template_sha256,
            "input_evidence_sha256": self.input_evidence_sha256,
        }


@dataclass(frozen=True)
class InterventionPrompt:
    """One immutable prompt and its selected activation for intervention."""

    example_id: str
    entity_unit_id: str
    split: str
    exposure: str
    answerability: str
    template_family: str
    domain: str
    target_string: str
    relation_id: str
    entity_type: str
    output_instruction: str
    registry_code: str
    target_familiarity: str
    distractor_familiarity: str
    model_revision: str
    input_ids: tuple[int, ...]
    assistant_prefix_token_ids: tuple[int, ...]
    rendered_prefix_utf8: bytes
    shared_query_suffix_token_ids: tuple[int, ...]
    shared_query_suffix_utf8: bytes
    shared_anchor_offsets: Mapping[str, int]
    tokenizer_revision: str
    chat_template_sha256: str
    anchor_positions: Mapping[str, int]
    activation_layer: int
    activation_anchor: str
    activation: np.ndarray
    activation_sha256: str | None
    activation_manifest_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.example_id, "example_id")
        _nonempty(self.entity_unit_id, "entity_unit_id")
        if self.split not in {"locked_validation", "intervention_test"}:
            raise ValueError("intervention prompts must use locked_validation or intervention_test")
        if self.exposure not in {"high_exposure", "low_exposure"}:
            raise ValueError("intervention exposure is invalid")
        if self.answerability not in {"target_bound", "code_absent"}:
            raise ValueError("intervention answerability is invalid")
        _nonempty(self.template_family, "template_family")
        if self.domain not in _REGISTERED_DOMAINS:
            raise ValueError("intervention domain is not registered")
        for name in (
            "target_string",
            "relation_id",
            "entity_type",
            "output_instruction",
            "registry_code",
        ):
            _nonempty(getattr(self, name), name)
        if self.target_familiarity not in {"screened_real", "matched_synthetic"}:
            raise ValueError("target familiarity is invalid")
        if self.distractor_familiarity not in {"screened_real", "matched_synthetic"}:
            raise ValueError("distractor familiarity is invalid")
        _revision(self.model_revision, "model_revision")
        input_ids = tuple(self.input_ids)
        if not input_ids or any(type(value) is not int or value < 0 for value in input_ids):
            raise ValueError("input_ids must contain nonnegative integers")
        object.__setattr__(self, "input_ids", input_ids)
        prefix_ids = tuple(self.assistant_prefix_token_ids)
        if not prefix_ids or any(
            type(value) is not int or value < 0 for value in prefix_ids
        ):
            raise ValueError("assistant prefix token IDs must be nonnegative integers")
        if len(prefix_ids) > len(input_ids) or input_ids[-len(prefix_ids) :] != prefix_ids:
            raise ValueError("assistant prefix token IDs must be a suffix of input_ids")
        object.__setattr__(self, "assistant_prefix_token_ids", prefix_ids)
        rendered = bytes(self.rendered_prefix_utf8)
        if not rendered:
            raise ValueError("rendered prefix bytes must be nonempty")
        object.__setattr__(self, "rendered_prefix_utf8", rendered)
        suffix_ids = tuple(self.shared_query_suffix_token_ids)
        if not suffix_ids or any(
            type(value) is not int or value < 0 for value in suffix_ids
        ):
            raise ValueError("shared query suffix token IDs are invalid")
        if len(suffix_ids) > len(input_ids) or input_ids[-len(suffix_ids) :] != suffix_ids:
            raise ValueError("shared query suffix token IDs must be a suffix of input_ids")
        object.__setattr__(self, "shared_query_suffix_token_ids", suffix_ids)
        suffix_bytes = bytes(self.shared_query_suffix_utf8)
        if not suffix_bytes or not rendered.endswith(suffix_bytes):
            raise ValueError("shared query suffix bytes must terminate the rendered prompt")
        object.__setattr__(self, "shared_query_suffix_utf8", suffix_bytes)
        offsets = dict(self.shared_anchor_offsets)
        if set(offsets) != _ANCHORS or any(
            type(value) is not int or not 0 <= value < len(suffix_ids)
            for value in offsets.values()
        ):
            raise ValueError("shared anchor offsets are invalid")
        object.__setattr__(self, "shared_anchor_offsets", MappingProxyType(offsets))
        _revision(self.tokenizer_revision, "tokenizer_revision")
        _sha256(self.chat_template_sha256, "chat_template_sha256")
        positions = dict(self.anchor_positions)
        if set(positions) != _ANCHORS:
            raise ValueError("anchor_positions must contain the registered intervention anchors")
        if any(type(value) is not int or not 0 <= value < len(input_ids) for value in positions.values()):
            raise ValueError("anchor position is outside input_ids")
        suffix_start = len(input_ids) - len(suffix_ids)
        if any(
            positions[name] != suffix_start + offsets[name] for name in _ANCHORS
        ):
            raise ValueError("anchor positions do not preserve shared suffix semantics")
        object.__setattr__(self, "anchor_positions", MappingProxyType(positions))
        if type(self.activation_layer) is not int or self.activation_layer < 0:
            raise ValueError("activation_layer must be nonnegative")
        if self.activation_anchor not in _ANCHORS:
            raise ValueError("activation_anchor is not registered")
        array = np.array(self.activation, dtype=np.float64, copy=True, order="C")
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError("activation must be a finite nonempty vector")
        array.setflags(write=False)
        object.__setattr__(self, "activation", array)
        digest = hashlib.sha256(_array_bytes(array)).hexdigest()
        if self.activation_sha256 is None:
            object.__setattr__(self, "activation_sha256", digest)
        elif self.activation_sha256 != digest:
            raise ValueError("activation hash does not match activation data")
        _sha256(self.activation_manifest_sha256, "activation_manifest_sha256")

    @property
    def assistant_prefix_sha256(self) -> str:
        return _prefix_evidence_sha256(
            self.rendered_prefix_utf8,
            self.tokenizer_revision,
            self.chat_template_sha256,
        )

    @property
    def input_evidence_sha256(self) -> str:
        return _hash(
            {
                "example_id": self.example_id,
                "entity_unit_id": self.entity_unit_id,
                "target_string": self.target_string,
                "relation_id": self.relation_id,
                "entity_type": self.entity_type,
                "output_instruction": self.output_instruction,
                "input_ids": list(self.input_ids),
                "assistant_prefix_sha256": self.assistant_prefix_sha256,
                "anchor_positions": dict(self.anchor_positions),
                "model_revision": self.model_revision,
            }
        )

    @property
    def causal_invariants_sha256(self) -> str:
        return _hash(
            {
                "entity_unit_id": self.entity_unit_id,
                "split": self.split,
                "answerability": self.answerability,
                "template_family": self.template_family,
                "domain": self.domain,
                "target_string": self.target_string,
                "relation_id": self.relation_id,
                "entity_type": self.entity_type,
                "output_instruction": self.output_instruction,
                "registry_code": self.registry_code,
                "target_familiarity": self.target_familiarity,
                "distractor_familiarity": self.distractor_familiarity,
                "shared_query_suffix_token_ids": list(
                    self.shared_query_suffix_token_ids
                ),
                "shared_query_suffix_utf8_sha256": hashlib.sha256(
                    self.shared_query_suffix_utf8
                ).hexdigest(),
                "shared_anchor_offsets": dict(self.shared_anchor_offsets),
                "model_revision": self.model_revision,
                "tokenizer_revision": self.tokenizer_revision,
                "chat_template_sha256": self.chat_template_sha256,
                "activation_layer": self.activation_layer,
                "activation_anchor": self.activation_anchor,
            }
        )

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "example_id": self.example_id,
            "entity_unit_id": self.entity_unit_id,
            "split": self.split,
            "exposure": self.exposure,
            "answerability": self.answerability,
            "template_family": self.template_family,
            "domain": self.domain,
            "target_string": self.target_string,
            "relation_id": self.relation_id,
            "entity_type": self.entity_type,
            "output_instruction": self.output_instruction,
            "registry_code": self.registry_code,
            "target_familiarity": self.target_familiarity,
            "distractor_familiarity": self.distractor_familiarity,
            "model_revision": self.model_revision,
            "input_ids": list(self.input_ids),
            "assistant_prefix_token_ids": list(self.assistant_prefix_token_ids),
            "rendered_prefix_utf8_hex": self.rendered_prefix_utf8.hex(),
            "shared_query_suffix_token_ids": list(
                self.shared_query_suffix_token_ids
            ),
            "shared_query_suffix_utf8_hex": self.shared_query_suffix_utf8.hex(),
            "shared_anchor_offsets": dict(self.shared_anchor_offsets),
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_sha256": self.chat_template_sha256,
            "assistant_prefix_sha256": self.assistant_prefix_sha256,
            "input_evidence_sha256": self.input_evidence_sha256,
            "causal_invariants_sha256": self.causal_invariants_sha256,
            "anchor_positions": dict(self.anchor_positions),
            "activation_layer": self.activation_layer,
            "activation_anchor": self.activation_anchor,
            "activation_sha256": self.activation_sha256,
            "activation_manifest_sha256": self.activation_manifest_sha256,
        }


@dataclass(frozen=True)
class InterventionPair:
    high: InterventionPrompt
    low: InterventionPrompt

    def __post_init__(self) -> None:
        if not isinstance(self.high, InterventionPrompt) or not isinstance(
            self.low, InterventionPrompt
        ):
            raise ValueError("intervention pair requires typed prompts")
        if self.high.exposure != "high_exposure" or self.low.exposure != "low_exposure":
            raise ValueError("intervention pair must be ordered high then low exposure")
        if self.high.entity_unit_id != self.low.entity_unit_id:
            raise ValueError("intervention pair must use the same entity unit")
        if self.high.split != self.low.split:
            raise ValueError("intervention pair cannot cross split boundaries")
        if self.high.answerability != self.low.answerability:
            raise ValueError("intervention pair must preserve answerability")
        if self.high.template_family != self.low.template_family:
            raise ValueError("intervention pair must preserve template family")
        if self.high.causal_invariants_sha256 != self.low.causal_invariants_sha256:
            raise ValueError("intervention pair must preserve every registered causal invariant")
        if self.high.domain != self.low.domain:
            raise ValueError("intervention pair must preserve domain")
        if (
            self.high.activation_layer != self.low.activation_layer
            or self.high.activation_anchor != self.low.activation_anchor
            or self.high.activation.shape != self.low.activation.shape
        ):
            raise ValueError("intervention pair activations must use one registered site")


@dataclass(frozen=True)
class PatchSpec:
    layer: int
    anchor: str
    direction: str
    mode: str
    alpha: float
    model_revision: str
    activation_manifest_sha256: str
    source_layer: int | None = None
    source_anchor: str | None = None
    control_name: str = "primary"

    def __post_init__(self) -> None:
        if type(self.layer) is not int or self.layer < 0:
            raise ValueError("patch layer must be nonnegative")
        if self.anchor not in _ANCHORS:
            raise ValueError("patch anchor is not registered")
        source_layer = self.layer if self.source_layer is None else self.source_layer
        source_anchor = self.anchor if self.source_anchor is None else self.source_anchor
        if type(source_layer) is not int or source_layer < 0:
            raise ValueError("source layer must be nonnegative")
        if source_anchor not in _ANCHORS:
            raise ValueError("source anchor is not registered")
        object.__setattr__(self, "source_layer", source_layer)
        object.__setattr__(self, "source_anchor", source_anchor)
        if self.control_name == "primary":
            if (
                self.anchor != "target_intro_end"
                or source_anchor != self.anchor
                or source_layer != self.layer
            ):
                raise ValueError("primary full replacement requires the selected target site")
        elif self.control_name == "wrong_layer":
            if (
                source_anchor != "target_intro_end"
                or self.anchor != source_anchor
                or self.layer == source_layer
            ):
                raise ValueError("wrong_layer control must change only the patch layer")
        elif self.control_name == "wrong_anchor":
            if (
                source_anchor != "target_intro_end"
                or self.anchor == source_anchor
                or self.layer != source_layer
            ):
                raise ValueError("wrong_anchor control must change only the patch anchor")
        else:
            raise ValueError("patch control_name is invalid")
        if self.direction not in _DIRECTIONS:
            raise ValueError("patch direction is invalid")
        if self.mode != "full_replacement":
            raise ValueError("primary intervention mode must be full_replacement")
        if not np.isclose(self.alpha, 1.0):
            raise ValueError("full replacement alpha must equal one")
        _revision(self.model_revision, "model_revision")
        _sha256(self.activation_manifest_sha256, "activation_manifest_sha256")


@dataclass(frozen=True)
class PatchAudit:
    """Adapter-observed evidence for one prefill-only intervention."""

    modified_sites: tuple[tuple[int, int], ...]
    decode_hook_calls: int
    input_ids: tuple[int, ...]
    assistant_prefix_token_ids: tuple[int, ...]
    rendered_prefix_utf8_sha256: str
    tokenizer_revision: str
    chat_template_sha256: str
    applied_intervention_sha256: str
    applied_vector_sha256: str
    replacement_sha256: str
    source_evidence_sha256: str
    source_activation_sha256: str | None
    source_example_id: str | None
    destination_example_id: str
    destination_entity_unit_id: str | None
    destination_evidence_sha256: str

    def __post_init__(self) -> None:
        sites = tuple(tuple(site) for site in self.modified_sites)
        if any(
            len(site) != 2
            or type(site[0]) is not int
            or type(site[1]) is not int
            or site[0] < 0
            or site[1] < 0
            for site in sites
        ):
            raise ValueError("modified sites must be nonnegative layer-position pairs")
        if len(set(sites)) != len(sites):
            raise ValueError("modified sites must be unique")
        object.__setattr__(self, "modified_sites", sites)
        if type(self.decode_hook_calls) is not int or self.decode_hook_calls < 0:
            raise ValueError("decode hook calls must be a nonnegative integer")
        input_ids = tuple(self.input_ids)
        prefix_ids = tuple(self.assistant_prefix_token_ids)
        if not input_ids or any(type(value) is not int or value < 0 for value in input_ids):
            raise ValueError("audit input IDs are invalid")
        if not prefix_ids or any(
            type(value) is not int or value < 0 for value in prefix_ids
        ):
            raise ValueError("audit prefix token IDs are invalid")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "assistant_prefix_token_ids", prefix_ids)
        for name in (
            "rendered_prefix_utf8_sha256",
            "applied_intervention_sha256",
            "applied_vector_sha256",
            "replacement_sha256",
            "source_evidence_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.source_activation_sha256 is not None:
            _sha256(self.source_activation_sha256, "audit source_activation_sha256")
        if self.source_example_id is not None:
            _nonempty(self.source_example_id, "audit source_example_id")
        _nonempty(self.destination_example_id, "audit destination_example_id")
        if self.destination_entity_unit_id is not None:
            _nonempty(
                self.destination_entity_unit_id,
                "audit destination_entity_unit_id",
            )
        _sha256(
            self.destination_evidence_sha256,
            "audit destination_evidence_sha256",
        )
        _revision(self.tokenizer_revision, "audit tokenizer_revision")
        _sha256(self.chat_template_sha256, "audit chat_template_sha256")

    @property
    def sha256(self) -> str:
        return _hash(self.canonical_payload())

    @property
    def audit_sha256(self) -> str:
        return self.sha256

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "modified_sites": [list(site) for site in self.modified_sites],
            "decode_hook_calls": self.decode_hook_calls,
            "input_ids": list(self.input_ids),
            "assistant_prefix_token_ids": list(self.assistant_prefix_token_ids),
            "rendered_prefix_utf8_sha256": self.rendered_prefix_utf8_sha256,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_sha256": self.chat_template_sha256,
            "applied_intervention_sha256": self.applied_intervention_sha256,
            "applied_vector_sha256": self.applied_vector_sha256,
            "replacement_sha256": self.replacement_sha256,
            "source_evidence_sha256": self.source_evidence_sha256,
            "source_activation_sha256": self.source_activation_sha256,
            "source_example_id": self.source_example_id,
            "destination_example_id": self.destination_example_id,
            "destination_entity_unit_id": self.destination_entity_unit_id,
            "destination_evidence_sha256": self.destination_evidence_sha256,
        }


@dataclass(frozen=True)
class _ScoringExample:
    example_id: str
    entity_unit_id: str
    template_family: str
    target_familiarity: str
    distractor_familiarity: str
    answerability: str
    registry_code: str
    exposure: str
    block: str = "same_string"


@dataclass(frozen=True)
class RawInterventionOutcome:
    """One executed observation with labels derived only from decoded text."""

    source_example_id: str
    destination_example_id: str
    entity_unit_id: str
    domain: str
    answerability: str
    template_family: str
    direction: str
    registry_code: str
    target_familiarity: str
    distractor_familiarity: str
    exposure: str
    applied_intervention: AppliedIntervention
    audit: PatchAudit
    provenance: Mapping[str, Any]
    decoded: Mapping[str, Any]
    baseline_answer_attempt: int = field(init=False)
    patched_answer_attempt: int = field(init=False)
    baseline_correct: int = field(init=False)
    patched_correct: int = field(init=False)
    baseline_refusal: int = field(init=False)
    patched_refusal: int = field(init=False)
    baseline_invalid_format: int = field(init=False)
    patched_invalid_format: int = field(init=False)
    familiarity_readout_change: float = field(init=False)
    answerability_readout_change: float = field(init=False)
    entity_type_readout_change: float = field(init=False)
    generic_confidence_change: float = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_example_id",
            "destination_example_id",
            "entity_unit_id",
            "domain",
            "template_family",
            "registry_code",
            "target_familiarity",
            "distractor_familiarity",
            "exposure",
        ):
            _nonempty(getattr(self, name), name)
        if self.answerability not in {"target_bound", "code_absent"}:
            raise ValueError("raw outcome answerability is invalid")
        if self.direction not in _DIRECTIONS:
            raise ValueError("raw outcome direction is invalid")
        if self.exposure not in {"high_exposure", "low_exposure"}:
            raise ValueError("raw outcome exposure is invalid")
        if not isinstance(self.applied_intervention, AppliedIntervention):
            raise ValueError("raw outcome requires a typed applied intervention")
        if not isinstance(self.audit, PatchAudit):
            raise ValueError("raw outcome requires typed patch audit evidence")
        object.__setattr__(self, "provenance", _deep_freeze(self.provenance))
        decoded = _parse_decoded_patch(self.decoded)
        object.__setattr__(self, "decoded", _deep_freeze(decoded))
        example = _ScoringExample(
            example_id=self.destination_example_id,
            entity_unit_id=self.entity_unit_id,
            template_family=self.template_family,
            target_familiarity=self.target_familiarity,
            distractor_familiarity=self.distractor_familiarity,
            answerability=self.answerability,
            registry_code=self.registry_code,
            exposure=self.exposure,
        )
        baseline = score_response(
            example,
            decoded["baseline_text"],
            registered_codes=(self.registry_code,),
            truncated=decoded["baseline_truncated"],
            infrastructure_marked=decoded["baseline_infrastructure_marked"],
        )
        patched = score_response(
            example,
            decoded["patched_text"],
            registered_codes=(self.registry_code,),
            truncated=decoded["patched_truncated"],
            infrastructure_marked=decoded["patched_infrastructure_marked"],
        )
        baseline_readouts = ReadoutSnapshot.from_mapping(
            decoded["baseline_readouts"], "baseline_readouts"
        )
        patched_readouts = ReadoutSnapshot.from_mapping(
            decoded["patched_readouts"], "patched_readouts"
        )
        derived = {
            "baseline_answer_attempt": baseline.answer_attempt,
            "patched_answer_attempt": patched.answer_attempt,
            "baseline_correct": int(baseline.outcome is OutcomeClass.EXACT_TARGET_CODE),
            "patched_correct": int(patched.outcome is OutcomeClass.EXACT_TARGET_CODE),
            "baseline_refusal": int(is_refusal(baseline.raw_output or "")),
            "patched_refusal": int(is_refusal(patched.raw_output or "")),
            "baseline_invalid_format": int(not baseline.valid_format),
            "patched_invalid_format": int(not patched.valid_format),
            "familiarity_readout_change": patched_readouts.familiarity
            - baseline_readouts.familiarity,
            "answerability_readout_change": patched_readouts.answerability
            - baseline_readouts.answerability,
            "entity_type_readout_change": patched_readouts.entity_type
            - baseline_readouts.entity_type,
            "generic_confidence_change": patched_readouts.generic_confidence
            - baseline_readouts.generic_confidence,
        }
        for name, value in derived.items():
            object.__setattr__(self, name, value)

    @property
    def control_name(self) -> str:
        return self.applied_intervention.control_name

    @property
    def source_activation_sha256(self) -> str | None:
        return self.applied_intervention.source_activation_sha256

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "source_example_id": self.source_example_id,
            "destination_example_id": self.destination_example_id,
            "entity_unit_id": self.entity_unit_id,
            "domain": self.domain,
            "answerability": self.answerability,
            "template_family": self.template_family,
            "control_name": self.control_name,
            "direction": self.direction,
            "registry_code": self.registry_code,
            "target_familiarity": self.target_familiarity,
            "distractor_familiarity": self.distractor_familiarity,
            "exposure": self.exposure,
            "baseline_answer_attempt": self.baseline_answer_attempt,
            "patched_answer_attempt": self.patched_answer_attempt,
            "baseline_correct": self.baseline_correct,
            "patched_correct": self.patched_correct,
            "baseline_refusal": self.baseline_refusal,
            "patched_refusal": self.patched_refusal,
            "baseline_invalid_format": self.baseline_invalid_format,
            "patched_invalid_format": self.patched_invalid_format,
            "source_activation_sha256": self.source_activation_sha256,
            "applied_intervention": self.applied_intervention.canonical_payload(),
            "applied_intervention_sha256": self.applied_intervention.sha256,
            "familiarity_readout_change": self.familiarity_readout_change,
            "answerability_readout_change": self.answerability_readout_change,
            "entity_type_readout_change": self.entity_type_readout_change,
            "generic_confidence_change": self.generic_confidence_change,
            "audit": self.audit.canonical_payload(),
            "provenance": _thaw_frozen(self.provenance),
            "decoded": _thaw_frozen(self.decoded),
        }


@dataclass(frozen=True)
class RawUnrelatedOutcome:
    """One raw H8 observation from a separately sealed unrelated prompt."""

    prompt_id: str
    task: str
    input_evidence_sha256: str
    applied_intervention: AppliedIntervention
    audit: PatchAudit
    provenance: Mapping[str, Any]
    decoded: Mapping[str, Any]
    baseline_refusal: int = field(init=False)
    patched_refusal: int = field(init=False)
    baseline_invalid_format: int = field(init=False)
    patched_invalid_format: int = field(init=False)
    generic_confidence_change: float = field(init=False)

    def __post_init__(self) -> None:
        _nonempty(self.prompt_id, "unrelated outcome prompt_id")
        if self.task not in {"unrelated_factual", "unrelated_instruction_following"}:
            raise ValueError("unrelated outcome task is not registered")
        _sha256(self.input_evidence_sha256, "unrelated input_evidence_sha256")
        if not isinstance(self.applied_intervention, AppliedIntervention):
            raise ValueError("unrelated outcome requires the selected intervention")
        if not isinstance(self.audit, PatchAudit):
            raise ValueError("unrelated outcome requires a typed patch audit")
        object.__setattr__(self, "provenance", _deep_freeze(self.provenance))
        decoded = _parse_decoded_patch(self.decoded)
        object.__setattr__(self, "decoded", _deep_freeze(decoded))
        baseline_readouts = ReadoutSnapshot.from_mapping(
            decoded["baseline_readouts"], "baseline_readouts"
        )
        patched_readouts = ReadoutSnapshot.from_mapping(
            decoded["patched_readouts"], "patched_readouts"
        )
        object.__setattr__(self, "baseline_refusal", int(is_refusal(decoded["baseline_text"] or "")))
        object.__setattr__(self, "patched_refusal", int(is_refusal(decoded["patched_text"] or "")))
        object.__setattr__(self, "baseline_invalid_format", _unrelated_invalid(decoded, "baseline"))
        object.__setattr__(self, "patched_invalid_format", _unrelated_invalid(decoded, "patched"))
        object.__setattr__(
            self,
            "generic_confidence_change",
            patched_readouts.generic_confidence - baseline_readouts.generic_confidence,
        )

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "task": self.task,
            "baseline_refusal": self.baseline_refusal,
            "patched_refusal": self.patched_refusal,
            "baseline_invalid_format": self.baseline_invalid_format,
            "patched_invalid_format": self.patched_invalid_format,
            "input_evidence_sha256": self.input_evidence_sha256,
            "applied_intervention": self.applied_intervention.canonical_payload(),
            "applied_intervention_sha256": self.applied_intervention.sha256,
            "generic_confidence_change": self.generic_confidence_change,
            "audit": self.audit.canonical_payload(),
            "provenance": _thaw_frozen(self.provenance),
            "decoded": _thaw_frozen(self.decoded),
        }


@dataclass(frozen=True)
class PatchRow:
    source_example_id: str
    destination_example_id: str
    entity_unit_id: str
    direction: str
    layer: int
    position: int
    source_activation_sha256: str
    assistant_prefix_sha256: str
    decoded: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "decoded", _deep_freeze(self.decoded))


@dataclass(frozen=True)
class PatchOutcome:
    rows: tuple[PatchRow, ...]
    changed_positions: frozenset[tuple[int, int]]
    decode_hook_calls: int
    spec_sha256: str


@dataclass(frozen=True)
class InterventionMetrics:
    """Deeply frozen summary derived from validation or raw test outcomes."""

    high_to_low_effect: float
    high_to_low_interval: tuple[float, float]
    low_to_high_effect: float
    low_to_high_interval: tuple[float, float]
    control_effects: Mapping[str, tuple[float, float]]
    target_bound_accuracy_change: float
    unrelated_refusal_change: float
    unrelated_invalid_format_change: float
    familiarity_readout_effect: float
    answerability_max_abs_change: float
    entity_type_max_abs_change: float
    generic_confidence_max_abs_change: float
    readout_constraints: ReadoutConstraints
    observed_domains: tuple[str, ...]
    passing_domains: tuple[str, ...]
    completed_fraction: float
    bootstrap_summary: Mapping[str, Any] = field(default_factory=dict)
    unrelated_refusal_change_by_direction: Mapping[str, float] = field(
        default_factory=dict
    )
    unrelated_invalid_format_change_by_direction: Mapping[str, float] = field(
        default_factory=dict
    )
    target_bound_accuracy_change_by_direction: Mapping[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        high_interval = tuple(self.high_to_low_interval)
        low_interval = tuple(self.low_to_high_interval)
        object.__setattr__(self, "high_to_low_interval", high_interval)
        object.__setattr__(self, "low_to_high_interval", low_interval)
        effects: dict[str, tuple[float, float]] = {}
        if not isinstance(self.control_effects, Mapping):
            raise ValueError("control effects must be a mapping")
        for name, values in self.control_effects.items():
            if name not in REQUIRED_CAUSAL_CONTROLS:
                raise ValueError("control effects contain an unregistered control")
            pair = tuple(values)
            if len(pair) != 2:
                raise ValueError("each control requires two directional effects")
            effects[name] = pair
        object.__setattr__(
            self,
            "control_effects",
            MappingProxyType(dict(sorted(effects.items()))),
        )
        numeric = (
            self.high_to_low_effect,
            self.low_to_high_effect,
            self.target_bound_accuracy_change,
            self.unrelated_refusal_change,
            self.unrelated_invalid_format_change,
            self.familiarity_readout_effect,
            self.answerability_max_abs_change,
            self.entity_type_max_abs_change,
            self.generic_confidence_max_abs_change,
            self.completed_fraction,
            *high_interval,
            *low_interval,
            *(value for pair in effects.values() for value in pair),
        )
        if any(type(value) not in {int, float} or not np.isfinite(value) for value in numeric):
            raise ValueError("intervention metrics must be finite")
        for interval in (high_interval, low_interval):
            if len(interval) != 2 or interval[0] > interval[1]:
                raise ValueError("intervention interval is invalid")
        if not isinstance(self.readout_constraints, ReadoutConstraints):
            raise ValueError("metrics require validation-frozen readout constraints")
        observed_domains = tuple(sorted(set(self.observed_domains)))
        passing_domains = tuple(sorted(set(self.passing_domains)))
        if any(domain not in _REGISTERED_DOMAINS for domain in observed_domains):
            raise ValueError("observed domains contain an unregistered domain")
        if any(domain not in observed_domains for domain in passing_domains):
            raise ValueError("passing domains must be observed")
        object.__setattr__(self, "observed_domains", observed_domains)
        object.__setattr__(self, "passing_domains", passing_domains)
        object.__setattr__(self, "bootstrap_summary", _deep_freeze(self.bootstrap_summary))
        for name in (
            "unrelated_refusal_change_by_direction",
            "unrelated_invalid_format_change_by_direction",
            "target_bound_accuracy_change_by_direction",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a directional mapping")
            directional = dict(value)
            if directional and set(directional) != _DIRECTIONS:
                raise ValueError(f"{name} must contain both registered directions")
            if any(
                type(change) not in {int, float} or not np.isfinite(change)
                for change in directional.values()
            ):
                raise ValueError(f"{name} must contain finite numeric changes")
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(sorted(directional.items()))),
            )
        if not 0.0 <= self.completed_fraction <= 1.0:
            raise ValueError("completed_fraction must be in [0, 1]")

    @property
    def h7_passed(self) -> bool:
        controls_pass = set(self.control_effects) == set(REQUIRED_CAUSAL_CONTROLS)
        directional = self.high_to_low_interval[0] > 0 and self.low_to_high_interval[0] > 0
        average = (self.high_to_low_effect + self.low_to_high_effect) / 2.0
        strictly_beats_controls = all(
            self.high_to_low_effect > effects[0]
            and self.low_to_high_effect > effects[1]
            for effects in self.control_effects.values()
        )
        registered_margin_pass = all(
            self.high_to_low_effect - self.control_effects[name][0] >= 0.02
            and self.low_to_high_effect - self.control_effects[name][1] >= 0.02
            for name in ("norm_matched_random", "cross_entity")
            if name in self.control_effects
        ) and all(
            name in self.control_effects
            for name in ("norm_matched_random", "cross_entity")
        )
        bootstrap = self.bootstrap_summary
        bootstrap_pass = _h7_bootstrap_evidence_passes(
            bootstrap,
            effects={
                "high_to_low": self.high_to_low_effect,
                "low_to_high": self.low_to_high_effect,
            },
            intervals={
                "high_to_low": self.high_to_low_interval,
                "low_to_high": self.low_to_high_interval,
            },
        )
        constraints = self.readout_constraints
        readouts_pass = (
            self.familiarity_readout_effect >= constraints.familiarity_min_effect
            and self.answerability_max_abs_change
            <= constraints.answerability_max_abs_change
            and self.entity_type_max_abs_change <= constraints.entity_type_max_abs_change
            and self.generic_confidence_max_abs_change
            <= constraints.generic_confidence_max_abs_change
        )
        return (
            self.completed_fraction >= 0.95
            and controls_pass
            and directional
            and bootstrap_pass
            and average >= 0.05
            and strictly_beats_controls
            and registered_margin_pass
            and set(self.observed_domains) == _REGISTERED_DOMAINS
            and len(self.passing_domains) >= 3
            and readouts_pass
        )

    @property
    def h8_passed(self) -> bool:
        refusal = self.unrelated_refusal_change_by_direction
        invalid = self.unrelated_invalid_format_change_by_direction
        target_accuracy = self.target_bound_accuracy_change_by_direction
        return (
            self.completed_fraction >= 0.95
            and set(target_accuracy) == _DIRECTIONS
            and all(
                target_accuracy[direction] >= -0.05 for direction in _DIRECTIONS
            )
            and set(refusal) == _DIRECTIONS
            and set(invalid) == _DIRECTIONS
            and all(abs(refusal[direction]) <= 0.03 for direction in _DIRECTIONS)
            and all(abs(invalid[direction]) <= 0.03 for direction in _DIRECTIONS)
        )

    @property
    def average_effect(self) -> float:
        return (self.high_to_low_effect + self.low_to_high_effect) / 2.0

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "high_to_low_effect": self.high_to_low_effect,
            "high_to_low_interval": list(self.high_to_low_interval),
            "low_to_high_effect": self.low_to_high_effect,
            "low_to_high_interval": list(self.low_to_high_interval),
            "control_effects": {
                name: list(values) for name, values in self.control_effects.items()
            },
            "target_bound_accuracy_change": self.target_bound_accuracy_change,
            "unrelated_refusal_change": self.unrelated_refusal_change,
            "unrelated_invalid_format_change": self.unrelated_invalid_format_change,
            "familiarity_readout_effect": self.familiarity_readout_effect,
            "answerability_max_abs_change": self.answerability_max_abs_change,
            "entity_type_max_abs_change": self.entity_type_max_abs_change,
            "generic_confidence_max_abs_change": self.generic_confidence_max_abs_change,
            "readout_constraints": self.readout_constraints.canonical_payload(),
            "observed_domains": list(self.observed_domains),
            "passing_domains": list(self.passing_domains),
            "completed_fraction": self.completed_fraction,
            "bootstrap_summary": _thaw_frozen(self.bootstrap_summary),
            "unrelated_refusal_change_by_direction": dict(
                self.unrelated_refusal_change_by_direction
            ),
            "unrelated_invalid_format_change_by_direction": dict(
                self.unrelated_invalid_format_change_by_direction
            ),
            "target_bound_accuracy_change_by_direction": dict(
                self.target_bound_accuracy_change_by_direction
            ),
        }


def _h7_bootstrap_evidence_passes(
    value: Mapping[str, Any],
    *,
    effects: Mapping[str, float],
    intervals: Mapping[str, tuple[float, float]],
) -> bool:
    if not isinstance(value, Mapping):
        return False
    alpha = value.get("alpha")
    directions = value.get("directions")
    requested_draws = value.get("requested_draws")
    valid_draws = value.get("valid_draws")
    discarded_draws = value.get("discarded_draws")
    if (
        set(value)
        != {
            "method",
            "seed",
            "replicates",
            "requested_draws",
            "valid_draws",
            "discarded_draws",
            "resampling_unit",
            "alpha",
            "directions",
        }
        or value.get("method") != "crossed_entity_unit_template_family_bootstrap"
        or value.get("seed") != _H7_BOOTSTRAP_SEED
        or value.get("replicates") != _H7_BOOTSTRAP_REPLICATES
        or requested_draws != _H7_BOOTSTRAP_REPLICATES
        or type(valid_draws) is not int
        or valid_draws <= 0
        or type(discarded_draws) is not int
        or discarded_draws < 0
        or valid_draws + discarded_draws != requested_draws
        or tuple(value.get("resampling_unit", ()))
        != ("entity_unit_id", "template_family")
        or type(alpha) not in {int, float}
        or not np.isclose(alpha, _H7_ALPHA)
        or not isinstance(directions, Mapping)
        or set(directions) != _DIRECTIONS
    ):
        return False
    expected_fields = {
        "point_estimate",
        "raw_interval",
        "raw_p",
        "entities",
        "template_families",
        "holm_interval",
        "holm_adjusted_p",
    }
    for direction, evidence in directions.items():
        if not isinstance(evidence, Mapping) or set(evidence) != expected_fields:
            return False
        point = evidence.get("point_estimate")
        raw_p = evidence.get("raw_p")
        adjusted_p = evidence.get("holm_adjusted_p")
        raw_interval = evidence.get("raw_interval")
        interval = evidence.get("holm_interval")
        if (
            type(point) not in {int, float}
            or not np.isfinite(point)
            or not np.isclose(point, effects[direction])
            or type(raw_p) not in {int, float}
            or not np.isfinite(raw_p)
            or not 0.0 <= raw_p <= 1.0
            or type(adjusted_p) not in {int, float}
            or not np.isfinite(adjusted_p)
            or not 0.0 <= adjusted_p <= 1.0
            or adjusted_p > _H7_ALPHA
            or not isinstance(raw_interval, (tuple, list))
            or len(raw_interval) != 2
            or any(
                type(bound) not in {int, float} or not np.isfinite(bound)
                for bound in raw_interval
            )
            or raw_interval[0] > raw_interval[1]
            or not isinstance(interval, (tuple, list))
            or len(interval) != 2
            or any(type(bound) not in {int, float} or not np.isfinite(bound) for bound in interval)
            or interval[0] <= 0
            or interval[0] > interval[1]
            or not np.allclose(interval, intervals[direction])
            or not isinstance(evidence.get("entities"), (tuple, list))
            or not evidence["entities"]
            or any(not isinstance(item, str) or not item for item in evidence["entities"])
            or not isinstance(evidence.get("template_families"), (tuple, list))
            or not evidence["template_families"]
            or any(
                not isinstance(item, str) or not item
                for item in evidence["template_families"]
            )
        ):
            return False
    return True


@dataclass(frozen=True)
class InterventionCandidate:
    layer: int
    anchor: str
    method: str
    alpha: float
    source_split: str
    metrics: InterventionMetrics
    direction_sha256: str

    def __post_init__(self) -> None:
        if type(self.layer) is not int or self.layer < 0:
            raise ValueError("candidate layer must be nonnegative")
        if self.anchor != "target_intro_end":
            raise ValueError("candidate anchor must be target_intro_end")
        if self.method not in {"full_replacement", "contrastive_direction"}:
            raise ValueError("candidate method is invalid")
        if self.method == "full_replacement" and not np.isclose(self.alpha, 1.0):
            raise ValueError("full replacement alpha must equal one")
        if self.method == "contrastive_direction" and self.alpha not in _ALPHA_GRID:
            raise ValueError("direction alpha is outside the registered grid")
        if self.source_split != "locked_validation":
            raise ValueError("intervention candidates must use locked_validation")
        if not isinstance(self.metrics, InterventionMetrics):
            raise ValueError("candidate metrics must be typed")
        _sha256(self.direction_sha256, "direction_sha256")

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "layer": self.layer,
            "anchor": self.anchor,
            "method": self.method,
            "alpha": self.alpha,
            "source_split": self.source_split,
            "metrics": self.metrics.canonical_payload(),
            "direction_sha256": self.direction_sha256,
        }


@dataclass(frozen=True)
class InterventionSelection:
    layer: int
    anchor: str
    method: str
    alpha: float
    source_split: str
    direction_sha256: str
    preregistration_sha256: str
    probe_selection_sha256: str
    confirmatory_pins_sha256: str
    f1_evidence_sha256: str
    f2a_evidence_sha256: str
    f1_result_sha256: str
    f2a_result_sha256: str
    f1_artifact_sha256: str
    f2a_artifact_sha256: str
    f1_manifest_sha256: str
    f2a_manifest_sha256: str
    control_source: ValidationControlSource
    control_source_artifact_sha256: str
    validation_metrics: InterventionMetrics
    candidate_count: int
    sha256: str

    def __post_init__(self) -> None:
        _sha256(self.preregistration_sha256, "preregistration_sha256")
        _sha256(self.probe_selection_sha256, "probe_selection_sha256")
        _sha256(self.confirmatory_pins_sha256, "confirmatory_pins_sha256")
        for name in (
            "f1_evidence_sha256",
            "f2a_evidence_sha256",
            "f1_result_sha256",
            "f2a_result_sha256",
            "f1_artifact_sha256",
            "f2a_artifact_sha256",
            "f1_manifest_sha256",
            "f2a_manifest_sha256",
        ):
            _sha256(getattr(self, name), name)
        _sha256(self.direction_sha256, "direction_sha256")
        _sha256(self.sha256, "selection sha256")
        if not isinstance(self.control_source, ValidationControlSource):
            raise ValueError("selection requires a verified validation control source")
        _sha256(
            self.control_source_artifact_sha256,
            "control_source_artifact_sha256",
        )
        if (
            self.control_source_artifact_sha256
            != self.control_source.source_artifact_sha256
            or self.control_source.source_split != "locked_validation"
            or self.control_source.preregistration_sha256
            != self.preregistration_sha256
            or self.control_source.probe_selection_sha256
            != self.probe_selection_sha256
        ):
            raise ValueError("selection control-source lineage does not match")
        if self.source_split != "locked_validation":
            raise ValueError("intervention selection must derive from locked_validation")
        if type(self.candidate_count) is not int or self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if self.sha256 != _selection_sha256(self):
            raise ValueError("intervention selection hash does not match canonical content")


@dataclass(frozen=True)
class InterventionTestResult:
    selection_sha256: str
    preregistration_sha256: str
    example_ids: tuple[str, ...]
    metrics: InterventionMetrics
    h7_passed: bool
    h8_passed: bool
    refit_performed: bool
    result_sha256: str

    def __post_init__(self) -> None:
        if self.refit_performed:
            raise ValueError("confirmatory intervention evaluation cannot refit")
        _sha256(self.selection_sha256, "selection_sha256")
        _sha256(self.preregistration_sha256, "preregistration_sha256")
        _sha256(self.result_sha256, "result_sha256")
        if tuple(sorted(set(self.example_ids))) != self.example_ids:
            raise ValueError("result example IDs must be unique and sorted")
        if self.h7_passed != self.metrics.h7_passed or self.h8_passed != self.metrics.h8_passed:
            raise ValueError("stored intervention gates do not match canonical metrics")
        if self.result_sha256 != _result_sha256(self):
            raise ValueError("intervention result hash does not match canonical content")


def _pair_direction(
    pair: InterventionPair, direction: str
) -> tuple[InterventionPrompt, InterventionPrompt]:
    if direction == "high_to_low":
        return pair.high, pair.low
    if direction == "low_to_high":
        return pair.low, pair.high
    raise ValueError("intervention direction is invalid")


def _applied_intervention(
    pair: InterventionPair,
    *,
    control_name: str,
    direction: str,
    layer: int,
    anchor: str,
    activation_manifest_sha256: str,
    cross_entity_pair: InterventionPair | None = None,
    control_source: ValidationControlSource | None = None,
) -> AppliedIntervention:
    source, destination = _pair_direction(pair, direction)
    base_direction = source.activation - destination.activation
    seed = int(
        _hash(
            {
                "entity_unit_id": pair.high.entity_unit_id,
                "direction": direction,
                "activation_manifest_sha256": activation_manifest_sha256,
            }
        )[:16],
        16,
    )
    cross_source = None
    if cross_entity_pair is not None:
        cross_source, _ = _pair_direction(cross_entity_pair, direction)
        cross_direction = cross_source.activation - destination.activation
    else:
        cross_direction = base_direction
    actual_layer = layer
    actual_anchor = anchor
    apply_patch = True
    source_prompt = source
    source_kind = "paired_full_replacement"
    vector = base_direction
    replacement = source.activation
    control_source_split = None
    control_source_artifact_sha256 = None
    control_source_component_sha256 = None
    if control_name == "no_intervention":
        apply_patch = False
        source_prompt = destination
        source_kind = "sealed_no_op_destination"
        vector = np.zeros_like(base_direction)
        replacement = destination.activation
    elif control_name == "wrong_layer":
        actual_layer = layer - 1 if layer > 0 else layer + 1
        source_kind = "paired_full_replacement_wrong_layer"
    elif control_name == "wrong_anchor":
        actual_anchor = (
            "user_prompt_end" if anchor == "target_intro_end" else "target_intro_end"
        )
        source_kind = "paired_full_replacement_wrong_anchor"
    elif control_name == "cross_entity":
        if cross_source is None or cross_source.entity_unit_id == source.entity_unit_id:
            raise ValueError("cross_entity requires a distinct sealed source entity")
        source_prompt = cross_source
        source_kind = "cross_entity_norm_matched_direction"
        vector = _match_norm(
            cross_source.activation - destination.activation,
            float(np.linalg.norm(base_direction)),
        )
        replacement = destination.activation + vector
    elif control_name in {
        "norm_matched_random",
        "orthogonal",
        "shuffled",
        "answerability_norm_residualized",
        "sign_reversed",
        "reverse_direction",
    }:
        if not isinstance(control_source, ValidationControlSource):
            raise ValueError("derived controls require a verified validation control source")
        controls = build_controls(
            base_direction,
            shuffled_direction=control_source.shuffled_direction,
            answerability_direction=control_source.answerability_direction,
            activation_norm_direction=control_source.activation_norm_direction,
            cross_entity_direction=cross_direction,
            seed=seed,
        )
        source_kind = f"derived_{control_name}_direction"
        vector = controls[control_name]
        replacement = destination.activation + vector
        if control_name in {"shuffled", "answerability_norm_residualized"}:
            control_source_split = control_source.source_split
            control_source_artifact_sha256 = control_source.source_artifact_sha256
            control_source_component_sha256 = (
                control_source.shuffled_direction_sha256
                if control_name == "shuffled"
                else control_source.residualizer_sha256
            )
    elif control_name not in {"primary", "target_bound"}:
        raise ValueError("applied intervention control is not executable")

    position = destination.anchor_positions[actual_anchor]
    source_activation_sha256 = str(source_prompt.activation_sha256)
    source_evidence_sha256 = _hash(
        {
            "source_kind": source_kind,
            "source_example_id": source_prompt.example_id,
            "source_entity_unit_id": source_prompt.entity_unit_id,
            "source_activation_sha256": source_activation_sha256,
            "activation_manifest_sha256": activation_manifest_sha256,
            "base_direction_sha256": hashlib.sha256(
                _array_bytes(np.asarray(base_direction, dtype=np.float64))
            ).hexdigest(),
            "applied_vector_sha256": hashlib.sha256(
                _array_bytes(np.asarray(vector, dtype=np.float64))
            ).hexdigest(),
            "replacement_sha256": hashlib.sha256(
                _array_bytes(np.asarray(replacement, dtype=np.float64))
            ).hexdigest(),
            "destination_example_id": destination.example_id,
            "destination_entity_unit_id": destination.entity_unit_id,
            "destination_evidence_sha256": destination.input_evidence_sha256,
            "control_source_split": control_source_split,
            "control_source_artifact_sha256": control_source_artifact_sha256,
            "control_source_component_sha256": control_source_component_sha256,
            "seed": seed,
        }
    )
    return AppliedIntervention(
        control_name=control_name,
        direction=direction,
        layer=actual_layer,
        anchor=actual_anchor,
        position=position,
        apply_patch=apply_patch,
        source_kind=source_kind,
        source_example_id=source_prompt.example_id,
        source_entity_unit_id=source_prompt.entity_unit_id,
        source_evidence_sha256=source_evidence_sha256,
        source_activation_sha256=source_activation_sha256,
        destination_example_id=destination.example_id,
        destination_entity_unit_id=destination.entity_unit_id,
        destination_evidence_sha256=destination.input_evidence_sha256,
        control_source_split=control_source_split,
        control_source_artifact_sha256=control_source_artifact_sha256,
        control_source_component_sha256=control_source_component_sha256,
        activation_manifest_sha256=activation_manifest_sha256,
        vector=vector,
        replacement=replacement,
    )


def _unrelated_intervention(
    source_pair: InterventionPair,
    prompt: UnrelatedCapabilityPrompt,
    *,
    direction: str,
    layer: int,
    anchor: str,
    activation_manifest_sha256: str,
) -> AppliedIntervention:
    source, destination = _pair_direction(source_pair, direction)
    vector = source.activation - destination.activation
    replacement = source.activation
    source_evidence_sha256 = _hash(
        {
            "source_kind": "selected_full_replacement_on_unrelated",
            "source_example_id": source.example_id,
            "source_entity_unit_id": source.entity_unit_id,
            "source_activation_sha256": source.activation_sha256,
            "activation_manifest_sha256": activation_manifest_sha256,
            "vector_sha256": hashlib.sha256(_array_bytes(vector)).hexdigest(),
            "replacement_sha256": hashlib.sha256(_array_bytes(replacement)).hexdigest(),
            "destination_example_id": prompt.prompt_id,
            "destination_evidence_sha256": prompt.input_evidence_sha256,
        }
    )
    return AppliedIntervention(
        control_name="unrelated",
        direction=direction,
        layer=layer,
        anchor=anchor,
        position=prompt.anchor_positions[anchor],
        apply_patch=True,
        source_kind="selected_full_replacement_on_unrelated",
        source_example_id=source.example_id,
        source_entity_unit_id=source.entity_unit_id,
        source_evidence_sha256=source_evidence_sha256,
        source_activation_sha256=str(source.activation_sha256),
        destination_example_id=prompt.prompt_id,
        destination_entity_unit_id=None,
        destination_evidence_sha256=prompt.input_evidence_sha256,
        control_source_split=None,
        control_source_artifact_sha256=None,
        control_source_component_sha256=None,
        activation_manifest_sha256=activation_manifest_sha256,
        vector=vector,
        replacement=replacement,
    )


def run_prefill_patch(
    runner: PrefillPatchRunner,
    pairs: Sequence[InterventionPair],
    spec: PatchSpec,
    *,
    verified_activation_manifest_sha256: str,
) -> PatchOutcome:
    """Apply a full replacement at one prefill site, then decode with no hook."""

    if not isinstance(spec, PatchSpec):
        raise ValueError("spec must be a PatchSpec")
    if getattr(runner, "model_revision", None) != spec.model_revision:
        raise ValueError("runner model revision does not match the patch specification")
    _sha256(
        verified_activation_manifest_sha256,
        "verified_activation_manifest_sha256",
    )
    if verified_activation_manifest_sha256 != spec.activation_manifest_sha256:
        raise ValueError("patch specification does not match verified activation manifest")
    typed_pairs = tuple(pairs)
    if not typed_pairs or any(not isinstance(pair, InterventionPair) for pair in typed_pairs):
        raise ValueError("run_prefill_patch requires typed intervention pairs")
    if len({pair.high.example_id for pair in typed_pairs} | {pair.low.example_id for pair in typed_pairs}) != 2 * len(typed_pairs):
        raise ValueError("intervention examples must be unique")
    before_decode_hooks = int(getattr(runner, "decode_hook_calls", 0))
    rows: list[PatchRow] = []
    changed: set[tuple[int, int]] = set()
    for pair in typed_pairs:
        source, destination = (
            (pair.high, pair.low)
            if spec.direction == "high_to_low"
            else (pair.low, pair.high)
        )
        if (
            source.activation_layer != spec.source_layer
            or source.activation_anchor != spec.source_anchor
        ):
            raise ValueError("source activation does not match the registered patch site")
        if (
            destination.activation_layer != spec.source_layer
            or destination.activation_anchor != spec.source_anchor
        ):
            raise ValueError("destination activation does not match the registered patch site")
        if (
            source.activation_manifest_sha256 != verified_activation_manifest_sha256
            or destination.activation_manifest_sha256
            != verified_activation_manifest_sha256
        ):
            raise ValueError("prompt activation does not belong to verified activation manifest")
        intervention = _applied_intervention(
            pair,
            control_name=spec.control_name,
            direction=spec.direction,
            layer=spec.source_layer,
            anchor=spec.source_anchor,
            activation_manifest_sha256=verified_activation_manifest_sha256,
        )
        if spec.control_name == "wrong_layer":
            intervention = AppliedIntervention(
                **{
                    **intervention.__dict__,
                    "layer": spec.layer,
                    "position": destination.anchor_positions[spec.anchor],
                }
            )
        if intervention.layer != spec.layer or intervention.anchor != spec.anchor:
            raise ValueError("patch specification site disagrees with applied intervention")
        position = intervention.position
        with runner.prefill_patch(intervention=intervention) as session:
            prefill_state = session.prefill(destination.input_ids)
        if bool(getattr(runner, "patch_active", False)):
            raise RuntimeError("prefill patch hook remained active before decoding")
        decoded = runner.decode(prefill_state)
        if not isinstance(decoded, Mapping):
            raise ValueError("runner.decode must return a mapping")
        audit_method = getattr(runner, "observed_patch_audit", None)
        if not callable(audit_method):
            raise ValueError("runner must expose adapter-observed patch audit data")
        audit = _coerce_patch_audit(audit_method())
        expected_site = ((spec.layer, position),)
        if audit.modified_sites != expected_site:
            raise ValueError("adapter-observed modified site differs from sealed patch site")
        if audit.decode_hook_calls != 0:
            raise ValueError("decode-time patch hook activity is forbidden")
        _validate_applied_audit(audit, intervention)
        _verify_prompt_evidence(destination, audit)
        changed.update(audit.modified_sites)
        rows.append(
            PatchRow(
                source_example_id=source.example_id,
                destination_example_id=destination.example_id,
                entity_unit_id=source.entity_unit_id,
                direction=spec.direction,
                layer=spec.layer,
                position=position,
                source_activation_sha256=str(source.activation_sha256),
                assistant_prefix_sha256=destination.assistant_prefix_sha256,
                decoded=decoded,
            )
        )
    decode_hook_calls = int(getattr(runner, "decode_hook_calls", 0)) - before_decode_hooks
    if decode_hook_calls != 0:
        raise ValueError("decode-time patch hook activity is forbidden")
    return PatchOutcome(
        rows=tuple(rows),
        changed_positions=frozenset(changed),
        decode_hook_calls=decode_hook_calls,
        spec_sha256=_hash(
            {
                "layer": spec.layer,
                "anchor": spec.anchor,
                "direction": spec.direction,
                "mode": spec.mode,
                "alpha": spec.alpha,
                "model_revision": spec.model_revision,
                "activation_manifest_sha256": spec.activation_manifest_sha256,
                "source_layer": spec.source_layer,
                "source_anchor": spec.source_anchor,
                "control_name": spec.control_name,
            }
        ),
    )


def build_controls(
    direction: np.ndarray,
    *,
    shuffled_direction: np.ndarray,
    answerability_direction: np.ndarray,
    activation_norm_direction: np.ndarray,
    cross_entity_direction: np.ndarray,
    seed: int,
) -> Mapping[str, np.ndarray]:
    """Construct deterministic norm-matched vectors for registered controls."""

    vector = _vector(direction, "direction")
    shuffled = _vector(shuffled_direction, "shuffled_direction", shape=vector.shape)
    answerability = _vector(
        answerability_direction, "answerability_direction", shape=vector.shape
    )
    activation_norm = _vector(
        activation_norm_direction, "activation_norm_direction", shape=vector.shape
    )
    cross_entity = _vector(
        cross_entity_direction, "cross_entity_direction", shape=vector.shape
    )
    if type(seed) is not int:
        raise ValueError("control seed must be an integer")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("direction must have nonzero norm")
    rng = np.random.default_rng(seed)
    random = rng.standard_normal(vector.shape)
    orthogonal = random - vector * (float(np.dot(random, vector)) / norm**2)
    if np.linalg.norm(orthogonal) <= 1e-12:
        basis = np.zeros_like(vector)
        basis[int(np.argmin(np.abs(vector)))] = 1.0
        orthogonal = basis - vector * (float(np.dot(basis, vector)) / norm**2)
    nuisance = np.column_stack((answerability, activation_norm))
    basis, _ = np.linalg.qr(nuisance)
    rank = int(np.linalg.matrix_rank(nuisance))
    residualized = vector - basis[:, :rank] @ (basis[:, :rank].T @ vector)
    if np.linalg.norm(residualized) <= 1e-12:
        raise ValueError(
            "familiarity direction has no residual after answerability/norm projection"
        )
    controls = {
        "orthogonal": _match_norm(orthogonal, norm),
        "shuffled": _match_norm(shuffled, norm),
        "answerability_norm_residualized": _match_norm(residualized, norm),
        "norm_matched_random": _match_norm(random, norm),
        "sign_reversed": -vector.copy(),
        "wrong_anchor": vector.copy(),
        "reverse_direction": -vector.copy(),
        "cross_entity": _match_norm(cross_entity, norm),
    }
    for value in controls.values():
        value.setflags(write=False)
    return MappingProxyType(controls)


def select_intervention(
    candidates: Sequence[InterventionCandidate],
    *,
    preregistration_sha256: str,
    probe_selection_sha256: str,
    confirmatory_pins: ConfirmatoryPinBundle,
    f1_evidence: GateArtifactEvidence,
    f2a_evidence: GateArtifactEvidence,
    control_source: ValidationControlSource,
) -> InterventionSelection:
    """Freeze the deterministic validation-selected primary intervention."""

    _sha256(preregistration_sha256, "preregistration_sha256")
    _sha256(probe_selection_sha256, "probe_selection_sha256")
    if not isinstance(confirmatory_pins, ConfirmatoryPinBundle):
        raise ValueError("selection requires typed confirmatory pins")
    if not isinstance(f1_evidence, GateArtifactEvidence) or not isinstance(
        f2a_evidence, GateArtifactEvidence
    ):
        raise ValueError("selection requires typed verified F1 and F2A evidence")
    if (f1_evidence.phase, f2a_evidence.phase) != ("F1", "F2A"):
        raise ValueError("selection requires F1 then F2A gate evidence")
    if (
        f1_evidence.artifact_store_sha256,
        f1_evidence.run_id,
    ) != (
        f2a_evidence.artifact_store_sha256,
        f2a_evidence.run_id,
    ):
        raise ValueError("F1 and F2A evidence must come from the same artifact store and run")
    if f1_evidence.status != "supported" or f2a_evidence.status != "supported":
        raise ValueError("F1 and F2A gates must pass before intervention selection")
    if any(
        evidence.preregistration_sha256 != preregistration_sha256
        or evidence.config_sha256 != confirmatory_pins.config_sha256
        for evidence in (f1_evidence, f2a_evidence)
    ):
        raise ValueError("gate evidence lineage does not match the selection")
    if f2a_evidence.probe_selection_sha256 != probe_selection_sha256:
        raise ValueError("F2A evidence does not bind the frozen probe selection")
    if not isinstance(control_source, ValidationControlSource):
        raise ValueError("selection requires a verified validation control source")
    if (
        control_source.source_split != "locked_validation"
        or control_source.preregistration_sha256 != preregistration_sha256
        or control_source.probe_selection_sha256 != probe_selection_sha256
    ):
        raise ValueError("validation control source does not match selection lineage")
    typed = tuple(candidates)
    if not typed or any(not isinstance(candidate, InterventionCandidate) for candidate in typed):
        raise ValueError("selection requires typed intervention candidates")
    if any(candidate.source_split != "locked_validation" for candidate in typed):
        raise ValueError("intervention selection may use locked_validation only")
    passing = [
        candidate
        for candidate in typed
        if candidate.method == "full_replacement"
        and candidate.metrics.h7_passed
        and candidate.metrics.h8_passed
    ]
    if not passing:
        raise ValueError("no passing validation candidate for F2B")
    selected = min(
        passing,
        key=lambda candidate: (
            -candidate.metrics.average_effect,
            candidate.layer,
            candidate.anchor,
            candidate.direction_sha256,
        ),
    )
    fields = {
        "layer": selected.layer,
        "anchor": selected.anchor,
        "method": selected.method,
        "alpha": selected.alpha,
        "source_split": selected.source_split,
        "direction_sha256": selected.direction_sha256,
        "preregistration_sha256": preregistration_sha256,
        "probe_selection_sha256": probe_selection_sha256,
        "confirmatory_pins_sha256": confirmatory_pins.sha256,
        "f1_evidence_sha256": f1_evidence.sha256,
        "f2a_evidence_sha256": f2a_evidence.sha256,
        "f1_result_sha256": f1_evidence.result_sha256,
        "f2a_result_sha256": f2a_evidence.result_sha256,
        "f1_artifact_sha256": f1_evidence.artifact_sha256,
        "f2a_artifact_sha256": f2a_evidence.artifact_sha256,
        "f1_manifest_sha256": f1_evidence.manifest_sha256,
        "f2a_manifest_sha256": f2a_evidence.manifest_sha256,
        "control_source": control_source,
        "control_source_artifact_sha256": control_source.source_artifact_sha256,
        "validation_metrics": selected.metrics,
        "candidate_count": len(typed),
    }
    provisional = InterventionSelection.__new__(InterventionSelection)
    for name, value in fields.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "sha256", "0" * 64)
    return InterventionSelection(**fields, sha256=_selection_sha256(provisional))


def evaluate_intervention_test_once(
    selection: InterventionSelection,
    store: FAArtifactStore,
    *,
    endpoint_manifest_path: str | Path,
    activation_manifest_path: str | Path,
    unrelated_manifest_path: str | Path,
    test_pairs: Sequence[InterventionPair],
    unrelated_prompts: Sequence[UnrelatedCapabilityPrompt],
    executor: ConfirmatoryInterventionExecutor,
    confirmatory_pins: ConfirmatoryPinBundle,
) -> InterventionTestResult:
    """Execute raw preregistered trials, derive metrics, and close the endpoint."""

    if not isinstance(selection, InterventionSelection):
        raise ValueError("selection must be an InterventionSelection")
    if not isinstance(store, FAArtifactStore):
        raise ValueError("evaluation requires an FAArtifactStore")
    if not isinstance(confirmatory_pins, ConfirmatoryPinBundle):
        raise ValueError("evaluation requires typed confirmatory pins")
    if selection.confirmatory_pins_sha256 != confirmatory_pins.sha256:
        raise ValueError("selection confirmatory pins do not match execution pins")
    endpoint_input = store.verify_endpoint_artifact(
        "intervention_test", endpoint_manifest_path
    )
    activation_manifest = store.verify_endpoint_artifact(
        "intervention_test", activation_manifest_path
    )
    unrelated_manifest = store.verify_endpoint_artifact(
        "intervention_test", unrelated_manifest_path
    )
    pairs = tuple(test_pairs)
    if not pairs or any(not isinstance(pair, InterventionPair) for pair in pairs):
        raise ValueError("evaluation requires typed intervention_test pairs")
    if any(pair.high.split != "intervention_test" for pair in pairs):
        raise ValueError("evaluation may use intervention_test rows only")
    _bind_pairs_to_endpoint(pairs, endpoint_input)
    _bind_activations_to_manifest(pairs, activation_manifest)
    _verify_shard_pins(endpoint_input, confirmatory_pins)
    _verify_shard_pins(activation_manifest, confirmatory_pins)
    _verify_pair_pins(pairs, confirmatory_pins)
    unrelated = tuple(unrelated_prompts)
    _bind_unrelated_prompts(unrelated, unrelated_manifest, pairs, confirmatory_pins)
    example_ids = tuple(
        sorted(prompt.example_id for pair in pairs for prompt in (pair.high, pair.low))
    )
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("intervention_test examples must be unique")
    if store.endpoint_state("intervention_test", endpoint_input.manifest_path) == "evaluated":
        metrics_shard = store.read_evaluated_metrics(
            "intervention_test", endpoint_input.manifest_path
        )
        return _recover_evaluated_intervention_result(
            store,
            metrics_shard=metrics_shard,
            selection=selection,
            example_ids=example_ids,
            endpoint_input=endpoint_input,
            activation_manifest=activation_manifest,
            unrelated_manifest=unrelated_manifest,
            confirmatory_pins=confirmatory_pins,
        )
    receipt = store.unlock_or_resume_endpoint(
        "intervention_test", endpoint_input.manifest_path
    )
    if receipt.preregistration_hash != selection.preregistration_sha256:
        raise ValueError("receipt preregistration does not match selection")
    if receipt.selection_manifest_hash != selection.sha256:
        raise ValueError("receipt selection manifest does not match selection")
    execute = getattr(executor, "execute", None)
    if not callable(execute):
        raise ValueError("confirmatory evaluation requires an execution adapter")
    outcomes: list[RawInterventionOutcome] = []
    h7_pairs = tuple(pair for pair in pairs if pair.high.answerability == "code_absent")
    target_pairs = tuple(pair for pair in pairs if pair.high.answerability == "target_bound")
    if not h7_pairs or not target_pairs:
        raise ValueError("intervention endpoint requires code_absent and target_bound pairs")
    h7_pairs = tuple(sorted(h7_pairs, key=lambda pair: pair.high.entity_unit_id))
    if {pair.high.domain for pair in h7_pairs} != _REGISTERED_DOMAINS:
        raise ValueError("intervention endpoint requires all four registered domains")
    for pair_index, pair in enumerate(h7_pairs):
        cross_entity_pair = h7_pairs[(pair_index + 1) % len(h7_pairs)]
        for control_name in ("primary", *REQUIRED_CAUSAL_CONTROLS):
            for direction in sorted(_DIRECTIONS):
                intervention = _applied_intervention(
                    pair,
                    control_name=control_name,
                    direction=direction,
                    layer=selection.layer,
                    anchor=selection.anchor,
                    activation_manifest_sha256=activation_manifest.sha256,
                    cross_entity_pair=cross_entity_pair,
                    control_source=selection.control_source,
                )
                executed = execute(
                    pair=pair,
                    intervention=intervention,
                    activation_manifest_sha256=activation_manifest.sha256,
                    confirmatory_pins=confirmatory_pins,
                )
                if executed is None:
                    raise ValueError(f"missing executed control evidence for {control_name}")
                raw = _derive_raw_intervention_outcome(
                    executed,
                    pair=pair,
                    intervention=intervention,
                    activation_manifest_sha256=activation_manifest.sha256,
                    confirmatory_pins=confirmatory_pins,
                )
                outcomes.append(raw)
    for pair in target_pairs:
        for direction in sorted(_DIRECTIONS):
            intervention = _applied_intervention(
                pair,
                control_name="target_bound",
                direction=direction,
                layer=selection.layer,
                anchor=selection.anchor,
                activation_manifest_sha256=activation_manifest.sha256,
                control_source=selection.control_source,
            )
            executed = execute(
                pair=pair,
                intervention=intervention,
                activation_manifest_sha256=activation_manifest.sha256,
                confirmatory_pins=confirmatory_pins,
            )
            if executed is None:
                raise ValueError("target_bound control has no executed numeric evidence")
            raw = _derive_raw_intervention_outcome(
                executed,
                pair=pair,
                intervention=intervention,
                activation_manifest_sha256=activation_manifest.sha256,
                confirmatory_pins=confirmatory_pins,
            )
            outcomes.append(raw)
    execute_unrelated = getattr(executor, "execute_unrelated", None)
    if not callable(execute_unrelated):
        raise ValueError("confirmatory evaluation requires unrelated execution evidence")
    unrelated_outcomes: list[RawUnrelatedOutcome] = []
    for source_pair in h7_pairs:
        for direction in sorted(_DIRECTIONS):
            for prompt in sorted(unrelated, key=lambda item: item.prompt_id):
                intervention = _unrelated_intervention(
                    source_pair,
                    prompt,
                    direction=direction,
                    layer=selection.layer,
                    anchor=selection.anchor,
                    activation_manifest_sha256=activation_manifest.sha256,
                )
                executed = execute_unrelated(
                    prompt=prompt,
                    intervention=intervention,
                    activation_manifest_sha256=activation_manifest.sha256,
                    confirmatory_pins=confirmatory_pins,
                )
                raw = _derive_raw_unrelated_outcome(
                    executed,
                    prompt=prompt,
                    intervention=intervention,
                    activation_manifest_sha256=activation_manifest.sha256,
                    confirmatory_pins=confirmatory_pins,
                )
                unrelated_outcomes.append(raw)
    run_id = _artifact_run_id(store, endpoint_input.data_path)
    raw_shard = _persist_raw_outcomes(
        store,
        run_id=run_id,
        outcomes=outcomes,
        unrelated_outcomes=unrelated_outcomes,
        selection=selection,
        endpoint_input=endpoint_input,
        activation_manifest=activation_manifest,
        unrelated_manifest=unrelated_manifest,
        confirmatory_pins=confirmatory_pins,
    )
    metrics = _derive_intervention_metrics(
        outcomes,
        h7_pairs,
        unrelated_outcomes,
        readout_constraints=selection.validation_metrics.readout_constraints,
    )
    fields = {
        "selection_sha256": selection.sha256,
        "preregistration_sha256": selection.preregistration_sha256,
        "example_ids": example_ids,
        "metrics": metrics,
        "h7_passed": metrics.h7_passed,
        "h8_passed": metrics.h8_passed,
        "refit_performed": False,
    }
    digest = _hash(
        {
            "selection_sha256": fields["selection_sha256"],
            "preregistration_sha256": fields["preregistration_sha256"],
            "example_ids": list(fields["example_ids"]),
            "metrics": fields["metrics"].canonical_payload(),
            "h7_passed": fields["h7_passed"],
            "h8_passed": fields["h8_passed"],
            "refit_performed": fields["refit_performed"],
        }
    )
    result = InterventionTestResult(**fields, result_sha256=digest)
    row = {
        "kind": "metrics",
        "result": _result_payload(result),
    }
    lineage = {
        "preregistration_sha256": selection.preregistration_sha256,
        "selection_sha256": selection.sha256,
        "f1_evidence_sha256": selection.f1_evidence_sha256,
        "f2a_evidence_sha256": selection.f2a_evidence_sha256,
        "f1_result_sha256": selection.f1_result_sha256,
        "f2a_result_sha256": selection.f2a_result_sha256,
        "f1_artifact_sha256": selection.f1_artifact_sha256,
        "f2a_artifact_sha256": selection.f2a_artifact_sha256,
        "f1_manifest_sha256": selection.f1_manifest_sha256,
        "f2a_manifest_sha256": selection.f2a_manifest_sha256,
        "control_source_sha256": selection.control_source.sha256,
        "control_source_artifact_sha256": selection.control_source_artifact_sha256,
        "endpoint_input_sha256": endpoint_input.sha256,
        "activation_manifest_sha256": activation_manifest.sha256,
        "unrelated_manifest_sha256": unrelated_manifest.sha256,
        "raw_outcomes_manifest_sha256": raw_shard.sha256,
        "confirmatory_pins_sha256": confirmatory_pins.sha256,
        "result_sha256": result.result_sha256,
    }
    shard_id = f"intervention-metrics-{result.result_sha256[:16]}"
    try:
        metrics_shard = store.write_completed_shard(
            run_id,
            "intervention_test",
            shard_id,
            [row],
            lineage,
            record_kind="metrics",
        )
    except FileExistsError:
        candidate = (
            store.root
            / "runs"
            / "familiarity_answerability"
            / run_id
            / "shards"
            / "intervention_test"
            / f"{shard_id}.jsonl.manifest.json"
        )
        metrics_shard = store.verify_shard(candidate)
        if metrics_shard.record_kind != "metrics":
            raise ValueError("existing intervention result is not a metrics artifact")
        expected_data_sha256 = hashlib.sha256(_canonical_json(row) + b"\n").hexdigest()
        if metrics_shard.sha256 != expected_data_sha256 or metrics_shard.row_count != 1:
            raise ValueError("existing intervention result does not match canonical metrics")
    store.mark_evaluated(receipt, metrics_shard.data_path)
    _verify_raw_outcomes_shard(store, raw_shard, lineage)
    store.close_endpoint("intervention_test")
    return result


def _recover_evaluated_intervention_result(
    store: FAArtifactStore,
    *,
    metrics_shard: Any,
    selection: InterventionSelection,
    example_ids: tuple[str, ...],
    endpoint_input: Any,
    activation_manifest: Any,
    unrelated_manifest: Any,
    confirmatory_pins: ConfirmatoryPinBundle,
) -> InterventionTestResult:
    rows = _read_canonical_rows(metrics_shard)
    if (
        metrics_shard.record_kind != "metrics"
        or len(rows) != 1
        or set(rows[0]) != {"kind", "result"}
        or rows[0].get("kind") != "metrics"
    ):
        raise ValueError("evaluated intervention metrics artifact has an invalid schema")
    result = _intervention_result_from_payload(rows[0]["result"])
    if (
        result.selection_sha256 != selection.sha256
        or result.preregistration_sha256 != selection.preregistration_sha256
        or result.example_ids != example_ids
    ):
        raise ValueError("evaluated intervention result does not match this execution")
    try:
        manifest = json.loads(Path(metrics_shard.manifest_path).read_bytes())
        lineage = manifest["lineage"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError("evaluated intervention metrics lineage is unreadable") from error
    raw_sha256 = lineage.get("raw_outcomes_manifest_sha256")
    _sha256(raw_sha256, "raw_outcomes_manifest_sha256")
    expected_lineage = {
        "preregistration_sha256": selection.preregistration_sha256,
        "selection_sha256": selection.sha256,
        "f1_evidence_sha256": selection.f1_evidence_sha256,
        "f2a_evidence_sha256": selection.f2a_evidence_sha256,
        "f1_result_sha256": selection.f1_result_sha256,
        "f2a_result_sha256": selection.f2a_result_sha256,
        "f1_artifact_sha256": selection.f1_artifact_sha256,
        "f2a_artifact_sha256": selection.f2a_artifact_sha256,
        "f1_manifest_sha256": selection.f1_manifest_sha256,
        "f2a_manifest_sha256": selection.f2a_manifest_sha256,
        "control_source_sha256": selection.control_source.sha256,
        "control_source_artifact_sha256": selection.control_source_artifact_sha256,
        "endpoint_input_sha256": endpoint_input.sha256,
        "activation_manifest_sha256": activation_manifest.sha256,
        "unrelated_manifest_sha256": unrelated_manifest.sha256,
        "raw_outcomes_manifest_sha256": raw_sha256,
        "confirmatory_pins_sha256": confirmatory_pins.sha256,
        "result_sha256": result.result_sha256,
    }
    if dict(lineage) != expected_lineage:
        raise ValueError("evaluated intervention metrics lineage does not verify")
    run_id = _artifact_run_id(store, endpoint_input.data_path)
    raw_matches = [
        shard
        for shard in store.resume_verified_shards(run_id, "intervention_test")
        if shard.record_kind == "raw_intervention_outcomes"
        and shard.sha256 == raw_sha256
    ]
    if len(raw_matches) != 1:
        raise ValueError("evaluated intervention raw evidence does not verify")
    _verify_raw_outcomes_shard(store, raw_matches[0], lineage)
    store.close_endpoint("intervention_test")
    return result


def _intervention_result_from_payload(value: Any) -> InterventionTestResult:
    expected = {
        "selection_sha256",
        "preregistration_sha256",
        "example_ids",
        "metrics",
        "h7_passed",
        "h8_passed",
        "refit_performed",
        "result_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("evaluated intervention result has an invalid schema")
    if not isinstance(value["example_ids"], (tuple, list)):
        raise ValueError("evaluated intervention example IDs are invalid")
    return InterventionTestResult(
        selection_sha256=value["selection_sha256"],
        preregistration_sha256=value["preregistration_sha256"],
        example_ids=tuple(value["example_ids"]),
        metrics=_intervention_metrics_from_payload(value["metrics"]),
        h7_passed=value["h7_passed"],
        h8_passed=value["h8_passed"],
        refit_performed=value["refit_performed"],
        result_sha256=value["result_sha256"],
    )


def _intervention_metrics_from_payload(value: Any) -> InterventionMetrics:
    expected = {
        "high_to_low_effect",
        "high_to_low_interval",
        "low_to_high_effect",
        "low_to_high_interval",
        "control_effects",
        "target_bound_accuracy_change",
        "unrelated_refusal_change",
        "unrelated_invalid_format_change",
        "familiarity_readout_effect",
        "answerability_max_abs_change",
        "entity_type_max_abs_change",
        "generic_confidence_max_abs_change",
        "readout_constraints",
        "observed_domains",
        "passing_domains",
        "completed_fraction",
        "bootstrap_summary",
        "unrelated_refusal_change_by_direction",
        "unrelated_invalid_format_change_by_direction",
        "target_bound_accuracy_change_by_direction",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("evaluated intervention metrics have an invalid schema")
    constraints = value["readout_constraints"]
    if not isinstance(constraints, Mapping) or set(constraints) != {
        "familiarity_min_effect",
        "answerability_max_abs_change",
        "entity_type_max_abs_change",
        "generic_confidence_max_abs_change",
    }:
        raise ValueError("evaluated readout constraints have an invalid schema")
    return InterventionMetrics(
        high_to_low_effect=value["high_to_low_effect"],
        high_to_low_interval=tuple(value["high_to_low_interval"]),
        low_to_high_effect=value["low_to_high_effect"],
        low_to_high_interval=tuple(value["low_to_high_interval"]),
        control_effects={
            name: tuple(effects) for name, effects in value["control_effects"].items()
        },
        target_bound_accuracy_change=value["target_bound_accuracy_change"],
        unrelated_refusal_change=value["unrelated_refusal_change"],
        unrelated_invalid_format_change=value["unrelated_invalid_format_change"],
        familiarity_readout_effect=value["familiarity_readout_effect"],
        answerability_max_abs_change=value["answerability_max_abs_change"],
        entity_type_max_abs_change=value["entity_type_max_abs_change"],
        generic_confidence_max_abs_change=value["generic_confidence_max_abs_change"],
        readout_constraints=ReadoutConstraints(**dict(constraints)),
        observed_domains=tuple(value["observed_domains"]),
        passing_domains=tuple(value["passing_domains"]),
        completed_fraction=value["completed_fraction"],
        bootstrap_summary=value["bootstrap_summary"],
        unrelated_refusal_change_by_direction=value[
            "unrelated_refusal_change_by_direction"
        ],
        unrelated_invalid_format_change_by_direction=value[
            "unrelated_invalid_format_change_by_direction"
        ],
        target_bound_accuracy_change_by_direction=value[
            "target_bound_accuracy_change_by_direction"
        ],
    )


def _bind_pairs_to_endpoint(
    pairs: Sequence[InterventionPair], endpoint_input: Any
) -> None:
    actual = _read_canonical_rows(endpoint_input)
    expected = [
        {"kind": "intervention_prompt", "prompt": prompt.canonical_payload()}
        for pair in pairs
        for prompt in (pair.high, pair.low)
    ]
    if _canonical_row_multiset(actual) != _canonical_row_multiset(expected):
        raise ValueError("intervention pairs do not exactly match the sealed endpoint")


def _bind_activations_to_manifest(
    pairs: Sequence[InterventionPair], activation_manifest: Any
) -> None:
    actual = _read_canonical_rows(activation_manifest)
    expected = [
        {
            "example_id": prompt.example_id,
            "activation_layer": prompt.activation_layer,
            "activation_anchor": prompt.activation_anchor,
            "activation_sha256": prompt.activation_sha256,
            "model_revision": prompt.model_revision,
            "tokenizer_revision": prompt.tokenizer_revision,
            "chat_template_sha256": prompt.chat_template_sha256,
        }
        for pair in pairs
        for prompt in (pair.high, pair.low)
    ]
    if any(
        prompt.activation_manifest_sha256 != activation_manifest.sha256
        for pair in pairs
        for prompt in (pair.high, pair.low)
    ):
        raise ValueError("prompt activation provenance does not reference verified manifest")
    if _canonical_row_multiset(actual) != _canonical_row_multiset(expected):
        raise ValueError("activation manifest does not match sealed intervention prompts")


def _verify_pair_pins(
    pairs: Sequence[InterventionPair], confirmatory_pins: ConfirmatoryPinBundle
) -> None:
    expected = (
        confirmatory_pins.model_revision,
        confirmatory_pins.tokenizer_revision,
        confirmatory_pins.chat_template_sha256,
    )
    if any(
        (prompt.model_revision, prompt.tokenizer_revision, prompt.chat_template_sha256)
        != expected
        for pair in pairs
        for prompt in (pair.high, pair.low)
    ):
        raise ValueError("intervention prompts do not match confirmatory pins")


def _bind_unrelated_prompts(
    prompts: Sequence[UnrelatedCapabilityPrompt],
    unrelated_manifest: Any,
    pairs: Sequence[InterventionPair],
    confirmatory_pins: ConfirmatoryPinBundle,
) -> None:
    if not prompts or any(not isinstance(prompt, UnrelatedCapabilityPrompt) for prompt in prompts):
        raise ValueError("evaluation requires typed separately sealed unrelated prompts")
    if len({prompt.prompt_id for prompt in prompts}) != len(prompts):
        raise ValueError("unrelated prompt IDs must be unique")
    _verify_shard_pins(unrelated_manifest, confirmatory_pins)
    expected = [
        {"kind": "unrelated_capability_prompt", "prompt": prompt.canonical_payload()}
        for prompt in prompts
    ]
    actual = _read_canonical_rows(unrelated_manifest)
    if _canonical_row_multiset(actual) != _canonical_row_multiset(expected):
        raise ValueError("unrelated prompts do not exactly match the sealed endpoint")
    pair_inputs = {prompt.input_ids for pair in pairs for prompt in (pair.high, pair.low)}
    if any(prompt.input_ids in pair_inputs for prompt in prompts):
        raise ValueError("unrelated prompts must not reuse same-string intervention inputs")
    if any(
        (prompt.model_revision, prompt.tokenizer_revision, prompt.chat_template_sha256)
        != (
            confirmatory_pins.model_revision,
            confirmatory_pins.tokenizer_revision,
            confirmatory_pins.chat_template_sha256,
        )
        for prompt in prompts
    ):
        raise ValueError("unrelated prompts do not match confirmatory pins")


def _verify_shard_pins(shard: Any, confirmatory_pins: ConfirmatoryPinBundle) -> None:
    try:
        manifest = json.loads(Path(shard.manifest_path).read_bytes())
        lineage = manifest["lineage"]
    except (OSError, TypeError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError("sealed artifact pins are unreadable") from error
    if (
        lineage.get("confirmatory_pins") != confirmatory_pins.canonical_payload()
        or lineage.get("confirmatory_pins_sha256") != confirmatory_pins.sha256
    ):
        raise ValueError("sealed artifact confirmatory pins do not match execution pins")


def _read_canonical_rows(shard: Any) -> tuple[Mapping[str, Any], ...]:
    data = Path(shard.data_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != shard.sha256:
        raise ValueError("verified shard changed before intervention evaluation")
    lines = data.splitlines()
    if len(lines) != shard.row_count:
        raise ValueError("verified shard row count changed before evaluation")
    rows: list[Mapping[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("intervention artifact contains invalid JSON") from error
        if not isinstance(row, dict) or _canonical_json(row) != line:
            raise ValueError("intervention artifacts must contain canonical JSON objects")
        rows.append(row)
    return tuple(rows)


def _canonical_row_multiset(rows: Sequence[Mapping[str, Any]]) -> tuple[bytes, ...]:
    return tuple(sorted(_canonical_json(row) for row in rows))


def _expected_execution_provenance(
    intervention: AppliedIntervention,
    confirmatory_pins: ConfirmatoryPinBundle,
) -> Mapping[str, str]:
    return {
        "activation_manifest_sha256": intervention.activation_manifest_sha256,
        "model_id": confirmatory_pins.model_id,
        "model_revision": confirmatory_pins.model_revision,
        "tokenizer_revision": confirmatory_pins.tokenizer_revision,
        "chat_template_sha256": confirmatory_pins.chat_template_sha256,
        "config_sha256": confirmatory_pins.config_sha256,
        "source_pins_sha256": confirmatory_pins.source_pins_sha256,
        "confirmatory_pins_sha256": confirmatory_pins.sha256,
        "applied_intervention_sha256": intervention.sha256,
    }


def _validate_executed_evidence(
    executed: Any,
    *,
    intervention: AppliedIntervention,
    prompt: InterventionPrompt | UnrelatedCapabilityPrompt,
    activation_manifest_sha256: str,
    confirmatory_pins: ConfirmatoryPinBundle,
    evidence_type: type[ExecutedPatchEvidence] | type[ExecutedUnrelatedEvidence],
) -> None:
    if not isinstance(executed, evidence_type):
        raise ValueError("executor must return typed hash-bound execution evidence")
    if intervention.activation_manifest_sha256 != activation_manifest_sha256:
        raise ValueError("applied intervention activation provenance is not verified")
    if executed.applied_intervention_sha256 != intervention.sha256:
        raise ValueError("executor returned evidence for a different intervention")
    if dict(executed.provenance) != dict(
        _expected_execution_provenance(intervention, confirmatory_pins)
    ):
        raise ValueError("raw outcome activation provenance is not verified")
    _validate_applied_audit(executed.audit, intervention)
    if executed.audit.decode_hook_calls != 0:
        raise ValueError("decode-time patch hook activity is forbidden")
    _verify_prompt_evidence(prompt, executed.audit)


def _derive_raw_intervention_outcome(
    executed: Any,
    *,
    pair: InterventionPair,
    intervention: AppliedIntervention,
    activation_manifest_sha256: str,
    confirmatory_pins: ConfirmatoryPinBundle,
) -> RawInterventionOutcome:
    _, destination = _pair_direction(pair, intervention.direction)
    _validate_executed_evidence(
        executed,
        intervention=intervention,
        prompt=destination,
        activation_manifest_sha256=activation_manifest_sha256,
        confirmatory_pins=confirmatory_pins,
        evidence_type=ExecutedPatchEvidence,
    )
    return RawInterventionOutcome(
        source_example_id=str(intervention.source_example_id),
        destination_example_id=destination.example_id,
        entity_unit_id=destination.entity_unit_id,
        domain=destination.domain,
        answerability=destination.answerability,
        template_family=destination.template_family,
        direction=intervention.direction,
        registry_code=destination.registry_code,
        target_familiarity=destination.target_familiarity,
        distractor_familiarity=destination.distractor_familiarity,
        exposure=destination.exposure,
        applied_intervention=intervention,
        audit=executed.audit,
        provenance=executed.provenance,
        decoded=executed.decoded,
    )


def _derive_raw_unrelated_outcome(
    executed: Any,
    *,
    prompt: UnrelatedCapabilityPrompt,
    intervention: AppliedIntervention,
    activation_manifest_sha256: str,
    confirmatory_pins: ConfirmatoryPinBundle,
) -> RawUnrelatedOutcome:
    _validate_executed_evidence(
        executed,
        intervention=intervention,
        prompt=prompt,
        activation_manifest_sha256=activation_manifest_sha256,
        confirmatory_pins=confirmatory_pins,
        evidence_type=ExecutedUnrelatedEvidence,
    )
    return RawUnrelatedOutcome(
        prompt_id=prompt.prompt_id,
        task=prompt.task,
        input_evidence_sha256=prompt.input_evidence_sha256,
        applied_intervention=intervention,
        audit=executed.audit,
        provenance=executed.provenance,
        decoded=executed.decoded,
    )


def _verify_prompt_evidence(
    prompt: InterventionPrompt | UnrelatedCapabilityPrompt, audit: PatchAudit
) -> None:
    if audit.input_ids != prompt.input_ids:
        raise ValueError("adapter-observed input IDs differ from sealed prompt")
    if (
        audit.assistant_prefix_token_ids != prompt.assistant_prefix_token_ids
        or audit.rendered_prefix_utf8_sha256
        != hashlib.sha256(prompt.rendered_prefix_utf8).hexdigest()
        or audit.tokenizer_revision != prompt.tokenizer_revision
        or audit.chat_template_sha256 != prompt.chat_template_sha256
        or _prefix_evidence_sha256(
            prompt.rendered_prefix_utf8,
            audit.tokenizer_revision,
            audit.chat_template_sha256,
        )
        != (
            prompt.assistant_prefix_sha256
            if isinstance(prompt, InterventionPrompt)
            else _prefix_evidence_sha256(
                prompt.rendered_prefix_utf8,
                prompt.tokenizer_revision,
                prompt.chat_template_sha256,
            )
        )
    ):
        raise ValueError("adapter-observed prefix evidence differs from pinned prefix")


def _validate_applied_audit(
    audit: PatchAudit, intervention: AppliedIntervention
) -> None:
    if (
        audit.applied_intervention_sha256 != intervention.sha256
        or audit.applied_vector_sha256 != intervention.vector_sha256
        or audit.replacement_sha256 != intervention.replacement_sha256
        or audit.source_evidence_sha256 != intervention.source_evidence_sha256
        or audit.source_activation_sha256 != intervention.source_activation_sha256
        or audit.source_example_id != intervention.source_example_id
        or audit.destination_example_id != intervention.destination_example_id
        or audit.destination_entity_unit_id
        != intervention.destination_entity_unit_id
        or audit.destination_evidence_sha256
        != intervention.destination_evidence_sha256
    ):
        raise ValueError("adapter audit does not bind the applied intervention tensor and source")
    expected_sites = (
        ((intervention.layer, intervention.position),)
        if intervention.apply_patch
        else ()
    )
    if audit.modified_sites != expected_sites:
        raise ValueError("adapter-observed modified site differs from sealed execution plan")


def _coerce_patch_audit(value: Any) -> PatchAudit:
    if isinstance(value, PatchAudit):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("adapter-observed patch audit must be a mapping")
    try:
        return PatchAudit(
            modified_sites=tuple(value["modified_sites"]),
            decode_hook_calls=value["decode_hook_calls"],
            input_ids=tuple(value["input_ids"]),
            assistant_prefix_token_ids=tuple(value["assistant_prefix_token_ids"]),
            rendered_prefix_utf8_sha256=value["rendered_prefix_utf8_sha256"],
            tokenizer_revision=value["tokenizer_revision"],
            chat_template_sha256=value["chat_template_sha256"],
            applied_intervention_sha256=value["applied_intervention_sha256"],
            applied_vector_sha256=value["applied_vector_sha256"],
            replacement_sha256=value["replacement_sha256"],
            source_evidence_sha256=value["source_evidence_sha256"],
            source_activation_sha256=value["source_activation_sha256"],
            source_example_id=value["source_example_id"],
            destination_example_id=value["destination_example_id"],
            destination_entity_unit_id=value["destination_entity_unit_id"],
            destination_evidence_sha256=value["destination_evidence_sha256"],
        )
    except KeyError as error:
        raise ValueError("adapter-observed patch audit is incomplete") from error


def _worst_directional_change(values: Mapping[str, float]) -> float:
    if set(values) != _DIRECTIONS:
        raise ValueError("directional capability changes require both directions")
    return max(
        (float(values[direction]) for direction in sorted(_DIRECTIONS)),
        key=abs,
    )


def _derive_intervention_metrics(
    outcomes: Sequence[RawInterventionOutcome],
    h7_pairs: Sequence[InterventionPair],
    unrelated_outcomes: Sequence[RawUnrelatedOutcome],
    *,
    readout_constraints: ReadoutConstraints,
) -> InterventionMetrics:
    by_condition: dict[tuple[str, str], list[RawInterventionOutcome]] = {}
    for outcome in outcomes:
        by_condition.setdefault((outcome.control_name, outcome.direction), []).append(
            outcome
        )
    expected_per_cell = len(h7_pairs)
    for control_name in ("primary", *REQUIRED_CAUSAL_CONTROLS):
        for direction in sorted(_DIRECTIONS):
            cell = by_condition.get((control_name, direction), [])
            if len(cell) != expected_per_cell:
                raise ValueError(f"missing executed control evidence for {control_name}")

    primary_high = _oriented_effects(by_condition[("primary", "high_to_low")])
    primary_low = _oriented_effects(by_condition[("primary", "low_to_high")])
    control_effects = {
        name: (
            float(np.mean(_oriented_effects(by_condition[(name, "high_to_low")]))),
            float(np.mean(_oriented_effects(by_condition[(name, "low_to_high")]))),
        )
        for name in REQUIRED_CAUSAL_CONTROLS
    }
    target_bound = [
        outcome
        for outcome in outcomes
        if outcome.control_name == "target_bound"
        and outcome.answerability == "target_bound"
    ]
    if not target_bound:
        raise ValueError("target_bound control has no executed numeric evidence")
    target_bound_by_direction = {
        direction: tuple(
            outcome for outcome in target_bound if outcome.direction == direction
        )
        for direction in sorted(_DIRECTIONS)
    }
    if any(not cell for cell in target_bound_by_direction.values()):
        raise ValueError("target_bound capability evidence requires both directions")
    target_accuracy_by_direction = {
        direction: float(
            np.mean(
                [
                    outcome.patched_correct - outcome.baseline_correct
                    for outcome in cell
                ]
            )
        )
        for direction, cell in target_bound_by_direction.items()
    }
    if not unrelated_outcomes:
        raise ValueError("unrelated control has no executed numeric evidence")
    unrelated_prompt_ids = {outcome.prompt_id for outcome in unrelated_outcomes}
    expected_unrelated_per_direction = len(h7_pairs) * len(unrelated_prompt_ids)
    unrelated_by_direction = {
        direction: tuple(
            outcome
            for outcome in unrelated_outcomes
            if outcome.applied_intervention.direction == direction
        )
        for direction in sorted(_DIRECTIONS)
    }
    expected_source_ids = {pair.high.entity_unit_id for pair in h7_pairs}
    if any(
        len(cell) != expected_unrelated_per_direction
        or {outcome.applied_intervention.source_entity_unit_id for outcome in cell}
        != expected_source_ids
        or {outcome.prompt_id for outcome in cell} != unrelated_prompt_ids
        for cell in unrelated_by_direction.values()
    ):
        raise ValueError(
            "unrelated controls require every deterministic source pair in both directions"
        )
    refusal_by_direction = {
        direction: float(
            np.mean(
                [
                    outcome.patched_refusal - outcome.baseline_refusal
                    for outcome in cell
                ]
            )
        )
        for direction, cell in unrelated_by_direction.items()
    }
    invalid_by_direction = {
        direction: float(
            np.mean(
                [
                    outcome.patched_invalid_format
                    - outcome.baseline_invalid_format
                    for outcome in cell
                ]
            )
        )
        for direction, cell in unrelated_by_direction.items()
    }

    passing_domains = []
    for domain in sorted({pair.high.domain for pair in h7_pairs}):
        high = [
            outcome
            for outcome in by_condition[("primary", "high_to_low")]
            if outcome.domain == domain
        ]
        low = [
            outcome
            for outcome in by_condition[("primary", "low_to_high")]
            if outcome.domain == domain
        ]
        if high and low and np.mean(_oriented_effects(high)) > 0 and np.mean(
            _oriented_effects(low)
        ) > 0:
            passing_domains.append(domain)

    bootstrap_summary = _h7_crossed_bootstrap(
        by_condition[("primary", "high_to_low")],
        by_condition[("primary", "low_to_high")],
    )
    directions = bootstrap_summary["directions"]
    primary_outcomes = (
        *by_condition[("primary", "high_to_low")],
        *by_condition[("primary", "low_to_high")],
    )
    familiarity_effects = [
        outcome.familiarity_readout_change
        if outcome.direction == "high_to_low"
        else -outcome.familiarity_readout_change
        for outcome in primary_outcomes
    ]
    return InterventionMetrics(
        high_to_low_effect=float(np.mean(primary_high)),
        high_to_low_interval=tuple(directions["high_to_low"]["holm_interval"]),
        low_to_high_effect=float(np.mean(primary_low)),
        low_to_high_interval=tuple(directions["low_to_high"]["holm_interval"]),
        control_effects=control_effects,
        target_bound_accuracy_change=min(target_accuracy_by_direction.values()),
        unrelated_refusal_change=_worst_directional_change(refusal_by_direction),
        unrelated_invalid_format_change=_worst_directional_change(
            invalid_by_direction
        ),
        familiarity_readout_effect=float(np.mean(familiarity_effects)),
        answerability_max_abs_change=float(
            max(abs(outcome.answerability_readout_change) for outcome in primary_outcomes)
        ),
        entity_type_max_abs_change=float(
            max(abs(outcome.entity_type_readout_change) for outcome in primary_outcomes)
        ),
        generic_confidence_max_abs_change=float(
            max(abs(outcome.generic_confidence_change) for outcome in primary_outcomes)
        ),
        readout_constraints=readout_constraints,
        observed_domains=tuple(sorted({pair.high.domain for pair in h7_pairs})),
        passing_domains=tuple(passing_domains),
        completed_fraction=1.0,
        bootstrap_summary=bootstrap_summary,
        unrelated_refusal_change_by_direction=refusal_by_direction,
        unrelated_invalid_format_change_by_direction=invalid_by_direction,
        target_bound_accuracy_change_by_direction=target_accuracy_by_direction,
    )


def _persist_raw_outcomes(
    store: FAArtifactStore,
    *,
    run_id: str,
    outcomes: Sequence[RawInterventionOutcome],
    unrelated_outcomes: Sequence[RawUnrelatedOutcome],
    selection: InterventionSelection,
    endpoint_input: Any,
    activation_manifest: Any,
    unrelated_manifest: Any,
    confirmatory_pins: ConfirmatoryPinBundle,
) -> Any:
    rows = [
        {
            "kind": "raw_intervention_outcomes",
            "outcome_kind": "raw_intervention_outcome",
            "outcome": outcome.canonical_payload(),
        }
        for outcome in outcomes
    ] + [
        {
            "kind": "raw_intervention_outcomes",
            "outcome_kind": "raw_unrelated_outcome",
            "outcome": outcome.canonical_payload(),
        }
        for outcome in unrelated_outcomes
    ]
    lineage = {
        "preregistration_sha256": selection.preregistration_sha256,
        "selection_sha256": selection.sha256,
        "f1_evidence_sha256": selection.f1_evidence_sha256,
        "f2a_evidence_sha256": selection.f2a_evidence_sha256,
        "f1_result_sha256": selection.f1_result_sha256,
        "f2a_result_sha256": selection.f2a_result_sha256,
        "f1_artifact_sha256": selection.f1_artifact_sha256,
        "f2a_artifact_sha256": selection.f2a_artifact_sha256,
        "f1_manifest_sha256": selection.f1_manifest_sha256,
        "f2a_manifest_sha256": selection.f2a_manifest_sha256,
        "control_source_sha256": selection.control_source.sha256,
        "control_source_artifact_sha256": selection.control_source_artifact_sha256,
        "endpoint_input_sha256": endpoint_input.sha256,
        "activation_manifest_sha256": activation_manifest.sha256,
        "unrelated_manifest_sha256": unrelated_manifest.sha256,
        "confirmatory_pins_sha256": confirmatory_pins.sha256,
    }
    shard_id = f"intervention-raw-outcomes-{selection.sha256[:16]}"
    try:
        shard = store.write_completed_shard(
            run_id,
            "intervention_test",
            shard_id,
            rows,
            lineage,
            record_kind="raw_intervention_outcomes",
        )
    except FileExistsError:
        candidate = (
            store.root
            / "runs"
            / "familiarity_answerability"
            / run_id
            / "shards"
            / "intervention_test"
            / f"{shard_id}.jsonl.manifest.json"
        )
        shard = store.verify_shard(candidate)
        expected = hashlib.sha256(
            b"".join(_canonical_json(row) + b"\n" for row in rows)
        ).hexdigest()
        if shard.record_kind != "raw_intervention_outcomes" or shard.sha256 != expected:
            raise ValueError("existing raw intervention artifact does not match executed outcomes")
    return store.verify_shard(shard.manifest_path)


def _verify_raw_outcomes_shard(
    store: FAArtifactStore, raw_shard: Any, metrics_lineage: Mapping[str, Any]
) -> None:
    verified = store.verify_shard(raw_shard.manifest_path)
    if (
        verified.record_kind != "raw_intervention_outcomes"
        or verified.sha256 != metrics_lineage.get("raw_outcomes_manifest_sha256")
        or verified.row_count < 1
    ):
        raise ValueError("raw intervention artifact does not verify before endpoint closure")
    rows = _read_canonical_rows(verified)
    if any(
        set(row) != {"kind", "outcome_kind", "outcome"}
        or row.get("kind") != "raw_intervention_outcomes"
        or row.get("outcome_kind")
        not in {"raw_intervention_outcome", "raw_unrelated_outcome"}
        or not isinstance(row.get("outcome"), Mapping)
        for row in rows
    ):
        raise ValueError("raw intervention artifact has an invalid evidence schema")


def _h7_crossed_bootstrap(
    high: Sequence[RawInterventionOutcome], low: Sequence[RawInterventionOutcome]
) -> Mapping[str, Any]:
    cells = {"high_to_low": tuple(high), "low_to_high": tuple(low)}
    if any(not values for values in cells.values()):
        raise ValueError("H7 bootstrap requires raw outcomes for both directions")
    rng = np.random.default_rng(_H7_BOOTSTRAP_SEED)
    entities = tuple(
        sorted({outcome.entity_unit_id for values in cells.values() for outcome in values})
    )
    templates = tuple(
        sorted({outcome.template_family for values in cells.values() for outcome in values})
    )
    if not entities or not templates:
        raise ValueError("H7 bootstrap requires entity and template identifiers")
    entity_index = {value: index for index, value in enumerate(entities)}
    template_index = {value: index for index, value in enumerate(templates)}
    indexed = {
        direction: (
            np.asarray(
                [entity_index[outcome.entity_unit_id] for outcome in outcomes]
            ),
            np.asarray(
                [template_index[outcome.template_family] for outcome in outcomes]
            ),
            _oriented_effects(outcomes),
        )
        for direction, outcomes in cells.items()
    }
    accepted: dict[str, list[float]] = {direction: [] for direction in cells}
    discarded_draws = 0
    for _ in range(_H7_BOOTSTRAP_REPLICATES):
        entity_counts = np.bincount(
            rng.integers(len(entities), size=len(entities)), minlength=len(entities)
        )
        template_counts = np.bincount(
            rng.integers(len(templates), size=len(templates)), minlength=len(templates)
        )
        draw: dict[str, float] = {}
        for direction, (entity_rows, template_rows, effects) in indexed.items():
            weights = entity_counts[entity_rows] * template_counts[template_rows]
            if int(np.sum(weights)) == 0:
                draw = {}
                break
            draw[direction] = float(np.average(effects, weights=weights))
        if not draw:
            discarded_draws += 1
            continue
        for direction, estimate in draw.items():
            accepted[direction].append(estimate)
    valid_draws = _H7_BOOTSTRAP_REPLICATES - discarded_draws
    if valid_draws == 0:
        raise ValueError("H7 bootstrap produced no valid crossed draws")

    raw: dict[str, Mapping[str, Any]] = {}
    samples_by_direction: dict[str, np.ndarray] = {}
    for direction, outcomes in cells.items():
        effects = _oriented_effects(outcomes)
        samples = np.asarray(accepted[direction], dtype=np.float64)
        point = float(np.mean(effects))
        raw_p = float((np.count_nonzero(samples <= 0.0) + 1) / (samples.size + 1))
        raw[direction] = {
            "point_estimate": point,
            "raw_interval": [
                float(np.quantile(samples, _H7_ALPHA / 2.0)),
                float(np.quantile(samples, 1.0 - _H7_ALPHA / 2.0)),
            ],
            "raw_p": raw_p,
            "entities": list(entities),
            "template_families": list(templates),
        }
        samples_by_direction[direction] = samples
    ordered = sorted(raw, key=lambda direction: (raw[direction]["raw_p"], direction))
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, direction in enumerate(ordered):
        running = max(running, min(1.0, float(raw[direction]["raw_p"]) * (len(ordered) - index)))
        adjusted[direction] = running
        alpha = _H7_ALPHA / (len(ordered) - index)
        # The Holm rank-specific two-sided interval is the interval paired with its test.
        raw[direction] = {
            **raw[direction],
            "holm_interval": [
                float(np.quantile(samples_by_direction[direction], alpha / 2.0)),
                float(np.quantile(samples_by_direction[direction], 1.0 - alpha / 2.0)),
            ],
            "holm_adjusted_p": adjusted[direction],
        }
    return _deep_freeze(
        {
            "method": "crossed_entity_unit_template_family_bootstrap",
            "seed": _H7_BOOTSTRAP_SEED,
            "replicates": _H7_BOOTSTRAP_REPLICATES,
            "requested_draws": _H7_BOOTSTRAP_REPLICATES,
            "valid_draws": valid_draws,
            "discarded_draws": discarded_draws,
            "resampling_unit": ["entity_unit_id", "template_family"],
            "alpha": _H7_ALPHA,
            "directions": raw,
        }
    )


def _oriented_effects(outcomes: Sequence[RawInterventionOutcome]) -> np.ndarray:
    return np.asarray(
        [
            outcome.patched_answer_attempt - outcome.baseline_answer_attempt
            if outcome.direction == "high_to_low"
            else outcome.baseline_answer_attempt - outcome.patched_answer_attempt
            for outcome in outcomes
        ],
        dtype=np.float64,
    )


def _mean_interval(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("cannot compute an interval without raw outcomes")
    mean = float(np.mean(values))
    if values.size == 1:
        return (mean, mean)
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    radius = 1.96 * standard_error
    return (mean - radius, mean + radius)


def _selection_sha256(selection: InterventionSelection) -> str:
    return _hash(
        {
            "layer": selection.layer,
            "anchor": selection.anchor,
            "method": selection.method,
            "alpha": selection.alpha,
            "source_split": selection.source_split,
            "direction_sha256": selection.direction_sha256,
            "preregistration_sha256": selection.preregistration_sha256,
            "probe_selection_sha256": selection.probe_selection_sha256,
            "confirmatory_pins_sha256": selection.confirmatory_pins_sha256,
            "f1_evidence_sha256": selection.f1_evidence_sha256,
            "f2a_evidence_sha256": selection.f2a_evidence_sha256,
            "f1_result_sha256": selection.f1_result_sha256,
            "f2a_result_sha256": selection.f2a_result_sha256,
            "f1_artifact_sha256": selection.f1_artifact_sha256,
            "f2a_artifact_sha256": selection.f2a_artifact_sha256,
            "f1_manifest_sha256": selection.f1_manifest_sha256,
            "f2a_manifest_sha256": selection.f2a_manifest_sha256,
            "control_source": selection.control_source.canonical_payload(),
            "control_source_artifact_sha256": selection.control_source_artifact_sha256,
            "validation_metrics": selection.validation_metrics.canonical_payload(),
            "candidate_count": selection.candidate_count,
        }
    )


def _result_sha256(result: InterventionTestResult) -> str:
    return _hash(
        {
            "selection_sha256": result.selection_sha256,
            "preregistration_sha256": result.preregistration_sha256,
            "example_ids": list(result.example_ids),
            "metrics": result.metrics.canonical_payload(),
            "h7_passed": result.h7_passed,
            "h8_passed": result.h8_passed,
            "refit_performed": result.refit_performed,
        }
    )


def _result_payload(result: InterventionTestResult) -> Mapping[str, Any]:
    return {
        "selection_sha256": result.selection_sha256,
        "preregistration_sha256": result.preregistration_sha256,
        "example_ids": list(result.example_ids),
        "metrics": result.metrics.canonical_payload(),
        "h7_passed": result.h7_passed,
        "h8_passed": result.h8_passed,
        "refit_performed": result.refit_performed,
        "result_sha256": result.result_sha256,
    }


def _artifact_run_id(store: FAArtifactStore, data_path: Any) -> str:
    base = store.root / "runs" / "familiarity_answerability"
    try:
        relative = data_path.relative_to(base)
    except (AttributeError, ValueError) as error:
        raise ValueError("endpoint artifact is outside the FA run namespace") from error
    if len(relative.parts) != 4 or relative.parts[1:3] != (
        "shards",
        "intervention_test",
    ):
        raise ValueError("endpoint artifact path has an invalid run identity")
    return relative.parts[0]


def _array_bytes(array: np.ndarray) -> bytes:
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return header + b"\n" + array.tobytes(order="C")


def _parse_decoded_patch(value: Any) -> Mapping[str, Any]:
    required = {
        "baseline_text",
        "patched_text",
        "baseline_truncated",
        "patched_truncated",
        "baseline_infrastructure_marked",
        "patched_infrastructure_marked",
        "baseline_readouts",
        "patched_readouts",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("decoded outcome must contain only registered text and readouts")
    parsed: dict[str, Any] = {}
    for name in ("baseline_text", "patched_text"):
        text = value[name]
        if text is not None and not isinstance(text, str):
            raise ValueError(f"{name} must be text or None")
        parsed[name] = text
    for name in (
        "baseline_truncated",
        "patched_truncated",
        "baseline_infrastructure_marked",
        "patched_infrastructure_marked",
    ):
        if type(value[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
        parsed[name] = value[name]
    for name in ("baseline_readouts", "patched_readouts"):
        parsed[name] = ReadoutSnapshot.from_mapping(value[name], name).canonical_payload()
    return parsed


def _unrelated_invalid(decoded: Mapping[str, Any], prefix: str) -> int:
    text = decoded[f"{prefix}_text"]
    marked = decoded[f"{prefix}_infrastructure_marked"]
    truncated = decoded[f"{prefix}_truncated"]
    if text is None or marked or truncated:
        return 1
    normalized = text.strip()
    return int(
        not normalized
        or "\n" in normalized
        or "\r" in normalized
        or not normalized.isprintable()
    )


def _finite_vector(
    value: np.ndarray,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    vector = np.array(value, dtype=np.float64, copy=True, order="C")
    if vector.ndim != 1 or vector.size < 2 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    if shape is not None and vector.shape != shape:
        raise ValueError(f"{name} shape does not match")
    return vector


def _vector(value: np.ndarray, name: str, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    vector = _finite_vector(value, name, shape=shape)
    if np.linalg.norm(vector) == 0:
        raise ValueError(f"{name} must have nonzero norm")
    return vector


def _match_norm(vector: np.ndarray, norm: float) -> np.ndarray:
    source_norm = float(np.linalg.norm(vector))
    if source_norm <= 1e-12:
        raise ValueError("control vector cannot be normalized")
    return np.asarray(vector * (norm / source_norm), dtype=np.float64)


def _hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _prefix_evidence_sha256(
    rendered_prefix_utf8: bytes,
    tokenizer_revision: str,
    chat_template_sha256: str,
) -> str:
    rendered = bytes(rendered_prefix_utf8)
    if not rendered:
        raise ValueError("rendered prefix evidence must be nonempty")
    return _hash(
        {
            "rendered_prefix_utf8_sha256": hashlib.sha256(rendered).hexdigest(),
            "tokenizer_revision": tokenizer_revision,
            "chat_template_sha256": chat_template_sha256,
        }
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = {
            str(key): _deep_freeze(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
        return MappingProxyType(frozen)
    if isinstance(value, np.ndarray):
        return tuple(_deep_freeze(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_deep_freeze(item) for item in value), key=repr))
    if value is None or type(value) in {str, int, float, bool, bytes}:
        return value
    raise ValueError(f"unsupported mutable payload type: {type(value).__name__}")


def _thaw_frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_frozen(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_frozen(item) for item in value]
    return value


def _nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _revision(value: Any, name: str) -> None:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase immutable revision")
