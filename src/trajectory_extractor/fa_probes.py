"""Leakage-resistant F2A probe selection, frozen evaluation, nulls, and OOD metrics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from trajectory_extractor.fa_artifacts import FAArtifactStore, UnlockReceipt
from trajectory_extractor.fa_config import CONFIRMATORY_THRESHOLDS, REGISTERED_ANCHORS


REGISTERED_LAYERS = tuple(range(26))
REGISTERED_DOMAINS = ("person", "place", "organization", "creative_work")
PCA_OPTIONS = (None, 16, 32, 64)
C_OPTIONS = (0.01, 0.1, 1.0, 10.0)
DEFAULT_CONFIRMATORY_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260722
_BOOTSTRAP_DRAW_OVERRIDE_FOR_TESTS: int | None = None
_TRAIN_ONLY_CV_FAST_PATH_FOR_TESTS = False
DEFAULT_FULL_SELECTION_NULL_SEEDS = tuple(range(2026072201, 2026072300))
DEFAULT_FULL_SELECTION_NULL_SEED_HASH = (
    "7aee4f4ee03201f4a8b7bee296294bc5c6a14a5251dfa71bb8cff15ce3d4e07f"
)
OUTPUT_CONTROL_FEATURE_NAMES = (
    "target_sequence_logp",
    "unknown_sequence_logp",
    "target_minus_unknown_logp",
    "maximum_sequence_logp",
    "candidate_logsumexp",
    "normalized_target_probability",
    "normalized_unknown_probability",
    "binary_candidate_entropy",
    "absolute_normalized_probability_margin",
    "signed_probability_margin",
    "maximum_candidate_confidence",
)
OUTPUT_CONTROL_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        {"dimension": 11, "ordered_feature_names": list(OUTPUT_CONTROL_FEATURE_NAMES)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
REGISTERED_BASELINES = (
    "surface",
    "output_margin",
    "residual_static",
    "static_plus_dynamics",
    "final_layer_excluded",
)
SAE_FAMILIES = ("sae_1_sparse", "sae_small_sparse")
NESTED_H5_BASELINE = "surface_plus_output_margin"
NESTED_H5_CANDIDATE = "surface_output_static"
NESTED_H6_CANDIDATE = "surface_output_static_dynamics"
DEFAULT_FEATURE_FAMILIES = REGISTERED_BASELINES + (
    NESTED_H5_BASELINE,
    NESTED_H5_CANDIDATE,
    NESTED_H6_CANDIDATE,
)
TASKS = ("familiarity", "answerability", "unsupported_answer")
ANSWERABILITY_CLASSES = ("target_bound", "distractor_bound", "code_absent")
TARGET_FAMILIARITY_CONDITIONS = ("screened_real", "matched_synthetic")
OUTCOME_STATUSES = ("valid", "missing", "invalid")
PROTECTED_SPLITS = frozenset({"behavior_test", "probe_test", "intervention_test"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ANCHOR_INDEX = {name: index for index, name in enumerate(REGISTERED_ANCHORS)}


def _registered_rotation_specs(task: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if task == "familiarity":
        conditions = ANSWERABILITY_CLASSES
    elif task == "answerability":
        conditions = TARGET_FAMILIARITY_CONDITIONS
    else:
        return ()
    return tuple(
        (condition, tuple(other for other in conditions if other != condition))
        for condition in conditions
    )


def _transfer_condition(row: "ProbeRow", task: str) -> str:
    if task == "familiarity":
        return row.answerability_condition
    if task == "answerability":
        return row.target_familiarity_condition
    raise ValueError("task has no registered cross-condition transfer")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be nonempty text")
    return value


def _required_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _readonly_array(value: Any, field_name: str, *, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim:
        raise ValueError(f"{field_name} must have {ndim} dimensions")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, np.ndarray):
        return _freeze_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@dataclass(frozen=True)
class ProbeRow:
    """One fully provenance-bound row used by probe selection or evaluation."""

    example_id: str
    split: str
    task: str
    label: int | str
    entity_id: str
    template_id: str
    relation_id: str
    domain: str
    condition: str
    answerability_condition: str
    target_familiarity_condition: str
    distractor_familiarity_condition: str
    surface_features: tuple[float, ...]
    output_margin_features: tuple[float, ...]
    residual_features: np.ndarray
    sae_features: np.ndarray | None
    outcome_status: str
    source_sha256: str
    activation_sha256: str
    metadata_manifest_sha256: str
    metadata_row_sha256: str
    output_control_schema_sha256: str
    output_evidence_sha256: str

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ProbeRow":
        expected = {
            "example_id", "split", "task", "label", "entity_id", "template_id",
            "relation_id", "domain", "condition", "answerability_condition",
            "target_familiarity_condition", "distractor_familiarity_condition",
            "surface_features", "output_margin_features", "residual_features",
            "sae_features", "outcome_status", "source_sha256", "activation_sha256",
            "metadata_manifest_sha256", "metadata_row_sha256",
            "output_control_schema_sha256", "output_evidence_sha256",
        }
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ValueError("probe row has an invalid schema")
        try:
            loaded = cls(
                **{
                    **dict(record),
                    "surface_features": tuple(record["surface_features"]),
                    "output_margin_features": tuple(record["output_margin_features"]),
                    "residual_features": np.asarray(record["residual_features"], dtype=np.float64),
                    "sae_features": (
                        None
                        if record["sae_features"] is None
                        else np.asarray(record["sae_features"], dtype=np.float64)
                    ),
                }
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"probe row is invalid: {error}") from error
        if loaded.to_record() != dict(record):
            raise ValueError("probe row is not canonical")
        return loaded

    def __post_init__(self) -> None:
        for name in (
            "example_id",
            "split",
            "entity_id",
            "template_id",
            "relation_id",
            "domain",
            "condition",
        ):
            _required_text(getattr(self, name), name)
        if self.answerability_condition not in ANSWERABILITY_CLASSES:
            raise ValueError("answerability_condition is not registered")
        if self.target_familiarity_condition not in TARGET_FAMILIARITY_CONDITIONS:
            raise ValueError("target_familiarity_condition is not registered")
        if self.distractor_familiarity_condition not in TARGET_FAMILIARITY_CONDITIONS:
            raise ValueError("distractor_familiarity_condition is not registered")
        if self.task not in TASKS:
            raise ValueError("task is not registered")
        if self.task == "answerability":
            if self.label not in ANSWERABILITY_CLASSES:
                raise ValueError("answerability label must use the frozen three-state target")
        elif type(self.label) is not int or self.label not in {0, 1}:
            raise ValueError("binary task label must be 0 or 1")
        if (
            self.task == "unsupported_answer"
            and self.answerability_condition == "target_bound"
        ):
            raise ValueError(
                "unsupported-answer rows require an evidence-absent condition"
            )
        if self.outcome_status not in OUTCOME_STATUSES:
            raise ValueError("outcome_status is not registered")
        surface = tuple(float(value) for value in self.surface_features)
        output = tuple(float(value) for value in self.output_margin_features)
        residual = _readonly_array(self.residual_features, "residual_features", ndim=3)
        sae = (
            None
            if self.sae_features is None
            else _readonly_array(self.sae_features, "sae_features", ndim=3)
        )
        if residual.shape[0] != len(REGISTERED_ANCHORS) or residual.shape[1] != 26:
            raise ValueError("residual_features must have shape [3 registered anchors, 26 layers, hidden]")
        if sae is not None and sae.shape[:2] != (len(REGISTERED_ANCHORS), 26):
            raise ValueError("sae_features must have shape [3 registered anchors, 26 layers, feature]")
        if (
            not np.isfinite(surface).all()
            or not np.isfinite(output).all()
            or not np.isfinite(residual).all()
            or (sae is not None and not np.isfinite(sae).all())
        ):
            raise ValueError("probe row features must be finite for canonical provenance")
        object.__setattr__(self, "surface_features", surface)
        object.__setattr__(self, "output_margin_features", output)
        object.__setattr__(self, "residual_features", residual)
        object.__setattr__(self, "sae_features", sae)
        _required_sha256(self.source_sha256, "source_sha256")
        _required_sha256(self.activation_sha256, "activation_sha256")
        _required_sha256(self.metadata_manifest_sha256, "metadata_manifest_sha256")
        _required_sha256(self.metadata_row_sha256, "metadata_row_sha256")
        if self.output_control_schema_sha256 != OUTPUT_CONTROL_SCHEMA_SHA256:
            raise ValueError("output control schema hash does not match the registered 11D schema")
        _required_sha256(self.output_evidence_sha256, "output_evidence_sha256")

    def to_record(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "task": self.task,
            "label": self.label,
            "entity_id": self.entity_id,
            "template_id": self.template_id,
            "relation_id": self.relation_id,
            "domain": self.domain,
            "condition": self.condition,
            "answerability_condition": self.answerability_condition,
            "target_familiarity_condition": self.target_familiarity_condition,
            "distractor_familiarity_condition": self.distractor_familiarity_condition,
            "surface_features": list(self.surface_features),
            "output_margin_features": list(self.output_margin_features),
            "residual_features": self.residual_features.tolist(),
            "sae_features": None if self.sae_features is None else self.sae_features.tolist(),
            "outcome_status": self.outcome_status,
            "source_sha256": self.source_sha256,
            "activation_sha256": self.activation_sha256,
            "metadata_manifest_sha256": self.metadata_manifest_sha256,
            "metadata_row_sha256": self.metadata_row_sha256,
            "output_control_schema_sha256": self.output_control_schema_sha256,
            "output_evidence_sha256": self.output_evidence_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())


@dataclass(frozen=True)
class ProbeSourceIdentity:
    """Pre-outcome identity available when the protected prompt is sealed."""

    example_id: str
    canonical_payload_sha256: str

    def __post_init__(self) -> None:
        _required_text(self.example_id, "example_id")
        _required_sha256(
            self.canonical_payload_sha256, "canonical_payload_sha256"
        )

    @classmethod
    def from_row(cls, row: ProbeRow) -> "ProbeSourceIdentity":
        if not isinstance(row, ProbeRow):
            raise ValueError("probe source identity requires a ProbeRow")
        return cls(row.example_id, row.source_sha256)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ProbeSourceIdentity":
        if not isinstance(record, Mapping):
            raise ValueError("sealed probe-test source identity has an invalid schema")
        required = {"example_id", "canonical_payload_sha256"}
        if set(record) != required:
            raise ValueError("sealed probe-test source identity has an invalid schema")
        return cls(record["example_id"], record["canonical_payload_sha256"])

    def to_record(self) -> dict[str, str]:
        return {
            "example_id": self.example_id,
            "canonical_payload_sha256": self.canonical_payload_sha256,
        }

    @property
    def probe_row_sha256(self) -> str:
        """Compatibility accessor for older report fixtures."""

        return self.canonical_payload_sha256


def _canonical_source_identities(
    identities: Sequence[ProbeSourceIdentity], *, field_name: str
) -> tuple[ProbeSourceIdentity, ...]:
    values = tuple(identities)
    if not values or any(
        not isinstance(identity, ProbeSourceIdentity) for identity in values
    ):
        raise ValueError(f"{field_name} must contain ProbeSourceIdentity records")
    if len(
        {
            (identity.example_id, identity.canonical_payload_sha256)
            for identity in values
        }
    ) != len(values):
        raise ValueError(f"{field_name} contains duplicate source identities")
    return tuple(
        sorted(
            values,
            key=lambda value: (
                value.example_id,
                value.canonical_payload_sha256,
            ),
        )
    )


def _source_identity_digest(identities: Sequence[ProbeSourceIdentity]) -> str:
    return _digest([identity.to_record() for identity in identities])


# Backward-compatible names for persisted pre-outcome identities created before
# the source/row distinction was made explicit.
ProbeRowIdentity = ProbeSourceIdentity
_canonical_identities = _canonical_source_identities
_identity_digest = _source_identity_digest


@dataclass(frozen=True)
class SAEGate:
    original_loss: float
    reconstructed_loss: float
    ablated_loss: float
    finite_fraction: float
    recovery: float
    reasons: tuple[str, ...]
    blocking: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        values = (
            self.original_loss,
            self.reconstructed_loss,
            self.ablated_loss,
            self.finite_fraction,
            self.recovery,
        )
        if any(type(value) not in {int, float} or not math.isfinite(float(value)) for value in values):
            raise ValueError("SAE gate values must be finite numbers")
        if not 0.0 <= float(self.finite_fraction) <= 1.0:
            raise ValueError("SAE gate finite_fraction must be in [0, 1]")
        denominator = float(self.ablated_loss) - float(self.original_loss)
        if denominator <= 0.0:
            raise ValueError("SAE gate recovery denominator must be positive")
        expected_recovery = (
            float(self.ablated_loss) - float(self.reconstructed_loss)
        ) / denominator
        if not math.isclose(float(self.recovery), expected_recovery, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("SAE gate recovery is inconsistent with losses")
        reasons = tuple(self.reasons)
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("SAE gate reasons must be nonempty text")
        expected_reasons = []
        if float(self.finite_fraction) < CONFIRMATORY_THRESHOLDS["sae_finite_fraction_min"]:
            expected_reasons.append("finite fraction is below 0.95")
        if float(self.recovery) < CONFIRMATORY_THRESHOLDS["sae_loss_recovery_min"]:
            expected_reasons.append("loss recovery is below 0.70")
        if reasons != tuple(expected_reasons):
            raise ValueError("SAE gate reasons are inconsistent with thresholds")
        object.__setattr__(self, "reasons", reasons)

    @property
    def passed(self) -> bool:
        return (
            self.finite_fraction >= CONFIRMATORY_THRESHOLDS["sae_finite_fraction_min"]
            and self.recovery >= CONFIRMATORY_THRESHOLDS["sae_loss_recovery_min"]
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "original_loss": self.original_loss,
            "reconstructed_loss": self.reconstructed_loss,
            "ablated_loss": self.ablated_loss,
            "finite_fraction": self.finite_fraction,
            "recovery": self.recovery,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "blocking": False,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())


def audit_sae_transfer(
    original_loss: float,
    reconstructed_loss: float,
    ablated_loss: float,
    finite_fraction: float,
) -> SAEGate:
    values = (original_loss, reconstructed_loss, ablated_loss, finite_fraction)
    if any(type(value) not in {int, float} or not math.isfinite(float(value)) for value in values):
        raise ValueError("SAE transfer inputs must be finite numbers")
    original = float(original_loss)
    reconstructed = float(reconstructed_loss)
    ablated = float(ablated_loss)
    finite = float(finite_fraction)
    if not 0.0 <= finite <= 1.0:
        raise ValueError("finite_fraction must be in [0, 1]")
    denominator = ablated - original
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("SAE recovery denominator must be finite and positive")
    recovery = (ablated - reconstructed) / denominator
    if not math.isfinite(recovery):
        raise ValueError("SAE recovery must be finite")
    reasons = []
    if finite < CONFIRMATORY_THRESHOLDS["sae_finite_fraction_min"]:
        reasons.append("finite fraction is below 0.95")
    if recovery < CONFIRMATORY_THRESHOLDS["sae_loss_recovery_min"]:
        reasons.append("loss recovery is below 0.70")
    return SAEGate(original, reconstructed, ablated, finite, recovery, tuple(reasons))


@dataclass(frozen=True)
class CandidateScore:
    feature_family: str
    anchor: str
    layer: int | None
    pca_components: int | None
    c: float
    estimator: str
    status: str
    validation_log_loss: float | None
    validation_auroc: float | None
    validation_balanced_accuracy: float | None
    threshold: float | None
    reasons: tuple[str, ...]
    cross_condition_transfer: CrossConditionTransferSummary | None = None

    def __post_init__(self) -> None:
        if self.status not in {"evaluable", "ineligible", "not_evaluable"}:
            raise ValueError("candidate status is invalid")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if self.cross_condition_transfer is not None and not isinstance(
            self.cross_condition_transfer, CrossConditionTransferSummary
        ):
            raise ValueError("candidate transfer result must be typed")

    def to_record(self) -> dict[str, Any]:
        return {
            "feature_family": self.feature_family,
            "anchor": self.anchor,
            "layer": self.layer,
            "pca_components": self.pca_components,
            "c": self.c,
            "estimator": self.estimator,
            "status": self.status,
            "validation_log_loss": self.validation_log_loss,
            "validation_auroc": self.validation_auroc,
            "validation_balanced_accuracy": self.validation_balanced_accuracy,
            "threshold": self.threshold,
            "reasons": list(self.reasons),
            "cross_condition_transfer": (
                None
                if self.cross_condition_transfer is None
                else self.cross_condition_transfer.to_record()
            ),
        }


@dataclass(frozen=True)
class FrozenProbeModel:
    feature_family: str
    anchor: str
    layer: int | None
    pca_components: int | None
    c: float
    estimator: str
    threshold: float | None
    classes: tuple[int | str, ...]
    selector_indices: tuple[int, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    pca_mean: tuple[float, ...]
    pca_components_matrix: tuple[tuple[float, ...], ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    validation_log_loss: float
    validation_auroc: float
    validation_balanced_accuracy: float
    claim_scope: str

    def __post_init__(self) -> None:
        if self.anchor not in REGISTERED_ANCHORS:
            raise ValueError("model anchor is not registered")
        if self.layer is not None and self.layer not in REGISTERED_LAYERS:
            raise ValueError("model layer is not registered")
        if self.feature_family == "final_layer_excluded" and self.layer == 25:
            raise ValueError("final_layer_excluded cannot contain layer 25")
        if self.anchor == "assistant_prefix_end" and self.claim_scope != "output_proximal_control":
            raise ValueError("assistant_prefix_end must remain an output-proximal control")
        for name in (
            "selector_indices",
            "scaler_mean",
            "scaler_scale",
            "pca_mean",
            "classes",
            "intercepts",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "pca_components_matrix",
            tuple(tuple(row) for row in self.pca_components_matrix),
        )
        object.__setattr__(self, "coefficients", tuple(tuple(row) for row in self.coefficients))
        if self.classes not in ((0, 1), ANSWERABILITY_CLASSES):
            raise ValueError("frozen model classes are not registered")
        expected_rows = 1 if self.classes == (0, 1) else len(self.classes)
        if len(self.coefficients) != expected_rows or len(self.intercepts) != expected_rows:
            raise ValueError("frozen model coefficient shape does not match classes")
        if self.classes == (0, 1) and self.threshold is None:
            raise ValueError("binary frozen model requires a threshold")
        if self.classes == ANSWERABILITY_CLASSES and self.threshold is not None:
            raise ValueError("multiclass frozen model cannot use a binary threshold")

    def to_record(self) -> dict[str, Any]:
        return {
            "feature_family": self.feature_family,
            "anchor": self.anchor,
            "layer": self.layer,
            "pca_components": self.pca_components,
            "c": self.c,
            "estimator": self.estimator,
            "threshold": self.threshold,
            "classes": list(self.classes),
            "selector_indices": list(self.selector_indices),
            "scaler_mean": list(self.scaler_mean),
            "scaler_scale": list(self.scaler_scale),
            "pca_mean": list(self.pca_mean),
            "pca_components_matrix": [list(row) for row in self.pca_components_matrix],
            "coefficients": [list(row) for row in self.coefficients],
            "intercepts": list(self.intercepts),
            "validation_log_loss": self.validation_log_loss,
            "validation_auroc": self.validation_auroc,
            "validation_balanced_accuracy": self.validation_balanced_accuracy,
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def predict_proba(self, rows: Sequence[ProbeRow]) -> np.ndarray:
        matrix = _feature_matrix(rows, self.feature_family, self.anchor, self.layer)
        if self.selector_indices:
            matrix = matrix[:, self.selector_indices]
        scaled = (matrix - np.asarray(self.scaler_mean)) / np.asarray(self.scaler_scale)
        if self.pca_components is not None:
            scaled = (scaled - np.asarray(self.pca_mean)) @ np.asarray(
                self.pca_components_matrix
            ).T
        logits = scaled @ np.asarray(self.coefficients).T + np.asarray(self.intercepts)
        if self.classes == ANSWERABILITY_CLASSES:
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            return exponentials / exponentials.sum(axis=1, keepdims=True)
        positive = logits >= 0
        scores = np.empty_like(logits, dtype=np.float64)
        scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        scores[~positive] = exp_logits / (1.0 + exp_logits)
        positive_scores = scores[:, 0]
        return np.column_stack((1.0 - positive_scores, positive_scores))


@dataclass(frozen=True)
class SelectionManifest:
    schema_version: int
    task: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    train_row_sha256s: tuple[str, ...]
    validation_row_sha256s: tuple[str, ...]
    train_entity_ids: tuple[str, ...]
    validation_entity_ids: tuple[str, ...]
    train_template_ids: tuple[str, ...]
    validation_template_ids: tuple[str, ...]
    train_relation_ids: tuple[str, ...]
    validation_relation_ids: tuple[str, ...]
    train_domain_ids: tuple[str, ...]
    validation_domain_ids: tuple[str, ...]
    models: tuple[FrozenProbeModel, ...]
    selected_feature_family: str
    candidate_scores: tuple[CandidateScore, ...]
    registered_layers: tuple[int, ...]
    registered_anchors: tuple[str, ...]
    pca_options: tuple[int | None, ...]
    c_options: tuple[float, ...]
    seed: int
    sae_gate_sha256: str | None
    null_provenance: Mapping[str, Any] | None
    transfer_rotations: tuple[FrozenTransferRotation, ...] = ()

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SelectionManifest":
        return _selection_manifest_from_record(record)

    def __post_init__(self) -> None:
        if self.schema_version != 2 or self.task not in TASKS:
            raise ValueError("selection manifest schema or task is invalid")
        tuple_fields = (
            "train_ids",
            "validation_ids",
            "train_row_sha256s",
            "validation_row_sha256s",
            "train_entity_ids",
            "validation_entity_ids",
            "train_template_ids",
            "validation_template_ids",
            "train_relation_ids",
            "validation_relation_ids",
            "train_domain_ids",
            "validation_domain_ids",
            "models",
            "candidate_scores",
            "registered_layers",
            "registered_anchors",
            "pca_options",
            "c_options",
            "transfer_rotations",
        )
        for name in tuple_fields:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "null_provenance", _freeze_mapping(self.null_provenance))
        if self.registered_layers != REGISTERED_LAYERS:
            raise ValueError("selection must record all 26 registered layers")
        if self.registered_anchors != tuple(REGISTERED_ANCHORS):
            raise ValueError("selection must record all registered anchors")
        if not self.models or self.selected_feature_family not in {
            model.feature_family for model in self.models if model.claim_scope == "pre_output"
        }:
            raise ValueError("selection must contain its selected feature family")
        if len({(model.feature_family, model.claim_scope) for model in self.models}) != len(
            self.models
        ):
            raise ValueError("selection can contain only one model per family and claim scope")
        if set(self.train_ids) & set(self.validation_ids):
            raise ValueError("selection IDs overlap")
        if self.sae_gate_sha256 is not None:
            _required_sha256(self.sae_gate_sha256, "sae_gate_sha256")
        observed_rotations = tuple(
            (rotation.train_condition, rotation.test_conditions)
            for rotation in self.transfer_rotations
            if isinstance(rotation, FrozenTransferRotation)
        )
        expected_rotations = _registered_rotation_specs(self.task)
        if self.task in {"familiarity", "answerability"}:
            if (
                len(observed_rotations) != len(self.transfer_rotations)
                or observed_rotations != expected_rotations
                or any(
                    rotation.task != self.task
                    or rotation.model.feature_family != self.selected_feature_family
                    or rotation.model.claim_scope != "pre_output"
                    for rotation in self.transfer_rotations
                )
            ):
                raise ValueError("selection requires exact registered transfer rotations")
        elif self.transfer_rotations:
            raise ValueError("unsupported-answer selection cannot contain transfer rotations")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "train_row_sha256s": list(self.train_row_sha256s),
            "validation_row_sha256s": list(self.validation_row_sha256s),
            "train_entity_ids": list(self.train_entity_ids),
            "validation_entity_ids": list(self.validation_entity_ids),
            "train_template_ids": list(self.train_template_ids),
            "validation_template_ids": list(self.validation_template_ids),
            "train_relation_ids": list(self.train_relation_ids),
            "validation_relation_ids": list(self.validation_relation_ids),
            "train_domain_ids": list(self.train_domain_ids),
            "validation_domain_ids": list(self.validation_domain_ids),
            "models": [model.to_record() for model in self.models],
            "selected_feature_family": self.selected_feature_family,
            "candidate_scores": [score.to_record() for score in self.candidate_scores],
            "registered_layers": list(self.registered_layers),
            "registered_anchors": list(self.registered_anchors),
            "pca_options": list(self.pca_options),
            "c_options": list(self.c_options),
            "seed": self.seed,
            "sae_gate_sha256": self.sae_gate_sha256,
            "null_provenance": _thaw(self.null_provenance),
            "cross_condition_transfer_rotations": [
                rotation.to_record() for rotation in self.transfer_rotations
            ],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def model_for(
        self, feature_family: str, *, claim_scope: str = "pre_output"
    ) -> FrozenProbeModel:
        for model in self.models:
            if model.feature_family == feature_family and model.claim_scope == claim_scope:
                return model
        raise KeyError((feature_family, claim_scope))


def _task_anchors(task: str) -> tuple[str, ...]:
    if task == "familiarity":
        return ("target_intro_end",)
    if task == "answerability":
        return ("user_prompt_end",)
    if task == "unsupported_answer":
        return ("user_prompt_end", "assistant_prefix_end")
    raise ValueError("task is not registered")


def _task_classes(task: str) -> tuple[int | str, ...]:
    if task == "answerability":
        return ANSWERABILITY_CLASSES
    if task in {"familiarity", "unsupported_answer"}:
        return (0, 1)
    raise ValueError("task is not registered")


def _validate_rows(rows: Sequence[ProbeRow], expected_split: str, field_name: str) -> tuple[ProbeRow, ...]:
    source = tuple(rows)
    if not source or any(not isinstance(row, ProbeRow) for row in source):
        raise ValueError(f"{field_name} must contain ProbeRow records")
    if any(row.split in PROTECTED_SPLITS for row in source):
        raise ValueError(f"{field_name} contains a protected split")
    if any(row.split != expected_split for row in source):
        raise ValueError(f"{field_name} must use {expected_split}")
    if len({row.example_id for row in source}) != len(source):
        raise ValueError(f"{field_name} contains duplicate example IDs")
    if any(len(row.output_margin_features) != 11 for row in source):
        raise ValueError("output_margin features must be exactly 11-dimensional")
    if any(row.outcome_status != "valid" for row in source):
        raise ValueError(f"{field_name} cannot contain missing or invalid rows")
    if any(
        not np.isfinite(row.residual_features).all()
        or not np.isfinite(row.surface_features).all()
        or not np.isfinite(row.output_margin_features).all()
        for row in source
    ):
        raise ValueError(f"{field_name} features must be finite")
    return source


def _validate_options(
    feature_families: Sequence[str],
    pca_options: Sequence[int | None],
    c_options: Sequence[float],
    sae_gate: SAEGate | None,
) -> tuple[tuple[str, ...], tuple[int | None, ...], tuple[float, ...]]:
    requested_families = tuple(feature_families)
    if requested_families != DEFAULT_FEATURE_FAMILIES:
        raise ValueError("feature_families must contain every required baseline in registered order")
    families = (
        DEFAULT_FEATURE_FAMILIES + SAE_FAMILIES
        if isinstance(sae_gate, SAEGate) and sae_gate.passed
        else DEFAULT_FEATURE_FAMILIES
    )
    pca_values = tuple(pca_options)
    c_values = tuple(float(value) for value in c_options)
    if pca_values != PCA_OPTIONS:
        raise ValueError("fit_selection requires the exact registered PCA grid")
    if c_values != C_OPTIONS:
        raise ValueError("fit_selection requires the exact registered C grid")
    return families, pca_values, c_values


def fit_selection(
    train_rows: Sequence[ProbeRow],
    validation_rows: Sequence[ProbeRow],
    *,
    estimators: Sequence[Any] | None = None,
    feature_families: Sequence[str] = DEFAULT_FEATURE_FAMILIES,
    pca_options: Sequence[int | None] = PCA_OPTIONS,
    c_options: Sequence[float] = C_OPTIONS,
    protected_test_ids: Collection[str] = (),
    sae_gate: SAEGate | None = None,
    seed: int = 20260722,
    null_provenance: Mapping[str, Any] | None = None,
) -> SelectionManifest:
    """Fit preprocessing and classifiers on train; use validation only for selection."""

    train = _validate_rows(train_rows, "mechanism_train", "train_rows")
    validation = _validate_rows(validation_rows, "locked_validation", "validation_rows")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    protected = set(protected_test_ids)
    if protected & {row.example_id for row in (*train, *validation)}:
        raise ValueError("protected test ID entered probe selection")
    train_ids = {row.example_id for row in train}
    validation_ids = {row.example_id for row in validation}
    if train_ids & validation_ids:
        raise ValueError("train/validation example leakage")
    _reject_group_overlap(train, validation, "entity_id", "entity leakage")
    _reject_group_overlap(train, validation, "template_id", "template leakage")
    _reject_group_overlap(train, validation, "relation_id", "relation leakage")
    _require_registered_domains(train, "train_rows")
    _require_registered_domains(validation, "validation_rows")
    tasks = {row.task for row in (*train, *validation)}
    if len(tasks) != 1:
        raise ValueError("selection rows must contain exactly one task")
    task = tasks.pop()
    classes = _task_classes(task)
    if {row.label for row in train} != set(classes) or {
        row.label for row in validation
    } != set(classes):
        raise ValueError("train and validation must each contain every registered task class")
    _validate_transfer_factors(train, task, "train_rows")
    _validate_transfer_factors(validation, task, "validation_rows")
    families, pca_values, c_values = _validate_options(
        feature_families, pca_options, c_options, sae_gate
    )
    if set(families) & set(SAE_FAMILIES):
        if any(row.sae_features is None for row in (*train, *validation)):
            raise ValueError("SAE candidates require audited SAE features on every row")
        if any(not np.isfinite(row.sae_features).all() for row in (*train, *validation)):
            raise ValueError("SAE features must be finite after the transfer gate")
    prototypes = tuple(estimators) if estimators is not None else (None,)
    if not prototypes:
        raise ValueError("estimators cannot be empty")

    scores: list[CandidateScore] = []
    best_by_family: dict[
        tuple[str, str],
        tuple[tuple[Any, ...], FrozenProbeModel, tuple[FrozenTransferRotation, ...]],
    ] = {}
    for family in families:
        for anchor in _task_anchors(task):
            layers: Sequence[int | None]
            if family in {"surface", "output_margin", NESTED_H5_BASELINE}:
                layers = (None,)
            elif family in {"final_layer_excluded", NESTED_H5_CANDIDATE, NESTED_H6_CANDIDATE}:
                layers = REGISTERED_LAYERS[:-1]
            else:
                layers = REGISTERED_LAYERS
            for layer in layers:
                if (
                    anchor != "assistant_prefix_end"
                    and layer == 25
                    and family not in {"surface", "output_margin", NESTED_H5_BASELINE}
                ):
                    scores.append(
                        CandidateScore(
                            family,
                            anchor,
                            layer,
                            None,
                            c_values[0],
                            _estimator_name(prototypes[0]),
                            "ineligible",
                            None,
                            None,
                            None,
                            None,
                            ("layer 25 is reserved for comparison baselines, not pre-output claims",),
                        )
                    )
                    continue
                raw_train = _feature_matrix(train, family, anchor, layer)
                raw_validation = _feature_matrix(validation, family, anchor, layer)
                selector = _fit_selector(raw_train, family)
                selected_train = raw_train[:, selector]
                selected_validation = raw_validation[:, selector]
                train_labels = np.asarray([row.label for row in train])
                validation_labels = np.asarray([row.label for row in validation])
                train_groups = tuple(row.entity_id for row in train)
                selected_hyperparams = _select_hyperparams_by_grouped_cv(
                    selected_train,
                    train_labels,
                    train_groups,
                    classes,
                    pca_values,
                    c_values,
                    prototypes,
                    seed,
                    rows=train,
                    task=task,
                )
                for pca_count in pca_values:
                    if pca_count is not None and pca_count > min(selected_train.shape):
                        for c_value in c_values:
                            for prototype in prototypes:
                                scores.append(
                                    CandidateScore(
                                        family,
                                        anchor,
                                        layer,
                                        pca_count,
                                        c_value,
                                        _estimator_name(prototype),
                                        "ineligible",
                                        None,
                                        None,
                                        None,
                                        None,
                                        ("PCA components exceed train-only rank",),
                                    )
                                )
                        continue
                    for c_value in c_values:
                        for prototype in prototypes:
                            estimator_name = _estimator_name(prototype)
                            if (pca_count, float(c_value), estimator_name) != selected_hyperparams:
                                scores.append(
                                    CandidateScore(
                                        family,
                                        anchor,
                                        layer,
                                        pca_count,
                                        c_value,
                                        estimator_name,
                                        "ineligible",
                                        None,
                                        None,
                                        None,
                                        None,
                                        ("not selected by train-only grouped CV",),
                                    )
                                )
                                continue
                            scaler = StandardScaler().fit(selected_train)
                            scaled_train = scaler.transform(selected_train)
                            scaled_validation = scaler.transform(selected_validation)
                            pca: PCA | None = None
                            transformed_train = scaled_train
                            transformed_validation = scaled_validation
                            if pca_count is not None:
                                pca = PCA(n_components=pca_count, svd_solver="full").fit(scaled_train)
                                transformed_train = pca.transform(scaled_train)
                                transformed_validation = pca.transform(scaled_validation)
                            _record_fit_ids(prototype, train_ids)
                            estimator = _fresh_estimator(prototype, c_value, seed, classes)
                            estimator.fit(transformed_train, train_labels)
                            probabilities = _class_probabilities(
                                estimator, transformed_validation, classes
                            )
                            threshold = (
                                _select_threshold(validation_labels, probabilities[:, 1])
                                if classes == (0, 1)
                                else None
                            )
                            metrics = compute_classification_metrics(
                                tuple(row.label for row in validation),
                                probabilities,
                                classes=classes,
                                threshold=threshold,
                            )
                            if metrics.status != "evaluable":
                                score = CandidateScore(
                                    family,
                                    anchor,
                                    layer,
                                    pca_count,
                                    c_value,
                                    _estimator_name(prototype),
                                    "not_evaluable",
                                    None,
                                    None,
                                    None,
                                    threshold,
                                    metrics.reasons,
                                )
                                scores.append(score)
                                continue
                            model = _freeze_model(
                                estimator,
                                family,
                                anchor,
                                layer,
                                pca_count,
                                c_value,
                                threshold,
                                selector,
                                scaler,
                                pca,
                                metrics,
                                classes,
                            )
                            transfer_summary: CrossConditionTransferSummary | None = None
                            transfer_rotations: tuple[FrozenTransferRotation, ...] = ()
                            if task in {"familiarity", "answerability"}:
                                transfer_rotations = _fit_transfer_rotations(
                                    task=task,
                                    train_rows=train,
                                    validation_rows=validation,
                                    train_matrix=selected_train,
                                    validation_matrix=selected_validation,
                                    family=family,
                                    anchor=anchor,
                                    layer=layer,
                                    pca_count=pca_count,
                                    c_value=c_value,
                                    prototype=prototype,
                                    seed=seed,
                                    classes=classes,
                                    selector=selector,
                                )
                                transfer_summary = CrossConditionTransferSummary(
                                    task,
                                    tuple(
                                        rotation.validation_result
                                        for rotation in transfer_rotations
                                    ),
                                )
                                if (
                                    transfer_summary.status != "evaluable"
                                    or transfer_summary.mean_log_loss is None
                                    or transfer_summary.mean_auroc is None
                                    or transfer_summary.mean_balanced_accuracy is None
                                ):
                                    scores.append(
                                        CandidateScore(
                                            family,
                                            anchor,
                                            layer,
                                            pca_count,
                                            c_value,
                                            model.estimator,
                                            "not_evaluable",
                                            None,
                                            None,
                                            None,
                                            threshold,
                                            ("registered transfer rotation is not evaluable",),
                                            transfer_summary,
                                        )
                                    )
                                    continue
                                model = replace(
                                    model,
                                    validation_log_loss=transfer_summary.mean_log_loss,
                                    validation_auroc=transfer_summary.mean_auroc,
                                    validation_balanced_accuracy=(
                                        transfer_summary.mean_balanced_accuracy
                                    ),
                                )
                            score = CandidateScore(
                                family,
                                anchor,
                                layer,
                                pca_count,
                                c_value,
                                model.estimator,
                                "evaluable",
                                model.validation_log_loss,
                                model.validation_auroc,
                                model.validation_balanced_accuracy,
                                threshold,
                                (),
                                transfer_summary,
                            )
                            scores.append(score)
                            rank = _candidate_rank(model)
                            key = (family, model.claim_scope)
                            current = best_by_family.get(key)
                            if current is None or rank < current[0]:
                                best_by_family[key] = (rank, model, transfer_rotations)
    pre_output_families = {
        family for family, claim_scope in best_by_family if claim_scope == "pre_output"
    }
    required_pre_output_families = set(families) - {"output_margin"}
    if pre_output_families != required_pre_output_families:
        missing = sorted(required_pre_output_families - pre_output_families)
        raise ValueError(f"no evaluable candidate for feature families: {missing}")
    models = tuple(
        best_by_family[(family, claim_scope)][1]
        for family in families
        for claim_scope in ("pre_output", "output_proximal_control")
        if (family, claim_scope) in best_by_family
    )
    selected = min(
        (model for model in models if model.claim_scope == "pre_output"),
        key=_candidate_rank,
    ).feature_family
    return SelectionManifest(
        schema_version=2,
        task=task,
        train_ids=tuple(sorted(train_ids)),
        validation_ids=tuple(sorted(validation_ids)),
        train_row_sha256s=tuple(sorted(row.sha256 for row in train)),
        validation_row_sha256s=tuple(sorted(row.sha256 for row in validation)),
        train_entity_ids=tuple(sorted({row.entity_id for row in train})),
        validation_entity_ids=tuple(sorted({row.entity_id for row in validation})),
        train_template_ids=tuple(sorted({row.template_id for row in train})),
        validation_template_ids=tuple(sorted({row.template_id for row in validation})),
        train_relation_ids=tuple(sorted({row.relation_id for row in train})),
        validation_relation_ids=tuple(sorted({row.relation_id for row in validation})),
        train_domain_ids=tuple(sorted({row.domain for row in train})),
        validation_domain_ids=tuple(sorted({row.domain for row in validation})),
        models=models,
        selected_feature_family=selected,
        candidate_scores=tuple(scores),
        registered_layers=REGISTERED_LAYERS,
        registered_anchors=tuple(REGISTERED_ANCHORS),
        pca_options=pca_values,
        c_options=c_values,
        seed=seed,
        sae_gate_sha256=None if sae_gate is None else sae_gate.sha256,
        null_provenance=null_provenance,
        transfer_rotations=(
            best_by_family[(selected, "pre_output")][2]
            if task in {"familiarity", "answerability"}
            else ()
        ),
    )


def _select_hyperparams_by_grouped_cv(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
    classes: tuple[int | str, ...],
    pca_values: Sequence[int | None],
    c_values: Sequence[float],
    prototypes: Sequence[Any],
    seed: int,
    *,
    rows: Sequence[ProbeRow],
    task: str,
) -> tuple[int | None, float, str]:
    if _TRAIN_ONLY_CV_FAST_PATH_FOR_TESTS:
        return (None, float(c_values[0]), _estimator_name(prototypes[0]))
    unique_groups = tuple(sorted(set(groups)))
    fold_count = min(5, max(2, len(unique_groups)))
    fold_by_group = {group: index % fold_count for index, group in enumerate(unique_groups)}
    ranked = []
    for pca_count in pca_values:
        if pca_count is not None and pca_count > min(matrix.shape):
            continue
        for c_value in c_values:
            for prototype in prototypes:
                fold_losses = []
                for fold in range(fold_count):
                    held_out = np.asarray([fold_by_group[group] == fold for group in groups])
                    specs = _registered_rotation_specs(task)
                    fold_specs = specs if specs else (("__pooled__", ("__pooled__",)),)
                    rotation_losses = []
                    for train_condition, test_conditions in fold_specs:
                        if specs:
                            train_mask = np.asarray(
                                [
                                    not held_out[index]
                                    and _transfer_condition(row, task) == train_condition
                                    for index, row in enumerate(rows)
                                ]
                            )
                        else:
                            train_mask = ~held_out
                        if not np.any(train_mask) or set(labels[train_mask].tolist()) != set(classes):
                            break
                        scaler = StandardScaler().fit(matrix[train_mask])
                        fold_train = scaler.transform(matrix[train_mask])
                        pca: PCA | None = None
                        if pca_count is not None:
                            if pca_count > min(fold_train.shape):
                                break
                            pca = PCA(n_components=pca_count, svd_solver="full").fit(fold_train)
                            fold_train = pca.transform(fold_train)
                        estimator = _fresh_estimator(prototype, float(c_value), seed, classes)
                        estimator.fit(fold_train, labels[train_mask])
                        threshold = (
                            _select_threshold(
                                labels[train_mask],
                                _class_probabilities(estimator, fold_train, classes)[:, 1],
                            )
                            if classes == (0, 1)
                            else None
                        )
                        condition_losses = []
                        for test_condition in test_conditions:
                            validation_mask = (
                                held_out
                                if not specs
                                else np.asarray(
                                    [
                                        held_out[index]
                                        and _transfer_condition(row, task) == test_condition
                                        for index, row in enumerate(rows)
                                    ]
                                )
                            )
                            if not np.any(validation_mask):
                                break
                            fold_validation = scaler.transform(matrix[validation_mask])
                            if pca is not None:
                                fold_validation = pca.transform(fold_validation)
                            probabilities = _class_probabilities(
                                estimator, fold_validation, classes
                            )
                            metrics = compute_classification_metrics(
                                tuple(labels[validation_mask].tolist()),
                                probabilities,
                                classes=classes,
                                threshold=threshold,
                            )
                            if metrics.status != "evaluable" or metrics.log_loss is None:
                                break
                            condition_losses.append(metrics.log_loss)
                        if len(condition_losses) != len(test_conditions):
                            break
                        rotation_losses.append(float(np.mean(condition_losses)))
                    if len(rotation_losses) == len(fold_specs):
                        fold_losses.append(float(np.mean(rotation_losses)))
                if fold_losses:
                    ranked.append(
                        (
                            float(np.mean(fold_losses)),
                            -len(fold_losses),
                            -1 if pca_count is None else pca_count,
                            float(c_value),
                            _estimator_name(prototype),
                            pca_count,
                            float(c_value),
                        )
                    )
    if not ranked:
        return (None, float(c_values[0]), _estimator_name(prototypes[0]))
    best = min(ranked)
    return best[5], best[6], best[4]


def _reject_group_overlap(
    train: Sequence[ProbeRow], validation: Sequence[ProbeRow], field_name: str, message: str
) -> None:
    if {getattr(row, field_name) for row in train} & {
        getattr(row, field_name) for row in validation
    }:
        raise ValueError(message)


def _require_registered_domains(rows: Sequence[ProbeRow], field_name: str) -> None:
    observed = {row.domain for row in rows}
    if observed != set(REGISTERED_DOMAINS):
        raise ValueError(f"{field_name} must contain exactly the four registered domains")


def _validate_transfer_factors(
    rows: Sequence[ProbeRow], task: str, field_name: str
) -> None:
    specs = _registered_rotation_specs(task)
    if not specs:
        return
    registered_conditions = tuple(train_condition for train_condition, _ in specs)
    observed_conditions = {_transfer_condition(row, task) for row in rows}
    if observed_conditions != set(registered_conditions):
        raise ValueError(f"{field_name} does not contain every registered transfer condition")
    for condition in registered_conditions:
        for distractor_familiarity in TARGET_FAMILIARITY_CONDITIONS:
            if not any(
                _transfer_condition(row, task) == condition
                and row.distractor_familiarity_condition == distractor_familiarity
                for row in rows
            ):
                raise ValueError(
                    f"{field_name} is missing a registered transfer/distractor-familiarity cell"
                )


def _feature_matrix(
    rows: Sequence[ProbeRow], family: str, anchor: str, layer: int | None
) -> np.ndarray:
    anchor_index = _ANCHOR_INDEX[anchor]
    vectors = []
    for row in rows:
        if family == "surface":
            vector = np.asarray(row.surface_features)
        elif family == "output_margin":
            vector = np.asarray(row.output_margin_features)
        elif family == NESTED_H5_BASELINE:
            vector = np.asarray(row.surface_features)
        elif family == NESTED_H5_CANDIDATE:
            if layer is None:
                raise ValueError("nested static feature family requires a layer")
            vector = np.concatenate(
                [
                    np.asarray(row.surface_features),
                    row.residual_features[anchor_index, layer],
                ]
            )
        elif family == NESTED_H6_CANDIDATE:
            if layer is None:
                raise ValueError("nested dynamics feature family requires a layer")
            static = row.residual_features[anchor_index, layer]
            previous = static if layer == 0 else row.residual_features[anchor_index, layer - 1]
            previous2 = previous if layer < 2 else row.residual_features[anchor_index, layer - 2]
            velocity = static - previous
            acceleration = velocity - (previous - previous2)
            vector = np.concatenate(
                [
                    np.asarray(row.surface_features),
                    *_dynamics_components(row.residual_features[anchor_index], layer),
                ]
            )
        elif family in {"residual_static", "final_layer_excluded"}:
            if layer is None:
                raise ValueError("residual feature family requires a layer")
            vector = row.residual_features[anchor_index, layer]
        elif family == "static_plus_dynamics":
            if layer is None:
                raise ValueError("dynamics feature family requires a layer")
            vector = np.concatenate(_dynamics_components(row.residual_features[anchor_index], layer))
        elif family in SAE_FAMILIES:
            if layer is None or row.sae_features is None:
                raise ValueError("SAE feature family requires audited layer features")
            vector = row.sae_features[anchor_index, layer]
        else:  # pragma: no cover - guarded by option validation
            raise ValueError("feature family is not registered")
        vectors.append(np.asarray(vector, dtype=np.float64))
    widths = {vector.shape for vector in vectors}
    if len(widths) != 1 or any(vector.ndim != 1 for vector in vectors):
        raise ValueError("feature vectors must have one consistent width")
    matrix = np.stack(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix must be finite")
    return matrix


def _dynamics_components(layer_states: np.ndarray, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    static = np.asarray(layer_states[layer], dtype=np.float64)
    previous = np.zeros_like(static) if layer == 0 else np.asarray(layer_states[layer - 1], dtype=np.float64)
    delta = static - previous
    if layer <= 1:
        previous_delta = previous - np.zeros_like(static)
    else:
        previous_delta = previous - np.asarray(layer_states[layer - 2], dtype=np.float64)
    previous_norm = float(np.linalg.norm(previous_delta))
    current_norm = float(np.linalg.norm(delta))
    if not math.isfinite(previous_norm) or not math.isfinite(current_norm):
        raise ValueError("dynamics direction-change norm is non-finite")
    denominator = max(previous_norm * current_norm, 1e-12)
    cosine = float(np.dot(previous_delta, delta) / denominator)
    if not math.isfinite(cosine):
        raise ValueError("dynamics direction-change cosine is non-finite")
    return static, delta, np.asarray([1.0 - max(-1.0, min(1.0, cosine))], dtype=np.float64)


def _fit_selector(validation_matrix: np.ndarray, family: str) -> tuple[int, ...]:
    if family not in SAE_FAMILIES:
        return tuple(range(validation_matrix.shape[1]))
    variances = np.var(validation_matrix, axis=0)
    order = np.lexsort((np.arange(len(variances)), -variances))
    count = 1 if family == "sae_1_sparse" else min(5, validation_matrix.shape[1])
    return tuple(sorted(int(value) for value in order[:count]))


def _record_fit_ids(prototype: Any, train_ids: Collection[str]) -> None:
    if prototype is None:
        return
    current = getattr(prototype, "fit_ids", None)
    if isinstance(current, set):
        current.update(train_ids)


def _fresh_estimator(
    prototype: Any,
    c_value: float,
    seed: int,
    classes: tuple[int | str, ...],
) -> Any:
    if prototype is None:
        return LogisticRegression(
            C=c_value,
            solver="liblinear" if classes == (0, 1) else "lbfgs",
            random_state=seed,
            max_iter=1000,
        )
    estimator = copy.deepcopy(prototype)
    get_params = getattr(estimator, "get_params", None)
    set_params = getattr(estimator, "set_params", None)
    if callable(get_params) and callable(set_params):
        params = get_params(deep=False)
        updates = {}
        if "C" in params:
            updates["C"] = c_value
        if "random_state" in params:
            updates["random_state"] = seed
        if classes == ANSWERABILITY_CLASSES and params.get("solver") == "liblinear":
            updates["solver"] = "lbfgs"
        if updates:
            set_params(**updates)
    if not callable(getattr(estimator, "fit", None)) or not callable(
        getattr(estimator, "predict_proba", None)
    ):
        raise ValueError("estimators must implement fit and predict_proba")
    return estimator


def _estimator_name(prototype: Any) -> str:
    return "LogisticRegression" if prototype is None else type(prototype).__name__


def _class_probabilities(
    estimator: Any,
    matrix: np.ndarray,
    expected_classes: tuple[int | str, ...],
) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
    classes = tuple(np.asarray(estimator.classes_).tolist())
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != len(expected_classes)
        or set(classes) != set(expected_classes)
    ):
        raise ValueError("estimator predict_proba classes do not match the registered target")
    values = probabilities[:, [classes.index(label) for label in expected_classes]]
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("estimator probabilities must be finite and in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("estimator class probabilities must sum to one")
    return values


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = sorted({0.5, *(float(value) for value in probabilities)})
    ranked = []
    for threshold in candidates:
        balanced = float(balanced_accuracy_score(labels, probabilities >= threshold))
        ranked.append((-balanced, abs(threshold - 0.5), threshold))
    return float(min(ranked)[2])


def _freeze_model(
    estimator: Any,
    family: str,
    anchor: str,
    layer: int | None,
    pca_count: int | None,
    c_value: float,
    threshold: float | None,
    selector: tuple[int, ...],
    scaler: StandardScaler,
    pca: PCA | None,
    metrics: "BinaryMetrics",
    classes: tuple[int | str, ...],
) -> FrozenProbeModel:
    coefficients = np.asarray(getattr(estimator, "coef_", None), dtype=np.float64)
    intercept = np.asarray(getattr(estimator, "intercept_", None), dtype=np.float64)
    estimator_classes = tuple(np.asarray(getattr(estimator, "classes_", ())).tolist())
    expected_rows = 1 if classes == (0, 1) else len(classes)
    if coefficients.ndim != 2 or coefficients.shape[0] != expected_rows or intercept.size != expected_rows:
        raise ValueError("fitted linear estimator shape does not match the registered target")
    if set(estimator_classes) != set(classes):
        raise ValueError("fitted linear estimator classes do not match the registered target")
    if classes == (0, 1):
        ordered_coefficients = coefficients
        ordered_intercepts = intercept.ravel()
    else:
        order = [estimator_classes.index(label) for label in classes]
        ordered_coefficients = coefficients[order]
        ordered_intercepts = intercept.ravel()[order]
    claim_scope = (
        "output_proximal_control"
        if anchor == "assistant_prefix_end" or family == "output_margin"
        else "pre_output"
    )
    return FrozenProbeModel(
        feature_family=family,
        anchor=anchor,
        layer=layer,
        pca_components=pca_count,
        c=c_value,
        estimator=type(estimator).__name__,
        threshold=threshold,
        classes=classes,
        selector_indices=selector,
        scaler_mean=tuple(float(value) for value in scaler.mean_),
        scaler_scale=tuple(float(value) for value in scaler.scale_),
        pca_mean=() if pca is None else tuple(float(value) for value in pca.mean_),
        pca_components_matrix=(
            ()
            if pca is None
            else tuple(tuple(float(value) for value in row) for row in pca.components_)
        ),
        coefficients=tuple(
            tuple(float(value) for value in row) for row in ordered_coefficients
        ),
        intercepts=tuple(float(value) for value in ordered_intercepts),
        validation_log_loss=float(metrics.log_loss),
        validation_auroc=float(metrics.auroc),
        validation_balanced_accuracy=float(metrics.balanced_accuracy),
        claim_scope=claim_scope,
    )


def _candidate_rank(model: FrozenProbeModel) -> tuple[Any, ...]:
    return (
        model.validation_log_loss,
        -model.validation_auroc,
        -model.validation_balanced_accuracy,
        model.feature_family,
        model.anchor,
        -1 if model.layer is None else model.layer,
        -1 if model.pca_components is None else model.pca_components,
        model.c,
        model.estimator,
    )


@dataclass(frozen=True)
class BinaryMetrics:
    status: str
    reasons: tuple[str, ...]
    total: int
    denominator: int
    missing: int
    invalid: int
    positives: int
    negatives: int
    auroc: float | None
    balanced_accuracy: float | None
    log_loss: float | None
    calibration_error: float | None
    threshold: float | None
    classes: tuple[int | str, ...] = ()
    class_counts: tuple[tuple[int | str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"evaluable", "not_evaluable"}:
            raise ValueError("metric status is invalid")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "classes", tuple(self.classes))
        object.__setattr__(
            self, "class_counts", tuple((label, int(count)) for label, count in self.class_counts)
        )
        if self.total != self.denominator + self.missing + self.invalid:
            raise ValueError("metric denominators do not sum to total")
        if self.classes and tuple(label for label, _ in self.class_counts) != self.classes:
            raise ValueError("metric class counts must follow registered class order")
        if self.class_counts and sum(count for _, count in self.class_counts) != self.denominator:
            raise ValueError("metric class counts do not sum to denominator")

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "total": self.total,
            "denominator": self.denominator,
            "missing": self.missing,
            "invalid": self.invalid,
            "positives": self.positives,
            "negatives": self.negatives,
            "auroc": self.auroc,
            "balanced_accuracy": self.balanced_accuracy,
            "log_loss": self.log_loss,
            "calibration_error": self.calibration_error,
            "threshold": self.threshold,
            "classes": list(self.classes),
            "class_counts": [
                {"label": label, "count": count} for label, count in self.class_counts
            ],
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())


@dataclass(frozen=True)
class DistractorFamiliarityCellResult:
    distractor_familiarity_condition: str
    metrics: BinaryMetrics

    def __post_init__(self) -> None:
        if self.distractor_familiarity_condition not in TARGET_FAMILIARITY_CONDITIONS:
            raise ValueError("distractor-familiarity cell is not registered")
        if not isinstance(self.metrics, BinaryMetrics):
            raise ValueError("distractor-familiarity cell requires typed metrics")

    def to_record(self) -> dict[str, Any]:
        return {
            "distractor_familiarity_condition": self.distractor_familiarity_condition,
            "metrics": self.metrics.to_record(),
        }


@dataclass(frozen=True)
class TransferConditionResult:
    test_condition: str
    metrics: BinaryMetrics
    distractor_familiarity_cells: tuple[DistractorFamiliarityCellResult, ...]

    def __post_init__(self) -> None:
        _required_text(self.test_condition, "test_condition")
        if not isinstance(self.metrics, BinaryMetrics):
            raise ValueError("transfer condition requires typed metrics")
        object.__setattr__(
            self, "distractor_familiarity_cells", tuple(self.distractor_familiarity_cells)
        )
        observed = tuple(
            cell.distractor_familiarity_condition
            for cell in self.distractor_familiarity_cells
            if isinstance(cell, DistractorFamiliarityCellResult)
        )
        if (
            len(observed) != len(self.distractor_familiarity_cells)
            or observed != TARGET_FAMILIARITY_CONDITIONS
        ):
            raise ValueError("transfer condition requires exact distractor-familiarity cells")

    def to_record(self) -> dict[str, Any]:
        return {
            "test_condition": self.test_condition,
            "metrics": self.metrics.to_record(),
            "distractor_familiarity_cells": [
                cell.to_record() for cell in self.distractor_familiarity_cells
            ],
        }


@dataclass(frozen=True)
class CrossConditionRotationResult:
    task: str
    train_condition: str
    test_conditions: tuple[str, ...]
    metrics: BinaryMetrics
    condition_results: tuple[TransferConditionResult, ...]

    def __post_init__(self) -> None:
        specs = dict(_registered_rotation_specs(self.task))
        if self.train_condition not in specs:
            raise ValueError("rotation train condition is not a registered train condition")
        object.__setattr__(self, "test_conditions", tuple(self.test_conditions))
        object.__setattr__(self, "condition_results", tuple(self.condition_results))
        if self.test_conditions != specs[self.train_condition]:
            raise ValueError("rotation test conditions are not the exact registered reciprocal tests")
        observed = tuple(
            result.test_condition
            for result in self.condition_results
            if isinstance(result, TransferConditionResult)
        )
        if len(observed) != len(self.condition_results) or observed != self.test_conditions:
            raise ValueError("rotation requires each exact reciprocal test condition once")
        if not isinstance(self.metrics, BinaryMetrics):
            raise ValueError("rotation requires typed aggregate metrics")

    def to_record(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "train_condition": self.train_condition,
            "test_conditions": list(self.test_conditions),
            "metrics": self.metrics.to_record(),
            "condition_results": [result.to_record() for result in self.condition_results],
        }


@dataclass(frozen=True)
class CrossConditionTransferSummary:
    task: str
    rotations: tuple[CrossConditionRotationResult, ...]
    aggregation: str = field(
        default="equal_weight_mean_across_registered_rotations", init=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotations", tuple(self.rotations))
        observed = tuple(
            (rotation.train_condition, rotation.test_conditions)
            for rotation in self.rotations
            if isinstance(rotation, CrossConditionRotationResult) and rotation.task == self.task
        )
        if len(observed) != len(self.rotations) or observed != _registered_rotation_specs(self.task):
            raise ValueError("transfer summary requires exact registered transfer rotations")

    @property
    def status(self) -> str:
        return (
            "evaluable"
            if all(rotation.metrics.status == "evaluable" for rotation in self.rotations)
            else "not_evaluable"
        )

    def _mean(self, field_name: str) -> float | None:
        values = tuple(getattr(rotation.metrics, field_name) for rotation in self.rotations)
        if any(value is None or not math.isfinite(float(value)) for value in values):
            return None
        return float(np.mean(values))

    @property
    def mean_auroc(self) -> float | None:
        return self._mean("auroc")

    @property
    def mean_balanced_accuracy(self) -> float | None:
        return self._mean("balanced_accuracy")

    @property
    def mean_log_loss(self) -> float | None:
        return self._mean("log_loss")

    @property
    def mean_calibration_error(self) -> float | None:
        return self._mean("calibration_error")

    @property
    def worst_cell_balanced_accuracy(self) -> float | None:
        values = tuple(
            cell.metrics.balanced_accuracy
            for rotation in self.rotations
            for condition in rotation.condition_results
            for cell in condition.distractor_familiarity_cells
        )
        if any(value is None or not math.isfinite(float(value)) for value in values):
            return None
        return min(float(value) for value in values)

    def to_record(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "aggregation": self.aggregation,
            "status": self.status,
            "mean_auroc": self.mean_auroc,
            "mean_balanced_accuracy": self.mean_balanced_accuracy,
            "mean_log_loss": self.mean_log_loss,
            "mean_calibration_error": self.mean_calibration_error,
            "worst_cell_balanced_accuracy": self.worst_cell_balanced_accuracy,
            "rotations": [rotation.to_record() for rotation in self.rotations],
        }


@dataclass(frozen=True)
class FrozenTransferRotation:
    task: str
    train_condition: str
    test_conditions: tuple[str, ...]
    model: FrozenProbeModel
    validation_result: CrossConditionRotationResult

    def __post_init__(self) -> None:
        specs = dict(_registered_rotation_specs(self.task))
        if self.train_condition not in specs:
            raise ValueError("rotation train condition is not a registered train condition")
        object.__setattr__(self, "test_conditions", tuple(self.test_conditions))
        if self.test_conditions != specs[self.train_condition]:
            raise ValueError("frozen rotation test conditions are not registered")
        if not isinstance(self.model, FrozenProbeModel):
            raise ValueError("frozen transfer rotation requires a typed model")
        if (
            not isinstance(self.validation_result, CrossConditionRotationResult)
            or self.validation_result.task != self.task
            or self.validation_result.train_condition != self.train_condition
            or self.validation_result.test_conditions != self.test_conditions
        ):
            raise ValueError("frozen rotation validation result does not match its rotation")

    def to_record(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "train_condition": self.train_condition,
            "test_conditions": list(self.test_conditions),
            "model": self.model.to_record(),
            "validation_result": self.validation_result.to_record(),
        }


def compute_binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float,
    total: int | None = None,
    missing: int = 0,
    invalid: int = 0,
) -> BinaryMetrics:
    return compute_classification_metrics(
        labels,
        probabilities,
        classes=(0, 1),
        threshold=threshold,
        total=total,
        missing=missing,
        invalid=invalid,
    )


def compute_classification_metrics(
    labels: Sequence[int | str],
    probabilities: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence[int | str],
    threshold: float | None = None,
    total: int | None = None,
    missing: int = 0,
    invalid: int = 0,
) -> BinaryMetrics:
    registered_classes = tuple(classes)
    if registered_classes not in ((0, 1), ANSWERABILITY_CLASSES):
        raise ValueError("metric classes are not a registered target")
    y = np.asarray(labels)
    raw_probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(y) == 0 and raw_probabilities.size == 0:
        p = np.empty((0, len(registered_classes)), dtype=np.float64)
    elif registered_classes == (0, 1) and raw_probabilities.ndim == 1:
        p = np.column_stack((1.0 - raw_probabilities, raw_probabilities))
    else:
        p = raw_probabilities
    if y.ndim != 1 or p.ndim != 2 or p.shape != (len(y), len(registered_classes)):
        raise ValueError("labels and class probabilities must be aligned")
    if any(type(value) is not int or value < 0 for value in (missing, invalid)):
        raise ValueError("missing and invalid counts must be nonnegative integers")
    if registered_classes == (0, 1):
        if type(threshold) not in {int, float} or not math.isfinite(float(threshold)):
            raise ValueError("binary threshold must be finite")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("binary threshold must be in [0, 1]")
        resolved_threshold: float | None = float(threshold)
    else:
        if threshold is not None:
            raise ValueError("multiclass metrics do not use a binary threshold")
        resolved_threshold = None
    if any(label not in registered_classes for label in y.tolist()):
        raise ValueError("labels must belong to the registered target classes")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("class probabilities must sum to one")
    denominator = int(len(y))
    expected_total = denominator + missing + invalid
    resolved_total = expected_total if total is None else total
    if type(resolved_total) is not int or resolved_total != expected_total:
        raise ValueError("total must equal denominator plus missing plus invalid")
    counts = tuple((label, int(np.sum(y == label))) for label in registered_classes)
    positives = counts[1][1] if registered_classes == (0, 1) else 0
    negatives = counts[0][1] if registered_classes == (0, 1) else 0
    if denominator == 0:
        reasons = ("no evaluable rows",)
    elif any(count == 0 for _, count in counts):
        reasons = (
            "single class after missing/invalid accounting"
            if registered_classes == (0, 1)
            else "missing target class after missing/invalid accounting",
        )
    else:
        reasons = ()
    if reasons:
        return BinaryMetrics(
            "not_evaluable",
            reasons,
            resolved_total,
            denominator,
            missing,
            invalid,
            positives,
            negatives,
            None,
            None,
            None,
            None,
            resolved_threshold,
            registered_classes,
            counts,
        )
    clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    clipped /= clipped.sum(axis=1, keepdims=True)
    if registered_classes == (0, 1):
        positive_probabilities = p[:, 1]
        predictions = np.where(positive_probabilities >= resolved_threshold, 1, 0)
        auroc = float(roc_auc_score(y.astype(np.int64), positive_probabilities))
        calibration = _calibration_error(y.astype(np.int64), positive_probabilities)
    else:
        predictions = np.asarray(registered_classes, dtype=object)[np.argmax(p, axis=1)]
        auroc = float(
            np.mean(
                [
                    roc_auc_score((y == label).astype(np.int64), p[:, index])
                    for index, label in enumerate(registered_classes)
                ]
            )
        )
        calibration = float(
            np.mean(
                [
                    _calibration_error((y == label).astype(np.int64), p[:, index])
                    for index, label in enumerate(registered_classes)
                ]
            )
        )
    if registered_classes == (0, 1):
        resolved_log_loss = float(log_loss(y, clipped, labels=[0, 1]))
    else:
        class_index = {label: index for index, label in enumerate(registered_classes)}
        resolved_log_loss = float(
            -np.mean(
                [math.log(clipped[row_index, class_index[label]]) for row_index, label in enumerate(y)]
            )
        )
    return BinaryMetrics(
        "evaluable",
        (),
        resolved_total,
        denominator,
        missing,
        invalid,
        positives,
        negatives,
        auroc,
        float(balanced_accuracy_score(y, predictions)),
        resolved_log_loss,
        calibration,
        resolved_threshold,
        registered_classes,
        counts,
    )


def _calibration_error(labels: np.ndarray, probabilities: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, 9)
    error = 0.0
    for index in range(10):
        selected = bins == index
        if np.any(selected):
            error += float(np.mean(selected)) * abs(
                float(np.mean(labels[selected])) - float(np.mean(probabilities[selected]))
            )
    return error


@dataclass(frozen=True, init=False)
class ProbeTestAuthorization:
    endpoint: str
    lease_id: str
    preregistration_hash: str
    selection_hash: str
    capability_id: str

    @classmethod
    def from_unlock_receipt(cls, receipt: UnlockReceipt) -> "ProbeTestAuthorization":
        if not isinstance(receipt, UnlockReceipt):
            raise ValueError("probe authorization requires an artifact-store UnlockReceipt")
        if receipt.endpoint != "probe_test" or receipt.state != "unlocked_once":
            raise ValueError("probe authorization requires an unlocked_once probe_test receipt")
        _required_sha256(receipt.preregistration_hash, "receipt preregistration_hash")
        _required_sha256(receipt.selection_manifest_hash, "receipt selection_manifest_hash")
        if not isinstance(receipt.lease_id, str) or not re.fullmatch(r"[0-9a-f]{32}", receipt.lease_id):
            raise ValueError("receipt lease_id is invalid")
        capability_id = _digest(
            {
                "endpoint": receipt.endpoint,
                "lease_id": receipt.lease_id,
                "preregistration_hash": receipt.preregistration_hash,
                "selection_hash": receipt.selection_manifest_hash,
            }
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "endpoint", receipt.endpoint)
        object.__setattr__(instance, "lease_id", receipt.lease_id)
        object.__setattr__(instance, "preregistration_hash", receipt.preregistration_hash)
        object.__setattr__(instance, "selection_hash", receipt.selection_manifest_hash)
        object.__setattr__(instance, "capability_id", capability_id)
        return instance

    def to_record(self) -> dict[str, str]:
        return {
            "endpoint": self.endpoint,
            "lease_id": self.lease_id,
            "preregistration_hash": self.preregistration_hash,
            "selection_hash": self.selection_hash,
            "capability_id": self.capability_id,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())


@dataclass(frozen=True)
class GateCriterion:
    name: str
    observed: float | None
    threshold: float
    comparison: str

    @property
    def satisfied(self) -> bool | None:
        if self.observed is None or not math.isfinite(self.observed):
            return None
        if self.comparison == ">=":
            return self.observed >= self.threshold
        if self.comparison == ">":
            return self.observed > self.threshold
        if self.comparison == "<=":
            return self.observed <= self.threshold
        raise ValueError("gate comparison must be >=, >, or <=")

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "observed": self.observed,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "satisfied": self.satisfied,
        }


@dataclass(frozen=True)
class HypothesisGate:
    hypothesis: str
    criteria: tuple[GateCriterion, ...]
    context_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.hypothesis not in {"H3", "H4", "H5", "H6"}:
            raise ValueError("hypothesis is not registered")
        object.__setattr__(self, "criteria", tuple(self.criteria))
        object.__setattr__(self, "context_reasons", tuple(self.context_reasons))

    @property
    def status(self) -> str:
        values = tuple(criterion.satisfied for criterion in self.criteria)
        if not values or any(value is None for value in values):
            return "not_evaluable"
        return "supported" if all(values) else "not_supported"

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons = list(self.context_reasons)
        for criterion in self.criteria:
            if criterion.satisfied is None:
                reasons.append(f"{criterion.name} is not evaluable")
            elif not criterion.satisfied:
                reasons.append(
                    f"{criterion.name}={criterion.observed:.6g} fails "
                    f"{criterion.comparison}{criterion.threshold:.6g}"
                )
        if not reasons and self.status == "supported":
            reasons.append("all registered criteria are satisfied")
        return tuple(reasons)

    def to_record(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "criteria": [criterion.to_record() for criterion in self.criteria],
            "status": self.status,
            "reasons": list(self.reasons),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())


@dataclass(frozen=True)
class CrossedBootstrapInterval:
    lower: float
    upper: float
    confidence: float = 0.95
    draws: int = DEFAULT_CONFIRMATORY_BOOTSTRAP_DRAWS
    requested_draws: int = DEFAULT_CONFIRMATORY_BOOTSTRAP_DRAWS
    valid_draws: int | None = None
    discarded_draws: int | None = None
    seed: int = DEFAULT_BOOTSTRAP_SEED
    resampling_unit: str = "crossed_entity_template"

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(value)) for value in (self.lower, self.upper)) or self.lower > self.upper:
            raise ValueError("bootstrap interval bounds are invalid")
        valid = self.draws if self.valid_draws is None else self.valid_draws
        discarded = 0 if self.discarded_draws is None else self.discarded_draws
        if (
            self.confidence != 0.95
            or type(self.draws) is not int
            or type(self.requested_draws) is not int
            or type(valid) is not int
            or type(discarded) is not int
            or self.draws < 50
            or valid != self.draws
            or discarded < 0
            or self.requested_draws < valid
        ):
            raise ValueError("bootstrap interval must be a 95% interval with at least 50 draws")
        if type(self.seed) is not int or self.resampling_unit != "crossed_entity_template":
            raise ValueError("bootstrap provenance is not registered")
        object.__setattr__(self, "valid_draws", valid)
        object.__setattr__(self, "discarded_draws", discarded)

    def to_record(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "draws": self.draws,
            "requested_draws": self.requested_draws,
            "valid_draws": self.valid_draws,
            "discarded_draws": self.discarded_draws,
            "seed": self.seed,
            "resampling_unit": self.resampling_unit,
        }


@dataclass(frozen=True)
class ProbeResult:
    schema_version: int
    task: str
    selection_hash: str
    authorization_sha256: str
    endpoint_input_sha256: str
    endpoint_input_identities_sha256: str
    test_ids: tuple[str, ...]
    test_row_sha256s: tuple[str, ...]
    selected_feature_family: str
    selected_anchor: str
    selected_layer: int | None
    claim_scope: str
    selected_model_sha256: str
    metrics: BinaryMetrics
    model_metrics: Mapping[str, BinaryMetrics]
    per_condition: Mapping[str, BinaryMetrics]
    worst_condition: BinaryMetrics | None
    ood_transfer: Mapping[str, Mapping[str, BinaryMetrics]]
    worst_ood_transfer: Mapping[str, BinaryMetrics | None]
    cross_condition_transfer: CrossConditionTransferSummary | None
    relative_h5_log_loss_improvement: float | None
    relative_h6_log_loss_improvement: float | None
    crossed_auroc_95: CrossedBootstrapInterval | None
    h5_absolute_log_loss_difference_95: CrossedBootstrapInterval | None
    h6_absolute_log_loss_difference_95: CrossedBootstrapInterval | None
    primary_gate: HypothesisGate
    null_results: tuple["NullSelectionResult", ...] = ()
    refit_performed: bool = field(default=False, init=False)

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        selection: SelectionManifest,
    ) -> "ProbeResult":
        return _probe_result_from_record(record, selection=selection)

    def __post_init__(self) -> None:
        if self.schema_version != 3 or self.task not in TASKS:
            raise ValueError("probe result schema or task is invalid")
        for field_name in (
            "selection_hash",
            "authorization_sha256",
            "endpoint_input_sha256",
            "endpoint_input_identities_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        _required_text(self.selected_feature_family, "selected_feature_family")
        if self.selected_anchor not in REGISTERED_ANCHORS:
            raise ValueError("selected anchor is not registered")
        if self.selected_layer is not None and (
            type(self.selected_layer) is not int
            or self.selected_layer not in REGISTERED_LAYERS
        ):
            raise ValueError("selected layer is not registered")
        if self.claim_scope not in {"pre_output", "output_proximal_control"}:
            raise ValueError("selected model claim scope is invalid")
        if self.claim_scope == "pre_output" and (
            self.selected_anchor == "assistant_prefix_end" or self.selected_layer == 25
        ):
            raise ValueError("pre-output scope cannot use assistant_prefix_end or layer 25")
        if self.selected_anchor == "assistant_prefix_end" and self.claim_scope != "output_proximal_control":
            raise ValueError("assistant_prefix_end is output-proximal only")
        _required_sha256(self.selected_model_sha256, "selected_model_sha256")
        object.__setattr__(self, "test_ids", tuple(self.test_ids))
        object.__setattr__(self, "test_row_sha256s", tuple(self.test_row_sha256s))
        object.__setattr__(self, "null_results", tuple(self.null_results))
        if any(
            not isinstance(result, NullSelectionResult) or result.test_metrics is None
            for result in self.null_results
        ):
            raise ValueError("probe result nulls must be typed test-scored null selections")
        if self.task in {"familiarity", "answerability"}:
            if (
                not isinstance(self.cross_condition_transfer, CrossConditionTransferSummary)
                or self.cross_condition_transfer.task != self.task
            ):
                raise ValueError(
                    "decoding result requires reciprocal cross-condition transfer"
                )
        elif self.cross_condition_transfer is not None:
            raise ValueError("unsupported-answer result cannot contain cross-condition transfer")
        object.__setattr__(self, "model_metrics", MappingProxyType(dict(self.model_metrics)))
        object.__setattr__(self, "per_condition", MappingProxyType(dict(self.per_condition)))
        object.__setattr__(
            self,
            "ood_transfer",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(group_metrics))
                    for name, group_metrics in self.ood_transfer.items()
                }
            ),
        )
        object.__setattr__(
            self, "worst_ood_transfer", MappingProxyType(dict(self.worst_ood_transfer))
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "selection_hash": self.selection_hash,
            "authorization_sha256": self.authorization_sha256,
            "endpoint_input_sha256": self.endpoint_input_sha256,
            "endpoint_input_identities_sha256": self.endpoint_input_identities_sha256,
            "endpoint_source_identities_sha256": self.endpoint_input_identities_sha256,
            "test_ids": list(self.test_ids),
            "test_row_sha256s": list(self.test_row_sha256s),
            "selected_feature_family": self.selected_feature_family,
            "selected_model_scope": {
                "feature_family": self.selected_feature_family,
                "anchor": self.selected_anchor,
                "layer": self.selected_layer,
                "claim_scope": self.claim_scope,
                "selected_model_sha256": self.selected_model_sha256,
            },
            "metrics": self.metrics.to_record(),
            "model_metrics": {
                name: metrics.to_record() for name, metrics in self.model_metrics.items()
            },
            "per_condition": {
                name: metrics.to_record() for name, metrics in self.per_condition.items()
            },
            "worst_condition": (
                None if self.worst_condition is None else self.worst_condition.to_record()
            ),
            "ood_transfer": {
                dimension: {
                    name: metrics.to_record() for name, metrics in groups.items()
                }
                for dimension, groups in self.ood_transfer.items()
            },
            "worst_ood_transfer": {
                dimension: None if metrics is None else metrics.to_record()
                for dimension, metrics in self.worst_ood_transfer.items()
            },
            "cross_condition_transfer": (
                None
                if self.cross_condition_transfer is None
                else self.cross_condition_transfer.to_record()
            ),
            "relative_h5_log_loss_improvement": self.relative_h5_log_loss_improvement,
            "relative_h6_log_loss_improvement": self.relative_h6_log_loss_improvement,
            "crossed_auroc_95": None if self.crossed_auroc_95 is None else self.crossed_auroc_95.to_record(),
            "h5_absolute_log_loss_difference_95": None if self.h5_absolute_log_loss_difference_95 is None else self.h5_absolute_log_loss_difference_95.to_record(),
            "h6_absolute_log_loss_difference_95": None if self.h6_absolute_log_loss_difference_95 is None else self.h6_absolute_log_loss_difference_95.to_record(),
            "primary_gate": self.primary_gate.to_record(),
            "null_results": [result.to_record() for result in self.null_results],
            "refit_performed": False,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    @property
    def endpoint_source_identities_sha256(self) -> str:
        return self.endpoint_input_identities_sha256

    @property
    def selected_claim_scope(self) -> str:
        return self.claim_scope


def _sealed_probe_test_identities(
    endpoint_artifact: Any,
) -> Mapping[str, tuple[ProbeSourceIdentity, ...]]:
    """Read task-specific source identities from the sealed prompt manifest."""

    if endpoint_artifact.record_kind != "prompt_manifest":
        raise ValueError("probe_test endpoint requires a prompt_manifest artifact")

    try:
        payload = endpoint_artifact.data_path.read_bytes()
    except OSError as error:
        raise ValueError("sealed probe-test input is unreadable") from error
    if hashlib.sha256(payload).hexdigest() != endpoint_artifact.sha256:
        raise ValueError("sealed probe-test input no longer matches its verified artifact")
    records = []
    for line in payload.splitlines():
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("sealed probe-test identity records must be JSON") from error
        if not isinstance(record, dict) or _canonical_json(record) != line:
            raise ValueError("sealed probe-test identity records must be canonical")
        records.append(record)
    if len(records) != endpoint_artifact.row_count:
        raise ValueError("sealed probe-test identity count does not match its artifact")
    if len(records) != 1:
        raise ValueError("probe_test prompt manifest must contain exactly one record")
    prompt_manifest = records[0]
    if (
        prompt_manifest.get("kind") != "prompt_manifest"
        or prompt_manifest.get("namespace") != "probe_test"
    ):
        raise ValueError("sealed probe-test prompt_manifest has an invalid schema")
    examples = prompt_manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("probe_test prompt manifest requires a nonempty examples list")
    identities = tuple(
        ProbeSourceIdentity.from_record(
            {
                "example_id": record.get("example_id"),
                "canonical_payload_sha256": record.get("canonical_payload_sha256"),
            }
        )
        for record in examples
    )
    all_identities = _canonical_source_identities(
        identities, field_name="sealed probe-test source identities"
    )
    unsupported = _canonical_source_identities(
        tuple(
            identity
            for identity, record in zip(identities, examples, strict=True)
            if record.get("answerability") in {"distractor_bound", "code_absent"}
        ),
        field_name="sealed unsupported-answer source identities",
    )
    return MappingProxyType(
        {
            "familiarity": all_identities,
            "answerability": all_identities,
            "unsupported_answer": unsupported,
        }
    )


def _task_identity_bundle_digest(
    identities_by_task: Mapping[str, Sequence[ProbeSourceIdentity]],
) -> str:
    if set(identities_by_task) != set(TASKS):
        raise ValueError("probe source identity bundle requires every registered task")
    return _digest(
        {
            task: _source_identity_digest(
                _canonical_source_identities(
                    identities_by_task[task],
                    field_name=f"{task} source identities",
                )
            )
            for task in TASKS
        }
    )


def _crossed_bootstrap_interval(
    rows: Sequence[ProbeRow], values: np.ndarray, statistic: Any
) -> CrossedBootstrapInterval | None:
    if not rows:
        return None
    entities = sorted({row.entity_id for row in rows})
    templates = sorted({row.template_id for row in rows})
    if not entities or not templates:
        return None
    entity_index = {value: index for index, value in enumerate(entities)}
    template_index = {value: index for index, value in enumerate(templates)}
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    requested = DEFAULT_CONFIRMATORY_BOOTSTRAP_DRAWS
    attempts = requested if _BOOTSTRAP_DRAW_OVERRIDE_FOR_TESTS is None else _BOOTSTRAP_DRAW_OVERRIDE_FOR_TESTS
    draws = []
    discarded = 0
    for _ in range(attempts):
        entity_counts = np.bincount(rng.integers(len(entities), size=len(entities)), minlength=len(entities))
        template_counts = np.bincount(rng.integers(len(templates), size=len(templates)), minlength=len(templates))
        weights = np.asarray([entity_counts[entity_index[row.entity_id]] * template_counts[template_index[row.template_id]] for row in rows])
        indices = np.repeat(np.arange(len(rows)), weights)
        if not len(indices):
            discarded += 1
            continue
        value = statistic(indices, values)
        if value is not None and math.isfinite(float(value)):
            draws.append(float(value))
        else:
            discarded += 1
    if len(draws) < 50:
        return None
    return CrossedBootstrapInterval(
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        draws=len(draws),
        requested_draws=requested,
        valid_draws=len(draws),
        discarded_draws=discarded,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )


def _auroc_interval(rows: Sequence[ProbeRow], probabilities: np.ndarray, classes: tuple[int | str, ...], threshold: float | None) -> CrossedBootstrapInterval | None:
    labels = tuple(row.label for row in rows)
    return _crossed_bootstrap_interval(rows, probabilities, lambda indices, values: compute_classification_metrics(tuple(labels[index] for index in indices), values[indices], classes=classes, threshold=threshold).auroc)


def _transfer_auroc_interval(
    selection: SelectionManifest, rows: Sequence[ProbeRow]
) -> CrossedBootstrapInterval | None:
    if selection.task not in {"familiarity", "answerability"} or not rows:
        return None
    row_tuple = tuple(rows)
    probabilities = np.stack(
        [rotation.model.predict_proba(row_tuple) for rotation in selection.transfer_rotations],
        axis=1,
    )

    def statistic(indices: np.ndarray, values: np.ndarray) -> float | None:
        rotation_aurocs = []
        for rotation_index, rotation in enumerate(selection.transfer_rotations):
            selected = np.asarray(
                [
                    index
                    for index in indices
                    if row_tuple[index].outcome_status == "valid"
                    and _transfer_condition(row_tuple[index], selection.task)
                    in rotation.test_conditions
                ],
                dtype=np.int64,
            )
            if not len(selected):
                return None
            metrics = compute_classification_metrics(
                tuple(row_tuple[index].label for index in selected),
                values[selected, rotation_index],
                classes=rotation.model.classes,
                threshold=rotation.model.threshold,
            )
            if metrics.auroc is None:
                return None
            rotation_aurocs.append(metrics.auroc)
        return float(np.mean(rotation_aurocs))

    return _crossed_bootstrap_interval(row_tuple, probabilities, statistic)


def _log_loss_difference_interval(rows: Sequence[ProbeRow], baseline: np.ndarray | None, candidate: np.ndarray | None, classes: tuple[int | str, ...]) -> CrossedBootstrapInterval | None:
    if baseline is None or candidate is None:
        return None
    labels = tuple(row.label for row in rows)
    class_index = {label: index for index, label in enumerate(classes)}
    differences = np.asarray([math.log(max(float(baseline[i, class_index[label]]), 1e-15)) - math.log(max(float(candidate[i, class_index[label]]), 1e-15)) for i, label in enumerate(labels)])
    return _crossed_bootstrap_interval(rows, differences, lambda indices, values: float(np.mean(values[indices])))


def _record_with_schema(
    value: Any, expected_keys: Collection[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        raise ValueError(f"{label} has an invalid schema")
    _require_finite_record(value, label)
    return value


def _require_finite_record(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must contain only finite numbers")
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite_record(item, label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_record(item, label)


def _require_canonical_record(
    loaded: Any, record: Mapping[str, Any], label: str
) -> Any:
    if loaded.to_record() != dict(record):
        raise ValueError(f"{label} is not canonical")
    return loaded


def _binary_metrics_from_record(value: Any) -> BinaryMetrics:
    record = _record_with_schema(
        value,
        {
            "status", "reasons", "total", "denominator", "missing", "invalid",
            "positives", "negatives", "auroc", "balanced_accuracy", "log_loss",
            "calibration_error", "threshold", "classes", "class_counts",
        },
        "binary metrics record",
    )
    raw_counts = record["class_counts"]
    if not isinstance(raw_counts, list) or any(
        not isinstance(item, Mapping) or set(item) != {"label", "count"}
        for item in raw_counts
    ):
        raise ValueError("binary metrics class counts have an invalid schema")
    loaded = BinaryMetrics(
        status=record["status"], reasons=tuple(record["reasons"]),
        total=record["total"], denominator=record["denominator"],
        missing=record["missing"], invalid=record["invalid"],
        positives=record["positives"], negatives=record["negatives"],
        auroc=record["auroc"], balanced_accuracy=record["balanced_accuracy"],
        log_loss=record["log_loss"], calibration_error=record["calibration_error"],
        threshold=record["threshold"], classes=tuple(record["classes"]),
        class_counts=tuple((item["label"], item["count"]) for item in raw_counts),
    )
    return _require_canonical_record(loaded, record, "binary metrics record")


def _distractor_cell_from_record(value: Any) -> DistractorFamiliarityCellResult:
    record = _record_with_schema(
        value, {"distractor_familiarity_condition", "metrics"},
        "distractor-familiarity cell",
    )
    loaded = DistractorFamiliarityCellResult(
        record["distractor_familiarity_condition"],
        _binary_metrics_from_record(record["metrics"]),
    )
    return _require_canonical_record(loaded, record, "distractor-familiarity cell")


def _transfer_condition_from_record(value: Any) -> TransferConditionResult:
    record = _record_with_schema(
        value, {"test_condition", "metrics", "distractor_familiarity_cells"},
        "transfer condition",
    )
    loaded = TransferConditionResult(
        record["test_condition"],
        _binary_metrics_from_record(record["metrics"]),
        tuple(_distractor_cell_from_record(item) for item in record["distractor_familiarity_cells"]),
    )
    return _require_canonical_record(loaded, record, "transfer condition")


def _rotation_result_from_record(value: Any) -> CrossConditionRotationResult:
    record = _record_with_schema(
        value,
        {"task", "train_condition", "test_conditions", "metrics", "condition_results"},
        "cross-condition rotation",
    )
    loaded = CrossConditionRotationResult(
        task=record["task"], train_condition=record["train_condition"],
        test_conditions=tuple(record["test_conditions"]),
        metrics=_binary_metrics_from_record(record["metrics"]),
        condition_results=tuple(_transfer_condition_from_record(item) for item in record["condition_results"]),
    )
    return _require_canonical_record(loaded, record, "cross-condition rotation")


def _transfer_summary_from_record(value: Any) -> CrossConditionTransferSummary:
    record = _record_with_schema(
        value,
        {
            "task", "aggregation", "status", "mean_auroc", "mean_balanced_accuracy",
            "mean_log_loss", "mean_calibration_error", "worst_cell_balanced_accuracy",
            "rotations",
        },
        "cross-condition transfer summary",
    )
    loaded = CrossConditionTransferSummary(
        record["task"],
        tuple(_rotation_result_from_record(item) for item in record["rotations"]),
    )
    return _require_canonical_record(loaded, record, "cross-condition transfer summary")


def _frozen_model_from_record(value: Any) -> FrozenProbeModel:
    record = _record_with_schema(
        value,
        {
            "feature_family", "anchor", "layer", "pca_components", "c", "estimator",
            "threshold", "classes", "selector_indices", "scaler_mean", "scaler_scale",
            "pca_mean", "pca_components_matrix", "coefficients", "intercepts",
            "validation_log_loss", "validation_auroc", "validation_balanced_accuracy",
            "claim_scope",
        },
        "frozen probe model",
    )
    loaded = FrozenProbeModel(
        feature_family=record["feature_family"], anchor=record["anchor"],
        layer=record["layer"], pca_components=record["pca_components"], c=record["c"],
        estimator=record["estimator"], threshold=record["threshold"],
        classes=tuple(record["classes"]), selector_indices=tuple(record["selector_indices"]),
        scaler_mean=tuple(record["scaler_mean"]), scaler_scale=tuple(record["scaler_scale"]),
        pca_mean=tuple(record["pca_mean"]),
        pca_components_matrix=tuple(tuple(row) for row in record["pca_components_matrix"]),
        coefficients=tuple(tuple(row) for row in record["coefficients"]),
        intercepts=tuple(record["intercepts"]),
        validation_log_loss=record["validation_log_loss"],
        validation_auroc=record["validation_auroc"],
        validation_balanced_accuracy=record["validation_balanced_accuracy"],
        claim_scope=record["claim_scope"],
    )
    return _require_canonical_record(loaded, record, "frozen probe model")


def _candidate_score_from_record(value: Any) -> CandidateScore:
    record = _record_with_schema(
        value,
        {
            "feature_family", "anchor", "layer", "pca_components", "c", "estimator",
            "status", "validation_log_loss", "validation_auroc",
            "validation_balanced_accuracy", "threshold", "reasons",
            "cross_condition_transfer",
        },
        "candidate score",
    )
    transfer = record["cross_condition_transfer"]
    loaded = CandidateScore(
        feature_family=record["feature_family"], anchor=record["anchor"],
        layer=record["layer"], pca_components=record["pca_components"], c=record["c"],
        estimator=record["estimator"], status=record["status"],
        validation_log_loss=record["validation_log_loss"],
        validation_auroc=record["validation_auroc"],
        validation_balanced_accuracy=record["validation_balanced_accuracy"],
        threshold=record["threshold"], reasons=tuple(record["reasons"]),
        cross_condition_transfer=None if transfer is None else _transfer_summary_from_record(transfer),
    )
    return _require_canonical_record(loaded, record, "candidate score")


def _frozen_rotation_from_record(value: Any) -> FrozenTransferRotation:
    record = _record_with_schema(
        value,
        {"task", "train_condition", "test_conditions", "model", "validation_result"},
        "frozen transfer rotation",
    )
    loaded = FrozenTransferRotation(
        task=record["task"], train_condition=record["train_condition"],
        test_conditions=tuple(record["test_conditions"]),
        model=_frozen_model_from_record(record["model"]),
        validation_result=_rotation_result_from_record(record["validation_result"]),
    )
    return _require_canonical_record(loaded, record, "frozen transfer rotation")


def _selection_manifest_from_record(value: Any) -> SelectionManifest:
    record = _record_with_schema(
        value,
        {
            "schema_version", "task", "train_ids", "validation_ids",
            "train_row_sha256s", "validation_row_sha256s", "train_entity_ids",
            "validation_entity_ids", "train_template_ids", "validation_template_ids",
            "train_relation_ids", "validation_relation_ids", "train_domain_ids",
            "validation_domain_ids", "models", "selected_feature_family",
            "candidate_scores", "registered_layers", "registered_anchors", "pca_options",
            "c_options", "seed", "sae_gate_sha256", "null_provenance",
            "cross_condition_transfer_rotations",
        },
        "selection manifest",
    )
    loaded = SelectionManifest(
        schema_version=record["schema_version"], task=record["task"],
        train_ids=tuple(record["train_ids"]), validation_ids=tuple(record["validation_ids"]),
        train_row_sha256s=tuple(record["train_row_sha256s"]),
        validation_row_sha256s=tuple(record["validation_row_sha256s"]),
        train_entity_ids=tuple(record["train_entity_ids"]),
        validation_entity_ids=tuple(record["validation_entity_ids"]),
        train_template_ids=tuple(record["train_template_ids"]),
        validation_template_ids=tuple(record["validation_template_ids"]),
        train_relation_ids=tuple(record["train_relation_ids"]),
        validation_relation_ids=tuple(record["validation_relation_ids"]),
        train_domain_ids=tuple(record["train_domain_ids"]),
        validation_domain_ids=tuple(record["validation_domain_ids"]),
        models=tuple(_frozen_model_from_record(item) for item in record["models"]),
        selected_feature_family=record["selected_feature_family"],
        candidate_scores=tuple(_candidate_score_from_record(item) for item in record["candidate_scores"]),
        registered_layers=tuple(record["registered_layers"]),
        registered_anchors=tuple(record["registered_anchors"]),
        pca_options=tuple(record["pca_options"]), c_options=tuple(record["c_options"]),
        seed=record["seed"], sae_gate_sha256=record["sae_gate_sha256"],
        null_provenance=record["null_provenance"],
        transfer_rotations=tuple(
            _frozen_rotation_from_record(item)
            for item in record["cross_condition_transfer_rotations"]
        ),
    )
    return _require_canonical_record(loaded, record, "selection manifest")


def _criterion_from_record(value: Any) -> GateCriterion:
    record = _record_with_schema(
        value, {"name", "observed", "threshold", "comparison", "satisfied"},
        "gate criterion",
    )
    loaded = GateCriterion(record["name"], record["observed"], record["threshold"], record["comparison"])
    return _require_canonical_record(loaded, record, "gate criterion")


def _hypothesis_gate_from_record(value: Any) -> HypothesisGate:
    record = _record_with_schema(
        value, {"hypothesis", "criteria", "status", "reasons"}, "hypothesis gate"
    )
    criteria = tuple(_criterion_from_record(item) for item in record["criteria"])
    base = HypothesisGate(record["hypothesis"], criteria)
    derived = base.reasons
    reasons = tuple(record["reasons"])
    if base.status == "supported":
        context = () if reasons == derived else reasons
    else:
        if len(reasons) < len(derived) or reasons[-len(derived):] != derived:
            raise ValueError("hypothesis gate reasons are not canonical")
        context = reasons[:len(reasons) - len(derived)]
    loaded = HypothesisGate(record["hypothesis"], criteria, context)
    return _require_canonical_record(loaded, record, "hypothesis gate")


def _bootstrap_interval_from_record(value: Any) -> CrossedBootstrapInterval:
    record = _record_with_schema(
        value,
        {"lower", "upper", "confidence", "draws", "requested_draws", "valid_draws",
         "discarded_draws", "seed", "resampling_unit"},
        "bootstrap interval",
    )
    return _require_canonical_record(
        CrossedBootstrapInterval(**record), record, "bootstrap interval"
    )


def _null_result_from_record(value: Any) -> "NullSelectionResult":
    record = _record_with_schema(
        value,
        {
            "kind", "seed", "config", "config_sha256", "selection_sha256",
            "selection", "max_norm_error", "test_source_identities", "test_transform",
            "test_ids", "test_row_sha256s", "test_metrics", "test_model_metrics",
            "test_cross_condition_transfer", "test_relative_h5_log_loss_improvement",
            "test_relative_h6_log_loss_improvement",
        },
        "null selection result",
    )
    selection = SelectionManifest.from_record(record["selection"])
    if selection.sha256 != record["selection_sha256"]:
        raise ValueError("null selected model manifest hash is invalid")
    metrics = record["test_metrics"]
    model_metrics = record["test_model_metrics"]
    transfer = record["test_cross_condition_transfer"]
    loaded = NullSelectionResult(
        kind=record["kind"], seed=record["seed"], config=record["config"],
        config_sha256=record["config_sha256"], selection=selection,
        max_norm_error=record["max_norm_error"],
        test_source_identities=tuple(
            ProbeSourceIdentity.from_record(item) for item in record["test_source_identities"]
        ),
        test_transform=record["test_transform"], test_ids=tuple(record["test_ids"]),
        test_row_sha256s=tuple(record["test_row_sha256s"]),
        test_metrics=None if metrics is None else _binary_metrics_from_record(metrics),
        test_model_metrics=None if model_metrics is None else {
            name: _binary_metrics_from_record(item) for name, item in model_metrics.items()
        },
        test_cross_condition_transfer=None if transfer is None else _transfer_summary_from_record(transfer),
        test_relative_h5_log_loss_improvement=record["test_relative_h5_log_loss_improvement"],
        test_relative_h6_log_loss_improvement=record["test_relative_h6_log_loss_improvement"],
    )
    return _require_canonical_record(loaded, record, "null selection result")


def _probe_result_from_record(value: Any, *, selection: SelectionManifest) -> ProbeResult:
    record = _record_with_schema(
        value,
        {
            "schema_version", "task", "selection_hash", "authorization_sha256",
            "endpoint_input_sha256", "endpoint_input_identities_sha256",
            "endpoint_source_identities_sha256", "test_ids", "test_row_sha256s",
            "selected_feature_family", "selected_model_scope", "metrics", "model_metrics",
            "per_condition", "worst_condition", "ood_transfer", "worst_ood_transfer",
            "cross_condition_transfer", "relative_h5_log_loss_improvement",
            "relative_h6_log_loss_improvement", "crossed_auroc_95",
            "h5_absolute_log_loss_difference_95", "h6_absolute_log_loss_difference_95",
            "primary_gate", "null_results", "refit_performed",
        },
        "probe result",
    )
    if not isinstance(selection, SelectionManifest):
        raise ValueError("probe result requires its typed selection manifest")
    scope = _record_with_schema(
        record["selected_model_scope"],
        {"feature_family", "anchor", "layer", "claim_scope", "selected_model_sha256"},
        "selected model scope",
    )
    if (
        record["selection_hash"] != selection.sha256
        or record["task"] != selection.task
        or record["selected_feature_family"] != selection.selected_feature_family
        or scope["feature_family"] != record["selected_feature_family"]
    ):
        raise ValueError("probe result selected model does not match its selection")
    selected_model = selection.model_for(record["selected_feature_family"])
    if (
        scope["anchor"] != selected_model.anchor
        or scope["layer"] != selected_model.layer
        or scope["claim_scope"] != selected_model.claim_scope
        or scope["selected_model_sha256"] != selected_model.sha256
    ):
        raise ValueError("probe result selected model hash or scope is invalid")
    if (
        record["endpoint_source_identities_sha256"] != record["endpoint_input_identities_sha256"]
        or record["refit_performed"] is not False
    ):
        raise ValueError("probe result provenance is not canonical")
    cross = record["cross_condition_transfer"]
    loaded = ProbeResult(
        schema_version=record["schema_version"], task=record["task"],
        selection_hash=record["selection_hash"], authorization_sha256=record["authorization_sha256"],
        endpoint_input_sha256=record["endpoint_input_sha256"],
        endpoint_input_identities_sha256=record["endpoint_input_identities_sha256"],
        test_ids=tuple(record["test_ids"]), test_row_sha256s=tuple(record["test_row_sha256s"]),
        selected_feature_family=record["selected_feature_family"],
        selected_anchor=scope["anchor"], selected_layer=scope["layer"],
        claim_scope=scope["claim_scope"], selected_model_sha256=scope["selected_model_sha256"],
        metrics=_binary_metrics_from_record(record["metrics"]),
        model_metrics={name: _binary_metrics_from_record(item) for name, item in record["model_metrics"].items()},
        per_condition={name: _binary_metrics_from_record(item) for name, item in record["per_condition"].items()},
        worst_condition=None if record["worst_condition"] is None else _binary_metrics_from_record(record["worst_condition"]),
        ood_transfer={
            dimension: {name: _binary_metrics_from_record(item) for name, item in groups.items()}
            for dimension, groups in record["ood_transfer"].items()
        },
        worst_ood_transfer={
            dimension: None if item is None else _binary_metrics_from_record(item)
            for dimension, item in record["worst_ood_transfer"].items()
        },
        cross_condition_transfer=None if cross is None else _transfer_summary_from_record(cross),
        relative_h5_log_loss_improvement=record["relative_h5_log_loss_improvement"],
        relative_h6_log_loss_improvement=record["relative_h6_log_loss_improvement"],
        crossed_auroc_95=None if record["crossed_auroc_95"] is None else _bootstrap_interval_from_record(record["crossed_auroc_95"]),
        h5_absolute_log_loss_difference_95=(None if record["h5_absolute_log_loss_difference_95"] is None else _bootstrap_interval_from_record(record["h5_absolute_log_loss_difference_95"])),
        h6_absolute_log_loss_difference_95=(None if record["h6_absolute_log_loss_difference_95"] is None else _bootstrap_interval_from_record(record["h6_absolute_log_loss_difference_95"])),
        primary_gate=_hypothesis_gate_from_record(record["primary_gate"]),
        null_results=tuple(_null_result_from_record(item) for item in record["null_results"]),
    )
    return _require_canonical_record(loaded, record, "probe result")


def f2a_selection_bundle_hash(selections: Mapping[str, SelectionManifest]) -> str:
    if set(selections) != set(TASKS) or any(
        not isinstance(selection, SelectionManifest) or selection.task != task
        for task, selection in selections.items()
    ):
        raise ValueError("F2A selection bundle requires one SelectionManifest per registered task")
    return _digest({task: selections[task].sha256 for task in TASKS})


@dataclass(frozen=True)
class ProbeBundleResult:
    schema_version: int
    selection_bundle_hash: str
    authorization_sha256: str
    endpoint_input_sha256: str
    endpoint_input_identities_sha256: str
    results: Mapping[str, ProbeResult]
    gates: F2AGates
    refit_performed: bool = field(default=False, init=False)

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        selections: Mapping[str, SelectionManifest],
    ) -> "ProbeBundleResult":
        return _probe_bundle_from_record(record, selections=selections)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("probe bundle schema is invalid")
        for name in (
            "selection_bundle_hash",
            "authorization_sha256",
            "endpoint_input_sha256",
            "endpoint_input_identities_sha256",
        ):
            _required_sha256(getattr(self, name), name)
        if set(self.results) != set(TASKS) or any(
            not isinstance(self.results[task], ProbeResult) or self.results[task].task != task
            for task in TASKS
        ):
            raise ValueError("probe bundle requires complete typed task results")
        if not isinstance(self.gates, F2AGates):
            raise ValueError("probe bundle requires typed joint gates")
        if any(
            result.authorization_sha256 != self.authorization_sha256
            or result.endpoint_input_sha256 != self.endpoint_input_sha256
            for result in self.results.values()
        ):
            raise ValueError("probe bundle result provenance is inconsistent")
        result_identity_bundle = _digest(
            {
                task: self.results[task].endpoint_input_identities_sha256
                for task in TASKS
            }
        )
        if result_identity_bundle != self.endpoint_input_identities_sha256:
            raise ValueError("probe bundle task identity provenance is inconsistent")
        if (
            self.gates.familiarity_result_sha256 != self.results["familiarity"].sha256
            or self.gates.answerability_result_sha256 != self.results["answerability"].sha256
            or self.gates.unsupported_result_sha256
            != self.results["unsupported_answer"].sha256
        ):
            raise ValueError("probe bundle gate hashes do not match task results")
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selection_bundle_hash": self.selection_bundle_hash,
            "authorization_sha256": self.authorization_sha256,
            "endpoint_input_sha256": self.endpoint_input_sha256,
            "endpoint_input_identities_sha256": self.endpoint_input_identities_sha256,
            "endpoint_source_identities_sha256": self.endpoint_input_identities_sha256,
            "results": {task: self.results[task].to_record() for task in TASKS},
            "gates": self.gates.to_record(),
            "refit_performed": False,
        }

    @property
    def endpoint_source_identities_sha256(self) -> str:
        return self.endpoint_input_identities_sha256

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())


def _calculate_probe_result(
    selection: SelectionManifest,
    authorization: ProbeTestAuthorization,
    rows: Sequence[ProbeRow],
    *,
    endpoint_input_sha256: str,
    endpoint_source_identities_sha256: str,
    expected_source_identities: Sequence[ProbeSourceIdentity],
    null_selections: Sequence["NullSelectionResult"] = (),
) -> ProbeResult:
    rows = tuple(rows)
    if not rows or any(not isinstance(row, ProbeRow) for row in rows):
        raise ValueError("test_rows must contain ProbeRow records")
    if any(row.split != "probe_test" for row in rows):
        raise ValueError("probe evaluation requires only the protected probe_test split")
    if any(row.task != selection.task for row in rows):
        raise ValueError("probe test task does not match the selection manifest")
    if len({row.example_id for row in rows}) != len(rows):
        raise ValueError("probe test contains duplicate IDs")
    expected = _canonical_source_identities(
        expected_source_identities,
        field_name="sealed probe-test source identities",
    )
    evaluated_identities = _canonical_source_identities(
        tuple(ProbeSourceIdentity.from_row(row) for row in rows),
        field_name="probe test source identities",
    )
    if Counter(evaluated_identities) != Counter(expected):
        raise ValueError(
            "probe test rows do not match the sealed probe-test source identities"
        )
    frozen_nulls = tuple(null_selections)
    if any(not isinstance(result, NullSelectionResult) for result in frozen_nulls):
        raise ValueError("probe evaluation requires typed NullSelectionResult records")
    if len({(result.kind, result.seed) for result in frozen_nulls}) != len(frozen_nulls):
        raise ValueError("probe evaluation contains duplicate frozen null selections")
    for null in frozen_nulls:
        if null.selection.task != selection.task or null.test_metrics is not None:
            raise ValueError("null selection is not a frozen unscored selection for this task")
        if null.test_source_identities != expected:
            raise ValueError("null selection was not frozen against the sealed probe-test identities")
    if {row.example_id for row in rows} & set(selection.train_ids + selection.validation_ids):
        raise ValueError("probe test ID leaked into selection")
    _reject_test_group_leakage(selection, rows)
    _validate_transfer_factors(rows, selection.task, "test_rows")
    if any(len(row.output_margin_features) != 11 for row in rows):
        raise ValueError("output_margin features must be exactly 11-dimensional")

    valid = tuple(row for row in rows if row.outcome_status == "valid")
    missing = sum(row.outcome_status == "missing" for row in rows)
    invalid = sum(row.outcome_status == "invalid" for row in rows)
    if any(
        not np.isfinite(row.residual_features).all()
        or not np.isfinite(row.surface_features).all()
        or not np.isfinite(row.output_margin_features).all()
        for row in valid
    ):
        raise ValueError("valid probe test rows must contain finite features")
    model_metrics: dict[str, BinaryMetrics] = {}
    probabilities_by_family: dict[str, np.ndarray] = {}
    classes = _task_classes(selection.task)
    for model in selection.models:
        probabilities = model.predict_proba(valid) if valid else np.empty((0, len(classes)), dtype=np.float64)
        model_key = (
            model.feature_family
            if model.claim_scope == "pre_output"
            else f"{model.feature_family}@{model.anchor}"
        )
        probabilities_by_family[model_key] = probabilities
        model_metrics[model_key] = compute_classification_metrics(
            tuple(row.label for row in valid),
            probabilities if valid else np.empty((0, len(classes)), dtype=np.float64),
            classes=classes,
            threshold=model.threshold,
            total=len(rows),
            missing=missing,
            invalid=invalid,
        )
    selected_model = selection.model_for(selection.selected_feature_family)
    selected_metrics = model_metrics[selection.selected_feature_family]
    selected_probabilities = probabilities_by_family[selection.selected_feature_family]
    selected_probability_by_id = {
        row.example_id: tuple(float(value) for value in probability)
        for row, probability in zip(valid, selected_probabilities)
    }
    per_condition = _group_metrics(
        rows,
        selected_probability_by_id,
        "condition",
        classes,
        selected_model.threshold,
    )
    evaluable_conditions = [
        metrics for metrics in per_condition.values() if metrics.status == "evaluable"
    ]
    worst_condition = (
        None
        if len(evaluable_conditions) != len(per_condition)
        else min(evaluable_conditions, key=lambda item: (item.auroc, item.balanced_accuracy, -item.log_loss))
    )
    ood_transfer = {
        name: _group_metrics(rows, selected_probability_by_id, field_name, classes, selected_model.threshold)
        for name, field_name in (
            ("entity", "entity_id"),
            ("template", "template_id"),
            ("relation", "relation_id"),
            ("domain", "domain"),
        )
    }
    worst_ood_transfer = {
        name: _worst_group_metrics(groups) for name, groups in ood_transfer.items()
    }
    cross_condition_transfer = _score_cross_condition_transfer(selection, rows)
    h5 = _relative_improvement(
        _metric_log_loss(model_metrics.get(NESTED_H5_CANDIDATE)),
        _metric_log_loss(model_metrics.get(NESTED_H5_BASELINE)),
    )
    h6 = _relative_improvement(
        _metric_log_loss(model_metrics.get(NESTED_H6_CANDIDATE)),
        _metric_log_loss(model_metrics.get(NESTED_H5_CANDIDATE)),
    )
    auroc_interval = _transfer_auroc_interval(selection, rows)
    h5_interval = _log_loss_difference_interval(
        valid,
        probabilities_by_family.get(NESTED_H5_BASELINE),
        probabilities_by_family.get(NESTED_H5_CANDIDATE),
        classes,
    )
    h6_interval = _log_loss_difference_interval(
        valid,
        probabilities_by_family.get(NESTED_H5_CANDIDATE),
        probabilities_by_family.get(NESTED_H6_CANDIDATE),
        classes,
    )
    result = ProbeResult(
        schema_version=3,
        task=selection.task,
        selection_hash=selection.sha256,
        authorization_sha256=_digest(authorization.to_record()),
        endpoint_input_sha256=endpoint_input_sha256,
        endpoint_input_identities_sha256=endpoint_source_identities_sha256,
        test_ids=tuple(sorted(row.example_id for row in rows)),
        test_row_sha256s=tuple(sorted(row.sha256 for row in rows)),
        selected_feature_family=selection.selected_feature_family,
        selected_anchor=selected_model.anchor,
        selected_layer=selected_model.layer,
        claim_scope=selected_model.claim_scope,
        selected_model_sha256=selected_model.sha256,
        metrics=selected_metrics,
        model_metrics=model_metrics,
        per_condition=per_condition,
        worst_condition=worst_condition,
        ood_transfer=ood_transfer,
        worst_ood_transfer=worst_ood_transfer,
        cross_condition_transfer=cross_condition_transfer,
        relative_h5_log_loss_improvement=h5,
        relative_h6_log_loss_improvement=h6,
        crossed_auroc_95=auroc_interval,
        h5_absolute_log_loss_difference_95=h5_interval,
        h6_absolute_log_loss_difference_95=h6_interval,
        primary_gate=_primary_gate(
            selection.task,
            selected_metrics,
            worst_condition,
            h5,
            h6,
            cross_condition_transfer,
        ),
    )
    return replace(
        result,
        null_results=tuple(_score_null_selection_on_test(null, rows) for null in frozen_nulls),
    )


def _read_canonical_probe_rows(shard: Any) -> tuple[Mapping[str, Any], ...]:
    try:
        payload = Path(shard.data_path).read_bytes()
    except OSError as error:
        raise ValueError("evaluated probe metrics are unreadable") from error
    if hashlib.sha256(payload).hexdigest() != shard.sha256:
        raise ValueError("evaluated probe metrics changed after verification")
    lines = payload.splitlines()
    if len(lines) != shard.row_count:
        raise ValueError("evaluated probe metrics row count changed")
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("evaluated probe metrics contain invalid JSON") from error
        if not isinstance(row, dict) or _canonical_json(row) != line:
            raise ValueError("evaluated probe metrics must contain canonical JSON objects")
        rows.append(row)
    return tuple(rows)


def _recover_evaluated_probe_bundle(
    metrics_shard: Any,
    *,
    selections: Mapping[str, SelectionManifest],
    authorization: ProbeTestAuthorization,
    endpoint_artifact: Any,
    sealed_identities: Mapping[str, Sequence[ProbeSourceIdentity]],
) -> ProbeBundleResult:
    rows = _read_canonical_probe_rows(metrics_shard)
    if (
        metrics_shard.record_kind != "metrics"
        or len(rows) != 1
        or set(rows[0]) != {"kind", "metric_type", "result"}
        or rows[0].get("kind") != "metrics"
        or rows[0].get("metric_type") != "f2a_bundle"
    ):
        raise ValueError("evaluated probe metrics artifact has an invalid schema")
    bundle = ProbeBundleResult.from_record(rows[0]["result"], selections=selections)
    identity_hash = _task_identity_bundle_digest(sealed_identities)
    if (
        bundle.selection_bundle_hash != f2a_selection_bundle_hash(selections)
        or bundle.authorization_sha256 != authorization.sha256
        or bundle.endpoint_input_sha256 != endpoint_artifact.sha256
        or bundle.endpoint_input_identities_sha256 != identity_hash
    ):
        raise ValueError("evaluated probe bundle does not match this protected execution")
    try:
        manifest = json.loads(Path(metrics_shard.manifest_path).read_bytes())
        lineage = manifest["lineage"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError("evaluated probe metrics lineage is unreadable") from error
    expected_lineage = {
        "selection_manifest": bundle.selection_bundle_hash,
        "authorization": authorization.sha256,
        "endpoint_input_sha256": endpoint_artifact.sha256,
        "endpoint_source_identities_sha256": identity_hash,
    }
    if not isinstance(lineage, Mapping) or dict(lineage) != expected_lineage:
        raise ValueError("evaluated probe metrics lineage does not verify")
    return bundle


def evaluate_probe_bundle_once(
    selections: Mapping[str, SelectionManifest],
    authorization: ProbeTestAuthorization,
    rows_by_task: Mapping[str, Sequence[ProbeRow]],
    *,
    null_selections_by_task: Mapping[str, Sequence["NullSelectionResult"]] | None = None,
    store: FAArtifactStore | None = None,
    endpoint_manifest_path: str | Path | None = None,
) -> ProbeBundleResult:
    if not isinstance(store, FAArtifactStore):
        raise ValueError("probe bundle evaluation requires a persistent FAArtifactStore")
    if endpoint_manifest_path is None:
        raise ValueError("probe bundle evaluation requires the sealed endpoint manifest path")
    selection_bundle_hash = f2a_selection_bundle_hash(selections)
    if not isinstance(authorization, ProbeTestAuthorization) or authorization.selection_hash != selection_bundle_hash:
        raise ValueError("authorization selection hash does not match the F2A selection bundle")
    endpoint_artifact = store.verify_endpoint_artifact("probe_test", endpoint_manifest_path)
    sealed_identities = _sealed_probe_test_identities(endpoint_artifact)
    state = store.endpoint_state("probe_test", endpoint_manifest_path)
    if state == "evaluated":
        metrics_shard = store.read_evaluated_metrics(
            "probe_test", endpoint_manifest_path
        )
        recovered = _recover_evaluated_probe_bundle(
            metrics_shard,
            selections=selections,
            authorization=authorization,
            endpoint_artifact=endpoint_artifact,
            sealed_identities=sealed_identities,
        )
        store.close_endpoint("probe_test")
        return recovered
    durable_receipt = store.unlock_or_resume_endpoint("probe_test", endpoint_manifest_path)
    if (
        durable_receipt.endpoint != authorization.endpoint
        or durable_receipt.lease_id != authorization.lease_id
        or durable_receipt.preregistration_hash != authorization.preregistration_hash
        or durable_receipt.selection_manifest_hash != authorization.selection_hash
    ):
        raise ValueError("authorization does not match the durable endpoint lease")
    if set(rows_by_task) != set(TASKS):
        raise ValueError("probe bundle rows must contain every registered task")
    for task in TASKS:
        evaluated = _canonical_source_identities(
            tuple(ProbeSourceIdentity.from_row(row) for row in rows_by_task[task]),
            field_name=f"{task} probe-test source identities",
        )
        if Counter(evaluated) != Counter(sealed_identities[task]):
            raise ValueError(
                "probe test rows do not match the sealed probe-test source identities"
            )
    null_map = null_selections_by_task or {}
    results = {
        task: _calculate_probe_result(
            selections[task],
            authorization,
            rows_by_task[task],
            endpoint_input_sha256=endpoint_artifact.sha256,
            endpoint_source_identities_sha256=_source_identity_digest(
                sealed_identities[task]
            ),
            expected_source_identities=sealed_identities[task],
            null_selections=null_map.get(task, ()),
        )
        for task in TASKS
    }
    gates = evaluate_f2a_gates(
        results["familiarity"], results["answerability"], results["unsupported_answer"]
    )
    bundle = ProbeBundleResult(
        schema_version=1,
        selection_bundle_hash=selection_bundle_hash,
        authorization_sha256=authorization.sha256,
        endpoint_input_sha256=endpoint_artifact.sha256,
        endpoint_input_identities_sha256=_task_identity_bundle_digest(
            sealed_identities
        ),
        results=results,
        gates=gates,
    )
    base = store.root / "runs" / "familiarity_answerability"
    relative_parts = endpoint_artifact.data_path.relative_to(base).parts
    if not relative_parts:
        raise ValueError("probe endpoint artifact has no durable run identity")
    metrics = store.write_completed_shard(
        relative_parts[0],
        "probe_test",
        f"probe-bundle-metrics-{authorization.lease_id}",
        ({"kind": "metrics", "metric_type": "f2a_bundle", "result": bundle.to_record()},),
        {
            "selection_manifest": selection_bundle_hash,
            "authorization": authorization.sha256,
            "endpoint_input_sha256": endpoint_artifact.sha256,
            "endpoint_source_identities_sha256": _task_identity_bundle_digest(
                sealed_identities
            ),
        },
        record_kind="metrics",
    )
    store.mark_evaluated(durable_receipt, metrics.data_path)
    store.close_endpoint("probe_test")
    return bundle


def evaluate_probe_test_once(
    selection: SelectionManifest,
    authorization: ProbeTestAuthorization,
    test_rows: Sequence[ProbeRow],
    *,
    null_selections: Sequence["NullSelectionResult"] = (),
    store: FAArtifactStore | None = None,
    endpoint_manifest_path: str | Path | None = None,
) -> ProbeResult:
    """Evaluate frozen pipelines exactly once under a hash-bound capability."""

    raise ValueError(
        "probe_test is bundle-only; evaluate all three registered tasks atomically"
    )

    if not isinstance(selection, SelectionManifest):
        raise ValueError("evaluation requires an immutable SelectionManifest")
    if not isinstance(authorization, ProbeTestAuthorization):
        raise ValueError("evaluation requires a ProbeTestAuthorization")
    if authorization.selection_hash != selection.sha256:
        raise ValueError("authorization selection hash does not match selection manifest")
    if not isinstance(store, FAArtifactStore):
        raise ValueError("probe evaluation requires a persistent FAArtifactStore")
    if endpoint_manifest_path is None:
        raise ValueError("probe evaluation requires the sealed endpoint manifest path")
    durable_receipt = store.unlock_or_resume_endpoint("probe_test", endpoint_manifest_path)
    endpoint_artifact = store.verify_endpoint_artifact("probe_test", endpoint_manifest_path)
    sealed_identities = _sealed_probe_test_identities(endpoint_artifact)
    if (
        durable_receipt.endpoint != authorization.endpoint
        or durable_receipt.lease_id != authorization.lease_id
        or durable_receipt.preregistration_hash != authorization.preregistration_hash
        or durable_receipt.selection_manifest_hash != authorization.selection_hash
    ):
        raise ValueError("authorization does not match the durable endpoint lease")

    rows = tuple(test_rows)
    if not rows or any(not isinstance(row, ProbeRow) for row in rows):
        raise ValueError("test_rows must contain ProbeRow records")
    if any(row.split != "probe_test" for row in rows):
        raise ValueError("probe evaluation requires only the protected probe_test split")
    if any(row.task != selection.task for row in rows):
        raise ValueError("probe test task does not match the selection manifest")
    if len({row.example_id for row in rows}) != len(rows):
        raise ValueError("probe test contains duplicate IDs")
    evaluated_identities = _canonical_identities(
        tuple(ProbeRowIdentity.from_row(row) for row in rows),
        field_name="probe test identities",
    )
    if Counter(evaluated_identities) != Counter(sealed_identities):
        raise ValueError("probe test rows do not match the sealed probe-test identities")
    frozen_nulls = tuple(null_selections)
    if any(not isinstance(result, NullSelectionResult) for result in frozen_nulls):
        raise ValueError("probe evaluation requires typed NullSelectionResult records")
    if len({(result.kind, result.seed) for result in frozen_nulls}) != len(frozen_nulls):
        raise ValueError("probe evaluation contains duplicate frozen null selections")
    for null in frozen_nulls:
        if null.selection.task != selection.task or null.test_metrics is not None:
            raise ValueError("null selection is not a frozen unscored selection for this task")
        if null.test_source_identities != sealed_identities:
            raise ValueError("null selection was not frozen against the sealed probe-test identities")
    if {row.example_id for row in rows} & set(selection.train_ids + selection.validation_ids):
        raise ValueError("probe test ID leaked into selection")
    _reject_test_group_leakage(selection, rows)
    if any(len(row.output_margin_features) != 11 for row in rows):
        raise ValueError("output_margin features must be exactly 11-dimensional")

    valid = tuple(row for row in rows if row.outcome_status == "valid")
    missing = sum(row.outcome_status == "missing" for row in rows)
    invalid = sum(row.outcome_status == "invalid" for row in rows)
    if any(
        not np.isfinite(row.residual_features).all()
        or not np.isfinite(row.surface_features).all()
        or not np.isfinite(row.output_margin_features).all()
        for row in valid
    ):
        raise ValueError("valid probe test rows must contain finite features")
    model_metrics: dict[str, BinaryMetrics] = {}
    probabilities_by_family: dict[str, np.ndarray] = {}
    classes = _task_classes(selection.task)
    for model in selection.models:
        probabilities = model.predict_proba(valid) if valid else np.asarray([], dtype=np.float64)
        model_key = (
            model.feature_family
            if model.claim_scope == "pre_output"
            else f"{model.feature_family}@{model.anchor}"
        )
        probabilities_by_family[model_key] = probabilities
        if not valid:
            probabilities = np.empty((0, len(classes)), dtype=np.float64)
        model_metrics[model_key] = compute_classification_metrics(
            tuple(row.label for row in valid),
            probabilities,
            classes=classes,
            threshold=model.threshold,
            total=len(rows),
            missing=missing,
            invalid=invalid,
        )
    selected_model = selection.model_for(selection.selected_feature_family)
    selected_metrics = model_metrics[selection.selected_feature_family]
    selected_probabilities = probabilities_by_family[selection.selected_feature_family]
    selected_probability_by_id = {
        row.example_id: tuple(float(value) for value in probability)
        for row, probability in zip(valid, selected_probabilities)
    }
    per_condition = _group_metrics(
        rows,
        selected_probability_by_id,
        "condition",
        classes,
        selected_model.threshold,
    )
    evaluable_conditions = [
        metrics for metrics in per_condition.values() if metrics.status == "evaluable"
    ]
    worst_condition = (
        None
        if len(evaluable_conditions) != len(per_condition)
        else min(
            evaluable_conditions,
            key=lambda item: (item.auroc, item.balanced_accuracy, -item.log_loss),
        )
    )
    ood_transfer = {
        name: _group_metrics(
            rows,
            selected_probability_by_id,
            field_name,
            classes,
            selected_model.threshold,
        )
        for name, field_name in (
            ("entity", "entity_id"),
            ("template", "template_id"),
            ("relation", "relation_id"),
            ("domain", "domain"),
        )
    }
    worst_ood_transfer = {
        name: _worst_group_metrics(groups) for name, groups in ood_transfer.items()
    }
    h5 = _relative_improvement(
        _metric_log_loss(model_metrics.get(NESTED_H5_CANDIDATE)),
        _metric_log_loss(model_metrics.get(NESTED_H5_BASELINE)),
    )
    h6 = _relative_improvement(
        _metric_log_loss(model_metrics.get(NESTED_H6_CANDIDATE)),
        _metric_log_loss(model_metrics.get(NESTED_H5_CANDIDATE)),
    )
    auroc_interval = _auroc_interval(valid, selected_probabilities, classes, selected_model.threshold) if selection.task in {"familiarity", "answerability"} else None
    h5_interval = _log_loss_difference_interval(valid, probabilities_by_family.get(NESTED_H5_BASELINE), probabilities_by_family.get(NESTED_H5_CANDIDATE), classes)
    h6_interval = _log_loss_difference_interval(valid, probabilities_by_family.get(NESTED_H5_CANDIDATE), probabilities_by_family.get(NESTED_H6_CANDIDATE), classes)
    primary_gate = _primary_gate(selection.task, selected_metrics, worst_condition, h5, h6)
    result = ProbeResult(
        schema_version=2,
        task=selection.task,
        selection_hash=selection.sha256,
        authorization_sha256=_digest(authorization.to_record()),
        endpoint_input_sha256=endpoint_artifact.sha256,
        endpoint_input_identities_sha256=_identity_digest(sealed_identities),
        test_ids=tuple(sorted(row.example_id for row in rows)),
        test_row_sha256s=tuple(sorted(row.sha256 for row in rows)),
        selected_feature_family=selection.selected_feature_family,
        metrics=selected_metrics,
        model_metrics=model_metrics,
        per_condition=per_condition,
        worst_condition=worst_condition,
        ood_transfer=ood_transfer,
        worst_ood_transfer=worst_ood_transfer,
        relative_h5_log_loss_improvement=h5,
        relative_h6_log_loss_improvement=h6,
        crossed_auroc_95=auroc_interval,
        h5_absolute_log_loss_difference_95=h5_interval,
        h6_absolute_log_loss_difference_95=h6_interval,
        primary_gate=primary_gate,
    )
    result = replace(
        result,
        null_results=tuple(_score_null_selection_on_test(null, rows) for null in frozen_nulls),
    )
    base = store.root / "runs" / "familiarity_answerability"
    relative_parts = endpoint_artifact.data_path.relative_to(base).parts
    if not relative_parts:
        raise ValueError("probe endpoint artifact has no durable run identity")
    metrics = store.write_completed_shard(
        relative_parts[0],
        "probe_test",
        f"probe-metrics-{authorization.lease_id}",
        ({"kind": "metrics", "result": result.to_record()},),
        {
            "selection_manifest": selection.sha256,
            "authorization": authorization.sha256,
            "endpoint_input_sha256": endpoint_artifact.sha256,
            "endpoint_input_identities_sha256": _identity_digest(sealed_identities),
        },
        record_kind="metrics",
    )
    store.mark_evaluated(durable_receipt, metrics.data_path)
    store.close_endpoint("probe_test")
    return result


def _reject_test_group_leakage(selection: SelectionManifest, rows: Sequence[ProbeRow]) -> None:
    _require_registered_domains(rows, "test_rows")
    for field_name, selected_values, message in (
        ("entity_id", selection.train_entity_ids + selection.validation_entity_ids, "entity leakage"),
        (
            "template_id",
            selection.train_template_ids + selection.validation_template_ids,
            "template leakage",
        ),
        (
            "relation_id",
            selection.train_relation_ids + selection.validation_relation_ids,
            "relation leakage",
        ),
    ):
        if {getattr(row, field_name) for row in rows} & set(selected_values):
            raise ValueError(f"probe test contains {message} from selection")


def _fit_transfer_rotations(
    *,
    task: str,
    train_rows: Sequence[ProbeRow],
    validation_rows: Sequence[ProbeRow],
    train_matrix: np.ndarray,
    validation_matrix: np.ndarray,
    family: str,
    anchor: str,
    layer: int | None,
    pca_count: int | None,
    c_value: float,
    prototype: Any,
    seed: int,
    classes: tuple[int | str, ...],
    selector: tuple[int, ...],
) -> tuple[FrozenTransferRotation, ...]:
    train_labels = np.asarray([row.label for row in train_rows])
    validation_labels = np.asarray([row.label for row in validation_rows])
    rotations = []
    for train_condition, test_conditions in _registered_rotation_specs(task):
        train_mask = np.asarray(
            [_transfer_condition(row, task) == train_condition for row in train_rows]
        )
        test_mask = np.asarray(
            [_transfer_condition(row, task) in test_conditions for row in validation_rows]
        )
        if set(train_labels[train_mask].tolist()) != set(classes):
            raise ValueError("mechanism_train rotation is missing a registered target class")
        scaler = StandardScaler().fit(train_matrix[train_mask])
        transformed_train = scaler.transform(train_matrix[train_mask])
        transformed_validation = scaler.transform(validation_matrix[test_mask])
        pca: PCA | None = None
        if pca_count is not None:
            if pca_count > min(transformed_train.shape):
                raise ValueError("PCA components exceed a train-condition rotation rank")
            pca = PCA(n_components=pca_count, svd_solver="full").fit(transformed_train)
            transformed_train = pca.transform(transformed_train)
            transformed_validation = pca.transform(transformed_validation)
        estimator = _fresh_estimator(prototype, c_value, seed, classes)
        estimator.fit(transformed_train, train_labels[train_mask])
        probabilities = _class_probabilities(estimator, transformed_validation, classes)
        threshold = (
            _select_threshold(validation_labels[test_mask], probabilities[:, 1])
            if classes == (0, 1)
            else None
        )
        metrics = compute_classification_metrics(
            tuple(validation_labels[test_mask].tolist()),
            probabilities,
            classes=classes,
            threshold=threshold,
        )
        if metrics.status != "evaluable":
            raise ValueError("locked_validation reciprocal rotation is not evaluable")
        model = _freeze_model(
            estimator,
            family,
            anchor,
            layer,
            pca_count,
            c_value,
            threshold,
            selector,
            scaler,
            pca,
            metrics,
            classes,
        )
        validation_result = _score_transfer_rotation(
            task,
            train_condition,
            test_conditions,
            model,
            validation_rows,
        )
        rotations.append(
            FrozenTransferRotation(
                task,
                train_condition,
                test_conditions,
                model,
                validation_result,
            )
        )
    return tuple(rotations)


def _metrics_for_transfer_rows(
    rows: Sequence[ProbeRow],
    probability_by_id: Mapping[str, tuple[float, ...]],
    classes: tuple[int | str, ...],
    threshold: float | None,
) -> BinaryMetrics:
    valid = tuple(row for row in rows if row.outcome_status == "valid")
    return compute_classification_metrics(
        tuple(row.label for row in valid),
        tuple(probability_by_id[row.example_id] for row in valid),
        classes=classes,
        threshold=threshold,
        total=len(rows),
        missing=sum(row.outcome_status == "missing" for row in rows),
        invalid=sum(row.outcome_status == "invalid" for row in rows),
    )


def _score_transfer_rotation(
    task: str,
    train_condition: str,
    test_conditions: tuple[str, ...],
    model: FrozenProbeModel,
    rows: Sequence[ProbeRow],
) -> CrossConditionRotationResult:
    test_rows = tuple(
        row for row in rows if _transfer_condition(row, task) in test_conditions
    )
    valid = tuple(row for row in test_rows if row.outcome_status == "valid")
    probabilities = (
        model.predict_proba(valid)
        if valid
        else np.empty((0, len(model.classes)), dtype=np.float64)
    )
    probability_by_id = {
        row.example_id: tuple(float(value) for value in probability)
        for row, probability in zip(valid, probabilities)
    }
    condition_results = []
    for test_condition in test_conditions:
        condition_rows = tuple(
            row
            for row in test_rows
            if _transfer_condition(row, task) == test_condition
        )
        cells = tuple(
            DistractorFamiliarityCellResult(
                distractor_familiarity,
                _metrics_for_transfer_rows(
                    tuple(
                        row
                        for row in condition_rows
                        if row.distractor_familiarity_condition == distractor_familiarity
                    ),
                    probability_by_id,
                    model.classes,
                    model.threshold,
                ),
            )
            for distractor_familiarity in TARGET_FAMILIARITY_CONDITIONS
        )
        condition_results.append(
            TransferConditionResult(
                test_condition,
                _metrics_for_transfer_rows(
                    condition_rows,
                    probability_by_id,
                    model.classes,
                    model.threshold,
                ),
                cells,
            )
        )
    return CrossConditionRotationResult(
        task,
        train_condition,
        test_conditions,
        _metrics_for_transfer_rows(
            test_rows,
            probability_by_id,
            model.classes,
            model.threshold,
        ),
        tuple(condition_results),
    )


def _score_cross_condition_transfer(
    selection: SelectionManifest, rows: Sequence[ProbeRow]
) -> CrossConditionTransferSummary | None:
    if selection.task not in {"familiarity", "answerability"}:
        return None
    return CrossConditionTransferSummary(
        selection.task,
        tuple(
            _score_transfer_rotation(
                selection.task,
                rotation.train_condition,
                rotation.test_conditions,
                rotation.model,
                rows,
            )
            for rotation in selection.transfer_rotations
        ),
    )


def _group_metrics(
    rows: Sequence[ProbeRow],
    probability_by_id: Mapping[str, tuple[float, ...]],
    field_name: str,
    classes: tuple[int | str, ...],
    threshold: float | None,
) -> Mapping[str, BinaryMetrics]:
    result = {}
    for value in sorted({getattr(row, field_name) for row in rows}):
        group = tuple(row for row in rows if getattr(row, field_name) == value)
        valid = tuple(row for row in group if row.outcome_status == "valid")
        result[value] = compute_classification_metrics(
            tuple(row.label for row in valid),
            tuple(probability_by_id[row.example_id] for row in valid),
            classes=classes,
            threshold=threshold,
            total=len(group),
            missing=sum(row.outcome_status == "missing" for row in group),
            invalid=sum(row.outcome_status == "invalid" for row in group),
        )
    return MappingProxyType(result)


def _worst_group_metrics(
    groups: Mapping[str, BinaryMetrics],
) -> BinaryMetrics | None:
    evaluable = [metrics for metrics in groups.values() if metrics.status == "evaluable"]
    if len(evaluable) != len(groups):
        return None
    return min(
        evaluable,
        key=lambda item: (item.auroc, item.balanced_accuracy, -item.log_loss),
    )


def _metric_log_loss(metrics: BinaryMetrics | None) -> float | None:
    return None if metrics is None or metrics.status != "evaluable" else metrics.log_loss


def _best_log_loss(metrics: Mapping[str, BinaryMetrics], names: Sequence[str]) -> float | None:
    values = [
        value
        for value in (_metric_log_loss(metrics.get(name)) for name in names)
        if value is not None
    ]
    return None if not values else min(values)


def _relative_improvement(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline <= 0.0:
        return None
    return (baseline - candidate) / baseline


def _primary_gate(
    task: str,
    metrics: BinaryMetrics,
    worst_condition: BinaryMetrics | None,
    h5_improvement: float | None,
    h6_improvement: float | None,
    cross_condition_transfer: CrossConditionTransferSummary | None = None,
) -> HypothesisGate:
    if task in {"familiarity", "answerability"}:
        hypothesis = "H3" if task == "familiarity" else "H4"
        criteria = (
            GateCriterion(
                "reciprocal-transfer mean AUROC",
                (
                    None
                    if cross_condition_transfer is None
                    else cross_condition_transfer.mean_auroc
                ),
                CONFIRMATORY_THRESHOLDS["probe_auroc_min"],
                ">=",
            ),
            GateCriterion(
                "reciprocal-transfer mean balanced accuracy",
                (
                    None
                    if cross_condition_transfer is None
                    else cross_condition_transfer.mean_balanced_accuracy
                ),
                CONFIRMATORY_THRESHOLDS["probe_balanced_accuracy_min"],
                ">=",
            ),
            GateCriterion(
                "worst reciprocal-transfer cell balanced accuracy",
                (
                    None
                    if cross_condition_transfer is None
                    else cross_condition_transfer.worst_cell_balanced_accuracy
                ),
                CONFIRMATORY_THRESHOLDS["probe_balanced_accuracy_min"],
                ">=",
            ),
        )
        return HypothesisGate(hypothesis, criteria)
    return HypothesisGate(
        "H5",
        (
            GateCriterion(
                "nested held-out log-loss improvement",
                h5_improvement,
                CONFIRMATORY_THRESHOLDS["h5_relative_log_loss_min"],
                ">=",
            ),
        ),
        ("H6 is secondary and cannot invalidate H5",)
        if h6_improvement is not None
        else ("H6 is not evaluable without both frozen dynamics and static models",),
    )


def holm_correct_h3_h4(p_values: Mapping[str, float]) -> Mapping[str, float]:
    if set(p_values) != {"H3", "H4"}:
        raise ValueError("Holm correction requires exactly H3 and H4")
    values = {name: float(value) for name, value in p_values.items()}
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError("Holm p-values must be finite and in [0, 1]")
    ordered = sorted(values, key=lambda name: (values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * values[name]))
        adjusted[name] = running
    return MappingProxyType(adjusted)


@dataclass(frozen=True)
class F2AGates:
    """Joint H3-H6 decisions derived only from immutable probe results."""

    familiarity_result_sha256: str
    answerability_result_sha256: str
    unsupported_result_sha256: str
    holm_adjusted_p: Mapping[str, float]
    h3: HypothesisGate
    h4: HypothesisGate
    h5: HypothesisGate
    h6: HypothesisGate
    h6_secondary: bool = field(default=True, init=False)

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        results: Mapping[str, ProbeResult],
    ) -> "F2AGates":
        return _f2a_gates_from_record(record, results=results)

    def __post_init__(self) -> None:
        for name in (
            "familiarity_result_sha256",
            "answerability_result_sha256",
            "unsupported_result_sha256",
        ):
            _required_sha256(getattr(self, name), name)
        object.__setattr__(self, "holm_adjusted_p", MappingProxyType(dict(self.holm_adjusted_p)))
        if set(self.holm_adjusted_p) != {"H3", "H4"}:
            raise ValueError("F2A gates require Holm-adjusted H3 and H4 p-values")
        if (self.h3.hypothesis, self.h4.hypothesis, self.h5.hypothesis, self.h6.hypothesis) != (
            "H3",
            "H4",
            "H5",
            "H6",
        ):
            raise ValueError("F2A gate records must be ordered H3 through H6")

    @property
    def status(self) -> str:
        primary = (self.h3.status, self.h4.status, self.h5.status)
        if "not_evaluable" in primary:
            return "not_evaluable"
        return "supported" if all(value == "supported" for value in primary) else "not_supported"

    def to_record(self) -> dict[str, Any]:
        return {
            "familiarity_result_sha256": self.familiarity_result_sha256,
            "answerability_result_sha256": self.answerability_result_sha256,
            "unsupported_result_sha256": self.unsupported_result_sha256,
            "holm_adjusted_p": dict(self.holm_adjusted_p),
            "h3": self.h3.to_record(),
            "h4": self.h4.to_record(),
            "h5": self.h5.to_record(),
            "h6": self.h6.to_record(),
            "h6_secondary": True,
            "status": self.status,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())


def evaluate_f2a_gates(
    familiarity: ProbeResult,
    answerability: ProbeResult,
    unsupported_answer: ProbeResult,
) -> F2AGates:
    """Apply joint Holm correction and registered H3-H6 thresholds."""

    inputs = (familiarity, answerability, unsupported_answer)
    if any(not isinstance(result, ProbeResult) for result in inputs):
        raise ValueError("F2A gates require immutable ProbeResult records")
    if tuple(result.task for result in inputs) != (
        "familiarity",
        "answerability",
        "unsupported_answer",
    ):
        raise ValueError("F2A gate results must be familiarity, answerability, unsupported_answer")
    for observed in inputs:
        if (
            observed.claim_scope != "pre_output"
            or observed.selected_anchor == "assistant_prefix_end"
            or observed.selected_layer == REGISTERED_LAYERS[-1]
        ):
            raise ValueError(
                "F2A confirmatory gates require a bound pre-output scope"
            )
    for observed in inputs:
        for null in observed.null_results:
            if (
                null.selection.task != observed.task
                or null.test_metrics is None
                or null.test_ids != observed.test_ids
                or _identity_digest(null.test_source_identities)
                != observed.endpoint_input_identities_sha256
            ):
                raise ValueError("F2A null result is not bound to the observed protected test")
    for observed in (familiarity, answerability):
        if (
            not isinstance(observed.cross_condition_transfer, CrossConditionTransferSummary)
            or observed.cross_condition_transfer.task != observed.task
        ):
            raise ValueError("H3/H4 require complete reciprocal cross-condition transfer")
        for null in observed.null_results:
            if (
                null.kind == "label_permutation"
                and (
                    not isinstance(
                        null.test_cross_condition_transfer, CrossConditionTransferSummary
                    )
                    or null.test_cross_condition_transfer.task != observed.task
                )
            ):
                raise ValueError(
                    "H3/H4 label-permutation nulls require reciprocal transfer results"
                )
    h3_nulls = _nulls_for(familiarity.null_results, "familiarity", "label_permutation")
    h4_nulls = _nulls_for(answerability.null_results, "answerability", "label_permutation")
    raw_p = {
        "H3": _full_selection_null_p(
            familiarity.cross_condition_transfer.mean_auroc, h3_nulls
        ),
        "H4": _full_selection_null_p(
            answerability.cross_condition_transfer.mean_auroc, h4_nulls
        ),
    }
    if any(value is None for value in raw_p.values()):
        adjusted = MappingProxyType(
            {name: 1.0 if value is None else float(value) for name, value in raw_p.items()}
        )
        p_reasons = {
            name: (
                "registered full-selection label-permutation nulls are missing",
            )
            if raw_p[name] is None
            else ("joint Holm correction is not evaluable because its paired null is missing",)
            for name in raw_p
        }
        adjusted_for_gate: Mapping[str, float | None] = {"H3": None, "H4": None}
    else:
        adjusted = holm_correct_h3_h4({name: float(value) for name, value in raw_p.items()})
        p_reasons = {"H3": (), "H4": ()}
        adjusted_for_gate = adjusted
    h3 = _decoding_gate(
        "H3",
        familiarity,
        adjusted_for_gate["H3"],
        context_reasons=p_reasons["H3"],
    )
    h4 = _decoding_gate(
        "H4",
        answerability,
        adjusted_for_gate["H4"],
        context_reasons=p_reasons["H4"],
    )
    h5 = HypothesisGate(
        "H5",
        (
            GateCriterion(
                "nested held-out log-loss improvement",
                unsupported_answer.relative_h5_log_loss_improvement,
                CONFIRMATORY_THRESHOLDS["h5_relative_log_loss_min"],
                ">=",
            ),
            GateCriterion(
                "crossed-bootstrap 95% absolute log-loss difference lower bound",
                None if unsupported_answer.h5_absolute_log_loss_difference_95 is None else unsupported_answer.h5_absolute_log_loss_difference_95.lower,
                0.0,
                ">",
            ),
        ),
    )
    layer_order_nulls = _nulls_for(
        unsupported_answer.null_results, "unsupported_answer", "layer_order"
    )
    random_map_nulls = _nulls_for(
        unsupported_answer.null_results, "unsupported_answer", "random_map"
    )
    h6_observed = unsupported_answer.relative_h6_log_loss_improvement
    layer_margin = _null_improvement_margin(h6_observed, layer_order_nulls)
    random_margin = _null_improvement_margin(h6_observed, random_map_nulls)
    h6_reasons = []
    if not layer_order_nulls:
        h6_reasons.append("registered full-selection layer-order nulls are missing")
    if not random_map_nulls:
        h6_reasons.append("registered norm-preserving random-map nulls are missing")
    h6_reasons.append("H6 is secondary and cannot invalidate H1-H5")
    h6 = HypothesisGate(
        "H6",
        (
            GateCriterion(
                "dynamics held-out log-loss improvement",
                h6_observed,
                CONFIRMATORY_THRESHOLDS["h6_relative_log_loss_min"],
                ">=",
            ),
            GateCriterion(
                "crossed-bootstrap 95% absolute log-loss difference lower bound",
                None if unsupported_answer.h6_absolute_log_loss_difference_95 is None else unsupported_answer.h6_absolute_log_loss_difference_95.lower,
                0.0,
                ">",
            ),
            GateCriterion(
                "dynamics improvement over all layer-order nulls",
                layer_margin,
                0.0,
                ">",
            ),
            GateCriterion(
                "dynamics improvement over all random-map nulls",
                random_margin,
                0.0,
                ">",
            ),
        ),
        tuple(h6_reasons),
    )
    return F2AGates(
        familiarity_result_sha256=familiarity.sha256,
        answerability_result_sha256=answerability.sha256,
        unsupported_result_sha256=unsupported_answer.sha256,
        holm_adjusted_p=adjusted,
        h3=h3,
        h4=h4,
        h5=h5,
        h6=h6,
    )


def _f2a_gates_from_record(
    value: Any, *, results: Mapping[str, ProbeResult]
) -> F2AGates:
    record = _record_with_schema(
        value,
        {
            "familiarity_result_sha256", "answerability_result_sha256",
            "unsupported_result_sha256", "holm_adjusted_p", "h3", "h4", "h5",
            "h6", "h6_secondary", "status",
        },
        "F2A gates",
    )
    if set(results) != set(TASKS) or any(
        not isinstance(results[task], ProbeResult) for task in TASKS
    ):
        raise ValueError("F2A gates require complete typed result records")
    loaded = F2AGates(
        familiarity_result_sha256=record["familiarity_result_sha256"],
        answerability_result_sha256=record["answerability_result_sha256"],
        unsupported_result_sha256=record["unsupported_result_sha256"],
        holm_adjusted_p=record["holm_adjusted_p"],
        h3=_hypothesis_gate_from_record(record["h3"]),
        h4=_hypothesis_gate_from_record(record["h4"]),
        h5=_hypothesis_gate_from_record(record["h5"]),
        h6=_hypothesis_gate_from_record(record["h6"]),
    )
    expected = evaluate_f2a_gates(
        results["familiarity"], results["answerability"], results["unsupported_answer"]
    )
    if loaded.to_record() != expected.to_record():
        raise ValueError("F2A gates are not canonical for their result records")
    return _require_canonical_record(loaded, record, "F2A gates")


def _probe_bundle_from_record(
    value: Any, *, selections: Mapping[str, SelectionManifest]
) -> ProbeBundleResult:
    record = _record_with_schema(
        value,
        {
            "schema_version", "selection_bundle_hash", "authorization_sha256",
            "endpoint_input_sha256", "endpoint_input_identities_sha256",
            "endpoint_source_identities_sha256", "results", "gates", "refit_performed",
        },
        "probe bundle",
    )
    expected_selection_hash = f2a_selection_bundle_hash(selections)
    if (
        record["selection_bundle_hash"] != expected_selection_hash
        or record["endpoint_source_identities_sha256"]
        != record["endpoint_input_identities_sha256"]
        or record["refit_performed"] is not False
        or not isinstance(record["results"], Mapping)
        or set(record["results"]) != set(TASKS)
    ):
        raise ValueError("probe bundle provenance or schema is invalid")
    results = {
        task: ProbeResult.from_record(record["results"][task], selection=selections[task])
        for task in TASKS
    }
    gates = F2AGates.from_record(record["gates"], results=results)
    loaded = ProbeBundleResult(
        schema_version=record["schema_version"],
        selection_bundle_hash=record["selection_bundle_hash"],
        authorization_sha256=record["authorization_sha256"],
        endpoint_input_sha256=record["endpoint_input_sha256"],
        endpoint_input_identities_sha256=record["endpoint_input_identities_sha256"],
        results=results,
        gates=gates,
    )
    return _require_canonical_record(loaded, record, "probe bundle")


def _full_selection_null_p(
    observed_auroc: float | None,
    nulls: Sequence["NullSelectionResult"],
) -> float | None:
    if observed_auroc is None or not nulls:
        return None
    values = tuple(
        (
            None
            if result.test_cross_condition_transfer is None
            else result.test_cross_condition_transfer.mean_auroc
        )
        for result in nulls
    )
    if any(value is None for value in values):
        return None
    exceedances = sum(float(value) >= observed_auroc for value in values)
    return (1.0 + exceedances) / (1.0 + len(values))


def _decoding_gate(
    hypothesis: str,
    result: ProbeResult,
    adjusted_p: float | None,
    *,
    context_reasons: Sequence[str],
) -> HypothesisGate:
    transfer = result.cross_condition_transfer
    criteria = [
        GateCriterion(
            "reciprocal-transfer mean AUROC",
            None if transfer is None else transfer.mean_auroc,
            CONFIRMATORY_THRESHOLDS["probe_auroc_min"],
            ">=",
        ),
        GateCriterion(
            "reciprocal-transfer mean balanced accuracy",
            None if transfer is None else transfer.mean_balanced_accuracy,
            CONFIRMATORY_THRESHOLDS["probe_balanced_accuracy_min"],
            ">=",
        ),
        GateCriterion(
            "worst reciprocal-transfer cell balanced accuracy",
            None if transfer is None else transfer.worst_cell_balanced_accuracy,
            CONFIRMATORY_THRESHOLDS["probe_balanced_accuracy_min"],
            ">=",
        ),
    ]
    criteria.append(
        GateCriterion(
            "crossed-bootstrap 95% AUROC lower bound",
            None if result.crossed_auroc_95 is None else result.crossed_auroc_95.lower,
            0.5,
            ">",
        )
    )
    criteria.append(GateCriterion("Holm-adjusted p-value", adjusted_p, 0.05, "<="))
    return HypothesisGate(hypothesis, tuple(criteria), tuple(context_reasons))


@dataclass(frozen=True)
class NullSelectionResult:
    kind: str
    seed: int
    config: Mapping[str, Any]
    config_sha256: str
    selection: SelectionManifest
    max_norm_error: float
    test_source_identities: tuple[ProbeSourceIdentity, ...]
    test_transform: Mapping[str, Any]
    test_ids: tuple[str, ...] = ()
    test_row_sha256s: tuple[str, ...] = ()
    test_metrics: BinaryMetrics | None = None
    test_model_metrics: Mapping[str, BinaryMetrics] | None = None
    test_cross_condition_transfer: CrossConditionTransferSummary | None = None
    test_relative_h5_log_loss_improvement: float | None = None
    test_relative_h6_log_loss_improvement: float | None = None

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "NullSelectionResult":
        return _null_result_from_record(record)

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _freeze_mapping(self.config))
        object.__setattr__(self, "test_transform", _freeze_mapping(self.test_transform))
        object.__setattr__(
            self,
            "test_source_identities",
            _canonical_source_identities(
                self.test_source_identities, field_name="null test source identities"
            ),
        )
        object.__setattr__(self, "test_ids", tuple(self.test_ids))
        object.__setattr__(self, "test_row_sha256s", tuple(self.test_row_sha256s))
        if self.kind not in {
            "label_permutation",
            "layer_order",
            "random_map",
            "output_aligned_11d",
        }:
            raise ValueError("null kind is not registered")
        if self.config_sha256 != _digest(_thaw(self.config)):
            raise ValueError("null config hash does not match config")
        if not isinstance(self.selection, SelectionManifest):
            raise ValueError("null result requires an immutable SelectionManifest")
        provenance = self.selection.null_provenance
        if (
            provenance is None
            or provenance.get("kind") != self.kind
            or provenance.get("seed") != self.seed
        ):
            raise ValueError("null selection provenance does not match the typed null result")
        if type(self.seed) is not int:
            raise ValueError("null seed must be an integer")
        if set(self.test_transform) != {"seed", "row_count"}:
            raise ValueError("null test transform must contain only seed and row_count")
        if self.test_transform.get("seed") != self.seed:
            raise ValueError("null test transform seed does not match null seed")
        if self.test_transform.get("row_count") != len(self.test_source_identities):
            raise ValueError("null test transform row_count does not match frozen identities")
        if type(self.max_norm_error) not in {int, float} or not math.isfinite(
            float(self.max_norm_error)
        ) or self.max_norm_error < 0.0:
            raise ValueError("null max_norm_error must be finite and nonnegative")
        if self.kind in {"layer_order", "random_map"} and self.max_norm_error > 1e-9:
            raise ValueError("registered geometric null must preserve vector norms")
        scored = self.test_metrics is not None
        if scored != (self.test_model_metrics is not None):
            raise ValueError("null test metrics must be present together")
        if scored:
            if not isinstance(self.test_metrics, BinaryMetrics):
                raise ValueError("null test metrics must be a BinaryMetrics record")
            if tuple(sorted(self.test_ids)) != tuple(
                identity.example_id for identity in self.test_source_identities
            ):
                raise ValueError("null test IDs do not match frozen test identities")
            if len(self.test_row_sha256s) != len(self.test_source_identities):
                raise ValueError("null transformed test identities are incomplete")
            if not isinstance(self.test_model_metrics, Mapping) or any(
                not isinstance(value, BinaryMetrics) for value in self.test_model_metrics.values()
            ):
                raise ValueError("null test model metrics must be typed records")
            object.__setattr__(
                self, "test_model_metrics", MappingProxyType(dict(self.test_model_metrics))
            )
            if self.selection.task in {"familiarity", "answerability"}:
                if (
                    not isinstance(
                        self.test_cross_condition_transfer, CrossConditionTransferSummary
                    )
                    or self.test_cross_condition_transfer.task != self.selection.task
                ):
                    raise ValueError(
                        "scored decoding null requires exact reciprocal transfer results"
                    )
            elif self.test_cross_condition_transfer is not None:
                raise ValueError(
                    "unsupported-answer null cannot contain cross-condition transfer"
                )
        elif any(
            value is not None
            for value in (
                self.test_relative_h5_log_loss_improvement,
                self.test_relative_h6_log_loss_improvement,
                self.test_cross_condition_transfer,
            )
        ) or self.test_ids or self.test_row_sha256s:
            raise ValueError("unscored null selection cannot contain test results")

    @property
    def validation_auroc(self) -> float | None:
        model = self.selection.model_for(self.selection.selected_feature_family)
        return model.validation_auroc

    @property
    def relative_h6_log_loss_improvement(self) -> float | None:
        try:
            baseline = self.selection.model_for(NESTED_H5_CANDIDATE)
            candidate = self.selection.model_for(NESTED_H6_CANDIDATE)
        except KeyError:
            return None
        return _relative_improvement(
            candidate.validation_log_loss,
            baseline.validation_log_loss,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "seed": self.seed,
            "config": _thaw(self.config),
            "config_sha256": self.config_sha256,
            "selection_sha256": self.selection.sha256,
            "selection": self.selection.to_record(),
            "max_norm_error": self.max_norm_error,
            "test_source_identities": [
                identity.to_record() for identity in self.test_source_identities
            ],
            "test_transform": _thaw(self.test_transform),
            "test_ids": list(self.test_ids),
            "test_row_sha256s": list(self.test_row_sha256s),
            "test_metrics": None if self.test_metrics is None else self.test_metrics.to_record(),
            "test_model_metrics": None
            if self.test_model_metrics is None
            else {name: value.to_record() for name, value in self.test_model_metrics.items()},
            "test_cross_condition_transfer": (
                None
                if self.test_cross_condition_transfer is None
                else self.test_cross_condition_transfer.to_record()
            ),
            "test_relative_h5_log_loss_improvement": self.test_relative_h5_log_loss_improvement,
            "test_relative_h6_log_loss_improvement": self.test_relative_h6_log_loss_improvement,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    @property
    def test_identities(self) -> tuple[ProbeSourceIdentity, ...]:
        """Compatibility alias for records created before source identities were explicit."""

        return self.test_source_identities


def _frozen_null_test_transform(
    kind: str, seed: int, config: Mapping[str, Any], row_count: int
) -> Mapping[str, Any]:
    del config
    if kind not in {
        "label_permutation",
        "layer_order",
        "random_map",
        "output_aligned_11d",
    }:
        raise ValueError("null kind is not registered")
    return {"seed": seed, "row_count": row_count}


def _transformed_null_test_rows(
    null: NullSelectionResult, rows: Sequence[ProbeRow]
) -> tuple[ProbeRow, ...]:
    ordered = tuple(sorted(rows, key=lambda row: (row.example_id, row.sha256)))
    if _canonical_source_identities(
        tuple(ProbeSourceIdentity.from_row(row) for row in ordered),
        field_name="null probe-test source identities",
    ) != null.test_source_identities:
        raise ValueError("null test rows do not match its frozen protected identities")
    transform = null.test_transform
    if set(transform) != {"seed", "row_count"}:
        raise ValueError("frozen null test transform has an invalid schema")
    seed = transform.get("seed")
    row_count = transform.get("row_count")
    if type(seed) is not int or row_count != len(ordered):
        raise ValueError("frozen null test transform does not match protected rows")
    rng = np.random.default_rng(seed)
    if null.kind == "label_permutation":
        permutation = tuple(int(value) for value in rng.permutation(len(ordered)))
        labels = tuple(row.label for row in ordered)
        return tuple(replace(row, label=labels[index]) for row, index in zip(ordered, permutation))
    if null.kind == "layer_order":
        order = np.asarray(rng.permutation(len(REGISTERED_LAYERS)), dtype=np.int64)
        return tuple(_replace_layer_order(row, order) for row in ordered)
    if null.kind == "random_map":
        return tuple(_random_map_row(row, rng) for row in ordered)
    if null.kind == "output_aligned_11d":
        return tuple(_output_aligned_row(row) for row in ordered)
    raise ValueError("null kind is not registered")


def _score_null_selection_on_test(
    null: NullSelectionResult, rows: Sequence[ProbeRow]
) -> NullSelectionResult:
    transformed = _transformed_null_test_rows(null, rows)
    valid = tuple(row for row in transformed if row.outcome_status == "valid")
    missing = sum(row.outcome_status == "missing" for row in transformed)
    invalid = sum(row.outcome_status == "invalid" for row in transformed)
    classes = _task_classes(null.selection.task)
    model_metrics: dict[str, BinaryMetrics] = {}
    for model in null.selection.models:
        probabilities = model.predict_proba(valid) if valid else np.empty((0, len(classes)))
        key = model.feature_family if model.claim_scope == "pre_output" else f"{model.feature_family}@{model.anchor}"
        model_metrics[key] = compute_classification_metrics(
            tuple(row.label for row in valid),
            probabilities,
            classes=classes,
            threshold=model.threshold,
            total=len(transformed),
            missing=missing,
            invalid=invalid,
        )
    selected = model_metrics[null.selection.selected_feature_family]
    h5 = _relative_improvement(
        _metric_log_loss(model_metrics.get(NESTED_H5_CANDIDATE)),
        _metric_log_loss(model_metrics.get(NESTED_H5_BASELINE)),
    )
    h6 = _relative_improvement(
        _metric_log_loss(model_metrics.get(NESTED_H6_CANDIDATE)),
        _metric_log_loss(model_metrics.get(NESTED_H5_CANDIDATE)),
    )
    cross_condition_transfer = _score_cross_condition_transfer(
        null.selection, transformed
    )
    return replace(
        null,
        test_ids=tuple(sorted(row.example_id for row in transformed)),
        test_row_sha256s=tuple(sorted(row.sha256 for row in transformed)),
        test_metrics=selected,
        test_model_metrics=model_metrics,
        test_cross_condition_transfer=cross_condition_transfer,
        test_relative_h5_log_loss_improvement=h5,
        test_relative_h6_log_loss_improvement=h6,
    )


def _nulls_for(
    nulls: Sequence[NullSelectionResult],
    task: str,
    kind: str,
) -> tuple[NullSelectionResult, ...]:
    return tuple(
        result
        for result in nulls
        if result.selection.task == task and result.kind == kind
    )


def _null_improvement_margin(
    observed: float | None,
    nulls: Sequence[NullSelectionResult],
) -> float | None:
    if observed is None or not nulls:
        return None
    values = tuple(result.test_relative_h6_log_loss_improvement for result in nulls)
    if any(value is None for value in values):
        return None
    return observed - max(float(value) for value in values)


def run_full_selection_nulls(
    train_rows: Sequence[ProbeRow],
    validation_rows: Sequence[ProbeRow],
    *,
    seeds: Sequence[int] = DEFAULT_FULL_SELECTION_NULL_SEEDS,
    estimators: Sequence[Any] | None = None,
    feature_families: Sequence[str] = DEFAULT_FEATURE_FAMILIES,
    pca_options: Sequence[int | None] = PCA_OPTIONS,
    c_options: Sequence[float] = C_OPTIONS,
    protected_test_ids: Collection[str] = (),
    sae_gate: SAEGate | None = None,
    probe_test_source_identities: Sequence[ProbeSourceIdentity] = (),
    _allow_test_seed_override: bool = False,
) -> tuple[NullSelectionResult, ...]:
    """Rerun the complete train/validation selection path for every registered null."""

    train = tuple(train_rows)
    validation = tuple(validation_rows)
    test_identities = _canonical_source_identities(
        probe_test_source_identities,
        field_name="probe_test_source_identities",
    )
    seed_values = tuple(seeds)
    if not seed_values or any(type(seed) is not int for seed in seed_values):
        raise ValueError("null seeds must be a nonempty integer sequence")
    seed_hash = hashlib.sha256(
        json.dumps(list(seed_values), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not _allow_test_seed_override and (
        seed_values != DEFAULT_FULL_SELECTION_NULL_SEEDS
        or seed_hash != DEFAULT_FULL_SELECTION_NULL_SEED_HASH
    ):
        raise ValueError("registered full-selection null seeds and hash are required")
    results = []
    for seed in seed_values:
        for kind in (
            "label_permutation",
            "layer_order",
            "random_map",
            "output_aligned_11d",
        ):
            transformed_train, transformed_validation, config, max_norm_error = _null_rows(
                train, validation, kind, seed
            )
            effective_families = tuple(feature_families)
            provenance = {
                "kind": kind,
                "seed": seed,
                "config": config,
                "requested_feature_families": list(feature_families),
                "effective_feature_families": list(effective_families),
                "seed_list_sha256": seed_hash,
                "registered_seed_count": len(seed_values),
            }
            selection = fit_selection(
                transformed_train,
                transformed_validation,
                estimators=estimators,
                feature_families=effective_families,
                pca_options=pca_options,
                c_options=c_options,
                protected_test_ids=protected_test_ids,
                sae_gate=None if kind == "output_aligned_11d" else sae_gate,
                seed=seed,
                null_provenance=provenance,
            )
            frozen_config = {
                "kind": kind,
                "seed": seed,
                "transform": config,
                "pca_options": list(pca_options),
                "c_options": list(c_options),
                "feature_families": list(effective_families),
                "seed_list_sha256": seed_hash,
                "registered_seed_count": len(seed_values),
            }
            test_transform = _frozen_null_test_transform(
                kind, seed, config, len(test_identities)
            )
            results.append(
                NullSelectionResult(
                    kind=kind,
                    seed=seed,
                    config=frozen_config,
                    config_sha256=_digest(frozen_config),
                    selection=selection,
                    max_norm_error=max_norm_error,
                    test_source_identities=test_identities,
                    test_transform=test_transform,
                )
            )
    return tuple(results)


def _null_rows(
    train: tuple[ProbeRow, ...],
    validation: tuple[ProbeRow, ...],
    kind: str,
    seed: int,
) -> tuple[tuple[ProbeRow, ...], tuple[ProbeRow, ...], dict[str, Any], float]:
    rng = np.random.default_rng(seed)
    if kind == "label_permutation":
        train_labels = _permuted_selection_labels(train, rng)
        validation_labels = _permuted_selection_labels(validation, rng)
        changed_train = tuple(
            replace(row, label=label.item() if isinstance(label, np.generic) else label)
            for row, label in zip(train, train_labels)
        )
        changed_validation = tuple(
            replace(row, label=label.item() if isinstance(label, np.generic) else label)
            for row, label in zip(validation, validation_labels)
        )
        return (
            changed_train,
            changed_validation,
            {
                "train_permutation": [
                    value.item() if isinstance(value, np.generic) else value
                    for value in train_labels
                ],
                "validation_permutation": [
                    value.item() if isinstance(value, np.generic) else value
                    for value in validation_labels
                ],
            },
            0.0,
        )
    if kind == "layer_order":
        order = rng.permutation(26)
        changed = tuple(_replace_layer_order(row, order) for row in (*train, *validation))
        return (
            changed[: len(train)],
            changed[len(train) :],
            {"layer_order": [int(value) for value in order]},
            _max_layer_norm_multiset_error((*train, *validation), changed),
        )
    if kind == "random_map":
        changed = tuple(_random_map_row(row, rng) for row in (*train, *validation))
        return (
            changed[: len(train)],
            changed[len(train) :],
            {"map": "rowwise signed permutation preserving each vector norm"},
            _max_norm_error((*train, *validation), changed),
        )
    if kind == "output_aligned_11d":
        if any(len(row.output_margin_features) != 11 for row in (*train, *validation)):
            raise ValueError("output-aligned control requires exact 11D output margins")
        changed = tuple(_output_aligned_row(row) for row in (*train, *validation))
        return (
            changed[: len(train)],
            changed[len(train) :],
            {"dimensions": 11, "projection": "registered output-margin subspace"},
            0.0,
        )
    raise ValueError("null kind is not registered")


def _permuted_selection_labels(
    rows: tuple[ProbeRow, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.asarray([row.label for row in rows], dtype=object)
    tasks = {row.task for row in rows}
    if len(tasks) != 1:
        raise ValueError("null-selection rows must contain exactly one task")
    task = next(iter(tasks))
    rotation_specs = _registered_rotation_specs(task)
    if not rotation_specs:
        return rng.permutation(labels)

    permuted = labels.copy()
    for train_condition, _ in rotation_specs:
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if _transfer_condition(row, task) == train_condition
            ],
            dtype=np.int64,
        )
        if indices.size == 0:
            raise ValueError(
                f"null-selection rows are missing registered condition {train_condition!r}"
            )
        permuted[indices] = rng.permutation(labels[indices])
    return permuted


def _replace_layer_order(row: ProbeRow, order: np.ndarray) -> ProbeRow:
    residual = row.residual_features[:, order, :]
    sae = None if row.sae_features is None else row.sae_features[:, order, :]
    return replace(row, residual_features=residual, sae_features=sae)


def _random_map_row(row: ProbeRow, rng: np.random.Generator) -> ProbeRow:
    def transform(array: np.ndarray) -> np.ndarray:
        result = np.empty_like(array)
        for anchor in range(array.shape[0]):
            for layer in range(array.shape[1]):
                order = rng.permutation(array.shape[2])
                signs = rng.choice((-1.0, 1.0), size=array.shape[2])
                result[anchor, layer] = array[anchor, layer, order] * signs
        return result

    return replace(
        row,
        residual_features=transform(row.residual_features),
        sae_features=None if row.sae_features is None else transform(row.sae_features),
    )


def _output_aligned_row(row: ProbeRow) -> ProbeRow:
    output = np.asarray(row.output_margin_features, dtype=np.float64)
    residual = np.broadcast_to(output, (len(REGISTERED_ANCHORS), 26, 11)).copy()
    return replace(row, residual_features=residual, sae_features=None)


def _max_norm_error(original: Sequence[ProbeRow], changed: Sequence[ProbeRow]) -> float:
    errors = []
    for before, after in zip(original, changed):
        errors.append(
            float(
                np.max(
                    np.abs(
                        np.linalg.norm(before.residual_features, axis=2)
                        - np.linalg.norm(after.residual_features, axis=2)
                    )
                )
            )
        )
    return max(errors, default=0.0)


def _max_layer_norm_multiset_error(
    original: Sequence[ProbeRow], changed: Sequence[ProbeRow]
) -> float:
    errors = []
    for before, after in zip(original, changed):
        before_norms = np.sort(np.linalg.norm(before.residual_features, axis=2), axis=1)
        after_norms = np.sort(np.linalg.norm(after.residual_features, axis=2), axis=1)
        errors.append(float(np.max(np.abs(before_norms - after_norms))))
    return max(errors, default=0.0)
