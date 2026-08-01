"""Exploratory representation-only analysis for the Same-String v2 pilot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from trajectory_extractor.fa_activations import ANCHOR_NAMES, ActivationRecord
from trajectory_extractor.fa_data import FAExample
from trajectory_extractor.fa_features import surface_feature_vector


REPRESENTATION_LAYER_IDS = (0, 6, 12, 18, 25)
REPRESENTATION_ANALYSIS_SEED = 20260802
REPRESENTATION_PERMUTATION_SEEDS = tuple(range(2026080201, 2026080300))
REPRESENTATION_BOOTSTRAP_DRAWS = 2_000
METRIC_RECORD_KIND = "same_string_representation_metrics"
PREDICTION_RECORD_KIND = "same_string_representation_prediction"
_SPLIT_GROUP_COUNTS = {
    "mechanism_train": 12,
    "locked_validation": 4,
    "probe_test": 4,
}
_TASK_CLASSES = {
    "exposure": ("low_exposure", "high_exposure"),
    "answerability": ("code_absent", "target_bound"),
}
_TASK_ANCHORS = {
    "exposure": ("target_intro_end", "user_prompt_end"),
    "answerability": ("user_prompt_end", "target_intro_end"),
}
_ANCHOR_INDEX = {name: index for index, name in enumerate(ANCHOR_NAMES)}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class SameStringRepresentationRow:
    example_id: str
    entity_unit_id: str
    split: str
    template_family: str
    exposure: str
    answerability: str
    surface_features: tuple[float, ...]
    layer_ids: tuple[int, ...]
    anchor_names: tuple[str, ...]
    activations: np.ndarray

    def __post_init__(self) -> None:
        if not self.example_id or not self.entity_unit_id or not self.template_family:
            raise ValueError("representation rows require nonempty identities")
        if self.split not in _SPLIT_GROUP_COUNTS:
            raise ValueError("representation row split is invalid")
        if self.exposure not in _TASK_CLASSES["exposure"]:
            raise ValueError("representation exposure label is invalid")
        if self.answerability not in _TASK_CLASSES["answerability"]:
            raise ValueError("representation answerability label is invalid")
        surface = tuple(float(value) for value in self.surface_features)
        layers = tuple(int(value) for value in self.layer_ids)
        anchors = tuple(self.anchor_names)
        activations = np.array(self.activations, dtype=np.float64, copy=True, order="C")
        if not surface or not np.isfinite(surface).all():
            raise ValueError("representation surface features must be finite")
        if layers != REPRESENTATION_LAYER_IDS:
            raise ValueError("representation analysis requires the fixed layer IDs")
        if anchors != ANCHOR_NAMES:
            raise ValueError("representation analysis requires the registered anchors")
        if activations.ndim != 3 or activations.shape[:2] != (
            len(ANCHOR_NAMES),
            len(REPRESENTATION_LAYER_IDS),
        ):
            raise ValueError("representation activations have an invalid shape")
        if activations.shape[2] < 16 or not np.isfinite(activations).all():
            raise ValueError("representation activations must be finite and support PCA")
        activations.setflags(write=False)
        object.__setattr__(self, "surface_features", surface)
        object.__setattr__(self, "layer_ids", layers)
        object.__setattr__(self, "anchor_names", anchors)
        object.__setattr__(self, "activations", activations)


@dataclass(frozen=True)
class SameStringRepresentationBundle:
    metric_records: tuple[Mapping[str, Any], ...]
    prediction_records: tuple[Mapping[str, Any], ...]
    analysis_sha256: str
    example_count: int
    training_group_count: int
    test_group_count: int


@dataclass(frozen=True)
class _Candidate:
    task: str
    anchor: str
    layer_id: int | None
    model_family: str
    labels: np.ndarray
    probabilities: np.ndarray
    metrics: Mapping[str, Any]
    null_aurocs: tuple[float, ...] = ()
    raw_permutation_p: float | None = None
    max_layer_permutation_p: float | None = None
    mean_layer_auroc: float | None = None
    mean_layer_permutation_p: float | None = None


def _fixed_layer_slice(
    layer_ids: Sequence[int], activations: np.ndarray
) -> tuple[tuple[int, ...], np.ndarray]:
    available = tuple(int(value) for value in layer_ids)
    try:
        indices = tuple(available.index(layer_id) for layer_id in REPRESENTATION_LAYER_IDS)
    except ValueError as error:
        raise ValueError("representation activations omit a fixed layer") from error
    return REPRESENTATION_LAYER_IDS, np.asarray(activations)[:, indices, :]


def build_same_string_representation_rows(
    examples: Sequence[FAExample],
    activation_records: Sequence[ActivationRecord],
) -> tuple[SameStringRepresentationRow, ...]:
    prepared = tuple(examples)
    activations = {row.example_id: row for row in activation_records}
    if not prepared or len(activations) != len(tuple(activation_records)):
        raise ValueError("representation inputs must be unique and nonempty")
    if set(activations) != {row.example_id for row in prepared}:
        raise ValueError("representation activations do not match prompt examples")
    rows = []
    for example in prepared:
        if example.block != "same_string":
            raise ValueError("representation analysis accepts only Same-String rows")
        activation = activations[example.example_id]
        if activation.anchors.input_ids != tuple(example.rendered_token_ids):
            raise ValueError("representation activation prompt provenance is invalid")
        fixed_layers, fixed_activations = _fixed_layer_slice(
            activation.layer_ids, activation.activations
        )
        rows.append(
            SameStringRepresentationRow(
                example_id=example.example_id,
                entity_unit_id=example.entity_unit_id,
                split=example.split,
                template_family=example.template_family,
                exposure=example.exposure,
                answerability=example.answerability,
                surface_features=surface_feature_vector(example)[:10],
                layer_ids=fixed_layers,
                anchor_names=activation.anchor_names,
                activations=fixed_activations,
            )
        )
    return _validate_rows(rows)


def analyze_same_string_representations(
    rows: Sequence[SameStringRepresentationRow],
    *,
    permutation_seeds: Sequence[int] = REPRESENTATION_PERMUTATION_SEEDS,
    bootstrap_seed: int = REPRESENTATION_ANALYSIS_SEED,
    bootstrap_draws: int = REPRESENTATION_BOOTSTRAP_DRAWS,
) -> SameStringRepresentationBundle:
    prepared = _validate_rows(rows)
    seeds = tuple(int(value) for value in permutation_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("representation permutation seeds must be unique")
    if type(bootstrap_draws) is not int or bootstrap_draws <= 0:
        raise ValueError("representation bootstrap draws must be positive")
    train_indices = np.asarray(
        [index for index, row in enumerate(prepared) if row.split != "probe_test"],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [index for index, row in enumerate(prepared) if row.split == "probe_test"],
        dtype=np.int64,
    )
    surface = np.asarray([row.surface_features for row in prepared], dtype=np.float64)
    candidates: list[_Candidate] = []
    for task, classes in _TASK_CLASSES.items():
        labels = np.asarray(
            [
                classes.index(row.exposure if task == "exposure" else row.answerability)
                for row in prepared
            ],
            dtype=np.int64,
        )
        null_labels = tuple(
            _permute_within_units(labels, prepared, task=task, seed=seed)
            for seed in seeds
        )
        candidates.append(
            _evaluate_candidate(
                task,
                "surface_only",
                None,
                "surface_morphology",
                surface,
                labels,
                train_indices,
                test_indices,
                prepared,
                (),
                bootstrap_seed,
                bootstrap_draws,
            )
        )
        for anchor in _TASK_ANCHORS[task]:
            anchor_index = _ANCHOR_INDEX[anchor]
            for layer_index, layer_id in enumerate(REPRESENTATION_LAYER_IDS):
                residual = np.stack(
                    [row.activations[anchor_index, layer_index] for row in prepared]
                )
                for family in ("residual_static", "morphology_plus_residual"):
                    candidates.append(
                        _evaluate_candidate(
                            task,
                            anchor,
                            layer_id,
                            family,
                            surface,
                            labels,
                            train_indices,
                            test_indices,
                            prepared,
                            null_labels,
                            bootstrap_seed,
                            bootstrap_draws,
                            residual=residual,
                        )
                    )
    candidates = _attach_layer_nulls(candidates)
    metrics = tuple(_metric_record(candidate) for candidate in candidates)
    predictions = tuple(
        record
        for candidate in candidates
        for record in _prediction_records(candidate, prepared, test_indices)
    )
    payload = {"metrics": metrics, "predictions": predictions}
    return SameStringRepresentationBundle(
        metric_records=metrics,
        prediction_records=predictions,
        analysis_sha256=_sha256(payload),
        example_count=len(prepared),
        training_group_count=len(
            {prepared[index].entity_unit_id for index in train_indices}
        ),
        test_group_count=len({prepared[index].entity_unit_id for index in test_indices}),
    )


def _validate_rows(
    rows: Sequence[SameStringRepresentationRow],
) -> tuple[SameStringRepresentationRow, ...]:
    prepared = tuple(sorted(rows, key=lambda row: row.example_id))
    if not prepared or any(not isinstance(row, SameStringRepresentationRow) for row in prepared):
        raise ValueError("representation analysis requires valid rows")
    if len({row.example_id for row in prepared}) != len(prepared):
        raise ValueError("representation example IDs must be unique")
    for split, expected in _SPLIT_GROUP_COUNTS.items():
        groups = {row.entity_unit_id for row in prepared if row.split == split}
        if len(groups) != expected:
            raise ValueError("representation split group counts do not match v2")
    for unit in sorted({row.entity_unit_id for row in prepared}):
        unit_rows = [row for row in prepared if row.entity_unit_id == unit]
        cells = Counter((row.exposure, row.answerability) for row in unit_rows)
        if cells != Counter(
            {
                (exposure, answerability): 1
                for exposure in _TASK_CLASSES["exposure"]
                for answerability in _TASK_CLASSES["answerability"]
            }
        ):
            raise ValueError("representation units must contain a complete 2x2")
        if len({row.split for row in unit_rows}) != 1:
            raise ValueError("representation units cannot cross splits")
    return prepared


def _fit_predict(
    surface: np.ndarray,
    residual: np.ndarray | None,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    family: str,
) -> np.ndarray:
    surface_scaler = StandardScaler()
    train_surface = surface_scaler.fit_transform(surface[train_indices])
    test_surface = surface_scaler.transform(surface[test_indices])
    if family == "surface_morphology":
        train_features, test_features = train_surface, test_surface
    else:
        if residual is None:
            raise ValueError("internal representation model requires activations")
        residual_scaler = StandardScaler()
        train_residual = residual_scaler.fit_transform(residual[train_indices])
        test_residual = residual_scaler.transform(residual[test_indices])
        pca = PCA(
            n_components=min(16, len(train_indices) - 1, train_residual.shape[1]),
            svd_solver="randomized",
            whiten=True,
            random_state=REPRESENTATION_ANALYSIS_SEED,
        )
        train_internal = pca.fit_transform(train_residual)
        test_internal = pca.transform(test_residual)
        if family == "residual_static":
            train_features, test_features = train_internal, test_internal
        elif family == "morphology_plus_residual":
            train_features = np.concatenate((train_surface, train_internal), axis=1)
            test_features = np.concatenate((test_surface, test_internal), axis=1)
        else:
            raise ValueError("representation model family is invalid")
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2_000,
        random_state=REPRESENTATION_ANALYSIS_SEED,
    )
    classifier.fit(train_features, labels[train_indices])
    return classifier.predict_proba(test_features)[:, 1]


def _evaluate_candidate(
    task: str,
    anchor: str,
    layer_id: int | None,
    family: str,
    surface: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    rows: Sequence[SameStringRepresentationRow],
    null_labels: Sequence[np.ndarray],
    bootstrap_seed: int,
    bootstrap_draws: int,
    *,
    residual: np.ndarray | None = None,
) -> _Candidate:
    probabilities = _fit_predict(
        surface, residual, labels, train_indices, test_indices, family
    )
    test_labels = labels[test_indices]
    metrics = _metrics(
        task,
        test_labels,
        probabilities,
        [rows[index] for index in test_indices],
        bootstrap_seed=bootstrap_seed,
        bootstrap_draws=bootstrap_draws,
    )
    null_aurocs = tuple(
        float(
            roc_auc_score(
                permuted[test_indices],
                _fit_predict(
                    surface,
                    residual,
                    permuted,
                    train_indices,
                    test_indices,
                    family,
                ),
            )
        )
        for permuted in null_labels
    )
    raw_p = (
        None
        if not null_aurocs
        else (1 + sum(value >= metrics["auroc"] for value in null_aurocs))
        / (len(null_aurocs) + 1)
    )
    return _Candidate(
        task,
        anchor,
        layer_id,
        family,
        test_labels,
        probabilities,
        metrics,
        null_aurocs,
        raw_p,
    )


def _metrics(
    task: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    rows: Sequence[SameStringRepresentationRow],
    *,
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    groups = np.asarray([row.entity_unit_id for row in rows], dtype=object)
    unique_groups = tuple(sorted(set(groups.tolist())))
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = []
    for _ in range(bootstrap_draws):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        bootstrap.append(float(roc_auc_score(labels[indices], probabilities[indices])))
    other = "answerability" if task == "exposure" else "exposure"
    conditions = _TASK_CLASSES[other]
    condition_labels = [
        row.answerability if task == "exposure" else row.exposure for row in rows
    ]
    condition_accuracy = {}
    for condition in conditions:
        indices = np.asarray([value == condition for value in condition_labels])
        condition_accuracy[condition] = float(
            balanced_accuracy_score(labels[indices], predictions[indices])
        )
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auroc_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "condition_balanced_accuracy": condition_accuracy,
        "worst_condition_balanced_accuracy": float(min(condition_accuracy.values())),
    }


def _permute_within_units(
    labels: np.ndarray,
    rows: Sequence[SameStringRepresentationRow],
    *,
    task: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permuted = np.array(labels, copy=True)
    strata = [
        (
            row.entity_unit_id,
            row.answerability if task == "exposure" else row.exposure,
        )
        for row in rows
    ]
    for stratum in sorted(set(strata)):
        indices = np.asarray(
            [index for index, value in enumerate(strata) if value == stratum],
            dtype=np.int64,
        )
        permuted[indices] = rng.permutation(permuted[indices])
    return permuted


def _attach_layer_nulls(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    prepared = list(candidates)
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for index, candidate in enumerate(prepared):
        if candidate.layer_id is not None and candidate.null_aurocs:
            grouped.setdefault(
                (candidate.task, candidate.anchor, candidate.model_family), []
            ).append(index)
    for indices in grouped.values():
        if {prepared[index].layer_id for index in indices} != set(
            REPRESENTATION_LAYER_IDS
        ):
            raise ValueError("representation max-layer null is incomplete")
        null_matrix = np.asarray(
            [prepared[index].null_aurocs for index in indices], dtype=np.float64
        )
        maxima = null_matrix.max(axis=0)
        null_means = null_matrix.mean(axis=0)
        observed_mean = float(
            np.mean([prepared[index].metrics["auroc"] for index in indices])
        )
        mean_p = (1 + int(np.sum(null_means >= observed_mean))) / (
            len(null_means) + 1
        )
        for index in indices:
            adjusted = (
                1
                + int(np.sum(maxima >= prepared[index].metrics["auroc"]))
            ) / (len(maxima) + 1)
            prepared[index] = replace(
                prepared[index],
                max_layer_permutation_p=float(adjusted),
                mean_layer_auroc=observed_mean,
                mean_layer_permutation_p=float(mean_p),
            )
    return prepared


def _metric_record(candidate: _Candidate) -> dict[str, Any]:
    return {
        "kind": METRIC_RECORD_KIND,
        "schema_version": 1,
        "task": candidate.task,
        "anchor": candidate.anchor,
        "layer_id": candidate.layer_id,
        "model_family": candidate.model_family,
        **dict(candidate.metrics),
        "permutation_count": len(candidate.null_aurocs),
        "permutation_auroc_mean": (
            None if not candidate.null_aurocs else float(np.mean(candidate.null_aurocs))
        ),
        "permutation_p_raw": candidate.raw_permutation_p,
        "permutation_p_max_layer": candidate.max_layer_permutation_p,
        "mean_layer_auroc": candidate.mean_layer_auroc,
        "mean_layer_permutation_p": candidate.mean_layer_permutation_p,
        "claim_scope": "exploratory_representation_only",
    }


def _prediction_records(
    candidate: _Candidate,
    rows: Sequence[SameStringRepresentationRow],
    test_indices: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    classes = _TASK_CLASSES[candidate.task]
    records = []
    for local_index, row_index in enumerate(test_indices):
        row = rows[int(row_index)]
        probability = float(candidate.probabilities[local_index])
        records.append(
            {
                "kind": PREDICTION_RECORD_KIND,
                "schema_version": 1,
                "example_id": row.example_id,
                "entity_unit_id": row.entity_unit_id,
                "split": row.split,
                "task": candidate.task,
                "anchor": candidate.anchor,
                "layer_id": candidate.layer_id,
                "model_family": candidate.model_family,
                "label": classes[int(candidate.labels[local_index])],
                "predicted_label": classes[int(probability >= 0.5)],
                "positive_probability": probability,
            }
        )
    return tuple(records)
