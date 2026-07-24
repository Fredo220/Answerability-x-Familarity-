"""Frozen development-only activation analysis for the FA Qwen pilot."""

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
from trajectory_extractor.fa_data import FAExample, TRAIN_TEMPLATE_FAMILIES
from trajectory_extractor.fa_features import surface_feature_vector


PILOT_LAYER_IDS = (0, 9, 18, 27)
PILOT_ANALYSIS_SEED = 20260723
PILOT_PCA_COMPONENTS = 16
PILOT_PERMUTATION_SEEDS = tuple(range(2026072300, 2026072400))
PILOT_TASK_ANCHORS = {
    "familiarity": "target_intro_end",
    "answerability": "user_prompt_end",
}
PILOT_CONTROL_ANCHOR = "assistant_prefix_end"
PILOT_MODEL_FAMILIES = (
    "majority",
    "surface_morphology",
    "surface_design_oracle",
    "residual_static",
    "morphology_plus_residual",
)
_TASK_CLASSES = {
    "familiarity": ("matched_synthetic", "screened_real"),
    "answerability": ("target_bound", "distractor_bound", "code_absent"),
}
_ANCHOR_INDEX = {name: index for index, name in enumerate(ANCHOR_NAMES)}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class PilotAnalysisRow:
    """One factorial example with provenance-bound selected activations."""

    example_id: str
    entity_unit_id: str
    template_family: str
    target_familiarity: str
    answerability: str
    surface_features: tuple[float, ...]
    layer_ids: tuple[int, ...]
    anchor_names: tuple[str, ...]
    activations: np.ndarray

    def __post_init__(self) -> None:
        if not self.example_id or not self.entity_unit_id:
            raise ValueError("pilot analysis rows require example and entity IDs")
        if self.template_family not in TRAIN_TEMPLATE_FAMILIES:
            raise ValueError("pilot template family is invalid")
        if self.target_familiarity not in _TASK_CLASSES["familiarity"]:
            raise ValueError("pilot familiarity label is invalid")
        if self.answerability not in _TASK_CLASSES["answerability"]:
            raise ValueError("pilot answerability label is invalid")
        surface = tuple(float(value) for value in self.surface_features)
        layers = tuple(int(value) for value in self.layer_ids)
        anchors = tuple(self.anchor_names)
        activations = np.array(self.activations, dtype=np.float64, copy=True, order="C")
        if not surface or not np.isfinite(surface).all():
            raise ValueError("pilot surface features must be finite and nonempty")
        if layers != PILOT_LAYER_IDS:
            raise ValueError("pilot analysis requires the frozen layer IDs")
        if anchors != ANCHOR_NAMES:
            raise ValueError("pilot analysis requires the registered anchor order")
        if activations.ndim != 3 or activations.shape[:2] != (
            len(ANCHOR_NAMES),
            len(PILOT_LAYER_IDS),
        ):
            raise ValueError("pilot activations must have shape [3, 4, hidden]")
        if activations.shape[2] < PILOT_PCA_COMPONENTS or not np.isfinite(
            activations
        ).all():
            raise ValueError("pilot activations must support the frozen PCA")
        activations.setflags(write=False)
        object.__setattr__(self, "surface_features", surface)
        object.__setattr__(self, "layer_ids", layers)
        object.__setattr__(self, "anchor_names", anchors)
        object.__setattr__(self, "activations", activations)


@dataclass(frozen=True)
class PilotAnalysisBundle:
    """Observed OOF predictions and aggregate metrics for the frozen pilot."""

    metric_records: tuple[Mapping[str, Any], ...]
    prediction_records: tuple[Mapping[str, Any], ...]
    analysis_sha256: str
    example_count: int
    group_count: int


@dataclass(frozen=True)
class _CandidateResult:
    task: str
    anchor: str
    layer_id: int | None
    model_family: str
    labels: np.ndarray
    probabilities: np.ndarray
    fold_ids: tuple[str, ...]
    metrics: Mapping[str, Any]
    null_aurocs: tuple[float, ...] = ()
    raw_permutation_p: float | None = None
    max_layer_permutation_p: float | None = None
    mean_layer_omnibus_auroc: float | None = None
    mean_layer_omnibus_p: float | None = None


def build_pilot_analysis_rows(
    examples: Sequence[FAExample],
    activation_records: Sequence[ActivationRecord],
) -> tuple[PilotAnalysisRow, ...]:
    """Bind the exact prompt rows to the exact activation records, then exclude controls."""

    prepared_examples = tuple(examples)
    prepared_activations = tuple(activation_records)
    if not prepared_examples or len(
        {row.example_id for row in prepared_examples}
    ) != len(prepared_examples):
        raise ValueError("pilot analysis requires unique nonempty examples")
    activation_by_id = {row.example_id: row for row in prepared_activations}
    if len(activation_by_id) != len(prepared_activations) or set(
        activation_by_id
    ) != {row.example_id for row in prepared_examples}:
        raise ValueError("pilot activations do not match the exact prompt examples")
    rows = []
    for example in prepared_examples:
        if example.split != "pilot":
            raise ValueError("pilot analysis cannot consume another namespace")
        activation = activation_by_id[example.example_id]
        if (
            activation.anchors.rendered_prompt_sha256
            != hashlib.sha256(activation.anchors.rendered_bytes).hexdigest()
            or activation.anchors.input_ids != tuple(example.rendered_token_ids)
        ):
            raise ValueError("pilot activation prompt provenance is invalid")
        if example.block == "same_string":
            continue
        if example.block != "factorial":
            raise ValueError("pilot analysis encountered an unregistered block")
        rows.append(
            PilotAnalysisRow(
                example_id=example.example_id,
                entity_unit_id=example.entity_unit_id,
                template_family=example.template_family,
                target_familiarity=example.target_familiarity,
                answerability=example.answerability,
                surface_features=surface_feature_vector(example),
                layer_ids=activation.layer_ids,
                anchor_names=activation.anchor_names,
                activations=activation.activations,
            )
        )
    return _validate_factorial_rows(rows)


def analyze_pilot_rows(
    rows: Sequence[PilotAnalysisRow],
    *,
    permutation_seeds: Sequence[int] = PILOT_PERMUTATION_SEEDS,
) -> PilotAnalysisBundle:
    """Run the fixed leave-one-entity-unit-out pilot analysis."""

    prepared = _validate_factorial_rows(rows)
    seeds = tuple(int(value) for value in permutation_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("pilot permutation seeds must be unique and nonempty")
    groups = np.asarray([row.entity_unit_id for row in prepared], dtype=object)
    example_ids = tuple(row.example_id for row in prepared)
    template = np.asarray(
        [
            tuple(
                float(row.template_family == family)
                for family in TRAIN_TEMPLATE_FAMILIES
            )
            for row in prepared
        ],
        dtype=np.float64,
    )
    surface_design_oracle = np.concatenate(
        (
            np.asarray(
                [row.surface_features for row in prepared], dtype=np.float64
            ),
            template,
        ),
        axis=1,
    )
    surface_morphology = np.concatenate(
        (surface_design_oracle[:, :10], template),
        axis=1,
    )
    candidates: list[_CandidateResult] = []

    for task in ("familiarity", "answerability"):
        classes = _TASK_CLASSES[task]
        labels = np.asarray(
            [
                classes.index(
                    row.target_familiarity if task == "familiarity" else row.answerability
                )
                for row in prepared
            ],
            dtype=np.int64,
        )
        null_labels = tuple(
            _permute_within_strata(labels, prepared, task=task, seed=seed)
            for seed in seeds
        )
        for family in (
            "majority",
            "surface_morphology",
            "surface_design_oracle",
        ):
            candidate = _evaluate_candidate(
                task=task,
                anchor="surface_only",
                layer_id=None,
                model_family=family,
                surface=(
                    surface_design_oracle
                    if family == "surface_design_oracle"
                    else surface_morphology
                ),
                residual=None,
                labels=labels,
                groups=groups,
                rows=prepared,
                null_labels=() if family == "majority" else null_labels,
            )
            candidates.append(candidate)

        for anchor in (PILOT_TASK_ANCHORS[task], PILOT_CONTROL_ANCHOR):
            anchor_index = _ANCHOR_INDEX[anchor]
            for layer_index, layer_id in enumerate(PILOT_LAYER_IDS):
                residual = np.stack(
                    [
                        row.activations[anchor_index, layer_index, :]
                        for row in prepared
                    ],
                    axis=0,
                )
                for family in ("residual_static", "morphology_plus_residual"):
                    candidates.append(
                        _evaluate_candidate(
                            task=task,
                            anchor=anchor,
                            layer_id=layer_id,
                            model_family=family,
                            surface=surface_morphology,
                            residual=residual,
                            labels=labels,
                            groups=groups,
                            rows=prepared,
                            null_labels=null_labels,
                        )
                    )

    candidates = _attach_layer_permutation_summaries(candidates)
    metric_records = tuple(
        _metric_record(candidate, len(prepared), len(set(groups)))
        for candidate in candidates
    )
    prediction_records = tuple(
        record
        for candidate in candidates
        for record in _prediction_records(candidate, prepared, example_ids)
    )
    payload = {
        "metrics": metric_records,
        "predictions": prediction_records,
    }
    return PilotAnalysisBundle(
        metric_records=metric_records,
        prediction_records=prediction_records,
        analysis_sha256=_sha256(payload),
        example_count=len(prepared),
        group_count=len(set(groups)),
    )


def _validate_factorial_rows(
    rows: Sequence[PilotAnalysisRow],
) -> tuple[PilotAnalysisRow, ...]:
    prepared = tuple(sorted(rows, key=lambda row: row.example_id))
    if not prepared or any(not isinstance(row, PilotAnalysisRow) for row in prepared):
        raise ValueError("pilot analysis requires PilotAnalysisRow records")
    if len({row.example_id for row in prepared}) != len(prepared):
        raise ValueError("pilot analysis example IDs must be unique")
    groups = sorted({row.entity_unit_id for row in prepared})
    if len(groups) != 8:
        raise ValueError("pilot analysis requires the frozen eight entity units")
    for group in groups:
        group_rows = [row for row in prepared if row.entity_unit_id == group]
        familiarity = Counter(row.target_familiarity for row in group_rows)
        answerability = Counter(row.answerability for row in group_rows)
        if set(familiarity) != set(_TASK_CLASSES["familiarity"]) or len(
            set(familiarity.values())
        ) != 1:
            raise ValueError("pilot familiarity labels must be balanced within entity units")
        if set(answerability) != set(_TASK_CLASSES["answerability"]) or len(
            set(answerability.values())
        ) != 1:
            raise ValueError("pilot answerability labels must be balanced within entity units")
    return prepared


def _evaluate_candidate(
    *,
    task: str,
    anchor: str,
    layer_id: int | None,
    model_family: str,
    surface: np.ndarray,
    residual: np.ndarray | None,
    labels: np.ndarray,
    groups: np.ndarray,
    rows: Sequence[PilotAnalysisRow],
    null_labels: Sequence[np.ndarray],
) -> _CandidateResult:
    fold_features = _prepare_fold_features(
        surface,
        residual,
        groups,
        model_family=model_family,
    )
    probabilities, fold_ids = _fit_oof_probabilities(
        fold_features,
        labels,
        groups,
        class_count=len(_TASK_CLASSES[task]),
        model_family=model_family,
    )
    metrics = _metrics(task, labels, probabilities, rows)
    null_aurocs = tuple(
        _metrics(
            task,
            permuted,
            _fit_oof_probabilities(
                fold_features,
                permuted,
                groups,
                class_count=len(_TASK_CLASSES[task]),
                model_family=model_family,
            )[0],
            rows,
            include_conditions=False,
        )["auroc"]
        for permuted in null_labels
    )
    raw_p = (
        None
        if not null_aurocs
        else (1 + sum(value >= metrics["auroc"] for value in null_aurocs))
        / (len(null_aurocs) + 1)
    )
    return _CandidateResult(
        task=task,
        anchor=anchor,
        layer_id=layer_id,
        model_family=model_family,
        labels=labels,
        probabilities=probabilities,
        fold_ids=fold_ids,
        metrics=metrics,
        null_aurocs=null_aurocs,
        raw_permutation_p=raw_p,
    )


def _prepare_fold_features(
    surface: np.ndarray,
    residual: np.ndarray | None,
    groups: np.ndarray,
    *,
    model_family: str,
) -> tuple[tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray], ...]:
    unique_groups = tuple(sorted(set(groups.tolist())))
    folds = []
    for held_out in unique_groups:
        test_indices = np.flatnonzero(groups == held_out)
        train_indices = np.flatnonzero(groups != held_out)
        if model_family == "majority":
            train_features = test_features = None
        else:
            surface_scaler = StandardScaler()
            train_surface = surface_scaler.fit_transform(surface[train_indices])
            test_surface = surface_scaler.transform(surface[test_indices])
            if model_family in {"surface_morphology", "surface_design_oracle"}:
                train_features, test_features = train_surface, test_surface
            else:
                if residual is None:
                    raise ValueError("residual model requires residual features")
                residual_scaler = StandardScaler()
                train_residual = residual_scaler.fit_transform(residual[train_indices])
                test_residual = residual_scaler.transform(residual[test_indices])
                pca = PCA(
                    n_components=PILOT_PCA_COMPONENTS,
                    svd_solver="randomized",
                    whiten=True,
                    random_state=PILOT_ANALYSIS_SEED,
                )
                train_internal = pca.fit_transform(train_residual)
                test_internal = pca.transform(test_residual)
                if model_family == "residual_static":
                    train_features, test_features = train_internal, test_internal
                elif model_family == "morphology_plus_residual":
                    train_features = np.concatenate(
                        (train_surface, train_internal), axis=1
                    )
                    test_features = np.concatenate(
                        (test_surface, test_internal), axis=1
                    )
                else:
                    raise ValueError("pilot model family is invalid")
        folds.append((train_features, test_features, train_indices, test_indices))
    return tuple(folds)


def _fit_oof_probabilities(
    folds: Sequence[tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]],
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    class_count: int,
    model_family: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    probabilities = np.zeros((len(labels), class_count), dtype=np.float64)
    fold_ids = [""] * len(labels)
    expected_classes = np.arange(class_count)
    for train_features, test_features, train_indices, test_indices in folds:
        held_out = sorted(set(groups[test_indices].tolist()))
        if len(held_out) != 1 or set(groups[train_indices].tolist()) & set(held_out):
            raise ValueError("pilot fold leaks an entity unit")
        train_labels = labels[train_indices]
        if set(train_labels.tolist()) != set(expected_classes.tolist()):
            raise ValueError("pilot training fold is missing a class")
        if model_family == "majority":
            counts = np.bincount(train_labels, minlength=class_count).astype(np.float64)
            fold_probabilities = np.tile(
                counts / counts.sum(), (len(test_indices), 1)
            )
        else:
            if train_features is None or test_features is None:
                raise ValueError("pilot classifier features are missing")
            classifier = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                class_weight=None,
                max_iter=2000,
                random_state=PILOT_ANALYSIS_SEED,
            )
            classifier.fit(train_features, train_labels)
            fold_probabilities = np.zeros(
                (len(test_indices), class_count), dtype=np.float64
            )
            fold_probabilities[:, classifier.classes_.astype(int)] = (
                classifier.predict_proba(test_features)
            )
        probabilities[test_indices] = fold_probabilities
        for index in test_indices:
            fold_ids[int(index)] = held_out[0]
    if any(not value for value in fold_ids) or not np.isfinite(probabilities).all():
        raise ValueError("pilot OOF prediction coverage is incomplete")
    return probabilities, tuple(fold_ids)


def _metrics(
    task: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    rows: Sequence[PilotAnalysisRow],
    *,
    include_conditions: bool = True,
) -> dict[str, Any]:
    predicted = probabilities.argmax(axis=1)
    if task == "familiarity":
        auroc = roc_auc_score(labels, probabilities[:, 1])
        condition_values = tuple(_TASK_CLASSES["answerability"])
        condition_labels = tuple(row.answerability for row in rows)
    else:
        auroc = roc_auc_score(
            labels,
            probabilities,
            labels=np.arange(len(_TASK_CLASSES[task])),
            multi_class="ovr",
            average="macro",
        )
        condition_values = tuple(_TASK_CLASSES["familiarity"])
        condition_labels = tuple(row.target_familiarity for row in rows)
    condition_scores: dict[str, float] = {}
    if include_conditions:
        for condition in condition_values:
            indices = np.asarray(
                [value == condition for value in condition_labels], dtype=bool
            )
            condition_scores[condition] = float(
                balanced_accuracy_score(labels[indices], predicted[indices])
            )
    return {
        "auroc": float(auroc),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "log_loss": float(
            log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))
        ),
        "condition_balanced_accuracy": condition_scores,
        "worst_condition_balanced_accuracy": (
            None if not condition_scores else float(min(condition_scores.values()))
        ),
    }


def _permute_within_strata(
    labels: np.ndarray,
    rows: Sequence[PilotAnalysisRow],
    *,
    task: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permuted = np.array(labels, copy=True)
    strata = [
        (
            row.entity_unit_id,
            row.answerability
            if task == "familiarity"
            else row.target_familiarity,
        )
        for row in rows
    ]
    for stratum in sorted(set(strata)):
        indices = np.asarray(
            [index for index, value in enumerate(strata) if value == stratum],
            dtype=np.int64,
        )
        permuted[indices] = rng.permutation(permuted[indices])
        if Counter(permuted[indices].tolist()) != Counter(labels[indices].tolist()):
            raise ValueError("within-stratum permutation changed class counts")
    return permuted


def _attach_layer_permutation_summaries(
    candidates: Sequence[_CandidateResult],
) -> list[_CandidateResult]:
    prepared = list(candidates)
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for index, candidate in enumerate(prepared):
        if candidate.layer_id is None or not candidate.null_aurocs:
            continue
        grouped.setdefault(
            (candidate.task, candidate.anchor, candidate.model_family), []
        ).append(index)
    for indices in grouped.values():
        if {prepared[index].layer_id for index in indices} != set(PILOT_LAYER_IDS):
            raise ValueError("max-layer null requires all frozen pilot layers")
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
            observed = prepared[index].metrics["auroc"]
            adjusted = (1 + int(np.sum(maxima >= observed))) / (len(maxima) + 1)
            prepared[index] = replace(
                prepared[index],
                max_layer_permutation_p=float(adjusted),
                mean_layer_omnibus_auroc=observed_mean,
                mean_layer_omnibus_p=float(mean_p),
            )
    return prepared


def _metric_record(
    candidate: _CandidateResult, example_count: int, group_count: int
) -> dict[str, Any]:
    return {
        "kind": "pilot_metrics",
        "schema_version": 1,
        "task": candidate.task,
        "anchor": candidate.anchor,
        "layer_id": candidate.layer_id,
        "model_family": candidate.model_family,
        "example_count": example_count,
        "group_count": group_count,
        **dict(candidate.metrics),
        "permutation_count": len(candidate.null_aurocs),
        "permutation_auroc_mean": (
            None
            if not candidate.null_aurocs
            else float(np.mean(candidate.null_aurocs))
        ),
        "permutation_auroc_q95": (
            None
            if not candidate.null_aurocs
            else float(np.quantile(candidate.null_aurocs, 0.95))
        ),
        "permutation_aurocs": list(candidate.null_aurocs),
        "permutation_p_raw": candidate.raw_permutation_p,
        "permutation_p_max_layer": candidate.max_layer_permutation_p,
        "mean_layer_omnibus_auroc": candidate.mean_layer_omnibus_auroc,
        "mean_layer_omnibus_p": candidate.mean_layer_omnibus_p,
        "claim_scope": "development_only_model_specific_decodability",
    }


def _prediction_records(
    candidate: _CandidateResult,
    rows: Sequence[PilotAnalysisRow],
    example_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    classes = _TASK_CLASSES[candidate.task]
    records = []
    for index, row in enumerate(rows):
        probabilities = candidate.probabilities[index]
        records.append(
            {
                "kind": "pilot_predictions",
                "schema_version": 1,
                "example_id": example_ids[index],
                "entity_unit_id": row.entity_unit_id,
                "held_out_entity_unit_id": candidate.fold_ids[index],
                "task": candidate.task,
                "anchor": candidate.anchor,
                "layer_id": candidate.layer_id,
                "model_family": candidate.model_family,
                "label": classes[int(candidate.labels[index])],
                "predicted_label": classes[int(np.argmax(probabilities))],
                "class_probabilities": {
                    name: float(probabilities[class_index])
                    for class_index, name in enumerate(classes)
                },
            }
        )
    return tuple(records)
