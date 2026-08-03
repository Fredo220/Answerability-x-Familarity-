"""Registered analysis for the Same-String representation replication v3."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from trajectory_extractor.fa_activations import ANCHOR_NAMES
from trajectory_extractor.fa_same_string_replication_v3 import (
    REP_V3_ANSWERABILITY,
    REP_V3_EXPOSURES,
    REP_V3_SPLIT_COUNTS,
    ReplicationPromptV3,
)


REP_V3_LAYERS = (0, 6, 12, 18, 25)
REP_V3_ANALYSIS_SEED = 20260803
REP_V3_BOOTSTRAP_DRAWS = 10_000
REP_V3_PERMUTATION_COUNT = 999
_ANCHOR_INDEX = {name: index for index, name in enumerate(ANCHOR_NAMES)}
_TASK_LABELS = {
    "answerability": REP_V3_ANSWERABILITY,
    "exposure": REP_V3_EXPOSURES,
}
_TASK_ANCHORS = {
    "answerability": ("user_prompt_end", "target_intro_end"),
    "exposure": ("target_intro_end", "user_prompt_end"),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


@dataclass(frozen=True)
class ReplicationAnalysisRowV3:
    example_id: str
    entity_unit_id: str
    split: str
    template_family: str
    exposure: str
    answerability: str
    target_text: str
    distractor_text: str
    registry_code: str
    neutral_property: str
    user_text: str
    target_intro_span: tuple[int, int]
    activations: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.activations, dtype=np.float64, copy=True, order="C")
        if values.ndim != 3 or values.shape[:2] != (
            len(ANCHOR_NAMES),
            len(REP_V3_LAYERS),
        ):
            raise ValueError("v3 activations must have shape [anchor, layer, hidden]")
        if values.shape[2] < 16 or not np.isfinite(values).all():
            raise ValueError("v3 activations must be finite and support fixed PCA")
        values.setflags(write=False)
        object.__setattr__(self, "target_intro_span", tuple(self.target_intro_span))
        object.__setattr__(self, "activations", values)

    @classmethod
    def from_prompt(
        cls, prompt: ReplicationPromptV3, activations: np.ndarray
    ) -> "ReplicationAnalysisRowV3":
        return cls(
            example_id=prompt.example_id,
            entity_unit_id=prompt.entity_unit_id,
            split=prompt.split,
            template_family=prompt.template_family,
            exposure=prompt.exposure,
            answerability=prompt.answerability,
            target_text=prompt.target_text,
            distractor_text=prompt.distractor_text,
            registry_code=prompt.registry_code,
            neutral_property=prompt.neutral_property,
            user_text=prompt.user_text,
            target_intro_span=prompt.target_intro_span,
            activations=activations,
        )


@dataclass(frozen=True)
class ReplicationAnalysisBundleV3:
    metric_records: tuple[Mapping[str, Any], ...]
    primary_records: tuple[Mapping[str, Any], ...]
    prediction_records: tuple[Mapping[str, Any], ...]
    analysis_sha256: str
    example_count: int
    training_unit_count: int
    test_unit_counts: Mapping[str, int]
    decision: str


@dataclass(frozen=True)
class ReplicationAnalysisPathsV3:
    result: Path
    report: Path
    metrics: Path
    primary: Path
    predictions: Path


@dataclass(frozen=True)
class _SurfaceTransform:
    train: sparse.csr_matrix
    test: sparse.csr_matrix


def build_replication_analysis_rows(
    prompts: Sequence[ReplicationPromptV3], activation_records: Sequence[Any]
) -> tuple[ReplicationAnalysisRowV3, ...]:
    records = {record.example_id: record for record in activation_records}
    if not prompts or len(records) != len(tuple(activation_records)):
        raise ValueError("v3 prompts and activation records must be unique and nonempty")
    if set(records) != {prompt.example_id for prompt in prompts}:
        raise ValueError("v3 activation IDs do not match prompt IDs")
    rows = []
    for prompt in prompts:
        record = records[prompt.example_id]
        if tuple(record.anchors.input_ids) != prompt.rendered_token_ids:
            raise ValueError("v3 activation prompt provenance does not match the corpus")
        if tuple(record.layer_ids) != REP_V3_LAYERS:
            raise ValueError("v3 activation records must use the fixed layers")
        if tuple(record.anchor_names) != ANCHOR_NAMES:
            raise ValueError("v3 activation records must use the registered anchors")
        rows.append(ReplicationAnalysisRowV3.from_prompt(prompt, record.activations))
    return tuple(sorted(rows, key=lambda row: row.example_id))


def _validate_rows(
    rows: Sequence[ReplicationAnalysisRowV3],
) -> tuple[ReplicationAnalysisRowV3, ...]:
    prepared = tuple(sorted(rows, key=lambda row: row.example_id))
    if len(prepared) != 320 or len({row.example_id for row in prepared}) != 320:
        raise ValueError("v3 analysis requires 320 unique examples")
    for split, count in REP_V3_SPLIT_COUNTS.items():
        units = {row.entity_unit_id for row in prepared if row.split == split}
        if len(units) != count:
            raise ValueError("v3 analysis split counts do not match the frozen design")
    expected = Counter(
        (exposure, answerability)
        for exposure in REP_V3_EXPOSURES
        for answerability in REP_V3_ANSWERABILITY
    )
    for unit in {row.entity_unit_id for row in prepared}:
        unit_rows = [row for row in prepared if row.entity_unit_id == unit]
        if Counter((row.exposure, row.answerability) for row in unit_rows) != expected:
            raise ValueError("v3 analysis requires complete 2x2 units")
    return prepared


def _labels(rows: Sequence[ReplicationAnalysisRowV3], task: str) -> np.ndarray:
    classes = _TASK_LABELS[task]
    return np.asarray(
        [classes.index(row.answerability if task == "answerability" else row.exposure) for row in rows],
        dtype=np.int64,
    )


def _numeric_surface(rows: Sequence[ReplicationAnalysisRowV3]) -> np.ndarray:
    return np.asarray(
        [
            (
                len(row.user_text),
                len(row.user_text.split()),
                row.user_text.count(row.target_text),
                row.user_text.count(row.distractor_text),
                row.user_text.count(row.registry_code),
                row.user_text.count(row.neutral_property),
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def _fit_surface(
    train_rows: Sequence[ReplicationAnalysisRowV3],
    test_rows: Sequence[ReplicationAnalysisRowV3],
    anchor: str,
) -> _SurfaceTransform:
    if anchor not in {"target_intro_end", "user_prompt_end"}:
        raise ValueError("surface features require a registered textual anchor")
    train_text = [
        row.user_text[: row.target_intro_span[1]]
        if anchor == "target_intro_end"
        else row.user_text
        for row in train_rows
    ]
    test_text = [
        row.user_text[: row.target_intro_span[1]]
        if anchor == "target_intro_end"
        else row.user_text
        for row in test_rows
    ]
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=4096,
        min_df=2,
        sublinear_tf=True,
        lowercase=False,
    )
    word = TfidfVectorizer(
        analyzer="word",
        tokenizer=str.split,
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=4096,
        min_df=2,
        sublinear_tf=True,
        lowercase=False,
    )
    train_char = char.fit_transform(train_text)
    test_char = char.transform(test_text)
    train_word = word.fit_transform(train_text)
    test_word = word.transform(test_text)
    scaler = StandardScaler()
    def numeric(rows: Sequence[ReplicationAnalysisRowV3], texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [
                (
                    len(text),
                    len(text.split()),
                    text.count(row.target_text),
                    text.count(row.distractor_text),
                    text.count(row.registry_code),
                    text.count(row.neutral_property),
                )
                for row, text in zip(rows, texts, strict=True)
            ],
            dtype=np.float64,
        )

    train_numeric = scaler.fit_transform(numeric(train_rows, train_text))
    test_numeric = scaler.transform(numeric(test_rows, test_text))
    return _SurfaceTransform(
        train=sparse.hstack(
            (train_char, train_word, sparse.csr_matrix(train_numeric)), format="csr"
        ),
        test=sparse.hstack(
            (test_char, test_word, sparse.csr_matrix(test_numeric)), format="csr"
        ),
    )


def _classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight=None,
        max_iter=2000,
        random_state=REP_V3_ANALYSIS_SEED,
    )


def _activation_projection(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    pca = PCA(
        n_components=min(16, train_scaled.shape[0] - 1, train_scaled.shape[1]),
        random_state=REP_V3_ANALYSIS_SEED,
    )
    return pca.fit_transform(train_scaled), pca.transform(test_scaled)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = (probabilities >= 0.5).astype(np.int64)
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration = _classifier().fit(logits, labels)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
    }


def _prediction_rows(
    *,
    rows: Sequence[ReplicationAnalysisRowV3],
    labels: np.ndarray,
    probabilities: np.ndarray,
    task: str,
    test_split: str,
    anchor: str,
    layer_id: int | None,
    family: str,
) -> tuple[dict[str, Any], ...]:
    classes = _TASK_LABELS[task]
    return tuple(
        {
            "kind": "same_string_replication_v3_prediction",
            "schema_version": 1,
            "example_id": row.example_id,
            "entity_unit_id": row.entity_unit_id,
            "test_split": test_split,
            "template_family": row.template_family,
            "task": task,
            "anchor": anchor,
            "layer_id": layer_id,
            "model_family": family,
            "label": classes[int(labels[index])],
            "positive_probability": float(probabilities[index]),
        }
        for index, row in enumerate(rows)
    )


def _metric_record(
    *,
    task: str,
    test_split: str,
    anchor: str,
    layer_id: int | None,
    family: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    role = "diagnostic"
    if task == "answerability" and anchor == "user_prompt_end":
        role = "primary_component"
    elif task == "answerability" and anchor == "target_intro_end":
        role = "temporal_negative_control"
    return {
        "kind": "same_string_replication_v3_metric",
        "schema_version": 1,
        "test_split": test_split,
        "task": task,
        "anchor": anchor,
        "layer_id": layer_id,
        "model_family": family,
        "claim_role": role,
        **_metrics(labels, probabilities),
    }


def _losses(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))


def _primary_summary(
    test_split: str,
    rows: Sequence[ReplicationAnalysisRowV3],
    labels: np.ndarray,
    surface_probabilities: np.ndarray,
    combined_by_layer: Mapping[int, np.ndarray],
    *,
    bootstrap_draws: int,
    permutation_count: int,
) -> dict[str, Any]:
    units = np.asarray([row.entity_unit_id for row in rows], dtype=object)
    unique_units = tuple(sorted(set(units.tolist())))
    surface_loss = _losses(labels, surface_probabilities)
    layer_differences = []
    auroc_differences = []
    for layer in REP_V3_LAYERS:
        combined = combined_by_layer[layer]
        layer_differences.append(surface_loss - _losses(labels, combined))
        auroc_differences.append(
            float(roc_auc_score(labels, combined) - roc_auc_score(labels, surface_probabilities))
        )
    mean_example_difference = np.mean(np.stack(layer_differences), axis=0)
    unit_effects = np.asarray(
        [float(np.mean(mean_example_difference[units == unit])) for unit in unique_units]
    )
    observed = float(np.mean(unit_effects))
    rng = np.random.default_rng(REP_V3_ANALYSIS_SEED + (0 if test_split == "entity_test" else 1))
    bootstrap = np.asarray(
        [float(np.mean(rng.choice(unit_effects, size=len(unit_effects), replace=True))) for _ in range(bootstrap_draws)]
    )
    permutation_values = []
    for _ in range(permutation_count):
        permuted = np.array(labels, copy=True)
        for unit in unique_units:
            for exposure in REP_V3_EXPOSURES:
                indices = np.asarray(
                    [
                        index
                        for index, row in enumerate(rows)
                        if row.entity_unit_id == unit and row.exposure == exposure
                    ],
                    dtype=np.int64,
                )
                permuted[indices] = rng.permutation(permuted[indices])
        surface_permuted = _losses(permuted, surface_probabilities)
        differences = [
            surface_permuted - _losses(permuted, combined_by_layer[layer])
            for layer in REP_V3_LAYERS
        ]
        permutation_values.append(float(np.mean(np.stack(differences))))
    p_value = (1 + sum(value >= observed for value in permutation_values)) / (
        permutation_count + 1
    )
    auroc_improvement = float(np.mean(auroc_differences))
    ci = [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
    supported = observed > 0 and ci[0] > 0 and auroc_improvement >= 0 and p_value <= 0.05
    return {
        "kind": "same_string_replication_v3_primary",
        "schema_version": 1,
        "test_split": test_split,
        "unit_count": len(unique_units),
        "fixed_layers": list(REP_V3_LAYERS),
        "mean_paired_log_loss_improvement": observed,
        "paired_log_loss_improvement_ci95": ci,
        "mean_auroc_improvement": auroc_improvement,
        "permutation_p": float(p_value),
        "permutation_count": permutation_count,
        "bootstrap_draws": bootstrap_draws,
        "supported_on_split": supported,
    }


def analyze_replication_v3(
    rows: Sequence[ReplicationAnalysisRowV3],
    *,
    bootstrap_draws: int = REP_V3_BOOTSTRAP_DRAWS,
    permutation_count: int = REP_V3_PERMUTATION_COUNT,
) -> ReplicationAnalysisBundleV3:
    if bootstrap_draws < 1 or permutation_count < 1:
        raise ValueError("v3 resampling counts must be positive")
    prepared = _validate_rows(rows)
    train_rows = tuple(row for row in prepared if row.split == "representation_train")
    metric_records = []
    prediction_records = []
    primary_records = []
    for test_split in ("entity_test", "template_test"):
        test_rows = tuple(row for row in prepared if row.split == test_split)
        for task in ("answerability", "exposure"):
            train_labels = _labels(train_rows, task)
            test_labels = _labels(test_rows, task)
            primary_combined: dict[int, np.ndarray] = {}
            primary_surface: np.ndarray | None = None
            for anchor in _TASK_ANCHORS[task]:
                surface = _fit_surface(train_rows, test_rows, anchor)
                surface_model = _classifier().fit(surface.train, train_labels)
                surface_probabilities = surface_model.predict_proba(surface.test)[:, 1]
                metric_records.append(
                    _metric_record(
                        task=task,
                        test_split=test_split,
                        anchor=anchor,
                        layer_id=None,
                        family="surface",
                        labels=test_labels,
                        probabilities=surface_probabilities,
                    )
                )
                prediction_records.extend(
                    _prediction_rows(
                        rows=test_rows,
                        labels=test_labels,
                        probabilities=surface_probabilities,
                        task=task,
                        test_split=test_split,
                        anchor=anchor,
                        layer_id=None,
                        family="surface",
                    )
                )
                anchor_index = _ANCHOR_INDEX[anchor]
                for layer_index, layer_id in enumerate(REP_V3_LAYERS):
                    train_activation = np.stack(
                        [row.activations[anchor_index, layer_index] for row in train_rows]
                    )
                    test_activation = np.stack(
                        [row.activations[anchor_index, layer_index] for row in test_rows]
                    )
                    train_projection, test_projection = _activation_projection(
                        train_activation, test_activation
                    )
                    activation_model = _classifier().fit(train_projection, train_labels)
                    activation_probabilities = activation_model.predict_proba(test_projection)[:, 1]
                    combined_train = sparse.hstack(
                        (surface.train, sparse.csr_matrix(train_projection)), format="csr"
                    )
                    combined_test = sparse.hstack(
                        (surface.test, sparse.csr_matrix(test_projection)), format="csr"
                    )
                    combined_model = _classifier().fit(combined_train, train_labels)
                    combined_probabilities = combined_model.predict_proba(combined_test)[:, 1]
                    for family, probabilities in (
                        ("activation", activation_probabilities),
                        ("surface_plus_activation", combined_probabilities),
                    ):
                        metric_records.append(
                            _metric_record(
                                task=task,
                                test_split=test_split,
                                anchor=anchor,
                                layer_id=layer_id,
                                family=family,
                                labels=test_labels,
                                probabilities=probabilities,
                            )
                        )
                        prediction_records.extend(
                            _prediction_rows(
                                rows=test_rows,
                                labels=test_labels,
                                probabilities=probabilities,
                                task=task,
                                test_split=test_split,
                                anchor=anchor,
                                layer_id=layer_id,
                                family=family,
                            )
                        )
                    if task == "answerability" and anchor == "user_prompt_end":
                        primary_combined[layer_id] = combined_probabilities
                        primary_surface = surface_probabilities
            if task == "answerability":
                if primary_surface is None:
                    raise ValueError("v3 primary surface prediction is missing")
                primary_records.append(
                    _primary_summary(
                        test_split,
                        test_rows,
                        test_labels,
                        primary_surface,
                        primary_combined,
                        bootstrap_draws=bootstrap_draws,
                        permutation_count=permutation_count,
                    )
                )
    metrics = tuple(metric_records)
    primary = tuple(primary_records)
    predictions = tuple(prediction_records)
    payload = {"metrics": metrics, "primary": primary, "predictions": predictions}
    decision = "supported" if all(row["supported_on_split"] for row in primary) else "not_supported"
    return ReplicationAnalysisBundleV3(
        metric_records=metrics,
        primary_records=primary,
        prediction_records=predictions,
        analysis_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
        example_count=len(prepared),
        training_unit_count=REP_V3_SPLIT_COUNTS["representation_train"],
        test_unit_counts={split: REP_V3_SPLIT_COUNTS[split] for split in ("entity_test", "template_test")},
        decision=decision,
    )


def simulate_replication_sensitivity(
    *, simulations: int = 2000, seed: int = REP_V3_ANALYSIS_SEED
) -> dict[str, Any]:
    if simulations < 1:
        raise ValueError("sensitivity simulations must be positive")
    rng = np.random.default_rng(seed)
    scenarios = []
    for standard_deviation in (0.05, 0.10):
        for effect in (0.01, 0.02, 0.05, 0.10):
            detected = 0
            for _ in range(simulations):
                sample = rng.normal(effect, standard_deviation, size=20)
                lower = float(np.mean(sample) - 1.96 * np.std(sample, ddof=1) / math.sqrt(20))
                detected += lower > 0
            scenarios.append(
                {
                    "mean_paired_log_loss_improvement": effect,
                    "unit_effect_standard_deviation": standard_deviation,
                    "estimated_detection_probability": detected / simulations,
                }
            )
    return {
        "kind": "same_string_replication_v3_sensitivity",
        "schema_version": 1,
        "seed": seed,
        "simulations": simulations,
        "unit_count_per_test_split": 20,
        "outcomes_opened": False,
        "decision_rule_proxy": "normal_approximation_lower_95_bound_above_zero",
        "scenarios": scenarios,
    }


def write_replication_analysis(
    bundle: ReplicationAnalysisBundleV3,
    destination: str | Path,
    *,
    lineage: Mapping[str, Any],
) -> ReplicationAnalysisPathsV3:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.jsonl"
    primary_path = root / "primary.jsonl"
    predictions_path = root / "predictions.jsonl"
    result_path = root / "result.json"
    report_path = root / "result.md"

    def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
        payload = b"".join(_canonical_json(dict(row)) + b"\n" for row in rows)
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    metrics_hash = write_jsonl(metrics_path, bundle.metric_records)
    primary_hash = write_jsonl(primary_path, bundle.primary_records)
    predictions_hash = write_jsonl(predictions_path, bundle.prediction_records)
    result = {
        "kind": "same_string_replication_v3_result",
        "schema_version": 1,
        "decision": bundle.decision,
        "claim_scope": "model_specific_noncausal_representation_decodability",
        "example_count": bundle.example_count,
        "training_unit_count": bundle.training_unit_count,
        "test_unit_counts": dict(bundle.test_unit_counts),
        "analysis_sha256": bundle.analysis_sha256,
        "metrics_sha256": metrics_hash,
        "primary_sha256": primary_hash,
        "predictions_sha256": predictions_hash,
        "lineage": dict(lineage),
        "primary": list(bundle.primary_records),
        "limitations": [
            "The registered surface comparator is a bag-of-ngrams TF-IDF model, not a symbolic binding parser.",
            "Answerability is explicitly encoded by the prompt sequence; the result concerns representation decodability, not hidden knowledge absent from input.",
            "Template transfer is limited to two held-out template families.",
            "Only one Gemma 2 2B checkpoint was tested.",
            "No causal intervention was performed.",
        ],
    }
    result_path.write_bytes(_canonical_json(result) + b"\n")
    lines = [
        "# Same-String Representation Replication v3",
        "",
        f"**Decision:** `{bundle.decision}`",
        "",
        (
            "This fixed analysis tests whether Gemma 2 2B residual activations add "
            "held-out answerability information beyond registered TF-IDF surface baselines."
        ),
        "",
        "| Test split | Units | Mean paired log-loss improvement | 95% CI | Mean AUROC improvement | Permutation p | Supported |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in bundle.primary_records:
        ci = row["paired_log_loss_improvement_ci95"]
        lines.append(
            f"| `{row['test_split']}` | {row['unit_count']} | "
            f"{row['mean_paired_log_loss_improvement']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{row['mean_auroc_improvement']:.4f} | {row['permutation_p']:.4f} | "
            f"{str(row['supported_on_split']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            (
                "A positive decision supports only model-specific, correlational held-out "
                "decodability on this controlled task. It does not establish causality, "
                "general metacognition, truth detection, or hallucination prevention."
            ),
            "",
            "The registered surface comparator is a bag-of-ngrams TF-IDF model, not a "
            "symbolic binding parser. Answerability is explicitly encoded by the prompt "
            "sequence, so this result concerns internal representation decodability rather "
            "than information absent from the input.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return ReplicationAnalysisPathsV3(
        result=result_path,
        report=report_path,
        metrics=metrics_path,
        primary=primary_path,
        predictions=predictions_path,
    )
