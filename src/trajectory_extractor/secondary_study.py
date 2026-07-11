from __future__ import annotations

import numpy as np

from trajectory_extractor.contrastive_directions import LayerwiseContrastiveDirection
from trajectory_extractor.evaluation import (
    binary_metrics,
    evaluate_and_predict_prefix_surfaces,
    paired_bootstrap_auc_delta,
    select_threshold,
    threshold_metrics,
)
from trajectory_extractor.features import make_method_tensor
from trajectory_extractor.operator_residual import LayerwiseOperatorResidual
from trajectory_extractor.types import TrajectoryBatch
from trajectory_extractor.vector_dynamics import StandardizedVectorDynamics


REGISTERED_BASELINE = "contrastive_vector"
REGISTERED_CANDIDATE = "contrastive_plus_dynamics"
MIN_AUROC_GAIN = 0.03


def causal_output_uncertainty(batch: TrajectoryBatch) -> tuple[np.ndarray, np.ndarray]:
    logprobs = np.zeros_like(batch.token_logprobs, dtype=np.float32)
    entropies = np.zeros_like(batch.token_entropies, dtype=np.float32)
    logprobs[:, 1:] = batch.token_logprobs[:, :-1]
    entropies[:, 1:] = batch.token_entropies[:, :-1]
    logprobs[~batch.token_mask] = 0.0
    entropies[~batch.token_mask] = 0.0
    return logprobs, entropies


def evaluate_concept_secondary(
    batch: TrajectoryBatch,
    *,
    pca_dims: int = 32,
    ridge_alpha: float = 1e-3,
    n_bootstrap: int = 2000,
) -> dict:
    train = np.flatnonzero(batch.splits == "train")
    validation = np.flatnonzero(batch.splits == "val")
    test = np.flatnonzero(batch.splits == "test")
    if min(train.size, validation.size, test.size) == 0:
        raise ValueError("secondary evaluation requires train, val, and test splits")
    for name, indices in (("train", train), ("validation", validation), ("test", test)):
        if np.unique(batch.labels[indices]).size < 2:
            raise ValueError(f"{name} split requires both classes")

    direction = LayerwiseContrastiveDirection().fit(batch, train)
    projection_scores = direction.transform(batch)
    dynamics_model = StandardizedVectorDynamics().fit(batch, projection_scores, train)
    vector_dynamics = dynamics_model.transform(batch, projection_scores)

    stable_train = train[batch.labels[train] == 0]
    operator = LayerwiseOperatorResidual(pca_dims, ridge_alpha).fit(batch, stable_train)
    operator_scores = operator.transform(batch)
    causal_batch = _with_causal_output_uncertainty(batch)
    full_features = make_method_tensor(
        causal_batch,
        static_scores=vector_dynamics.static,
        operator_residuals=operator_scores,
    )
    full_features = np.concatenate(
        [
            full_features,
            vector_dynamics.layer_delta[..., None],
            vector_dynamics.token_delta[..., None],
        ],
        axis=-1,
    )
    methods = {
        REGISTERED_BASELINE: vector_dynamics.static[..., None],
        REGISTERED_CANDIDATE: vector_dynamics.as_feature_tensor(include_static=True),
        "full_metacognitive_monitor": full_features.astype(np.float32),
    }

    method_results: dict[str, dict] = {}
    selected_test_probabilities: dict[str, np.ndarray] = {}
    test_probability_surfaces: dict[str, np.ndarray] = {}
    for name, features in methods.items():
        surface, validation_probabilities, test_probabilities = (
            evaluate_and_predict_prefix_surfaces(
                features,
                batch.labels,
                token_mask=batch.token_mask,
                train_indices=train,
                validation_indices=validation,
                test_indices=test,
            )
        )
        selected = np.unravel_index(int(np.nanargmax(surface.auroc)), surface.auroc.shape)
        validation_scores = validation_probabilities[:, selected[0], selected[1]]
        test_scores = test_probabilities[:, selected[0], selected[1]]
        threshold = select_threshold(batch.labels[validation], validation_scores)
        metrics = binary_metrics(batch.labels[test], test_scores, threshold=threshold)
        causal_scores = test_probabilities.copy()
        valid_scores = np.broadcast_to(batch.token_mask[test, :, None], causal_scores.shape)
        causal_scores[~valid_scores] = -np.inf
        crossings = threshold_metrics(causal_scores, batch.labels[test], threshold=threshold)
        positive_crossings = crossings.earliest_crossing[batch.labels[test] == 1]
        observed_crossings = positive_crossings[positive_crossings >= 0]
        positive_tokens = crossings.earliest_token[batch.labels[test] == 1]
        observed_tokens = positive_tokens[positive_tokens >= 0]
        positive_layers = crossings.earliest_layer[batch.labels[test] == 1]
        observed_layers = positive_layers[positive_layers >= 0]
        method_results[name] = {
            "selected_token": int(selected[0]),
            "selected_layer": int(selected[1]),
            "validation_auroc": float(surface.auroc[selected]),
            "validation_auprc": float(surface.auprc[selected]),
            "test": metrics,
            "test_auroc": metrics["auroc"],
            "test_auprc": metrics["auprc"],
            "test_calibration_error": metrics["calibration_error"],
            "test_false_positive_rate": crossings.false_positive_rate,
            "median_positive_crossing": (
                float(np.median(observed_crossings)) if observed_crossings.size else None
            ),
            "median_positive_crossing_token": (
                float(np.median(observed_tokens)) if observed_tokens.size else None
            ),
            "median_positive_crossing_layer": (
                float(np.median(observed_layers)) if observed_layers.size else None
            ),
            "validation_surface": {
                "auroc": surface.auroc.tolist(),
                "auprc": surface.auprc.tolist(),
            },
        }
        selected_test_probabilities[name] = test_scores.astype(np.float32)
        test_probability_surfaces[name] = test_probabilities.astype(np.float32)

    cluster_groups = _bootstrap_groups(batch, test)
    bootstrap = paired_bootstrap_auc_delta(
        batch.labels[test],
        selected_test_probabilities[REGISTERED_CANDIDATE],
        selected_test_probabilities[REGISTERED_BASELINE],
        n_bootstrap=n_bootstrap,
        groups=cluster_groups,
    )
    raw_p = _two_sided_bootstrap_p(bootstrap.samples)
    adjusted_p = float(benjamini_hochberg(np.array([raw_p, 1.0]))[0])
    endpoint = secondary_endpoint_status(batch, test)
    supported = bool(
        endpoint["evaluable"]
        and bootstrap.delta >= MIN_AUROC_GAIN
        and bootstrap.lower > 0.0
        and adjusted_p < 0.05
    )
    claim_status = (
        "not_evaluable"
        if not endpoint["evaluable"]
        else "provisional_supported" if supported else "not_supported"
    )

    return {
        "scientific_name": "metacognitive_internal_reliability_signal",
        "user_facing_metaphor": "artificial_intuition",
        "claim_status": claim_status,
        "methods": method_results,
        "registered_comparison": {
            "candidate": REGISTERED_CANDIDATE,
            "baseline": REGISTERED_BASELINE,
            "delta_auroc": bootstrap.delta,
            "lower": bootstrap.lower,
            "upper": bootstrap.upper,
            "raw_p": raw_p,
            "bh_adjusted_p": adjusted_p,
            "fdr_family": [
                "detection_vector_dynamics",
                "intervention_capping_vs_triggered_pending",
            ],
            "minimum_effect": MIN_AUROC_GAIN,
            "supported": supported,
        },
        "endpoint_status": endpoint,
        "fit_example_ids": {
            "direction": list(direction.fit_example_ids),
            "vector_standardization": list(dynamics_model.fit_example_ids),
            "operator": list(operator.fit_example_ids),
        },
        "operator_reference_class": 0,
        "artifacts": {
            "directions": direction.directions,
            "centers": direction.centers,
            "vector_means": dynamics_model.means,
            "vector_scales": dynamics_model.scales,
            "test_indices": test.astype(np.int64),
            "test_labels": batch.labels[test].astype(np.int64),
            "contrastive_vector_probability": selected_test_probabilities[REGISTERED_BASELINE],
            "metacognitive_risk_probability": selected_test_probabilities[REGISTERED_CANDIDATE],
            "metacognitive_risk_surface": test_probability_surfaces[REGISTERED_CANDIDATE],
            "full_monitor_probability": selected_test_probabilities["full_metacognitive_monitor"],
            "bootstrap_delta_samples": bootstrap.samples,
        },
    }


def secondary_endpoint_status(
    batch: TrajectoryBatch,
    test_indices: np.ndarray,
    *,
    min_positive: int = 20,
    min_clusters: int = 10,
) -> dict:
    indices = np.asarray(test_indices, dtype=int)
    positive = indices[batch.labels[indices] == 1]
    clusters = {
        str(batch.provenance[index].get("entity_family", ""))
        for index in positive
        if batch.provenance and batch.provenance[index].get("entity_family")
    }
    reasons: list[str] = []
    if positive.size < min_positive:
        reasons.append(f"positive_examples<{min_positive}")
    if len(clusters) < min_clusters:
        reasons.append(f"positive_clusters<{min_clusters}")
    return {
        "evaluable": not reasons,
        "positive_examples": int(positive.size),
        "positive_clusters": len(clusters),
        "minimum_positive_examples": min_positive,
        "minimum_positive_clusters": min_clusters,
        "reasons": reasons,
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.empty_like(ranked)
    running = 1.0
    count = len(ranked)
    for position in range(count - 1, -1, -1):
        rank = position + 1
        running = min(running, ranked[position] * count / rank)
        adjusted_ranked[position] = min(1.0, running)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_ranked
    return adjusted


def _two_sided_bootstrap_p(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=float)
    lower = (np.count_nonzero(values <= 0.0) + 1) / (values.size + 1)
    upper = (np.count_nonzero(values >= 0.0) + 1) / (values.size + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def _bootstrap_groups(batch: TrajectoryBatch, indices: np.ndarray) -> np.ndarray:
    groups = []
    for index in indices:
        family = batch.provenance[index].get("entity_family") if batch.provenance else None
        groups.append(str(family) if family else batch.example_ids[index])
    return np.asarray(groups)


def _with_causal_output_uncertainty(batch: TrajectoryBatch) -> TrajectoryBatch:
    logprobs, entropies = causal_output_uncertainty(batch)
    return TrajectoryBatch(
        example_ids=batch.example_ids,
        labels=batch.labels,
        splits=batch.splits,
        hidden_states=batch.hidden_states,
        token_mask=batch.token_mask,
        token_logprobs=logprobs,
        token_entropies=entropies,
        provenance=batch.provenance,
    )
