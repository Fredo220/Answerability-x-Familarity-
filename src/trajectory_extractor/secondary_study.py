from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from trajectory_extractor.contrastive_directions import LayerwiseContrastiveDirection
from trajectory_extractor.evaluation import (
    binary_metrics,
    evaluate_prefix_surface,
    paired_bootstrap_auc_delta,
    predict_at_prefix,
    predict_prefix_probability_surface,
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
    cluster_groups = _test_entity_families(batch, test)

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
    validation_probability_surfaces: dict[str, np.ndarray] = {}
    for name, features in methods.items():
        surface = evaluate_prefix_surface(
            features,
            batch.labels,
            token_mask=batch.token_mask,
            train_indices=train,
            test_indices=validation,
        )
        selected = np.unravel_index(int(np.nanargmax(surface.auroc)), surface.auroc.shape)
        validation_scores = predict_at_prefix(
            features,
            batch.labels,
            token_mask=batch.token_mask,
            train_indices=train,
            predict_indices=validation,
            token_end=int(selected[0]),
            layer_end=int(selected[1]),
        )
        threshold = select_threshold(batch.labels[validation], validation_scores)
        test_scores = predict_at_prefix(
            features,
            batch.labels,
            token_mask=batch.token_mask,
            train_indices=train,
            predict_indices=test,
            token_end=int(selected[0]),
            layer_end=int(selected[1]),
        )
        metrics = binary_metrics(batch.labels[test], test_scores, threshold=threshold)
        validation_surface = predict_prefix_probability_surface(
            features,
            batch.labels,
            token_mask=batch.token_mask,
            train_indices=train,
            predict_indices=validation,
        )
        causal_validation_scores = validation_surface.copy()
        valid_validation_scores = np.broadcast_to(
            batch.token_mask[validation, :, None], causal_validation_scores.shape
        )
        causal_validation_scores[~valid_validation_scores] = -np.inf
        crossings = threshold_metrics(
            causal_validation_scores,
            batch.labels[validation],
            threshold=threshold,
        )
        positive_crossings = crossings.earliest_crossing[batch.labels[validation] == 1]
        observed_crossings = positive_crossings[positive_crossings >= 0]
        positive_tokens = crossings.earliest_token[batch.labels[validation] == 1]
        observed_tokens = positive_tokens[positive_tokens >= 0]
        positive_layers = crossings.earliest_layer[batch.labels[validation] == 1]
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
            "test_false_positive_rate": metrics["false_positive_rate"],
            "validation_surface": {
                "auroc": surface.auroc.tolist(),
                "auprc": surface.auprc.tolist(),
            },
            "validation_diagnostics": {
                "false_positive_rate": crossings.false_positive_rate,
                "median_positive_crossing": (
                    float(np.median(observed_crossings)) if observed_crossings.size else None
                ),
                "median_positive_crossing_token": (
                    float(np.median(observed_tokens)) if observed_tokens.size else None
                ),
                "median_positive_crossing_layer": (
                    float(np.median(observed_layers)) if observed_layers.size else None
                ),
            },
            "metadata": {
                "analysis_role": (
                    "exploratory" if name == "full_metacognitive_monitor" else "confirmatory"
                )
            },
        }
        selected_test_probabilities[name] = test_scores.astype(np.float32)
        validation_probability_surfaces[name] = validation_surface.astype(np.float32)

    bootstrap = paired_bootstrap_auc_delta(
        batch.labels[test],
        selected_test_probabilities[REGISTERED_CANDIDATE],
        selected_test_probabilities[REGISTERED_BASELINE],
        n_bootstrap=n_bootstrap,
        groups=cluster_groups,
    )
    raw_p = paired_entity_family_permutation_p(
        batch.labels[test],
        selected_test_probabilities[REGISTERED_CANDIDATE],
        selected_test_probabilities[REGISTERED_BASELINE],
        groups=cluster_groups,
    )
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
            "p_value_method": "paired_entity_family_permutation",
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
            "validation_indices": validation.astype(np.int64),
            "validation_labels": batch.labels[validation].astype(np.int64),
            "test_indices": test.astype(np.int64),
            "test_labels": batch.labels[test].astype(np.int64),
            "contrastive_vector_probability": selected_test_probabilities[REGISTERED_BASELINE],
            "metacognitive_risk_probability": selected_test_probabilities[REGISTERED_CANDIDATE],
            "validation_metacognitive_risk_surface": validation_probability_surfaces[
                REGISTERED_CANDIDATE
            ],
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


def paired_entity_family_permutation_p(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    groups: np.ndarray,
    n_permutations: int = 2000,
    seed: int = 42,
) -> float:
    """Two-sided paired cluster permutation p-value under method exchangeability."""
    label_values = np.asarray(labels, dtype=int)
    candidate_values = np.asarray(candidate, dtype=float)
    baseline_values = np.asarray(baseline, dtype=float)
    group_values = np.asarray(groups)
    if (
        label_values.ndim != 1
        or candidate_values.shape != label_values.shape
        or baseline_values.shape != label_values.shape
        or group_values.shape != label_values.shape
        or label_values.size == 0
    ):
        raise ValueError("labels, predictions, and groups must be aligned non-empty vectors")
    if np.unique(label_values).size < 2:
        raise ValueError("paired permutation requires both classes")
    if not np.isfinite(candidate_values).all() or not np.isfinite(baseline_values).all():
        raise ValueError("paired permutation predictions must be finite")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")

    unique_groups = np.unique(group_values)
    observed = roc_auc_score(label_values, candidate_values) - roc_auc_score(
        label_values, baseline_values
    )
    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        permuted_candidate = candidate_values.copy()
        permuted_baseline = baseline_values.copy()
        swaps = rng.integers(0, 2, size=unique_groups.size).astype(bool)
        for should_swap, group in zip(swaps, unique_groups, strict=True):
            if should_swap:
                selected = group_values == group
                permuted_candidate[selected], permuted_baseline[selected] = (
                    baseline_values[selected],
                    candidate_values[selected],
                )
        statistic = roc_auc_score(label_values, permuted_candidate) - roc_auc_score(
            label_values, permuted_baseline
        )
        at_least_as_extreme += abs(statistic) >= abs(observed)
    return float((at_least_as_extreme + 1) / (n_permutations + 1))


def _test_entity_families(batch: TrajectoryBatch, indices: np.ndarray) -> np.ndarray:
    if not batch.provenance:
        raise ValueError("every test example requires a non-empty entity_family")
    groups: list[str] = []
    for index in indices:
        family = batch.provenance[index].get("entity_family")
        if not family:
            raise ValueError("every test example requires a non-empty entity_family")
        groups.append(str(family))
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
