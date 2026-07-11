from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


@dataclass(frozen=True)
class PrefixSurface:
    auroc: np.ndarray
    auprc: np.ndarray


@dataclass(frozen=True)
class BootstrapDelta:
    delta: float
    lower: float
    upper: float
    samples: np.ndarray


@dataclass(frozen=True)
class ThresholdMetrics:
    false_positive_rate: float
    earliest_crossing: np.ndarray
    earliest_token: np.ndarray
    earliest_layer: np.ndarray


@dataclass(frozen=True)
class BootstrapRateDelta:
    delta: float
    lower: float
    upper: float
    samples: np.ndarray


def evaluate_prefix_surface(
    feature_tensor: np.ndarray,
    labels: np.ndarray,
    *,
    token_mask: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> PrefixSurface:
    labels = np.asarray(labels, dtype=int)
    if np.unique(labels[test_indices]).size < 2:
        raise ValueError("prefix evaluation requires both classes in fit and evaluation splits")
    probabilities = fit_predict_prefix_probability_surfaces(
        feature_tensor,
        labels,
        token_mask=token_mask,
        train_indices=train_indices,
        predict_groups=(test_indices,),
    )[0]
    return prefix_surface_from_probabilities(labels[test_indices], probabilities)


def prefix_surface_from_probabilities(
    labels: np.ndarray, probabilities: np.ndarray
) -> PrefixSurface:
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != labels.size:
        raise ValueError("probabilities must have shape [example, token, layer]")
    if np.unique(labels).size < 2:
        raise ValueError("prefix surface requires both evaluation classes")
    n_tokens, n_layers = values.shape[1:3]
    auroc = np.zeros((n_tokens, n_layers), dtype=np.float32)
    auprc = np.zeros_like(auroc)
    for token_end in range(n_tokens):
        for layer_end in range(n_layers):
            scores = values[:, token_end, layer_end]
            auroc[token_end, layer_end] = roc_auc_score(labels, scores)
            auprc[token_end, layer_end] = average_precision_score(labels, scores)
    return PrefixSurface(auroc=auroc, auprc=auprc)


def predict_prefix_probability_surface(
    feature_tensor: np.ndarray,
    labels: np.ndarray,
    *,
    token_mask: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
) -> np.ndarray:
    return fit_predict_prefix_probability_surfaces(
        feature_tensor,
        labels,
        token_mask=token_mask,
        train_indices=train_indices,
        predict_groups=(predict_indices,),
    )[0]


def fit_predict_prefix_probability_surfaces(
    feature_tensor: np.ndarray,
    labels: np.ndarray,
    *,
    token_mask: np.ndarray,
    train_indices: np.ndarray,
    predict_groups: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    """Fit each causal-prefix classifier once and score multiple held-out groups."""
    features = np.asarray(feature_tensor, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    if np.unique(labels[train_indices]).size < 2:
        raise ValueError("prefix prediction requires both classes in the training split")
    n_tokens, n_layers = features.shape[1:3]
    groups = tuple(np.asarray(indices, dtype=int) for indices in predict_groups)
    results = tuple(
        np.zeros((len(indices), n_tokens, n_layers), dtype=np.float32) for indices in groups
    )
    for token_end in range(n_tokens):
        for layer_end in range(n_layers):
            summary = _prefix_summary(features, token_mask, token_end, layer_end)
            classifier = LogisticRegression(max_iter=1000, solver="liblinear", random_state=0)
            classifier.fit(summary[train_indices], labels[train_indices])
            for result, indices in zip(results, groups, strict=True):
                result[:, token_end, layer_end] = classifier.predict_proba(summary[indices])[:, 1]
    return results


def evaluate_and_predict_prefix_surfaces(
    feature_tensor: np.ndarray,
    labels: np.ndarray,
    *,
    token_mask: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[PrefixSurface, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    validation, test = fit_predict_prefix_probability_surfaces(
        feature_tensor,
        labels,
        token_mask=token_mask,
        train_indices=train_indices,
        predict_groups=(validation_indices, test_indices),
    )
    surface = prefix_surface_from_probabilities(labels[validation_indices], validation)
    return surface, validation, test


def predict_at_prefix(
    feature_tensor: np.ndarray,
    labels: np.ndarray,
    *,
    token_mask: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    token_end: int,
    layer_end: int,
) -> np.ndarray:
    if np.unique(np.asarray(labels)[train_indices]).size < 2:
        raise ValueError("prefix prediction requires both classes in the training split")
    summary = _prefix_summary(feature_tensor, token_mask, token_end, layer_end)
    classifier = LogisticRegression(max_iter=1000, solver="liblinear", random_state=0)
    classifier.fit(summary[train_indices], np.asarray(labels)[train_indices])
    return classifier.predict_proba(summary[predict_indices])[:, 1]


def calibration_error(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            selected = (probabilities >= bins[index]) & (probabilities <= bins[index + 1])
        else:
            selected = (probabilities >= bins[index]) & (probabilities < bins[index + 1])
        if selected.any():
            error += selected.mean() * abs(labels[selected].mean() - probabilities[selected].mean())
    return float(error)


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if np.unique(labels).size < 2:
        raise ValueError("threshold selection requires both classes")
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        predicted = probabilities >= threshold
        sensitivity = predicted[labels == 1].mean()
        specificity = (~predicted[labels == 0]).mean()
        score = float((sensitivity + specificity) / 2)
        if score > best[0]:
            best = (score, float(threshold))
    return best[1]


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, *, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if np.unique(labels).size < 2:
        raise ValueError("binary metrics require both classes")
    predicted = probabilities >= threshold
    negatives = labels == 0
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "calibration_error": calibration_error(labels, probabilities),
        "false_positive_rate": float(predicted[negatives].mean()),
        "threshold": float(threshold),
    }


def paired_bootstrap_auc_delta(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> BootstrapDelta:
    labels = np.asarray(labels, dtype=int)
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    observed = roc_auc_score(labels, candidate) - roc_auc_score(labels, baseline)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    attempts = 0
    while len(samples) < n_bootstrap:
        attempts += 1
        if attempts > n_bootstrap * 100:
            raise ValueError("Could not draw enough two-class bootstrap samples")
        indices = _resample_indices(labels.size, rng, groups)
        if np.unique(labels[indices]).size < 2:
            continue
        samples.append(
            roc_auc_score(labels[indices], candidate[indices])
            - roc_auc_score(labels[indices], baseline[indices])
        )
    sample_array = np.asarray(samples, dtype=np.float32)
    return BootstrapDelta(
        delta=float(observed),
        lower=float(np.quantile(sample_array, 0.025)),
        upper=float(np.quantile(sample_array, 0.975)),
        samples=sample_array,
    )


def paired_bootstrap_rate_delta(
    candidate_success: np.ndarray,
    baseline_success: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> BootstrapRateDelta:
    candidate = np.asarray(candidate_success, dtype=float)
    baseline = np.asarray(baseline_success, dtype=float)
    if candidate.shape != baseline.shape or candidate.ndim != 1 or candidate.size == 0:
        raise ValueError("paired success arrays must be aligned and non-empty")
    observed = float(candidate.mean() - baseline.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float32)
    for index in range(n_bootstrap):
        selected = _resample_indices(candidate.size, rng, groups)
        samples[index] = candidate[selected].mean() - baseline[selected].mean()
    return BootstrapRateDelta(
        delta=observed,
        lower=float(np.quantile(samples, 0.025)),
        upper=float(np.quantile(samples, 0.975)),
        samples=samples,
    )


def threshold_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
) -> ThresholdMetrics:
    """Evaluate a causal token x layer score surface at a frozen threshold."""
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if values.ndim != 3 or values.shape[0] != labels.size:
        raise ValueError("scores must have shape [example, token, layer]")
    positive = values >= threshold
    false_positive_rate = float(positive[labels == 0].any(axis=(1, 2)).mean()) if (labels == 0).any() else 0.0
    crossing = np.full(labels.size, -1, dtype=np.int64)
    crossing_token = np.full(labels.size, -1, dtype=np.int64)
    crossing_layer = np.full(labels.size, -1, dtype=np.int64)
    flat = positive.reshape(labels.size, -1)
    for index in range(labels.size):
        found = np.flatnonzero(flat[index])
        if found.size:
            crossing[index] = int(found[0])
            crossing_token[index], crossing_layer[index] = np.unravel_index(
                int(found[0]), values.shape[1:]
            )
    return ThresholdMetrics(false_positive_rate, crossing, crossing_token, crossing_layer)


def _resample_indices(
    size: int,
    rng: np.random.Generator,
    groups: np.ndarray | None,
) -> np.ndarray:
    if groups is None:
        return rng.integers(0, size, size=size)
    group_values = np.asarray(groups)
    if group_values.shape != (size,):
        raise ValueError("bootstrap groups must align with observations")
    unique = np.unique(group_values)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(group_values == value) for value in sampled])


def _prefix_summary(features: np.ndarray, mask: np.ndarray, token_end: int, layer_end: int) -> np.ndarray:
    prefix = features[:, : token_end + 1, : layer_end + 1, :]
    prefix_mask = mask[:, : token_end + 1, None, None]
    valid = np.broadcast_to(prefix_mask, prefix.shape)
    count = valid.sum(axis=(1, 2)).clip(min=1)
    mean = (prefix * valid).sum(axis=(1, 2)) / count
    masked = np.where(valid, prefix, -np.inf)
    maximum = masked.max(axis=(1, 2))
    maximum[~np.isfinite(maximum)] = 0.0
    last = np.zeros((features.shape[0], features.shape[-1]), dtype=features.dtype)
    valid_tokens = mask[:, : token_end + 1]
    for example in range(features.shape[0]):
        found = np.flatnonzero(valid_tokens[example])
        if found.size:
            last[example] = features[example, found[-1], layer_end, :]
    return np.concatenate([mean, maximum, last], axis=1)
