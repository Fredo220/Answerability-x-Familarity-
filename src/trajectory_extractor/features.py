from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajectory_extractor.types import TrajectoryBatch


@dataclass(frozen=True)
class RawDynamics:
    velocity: np.ndarray
    curvature: np.ndarray
    direction_change: np.ndarray


def compute_raw_dynamics(hidden_states: np.ndarray, epsilon: float = 1e-8) -> RawDynamics:
    hidden = np.asarray(hidden_states)
    if hidden.ndim != 4 or hidden.shape[2] < 2:
        raise ValueError("hidden_states must have shape [example, token, layer, hidden_dim]")
    prefix = hidden.shape[:2]
    n_layers = hidden.shape[2]
    velocity = np.zeros((*prefix, n_layers - 1), dtype=np.float32)
    curvature = np.zeros((*prefix, max(0, n_layers - 2)), dtype=np.float32)
    direction = np.zeros_like(curvature)
    previous_delta = None
    for layer in range(n_layers - 1):
        current = hidden[:, :, layer, :].astype(np.float32)
        following = hidden[:, :, layer + 1, :].astype(np.float32)
        delta = following - current
        delta_norm = np.linalg.norm(delta, axis=-1)
        velocity[:, :, layer] = delta_norm / (np.linalg.norm(current, axis=-1) + epsilon)
        if previous_delta is not None:
            previous_norm = np.linalg.norm(previous_delta, axis=-1)
            curvature[:, :, layer - 1] = np.linalg.norm(delta - previous_delta, axis=-1) / (
                previous_norm + epsilon
            )
            cosine = np.sum(previous_delta * delta, axis=-1) / (
                previous_norm * delta_norm + epsilon
            )
            direction[:, :, layer - 1] = 1.0 - np.clip(cosine, -1.0, 1.0)
        previous_delta = delta
        del current, following
    return RawDynamics(velocity=velocity, curvature=curvature, direction_change=direction)


def make_method_tensor(
    batch: TrajectoryBatch,
    *,
    static_scores: np.ndarray | None = None,
    operator_residuals: np.ndarray | None = None,
) -> np.ndarray:
    hidden = batch.hidden_states
    n_examples, n_tokens, n_layers = hidden.shape[:3]
    dynamics = compute_raw_dynamics(hidden)
    features: list[np.ndarray] = []
    norm = np.empty(hidden.shape[:3], dtype=np.float32)
    for layer in range(n_layers):
        norm[:, :, layer] = np.linalg.norm(hidden[:, :, layer, :].astype(np.float32), axis=-1)
    features.append(norm[..., None])
    features.append(np.broadcast_to(batch.token_logprobs[:, :, None, None], (n_examples, n_tokens, n_layers, 1)))
    features.append(np.broadcast_to(batch.token_entropies[:, :, None, None], (n_examples, n_tokens, n_layers, 1)))
    features.append(_pad_layer_metric(dynamics.velocity, n_layers)[..., None])
    features.append(_pad_layer_metric(dynamics.curvature, n_layers)[..., None])
    features.append(_pad_layer_metric(dynamics.direction_change, n_layers)[..., None])
    if operator_residuals is not None:
        features.append(_pad_layer_metric(operator_residuals, n_layers)[..., None])
    if static_scores is not None:
        static = np.asarray(static_scores, dtype=np.float32)
        if static.shape == (n_examples, n_layers):
            static = np.broadcast_to(static[:, None, :], (n_examples, n_tokens, n_layers))
        if static.shape != (n_examples, n_tokens, n_layers):
            raise ValueError("static_scores must have shape [example, token, layer]")
        features.append(static[..., None])
    return np.concatenate(features, axis=-1).astype(np.float32)


def _pad_layer_metric(metric: np.ndarray, n_layers: int) -> np.ndarray:
    padded = np.zeros((*metric.shape[:2], n_layers), dtype=np.float32)
    padded[:, :, -metric.shape[2] :] = metric
    return padded
