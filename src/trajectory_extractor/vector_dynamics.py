from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajectory_extractor.types import TrajectoryBatch


@dataclass(frozen=True)
class VectorDynamics:
    static: np.ndarray
    layer_delta: np.ndarray
    token_delta: np.ndarray

    def as_feature_tensor(self, *, include_static: bool = True) -> np.ndarray:
        values = [self.layer_delta, self.token_delta]
        if include_static:
            values.insert(0, self.static)
        return np.stack(values, axis=-1).astype(np.float32)


class StandardizedVectorDynamics:
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.means = np.empty(0, dtype=np.float32)
        self.scales = np.empty(0, dtype=np.float32)
        self.fit_example_ids: tuple[str, ...] = ()

    def fit(
        self,
        batch: TrajectoryBatch,
        scores: np.ndarray,
        train_indices: np.ndarray,
    ) -> "StandardizedVectorDynamics":
        values = _validated_scores(batch, scores)
        indices = _validated_train_indices(batch, train_indices)
        means = np.zeros(values.shape[2], dtype=np.float32)
        scales = np.ones(values.shape[2], dtype=np.float32)
        selected_mask = batch.token_mask[indices]
        if not selected_mask.any(axis=1).all():
            raise ValueError("selected training examples have no valid tokens")
        means64 = np.zeros(values.shape[2], dtype=np.float64)
        scales64 = np.ones(values.shape[2], dtype=np.float64)
        for layer in range(values.shape[2]):
            samples = values[indices, :, layer][selected_mask].astype(np.float64)
            means64[layer] = samples.mean()
            standard_deviation = float(samples.std())
            scales64[layer] = standard_deviation if standard_deviation > self.epsilon else 1.0
        means = _checked_float32(means64, "fitted means")
        scales = _checked_float32(scales64, "fitted scales")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(scales)) or not np.all(scales > 0):
            raise ValueError("fitted means/scales must be finite and scales positive")
        self.means = means
        self.scales = scales
        self.fit_example_ids = tuple(batch.example_ids[index] for index in indices)
        return self

    def transform(self, batch: TrajectoryBatch, scores: np.ndarray) -> VectorDynamics:
        values = _validated_scores(batch, scores)
        if not self.fit_example_ids:
            raise RuntimeError("vector dynamics must be fitted before transform")
        if self.means.shape != (values.shape[2],):
            raise ValueError("score layer count does not match fitted dynamics")
        values64 = values.astype(np.float64)
        static64 = (values64 - self.means[None, None, :]) / self.scales[None, None, :]
        static64[~batch.token_mask] = 0.0
        static = _checked_float32(static64, "standardized state")

        layer_delta64 = np.zeros_like(static64)
        layer_delta64[:, :, 1:] = static64[:, :, 1:] - static64[:, :, :-1]
        layer_delta64[~batch.token_mask] = 0.0
        layer_delta = _checked_float32(layer_delta64, "layer deltas")

        token_delta64 = np.zeros_like(static64)
        valid_pair = batch.token_mask[:, 1:] & batch.token_mask[:, :-1]
        difference = static64[:, 1:, :] - static64[:, :-1, :]
        token_delta64[:, 1:, :] = difference * valid_pair[:, :, None]
        token_delta64[~batch.token_mask] = 0.0
        token_delta = _checked_float32(token_delta64, "token deltas")
        return VectorDynamics(static, layer_delta, token_delta)


def _validated_scores(batch: TrajectoryBatch, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != batch.hidden_states.shape[:3]:
        raise ValueError("scores must have shape [example, token, layer]")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return values


def _checked_float32(values: np.ndarray, name: str) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    float32_max = np.finfo(np.float32).max
    if not np.isfinite(values64).all() or np.any(np.abs(values64) > float32_max):
        raise ValueError(f"{name} is not finite or representable as float32")
    return values64.astype(np.float32)


def _validated_train_indices(batch: TrajectoryBatch, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("train_indices must be a non-empty vector")
    if np.unique(values).size != values.size:
        raise ValueError("train_indices must not contain duplicates")
    if values.min() < 0 or values.max() >= len(batch.example_ids):
        raise ValueError("train_indices are out of range")
    if np.any(batch.splits[values] != "train"):
        raise ValueError("vector standardization may use the training split only")
    return values
