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
        for layer in range(values.shape[2]):
            samples = values[indices, :, layer][selected_mask]
            means[layer] = samples.mean()
            standard_deviation = float(samples.std())
            scales[layer] = standard_deviation if standard_deviation > self.epsilon else 1.0
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
        static = (values - self.means[None, None, :]) / self.scales[None, None, :]
        static = static.astype(np.float32)
        static[~batch.token_mask] = 0.0

        layer_delta = np.zeros_like(static)
        layer_delta[:, :, 1:] = static[:, :, 1:] - static[:, :, :-1]
        layer_delta[~batch.token_mask] = 0.0

        token_delta = np.zeros_like(static)
        valid_pair = batch.token_mask[:, 1:] & batch.token_mask[:, :-1]
        difference = static[:, 1:, :] - static[:, :-1, :]
        token_delta[:, 1:, :] = difference * valid_pair[:, :, None]
        token_delta[~batch.token_mask] = 0.0
        return VectorDynamics(static, layer_delta, token_delta)


def _validated_scores(batch: TrajectoryBatch, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != batch.hidden_states.shape[:3]:
        raise ValueError("scores must have shape [example, token, layer]")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return values


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
