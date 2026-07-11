from __future__ import annotations

import numpy as np

from trajectory_extractor.types import TrajectoryBatch


class LayerwiseContrastiveDirection:
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.directions = np.empty((0, 0), dtype=np.float32)
        self.centers = np.empty((0, 0), dtype=np.float32)
        self.fit_example_ids: tuple[str, ...] = ()

    def fit(
        self,
        batch: TrajectoryBatch,
        train_indices: np.ndarray,
    ) -> "LayerwiseContrastiveDirection":
        indices = _validated_train_indices(batch, train_indices)
        labels = batch.labels[indices]
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("contrastive direction requires both classes")

        pooled = _pool_valid_tokens_per_example(batch)
        selected = pooled[indices]
        risk_mean = selected[labels == 1].mean(axis=0)
        control_mean = selected[labels == 0].mean(axis=0)
        raw = risk_mean - control_mean
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        if np.any(norms <= self.epsilon):
            raise ValueError("contrastive direction has a near-zero layer")

        self.directions = (raw / norms).astype(np.float32)
        self.centers = selected.mean(axis=0).astype(np.float32)
        self.fit_example_ids = tuple(batch.example_ids[index] for index in indices)
        return self

    def transform(self, batch: TrajectoryBatch) -> np.ndarray:
        if not self.fit_example_ids:
            raise RuntimeError("contrastive direction must be fitted before transform")
        expected = batch.hidden_states.shape[2:]
        if self.directions.shape != expected or self.centers.shape != expected:
            raise ValueError("batch layer and hidden dimensions do not match fitted direction")
        scores = np.zeros(batch.hidden_states.shape[:3], dtype=np.float32)
        for layer in range(batch.hidden_states.shape[2]):
            current = batch.hidden_states[:, :, layer, :].astype(np.float32)
            current -= self.centers[layer][None, None, :]
            scores[:, :, layer] = np.einsum(
                "ntd,d->nt",
                current,
                self.directions[layer],
                optimize=True,
            )
        scores[~batch.token_mask] = 0.0
        return scores


def _pool_valid_tokens_per_example(batch: TrajectoryBatch) -> np.ndarray:
    counts = batch.token_mask.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("every example needs at least one valid token")
    n_examples, _, n_layers, hidden_dim = batch.hidden_states.shape
    pooled = np.empty((n_examples, n_layers, hidden_dim), dtype=np.float32)
    for layer in range(n_layers):
        current = batch.hidden_states[:, :, layer, :].astype(np.float32)
        current *= batch.token_mask[:, :, None]
        pooled[:, layer, :] = current.sum(axis=1) / counts[:, None]
    return pooled


def _validated_train_indices(batch: TrajectoryBatch, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("train_indices must be a non-empty vector")
    if np.unique(values).size != values.size:
        raise ValueError("train_indices must not contain duplicates")
    if values.min() < 0 or values.max() >= len(batch.example_ids):
        raise ValueError("train_indices are out of range")
    if np.any(batch.splits[values] != "train"):
        raise ValueError("contrastive directions may use the training split only")
    selected_ids = [batch.example_ids[index] for index in values]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("fit example IDs must be unique")
    return values
