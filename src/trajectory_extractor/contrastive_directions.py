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

        pooled = _pool_valid_tokens_per_example(batch, indices)
        selected = pooled
        risk_mean = selected[labels == 1].mean(axis=0, dtype=np.float64)
        control_mean = selected[labels == 0].mean(axis=0, dtype=np.float64)
        raw = risk_mean - control_mean
        centers = selected.mean(axis=0, dtype=np.float64)
        if not np.isfinite(raw).all() or not np.isfinite(centers).all():
            raise ValueError("fitted contrastive state must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= self.epsilon):
            raise ValueError("contrastive direction requires a finite positive norm per layer")

        self.directions = (raw / norms).astype(np.float32)
        self.centers = centers.astype(np.float32)
        if not np.isfinite(self.directions).all() or not np.isfinite(self.centers).all():
            raise ValueError("fitted contrastive state must be finite")
        cast_norms = np.linalg.norm(self.directions.astype(np.float64), axis=1)
        if (
            not np.isfinite(cast_norms).all()
            or np.any(cast_norms <= 0.0)
            or not np.allclose(cast_norms, 1.0, rtol=1e-6, atol=1e-6)
        ):
            raise ValueError("normalized float32 directions must remain finite and unit length")
        self.fit_example_ids = tuple(batch.example_ids[index] for index in indices)
        return self

    def transform(self, batch: TrajectoryBatch) -> np.ndarray:
        if not self.fit_example_ids:
            raise RuntimeError("contrastive direction must be fitted before transform")
        if not np.isfinite(self.directions).all() or not np.isfinite(self.centers).all():
            raise ValueError("fitted contrastive state must be finite")
        expected = batch.hidden_states.shape[2:]
        if self.directions.shape != expected or self.centers.shape != expected:
            raise ValueError("batch layer and hidden dimensions do not match fitted direction")
        scores = np.zeros(batch.hidden_states.shape[:3], dtype=np.float32)
        valid = batch.token_mask[..., None]
        for layer in range(batch.hidden_states.shape[2]):
            current = batch.hidden_states[:, :, layer, :].astype(np.float32)
            if not np.isfinite(current[batch.token_mask]).all():
                raise ValueError("finite valid-token activations are required for transform")
            current = np.where(valid, current, 0.0)
            current -= self.centers[layer][None, None, :]
            scores[:, :, layer] = np.einsum(
                "ntd,d->nt",
                current,
                self.directions[layer],
                optimize=True,
            )
        scores = np.where(batch.token_mask[..., None], scores, 0.0)
        if not np.isfinite(scores).all():
            raise ValueError("transform produced non-finite scores")
        return scores


def _pool_valid_tokens_per_example(
    batch: TrajectoryBatch,
    indices: np.ndarray,
) -> np.ndarray:
    token_mask = batch.token_mask[indices]
    counts = token_mask.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("every selected example needs at least one valid token")
    n_examples = indices.size
    n_layers = batch.hidden_states.shape[2]
    hidden_dim = batch.hidden_states.shape[3]
    pooled = np.empty((n_examples, n_layers, hidden_dim), dtype=np.float32)
    valid = token_mask[..., None]
    for layer in range(n_layers):
        current = batch.hidden_states[indices, :, layer, :].astype(np.float32)
        if not np.isfinite(current[token_mask]).all():
            raise ValueError("finite valid-token activations are required for fitting")
        current = np.where(valid, current, 0.0)
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
