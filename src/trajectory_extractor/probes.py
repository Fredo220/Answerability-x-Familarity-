from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trajectory_extractor.types import TrajectoryBatch


class LayerwiseStaticProbe:
    """Train-only layerwise probes evaluated independently at each causal token."""

    def __init__(self) -> None:
        self.scalers: list[StandardScaler] = []
        self.probes: list[LogisticRegression] = []
        self.fit_example_ids: tuple[str, ...] = ()

    def fit(self, batch: TrajectoryBatch, train_indices: np.ndarray) -> "LayerwiseStaticProbe":
        indices = np.asarray(train_indices, dtype=int)
        if indices.size == 0:
            raise ValueError("train_indices must not be empty")
        self.scalers = []
        self.probes = []
        hidden = batch.hidden_states
        labels = np.repeat(batch.labels[indices, None], hidden.shape[1], axis=1)
        for layer in range(hidden.shape[2]):
            samples = hidden[indices, :, layer][batch.token_mask[indices]].astype(np.float32)
            sample_labels = labels[batch.token_mask[indices]]
            scaler = StandardScaler().fit(samples)
            probe = LogisticRegression(max_iter=2000, solver="liblinear", random_state=0)
            probe.fit(scaler.transform(samples), sample_labels)
            self.scalers.append(scaler)
            self.probes.append(probe)
        self.fit_example_ids = tuple(batch.example_ids[index] for index in indices)
        return self

    def predict_scores(self, batch: TrajectoryBatch) -> np.ndarray:
        if not self.probes:
            raise RuntimeError("Probe must be fitted before prediction")
        hidden = batch.hidden_states
        scores = np.zeros(hidden.shape[:3], dtype=np.float32)
        for layer, (scaler, probe) in enumerate(zip(self.scalers, self.probes, strict=True)):
            flattened = hidden[:, :, layer].reshape(-1, hidden.shape[-1]).astype(np.float32)
            scores[:, :, layer] = probe.predict_proba(scaler.transform(flattened))[:, 1].reshape(
                hidden.shape[:2]
            )
        scores[~batch.token_mask] = 0.0
        return scores
