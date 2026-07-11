from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from trajectory_extractor.types import TrajectoryBatch


class LayerwiseOperatorResidual:
    def __init__(self, n_components: int = 32, ridge_alpha: float = 1e-3, epsilon: float = 1e-8):
        self.n_components = n_components
        self.ridge_alpha = ridge_alpha
        self.epsilon = epsilon
        self.pcas: list[PCA] = []
        self.operators: list[np.ndarray] = []
        self.fit_example_ids: tuple[str, ...] = ()

    def fit(self, batch: TrajectoryBatch, train_indices: np.ndarray) -> "LayerwiseOperatorResidual":
        indices = np.asarray(train_indices, dtype=int)
        if indices.size == 0:
            raise ValueError("train_indices must not be empty")
        hidden = batch.hidden_states
        mask = batch.token_mask
        n_layers = hidden.shape[2]
        self.pcas = []
        for layer in range(n_layers):
            samples = hidden[indices, :, layer, :][mask[indices]].astype(np.float32)
            n_components = min(self.n_components, samples.shape[0], samples.shape[1])
            solver = "randomized" if n_components < min(samples.shape) else "full"
            self.pcas.append(PCA(n_components=n_components, svd_solver=solver, random_state=0).fit(samples))
        self.operators = []
        for layer in range(n_layers - 1):
            current = hidden[indices, :, layer, :][mask[indices]].astype(np.float32)
            following = hidden[indices, :, layer + 1, :][mask[indices]].astype(np.float32)
            x = self.pcas[layer].transform(current)
            y = self.pcas[layer + 1].transform(following)
            cross_dim = min(x.shape[1], y.shape[1])
            x = x[:, :cross_dim]
            y = y[:, :cross_dim]
            gram = x.T @ x + self.ridge_alpha * np.eye(cross_dim)
            operator = np.linalg.solve(gram, x.T @ y)
            self.operators.append(operator)
        self.fit_example_ids = tuple(batch.example_ids[index] for index in indices)
        return self

    def transform(self, batch: TrajectoryBatch) -> np.ndarray:
        if not self.operators:
            raise RuntimeError("Operator must be fitted before transform")
        hidden = batch.hidden_states
        n_examples, n_tokens, n_layers = hidden.shape[:3]
        residuals = np.zeros((n_examples, n_tokens, n_layers - 1), dtype=np.float32)
        for layer, operator in enumerate(self.operators):
            current = hidden[:, :, layer, :].reshape(-1, hidden.shape[-1]).astype(np.float32)
            following = hidden[:, :, layer + 1, :].reshape(-1, hidden.shape[-1]).astype(np.float32)
            x = self.pcas[layer].transform(current)[:, : operator.shape[0]]
            y = self.pcas[layer + 1].transform(following)[:, : operator.shape[1]]
            prediction = x @ operator
            score = np.linalg.norm(y - prediction, axis=1) / (np.linalg.norm(y, axis=1) + self.epsilon)
            residuals[:, :, layer] = score.reshape(n_examples, n_tokens)
        residuals[~batch.token_mask] = 0.0
        return residuals
