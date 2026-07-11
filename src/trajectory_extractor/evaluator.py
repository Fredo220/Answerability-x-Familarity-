from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryEvaluator:
    hidden_dim: int
    epsilon: float = 1e-8

    def evaluate_step(
        self,
        previous_layer_vec: np.ndarray,
        current_layer_vec: np.ndarray,
        next_layer_vec: np.ndarray,
    ) -> float:
        previous = self._as_vector(previous_layer_vec, "previous_layer_vec")
        current = self._as_vector(current_layer_vec, "current_layer_vec")
        next_layer = self._as_vector(next_layer_vec, "next_layer_vec")

        del previous
        delta_vector = next_layer - current
        norm_delta = np.linalg.norm(delta_vector)
        norm_current = np.linalg.norm(current) + self.epsilon
        return float(norm_delta / norm_current)

    def _as_vector(self, value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (self.hidden_dim,):
            raise ValueError(
                f"{name} must have shape ({self.hidden_dim},) for hidden_dim={self.hidden_dim}; "
                f"got {vector.shape}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values.")
        return vector
