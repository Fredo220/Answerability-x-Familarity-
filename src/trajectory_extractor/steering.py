from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Callable

import numpy as np
import torch


def normalize_direction(direction: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    norm = torch.linalg.vector_norm(direction.float())
    if norm <= epsilon:
        raise ValueError("Steering direction must be non-zero")
    return direction / norm.to(direction.dtype)


def transformer_layers(model) -> torch.nn.ModuleList:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "layers", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise ValueError("Could not locate transformer layers on model")


class SteeringHook(AbstractContextManager):
    def __init__(
        self,
        model,
        *,
        layer_idx: int,
        direction: torch.Tensor,
        strength: float,
        trigger: Callable[[torch.Tensor], bool] | None = None,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.direction = normalize_direction(direction.detach())
        self.strength = float(strength)
        self.trigger = trigger or (lambda _: True)
        self.handle = None

    def __enter__(self) -> "SteeringHook":
        layer = transformer_layers(self.model)[self.layer_idx]

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not self.trigger(hidden):
                return output
            changed = hidden.clone()
            direction = self.direction.to(device=changed.device, dtype=changed.dtype)
            changed[:, -1, :] = changed[:, -1, :] + self.strength * direction
            if isinstance(output, tuple):
                return (changed, *output[1:])
            return changed

        self.handle = layer.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def mean_difference_direction(
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    if positive.shape[1:] != negative.shape[1:]:
        raise ValueError("Positive and negative activations must share feature dimensions")
    return normalize_direction(positive.float().mean(dim=0) - negative.float().mean(dim=0))


class OperatorResidualSteering(AbstractContextManager):
    """Steer only when a fitted adjacent-layer operator exceeds its threshold."""

    def __init__(
        self,
        model,
        *,
        from_layer_idx: int,
        pca_from,
        pca_to,
        operator: np.ndarray,
        threshold: float,
        direction: torch.Tensor,
        strength: float,
        epsilon: float = 1e-8,
    ):
        self.model = model
        self.from_layer_idx = from_layer_idx
        self.pca_from = pca_from
        self.pca_to = pca_to
        self.operator = np.asarray(operator)
        self.threshold = float(threshold)
        self.direction = normalize_direction(direction.detach())
        self.strength = float(strength)
        self.epsilon = epsilon
        self.handles = []
        self.previous: torch.Tensor | None = None
        self.triggered = False
        self.last_score: float | None = None

    def __enter__(self) -> "OperatorResidualSteering":
        layers = transformer_layers(self.model)

        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.previous = hidden[:, -1, :].detach().float().cpu()

        def evaluate_and_steer(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.previous is None:
                return output
            current = hidden[:, -1, :].detach().float().cpu().numpy()
            previous = self.previous.numpy()
            x = self.pca_from.transform(previous)[:, : self.operator.shape[0]]
            y = self.pca_to.transform(current)[:, : self.operator.shape[1]]
            score = np.linalg.norm(y - x @ self.operator) / (np.linalg.norm(y) + self.epsilon)
            self.last_score = float(score)
            if score <= self.threshold:
                return output
            self.triggered = True
            changed = hidden.clone()
            direction = self.direction.to(device=changed.device, dtype=changed.dtype)
            changed[:, -1, :] += self.strength * direction
            if isinstance(output, tuple):
                return (changed, *output[1:])
            return changed

        self.handles = [
            layers[self.from_layer_idx].register_forward_hook(capture),
            layers[self.from_layer_idx + 1].register_forward_hook(evaluate_and_steer),
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.previous = None
