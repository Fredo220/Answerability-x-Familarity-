from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    model_revision: str = "main"
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 42
    max_new_tokens: int = 12
    pca_dims: int = 32
    ridge_alpha: float = 1e-3
    temperature: float = 0.0
    output_dir: str = "runs"

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.pca_dims < 1:
            raise ValueError("pca_dims must be positive")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        if self.temperature != 0.0:
            raise ValueError("The preregistered pipeline requires deterministic temperature=0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class ActivationRun:
    run_id: str
    example_id: str
    track: str
    split: str
    prompt: str
    response: str
    label: int
    input_token_count: int
    response_token_ids: np.ndarray
    hidden_states: np.ndarray
    token_logprobs: np.ndarray
    token_entropies: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [response_token, layer, hidden_dim]")
        token_count = self.hidden_states.shape[0]
        if self.response_token_ids.shape != (token_count,):
            raise ValueError("response_token_ids must align with hidden_states")
        if self.token_logprobs.shape != (token_count,):
            raise ValueError("token_logprobs must align with hidden_states")
        if self.token_entropies.shape != (token_count,):
            raise ValueError("token_entropies must align with hidden_states")
        if not np.isfinite(self.hidden_states).all():
            raise ValueError("hidden_states must be finite")
        start = self.provenance.get("response_token_start")
        end = self.provenance.get("response_token_end")
        if start is not None and end is not None and int(end) - int(start) != token_count:
            raise ValueError("provenance token boundaries must align with response tokens")


@dataclass
class ResponseRun:
    """Generation-only artifact used when activation replay is not part of the endpoint."""

    run_id: str
    example_id: str
    track: str
    split: str
    prompt: str
    response: str
    label: int
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryBatch:
    example_ids: tuple[str, ...]
    labels: np.ndarray
    splits: np.ndarray
    hidden_states: np.ndarray
    token_mask: np.ndarray
    token_logprobs: np.ndarray
    token_entropies: np.ndarray
    provenance: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.hidden_states.ndim != 4:
            raise ValueError("hidden_states must have shape [example, token, layer, hidden_dim]")
        n_examples, n_tokens = self.hidden_states.shape[:2]
        if len(self.example_ids) != n_examples:
            raise ValueError("example_ids must align with hidden_states")
        if self.labels.shape != (n_examples,) or self.splits.shape != (n_examples,):
            raise ValueError("labels and splits must align with examples")
        if self.token_mask.shape != (n_examples, n_tokens):
            raise ValueError("token_mask must align with example and token axes")
        if self.token_logprobs.shape != (n_examples, n_tokens):
            raise ValueError("token_logprobs must align with token_mask")
        if self.token_entropies.shape != (n_examples, n_tokens):
            raise ValueError("token_entropies must align with token_mask")
        if self.provenance and len(self.provenance) != n_examples:
            raise ValueError("provenance must align with examples")
