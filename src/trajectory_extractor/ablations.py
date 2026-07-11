from __future__ import annotations

import numpy as np

from trajectory_extractor.types import TrajectoryBatch


def last_token_only(batch: TrajectoryBatch) -> TrajectoryBatch:
    mask = np.zeros_like(batch.token_mask)
    last = batch.token_mask.sum(axis=1).astype(int) - 1
    if (last < 0).any():
        raise ValueError("Every example must contain a response token")
    mask[np.arange(len(last)), last] = True
    return _replace(batch, token_mask=mask)


def shuffled_layers(batch: TrajectoryBatch, *, seed: int = 42) -> TrajectoryBatch:
    order = np.random.default_rng(seed).permutation(batch.hidden_states.shape[2])
    return _replace(batch, hidden_states=batch.hidden_states[:, :, order, :])


def random_projection(batch: TrajectoryBatch, *, dimensions: int, seed: int = 42) -> TrajectoryBatch:
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    rng = np.random.default_rng(seed)
    source_dim = batch.hidden_states.shape[-1]
    projection = rng.normal(size=(source_dim, dimensions)).astype(np.float32) / np.sqrt(dimensions)
    hidden = np.einsum("ntld,dk->ntlk", batch.hidden_states.astype(np.float32), projection)
    return _replace(batch, hidden_states=hidden.astype(np.float16))


def steering_grid(layers: list[int], strengths: list[float]) -> list[dict[str, float | int]]:
    return [{"layer": layer, "strength": strength} for layer in layers for strength in strengths]


def _replace(batch: TrajectoryBatch, **changes) -> TrajectoryBatch:
    values = {
        "example_ids": batch.example_ids,
        "labels": batch.labels,
        "splits": batch.splits,
        "hidden_states": batch.hidden_states,
        "token_mask": batch.token_mask,
        "token_logprobs": batch.token_logprobs,
        "token_entropies": batch.token_entropies,
        "provenance": batch.provenance,
    }
    values.update(changes)
    return TrajectoryBatch(**values)
