"""Pure, parity-tested advantage functions for the pinned RLMF implementation.

The RLMF branch follows yale-nlp/RLMF commit
``a087e7a1e49f52aaa701add19cd80699b709fdef``,
``src/exp2_rlmf/c_rl_training/rlmf_trainer.py:625-653``.  It deliberately
uses no standard-deviation normalization and never adds the metacognitive
score to the reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class RewardBatch:
    """Pre-weighted rollout rewards arranged as consecutive equal-size groups."""

    other_reward: Tensor
    faith_reward: Tensor
    metacognitive_reward: Tensor
    group_size: int


def standard_grpo_advantage(other_reward: Tensor, faith_reward: Tensor) -> Tensor:
    """Return detached GRPO advantages for one complete rollout group."""
    _validate_rewards(other_reward, faith_reward)
    other_reward = other_reward.detach()
    faith_reward = faith_reward.detach()
    total_reward = _require_finite(other_reward + faith_reward, "total reward")
    total_mean = _require_finite(total_reward.mean(), "total reward mean")
    advantage = _require_finite(total_reward - total_mean, "standard advantage")
    return advantage.detach()


def rlmf_advantage(
    other_reward: Tensor,
    faith_reward: Tensor,
    metacognitive_reward: Tensor,
    *,
    k: float = 1.0,
) -> Tensor:
    """Return detached official piecewise RLMF advantages for one rollout group."""
    _validate_rewards(other_reward, faith_reward, metacognitive_reward)
    if not isinstance(k, Real) or not isfinite(k):
        raise ValueError("k must be a finite real number")

    other_reward = other_reward.detach()
    faith_reward = faith_reward.detach()
    metacognitive_reward = metacognitive_reward.detach()
    try:
        k_tensor = torch.as_tensor(
            k,
            dtype=metacognitive_reward.dtype,
            device=metacognitive_reward.device,
        )
    except (OverflowError, RuntimeError, TypeError) as error:
        raise ValueError("k must be finite and representable in reward dtype") from error
    _require_finite(k_tensor, "k")

    other_mean = _require_finite(other_reward.mean(), "other reward mean")
    other_delta = _require_finite(other_reward - other_mean, "other reward centering")
    faith_mean = _require_finite(faith_reward.mean(), "faith reward mean")
    faith_delta = _require_finite(faith_reward - faith_mean, "faith reward centering")
    scale = _require_finite(
        torch.clamp(k_tensor + metacognitive_reward, min=0.0),
        "metacognitive scale",
    )
    scaled_faith_delta = _require_finite(
        faith_delta * scale,
        "scaled faith advantage",
    )
    faith_advantage = _require_finite(
        torch.where(faith_reward > faith_mean, scaled_faith_delta, faith_delta),
        "faith advantage",
    )
    advantage = _require_finite(
        other_delta + faith_advantage,
        "RLMF advantage",
    )
    return advantage.detach()


def compute_group_advantages(
    batch: RewardBatch, arm: Literal["standard", "rlmf"]
) -> Tensor:
    """Compute detached advantages independently for every complete rollout group."""
    if not isinstance(batch, RewardBatch):
        raise ValueError("batch must be a RewardBatch")
    if arm not in {"standard", "rlmf"}:
        raise ValueError("arm must be 'standard' or 'rlmf'")
    if type(batch.group_size) is not int or batch.group_size < 2:
        raise ValueError("group_size must be an integer of at least two")
    _validate_rewards(
        batch.other_reward,
        batch.faith_reward,
        batch.metacognitive_reward,
    )
    if batch.other_reward.numel() % batch.group_size:
        raise ValueError("reward batch contains an incomplete rollout group")

    other_groups = batch.other_reward.reshape(-1, batch.group_size)
    faith_groups = batch.faith_reward.reshape(-1, batch.group_size)
    meta_groups = batch.metacognitive_reward.reshape(-1, batch.group_size)
    if arm == "standard":
        advantages = [
            standard_grpo_advantage(other_group, faith_group)
            for other_group, faith_group in zip(other_groups, faith_groups, strict=True)
        ]
    else:
        advantages = [
            rlmf_advantage(other_group, faith_group, meta_group)
            for other_group, faith_group, meta_group in zip(
                other_groups, faith_groups, meta_groups, strict=True
            )
        ]
    return torch.cat(advantages).detach()


def _validate_rewards(*rewards: Tensor) -> None:
    first = rewards[0]
    if not isinstance(first, Tensor) or first.ndim != 1 or first.numel() == 0:
        raise ValueError("rewards must be non-empty one-dimensional tensors")
    if not torch.is_floating_point(first):
        raise ValueError("rewards must use floating-point tensors")
    for reward in rewards:
        if (
            not isinstance(reward, Tensor)
            or reward.shape != first.shape
            or reward.dtype != first.dtype
            or reward.device != first.device
        ):
            raise ValueError("rewards must have matching shape, dtype, and device")
        if not torch.isfinite(reward).all().item():
            raise ValueError("rewards must be finite")


def _require_finite(value: Tensor, name: str) -> Tensor:
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must remain finite")
    return value
