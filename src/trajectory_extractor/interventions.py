from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch

from trajectory_extractor.steering import mean_difference_direction, normalize_direction


class InterventionArm(StrEnum):
    NONE = "none"
    RANDOM = "norm_matched_random"
    SHUFFLED = "shuffled_label"
    ALWAYS_ON = "iti_always_on"
    TRIGGERED = "dynamics_triggered"


@dataclass(frozen=True)
class InterventionPlan:
    arm: InterventionArm
    direction: torch.Tensor | None
    strength: float
    threshold: float | None


def build_intervention_plans(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    strength: float,
    threshold: float,
    seed: int = 42,
) -> tuple[InterventionPlan, ...]:
    direction = mean_difference_direction(positive, negative)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_direction = normalize_direction(torch.randn(direction.shape, generator=generator))
    joined = torch.cat([positive, negative], dim=0)
    labels = np.array([1] * len(positive) + [0] * len(negative))
    shuffled = np.random.default_rng(seed).permutation(labels)
    shuffled_direction = mean_difference_direction(joined[shuffled == 1], joined[shuffled == 0])
    return (
        InterventionPlan(InterventionArm.NONE, None, 0.0, None),
        InterventionPlan(InterventionArm.RANDOM, random_direction, strength, None),
        InterventionPlan(InterventionArm.SHUFFLED, shuffled_direction, strength, None),
        InterventionPlan(InterventionArm.ALWAYS_ON, direction, strength, None),
        InterventionPlan(InterventionArm.TRIGGERED, direction, strength, threshold),
    )
