from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionDecision:
    outcome: str
    improvement: float
    confidence_interval: tuple[float, float]


@dataclass(frozen=True)
class InterventionDecision:
    outcome: str
    relative_reduction: float
    control_loss_points: float
    confidence_interval_vs_random: tuple[float, float]


def classify_detection(improvement: float, lower: float, upper: float) -> DetectionDecision:
    if improvement >= 0.03 and lower > 0:
        outcome = "supported"
    elif improvement > 0:
        outcome = "partially_supported"
    else:
        outcome = "not_supported"
    return DetectionDecision(outcome, improvement, (lower, upper))


def classify_intervention(
    relative_reduction: float,
    control_loss_points: float,
    lower_vs_random: float,
    upper_vs_random: float,
) -> InterventionDecision:
    if relative_reduction >= 0.20 and control_loss_points <= 0.05 and lower_vs_random > 0:
        outcome = "supported"
    elif relative_reduction > 0 and control_loss_points <= 0.05:
        outcome = "partially_supported"
    else:
        outcome = "not_supported"
    return InterventionDecision(
        outcome,
        relative_reduction,
        control_loss_points,
        (lower_vs_random, upper_vs_random),
    )
