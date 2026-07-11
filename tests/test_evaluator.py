import math

import numpy as np
import pytest

from trajectory_extractor import TrajectoryEvaluator


def test_evaluate_step_returns_normalized_delta_velocity():
    evaluator = TrajectoryEvaluator(hidden_dim=3)
    previous = np.array([0.0, 0.0, 0.0])
    current = np.array([3.0, 4.0, 0.0])
    next_layer = np.array([6.0, 8.0, 0.0])

    score = evaluator.evaluate_step(previous, current, next_layer)

    assert score == pytest.approx(1.0)


def test_evaluate_step_is_stable_for_zero_current_vector():
    evaluator = TrajectoryEvaluator(hidden_dim=2, epsilon=1e-8)
    previous = np.array([0.0, 0.0])
    current = np.array([0.0, 0.0])
    next_layer = np.array([0.0, 0.0])

    score = evaluator.evaluate_step(previous, current, next_layer)

    assert score == 0.0
    assert math.isfinite(score)


def test_evaluate_step_rejects_mismatched_vector_shapes():
    evaluator = TrajectoryEvaluator(hidden_dim=3)

    with pytest.raises(ValueError, match="hidden_dim=3"):
        evaluator.evaluate_step(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
        )


def test_evaluate_step_rejects_non_finite_values():
    evaluator = TrajectoryEvaluator(hidden_dim=2)

    with pytest.raises(ValueError, match="finite"):
        evaluator.evaluate_step(
            np.array([0.0, 0.0]),
            np.array([1.0, np.nan]),
            np.array([1.0, 2.0]),
        )
