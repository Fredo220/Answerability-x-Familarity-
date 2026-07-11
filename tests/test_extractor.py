import numpy as np
import pytest

from trajectory_extractor import (
    TrajectoryEvaluator,
    extract_layer_scores,
    run_conditions,
)


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeConfig:
    n_layers = 3
    d_model = 2


class FakeModel:
    cfg = FakeConfig()

    def __init__(self):
        self.prompts_seen = []

    def run_with_cache(self, prompt):
        self.prompts_seen.append(prompt)
        cache = {
            "blocks.0.hook_resid_post": np.array([[[1.0, 0.0], [1.0, 1.0]]]),
            "blocks.1.hook_resid_post": np.array([[[2.0, 0.0], [2.0, 1.0]]]),
            "blocks.2.hook_resid_post": np.array([[[4.0, 0.0], [4.0, 1.0]]]),
        }
        return object(), cache


def test_extract_layer_scores_uses_last_token_residual_stream():
    scores = extract_layer_scores(
        FakeModel(),
        TrajectoryEvaluator(hidden_dim=2),
        "When was Albert Einstein born?",
    )

    assert scores == [0.0, pytest.approx(2.0 / (np.linalg.norm([2.0, 1.0]) + 1e-8))]


def test_run_conditions_returns_scores_for_each_named_prompt():
    model = FakeModel()
    prompts = {
        "Baseline State": "When was Albert Einstein born?",
        "Anomalous State": "Tell me about a fabricated discovery.",
    }

    results = run_conditions(model, prompts, TrajectoryEvaluator(hidden_dim=2))

    assert set(results) == {"Baseline State", "Anomalous State"}
    assert model.prompts_seen == list(prompts.values())
    assert all(len(layer_scores) == 2 for layer_scores in results.values())
