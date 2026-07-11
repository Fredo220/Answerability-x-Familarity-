import json

import numpy as np

from trajectory_extractor.artifacts import RunStore
from trajectory_extractor.types import ActivationRun, ExperimentConfig


def make_run(example_id="example-1"):
    return ActivationRun(
        run_id="run-1",
        example_id=example_id,
        track="concept_mixing",
        split="train",
        prompt="Facts...",
        response="Answer",
        label=1,
        input_token_count=3,
        response_token_ids=np.array([4, 5]),
        hidden_states=np.ones((2, 3, 4), dtype=np.float16),
        token_logprobs=np.array([-0.2, -0.3], dtype=np.float32),
        token_entropies=np.array([0.7, 0.8], dtype=np.float32),
        provenance={"model_id": "fake/model", "seed": 7, "binding_error": 0},
    )


def test_experiment_config_round_trips_json(tmp_path):
    config = ExperimentConfig(model_id="fake/model", seed=7, pca_dims=16)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))

    loaded = ExperimentConfig.from_json(path)

    assert loaded == config


def test_run_store_round_trips_activation_run(tmp_path):
    store = RunStore(tmp_path / "runs")
    original = make_run()

    store.write(original)
    loaded = store.read("run-1", "example-1")

    assert loaded.example_id == original.example_id
    assert loaded.provenance == original.provenance
    np.testing.assert_array_equal(loaded.hidden_states, original.hidden_states)
    np.testing.assert_array_equal(loaded.response_token_ids, original.response_token_ids)


def test_run_store_collates_variable_length_runs(tmp_path):
    store = RunStore(tmp_path / "runs")
    first = make_run("first")
    second = make_run("second")
    second.response_token_ids = np.array([9])
    second.hidden_states = np.ones((1, 3, 4), dtype=np.float16) * 2
    second.token_logprobs = np.array([-0.1], dtype=np.float32)
    second.token_entropies = np.array([0.4], dtype=np.float32)
    store.write(first)
    store.write(second)

    batch = store.load_batch("run-1")

    assert batch.hidden_states.shape == (2, 2, 3, 4)
    assert batch.token_mask.tolist() == [[True, True], [True, False]]
    assert batch.example_ids == ("first", "second")
    assert batch.provenance[0]["model_id"] == "fake/model"
    binding_batch = store.load_batch("run-1", label_key="binding_error")
    assert binding_batch.labels.tolist() == [0, 0]
