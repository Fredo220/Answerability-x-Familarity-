import numpy as np
import pytest

from trajectory_extractor.types import TrajectoryBatch
from trajectory_extractor.vector_dynamics import StandardizedVectorDynamics


def make_batch() -> TrajectoryBatch:
    hidden = np.zeros((4, 3, 3, 2), dtype=np.float32)
    return TrajectoryBatch(
        example_ids=("a", "b", "c", "d"),
        labels=np.array([0, 1, 0, 1]),
        splits=np.array(["train", "train", "val", "test"]),
        hidden_states=hidden,
        token_mask=np.array(
            [[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
            dtype=bool,
        ),
        token_logprobs=np.zeros((4, 3), dtype=np.float32),
        token_entropies=np.zeros((4, 3), dtype=np.float32),
    )


def test_standardized_dynamics_have_expected_values_and_masking():
    batch = make_batch()
    scores = np.array(
        [
            [[0, 1, 3], [1, 2, 4], [0, 0, 0]],
            [[2, 3, 5], [3, 4, 6], [4, 5, 7]],
            [[1, 2, 4], [2, 3, 5], [3, 4, 6]],
            [[8, 9, 11], [9, 10, 12], [10, 11, 13]],
        ],
        dtype=np.float32,
    )
    model = StandardizedVectorDynamics().fit(batch, scores, np.array([0, 1]))

    result = model.transform(batch, scores)

    assert result.static.shape == (4, 3, 3)
    assert result.layer_delta.shape == (4, 3, 3)
    assert result.token_delta.shape == (4, 3, 3)
    assert result.as_feature_tensor().shape == (4, 3, 3, 3)
    assert np.all(result.static[~batch.token_mask] == 0.0)
    assert np.all(result.token_delta[:, 0] == 0.0)
    assert np.all(result.layer_delta[:, :, 0] == 0.0)
    assert model.fit_example_ids == ("a", "b")


def test_future_tokens_cannot_change_an_earlier_score():
    batch = make_batch()
    scores = np.arange(36, dtype=np.float32).reshape(4, 3, 3)
    model = StandardizedVectorDynamics().fit(batch, scores, np.array([0, 1]))
    original = model.transform(batch, scores)
    changed = scores.copy()
    changed[:, 2, :] += 10_000

    modified = model.transform(batch, changed)

    np.testing.assert_allclose(original.static[:, :2], modified.static[:, :2])
    np.testing.assert_allclose(original.token_delta[:, :2], modified.token_delta[:, :2])


def test_fit_rejects_non_training_examples():
    batch = make_batch()
    scores = np.ones((4, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="training split only"):
        StandardizedVectorDynamics().fit(batch, scores, np.array([0, 2]))
