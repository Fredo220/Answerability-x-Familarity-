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
    np.testing.assert_allclose(model.means, np.array([2.0, 3.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(
        model.scales,
        np.full(3, np.sqrt(2.0), dtype=np.float32),
    )
    invalid_tokens = ~batch.token_mask
    assert np.all(result.static[invalid_tokens] == 0.0)
    assert np.all(result.layer_delta[invalid_tokens] == 0.0)
    assert np.all(result.token_delta[invalid_tokens] == 0.0)
    assert np.all(result.token_delta[:, 0] == 0.0)
    assert np.all(result.layer_delta[:, :, 0] == 0.0)
    invalid_pairs = ~(batch.token_mask[:, 1:] & batch.token_mask[:, :-1])
    assert np.all(result.token_delta[:, 1:][invalid_pairs] == 0.0)
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


def test_fit_rejects_selected_training_examples_with_no_valid_tokens():
    batch = make_batch()
    batch.token_mask[1] = False
    scores = np.ones((4, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="no valid tokens"):
        StandardizedVectorDynamics().fit(batch, scores, np.array([0, 1]))


def test_transform_rejects_float32_unrepresentable_standardized_deltas():
    batch = make_batch()
    fit_scores = np.zeros((4, 3, 3), dtype=np.float32)
    model = StandardizedVectorDynamics().fit(batch, fit_scores, np.array([0, 1]))
    scores = fit_scores.copy()
    scores[0, 0] = np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max, 0.0])

    with pytest.raises(ValueError, match="representable as float32"):
        model.transform(batch, scores)
