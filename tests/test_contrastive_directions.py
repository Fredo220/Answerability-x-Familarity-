import numpy as np
import pytest

from trajectory_extractor.contrastive_directions import LayerwiseContrastiveDirection
from trajectory_extractor.types import TrajectoryBatch


def make_batch() -> TrajectoryBatch:
    hidden = np.zeros((6, 3, 2, 4), dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 0, 1])
    splits = np.array(["train", "train", "train", "train", "test", "test"])
    mask = np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=bool,
    )
    hidden[labels == 1, :, :, 0] = 3.0
    hidden[labels == 0, :, :, 0] = -1.0
    return TrajectoryBatch(
        example_ids=tuple(f"e{index}" for index in range(6)),
        labels=labels,
        splits=splits,
        hidden_states=hidden,
        token_mask=mask,
        token_logprobs=np.zeros((6, 3), dtype=np.float32),
        token_entropies=np.zeros((6, 3), dtype=np.float32),
    )


def test_direction_is_unit_length_and_risk_projection_is_larger():
    batch = make_batch()
    model = LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 3]))

    scores = model.transform(batch)

    np.testing.assert_allclose(np.linalg.norm(model.directions, axis=1), 1.0)
    assert scores[2, 0, 0] > scores[0, 0, 0]
    assert model.fit_example_ids == ("e0", "e1", "e2", "e3")
    assert np.all(scores[~batch.token_mask] == 0.0)


def test_each_example_is_pooled_before_class_means_are_computed():
    batch = make_batch()
    model = LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 3]))

    np.testing.assert_allclose(model.centers[:, 0], 1.0)


def test_fit_rejects_test_examples_and_missing_classes():
    batch = make_batch()

    with pytest.raises(ValueError, match="training split only"):
        LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 5]))
    with pytest.raises(ValueError, match="both classes"):
        LayerwiseContrastiveDirection().fit(batch, np.array([0, 1]))
