import numpy as np
import pytest

import trajectory_extractor.contrastive_directions as contrastive_directions
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
    hidden = np.zeros((4, 3, 1, 1), dtype=np.float32)
    hidden[0, :, 0, 0] = 0.0
    hidden[1, :, 0, 0] = 4.0
    hidden[2, :, 0, 0] = 10.0
    hidden[3, :, 0, 0] = 12.0
    batch = TrajectoryBatch(
        example_ids=("c0", "c1", "r0", "r1"),
        labels=np.array([0, 0, 1, 1]),
        splits=np.array(["train", "train", "train", "train"]),
        hidden_states=hidden,
        token_mask=np.array(
            [
                [1, 0, 0],
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
        token_logprobs=np.zeros((4, 3), dtype=np.float32),
        token_entropies=np.zeros((4, 3), dtype=np.float32),
    )
    model = LayerwiseContrastiveDirection().fit(batch, np.arange(4))

    np.testing.assert_allclose(model.centers[:, 0], 6.5)


def test_fit_rejects_test_examples_and_missing_classes():
    batch = make_batch()

    with pytest.raises(ValueError, match="training split only"):
        LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 5]))
    with pytest.raises(ValueError, match="both classes"):
        LayerwiseContrastiveDirection().fit(batch, np.array([0, 1]))


def test_malformed_held_out_examples_do_not_affect_train_only_fit():
    batch = make_batch()
    batch.token_mask[4] = False
    batch.hidden_states[5, 0, 0, 0] = np.nan

    model = LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 3]))

    np.testing.assert_allclose(model.centers[:, 0], 1.0)


def test_fit_rejects_non_finite_valid_train_activation():
    batch = make_batch()
    batch.hidden_states[0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite valid-token activations"):
        LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 3]))


def test_transform_rejects_non_finite_valid_activation():
    batch = make_batch()
    model = LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 3]))
    batch.hidden_states[0, 0, 0, 0] = np.inf

    with pytest.raises(ValueError, match="finite valid-token activations"):
        model.transform(batch)


def test_masked_non_finite_train_padding_does_not_affect_fit():
    batch = make_batch()
    batch.hidden_states[0, 1, 0, :] = np.nan
    batch.hidden_states[0, 2, 1, :] = np.inf

    model = LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 3]))

    np.testing.assert_allclose(model.centers[:, 0], 1.0)


def test_masked_non_finite_transform_padding_produces_zero_scores():
    batch = make_batch()
    model = LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 3]))
    batch.hidden_states[0, 1, 0, :] = np.nan
    batch.hidden_states[4, 2, 1, :] = np.inf

    scores = model.transform(batch)

    assert np.all(scores[~batch.token_mask] == 0.0)


def test_fit_uses_float64_means_and_norms_before_cast(monkeypatch):
    batch = make_batch()
    pooled = np.zeros((4, 2, 4), dtype=np.float32)
    pooled[2:, :, :] = np.float32(1e20)
    monkeypatch.setattr(
        contrastive_directions,
        "_pool_valid_tokens_per_example",
        lambda batch, indices: pooled,
    )

    model = LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 3]))

    assert model.directions.dtype == np.float32
    assert np.isfinite(model.directions).all()
    np.testing.assert_allclose(
        np.linalg.norm(model.directions.astype(np.float64), axis=1),
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )


def test_fit_rejects_non_finite_float64_norm(monkeypatch):
    batch = make_batch()
    pooled = np.zeros((4, 2, 4), dtype=np.float64)
    pooled[2:, :, :] = 8e307
    monkeypatch.setattr(
        contrastive_directions,
        "_pool_valid_tokens_per_example",
        lambda batch, indices: pooled,
    )

    with pytest.raises(ValueError, match="finite positive norm"):
        LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 3]))
