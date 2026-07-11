import numpy as np

from trajectory_extractor.features import compute_raw_dynamics, make_method_tensor
from trajectory_extractor.operator_residual import LayerwiseOperatorResidual
from trajectory_extractor.types import TrajectoryBatch


def make_batch():
    rng = np.random.default_rng(4)
    hidden = rng.normal(size=(6, 3, 4, 5)).astype(np.float32)
    return TrajectoryBatch(
        example_ids=tuple(f"e{i}" for i in range(6)),
        labels=np.array([0, 1, 0, 1, 0, 1]),
        splits=np.array(["train", "train", "train", "train", "test", "test"]),
        hidden_states=hidden,
        token_mask=np.array([[1, 1, 1]] * 6, dtype=bool),
        token_logprobs=np.zeros((6, 3), dtype=np.float32),
        token_entropies=np.ones((6, 3), dtype=np.float32),
    )


def test_raw_dynamics_have_token_layer_feature_axes():
    batch = make_batch()

    dynamics = compute_raw_dynamics(batch.hidden_states)

    assert dynamics.velocity.shape == (6, 3, 3)
    assert dynamics.curvature.shape == (6, 3, 2)
    assert dynamics.direction_change.shape == (6, 3, 2)
    assert np.isfinite(dynamics.velocity).all()


def test_operator_fits_training_examples_only_and_scores_all_examples():
    batch = make_batch()
    operator = LayerwiseOperatorResidual(n_components=2, ridge_alpha=1e-3)

    operator.fit(batch, train_indices=np.array([0, 1, 2, 3]))
    residuals = operator.transform(batch)

    assert operator.fit_example_ids == ("e0", "e1", "e2", "e3")
    assert residuals.shape == (6, 3, 3)
    assert np.isfinite(residuals).all()


def test_method_tensor_combines_output_static_and_dynamic_features():
    batch = make_batch()
    operator = LayerwiseOperatorResidual(n_components=2).fit(batch, np.array([0, 1, 2, 3]))

    features = make_method_tensor(batch, operator_residuals=operator.transform(batch))

    assert features.shape[:3] == (6, 3, 4)
    assert features.shape[-1] >= 6
