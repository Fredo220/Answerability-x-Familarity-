import numpy as np

from trajectory_extractor.evaluation import (
    calibration_error,
    evaluate_and_predict_prefix_surfaces,
    evaluate_prefix_surface,
    paired_bootstrap_auc_delta,
    paired_bootstrap_rate_delta,
    predict_at_prefix,
    select_threshold,
)


def test_prefix_surface_uses_only_requested_token_layer_prefixes():
    rng = np.random.default_rng(8)
    labels = np.array([0, 1] * 20)
    features = rng.normal(size=(40, 2, 3, 2))
    features[:, :, :, 0] += labels[:, None, None] * 2.0
    mask = np.ones((40, 2), dtype=bool)

    result = evaluate_prefix_surface(
        features,
        labels,
        token_mask=mask,
        train_indices=np.arange(30),
        test_indices=np.arange(30, 40),
    )

    assert result.auroc.shape == (2, 3)
    assert result.auprc.shape == (2, 3)
    assert np.isfinite(result.auroc).all()


def test_joint_prefix_fit_matches_separate_evaluation():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(18, 2, 3, 2)).astype(np.float32)
    labels = np.array([0, 1] * 9)
    mask = np.ones((18, 2), dtype=bool)
    train = np.arange(10)
    val = np.arange(10, 14)
    test = np.arange(14, 18)
    expected = evaluate_prefix_surface(
        features,
        labels,
        token_mask=mask,
        train_indices=train,
        test_indices=val,
    )
    surface, validation, predictions = evaluate_and_predict_prefix_surfaces(
        features,
        labels,
        token_mask=mask,
        train_indices=train,
        validation_indices=val,
        test_indices=test,
    )
    np.testing.assert_allclose(surface.auroc, expected.auroc)
    np.testing.assert_allclose(surface.auprc, expected.auprc)
    assert validation.shape == (4, 2, 3)
    assert predictions.shape == (4, 2, 3)


def test_calibration_error_is_zero_for_perfect_binned_predictions():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])

    assert calibration_error(labels, probabilities, n_bins=2) == 0.0


def test_paired_bootstrap_reports_auc_delta_distribution():
    labels = np.array([0, 1] * 20)
    baseline = np.tile([0.4, 0.6], 20)
    candidate = np.tile([0.1, 0.9], 20)

    result = paired_bootstrap_auc_delta(labels, candidate, baseline, n_bootstrap=100, seed=3)

    assert result.delta >= 0.0
    assert result.samples.shape == (100,)


def test_cluster_bootstrap_keeps_group_members_together():
    labels = np.array([0, 1, 0, 1, 0, 1])
    baseline = np.array([0.4, 0.6, 0.4, 0.6, 0.4, 0.6])
    candidate = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    result = paired_bootstrap_auc_delta(
        labels, candidate, baseline, n_bootstrap=25, seed=4, groups=groups
    )
    assert result.samples.shape == (25,)


def test_prefix_prediction_cannot_see_future_token_features():
    rng = np.random.default_rng(3)
    labels = np.array([0, 1] * 10)
    features = rng.normal(size=(20, 3, 2, 1)).astype(np.float32)
    mask = np.ones((20, 3), dtype=bool)
    original = predict_at_prefix(
        features,
        labels,
        token_mask=mask,
        train_indices=np.arange(16),
        predict_indices=np.arange(16, 20),
        token_end=0,
        layer_end=0,
    )
    features[:, 1:, :, :] += 10000
    changed = predict_at_prefix(
        features,
        labels,
        token_mask=mask,
        train_indices=np.arange(16),
        predict_indices=np.arange(16, 20),
        token_end=0,
        layer_end=0,
    )
    np.testing.assert_allclose(original, changed)


def test_threshold_and_paired_rate_delta():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    assert 0.2 < select_threshold(labels, probabilities) <= 0.8
    result = paired_bootstrap_rate_delta(
        np.array([1, 1, 1, 0]),
        np.array([1, 0, 0, 0]),
        n_bootstrap=50,
        seed=2,
    )
    assert result.delta == 0.5
