import numpy as np

from trajectory_extractor.study import build_real_transfer_evaluation_batch, evaluate_detection_methods
from trajectory_extractor.types import TrajectoryBatch


def test_detection_pipeline_runs_end_to_end_on_local_tiny_batch():
    rng = np.random.default_rng(12)
    n = 36
    labels = np.array([0, 1] * (n // 2))
    hidden = rng.normal(size=(n, 2, 4, 8)).astype(np.float16)
    hidden[:, :, 2:, 0] += labels[:, None, None] * 0.8
    batch = TrajectoryBatch(
        example_ids=tuple(f"example-{index}" for index in range(n)),
        labels=labels,
        splits=np.array(["train"] * 20 + ["val"] * 8 + ["test"] * 8),
        hidden_states=hidden,
        token_mask=np.ones((n, 2), dtype=bool),
        token_logprobs=rng.normal(size=(n, 2)).astype(np.float32),
        token_entropies=np.abs(rng.normal(size=(n, 2))).astype(np.float32),
    )
    result = evaluate_detection_methods(
        batch,
        pca_dims=3,
        ridge_alpha=1e-3,
        n_bootstrap=20,
    )
    assert set(result["methods"]) == {
        "output",
        "static",
        "raw_dynamics",
        "operator_residual",
        "combined",
    }
    assert result["decision"]["outcome"] in {
        "supported",
        "partially_supported",
        "not_supported",
    }
    assert len(result["bootstrap_samples"]) == 20
    assert result["operator_reference_class"] == 0
    label_by_id = dict(zip(batch.example_ids, batch.labels, strict=True))
    assert all(label_by_id[example_id] == 0 for example_id in result["fit_example_ids"]["operator"])
    for method in result["methods"].values():
        assert 0.0 <= method["test_auroc"] <= 1.0
        assert 0.0 <= method["test_auprc"] <= 1.0


def test_primary_evaluation_marks_cross_cell_threshold_diagnostics_not_interpretable():
    rng = np.random.default_rng(27)
    labels = np.array([0, 1] * 18)
    batch = TrajectoryBatch(
        example_ids=tuple(f"example-{index}" for index in range(len(labels))),
        labels=labels,
        splits=np.array(["train"] * 20 + ["val"] * 8 + ["test"] * 8),
        hidden_states=rng.normal(size=(len(labels), 2, 4, 8)).astype(np.float16),
        token_mask=np.ones((len(labels), 2), dtype=bool),
        token_logprobs=rng.normal(size=(len(labels), 2)).astype(np.float32),
        token_entropies=np.abs(rng.normal(size=(len(labels), 2))).astype(np.float32),
    )

    result = evaluate_detection_methods(batch, pca_dims=3, n_bootstrap=20)

    for method in result["methods"].values():
        assert method["validation_diagnostics"] == {
            "status": "not_interpretable",
            "reason": (
                "independently_fitted_prefix_classifiers_do_not_share_a_transferable_threshold"
            ),
        }
        assert method["test_false_positive_rate"] == method["test"]["false_positive_rate"]
        assert "median_positive_crossing" not in method
        assert "median_positive_crossing_token" not in method
        assert "median_positive_crossing_layer" not in method


def test_real_transfer_batch_uses_only_concept_train_and_validation_for_fitting():
    rng = np.random.default_rng(2)

    def batch(prefix, splits):
        count = len(splits)
        return TrajectoryBatch(
            example_ids=tuple(f"{prefix}-{index}" for index in range(count)),
            labels=np.array([0, 1] * (count // 2)),
            splits=np.asarray(splits),
            hidden_states=rng.normal(size=(count, 2, 3, 4)).astype(np.float16),
            token_mask=np.ones((count, 2), dtype=bool),
            token_logprobs=np.zeros((count, 2), dtype=np.float32),
            token_entropies=np.zeros((count, 2), dtype=np.float32),
        )

    concept = batch("concept", ["train", "train", "val", "val", "test", "test"])
    transfer = batch("transfer", ["train", "train", "val", "val"])
    combined = build_real_transfer_evaluation_batch(concept, transfer)
    assert combined.example_ids == (
        "concept::concept-0",
        "concept::concept-1",
        "concept::concept-2",
        "concept::concept-3",
        "transfer::transfer-0",
        "transfer::transfer-1",
        "transfer::transfer-2",
        "transfer::transfer-3",
    )
    assert combined.splits.tolist() == ["train", "train", "val", "val", "test", "test", "test", "test"]
