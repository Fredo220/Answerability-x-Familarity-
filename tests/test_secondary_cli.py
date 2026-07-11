import json

import numpy as np

from trajectory_extractor import cli
from trajectory_extractor.secondary_artifacts import SecondaryArtifactStore


def test_evaluate_secondary_concept_writes_isolated_artifacts(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_id": "test/model",
                "model_revision": "revision",
                "device": "cpu",
                "dtype": "float32",
                "seed": 42,
                "max_new_tokens": 2,
                "pca_dims": 2,
                "ridge_alpha": 0.001,
                "temperature": 0.0,
                "output_dir": str(tmp_path / "runs"),
            }
        )
    )
    fake_batch = object()
    monkeypatch.setattr(cli.RunStore, "load_batch", lambda self, run_id, label_key: fake_batch)

    def fake_evaluation(batch, **kwargs):
        assert batch is fake_batch
        assert kwargs == {"pca_dims": 2, "ridge_alpha": 0.001, "n_bootstrap": 20}
        return {
            "methods": {
                "contrastive_vector": {"test_auroc": 0.60},
                "contrastive_plus_dynamics": {"test_auroc": 0.70},
                "full_metacognitive_monitor": {"test_auroc": 0.72},
            },
            "registered_comparison": {"supported": True},
            "endpoint_status": {"evaluable": True},
            "claim_status": "provisional_supported",
            "artifacts": {
                "directions": np.eye(2, dtype=np.float32),
                "centers": np.zeros((2, 2), dtype=np.float32),
                "vector_means": np.zeros(2, dtype=np.float32),
                "vector_scales": np.ones(2, dtype=np.float32),
                "validation_indices": np.array([2, 3]),
                "validation_labels": np.array([0, 1]),
                "test_indices": np.array([0, 1]),
                "test_labels": np.array([0, 1]),
                "contrastive_vector_probability": np.array([0.2, 0.6]),
                "metacognitive_risk_probability": np.array([0.1, 0.9]),
                "validation_metacognitive_risk_surface": np.array([[[0.1]], [[0.9]]]),
                "full_monitor_probability": np.array([0.1, 0.95]),
                "bootstrap_delta_samples": np.array([0.05, 0.10]),
            },
        }

    monkeypatch.setattr(cli, "evaluate_concept_secondary", fake_evaluation)
    monkeypatch.setattr(cli, "_write_secondary_figures", lambda *args, **kwargs: None)

    exit_code = cli.main(
        [
            "evaluate-secondary-concept",
            "--config",
            str(config_path),
            "--run-id",
            "concept-main",
            "--bootstrap",
            "20",
            "--endpoint",
            "exact_error",
        ]
    )

    root = tmp_path / "runs" / "concept-main" / "secondary"
    assert exit_code == 0
    assert (root / "comparisons" / "detection_exact_error.json").exists()
    assert (root / "comparisons" / "predictions_exact_error.npz").exists()
    assert (root / "contrastive_vectors" / "exact_error.npz").exists()
    assert (root / "vector_dynamics" / "exact_error.npz").exists()
    assert not (tmp_path / "runs" / "concept-main" / "metrics" / "detection.json").exists()

    with np.load(root / "comparisons" / "predictions_exact_error.npz") as predictions:
        assert set(predictions.files) == {
            "validation_indices",
            "validation_labels",
            "test_indices",
            "test_labels",
            "contrastive_vector_probability",
            "metacognitive_risk_probability",
            "validation_metacognitive_risk_surface",
            "full_monitor_probability",
            "bootstrap_delta_samples",
        }
        assert "metacognitive_risk_surface" not in predictions.files


def test_secondary_figures_stay_within_sanitized_artifact_namespace(tmp_path):
    store = SecondaryArtifactStore(tmp_path)
    metrics_path = store.write_json("../escaped", "comparisons", "detection", {})
    result = {
        "methods": {
            "contrastive_vector": {"test_auroc": 0.60},
            "contrastive_plus_dynamics": {"test_auroc": 0.70},
        }
    }
    arrays = {
        "validation_labels": np.array([0, 1]),
        "validation_metacognitive_risk_surface": np.array([[[0.1]], [[0.9]]]),
    }

    cli._write_secondary_figures(store, "../escaped", result, arrays, endpoint="exact_error")

    assert (metrics_path.parent / "figures" / "method_comparison_exact_error.png").exists()
    assert (
        metrics_path.parent / "figures" / "validation_metacognitive_risk_gap_exact_error.png"
    ).exists()
