import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from trajectory_extractor import cli
from trajectory_extractor.secondary_artifacts import SecondaryArtifactStore


def _write_config_and_run_provenance(tmp_path):
    output_dir = tmp_path / "runs"
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
                "output_dir": str(output_dir),
            }
        )
    )
    run_root = output_dir / "concept-main"
    examples = run_root / "examples"
    examples.mkdir(parents=True)
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "config": {"model_id": "test/model", "model_revision": "revision"},
                "dataset": {
                    "path": "data/processed/concept_mixing.jsonl",
                    "sha256": "a" * 64,
                    "manifest_path": "data/processed/concept_mixing.jsonl.manifest.json",
                    "manifest_sha256": "b" * 64,
                },
            }
        )
    )
    (examples / "train-0001.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "model_id": "test/model",
                    "model_revision": "revision",
                    "resolved_model_revision": "resolved-revision",
                }
            }
        )
    )
    return config_path, output_dir


def _fake_secondary_result():
    return {
        "methods": {
            "contrastive_vector": {"test_auroc": 0.60},
            "contrastive_plus_dynamics": {"test_auroc": 0.70},
            "full_metacognitive_monitor": {"test_auroc": 0.72},
        },
        "registered_comparison": {
            "supported": True,
            "p_value_method": "paired_entity_family_permutation",
            "permutation_seed": 42,
            "n_permutations": 2000,
            "fdr_family": [
                "detection_vector_dynamics",
                "intervention_capping_vs_triggered_pending",
            ],
        },
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


def test_evaluate_secondary_concept_writes_isolated_artifacts(tmp_path, monkeypatch):
    config_path, output_dir = _write_config_and_run_provenance(tmp_path)
    run_root = output_dir / "concept-main"
    primary_before = {
        path.relative_to(run_root): path.read_bytes()
        for path in run_root.rglob("*")
        if path.is_file()
    }
    fake_batch = object()
    monkeypatch.setattr(cli.RunStore, "load_batch", lambda self, run_id, label_key: fake_batch)

    def fake_evaluation(batch, **kwargs):
        assert batch is fake_batch
        assert kwargs == {"pca_dims": 2, "ridge_alpha": 0.001, "n_bootstrap": 20}
        return _fake_secondary_result()

    monkeypatch.setattr(cli, "evaluate_concept_secondary", fake_evaluation)

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

    root = output_dir / "concept-main" / "secondary"
    assert exit_code == 0
    assert (root / "comparisons" / "detection_exact_error.json").exists()
    assert (root / "comparisons" / "predictions_exact_error.npz").exists()
    assert (root / "contrastive_vectors" / "exact_error.npz").exists()
    assert (root / "vector_dynamics" / "exact_error.npz").exists()
    assert (root / "comparisons" / "completion_exact_error.json").exists()
    assert not (root / "comparisons" / "claim_exact_error.json").exists()
    assert not (output_dir / "concept-main" / "metrics" / "detection.json").exists()
    assert not (
        output_dir / "concept-main" / "metrics" / "detection_exact_error.json"
    ).exists()
    primary_after = {
        path.relative_to(run_root): path.read_bytes()
        for path in run_root.rglob("*")
        if path.is_file() and "secondary" not in path.relative_to(run_root).parts
    }
    assert primary_after == primary_before

    metrics = json.loads((root / "comparisons" / "detection_exact_error.json").read_text())
    assert metrics["claim_status"] == "provisional_supported"
    assert set(metrics["runtime"]) == {"seconds", "max_resident_set_size"}
    assert "artifacts" not in metrics
    provenance = metrics["analysis_provenance"]
    assert metrics["analysis_id"] == provenance["analysis_id"]
    assert provenance["inputs"]["config"]["sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert provenance["model"] == {
        "id": "test/model",
        "requested_revision": "revision",
        "resolved_revision": "resolved-revision",
    }
    assert provenance["analysis"] == {
        "endpoint": "exact_error",
        "pca_dims": 2,
        "ridge_alpha": 0.001,
    }
    assert provenance["permutation"]["count"] == 2000
    secondary_preregistration = (
        Path(cli.__file__).resolve().parents[2] / "docs" / "secondary_preregistration.md"
    )
    assert provenance["inputs"]["preregistration"] == {
        "path": str(secondary_preregistration),
        "sha256": hashlib.sha256(secondary_preregistration.read_bytes()).hexdigest(),
    }

    completion = json.loads(
        (root / "comparisons" / "completion_exact_error.json").read_text()
    )
    expected_artifacts = {
        "comparisons/detection_exact_error.json",
        "comparisons/figures/method_comparison_exact_error.png",
        "comparisons/figures/validation_metacognitive_risk_gap_exact_error.png",
        "comparisons/predictions_exact_error.npz",
        "contrastive_vectors/exact_error.npz",
        "vector_dynamics/exact_error.npz",
    }
    assert completion["schema_version"] == 1
    assert completion["analysis_id"] == metrics["analysis_id"]
    assert set(completion["artifacts"]) == expected_artifacts
    assert SecondaryArtifactStore(output_dir).verify_completion(
        "concept-main", "exact_error"
    ) == completion

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


def test_existing_legacy_metrics_refuse_rerun_before_batch_load(tmp_path, monkeypatch):
    config_path, output_dir = _write_config_and_run_provenance(tmp_path)
    SecondaryArtifactStore(output_dir).write_json(
        "concept-main", "comparisons", "detection_exact_error", {"legacy": True}
    )

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("completed endpoint must not load its batch")

    monkeypatch.setattr(cli.RunStore, "load_batch", fail_if_loaded)
    monkeypatch.setattr(cli, "evaluate_concept_secondary", fail_if_loaded)

    with pytest.raises(FileExistsError, match="detection_exact_error.json"):
        cli.main(
            [
                "evaluate-secondary-concept",
                "--config",
                str(config_path),
                "--run-id",
                "concept-main",
                "--endpoint",
                "exact_error",
            ]
        )
    assert not (
        output_dir
        / "concept-main"
        / "secondary"
        / "comparisons"
        / "claim_exact_error.json"
    ).exists()


def test_concurrent_claim_refuses_before_batch_load(tmp_path, monkeypatch):
    config_path, output_dir = _write_config_and_run_provenance(tmp_path)
    store = SecondaryArtifactStore(output_dir)
    claim = store.acquire_claim("concept-main", "exact_error")

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("contended endpoint must not load its batch")

    monkeypatch.setattr(cli.RunStore, "load_batch", fail_if_loaded)

    with pytest.raises(FileExistsError, match="claim_exact_error.json"):
        cli.main(
            [
                "evaluate-secondary-concept",
                "--config",
                str(config_path),
                "--run-id",
                "concept-main",
                "--endpoint",
                "exact_error",
            ]
        )

    assert claim.path.exists()


def test_failed_evaluation_leaves_claim_as_permanent_block(tmp_path, monkeypatch):
    config_path, output_dir = _write_config_and_run_provenance(tmp_path)
    monkeypatch.setattr(cli.RunStore, "load_batch", lambda *args, **kwargs: object())

    def fail_evaluation(*args, **kwargs):
        raise RuntimeError("injected evaluation failure")

    monkeypatch.setattr(cli, "evaluate_concept_secondary", fail_evaluation)
    command = [
        "evaluate-secondary-concept",
        "--config",
        str(config_path),
        "--run-id",
        "concept-main",
        "--endpoint",
        "exact_error",
    ]

    with pytest.raises(RuntimeError, match="injected evaluation failure"):
        cli.main(command)

    claim = (
        output_dir
        / "concept-main"
        / "secondary"
        / "comparisons"
        / "claim_exact_error.json"
    )
    assert claim.exists()

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("failed claimed endpoint must not reload its batch")

    monkeypatch.setattr(cli.RunStore, "load_batch", fail_if_loaded)
    with pytest.raises(FileExistsError, match="claim_exact_error.json"):
        cli.main(command)


def test_completion_marker_is_not_written_when_figure_generation_fails(tmp_path, monkeypatch):
    config_path, output_dir = _write_config_and_run_provenance(tmp_path)
    monkeypatch.setattr(cli.RunStore, "load_batch", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cli,
        "evaluate_concept_secondary",
        lambda *args, **kwargs: _fake_secondary_result(),
    )

    def fail_figures(*args, **kwargs):
        raise RuntimeError("injected figure failure")

    monkeypatch.setattr(cli, "_write_secondary_figures", fail_figures)

    with pytest.raises(RuntimeError, match="injected figure failure"):
        cli.main(
            [
                "evaluate-secondary-concept",
                "--config",
                str(config_path),
                "--run-id",
                "concept-main",
                "--endpoint",
                "exact_error",
            ]
        )

    comparisons = output_dir / "concept-main" / "secondary" / "comparisons"
    assert (comparisons / "detection_exact_error.json").exists()
    assert not (comparisons / "completion_exact_error.json").exists()
    assert (comparisons / "claim_exact_error.json").exists()


def test_secondary_figures_stay_within_artifact_namespace(tmp_path):
    store = SecondaryArtifactStore(tmp_path)
    metrics_path = store.write_json("concept-main", "comparisons", "detection", {})
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

    directory = metrics_path.parent / "figures"
    cli._write_secondary_figures(directory, result, arrays, endpoint="exact_error")

    assert (directory / "method_comparison_exact_error.png").exists()
    assert (
        directory / "validation_metacognitive_risk_gap_exact_error.png"
    ).exists()


def test_secondary_figures_use_durable_directory_creation(tmp_path, monkeypatch):
    store = SecondaryArtifactStore(tmp_path)
    metrics_path = store.write_json("concept-main", "comparisons", "detection", {})
    directory = metrics_path.parent / "figures"
    created = []
    monkeypatch.setattr(
        cli,
        "ensure_durable_directory",
        lambda path: created.append(path),
        raising=False,
    )

    cli._write_secondary_figures(
        directory,
        {
            "methods": {
                "contrastive_vector": {"test_auroc": 0.60},
                "contrastive_plus_dynamics": {"test_auroc": 0.70},
            }
        },
        {
            "validation_labels": np.array([0, 1]),
            "validation_metacognitive_risk_surface": np.array(
                [[[0.1]], [[0.9]]]
            ),
        },
        endpoint="exact_error",
    )

    assert created == [directory]


def test_report_study_defaults_to_generated_report(tmp_path, monkeypatch):
    config_path, output_dir = _write_config_and_run_provenance(tmp_path)
    captured = {}

    def fake_write(store, output):
        captured["store_root"] = store.root
        captured["output"] = output
        return Path(output)

    monkeypatch.setattr(cli, "write_study_report", fake_write)

    assert cli.main(["report-study", "--config", str(config_path)]) == 0
    assert captured == {
        "store_root": output_dir,
        "output": "docs/generated_study_report.md",
    }


@pytest.mark.parametrize(
    "output",
    [
        "docs/results.md",
        str(Path(cli.__file__).resolve().parents[2] / "docs" / "results.md"),
        "docs/../docs/results.md",
    ],
)
def test_report_study_rejects_repository_results_path(tmp_path, monkeypatch, output):
    config_path, _ = _write_config_and_run_provenance(tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("protected results report must not be written")

    monkeypatch.setattr(cli, "write_study_report", fail_if_called)

    with pytest.raises(ValueError, match="docs/results.md"):
        cli.main(
            [
                "report-study",
                "--config",
                str(config_path),
                "--output",
                output,
            ]
        )


def test_report_study_rejects_hard_link_to_repository_results(
    tmp_path, monkeypatch
):
    config_path, _ = _write_config_and_run_provenance(tmp_path)
    protected = Path(cli.__file__).resolve().parents[2] / "docs" / "results.md"
    alias = tmp_path / "results-hard-link.md"
    alias.hardlink_to(protected)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("protected results report must not be written")

    monkeypatch.setattr(cli, "write_study_report", fail_if_called)

    with pytest.raises(ValueError, match="docs/results.md"):
        cli.main(
            [
                "report-study",
                "--config",
                str(config_path),
                "--output",
                str(alias),
            ]
        )


def test_report_study_rejects_case_alias_when_filesystem_exposes_it(
    tmp_path, monkeypatch
):
    config_path, _ = _write_config_and_run_provenance(tmp_path)
    protected = Path(cli.__file__).resolve().parents[2] / "docs" / "results.md"
    alias = protected.with_name(protected.name.swapcase())
    if not alias.exists() or not alias.samefile(protected):
        pytest.skip("filesystem does not expose case-insensitive aliases")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("protected results report must not be written")

    monkeypatch.setattr(cli, "write_study_report", fail_if_called)

    with pytest.raises(ValueError, match="docs/results.md"):
        cli.main(
            [
                "report-study",
                "--config",
                str(config_path),
                "--output",
                str(alias),
            ]
        )


def test_report_output_identity_retains_resolved_fallback_for_nonexistent_path(
    tmp_path,
):
    protected = tmp_path / "docs" / "results.md"
    protected.parent.mkdir()
    protected.write_text("protected")
    candidate = tmp_path / "missing" / ".." / "docs" / "results.md"

    assert not candidate.exists()
    assert cli._same_output_file(candidate, protected)
