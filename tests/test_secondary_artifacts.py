import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import trajectory_extractor.secondary_artifacts as secondary_artifacts
from trajectory_extractor.secondary_artifacts import SecondaryArtifactStore


def test_secondary_artifacts_round_trip_in_isolated_namespace(tmp_path):
    store = SecondaryArtifactStore(tmp_path)

    json_path = store.write_json(
        "concept-main",
        "comparisons",
        "detection_exact_error",
        {"supported": False, "delta": 0.01},
    )
    array_path = store.write_npz(
        "concept-main",
        "contrastive_vectors",
        "exact_error",
        directions=np.eye(2, dtype=np.float32),
        centers=np.ones((2, 2), dtype=np.float32),
    )

    assert json_path == tmp_path / "concept-main" / "secondary" / "comparisons" / "detection_exact_error.json"
    assert array_path == tmp_path / "concept-main" / "secondary" / "contrastive_vectors" / "exact_error.npz"
    assert store.read_json("concept-main", "comparisons", "detection_exact_error")["delta"] == 0.01
    arrays = store.read_npz("concept-main", "contrastive_vectors", "exact_error")
    np.testing.assert_array_equal(arrays["directions"], np.eye(2, dtype=np.float32))


def test_secondary_artifacts_reject_unknown_sections(tmp_path):
    store = SecondaryArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="secondary section"):
        store.write_json("concept-main", "metrics", "x", {})


@pytest.mark.parametrize(
    "identifier",
    ["a/b", "a b", "../outside", ".", "..", "_leading", "trailing_"],
)
def test_secondary_artifacts_reject_lossy_or_unsafe_identifiers(tmp_path, identifier):
    store = SecondaryArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="safe grammar"):
        store.write_json(identifier, "comparisons", "result", {})
    with pytest.raises(ValueError, match="safe grammar"):
        store.write_json("concept-main", "comparisons", identifier, {})


def test_secondary_artifacts_keep_distinct_valid_identifiers_exact(tmp_path):
    store = SecondaryArtifactStore(tmp_path)

    slashless = store.write_json("concept-main", "comparisons", "a_b", {"ok": True})
    endpoint = store.write_json(
        "concept-main", "comparisons", "detection_exact_error", {"ok": True}
    )

    assert slashless.name == "a_b.json"
    assert endpoint.name == "detection_exact_error.json"


def test_replacement_leaves_a_complete_readable_artifact(tmp_path):
    store = SecondaryArtifactStore(tmp_path)
    store.write_json("run", "comparisons", "result", {"version": 1})
    store.write_json("run", "comparisons", "result", {"version": 2})

    assert store.read_json("run", "comparisons", "result") == {"version": 2}
    assert list((tmp_path / "run" / "secondary" / "comparisons").glob("*.tmp")) == []


def test_npz_replacement_leaves_version_two_complete_and_readable(tmp_path):
    store = SecondaryArtifactStore(tmp_path)
    first = np.array([1, 2], dtype=np.int64)
    second = np.array([3, 4], dtype=np.int64)

    store.write_npz("run", "vector_dynamics", "result", values=first)
    store.write_npz("run", "vector_dynamics", "result", values=second)

    arrays = store.read_npz("run", "vector_dynamics", "result")
    np.testing.assert_array_equal(arrays["values"], second)
    assert list((tmp_path / "run" / "secondary" / "vector_dynamics").glob("*.tmp")) == []


def test_npz_write_failure_preserves_existing_artifact_and_cleans_temp(tmp_path, monkeypatch):
    store = SecondaryArtifactStore(tmp_path)
    original = np.array([1, 2], dtype=np.int64)
    replacement = np.array([3, 4], dtype=np.int64)
    store.write_npz("run", "vector_dynamics", "result", values=original)

    def write_partial_then_raise(handle, **arrays):
        handle.write(b"partial npz")
        raise RuntimeError("injected npz failure")

    monkeypatch.setattr(secondary_artifacts.np, "savez_compressed", write_partial_then_raise)
    with pytest.raises(RuntimeError, match="injected npz failure"):
        store.write_npz("run", "vector_dynamics", "result", values=replacement)

    arrays = store.read_npz("run", "vector_dynamics", "result")
    np.testing.assert_array_equal(arrays["values"], original)
    assert list((tmp_path / "run" / "secondary" / "vector_dynamics").glob("*.tmp")) == []


def test_json_replace_failure_preserves_existing_artifact_and_cleans_temp(tmp_path, monkeypatch):
    store = SecondaryArtifactStore(tmp_path)
    store.write_json("run", "comparisons", "result", {"version": 1})

    def raise_on_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(secondary_artifacts.os, "replace", raise_on_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        store.write_json("run", "comparisons", "result", {"version": 2})

    assert store.read_json("run", "comparisons", "result") == {"version": 1}
    assert list((tmp_path / "run" / "secondary" / "comparisons").glob("*.tmp")) == []


def test_existing_metrics_or_completion_marker_blocks_secondary_rerun(tmp_path):
    store = SecondaryArtifactStore(tmp_path)
    metrics = store.write_json(
        "concept-main", "comparisons", "detection_exact_error", {"legacy": True}
    )

    with pytest.raises(FileExistsError, match=str(metrics)):
        store.assert_incomplete("concept-main", "exact_error")

    metrics.unlink()
    completion = store.write_json(
        "concept-main", "comparisons", "completion_exact_error", {"analysis_id": "abc"}
    )

    with pytest.raises(FileExistsError, match=str(completion)):
        store.assert_incomplete("concept-main", "exact_error")


def test_completion_marker_binds_analysis_id_to_metrics_hash(tmp_path):
    store = SecondaryArtifactStore(tmp_path)
    analysis_id = "a" * 64
    metrics = store.write_json(
        "concept-main", "comparisons", "detection_exact_error", {"analysis_id": analysis_id}
    )

    marker = store.write_completion(
        "concept-main", "exact_error", analysis_id=analysis_id, metrics_path=metrics
    )

    assert store.read_json("concept-main", "comparisons", "completion_exact_error") == {
        "analysis_id": analysis_id,
        "metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
    }
    assert marker.name == "completion_exact_error.json"


def test_analysis_provenance_is_canonical_and_rejects_ambiguous_model_revision(tmp_path):
    run_root = tmp_path / "runs" / "concept-main"
    examples = run_root / "examples"
    examples.mkdir(parents=True)
    config = tmp_path / "config.json"
    preregistration = tmp_path / "preregistration.md"
    config.write_text('{"model_id":"test/model","model_revision":"requested"}')
    preregistration.write_text("frozen protocol\n")
    manifest = {
        "config": {"model_id": "test/model", "model_revision": "requested"},
        "dataset": {
            "path": "data/concept.jsonl",
            "sha256": "a" * 64,
            "manifest_path": "data/concept.jsonl.manifest.json",
            "manifest_sha256": "b" * 64,
        },
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest))
    example = {
        "provenance": {
            "model_id": "test/model",
            "model_revision": "requested",
            "resolved_model_revision": "resolved-commit",
        }
    }
    (examples / "train-0001.json").write_text(json.dumps(example))

    kwargs = {
        "repo_root": Path(__file__).resolve().parents[1],
        "preregistration_path": preregistration,
        "config_path": config,
        "run_root": run_root,
        "endpoint": "exact_error",
        "pca_dims": 32,
        "ridge_alpha": 0.001,
        "bootstrap_count": 2000,
        "permutation_method": "paired_entity_family_permutation",
        "permutation_seed": 42,
        "permutation_count": 2000,
        "fdr_family": ["detection", "intervention"],
    }
    first = secondary_artifacts.build_analysis_provenance(**kwargs)
    second = secondary_artifacts.build_analysis_provenance(**kwargs)

    assert first == second
    assert first["analysis_id"] == hashlib.sha256(
        json.dumps(
            {key: value for key, value in first.items() if key != "analysis_id"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert first["implementation"]["git_commit"]
    assert isinstance(first["implementation"]["tracked_dirty"], bool)
    assert len(first["implementation"]["source_sha256"]) == 64
    assert first["inputs"]["dataset"] == manifest["dataset"]
    assert first["model"]["resolved_revision"] == "resolved-commit"
    assert first["bootstrap"] == {
        "method": "paired_entity_family_cluster_bootstrap",
        "unit": "entity_family",
        "seed": 42,
        "count": 2000,
    }

    conflicting = dict(example)
    conflicting["provenance"] = dict(example["provenance"])
    conflicting["provenance"]["resolved_model_revision"] = "other-commit"
    (examples / "train-0002.json").write_text(json.dumps(conflicting))

    with pytest.raises(ValueError, match="unique resolved model revision"):
        secondary_artifacts.build_analysis_provenance(**kwargs)
