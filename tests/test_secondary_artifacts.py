import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

import trajectory_extractor.secondary_artifacts as secondary_artifacts
from trajectory_extractor.secondary_artifacts import SecondaryArtifactStore


def _metrics_with_provenance(*, path_prefix: str = "/local/one"):
    provenance = {
        "schema_version": 1,
        "implementation": {
            "git_commit": "commit",
            "tracked_dirty": False,
            "source_sha256": "a" * 64,
        },
        "inputs": {
            "preregistration": {
                "path": f"{path_prefix}/docs/preregistration.md",
                "sha256": "b" * 64,
            },
            "config": {
                "path": f"{path_prefix}/configs/model.json",
                "sha256": "c" * 64,
            },
            "dataset": {
                "path": f"{path_prefix}/data/concept.jsonl",
                "sha256": "d" * 64,
                "manifest_path": f"{path_prefix}/data/concept.jsonl.manifest.json",
                "manifest_sha256": "e" * 64,
            },
        },
        "model": {
            "id": "test/model",
            "requested_revision": "main",
            "resolved_revision": "resolved",
        },
        "analysis": {"endpoint": "exact_error", "pca_dims": 2, "ridge_alpha": 0.001},
        "bootstrap": {
            "method": "paired_entity_family_cluster_bootstrap",
            "unit": "entity_family",
            "seed": 42,
            "count": 20,
        },
        "permutation": {
            "method": "paired_entity_family_permutation",
            "seed": 42,
            "count": 2000,
            "fdr_family": ["detection", "intervention"],
        },
    }
    analysis_id = secondary_artifacts.analysis_fingerprint(provenance)
    provenance["analysis_id"] = analysis_id
    return {
        "analysis_id": analysis_id,
        "analysis_provenance": provenance,
        "claim_status": "not_supported",
    }


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


def test_write_all_retries_short_writes(monkeypatch):
    read_descriptor, write_descriptor = os.pipe()
    real_write = os.write
    calls = []

    def short_write(descriptor, payload):
        calls.append(len(payload))
        return real_write(descriptor, payload[:2])

    monkeypatch.setattr(secondary_artifacts.os, "write", short_write)
    try:
        secondary_artifacts._write_all(write_descriptor, b"complete-payload")
    finally:
        os.close(write_descriptor)
    try:
        assert os.read(read_descriptor, 1024) == b"complete-payload"
    finally:
        os.close(read_descriptor)
    assert len(calls) > 1


def test_exclusive_publish_write_failure_leaves_no_destination_or_temp(
    tmp_path, monkeypatch
):
    destination = tmp_path / "claim.json"
    real_write = os.write
    calls = 0

    def partial_then_fail(descriptor, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, payload[:3])
        raise OSError("injected write failure")

    monkeypatch.setattr(secondary_artifacts.os, "write", partial_then_fail)

    with pytest.raises(OSError, match="injected write failure"):
        secondary_artifacts._exclusive_write_bytes(destination, b"complete-payload")

    assert not destination.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_exclusive_publish_fsync_failure_leaves_no_destination_or_temp(
    tmp_path, monkeypatch
):
    destination = tmp_path / "completion.json"

    def fail_fsync(descriptor):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(secondary_artifacts.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        secondary_artifacts._exclusive_write_bytes(destination, b"complete-payload")

    assert not destination.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_concurrent_exclusive_publishers_create_one_complete_winner(tmp_path):
    destination = tmp_path / "claim.json"
    payloads = (b"publisher-one", b"publisher-two-with-more-bytes")
    barrier = Barrier(2)

    def publish(payload):
        barrier.wait()
        try:
            secondary_artifacts._exclusive_write_bytes(destination, payload)
        except FileExistsError:
            return "blocked"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, payloads))

    assert sorted(outcomes) == ["blocked", "published"]
    assert destination.read_bytes() in payloads
    assert list(tmp_path.glob("*.tmp")) == []


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


def test_endpoint_claim_is_exclusive_and_persists_until_completed(tmp_path):
    store = SecondaryArtifactStore(tmp_path)

    claim = store.acquire_claim("concept-main", "exact_error")

    assert claim.path.exists()
    claim_payload = json.loads(claim.path.read_text())
    assert claim_payload["claim_id"] == claim.claim_id
    assert claim_payload["run_id"] == "concept-main"
    assert claim_payload["endpoint"] == "exact_error"
    with pytest.raises(FileExistsError, match="claim_exact_error.json"):
        store.acquire_claim("concept-main", "exact_error")
    with pytest.raises(RuntimeError, match="completion marker"):
        store.release_claim(claim)
    assert claim.path.exists()

    metrics = store.write_json(
        "concept-main", "comparisons", "detection_exact_error", _metrics_with_provenance()
    )
    marker = store.write_completion(
        "concept-main",
        "exact_error",
        analysis_id=_metrics_with_provenance()["analysis_id"],
        metrics_path=metrics,
    )
    store.release_claim(claim)

    assert marker.exists()
    assert not claim.path.exists()
    with pytest.raises(FileExistsError, match="detection_exact_error.json"):
        store.acquire_claim("concept-main", "exact_error")


def test_completion_marker_binds_analysis_id_to_metrics_hash(tmp_path):
    store = SecondaryArtifactStore(tmp_path)
    payload = _metrics_with_provenance()
    analysis_id = payload["analysis_id"]
    metrics = store.write_json(
        "concept-main", "comparisons", "detection_exact_error", payload
    )

    marker = store.write_completion(
        "concept-main", "exact_error", analysis_id=analysis_id, metrics_path=metrics
    )

    assert store.read_json("concept-main", "comparisons", "completion_exact_error") == {
        "analysis_id": analysis_id,
        "metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
    }
    assert marker.name == "completion_exact_error.json"


def test_completion_rejects_mismatched_ids_and_tampered_provenance(tmp_path):
    store = SecondaryArtifactStore(tmp_path)
    payload = _metrics_with_provenance()
    metrics = store.write_json(
        "concept-main", "comparisons", "detection_exact_error", payload
    )

    with pytest.raises(ValueError, match="analysis IDs"):
        store.write_completion(
            "concept-main",
            "exact_error",
            analysis_id="f" * 64,
            metrics_path=metrics,
        )

    payload["analysis_provenance"]["analysis"]["pca_dims"] = 99
    metrics.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="canonical provenance fingerprint"):
        store.write_completion(
            "concept-main",
            "exact_error",
            analysis_id=payload["analysis_id"],
            metrics_path=metrics,
        )


def test_completion_marker_is_no_clobber(tmp_path):
    store = SecondaryArtifactStore(tmp_path)
    payload = _metrics_with_provenance()
    metrics = store.write_json(
        "concept-main", "comparisons", "detection_exact_error", payload
    )
    existing = store.write_json(
        "concept-main", "comparisons", "completion_exact_error", {"legacy": True}
    )

    with pytest.raises(FileExistsError, match="completion_exact_error.json"):
        store.write_completion(
            "concept-main",
            "exact_error",
            analysis_id=payload["analysis_id"],
            metrics_path=metrics,
        )

    assert json.loads(existing.read_text()) == {"legacy": True}


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
    assert first["analysis_id"] == secondary_artifacts.analysis_fingerprint(first)
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


def test_analysis_id_is_stable_when_identical_inputs_are_relocated(tmp_path):
    first_root = tmp_path / "first"
    first_run = first_root / "runs" / "concept-main"
    examples = first_run / "examples"
    examples.mkdir(parents=True)
    (first_root / "config.json").write_text(
        '{"model_id":"test/model","model_revision":"requested"}'
    )
    (first_root / "preregistration.md").write_text("frozen protocol\n")
    (first_run / "manifest.json").write_text(
        json.dumps(
            {
                "config": {"model_id": "test/model", "model_revision": "requested"},
                "dataset": {
                    "path": "/machine/one/data/concept.jsonl",
                    "sha256": "a" * 64,
                    "manifest_path": "/machine/one/data/concept.manifest.json",
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
                    "model_revision": "requested",
                    "resolved_model_revision": "resolved-commit",
                }
            }
        )
    )
    second_root = tmp_path / "second"
    shutil.copytree(first_root, second_root)
    second_manifest = json.loads(
        (second_root / "runs" / "concept-main" / "manifest.json").read_text()
    )
    second_manifest["dataset"]["path"] = "/machine/two/data/concept.jsonl"
    second_manifest["dataset"]["manifest_path"] = (
        "/machine/two/data/concept.manifest.json"
    )
    (second_root / "runs" / "concept-main" / "manifest.json").write_text(
        json.dumps(second_manifest)
    )

    common = {
        "repo_root": Path(__file__).resolve().parents[1],
        "endpoint": "exact_error",
        "pca_dims": 32,
        "ridge_alpha": 0.001,
        "bootstrap_count": 2000,
        "permutation_method": "paired_entity_family_permutation",
        "permutation_seed": 42,
        "permutation_count": 2000,
        "fdr_family": ["detection", "intervention"],
    }
    first = secondary_artifacts.build_analysis_provenance(
        preregistration_path=first_root / "preregistration.md",
        config_path=first_root / "config.json",
        run_root=first_run,
        **common,
    )
    second = secondary_artifacts.build_analysis_provenance(
        preregistration_path=second_root / "preregistration.md",
        config_path=second_root / "config.json",
        run_root=second_root / "runs" / "concept-main",
        **common,
    )

    assert first["inputs"] != second["inputs"]
    assert first["analysis_id"] == second["analysis_id"]
