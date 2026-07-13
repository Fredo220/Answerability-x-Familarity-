import json

import numpy as np
import pytest

from trajectory_extractor import rlmf_artifacts
from trajectory_extractor.rlmf_artifacts import RLMFArtifactStore


def test_store_writes_only_under_isolated_rlmf_namespace(tmp_path):
    store = RLMFArtifactStore(tmp_path)

    jsonl = store.write_jsonl("rlmf-qwen06b-v1", "data", "rows", [{"id": 1}, {"id": 2}])
    metadata = store.write_json("rlmf-qwen06b-v1", "metadata", "config", {"ok": True})
    arrays = store.write_npz("rlmf-qwen06b-v1", "rollouts", "scores", score=np.array([0.1]))

    expected = tmp_path / "runs" / "rlmf" / "rlmf-qwen06b-v1"
    assert jsonl.parent.parent == expected
    assert metadata.parent.parent == expected
    assert arrays.parent.parent == expected
    assert jsonl.read_text().splitlines() == ['{"id": 1}', '{"id": 2}']


@pytest.mark.parametrize("study_id", ["../escape", "", ".", "study/id"])
def test_store_rejects_unsafe_study_ids(tmp_path, study_id):
    with pytest.raises(ValueError, match="study_id"):
        RLMFArtifactStore(tmp_path).write_json(study_id, "metadata", "config", {})


def test_artifact_writes_are_exclusive(tmp_path):
    store = RLMFArtifactStore(tmp_path)
    path = store.write_json("rlmf-qwen06b-v1", "metadata", "config", {"version": 1})

    with pytest.raises(FileExistsError, match="config.json"):
        store.write_json("rlmf-qwen06b-v1", "metadata", "config", {"version": 2})

    assert json.loads(path.read_text()) == {"version": 1}


def test_failed_temporary_write_is_cleaned_up(tmp_path, monkeypatch):
    store = RLMFArtifactStore(tmp_path)

    def fail_fsync(_descriptor):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(rlmf_artifacts.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        store.write_json("rlmf-qwen06b-v1", "metadata", "config", {"version": 1})

    namespace = tmp_path / "runs" / "rlmf" / "rlmf-qwen06b-v1" / "metadata"
    assert not (namespace / "config.json").exists()
    assert list(namespace.glob("*.tmp")) == []


def test_completion_binds_parent_config_and_artifact_hashes(tmp_path):
    store = RLMFArtifactStore(tmp_path)
    artifact = store.write_json("rlmf-qwen06b-v1", "metrics", "behavior", {"score": 0.5})
    marker = store.complete_endpoint(
        "rlmf-qwen06b-v1", "behavior", "a" * 64, [artifact]
    )

    verified = store.verify_endpoint("rlmf-qwen06b-v1", "behavior")

    assert marker.name == "behavior.complete.json"
    assert verified["parent_hashes"]["config"] == "a" * 64
    assert verified["artifact_hashes"] == {
        "metrics/behavior.json": rlmf_artifacts.sha256_file(artifact)
    }


def test_completion_rejects_invalid_parent_hash_and_is_immutable(tmp_path):
    store = RLMFArtifactStore(tmp_path)
    artifact = store.write_json("rlmf-qwen06b-v1", "metrics", "behavior", {"score": 0.5})

    with pytest.raises(ValueError, match="config_hash"):
        store.complete_endpoint("rlmf-qwen06b-v1", "behavior", "main", [artifact])

    store.complete_endpoint("rlmf-qwen06b-v1", "behavior", "a" * 64, [artifact])
    with pytest.raises(FileExistsError, match="behavior.complete.json"):
        store.complete_endpoint("rlmf-qwen06b-v1", "behavior", "a" * 64, [artifact])


def test_verify_endpoint_detects_tampered_bound_artifact_and_parent_hash(tmp_path):
    store = RLMFArtifactStore(tmp_path)
    artifact = store.write_json("rlmf-qwen06b-v1", "metrics", "behavior", {"score": 0.5})
    marker = store.complete_endpoint(
        "rlmf-qwen06b-v1", "behavior", "a" * 64, [artifact]
    )

    artifact.write_text('{"score": 0.9}')
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        store.verify_endpoint("rlmf-qwen06b-v1", "behavior")

    artifact.write_text('{"score": 0.5}')
    payload = json.loads(marker.read_text())
    payload["parent_hashes"]["config"] = "not-a-sha256"
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="parent_hashes"):
        store.verify_endpoint("rlmf-qwen06b-v1", "behavior")
