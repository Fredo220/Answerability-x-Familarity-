import json
from pathlib import Path

import numpy as np
import pytest

from trajectory_extractor import rlmf_artifacts
from trajectory_extractor.rlmf_artifacts import RLMFArtifactStore
from trajectory_extractor.rlmf_types import RLMFConfig


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def confirmatory_config():
    return RLMFConfig.from_json(CONFIGS / "rlmf_qwen06b_confirmatory.json")


def smoke_config():
    return RLMFConfig.from_json(CONFIGS / "rlmf_qwen06b_smoke.json")


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
        "rlmf-qwen06b-v1", "behavior", confirmatory_config(), [artifact]
    )

    verified = store.verify_endpoint("rlmf-qwen06b-v1", "behavior")

    assert marker.name == "behavior.complete.json"
    assert verified["parent_hashes"]["config"] == confirmatory_config().config_hash
    assert (tmp_path / "runs" / "rlmf" / "rlmf-qwen06b-v1" / "metadata" / "config.json").read_bytes() == confirmatory_config().canonical_bytes
    assert verified["artifact_hashes"] == {
        "metrics/behavior.json": rlmf_artifacts.sha256_file(artifact)
    }


def test_completion_requires_a_config_record_and_is_immutable(tmp_path):
    store = RLMFArtifactStore(tmp_path)
    artifact = store.write_json("rlmf-qwen06b-v1", "metrics", "behavior", {"score": 0.5})

    with pytest.raises(ValueError, match="config"):
        store.complete_endpoint("rlmf-qwen06b-v1", "behavior", "main", [artifact])

    store.complete_endpoint("rlmf-qwen06b-v1", "behavior", confirmatory_config(), [artifact])
    with pytest.raises(FileExistsError, match="behavior.complete.json"):
        store.complete_endpoint("rlmf-qwen06b-v1", "behavior", confirmatory_config(), [artifact])


def test_verify_endpoint_detects_tampered_bound_artifact_and_parent_hash(tmp_path):
    store = RLMFArtifactStore(tmp_path)
    artifact = store.write_json("rlmf-qwen06b-v1", "metrics", "behavior", {"score": 0.5})
    marker = store.complete_endpoint(
        "rlmf-qwen06b-v1", "behavior", confirmatory_config(), [artifact]
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


def test_completion_rejects_mismatched_existing_canonical_config_artifact(tmp_path):
    store = RLMFArtifactStore(tmp_path)
    config_path = tmp_path / "runs" / "rlmf" / "rlmf-qwen06b-v1" / "metadata" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(smoke_config().canonical_bytes)
    artifact = store.write_json("rlmf-qwen06b-v1", "metrics", "behavior", {"score": 0.5})

    with pytest.raises(ValueError, match="config artifact"):
        store.complete_endpoint("rlmf-qwen06b-v1", "behavior", confirmatory_config(), [artifact])


def test_verify_endpoint_recomputes_config_hash_and_rejects_valid_length_rewrite(tmp_path):
    store = RLMFArtifactStore(tmp_path)
    artifact = store.write_json("rlmf-qwen06b-v1", "metrics", "behavior", {"score": 0.5})
    marker = store.complete_endpoint(
        "rlmf-qwen06b-v1", "behavior", confirmatory_config(), [artifact]
    )
    payload = json.loads(marker.read_text())
    payload["parent_hashes"]["config"] = "a" * 64
    marker.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="configuration hash mismatch"):
        store.verify_endpoint("rlmf-qwen06b-v1", "behavior")


def test_verify_endpoint_rejects_rewritten_config_artifact(tmp_path):
    store = RLMFArtifactStore(tmp_path)
    artifact = store.write_json("rlmf-qwen06b-v1", "metrics", "behavior", {"score": 0.5})
    store.complete_endpoint("rlmf-qwen06b-v1", "behavior", confirmatory_config(), [artifact])
    config_path = tmp_path / "runs" / "rlmf" / "rlmf-qwen06b-v1" / "metadata" / "config.json"
    config_path.write_bytes(smoke_config().canonical_bytes)

    with pytest.raises(ValueError, match="study_id"):
        store.verify_endpoint("rlmf-qwen06b-v1", "behavior")


def test_verify_endpoint_rejects_symlinked_config_section(tmp_path):
    store = RLMFArtifactStore(tmp_path)
    artifact = store.write_json("rlmf-qwen06b-v1", "metrics", "behavior", {"score": 0.5})
    store.complete_endpoint("rlmf-qwen06b-v1", "behavior", confirmatory_config(), [artifact])
    namespace = tmp_path / "runs" / "rlmf" / "rlmf-qwen06b-v1"
    metadata = namespace / "metadata"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_bytes(confirmatory_config().canonical_bytes)
    metadata.rename(tmp_path / "saved-metadata")
    metadata.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        store.verify_endpoint("rlmf-qwen06b-v1", "behavior")


@pytest.mark.parametrize("component", ["runs", "rlmf", "study", "section"])
def test_store_rejects_symlinked_namespace_components(tmp_path, component):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    runs = root / "runs"
    rlmf = runs / "rlmf"
    study = rlmf / "rlmf-qwen06b-v1"
    section = study / "metadata"
    target = {"runs": runs, "rlmf": rlmf, "study": study, "section": section}[component]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        RLMFArtifactStore(root).write_json("rlmf-qwen06b-v1", "metadata", "config", {})

    assert list(outside.iterdir()) == []
