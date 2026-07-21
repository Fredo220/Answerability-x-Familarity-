import hashlib
import json
import os
from pathlib import Path

import pytest

import trajectory_extractor.fa_artifacts as fa_artifacts
from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_config import FAConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "familiarity_answerability_qwen06b_smoke.json"


@pytest.fixture
def config():
    return FAConfig.from_json(SMOKE_CONFIG)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def lineage(config: FAConfig) -> dict[str, str]:
    return {"config_sha256": config.config_hash, "source_manifest_sha256": digest("source")}


def parents() -> dict[str, str]:
    return {
        "preregistration": digest("preregistration"),
        "selection_manifest": digest("probe selection"),
    }


def test_completed_shard_is_no_clobber_and_hash_verified(tmp_path, config):
    store = FAArtifactStore(tmp_path)

    sealed = store.write_completed_shard(
        config.run_id, "pilot", "0001", [{"example_id": "a"}], lineage(config)
    )

    assert store.verify_shard(sealed.manifest_path).sha256 == sealed.sha256
    with pytest.raises(FileExistsError):
        store.write_completed_shard(
            config.run_id, "pilot", "0001", [{"example_id": "b"}], lineage(config)
        )


def test_probe_endpoint_rejects_behavior_parent_and_second_unlock(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    probe_shard = store.write_completed_shard(
        config.run_id, "probe_test", "selection", [{"example_id": "probe"}], lineage(config)
    )
    store.seal_endpoint("probe_test", [probe_shard], parents())

    with pytest.raises(ValueError, match="probe selection"):
        store.unlock_endpoint(
            "probe_test", parents()["preregistration"], digest("behavior selection")
        )
    receipt = store.unlock_endpoint(
        "probe_test", parents()["preregistration"], parents()["selection_manifest"]
    )
    with pytest.raises(ValueError, match="already unlocked"):
        store.unlock_endpoint(
            "probe_test", parents()["preregistration"], parents()["selection_manifest"]
        )

    assert receipt.state == "unlocked_once"


def test_tampered_shard_fails_verification_and_cannot_seal(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    shard = store.write_completed_shard(
        config.run_id, "probe_test", "selection", [{"example_id": "probe"}], lineage(config)
    )
    shard.data_path.write_bytes(b'{"example_id":"tampered"}\n')

    with pytest.raises(ValueError, match="hash mismatch"):
        store.verify_shard(shard.manifest_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        store.seal_endpoint("probe_test", [shard], parents())


def test_interrupted_shard_write_leaves_no_completed_marker_or_temporary_file(
    tmp_path, config, monkeypatch
):
    store = FAArtifactStore(tmp_path)

    def partial_then_fail(descriptor, payload):
        os.write(descriptor, payload[:1])
        raise OSError("injected write failure")

    monkeypatch.setattr(fa_artifacts, "_write_all", partial_then_fail)
    with pytest.raises(OSError, match="injected write failure"):
        store.write_completed_shard(
            config.run_id, "pilot", "0001", [{"example_id": "a"}], lineage(config)
        )

    shard_dir = tmp_path / "runs" / "familiarity_answerability" / config.run_id / "shards" / "pilot"
    assert not (shard_dir / "0001.jsonl").exists()
    assert not (shard_dir / "0001.jsonl.manifest.json").exists()
    assert list(shard_dir.glob("*.tmp")) == []


def test_fsync_failure_removes_published_data_before_a_completion_marker(
    tmp_path, config, monkeypatch
):
    store = FAArtifactStore(tmp_path)
    shard_dir = tmp_path / "runs" / "familiarity_answerability" / config.run_id / "shards" / "pilot"
    real_fsync_directory = fa_artifacts._fsync_directory
    failed = False

    def fail_once(path):
        nonlocal failed
        if path == shard_dir and not failed:
            failed = True
            raise OSError("injected directory fsync failure")
        return real_fsync_directory(path)

    monkeypatch.setattr(fa_artifacts, "_fsync_directory", fail_once)
    with pytest.raises(OSError, match="injected directory fsync failure"):
        store.write_completed_shard(
            config.run_id, "pilot", "0001", [{"example_id": "a"}], lineage(config)
        )

    assert failed
    assert not (shard_dir / "0001.jsonl").exists()
    assert not (shard_dir / "0001.jsonl.manifest.json").exists()
    assert list(shard_dir.glob("*.tmp")) == []


@pytest.mark.parametrize("unsafe_id", ["../escape", "a/b", ".", "..", "with space"])
def test_shards_reject_path_traversal_and_lossy_identifiers(tmp_path, config, unsafe_id):
    store = FAArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="safe identifier"):
        store.write_completed_shard(
            unsafe_id, "pilot", "0001", [{"example_id": "a"}], lineage(config)
        )
    with pytest.raises(ValueError, match="safe identifier"):
        store.write_completed_shard(
            config.run_id, "pilot", unsafe_id, [{"example_id": "a"}], lineage(config)
        )


def test_store_rejects_symlinked_artifact_directories(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "runs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="real directory"):
        store.write_completed_shard(
            config.run_id, "pilot", "0001", [{"example_id": "a"}], lineage(config)
        )


def test_endpoint_rejects_artifacts_from_another_test_namespace(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    behavior = store.write_completed_shard(
        config.run_id,
        "behavior_test",
        "selection",
        [{"example_id": "behavior"}],
        lineage(config),
    )

    with pytest.raises(ValueError, match="matching namespace"):
        store.seal_endpoint("probe_test", [behavior], parents())


def test_unlock_rechecks_a_sealed_selection_sidecar_before_opening(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    selection = store.write_completed_shard(
        config.run_id, "probe_test", "selection", [{"example_id": "probe"}], lineage(config)
    )
    store.seal_endpoint("probe_test", [selection], parents())
    selection.data_path.write_bytes(b'{"example_id":"tampered"}\n')

    with pytest.raises(ValueError, match="hash mismatch"):
        store.unlock_endpoint(
            "probe_test", parents()["preregistration"], parents()["selection_manifest"]
        )


def test_resume_scan_returns_only_verified_sidecars_and_fails_closed_on_corruption(
    tmp_path, config
):
    store = FAArtifactStore(tmp_path)
    verified = store.write_completed_shard(
        config.run_id, "pilot", "verified", [{"example_id": "a"}], lineage(config)
    )
    orphan = verified.data_path.with_name("orphan.jsonl")
    orphan.write_bytes(b'{"example_id":"orphan"}\n')

    assert store.resume_verified_shards(config.run_id, "pilot") == (verified,)

    corrupt = store.write_completed_shard(
        config.run_id, "pilot", "corrupt", [{"example_id": "b"}], lineage(config)
    )
    corrupt.data_path.write_bytes(b'{"example_id":"tampered"}\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        store.resume_verified_shards(config.run_id, "pilot")


def test_endpoint_lifecycle_requires_verified_matching_metrics_and_closes_once(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    selection = store.write_completed_shard(
        config.run_id, "probe_test", "selection", [{"example_id": "probe"}], lineage(config)
    )
    store.seal_endpoint("probe_test", [selection], parents())
    receipt = store.unlock_endpoint(
        "probe_test", parents()["preregistration"], parents()["selection_manifest"]
    )
    behavior_metrics = store.write_completed_shard(
        config.run_id,
        "behavior_test",
        "metrics",
        [{"metric": "wrong endpoint"}],
        lineage(config),
    )
    with pytest.raises(ValueError, match="matching endpoint namespace"):
        store.mark_evaluated(receipt, behavior_metrics.data_path)

    probe_metrics = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "metrics",
        [{"metric": "complete"}],
        lineage(config),
    )
    evaluated = store.mark_evaluated(receipt, probe_metrics.data_path)
    closed = store.close_endpoint("probe_test")

    assert json.loads(evaluated.read_text(encoding="utf-8"))["state"] == "evaluated"
    assert json.loads(closed.read_text(encoding="utf-8"))["state"] == "closed"
    with pytest.raises(ValueError, match="already closed"):
        store.close_endpoint("probe_test")


def test_mark_evaluated_rejects_a_tampered_metrics_manifest(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    selection = store.write_completed_shard(
        config.run_id, "probe_test", "selection", [{"example_id": "probe"}], lineage(config)
    )
    store.seal_endpoint("probe_test", [selection], parents())
    receipt = store.unlock_endpoint(
        "probe_test", parents()["preregistration"], parents()["selection_manifest"]
    )
    metrics = store.write_completed_shard(
        config.run_id, "probe_test", "metrics", [{"metric": "complete"}], lineage(config)
    )
    manifest = json.loads(metrics.manifest_path.read_text(encoding="utf-8"))
    manifest["data_file"] = "other.jsonl"
    metrics.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="data path is invalid"):
        store.mark_evaluated(receipt, metrics.data_path)
    assert not (
        tmp_path
        / "runs"
        / "familiarity_answerability"
        / config.run_id
        / "endpoints"
        / "probe_test"
        / "evaluated.json"
    ).exists()


def test_mark_evaluated_rejects_reusing_a_sealed_input_as_metrics(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    selection = store.write_completed_shard(
        config.run_id, "probe_test", "selection", [{"example_id": "probe"}], lineage(config)
    )
    store.seal_endpoint("probe_test", [selection], parents())
    receipt = store.unlock_endpoint(
        "probe_test", parents()["preregistration"], parents()["selection_manifest"]
    )

    with pytest.raises(ValueError, match="sealed input"):
        store.mark_evaluated(receipt, selection.data_path)
