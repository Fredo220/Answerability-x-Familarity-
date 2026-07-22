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


def test_completed_shard_manifest_enforces_an_explicit_record_kind(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    shard = store.write_completed_shard(
        config.run_id,
        "pilot",
        "typed",
        [{"kind": "generation", "status": "completed"}],
        lineage(config),
        record_kind="generation",
    )

    assert store.verify_shard(shard.manifest_path).record_kind == "generation"
    manifest = json.loads(shard.manifest_path.read_text(encoding="utf-8"))
    manifest["record_kind"] = "../metrics"
    shard.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="record kind"):
        store.verify_shard(shard.manifest_path)


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


def test_registered_endpoint_artifact_must_be_explicitly_sealed(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    registered = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "registered-prompts",
        [{"manifest_sha256": digest("registered prompts"), "examples": []}],
        lineage(config),
    )
    unregistered = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "caller-prompts",
        [{"manifest_sha256": digest("caller prompts"), "examples": []}],
        lineage(config),
    )
    store.seal_endpoint("probe_test", [registered], parents())

    assert store.verify_endpoint_artifact("probe_test", registered.manifest_path) == registered
    with pytest.raises(ValueError, match="not registered"):
        store.verify_endpoint_artifact("probe_test", unregistered.manifest_path)
    with pytest.raises(ValueError, match="not registered"):
        store.unlock_or_resume_endpoint("probe_test", unregistered.manifest_path)


def test_endpoint_retry_resumes_the_same_still_open_unlock_receipt(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    prompts = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "registered-prompts",
        [{"manifest_sha256": digest("registered prompts"), "examples": []}],
        lineage(config),
    )
    store.seal_endpoint("probe_test", [prompts], parents())
    assert store.endpoint_state("probe_test", prompts.manifest_path) == "sealed"

    first = store.unlock_or_resume_endpoint("probe_test", prompts.manifest_path)
    resumed = store.unlock_or_resume_endpoint("probe_test", prompts.manifest_path)

    assert resumed == first
    assert resumed.state == "unlocked_once"
    assert store.endpoint_state("probe_test", prompts.manifest_path) == "unlocked_once"
    endpoint_dir = (
        tmp_path
        / "runs"
        / "familiarity_answerability"
        / config.run_id
        / "endpoints"
        / "probe_test"
    )
    assert endpoint_dir.is_dir()
    assert sorted(path.name for path in endpoint_dir.iterdir()) == ["sealed.json", "unlocked_once.json"]

    metrics = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "metrics",
        [{"kind": "metrics", "metric": "complete"}],
        lineage(config),
        record_kind="metrics",
    )
    store.mark_evaluated(first, metrics.data_path)
    assert store.endpoint_state("probe_test", prompts.manifest_path) == "evaluated"
    with pytest.raises(ValueError, match="already evaluated"):
        store.unlock_or_resume_endpoint("probe_test", prompts.manifest_path)


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
    store._ensure_directory(shard_dir)
    shard_directory_inode = os.stat(shard_dir).st_ino
    real_fsync_descriptor = fa_artifacts._fsync_descriptor
    failed = False

    def fail_once(descriptor):
        nonlocal failed
        if os.fstat(descriptor).st_ino == shard_directory_inode and not failed:
            failed = True
            raise OSError("injected directory fsync failure")
        return real_fsync_descriptor(descriptor)

    monkeypatch.setattr(fa_artifacts, "_fsync_descriptor", fail_once)
    with pytest.raises(OSError, match="injected directory fsync failure"):
        store.write_completed_shard(
            config.run_id, "pilot", "0001", [{"example_id": "a"}], lineage(config)
        )

    assert failed
    assert not (shard_dir / "0001.jsonl").exists()
    assert not (shard_dir / "0001.jsonl.manifest.json").exists()
    assert list(shard_dir.glob("*.tmp")) == []


def test_completed_shard_fails_closed_when_a_temp_leaf_is_replaced_with_a_regular_file(
    tmp_path, config, monkeypatch
):
    store = FAArtifactStore(tmp_path)
    real_link = os.link
    replaced = False

    def replace_temp_then_link(source, destination, *, src_dir_fd=None, dst_dir_fd=None, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            os.unlink(source, dir_fd=src_dir_fd)
            descriptor = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=src_dir_fd,
            )
            try:
                os.write(descriptor, b"attacker-controlled")
            finally:
                os.close(descriptor)
        return real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            **kwargs,
        )

    monkeypatch.setattr(fa_artifacts.os, "link", replace_temp_then_link)

    with pytest.raises(ValueError, match="publication"):
        store.write_completed_shard(
            config.run_id, "pilot", "0001", [{"example_id": "a"}], lineage(config)
        )

    assert replaced


def test_completed_shard_fails_closed_when_a_temp_leaf_is_replaced_with_a_symlink(
    tmp_path, config, monkeypatch
):
    store = FAArtifactStore(tmp_path)
    outside = tmp_path / "attacker-controlled"
    outside.write_bytes(b"attacker-controlled")
    real_link = os.link
    replaced = False

    def replace_temp_then_link(source, destination, *, src_dir_fd=None, dst_dir_fd=None, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            os.unlink(source, dir_fd=src_dir_fd)
            os.symlink(outside, source, dir_fd=src_dir_fd)
        return real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            **kwargs,
        )

    monkeypatch.setattr(fa_artifacts.os, "link", replace_temp_then_link)

    with pytest.raises(ValueError, match="publication"):
        store.write_completed_shard(
            config.run_id, "pilot", "0001", [{"example_id": "a"}], lineage(config)
        )

    assert replaced


def test_interrupted_write_quarantines_a_temp_leaf_replaced_during_cleanup(
    tmp_path, config, monkeypatch
):
    store = FAArtifactStore(tmp_path)
    real_rename = os.rename
    real_unlink = os.unlink
    real_stat = os.stat
    replacement_identity = None
    replaced = False

    def partial_then_fail(descriptor, payload):
        os.write(descriptor, payload[:1])
        raise OSError("injected write failure")

    def replace_before_quarantine(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal replaced, replacement_identity
        if not replaced and source.endswith(".tmp"):
            replaced = True
            real_unlink(source, dir_fd=src_dir_fd)
            descriptor = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=src_dir_fd,
            )
            try:
                os.write(descriptor, b"replacement")
            finally:
                os.close(descriptor)
            replacement_identity = real_stat(source, dir_fd=src_dir_fd, follow_symlinks=False)
        return real_rename(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    def reject_replacement_unlink(path, *, dir_fd=None):
        if replacement_identity is not None:
            try:
                candidate = real_stat(path, dir_fd=dir_fd, follow_symlinks=False)
            except FileNotFoundError:
                candidate = None
            if candidate is not None and (
                candidate.st_dev,
                candidate.st_ino,
            ) == (
                replacement_identity.st_dev,
                replacement_identity.st_ino,
            ):
                raise AssertionError("cleanup attempted to unlink an unverified replacement")
        return real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(fa_artifacts, "_write_all", partial_then_fail)
    monkeypatch.setattr(fa_artifacts.os, "rename", replace_before_quarantine)
    monkeypatch.setattr(fa_artifacts.os, "unlink", reject_replacement_unlink)

    with pytest.raises(OSError, match="injected write failure"):
        store.write_completed_shard(
            config.run_id, "pilot", "0001", [{"example_id": "a"}], lineage(config)
        )

    assert replaced
    assert replacement_identity is not None


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
        [{"kind": "metrics", "metric": "wrong endpoint"}],
        lineage(config),
        record_kind="metrics",
    )
    with pytest.raises(ValueError, match="matching endpoint namespace"):
        store.mark_evaluated(receipt, behavior_metrics.data_path)

    probe_metrics = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "metrics",
        [{"kind": "metrics", "metric": "complete"}],
        lineage(config),
        record_kind="metrics",
    )
    evaluated = store.mark_evaluated(receipt, probe_metrics.data_path)
    closed = store.close_endpoint("probe_test")

    assert json.loads(evaluated.read_text(encoding="utf-8"))["state"] == "evaluated"
    assert json.loads(closed.read_text(encoding="utf-8"))["state"] == "closed"
    with pytest.raises(ValueError, match="already closed"):
        store.close_endpoint("probe_test")


def test_endpoint_rejects_generation_shard_as_metrics(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    selection = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "selection",
        [{"example_id": "probe"}],
        lineage(config),
    )
    store.seal_endpoint("probe_test", [selection], parents())
    receipt = store.unlock_endpoint(
        "probe_test", parents()["preregistration"], parents()["selection_manifest"]
    )
    generation = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "generation",
        [{"kind": "generation", "example_id": "probe", "response": "UNKNOWN"}],
        lineage(config),
        record_kind="generation",
    )

    with pytest.raises(ValueError, match="metrics artifact"):
        store.mark_evaluated(receipt, generation.data_path)

    sidecar = json.loads(generation.manifest_path.read_text(encoding="utf-8"))
    sidecar["record_kind"] = "metrics"
    generation.manifest_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="row kind"):
        store.mark_evaluated(receipt, generation.data_path)


def test_endpoint_rejects_empty_metrics_shard(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    selection = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "selection",
        [{"example_id": "probe"}],
        lineage(config),
    )
    store.seal_endpoint("probe_test", [selection], parents())
    receipt = store.unlock_endpoint(
        "probe_test", parents()["preregistration"], parents()["selection_manifest"]
    )
    empty = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "empty-metrics",
        [],
        lineage(config),
        record_kind="metrics",
    )

    with pytest.raises(ValueError, match="at least one row"):
        store.mark_evaluated(receipt, empty.data_path)


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
        config.run_id,
        "probe_test",
        "metrics",
        [{"kind": "metrics", "metric": "complete"}],
        lineage(config),
        record_kind="metrics",
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

    with pytest.raises(ValueError, match="metrics artifact"):
        store.mark_evaluated(receipt, selection.data_path)


def _evaluated_endpoint(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    selection = store.write_completed_shard(
        config.run_id, "probe_test", "selection", [{"example_id": "probe"}], lineage(config)
    )
    store.seal_endpoint("probe_test", [selection], parents())
    receipt = store.unlock_endpoint(
        "probe_test", parents()["preregistration"], parents()["selection_manifest"]
    )
    metrics = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "metrics",
        [{"kind": "metrics", "metric": "complete"}],
        lineage(config),
        record_kind="metrics",
    )
    evaluated = store.mark_evaluated(receipt, metrics.data_path)
    return store, selection, metrics, receipt, evaluated


def test_close_revalidates_every_sealed_input(tmp_path, config):
    store, selection, _, _, _ = _evaluated_endpoint(tmp_path, config)
    selection.data_path.write_bytes(b'{"example_id":"tampered"}\n')

    with pytest.raises(ValueError, match="hash mismatch"):
        store.close_endpoint("probe_test")


def test_close_requires_exact_evaluated_schema_and_active_matching_lease(tmp_path, config):
    store, _, _, receipt, evaluated = _evaluated_endpoint(tmp_path, config)
    record = json.loads(evaluated.read_text(encoding="utf-8"))
    record["unexpected"] = True
    evaluated.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid schema"):
        store.close_endpoint("probe_test")

    record.pop("unexpected")
    record["lease_id"] = "0" * 32
    evaluated.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="lease"):
        store.close_endpoint("probe_test")

    record["lease_id"] = receipt.lease_id
    evaluated.write_text(json.dumps(record), encoding="utf-8")
    unlocked = evaluated.with_name("unlocked_once.json")
    lease = json.loads(unlocked.read_text(encoding="utf-8"))
    lease["lease_id"] = "1" * 32
    unlocked.write_text(json.dumps(lease), encoding="utf-8")
    with pytest.raises(ValueError, match="lease"):
        store.close_endpoint("probe_test")


def test_close_rejects_evaluated_metrics_bound_to_another_run(tmp_path, config):
    store, _, _, receipt, evaluated = _evaluated_endpoint(tmp_path, config)
    other = store.write_completed_shard(
        "other-run",
        "probe_test",
        "metrics",
        [{"kind": "metrics", "metric": "other"}],
        lineage(config),
        record_kind="metrics",
    )
    record = json.loads(evaluated.read_text(encoding="utf-8"))
    record["metrics_manifest_path"] = str(other.manifest_path.relative_to(tmp_path))
    record["metrics_sha256"] = other.sha256
    record["lease_id"] = receipt.lease_id
    evaluated.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="endpoint run"):
        store.close_endpoint("probe_test")


def test_close_rejects_an_active_lease_detached_from_sealed_parents(tmp_path, config):
    store, _, _, _, evaluated = _evaluated_endpoint(tmp_path, config)
    unlocked = evaluated.with_name("unlocked_once.json")
    record = json.loads(unlocked.read_text(encoding="utf-8"))
    record["preregistration_hash"] = digest("different preregistration")
    unlocked.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="sealed endpoint"):
        store.close_endpoint("probe_test")


def test_close_rejects_evaluated_marker_for_another_run(tmp_path, config):
    store, _, _, _, evaluated = _evaluated_endpoint(tmp_path, config)
    record = json.loads(evaluated.read_text(encoding="utf-8"))
    record["run_id"] = "other-run"
    evaluated.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="another run"):
        store.close_endpoint("probe_test")


def test_verify_rejects_an_intermediate_symlink(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    sealed = store.write_completed_shard(
        config.run_id, "pilot", "0001", [{"example_id": "a"}], lineage(config)
    )
    shard_root = sealed.manifest_path.parents[2]
    outside = tmp_path / "outside"
    outside.mkdir()
    shard_root.rename(tmp_path / "parked-shards")
    shard_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="real directory"):
        store.verify_shard(sealed.manifest_path)


def test_descriptor_read_fails_closed_when_an_intermediate_directory_is_replaced(
    tmp_path, config, monkeypatch
):
    store = FAArtifactStore(tmp_path)
    sealed = store.write_completed_shard(
        config.run_id, "pilot", "0001", [{"example_id": "a"}], lineage(config)
    )
    runs = tmp_path / "runs"
    parked = tmp_path / "parked-runs"
    outside = tmp_path / "outside"
    outside.mkdir()
    replaced = False

    def replace_after_safe_traversal(path):
        nonlocal replaced
        if path == sealed.manifest_path and not replaced:
            replaced = True
            runs.rename(parked)
            runs.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(fa_artifacts, "_before_final_open", replace_after_safe_traversal)
    with pytest.raises(ValueError, match="real directory"):
        store.verify_shard(sealed.manifest_path)
    assert replaced
