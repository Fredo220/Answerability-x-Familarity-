from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from trajectory_extractor.fa_artifacts import FAArtifactStore
import trajectory_extractor.fa_colab_checkpoint as checkpoint_module
from trajectory_extractor.fa_colab_checkpoint import ColabSplitCheckpointStore


RUN_ID = "confirmatory-v1"
SPLIT = "mechanism_train"


def checkpoint_store(tmp_path: Path) -> ColabSplitCheckpointStore:
    return ColabSplitCheckpointStore(
        repo_root=tmp_path / "repo",
        checkpoint_root=tmp_path / "drive",
        scratch_root=tmp_path / "scratch",
        run_id=RUN_ID,
        git_commit="a" * 40,
        config_sha256="b" * 64,
    )


def write_completion(
    checkpoint: ColabSplitCheckpointStore,
    *,
    shard_id: str,
    status: str,
) -> Path:
    shard = checkpoint.store.write_completed_shard(
        RUN_ID,
        SPLIT,
        shard_id,
        [
            {
                "kind": "screening_completion",
                "schema_version": 1,
                "status": status,
            }
        ],
        {"config_sha256": "b" * 64},
        record_kind="screening_completion",
    )
    return shard.manifest_path


def write_screened_match(checkpoint: ColabSplitCheckpointStore) -> Path:
    shard = checkpoint.store.write_completed_shard(
        RUN_ID,
        SPLIT,
        "screened-matches-test",
        [
            {
                "kind": "screened_match",
                "schema_version": 1,
                "pair_id": "pair-1",
            }
        ],
        {"config_sha256": "b" * 64},
        record_kind="screened_match",
    )
    return shard.manifest_path


def write_protected_metrics(checkpoint: ColabSplitCheckpointStore) -> Path:
    shard = checkpoint.store.write_completed_shard(
        RUN_ID,
        SPLIT,
        "protected-metrics-test",
        [{"kind": "metrics", "schema_version": 1, "value": 0.5}],
        {"config_sha256": "b" * 64},
        record_kind="metrics",
    )
    return shard.manifest_path


def clear_local_artifacts(checkpoint: ColabSplitCheckpointStore) -> None:
    shutil.rmtree(
        checkpoint.repo_root / "runs",
        ignore_errors=True,
    )


def test_completion_checkpoint_restores_after_runtime_loss(tmp_path):
    checkpoint = checkpoint_store(tmp_path)
    manifest = write_completion(
        checkpoint,
        shard_id="confirmatory-mechanism_train-screening-v1",
        status="completed",
    )

    metadata = checkpoint.checkpoint_split(SPLIT, "completion")
    assert metadata.is_file()
    assert (checkpoint.checkpoint_root / f"screening-{SPLIT}-LATEST.json").is_file()
    clear_local_artifacts(checkpoint)

    assert checkpoint.restore_split_checkpoint(SPLIT) is True
    assert checkpoint.successful_completion_manifest(SPLIT) == manifest


def test_local_shards_cannot_be_reused_by_a_different_git_commit(tmp_path):
    checkpoint = checkpoint_store(tmp_path)
    write_completion(
        checkpoint,
        shard_id="confirmatory-mechanism_train-screening-v1",
        status="completed",
    )

    with pytest.raises(ValueError, match="local run execution identity mismatch"):
        ColabSplitCheckpointStore(
            repo_root=checkpoint.repo_root,
            checkpoint_root=checkpoint.checkpoint_root / ("c" * 40),
            scratch_root=checkpoint.scratch_root,
            run_id=RUN_ID,
            git_commit="c" * 40,
            config_sha256="b" * 64,
        )


def test_legacy_local_shards_without_execution_identity_are_rejected(tmp_path):
    checkpoint = checkpoint_store(tmp_path)
    write_completion(
        checkpoint,
        shard_id="confirmatory-mechanism_train-screening-v1",
        status="completed",
    )
    (
        checkpoint.repo_root
        / "runs"
        / "familiarity_answerability"
        / RUN_ID
        / ".colab-screening-execution.json"
    ).unlink(missing_ok=True)

    with pytest.raises(ValueError, match="local run execution identity is missing"):
        checkpoint_store(tmp_path)


def test_restore_uses_highest_content_addressed_stage_when_pointer_is_corrupt(
    tmp_path,
):
    checkpoint = checkpoint_store(tmp_path)
    write_completion(
        checkpoint,
        shard_id="confirmatory-mechanism_train-screening-v1",
        status="completed",
    )
    completion_metadata = checkpoint.checkpoint_split(SPLIT, "completion")
    write_screened_match(checkpoint)
    screened_metadata = checkpoint.checkpoint_split(SPLIT, "screened")
    assert completion_metadata != screened_metadata
    pointer = checkpoint.checkpoint_root / f"screening-{SPLIT}-LATEST.json"
    pointer.write_text("interrupted pointer", encoding="utf-8")
    clear_local_artifacts(checkpoint)

    assert checkpoint.restore_split_checkpoint(SPLIT) is True
    assert checkpoint.unique_manifest(SPLIT, "screened_match") is not None


def test_failure_checkpoint_is_preserved_but_not_resumable_as_success(tmp_path):
    checkpoint = checkpoint_store(tmp_path)
    write_completion(
        checkpoint,
        shard_id="confirmatory-mechanism_train-screening-v1",
        status="infrastructure_failure",
    )

    checkpoint.checkpoint_split(SPLIT, "failure")
    clear_local_artifacts(checkpoint)

    assert checkpoint.restore_split_checkpoint(SPLIT) is False
    assert checkpoint.next_screening_shard_id(SPLIT).endswith("-retry-01")
    assert len(checkpoint.verified_split_shards(SPLIT)) == 1


def test_restore_rejects_a_tampered_checkpoint_archive(tmp_path):
    checkpoint = checkpoint_store(tmp_path)
    write_completion(
        checkpoint,
        shard_id="confirmatory-mechanism_train-screening-v1",
        status="completed",
    )
    metadata_path = checkpoint.checkpoint_split(SPLIT, "completion")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    archive = checkpoint.checkpoint_root / metadata["archive_file"]
    archive.write_bytes(archive.read_bytes() + b"tamper")
    clear_local_artifacts(checkpoint)

    with pytest.raises(ValueError, match="archive hash mismatch"):
        checkpoint.restore_split_checkpoint(SPLIT)


def test_screening_checkpoint_excludes_later_protected_shards(tmp_path):
    checkpoint = checkpoint_store(tmp_path)
    write_completion(
        checkpoint,
        shard_id="confirmatory-mechanism_train-screening-v1",
        status="completed",
    )
    write_screened_match(checkpoint)
    protected = write_protected_metrics(checkpoint)

    metadata_path = checkpoint.checkpoint_split(SPLIT, "screened")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    archive = checkpoint.checkpoint_root / metadata["archive_file"]
    with zipfile.ZipFile(archive) as bundle:
        members = set(bundle.namelist())

    assert str(protected.relative_to(checkpoint.repo_root)) not in members
    assert all("protected-metrics-test" not in member for member in members)


def test_restore_rejects_self_consistent_archive_member_outside_split(tmp_path):
    checkpoint = checkpoint_store(tmp_path)
    write_completion(
        checkpoint,
        shard_id="confirmatory-mechanism_train-screening-v1",
        status="completed",
    )
    metadata_path = checkpoint.checkpoint_split(SPLIT, "completion")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    original_archive = checkpoint.checkpoint_root / metadata["archive_file"]
    malicious_archive = checkpoint.scratch_root / "malicious.zip"
    payload = b"not allowed"
    with zipfile.ZipFile(original_archive) as source, zipfile.ZipFile(
        malicious_archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as destination:
        for member in source.infolist():
            destination.writestr(member, source.read(member.filename))
        destination.writestr("src/trajectory_extractor/owned.py", payload)
    archive_sha256 = checkpoint_module._sha256_file(malicious_archive)
    stored_archive = checkpoint.checkpoint_root / f"malicious-{archive_sha256[:16]}.zip"
    shutil.copy2(malicious_archive, stored_archive)
    metadata["archive_file"] = stored_archive.name
    metadata["archive_sha256"] = archive_sha256
    metadata["members"]["src/trajectory_extractor/owned.py"] = hashlib.sha256(
        payload
    ).hexdigest()
    metadata_bytes = checkpoint_module._canonical_bytes(metadata)
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    malicious_metadata = checkpoint.checkpoint_root / (
        f"screening-{SPLIT}-completion-{archive_sha256[:16]}-"
        f"{metadata_sha256[:16]}.checkpoint.json"
    )
    malicious_metadata.write_bytes(metadata_bytes)
    metadata_path.unlink()
    clear_local_artifacts(checkpoint)

    with pytest.raises(ValueError, match="outside the split"):
        checkpoint.restore_split_checkpoint(SPLIT)
    assert not (checkpoint.repo_root / "src/trajectory_extractor/owned.py").exists()


def test_restore_detects_manifest_lineage_tampering_with_unchanged_data(tmp_path):
    checkpoint = checkpoint_store(tmp_path)
    manifest = write_completion(
        checkpoint,
        shard_id="confirmatory-mechanism_train-screening-v1",
        status="completed",
    )
    checkpoint.checkpoint_split(SPLIT, "completion")
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["lineage"]["tampered"] = True
    manifest.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differs from its checkpoint"):
        checkpoint.restore_split_checkpoint(SPLIT)


def test_restore_rejects_self_consistent_extra_file_inside_split(tmp_path):
    checkpoint = checkpoint_store(tmp_path)
    write_completion(
        checkpoint,
        shard_id="confirmatory-mechanism_train-screening-v1",
        status="completed",
    )
    metadata_path = checkpoint.checkpoint_split(SPLIT, "completion")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    original_archive = checkpoint.checkpoint_root / metadata["archive_file"]
    extra_name = (
        f"runs/familiarity_answerability/{RUN_ID}/shards/{SPLIT}/extra.bin"
    )
    payload = b"extra"
    malicious_archive = checkpoint.scratch_root / "extra-file.zip"
    with zipfile.ZipFile(original_archive) as source, zipfile.ZipFile(
        malicious_archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as destination:
        for member in source.infolist():
            destination.writestr(member, source.read(member.filename))
        destination.writestr(extra_name, payload)
    archive_sha256 = checkpoint_module._sha256_file(malicious_archive)
    stored_archive = checkpoint.checkpoint_root / f"extra-{archive_sha256[:16]}.zip"
    shutil.copy2(malicious_archive, stored_archive)
    metadata["archive_file"] = stored_archive.name
    metadata["archive_sha256"] = archive_sha256
    metadata["members"][extra_name] = hashlib.sha256(payload).hexdigest()
    metadata_bytes = checkpoint_module._canonical_bytes(metadata)
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    malicious_metadata = checkpoint.checkpoint_root / (
        f"screening-{SPLIT}-completion-{archive_sha256[:16]}-"
        f"{metadata_sha256[:16]}.checkpoint.json"
    )
    malicious_metadata.write_bytes(metadata_bytes)
    metadata_path.unlink()
    clear_local_artifacts(checkpoint)

    with pytest.raises(ValueError, match="non-shard files"):
        checkpoint.restore_split_checkpoint(SPLIT)
