from __future__ import annotations

import json
import shutil
from pathlib import Path

from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_colab_snapshot import VerifiedColabSnapshotStore


def snapshot_store(tmp_path: Path) -> VerifiedColabSnapshotStore:
    return VerifiedColabSnapshotStore(
        artifact_root=tmp_path / "local",
        checkpoint_root=tmp_path / "drive",
        scratch_root=tmp_path / "scratch",
    )


def clear_local(store: VerifiedColabSnapshotStore) -> None:
    shutil.rmtree(store.artifact_root)
    store.artifact_root.mkdir()


def test_snapshot_excludes_interrupted_or_orphaned_shard_publication(tmp_path):
    snapshot = snapshot_store(tmp_path)
    store = FAArtifactStore(snapshot.artifact_root)
    valid = store.write_completed_shard(
        "run-v1",
        "pilot",
        "complete",
        ({"kind": "example", "value": 1},),
        {"source": "test"},
    )
    orphan = valid.data_path.with_name("interrupted.jsonl")
    orphan.write_text(json.dumps({"kind": "example"}) + "\n", encoding="utf-8")

    archive = snapshot.checkpoint()
    clear_local(snapshot)
    assert snapshot.restore_latest() == archive

    assert store.verify_shard(valid.manifest_path).sha256 == valid.sha256
    assert not orphan.exists()


def test_snapshot_restores_consistent_unlocked_endpoint_and_inputs(tmp_path):
    snapshot = snapshot_store(tmp_path)
    store = FAArtifactStore(snapshot.artifact_root)
    input_path = snapshot.artifact_root / "inputs" / "source.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text('{"source":"fixed"}\n', encoding="utf-8")
    capability = store.write_completed_shard(
        "run-v1",
        "behavior_test",
        "capability",
        ({"kind": "capability"},),
        {"source": "test"},
    )
    store.seal_endpoint(
        "behavior_test",
        (capability,),
        {"preregistration": "a" * 64, "selection_manifest": "b" * 64},
    )
    store.unlock_endpoint("behavior_test", "a" * 64, "b" * 64)

    archive = snapshot.checkpoint()
    clear_local(snapshot)
    snapshot.restore_latest()

    restored = FAArtifactStore(snapshot.artifact_root)
    assert restored.endpoint_state("behavior_test", capability.manifest_path) == (
        "unlocked_once"
    )
    assert input_path.read_text(encoding="utf-8") == '{"source":"fixed"}\n'
    assert archive.exists()
