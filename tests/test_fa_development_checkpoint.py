from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from trajectory_extractor.fa_development_checkpoint import (
    DevelopmentCheckpointMirror,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _make_run(tmp_path: Path) -> tuple[Path, str]:
    identity = {"schema_version": 1, "split": "instrument_development"}
    identity_bytes = _canonical_bytes(identity)
    identity_sha256 = hashlib.sha256(identity_bytes).hexdigest()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "execution_identity.json").write_bytes(identity_bytes)
    (run_dir / "batch-000000.jsonl").write_text(
        '{"completion":"Paris"}\n',
        encoding="utf-8",
    )
    return run_dir, identity_sha256


def _latest_metadata(
    checkpoint_root: Path,
    identity_sha256: str,
) -> tuple[Path, dict[str, object]]:
    identity_root = checkpoint_root / identity_sha256
    pointer = json.loads(
        (identity_root / "LATEST.json").read_text(encoding="utf-8")
    )
    metadata_path = identity_root / str(pointer["metadata_file"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata_path, metadata


def _publish_modified_checkpoint(
    checkpoint_root: Path,
    identity_sha256: str,
    archive_members: dict[str, bytes],
    metadata: dict[str, object],
) -> None:
    identity_root = checkpoint_root / identity_sha256
    temporary_archive = identity_root / "modified.zip"
    with zipfile.ZipFile(temporary_archive, "w") as archive:
        for name, content in archive_members.items():
            archive.writestr(name, content)
    archive_bytes = temporary_archive.read_bytes()
    temporary_archive.unlink()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    archive_name = f"archive-{archive_sha256}.zip"
    (identity_root / archive_name).write_bytes(archive_bytes)

    metadata["archive_file"] = archive_name
    metadata["archive_sha256"] = archive_sha256
    metadata_bytes = _canonical_bytes(metadata)
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    metadata_name = f"metadata-{metadata_sha256}.json"
    (identity_root / metadata_name).write_bytes(metadata_bytes)
    pointer = {
        "schema_version": 1,
        "identity_sha256": identity_sha256,
        "metadata_file": metadata_name,
        "metadata_sha256": metadata_sha256,
    }
    (identity_root / "LATEST.json").write_bytes(_canonical_bytes(pointer))


def test_snapshot_and_restore_round_trip_without_hardlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, identity_sha256 = _make_run(tmp_path)
    checkpoint_root = tmp_path / "drive"
    mirror = DevelopmentCheckpointMirror(checkpoint_root)

    def reject_hardlink(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("checkpoint mirrors must not use hardlinks")

    monkeypatch.setattr(os, "link", reject_hardlink)
    metadata_path = mirror.snapshot(run_dir, identity_sha256)
    _, metadata = _latest_metadata(checkpoint_root, identity_sha256)
    assert metadata_path.is_file()
    assert metadata["members"] == {
        "batch-000000.jsonl": hashlib.sha256(
            b'{"completion":"Paris"}\n'
        ).hexdigest(),
        "execution_identity.json": identity_sha256,
    }

    for path in sorted(run_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    run_dir.rmdir()

    assert mirror.restore(run_dir, identity_sha256) is True
    assert (run_dir / "batch-000000.jsonl").read_text() == (
        '{"completion":"Paris"}\n'
    )
    assert hashlib.sha256(
        (run_dir / "execution_identity.json").read_bytes()
    ).hexdigest() == identity_sha256


def test_snapshot_is_content_addressed_and_preserves_prior_versions(
    tmp_path: Path,
) -> None:
    run_dir, identity_sha256 = _make_run(tmp_path)
    checkpoint_root = tmp_path / "drive"
    mirror = DevelopmentCheckpointMirror(checkpoint_root)

    first = mirror.snapshot(run_dir, identity_sha256)
    assert mirror.snapshot(run_dir, identity_sha256) == first
    (run_dir / "screening_yield.json").write_text('{"status":"done"}\n')
    second = mirror.snapshot(run_dir, identity_sha256)

    identity_root = checkpoint_root / identity_sha256
    assert second != first
    assert first.is_file()
    assert len(list(identity_root.glob("archive-*.zip"))) == 2
    assert len(list(identity_root.glob("metadata-*.json"))) == 2
    assert not list(identity_root.glob("*.partial"))


def test_snapshot_rejects_execution_identity_mismatch(tmp_path: Path) -> None:
    run_dir, identity_sha256 = _make_run(tmp_path)
    mirror = DevelopmentCheckpointMirror(tmp_path / "drive")

    with pytest.raises(ValueError, match="execution identity hash mismatch"):
        mirror.snapshot(run_dir, "f" * 64)

    assert not (tmp_path / "drive" / ("f" * 64)).exists()
    assert identity_sha256 != "f" * 64


def test_restore_rejects_pointer_identity_mismatch(tmp_path: Path) -> None:
    run_dir, identity_sha256 = _make_run(tmp_path)
    checkpoint_root = tmp_path / "drive"
    mirror = DevelopmentCheckpointMirror(checkpoint_root)
    mirror.snapshot(run_dir, identity_sha256)
    pointer_path = checkpoint_root / identity_sha256 / "LATEST.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["identity_sha256"] = "e" * 64
    pointer_path.write_bytes(_canonical_bytes(pointer))

    with pytest.raises(ValueError, match="pointer identity mismatch"):
        mirror.restore(tmp_path / "restored", identity_sha256)


def test_restore_rejects_archive_hash_mismatch(tmp_path: Path) -> None:
    run_dir, identity_sha256 = _make_run(tmp_path)
    checkpoint_root = tmp_path / "drive"
    mirror = DevelopmentCheckpointMirror(checkpoint_root)
    mirror.snapshot(run_dir, identity_sha256)
    _, metadata = _latest_metadata(checkpoint_root, identity_sha256)
    archive_path = (
        checkpoint_root / identity_sha256 / str(metadata["archive_file"])
    )
    archive_path.write_bytes(archive_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="archive hash mismatch"):
        mirror.restore(tmp_path / "restored", identity_sha256)


def test_restore_rejects_member_hash_mismatch(tmp_path: Path) -> None:
    run_dir, identity_sha256 = _make_run(tmp_path)
    checkpoint_root = tmp_path / "drive"
    mirror = DevelopmentCheckpointMirror(checkpoint_root)
    mirror.snapshot(run_dir, identity_sha256)
    _, metadata = _latest_metadata(checkpoint_root, identity_sha256)
    _publish_modified_checkpoint(
        checkpoint_root,
        identity_sha256,
        {
            "execution_identity.json": (
                run_dir / "execution_identity.json"
            ).read_bytes(),
            "batch-000000.jsonl": b'{"completion":"London"}\n',
        },
        metadata,
    )

    with pytest.raises(ValueError, match="checkpoint member hash mismatch"):
        mirror.restore(tmp_path / "restored", identity_sha256)


def test_restore_rejects_zip_path_traversal(tmp_path: Path) -> None:
    run_dir, identity_sha256 = _make_run(tmp_path)
    checkpoint_root = tmp_path / "drive"
    mirror = DevelopmentCheckpointMirror(checkpoint_root)
    mirror.snapshot(run_dir, identity_sha256)
    _, metadata = _latest_metadata(checkpoint_root, identity_sha256)
    members = dict(metadata["members"])
    members["../escape.txt"] = hashlib.sha256(b"escape").hexdigest()
    metadata["members"] = members
    _publish_modified_checkpoint(
        checkpoint_root,
        identity_sha256,
        {
            "execution_identity.json": (
                run_dir / "execution_identity.json"
            ).read_bytes(),
            "batch-000000.jsonl": (
                run_dir / "batch-000000.jsonl"
            ).read_bytes(),
            "../escape.txt": b"escape",
        },
        metadata,
    )

    with pytest.raises(ValueError, match="unsafe checkpoint member path"):
        mirror.restore(tmp_path / "restored", identity_sha256)
    assert not (tmp_path / "escape.txt").exists()


def test_restore_rejects_conflicting_local_files(tmp_path: Path) -> None:
    run_dir, identity_sha256 = _make_run(tmp_path)
    checkpoint_root = tmp_path / "drive"
    mirror = DevelopmentCheckpointMirror(checkpoint_root)
    mirror.snapshot(run_dir, identity_sha256)
    (run_dir / "batch-000000.jsonl").write_text(
        '{"completion":"tampered"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="local run member hash mismatch"):
        mirror.restore(run_dir, identity_sha256)
