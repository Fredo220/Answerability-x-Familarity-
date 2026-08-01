"""Verified, content-addressed snapshots for resumable Colab transactions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

from trajectory_extractor.fa_artifacts import FAArtifactStore


_INDEX_NAME = "_snapshot_index.json"
_STATIC_DIRECTORIES = (
    "inputs",
    "notebook_state",
    "rater-packets",
    "adjudication-packet",
)


class VerifiedColabSnapshotStore:
    """Mirror only complete FA transactions to a non-POSIX checkpoint mount."""

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        checkpoint_root: str | Path,
        scratch_root: str | Path,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.checkpoint_root = Path(checkpoint_root).resolve()
        self.scratch_root = Path(scratch_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.scratch_root.mkdir(parents=True, exist_ok=True)

    def checkpoint(self) -> Path:
        members = self._verified_members(self.artifact_root)
        index = {
            "schema_version": 1,
            "members": {
                path.relative_to(self.artifact_root).as_posix(): _sha256(path)
                for path in members
            },
        }
        with tempfile.TemporaryDirectory(
            dir=self.scratch_root, prefix="fa-snapshot-"
        ) as temporary:
            archive = Path(temporary) / "artifacts.zip"
            with zipfile.ZipFile(
                archive, "w", compression=zipfile.ZIP_DEFLATED
            ) as bundle:
                bundle.writestr(_INDEX_NAME, _canonical_bytes(index))
                for path in members:
                    bundle.write(
                        path, path.relative_to(self.artifact_root).as_posix()
                    )
            archive_sha256 = _sha256(archive)
            destination = self.checkpoint_root / f"fa-{archive_sha256}.zip"
            if destination.exists():
                if _sha256(destination) != archive_sha256:
                    raise ValueError("content-addressed snapshot changed")
                return destination
            partial = self.checkpoint_root / (
                f".{destination.name}.{uuid.uuid4().hex}.partial"
            )
            shutil.copy2(archive, partial)
            if _sha256(partial) != archive_sha256:
                partial.unlink(missing_ok=True)
                raise ValueError("snapshot copy hash mismatch")
            os.replace(partial, destination)
            return destination

    def restore_latest(self) -> Path | None:
        if self.artifact_root.exists() and any(self.artifact_root.iterdir()):
            return None
        candidates = sorted(
            self.checkpoint_root.glob("fa-*.zip"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for archive in candidates:
            expected = archive.stem.removeprefix("fa-")
            with tempfile.TemporaryDirectory(
                dir=self.artifact_root.parent, prefix="fa-restore-"
            ) as temporary:
                local_archive = Path(temporary) / "snapshot.zip"
                try:
                    shutil.copy2(archive, local_archive)
                except OSError:
                    continue
                if _sha256(local_archive) != expected:
                    continue
                staging = Path(temporary) / "artifacts"
                staging.mkdir()
                try:
                    self._extract_verified(local_archive, staging)
                    observed = {
                        path.relative_to(staging).as_posix()
                        for path in self._verified_members(staging)
                    }
                    expected_members = _read_index(local_archive)["members"]
                    if observed != set(expected_members):
                        raise ValueError("snapshot transaction set does not verify")
                except (OSError, ValueError, zipfile.BadZipFile):
                    continue
                if self.artifact_root.exists():
                    self.artifact_root.rmdir()
                os.replace(staging, self.artifact_root)
                return archive
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        return None

    def _extract_verified(self, archive: Path, staging: Path) -> None:
        index = _read_index(archive)
        expected = index["members"]
        with zipfile.ZipFile(archive) as bundle:
            names = tuple(member.filename for member in bundle.infolist())
            if set(names) != {*expected, _INDEX_NAME} or len(names) != len(set(names)):
                raise ValueError("snapshot member list does not match its index")
            for name in expected:
                destination = (staging / name).resolve()
                destination.relative_to(staging.resolve())
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if _sha256(destination) != expected[name]:
                    raise ValueError("snapshot member hash mismatch")

    @staticmethod
    def _verified_members(root: Path) -> tuple[Path, ...]:
        store = FAArtifactStore(root)
        members: set[Path] = set()
        for directory_name in _STATIC_DIRECTORIES:
            directory = root / directory_name
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if path.is_symlink():
                    raise ValueError("snapshot input contains a symlink")
                if path.is_file() and not path.name.endswith(".tmp"):
                    members.add(path.resolve())

        def add_verified_shards() -> None:
            pattern = "runs/familiarity_answerability/*/shards/*/*.jsonl.manifest.json"
            for manifest in root.glob(pattern):
                try:
                    shard = store.verify_shard(manifest)
                except (OSError, ValueError):
                    continue
                members.update(
                    (shard.data_path.resolve(), shard.manifest_path.resolve())
                )

        add_verified_shards()
        endpoint_pattern = (
            "runs/familiarity_answerability/*/endpoints/*/sealed.json"
        )
        for sealed_path in root.glob(endpoint_pattern):
            sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
            artifacts = sealed.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError("sealed endpoint has no artifact capability")
            endpoint = sealed_path.parent.name
            manifest = root / artifacts[0]["manifest_path"]
            state = store.endpoint_state(endpoint, manifest)
            if state == "closed":
                store.read_closed_metrics(endpoint, manifest)
                states = ("sealed", "unlocked_once", "evaluated", "closed")
            elif state == "evaluated":
                store.read_evaluated_metrics(endpoint, manifest)
                states = ("sealed", "unlocked_once", "evaluated")
            elif state == "unlocked_once":
                states = ("sealed", "unlocked_once")
            else:
                states = ("sealed",)
            members.update(
                (sealed_path.parent / f"{name}.json").resolve() for name in states
            )
        add_verified_shards()
        return tuple(sorted(members))


def _read_index(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as bundle:
        try:
            value = json.loads(bundle.read(_INDEX_NAME))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("snapshot index is unreadable") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or not isinstance(value.get("members"), dict)
        or any(
            not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            for path, digest in value["members"].items()
        )
    ):
        raise ValueError("snapshot index has an invalid schema")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
