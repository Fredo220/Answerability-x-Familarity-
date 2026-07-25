"""Content-addressed checkpoint mirrors for Source-v6 development runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_FILE = "execution_identity.json"


class DevelopmentCheckpointMirror:
    """Mirror one development run to a Drive-compatible checkpoint root."""

    def __init__(self, checkpoint_root: str | Path) -> None:
        self.checkpoint_root = Path(checkpoint_root).expanduser().resolve()

    def snapshot(
        self,
        run_dir: str | Path,
        identity_sha256: str,
    ) -> Path:
        """Publish an immutable snapshot and atomically advance ``LATEST``."""
        _validate_sha256(identity_sha256, "identity")
        run_path = Path(run_dir).expanduser().resolve()
        members = _member_sha256_map(run_path)
        if members.get(_IDENTITY_FILE) != identity_sha256:
            raise ValueError("execution identity hash mismatch")
        if self.checkpoint_root.is_relative_to(run_path):
            raise ValueError("checkpoint root must be outside the run directory")

        identity_root = self.checkpoint_root / identity_sha256
        identity_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fa-checkpoint-") as temporary:
            archive_source = Path(temporary) / "checkpoint.zip"
            _write_archive(run_path, members, archive_source)
            _verify_archive_members(archive_source, members)
            archive_sha256 = _sha256_file(archive_source)
            archive_name = f"archive-{archive_sha256}.zip"
            archive_path = identity_root / archive_name
            _publish_immutable_file(
                archive_source,
                archive_path,
                archive_sha256,
            )

        metadata = {
            "schema_version": 1,
            "kind": "fa_development_checkpoint",
            "identity_sha256": identity_sha256,
            "archive_file": archive_name,
            "archive_sha256": archive_sha256,
            "members": members,
        }
        metadata_bytes = _canonical_bytes(metadata)
        metadata_sha256 = _sha256_bytes(metadata_bytes)
        metadata_path = (
            identity_root / f"metadata-{metadata_sha256}.json"
        )
        _publish_immutable_bytes(
            metadata_bytes,
            metadata_path,
            metadata_sha256,
        )
        pointer = {
            "schema_version": 1,
            "identity_sha256": identity_sha256,
            "metadata_file": metadata_path.name,
            "metadata_sha256": metadata_sha256,
        }
        _atomic_write(identity_root / "LATEST.json", _canonical_bytes(pointer))
        return metadata_path

    def restore(
        self,
        run_dir: str | Path,
        identity_sha256: str,
    ) -> bool:
        """Restore the latest verified snapshot, or return false if absent."""
        _validate_sha256(identity_sha256, "identity")
        identity_root = self.checkpoint_root / identity_sha256
        pointer_path = identity_root / "LATEST.json"
        if not pointer_path.is_file():
            return False

        metadata = _load_verified_metadata(
            pointer_path,
            identity_root,
            identity_sha256,
        )
        members = metadata["members"]
        archive_path = identity_root / metadata["archive_file"]
        if not archive_path.is_file():
            raise ValueError("checkpoint archive is missing")
        if _sha256_file(archive_path) != metadata["archive_sha256"]:
            raise ValueError("archive hash mismatch")

        run_path = Path(run_dir).expanduser().resolve()
        existing = _member_sha256_map(run_path, allow_missing=True)
        if existing and existing.get(_IDENTITY_FILE) != identity_sha256:
            raise ValueError("local execution identity hash mismatch")
        for name, observed_sha256 in existing.items():
            if name not in members:
                raise ValueError("local run contains uncheckpointed member")
            if observed_sha256 != members[name]:
                raise ValueError("local run member hash mismatch")

        run_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="fa-restore-",
            dir=run_path.parent,
        ) as temporary:
            staging = Path(temporary)
            _restore_archive(archive_path, staging, members)
            for name in sorted(members):
                destination = run_path / PurePosixPath(name)
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / PurePosixPath(name), destination)

        if _member_sha256_map(run_path) != members:
            raise ValueError("restored checkpoint member map mismatch")
        return True


def _load_verified_metadata(
    pointer_path: Path,
    identity_root: Path,
    identity_sha256: str,
) -> dict[str, Any]:
    pointer = _read_object(pointer_path, "checkpoint pointer")
    if pointer.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint pointer schema")
    if pointer.get("identity_sha256") != identity_sha256:
        raise ValueError("pointer identity mismatch")
    metadata_sha256 = pointer.get("metadata_sha256")
    _validate_sha256(metadata_sha256, "metadata")
    metadata_name = pointer.get("metadata_file")
    if (
        not isinstance(metadata_name, str)
        or Path(metadata_name).name != metadata_name
        or metadata_name != f"metadata-{metadata_sha256}.json"
    ):
        raise ValueError("invalid checkpoint metadata path")
    metadata_path = identity_root / metadata_name
    if not metadata_path.is_file():
        raise ValueError("checkpoint metadata is missing")
    metadata_bytes = metadata_path.read_bytes()
    if _sha256_bytes(metadata_bytes) != metadata_sha256:
        raise ValueError("metadata hash mismatch")
    metadata = _decode_object(metadata_bytes, "checkpoint metadata")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "fa_development_checkpoint"
    ):
        raise ValueError("unsupported checkpoint metadata schema")
    if metadata.get("identity_sha256") != identity_sha256:
        raise ValueError("metadata identity mismatch")

    archive_sha256 = metadata.get("archive_sha256")
    _validate_sha256(archive_sha256, "archive")
    archive_name = metadata.get("archive_file")
    if (
        not isinstance(archive_name, str)
        or Path(archive_name).name != archive_name
        or archive_name != f"archive-{archive_sha256}.zip"
    ):
        raise ValueError("invalid checkpoint archive path")
    members = metadata.get("members")
    if not isinstance(members, dict) or not members:
        raise ValueError("checkpoint member map is missing")
    for name, member_sha256 in members.items():
        _validate_member_name(name)
        _validate_sha256(member_sha256, "checkpoint member")
    if members.get(_IDENTITY_FILE) != identity_sha256:
        raise ValueError("checkpoint execution identity mismatch")
    return metadata


def _write_archive(
    run_dir: Path,
    members: dict[str, str],
    destination: Path,
) -> None:
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with (run_dir / PurePosixPath(name)).open("rb") as source:
                with archive.open(info, "w") as target:
                    shutil.copyfileobj(source, target)


def _restore_archive(
    archive_path: Path,
    staging: Path,
    members: dict[str, str],
) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("duplicate checkpoint archive member")
        for info in infos:
            _validate_zip_info(info)
        if set(names) != set(members):
            raise ValueError("checkpoint archive member list mismatch")
        for info in infos:
            destination = staging / PurePosixPath(info.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(info) as source, destination.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
            if digest.hexdigest() != members[info.filename]:
                raise ValueError("checkpoint member hash mismatch")


def _verify_archive_members(
    archive_path: Path,
    members: dict[str, str],
) -> None:
    with tempfile.TemporaryDirectory(prefix="fa-checkpoint-verify-") as temporary:
        _restore_archive(archive_path, Path(temporary), members)


def _member_sha256_map(
    root: Path,
    *,
    allow_missing: bool = False,
) -> dict[str, str]:
    if not root.exists():
        if allow_missing:
            return {}
        raise ValueError("run directory is missing")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("run directory must be a regular directory")
    members: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("run directory contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("run directory contains a non-regular file")
        name = path.relative_to(root).as_posix()
        _validate_member_name(name)
        members[name] = _sha256_file(path)
    return members


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    _validate_member_name(info.filename)
    if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError("checkpoint archive contains a non-regular member")


def _validate_member_name(name: object) -> None:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("unsafe checkpoint member path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("unsafe checkpoint member path")
    if path.as_posix() != name:
        raise ValueError("unsafe checkpoint member path")


def _publish_immutable_file(
    source: Path,
    destination: Path,
    expected_sha256: str,
) -> None:
    if destination.exists():
        if _sha256_file(destination) != expected_sha256:
            raise ValueError("immutable checkpoint object hash mismatch")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        with source.open("rb") as input_file:
            shutil.copyfileobj(input_file, temporary)
    try:
        if _sha256_file(temporary_path) != expected_sha256:
            raise ValueError("checkpoint object changed while publishing")
        if destination.exists():
            if _sha256_file(destination) != expected_sha256:
                raise ValueError("immutable checkpoint object hash mismatch")
        else:
            os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_immutable_bytes(
    content: bytes,
    destination: Path,
    expected_sha256: str,
) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        _publish_immutable_file(
            temporary_path,
            destination,
            expected_sha256,
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        return _decode_object(path.read_bytes(), description)
    except OSError as error:
        raise ValueError(f"cannot read {description}") from error


def _decode_object(content: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _validate_sha256(value: object, description: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"invalid {description} SHA-256")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
