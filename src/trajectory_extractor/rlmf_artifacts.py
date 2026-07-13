from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from trajectory_extractor.rlmf_types import RLMFConfig


_SAFE_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RLMFArtifactStore:
    """Append-only RLMF artifacts, isolated from the existing Study 0 run store."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_jsonl(
        self, study_id: str, section: str, name: str, rows: Iterable[Mapping[str, Any]]
    ) -> Path:
        payload = b"".join(
            json.dumps(row, sort_keys=True).encode("utf-8") + b"\n"
            for row in rows
        )
        destination = self._path(study_id, section, name, ".jsonl")
        _exclusive_write_bytes(destination, payload)
        return destination

    def append_jsonl(
        self, study_id: str, section: str, name: str, row: Mapping[str, Any]
    ) -> Path:
        if not isinstance(row, Mapping):
            raise ValueError("append-only JSONL rows must be mappings")
        destination = self._path(study_id, section, name, ".jsonl")
        if destination.is_symlink():
            raise ValueError("append-only artifacts must not be symlinks")
        if destination.exists() and not destination.is_file():
            raise ValueError("append-only artifact destination must be a regular file")
        payload = json.dumps(row, sort_keys=True).encode("utf-8") + b"\n"
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(destination, flags, 0o644)
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(destination.parent)
        return destination

    def write_json(self, study_id: str, section: str, name: str, value: Any) -> Path:
        destination = self._path(study_id, section, name, ".json")
        _exclusive_write_bytes(
            destination, json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
        )
        return destination

    def write_bytes(
        self,
        study_id: str,
        section: str,
        name: str,
        payload: bytes,
        *,
        suffix: str = ".bin",
    ) -> Path:
        if not isinstance(payload, bytes):
            raise ValueError("binary artifact payload must be bytes")
        if not isinstance(suffix, str) or not re.fullmatch(r"(?:\.[A-Za-z0-9_-]+)?", suffix):
            raise ValueError("binary artifact suffix is unsafe")
        destination = self._path(study_id, section, name, suffix)
        _exclusive_write_bytes(destination, payload)
        return destination

    def directory_path(
        self, study_id: str, section: str, name: str, *, create_parent: bool = False
    ) -> Path:
        return self._section(study_id, section, create=create_parent) / _safe_id(name, "name")

    def owns_path(self, study_id: str, path: str | Path) -> bool:
        try:
            namespace = self._namespace(study_id, create=False)
        except (FileNotFoundError, NotADirectoryError, ValueError):
            return False
        candidate = Path(path)
        if candidate.is_symlink() or not candidate.exists():
            return False
        try:
            candidate.resolve().relative_to(namespace)
        except ValueError:
            return False
        return True

    def publish_directory(
        self, study_id: str, section: str, name: str, source: str | Path
    ) -> Path:
        source_path = Path(source)
        if source_path.is_symlink() or not source_path.is_dir():
            raise ValueError("directory artifact source must be a real directory")
        entries = _validated_directory_entries(source_path)
        destination = self.directory_path(study_id, section, name, create_parent=True)
        if destination.is_symlink() or destination.exists():
            raise FileExistsError(destination)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            for relative, source_entry, is_directory in entries:
                target = temporary / relative
                if is_directory:
                    target.mkdir()
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source_entry.open("rb") as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
                    writer.flush()
                    os.fchmod(writer.fileno(), 0o644)
                    os.fsync(writer.fileno())
                _fsync_directory(target.parent)
            _fsync_directory(temporary)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            os.rename(temporary, destination)
            _fsync_directory(destination.parent)
            return destination
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def write_npz(self, study_id: str, section: str, name: str, **arrays: np.ndarray) -> Path:
        if not arrays:
            raise ValueError("write_npz requires at least one named array")
        destination = self._path(study_id, section, name, ".npz")
        ensure_durable_directory(destination.parent)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fchmod(handle.fileno(), 0o644)
                os.fsync(handle.fileno())
            os.link(temporary, destination)
            temporary.unlink()
            temporary = None
            _fsync_directory(destination.parent)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return destination

    def complete_endpoint(
        self,
        study_id: str,
        endpoint: str,
        config: RLMFConfig,
        paths: Iterable[str | Path] | Mapping[str, str | Path],
        *,
        parent_hashes: Mapping[str, str] | None = None,
    ) -> Path:
        if not isinstance(config, RLMFConfig):
            raise ValueError("config must be an RLMFConfig")
        safe_study = _safe_id(study_id, "study_id")
        if config.study_id != safe_study:
            raise ValueError("config study_id must match endpoint study_id")
        endpoint = _safe_id(endpoint, "endpoint")
        artifacts = self._bound_artifacts(safe_study, paths)
        marker = self._section(safe_study, "endpoints", create=True) / f"{endpoint}.complete.json"
        if marker.is_symlink():
            raise ValueError("endpoint marker must not be a symlink")
        if marker.exists():
            raise FileExistsError(marker)
        config_artifact = self._bind_config_artifact(safe_study, config)
        sealed_parents = {"config": config.config_hash}
        if parent_hashes is not None:
            if not isinstance(parent_hashes, Mapping):
                raise ValueError("parent_hashes must be a mapping")
            for name, digest in parent_hashes.items():
                safe_name = _safe_id(name, "parent_hashes key")
                if safe_name == "config":
                    raise ValueError("parent_hashes must not replace the config hash")
                _validate_sha256(digest, "parent_hashes")
                sealed_parents[safe_name] = digest
        for path in artifacts.values():
            _fsync_file(path)
            _fsync_directory(path.parent)
        _fsync_file(config_artifact)
        _fsync_directory(config_artifact.parent)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "study_id": safe_study,
            "endpoint": endpoint,
            "config_artifact": "metadata/config.json",
            "parent_hashes": dict(sorted(sealed_parents.items())),
            "artifact_hashes": {
                relative: sha256_file(path) for relative, path in artifacts.items()
            },
        }
        _exclusive_write_bytes(
            marker, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        )
        return marker

    def verify_endpoint(self, study_id: str, endpoint: str) -> dict[str, Any]:
        safe_study = _safe_id(study_id, "study_id")
        safe_endpoint = _safe_id(endpoint, "endpoint")
        namespace = self._namespace(safe_study, create=False)
        marker = self._section(safe_study, "endpoints", create=False) / f"{safe_endpoint}.complete.json"
        if marker.is_symlink():
            raise ValueError("endpoint marker must not be a symlink")
        try:
            value = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("endpoint marker is unreadable") from error
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("endpoint marker has an invalid schema")
        if value.get("study_id") != safe_study or value.get("endpoint") != safe_endpoint:
            raise ValueError("endpoint marker IDs do not match")
        created_at = value.get("created_at")
        if created_at is not None:
            try:
                parsed_created_at = datetime.fromisoformat(created_at)
            except (TypeError, ValueError) as error:
                raise ValueError("endpoint marker has invalid created_at") from error
            if parsed_created_at.tzinfo is None or parsed_created_at.utcoffset() is None:
                raise ValueError("endpoint marker created_at must include a timezone")
        parents = value.get("parent_hashes")
        if not isinstance(parents, dict) or "config" not in parents:
            raise ValueError("endpoint marker has invalid parent_hashes")
        for name, digest in parents.items():
            _safe_id(name, "parent_hashes key")
            _validate_sha256(digest, "parent_hashes")
        if value.get("config_artifact") != "metadata/config.json":
            raise ValueError("endpoint marker has invalid config_artifact")
        config = self._read_config_artifact(
            self._section(safe_study, "metadata", create=False) / "config.json"
        )
        if config.study_id != safe_study:
            raise ValueError("config artifact study_id does not match endpoint study_id")
        if config.config_hash != parents["config"]:
            raise ValueError("configuration hash mismatch")
        hashes = value.get("artifact_hashes")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("endpoint marker has invalid artifact_hashes")
        for relative, expected in hashes.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ValueError("endpoint marker has invalid artifact_hashes")
            _validate_sha256(expected, "artifact_hashes")
            path = _relative_artifact_path(namespace, relative)
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(namespace):
                raise ValueError("bound artifact is outside the RLMF namespace")
            if sha256_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {relative}")
        return value

    def _namespace(self, study_id: str, *, create: bool) -> Path:
        safe_study = _safe_id(study_id, "study_id")
        return self._namespace_components(("runs", "rlmf", safe_study), create=create)

    def _path(self, study_id: str, section: str, name: str, suffix: str) -> Path:
        return self._section(study_id, section, create=True) / (
            f"{_safe_id(name, 'name')}{suffix}"
        )

    def _section(self, study_id: str, section: str, *, create: bool) -> Path:
        return self._namespace_components(
            ("runs", "rlmf", _safe_id(study_id, "study_id"), _safe_id(section, "section")),
            create=create,
        )

    def _namespace_components(self, components: tuple[str, ...], *, create: bool) -> Path:
        root = self._trusted_root(create=create)
        current = root
        for component in components:
            current = current / component
            if current.is_symlink():
                raise ValueError("RLMF namespace components must not be symlinks")
            if current.exists():
                if not current.is_dir():
                    raise NotADirectoryError(current)
                continue
            if not create:
                raise FileNotFoundError(current)
            try:
                os.mkdir(current)
            except FileExistsError:
                if current.is_symlink():
                    raise ValueError("RLMF namespace components must not be symlinks")
                if not current.is_dir():
                    raise
            _fsync_directory(current.parent)
        return current

    def _trusted_root(self, *, create: bool) -> Path:
        if self.root.is_symlink():
            raise ValueError("trusted root must not be a symlink")
        if create:
            ensure_durable_directory(self.root)
        if not self.root.exists() or not self.root.is_dir():
            raise NotADirectoryError(self.root)
        return self.root.resolve()

    def _bind_config_artifact(self, study_id: str, config: RLMFConfig) -> Path:
        destination = self._section(study_id, "metadata", create=True) / "config.json"
        if destination.is_symlink():
            raise ValueError("config artifact must not be a symlink")
        if destination.exists():
            existing = self._read_config_artifact(destination)
            if existing.study_id != study_id or existing.config_hash != config.config_hash:
                raise ValueError("config artifact does not match supplied study config")
            return destination
        _exclusive_write_bytes(destination, config.canonical_bytes)
        return destination

    def _read_config_artifact(self, path: Path) -> RLMFConfig:
        if path.is_symlink() or not path.is_file():
            raise ValueError("config artifact is missing or outside the RLMF namespace")
        try:
            payload = path.read_bytes()
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError("config artifact must contain a JSON object")
            config = RLMFConfig(**value)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("config artifact is invalid") from error
        if payload != config.canonical_bytes:
            raise ValueError("config artifact is not canonical")
        return config

    def _bound_artifacts(
        self,
        study_id: str,
        paths: Iterable[str | Path] | Mapping[str, str | Path],
    ) -> dict[str, Path]:
        values = paths.values() if isinstance(paths, Mapping) else paths
        namespace = self._namespace(study_id, create=False)
        artifacts: dict[str, Path] = {}
        for value in values:
            path = Path(value)
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(namespace):
                raise ValueError("endpoint artifacts must exist under runs/rlmf/<study_id>")
            relative = path.resolve().relative_to(namespace).as_posix()
            if relative in artifacts:
                raise ValueError("endpoint artifacts must be unique")
            artifacts[relative] = path
        if not artifacts:
            raise ValueError("endpoint requires at least one artifact")
        return dict(sorted(artifacts.items()))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_directory_entries(source: Path) -> list[tuple[Path, Path, bool]]:
    entries: list[tuple[Path, Path, bool]] = []
    for root, directories, files in os.walk(source, topdown=True, followlinks=False):
        root_path = Path(root)
        for name in sorted(directories):
            path = root_path / name
            if path.is_symlink():
                raise ValueError("directory artifacts must not contain symlinks")
            mode = path.stat(follow_symlinks=False).st_mode
            if not stat.S_ISDIR(mode):
                raise ValueError("directory artifacts must contain only regular entries")
            entries.append((path.relative_to(source), path, True))
        for name in sorted(files):
            path = root_path / name
            if path.is_symlink():
                raise ValueError("directory artifacts must not contain symlinks")
            metadata = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("directory artifacts must contain only regular files")
            if metadata.st_nlink != 1:
                raise ValueError("directory artifacts must not contain hardlinks")
            entries.append((path.relative_to(source), path, False))
    return sorted(entries, key=lambda item: (len(item[0].parts), item[0].as_posix()))


def _relative_artifact_path(namespace: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise ValueError("endpoint marker has invalid artifact path")
    return namespace / candidate


# Adapted from secondary_artifacts.py's durable exclusive-publish pattern.
def _exclusive_write_bytes(destination: Path, payload: bytes) -> None:
    ensure_durable_directory(destination.parent)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            _write_all(handle.fileno(), payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        temporary = None
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("write made no progress")
        remaining = remaining[written:]


def ensure_durable_directory(directory: str | Path) -> Path:
    destination = Path(directory)
    missing: list[Path] = []
    current = destination
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            raise FileNotFoundError(f"no existing ancestor for directory: {destination}")
        current = current.parent
    if not current.is_dir():
        raise NotADirectoryError(current)
    for child in reversed(missing):
        try:
            os.mkdir(child)
        except FileExistsError:
            if not child.is_dir():
                raise
        _fsync_directory(child.parent)
    return destination


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as error:
        if error.errno in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_id(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} must be a safe identifier")
    return value


def _validate_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hash")
