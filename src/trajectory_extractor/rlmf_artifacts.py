from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


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

    def write_json(self, study_id: str, section: str, name: str, value: Any) -> Path:
        destination = self._path(study_id, section, name, ".json")
        _exclusive_write_bytes(
            destination, json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
        )
        return destination

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
        config_hash: str,
        paths: Iterable[str | Path] | Mapping[str, str | Path],
    ) -> Path:
        _validate_sha256(config_hash, "config_hash")
        endpoint = _safe_id(endpoint, "endpoint")
        artifacts = self._bound_artifacts(study_id, paths)
        for path in artifacts.values():
            _fsync_file(path)
            _fsync_directory(path.parent)
        marker = self._namespace(study_id) / "endpoints" / f"{endpoint}.complete.json"
        payload = {
            "schema_version": 1,
            "study_id": _safe_id(study_id, "study_id"),
            "endpoint": endpoint,
            "parent_hashes": {"config": config_hash},
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
        namespace = self._namespace(safe_study)
        marker = namespace / "endpoints" / f"{safe_endpoint}.complete.json"
        try:
            value = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("endpoint marker is unreadable") from error
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("endpoint marker has an invalid schema")
        if value.get("study_id") != safe_study or value.get("endpoint") != safe_endpoint:
            raise ValueError("endpoint marker IDs do not match")
        parents = value.get("parent_hashes")
        if not isinstance(parents, dict) or set(parents) != {"config"}:
            raise ValueError("endpoint marker has invalid parent_hashes")
        _validate_sha256(parents["config"], "parent_hashes")
        hashes = value.get("artifact_hashes")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("endpoint marker has invalid artifact_hashes")
        for relative, expected in hashes.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ValueError("endpoint marker has invalid artifact_hashes")
            _validate_sha256(expected, "artifact_hashes")
            path = namespace / relative
            if not path.is_file() or not path.resolve().is_relative_to(namespace.resolve()):
                raise ValueError("bound artifact is outside the RLMF namespace")
            if sha256_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {relative}")
        return value

    def _namespace(self, study_id: str) -> Path:
        return self.root / "runs" / "rlmf" / _safe_id(study_id, "study_id")

    def _path(self, study_id: str, section: str, name: str, suffix: str) -> Path:
        return self._namespace(study_id) / _safe_id(section, "section") / (
            f"{_safe_id(name, 'name')}{suffix}"
        )

    def _bound_artifacts(
        self,
        study_id: str,
        paths: Iterable[str | Path] | Mapping[str, str | Path],
    ) -> dict[str, Path]:
        values = paths.values() if isinstance(paths, Mapping) else paths
        namespace = self._namespace(study_id)
        resolved_namespace = namespace.resolve()
        artifacts: dict[str, Path] = {}
        for value in values:
            path = Path(value)
            if not path.is_file() or not path.resolve().is_relative_to(resolved_namespace):
                raise ValueError("endpoint artifacts must exist under runs/rlmf/<study_id>")
            relative = path.resolve().relative_to(resolved_namespace).as_posix()
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
