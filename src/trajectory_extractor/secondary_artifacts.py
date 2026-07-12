from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import socket
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


_SECTIONS = {
    "contrastive_vectors",
    "vector_dynamics",
    "activation_capping",
    "comparisons",
}
_SAFE_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class EndpointClaim:
    path: Path
    run_id: str
    endpoint: str
    claim_id: str


class SecondaryArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_json(self, run_id: str, section: str, name: str, value) -> Path:
        destination = self._path(run_id, section, name, ".json")
        payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
        _exclusive_write_bytes(destination, payload)
        return destination

    def write_npz(self, run_id: str, section: str, name: str, **arrays: np.ndarray) -> Path:
        if not arrays:
            raise ValueError("write_npz requires at least one named array")
        destination = self._path(run_id, section, name, ".npz")
        ensure_durable_directory(destination.parent)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, destination)
            temporary.unlink()
            temporary = None
            _fsync_directory(destination.parent)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return destination

    def read_json(self, run_id: str, section: str, name: str):
        return json.loads(self._path(run_id, section, name, ".json").read_text())

    def read_npz(self, run_id: str, section: str, name: str) -> dict[str, np.ndarray]:
        with np.load(self._path(run_id, section, name, ".npz")) as arrays:
            return {key: arrays[key].copy() for key in arrays.files}

    def assert_incomplete(self, run_id: str, endpoint: str) -> None:
        candidates = (
            self._path(run_id, "comparisons", f"detection_{endpoint}", ".json"),
            self._path(run_id, "comparisons", f"completion_{endpoint}", ".json"),
        )
        for path in candidates:
            if path.exists():
                raise FileExistsError(
                    f"secondary endpoint is immutable after metrics or completion: {path}"
                )

    def acquire_claim(self, run_id: str, endpoint: str) -> EndpointClaim:
        self.assert_incomplete(run_id, endpoint)
        path = self._path(run_id, "comparisons", f"claim_{endpoint}", ".json")
        claim = EndpointClaim(
            path=path,
            run_id=run_id,
            endpoint=endpoint,
            claim_id=uuid.uuid4().hex,
        )
        payload = json.dumps(
            {
                "claim_id": claim.claim_id,
                "run_id": run_id,
                "endpoint": endpoint,
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "status": "in_progress",
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        _exclusive_write_bytes(path, payload)
        self.assert_incomplete(run_id, endpoint)
        return claim

    def release_claim(self, claim: EndpointClaim) -> None:
        expected = self._path(
            claim.run_id, "comparisons", f"claim_{claim.endpoint}", ".json"
        )
        if claim.path != expected:
            raise ValueError("claim path does not match its endpoint")
        completion = self._path(
            claim.run_id, "comparisons", f"completion_{claim.endpoint}", ".json"
        )
        if not completion.is_file():
            raise RuntimeError("completion marker must exist before releasing a claim")
        payload = _read_json_object(claim.path, "endpoint claim")
        if payload.get("claim_id") != claim.claim_id:
            raise ValueError("endpoint claim ownership does not match")
        claim.path.unlink()

    def write_completion(
        self,
        run_id: str,
        endpoint: str,
        *,
        analysis_id: str,
        artifact_paths: list[str | Path],
    ) -> Path:
        if not _SHA256.fullmatch(analysis_id):
            raise ValueError("analysis_id must be a lowercase SHA-256 fingerprint")
        artifacts = self._validated_endpoint_artifacts(run_id, endpoint, artifact_paths)
        metrics = artifacts[f"comparisons/detection_{endpoint}.json"]
        metrics_payload = _read_json_object(metrics, "secondary metrics")
        _validate_metrics_analysis(metrics_payload, analysis_id)
        for _, path in sorted(artifacts.items()):
            _fsync_file(path)
            _fsync_directory(path.parent)
        destination = self._path(
            run_id, "comparisons", f"completion_{endpoint}", ".json"
        )
        payload = json.dumps(
            {
                "schema_version": 1,
                "analysis_id": analysis_id,
                "artifacts": {
                    relative: _sha256(path)
                    for relative, path in sorted(artifacts.items())
                },
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        _exclusive_write_bytes(destination, payload)
        return destination

    def verify_completion(self, run_id: str, endpoint: str) -> dict:
        marker_path = self._path(
            run_id, "comparisons", f"completion_{endpoint}", ".json"
        )
        marker = _read_json_object(marker_path, "secondary completion marker")
        analysis_id = marker.get("analysis_id")
        hashes = marker.get("artifacts")
        if marker.get("schema_version") != 1 or not isinstance(analysis_id, str):
            raise ValueError("secondary completion marker has an invalid schema")
        if not _SHA256.fullmatch(analysis_id) or not isinstance(hashes, dict):
            raise ValueError("secondary completion marker has invalid hashes")
        artifacts = self._validated_endpoint_artifacts(
            run_id,
            endpoint,
            [
                self.root / _safe_id(run_id) / "secondary" / path
                for path in hashes
            ],
        )
        if set(hashes) != set(artifacts):
            raise ValueError(
                "completion marker does not bind the expected endpoint artifacts"
            )
        for relative, path in artifacts.items():
            recorded = hashes.get(relative)
            if not isinstance(recorded, str) or not _SHA256.fullmatch(recorded):
                raise ValueError("secondary completion marker has invalid hashes")
            if _sha256(path) != recorded:
                raise ValueError(f"artifact hash mismatch: {relative}")
        metrics = artifacts[f"comparisons/detection_{endpoint}.json"]
        _validate_metrics_analysis(
            _read_json_object(metrics, "secondary metrics"), analysis_id
        )
        return marker

    def _validated_endpoint_artifacts(
        self,
        run_id: str,
        endpoint: str,
        artifact_paths: list[str | Path],
    ) -> dict[str, Path]:
        run = _safe_id(run_id)
        endpoint = _safe_id(endpoint)
        namespace = self.root / run / "secondary"
        relative_paths = (
            f"contrastive_vectors/{endpoint}.npz",
            f"vector_dynamics/{endpoint}.npz",
            f"comparisons/predictions_{endpoint}.npz",
            f"comparisons/detection_{endpoint}.json",
            f"comparisons/figures/method_comparison_{endpoint}.png",
            f"comparisons/figures/validation_metacognitive_risk_gap_{endpoint}.png",
        )
        expected = {relative: namespace / relative for relative in relative_paths}
        provided = {Path(path).absolute() for path in artifact_paths}
        expected_absolute = {path.absolute() for path in expected.values()}
        if provided != expected_absolute or len(artifact_paths) != len(expected):
            raise ValueError("artifact_paths must be the expected endpoint artifact paths")
        resolved_namespace = namespace.resolve()
        for path in expected.values():
            if not path.is_file() or not path.resolve().is_relative_to(
                resolved_namespace
            ):
                raise ValueError("artifact_paths must stay under the run secondary namespace")
        return expected

    def _path(self, run_id: str, section: str, name: str, suffix: str) -> Path:
        if section not in _SECTIONS:
            raise ValueError(f"unknown secondary section: {section}")
        return (
            self.root
            / _safe_id(run_id)
            / "secondary"
            / section
            / f"{_safe_id(name)}{suffix}"
        )


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
    missing = []
    current = destination
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(f"no existing ancestor for directory: {destination}")
        current = parent
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


def _validate_metrics_analysis(metrics: dict, analysis_id: str) -> None:
    provenance = metrics.get("analysis_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("secondary metrics must contain analysis_provenance")
    if (
        metrics.get("analysis_id") != analysis_id
        or provenance.get("analysis_id") != analysis_id
    ):
        raise ValueError("metrics, provenance, and completion analysis IDs must match")
    if analysis_fingerprint(provenance) != analysis_id:
        raise ValueError(
            "analysis ID does not match the canonical provenance fingerprint"
        )


def build_analysis_provenance(
    *,
    repo_root: str | Path,
    preregistration_path: str | Path,
    config_path: str | Path,
    run_root: str | Path,
    endpoint: str,
    pca_dims: int,
    ridge_alpha: float,
    bootstrap_count: int,
    permutation_method: str,
    permutation_seed: int,
    permutation_count: int,
    fdr_family: list[str],
) -> dict:
    _safe_id(endpoint)
    repository = _repository_provenance(Path(repo_root))
    preregistration = Path(preregistration_path)
    config = Path(config_path)
    run_directory = Path(run_root)
    manifest_path = run_directory / "manifest.json"
    manifest = _read_json_object(manifest_path, "run manifest")
    config_payload = _read_json_object(config, "config")
    dataset = _validated_dataset_provenance(manifest)
    model = _validated_model_provenance(
        run_directory / "examples", manifest=manifest, config=config_payload
    )
    if bootstrap_count < 1 or permutation_count < 1:
        raise ValueError("bootstrap and permutation counts must be positive")
    if permutation_seed != 42:
        raise ValueError("registered permutation seed must be 42")
    if not permutation_method or not fdr_family or any(not item for item in fdr_family):
        raise ValueError("permutation method and FDR family are required")

    provenance = {
        "schema_version": 1,
        "implementation": repository,
        "inputs": {
            "preregistration": {
                "path": str(preregistration),
                "sha256": _sha256_required(preregistration, "preregistration"),
            },
            "config": {
                "path": str(config),
                "sha256": _sha256_required(config, "config"),
            },
            "dataset": dataset,
        },
        "model": model,
        "analysis": {
            "endpoint": endpoint,
            "pca_dims": int(pca_dims),
            "ridge_alpha": float(ridge_alpha),
        },
        "bootstrap": {
            "method": "paired_entity_family_cluster_bootstrap",
            "unit": "entity_family",
            "seed": 42,
            "count": int(bootstrap_count),
        },
        "permutation": {
            "method": permutation_method,
            "seed": int(permutation_seed),
            "count": int(permutation_count),
            "fdr_family": list(fdr_family),
        },
    }
    provenance["analysis_id"] = analysis_fingerprint(provenance)
    return provenance


def analysis_fingerprint(provenance: dict) -> str:
    if not isinstance(provenance, dict):
        raise ValueError("analysis provenance must be a JSON object")
    semantic = _semantic_provenance(provenance)
    return hashlib.sha256(_canonical_json(semantic)).hexdigest()


def _semantic_provenance(value):
    if isinstance(value, dict):
        return {
            key: _semantic_provenance(item)
            for key, item in value.items()
            if key not in {"analysis_id", "path", "manifest_path"}
        }
    if isinstance(value, list):
        return [_semantic_provenance(item) for item in value]
    return value


def _repository_provenance(repo_root: Path) -> dict:
    try:
        commit = _git(repo_root, "rev-parse", "HEAD").strip()
        dirty = bool(
            _git(repo_root, "status", "--porcelain", "--untracked-files=no").strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("critical git provenance is unavailable") from error
    if not commit:
        raise ValueError("critical git provenance is incomplete")
    source_root = repo_root / "src" / "trajectory_extractor"
    source_paths = sorted(source_root.rglob("*.py"))
    if not source_paths:
        raise ValueError("critical implementation source is unavailable")
    digest = hashlib.sha256()
    for path in source_paths:
        relative = path.relative_to(repo_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "git_commit": commit,
        "tracked_dirty": dirty,
        "source_sha256": digest.hexdigest(),
    }


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _validated_dataset_provenance(manifest: dict) -> dict:
    dataset = manifest.get("dataset")
    required = ("path", "sha256", "manifest_path", "manifest_sha256")
    if not isinstance(dataset, dict) or any(not dataset.get(key) for key in required):
        raise ValueError("run manifest has incomplete dataset provenance")
    if not _SHA256.fullmatch(str(dataset["sha256"])) or not _SHA256.fullmatch(
        str(dataset["manifest_sha256"])
    ):
        raise ValueError("run manifest dataset hashes must be lowercase SHA-256 values")
    return {key: dataset[key] for key in required}


def _validated_model_provenance(
    examples_directory: Path, *, manifest: dict, config: dict
) -> dict:
    paths = sorted(examples_directory.glob("*.json"))
    if not paths:
        raise ValueError("run provenance has no example records")
    values = {"model_id": set(), "model_revision": set(), "resolved_model_revision": set()}
    for path in paths:
        payload = _read_json_object(path, "example provenance")
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"example has no provenance object: {path}")
        for key in values:
            value = provenance.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"example provenance is missing critical field {key}: {path}")
            values[key].add(value)
    if len(values["resolved_model_revision"]) != 1:
        raise ValueError("run provenance must contain one unique resolved model revision")
    if len(values["model_id"]) != 1 or len(values["model_revision"]) != 1:
        raise ValueError("run provenance must contain one unique requested model identity")
    model_id = next(iter(values["model_id"]))
    requested = next(iter(values["model_revision"]))
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict):
        raise ValueError("run manifest config provenance is missing")
    expected_pairs = (
        (manifest_config.get("model_id"), model_id),
        (manifest_config.get("model_revision"), requested),
        (config.get("model_id"), model_id),
        (config.get("model_revision"), requested),
    )
    if any(expected != actual for expected, actual in expected_pairs):
        raise ValueError("config, manifest, and run model provenance do not agree")
    return {
        "id": model_id,
        "requested_revision": requested,
        "resolved_revision": next(iter(values["resolved_model_revision"])),
    }


def _read_json_object(path: Path, description: str) -> dict:
    if not path.is_file():
        raise ValueError(f"critical {description} is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"critical {description} must be a JSON object: {path}")
    return payload


def _sha256_required(path: Path, description: str) -> str:
    if not path.is_file():
        raise ValueError(f"critical {description} file is missing: {path}")
    return _sha256(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _safe_id(value: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(
            "identifier must match the exact safe grammar: alphanumeric boundaries with "
            "only alphanumeric, dot, underscore, or hyphen characters"
        )
    return value
