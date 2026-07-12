from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
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


class SecondaryArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_json(self, run_id: str, section: str, name: str, value) -> Path:
        destination = self._path(run_id, section, name, ".json")
        payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
        _atomic_write_bytes(destination, payload)
        return destination

    def write_npz(self, run_id: str, section: str, name: str, **arrays: np.ndarray) -> Path:
        if not arrays:
            raise ValueError("write_npz requires at least one named array")
        destination = self._path(run_id, section, name, ".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
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
            os.replace(temporary, destination)
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

    def write_completion(
        self,
        run_id: str,
        endpoint: str,
        *,
        analysis_id: str,
        metrics_path: str | Path,
    ) -> Path:
        if not _SHA256.fullmatch(analysis_id):
            raise ValueError("analysis_id must be a lowercase SHA-256 fingerprint")
        expected = self._path(
            run_id, "comparisons", f"detection_{endpoint}", ".json"
        ).resolve()
        metrics = Path(metrics_path).resolve()
        if metrics != expected or not metrics.is_file():
            raise ValueError("metrics_path must be the endpoint's existing metrics file")
        return self.write_json(
            run_id,
            "comparisons",
            f"completion_{endpoint}",
            {
                "analysis_id": analysis_id,
                "metrics_sha256": _sha256(metrics),
            },
        )

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


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


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
    provenance["analysis_id"] = hashlib.sha256(_canonical_json(provenance)).hexdigest()
    return provenance


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
