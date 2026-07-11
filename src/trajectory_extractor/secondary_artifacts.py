from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np


_SECTIONS = {
    "contrastive_vectors",
    "vector_dynamics",
    "activation_capping",
    "comparisons",
}


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


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError("identifier must contain at least one safe character")
    return safe
