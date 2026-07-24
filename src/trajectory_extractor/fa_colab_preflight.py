"""Fail-closed runtime verification for isolated Colab FA environments."""

from __future__ import annotations

import hashlib
import importlib
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


REQUIRED_RUNTIME_PINS = {
    "accelerate": "1.12.0",
    "torch": "2.7.1",
    "transformers": "4.57.1",
}


def read_exact_lock_pins(lock_path: str | Path) -> dict[str, str]:
    path = Path(lock_path)
    if not path.is_file():
        raise FileNotFoundError(f"lock file does not exist: {path}")
    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line or raw_line.isspace() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line[0].isspace():
            continue
        requirement = Requirement(raw_line)
        specifiers = tuple(requirement.specifier)
        if (
            requirement.url is not None
            or requirement.extras
            or requirement.marker is not None
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
            or specifiers[0].version.endswith(".*")
        ):
            raise ValueError(
                f"lock line {line_number} is not one exact == pin: {raw_line}"
            )
        name = canonicalize_name(requirement.name)
        if name in pins:
            raise ValueError(f"duplicate lock pin for {name}")
        pins[name] = specifiers[0].version
    if not pins:
        raise ValueError("lock file contains no exact pins")
    return pins


def run_colab_preflight(
    root: str | Path, lock_path: str | Path
) -> dict[str, Any]:
    repo_root = Path(root).resolve()
    lock = Path(lock_path)
    if not lock.is_absolute():
        lock = repo_root / lock
    lock = lock.resolve()
    if not lock.is_relative_to(repo_root):
        raise ValueError("lock path must resolve inside the repository")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("fa-colab-preflight must run inside an isolated venv")

    pins = read_exact_lock_pins(lock)
    for package, required in REQUIRED_RUNTIME_PINS.items():
        if pins.get(package) != required:
            raise RuntimeError(
                f"lock must pin {package}=={required}, observed {pins.get(package)!r}"
            )

    installed: dict[str, str] = {}
    for package, expected in sorted(pins.items()):
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"locked distribution is not installed: {package}=={expected}"
            ) from error
        if actual != expected:
            raise RuntimeError(
                f"installed {package} version {actual} does not match lock pin {expected}"
            )
        installed[package] = actual

    project = importlib.import_module("trajectory_extractor")
    project_path_value = getattr(project, "__file__", None)
    if not project_path_value:
        raise RuntimeError("trajectory_extractor import has no file path")
    project_path = Path(project_path_value).resolve()
    project_package_root = (repo_root / "src/trajectory_extractor").resolve()
    if not project_path.is_relative_to(project_package_root):
        raise RuntimeError(
            "trajectory_extractor import does not resolve inside the requested checkout"
        )

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    accelerate = importlib.import_module("accelerate")
    runtime_modules = {
        "torch": torch,
        "transformers": transformers,
        "accelerate": accelerate,
    }
    for package, module in runtime_modules.items():
        observed = str(getattr(module, "__version__", ""))
        if Version(observed).base_version != REQUIRED_RUNTIME_PINS[package]:
            raise RuntimeError(
                f"{package} runtime version {observed!r} does not match "
                f"{REQUIRED_RUNTIME_PINS[package]}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA reports no devices")
    properties = torch.cuda.get_device_properties(0)
    gpu_total_memory = int(properties.total_memory)
    if gpu_total_memory <= 0:
        raise RuntimeError("CUDA device reports non-positive total memory")

    lock_sha256 = hashlib.sha256(lock.read_bytes()).hexdigest()
    return {
        "status": "ready",
        "lock_path": str(lock),
        "lock_sha256": lock_sha256,
        "lock_pin_count": len(pins),
        "verified_pin_count": len(installed),
        "project_import_path": str(project_path),
        "project_version": metadata.version("trajectory-extractor"),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_version": platform.python_version(),
        "torch_version": installed["torch"],
        "torch_runtime_version": str(torch.__version__),
        "transformers_version": installed["transformers"],
        "transformers_runtime_version": str(transformers.__version__),
        "accelerate_version": installed["accelerate"],
        "accelerate_runtime_version": str(accelerate.__version__),
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "gpu_count": int(torch.cuda.device_count()),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "gpu_total_memory_bytes": gpu_total_memory,
        "gpu_total_memory_gib": round(gpu_total_memory / 1024**3, 3),
    }
