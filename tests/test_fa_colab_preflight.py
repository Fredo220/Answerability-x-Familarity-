from pathlib import Path
from types import SimpleNamespace

import pytest

import trajectory_extractor.fa_colab_preflight as preflight


def _write_lock(path: Path) -> dict[str, str]:
    pins = {
        "accelerate": "1.12.0",
        "torch": "2.7.1",
        "transformers": "4.57.1",
    }
    path.parent.mkdir(parents=True)
    path.write_text(
        "# generated lock\n"
        + "".join(f"{name}=={version}\n" for name, version in pins.items()),
        encoding="utf-8",
    )
    return pins


def _fake_modules(project_file: Path, *, cuda_available: bool = True):
    cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: 1 if cuda_available else 0,
        get_device_name=lambda index: "NVIDIA Test GPU",
        get_device_properties=lambda index: SimpleNamespace(
            total_memory=24 * 1024**3
        ),
    )
    return {
        "trajectory_extractor": SimpleNamespace(__file__=str(project_file)),
        "torch": SimpleNamespace(
            __version__="2.7.1+cu126",
            version=SimpleNamespace(cuda="12.6"),
            cuda=cuda,
        ),
        "transformers": SimpleNamespace(__version__="4.57.1"),
        "accelerate": SimpleNamespace(__version__="1.12.0"),
    }


def test_lock_parser_rejects_any_non_exact_requirement(tmp_path):
    lock = tmp_path / "fa-core.lock"
    lock.write_text("torch==2.7.1\ntransformers>=4.57.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact == pin"):
        preflight.read_exact_lock_pins(lock)


def test_preflight_verifies_all_pins_project_import_and_cuda(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    project_file = root / "src/trajectory_extractor/__init__.py"
    project_file.parent.mkdir(parents=True)
    project_file.write_text("", encoding="utf-8")
    lock = root / "requirements/fa-core.lock"
    pins = _write_lock(lock)
    modules = _fake_modules(project_file)
    monkeypatch.setattr(
        preflight.metadata,
        "version",
        lambda name: (
            "0.1.0"
            if name == "trajectory-extractor"
            else pins[name.replace("_", "-").lower()]
        ),
    )
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(preflight.sys, "prefix", "/content/fa-venv")
    monkeypatch.setattr(preflight.sys, "base_prefix", "/usr")
    monkeypatch.setattr(preflight.sys, "executable", "/content/fa-venv/bin/python")

    payload = preflight.run_colab_preflight(
        root, Path("requirements/fa-core.lock")
    )

    assert payload["status"] == "ready"
    assert payload["lock_pin_count"] == 3
    assert payload["verified_pin_count"] == 3
    assert payload["project_import_path"] == str(project_file.resolve())
    assert payload["torch_version"] == "2.7.1"
    assert payload["torch_runtime_version"] == "2.7.1+cu126"
    assert payload["transformers_version"] == "4.57.1"
    assert payload["accelerate_version"] == "1.12.0"
    assert payload["cuda_available"] is True
    assert payload["cuda_version"] == "12.6"
    assert payload["gpu_name"] == "NVIDIA Test GPU"
    assert payload["gpu_total_memory_gib"] == 24.0
    assert payload["ram_total_bytes"] > 0
    assert payload["disk_free_bytes"] > 0


def test_preflight_rejects_python_other_than_lock_target(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    lock = root / "requirements/fa-core.lock"
    _write_lock(lock)
    monkeypatch.setattr(preflight.sys, "prefix", "/content/fa-venv")
    monkeypatch.setattr(preflight.sys, "base_prefix", "/usr")
    monkeypatch.setattr(preflight.sys, "version_info", (3, 11, 9))

    with pytest.raises(RuntimeError, match="requires Python 3.12"):
        preflight.run_colab_preflight(root, lock)


def test_preflight_fails_closed_on_pin_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    project_file = root / "src/trajectory_extractor/__init__.py"
    project_file.parent.mkdir(parents=True)
    project_file.write_text("", encoding="utf-8")
    lock = root / "requirements/fa-core.lock"
    pins = _write_lock(lock)
    modules = _fake_modules(project_file)
    monkeypatch.setattr(
        preflight.metadata,
        "version",
        lambda name: (
            "9.9.9"
            if name == "transformers"
            else (
                "0.1.0"
                if name == "trajectory-extractor"
                else pins[name.replace("_", "-").lower()]
            )
        ),
    )
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(preflight.sys, "prefix", "/content/fa-venv")
    monkeypatch.setattr(preflight.sys, "base_prefix", "/usr")

    with pytest.raises(RuntimeError, match="transformers.*9.9.9.*4.57.1"):
        preflight.run_colab_preflight(root, lock)


def test_preflight_fails_closed_without_cuda(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    project_file = root / "src/trajectory_extractor/__init__.py"
    project_file.parent.mkdir(parents=True)
    project_file.write_text("", encoding="utf-8")
    lock = root / "requirements/fa-core.lock"
    pins = _write_lock(lock)
    modules = _fake_modules(project_file, cuda_available=False)
    monkeypatch.setattr(
        preflight.metadata,
        "version",
        lambda name: (
            "0.1.0"
            if name == "trajectory-extractor"
            else pins[name.replace("_", "-").lower()]
        ),
    )
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(preflight.sys, "prefix", "/content/fa-venv")
    monkeypatch.setattr(preflight.sys, "base_prefix", "/usr")

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        preflight.run_colab_preflight(root, lock)
