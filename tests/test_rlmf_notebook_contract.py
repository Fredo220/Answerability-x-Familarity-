import json
import re
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/05_rlmf_colab.ipynb")


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text())


def _code_cells():
    return [
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    ]


def _all_code():
    return "\n".join(_code_cells())


def test_notebook_is_a_clean_thin_orchestrator():
    notebook = _notebook()
    code_cells = _code_cells()

    assert notebook["nbformat"] == 4
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("outputs") == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    for index, source in enumerate(code_cells):
        compile(source, f"{NOTEBOOK_PATH}:cell-{index}", "exec")
    assert all(not re.search(r"^\s*(def|class)\s+", source, re.MULTILINE) for source in code_cells)

    forbidden_training_logic = (
        "backward(",
        "optimizer.step(",
        "loss =",
        "advantages =",
        "torch.nn",
        "GRPOTrainer(",
        "RLMFTrainer(",
    )
    assert not any(token in _all_code() for token in forbidden_training_logic)
    assert "trajectory_extractor" in _all_code()
    assert "feature-dynamics" in _all_code()


def test_checkout_and_dependency_environment_are_exactly_verified():
    code = _all_code()

    assert "PROJECT_COMMIT" in code
    assert "re.fullmatch(r\"[0-9a-f]{40}\"" in code
    assert all(token in code for token in ('"git"', '"checkout"'))
    assert "rev-parse" in code
    assert "requirements-rlmf-colab.txt" in code
    assert all(token in code for token in ('"pip"', '"install"'))
    assert all(token in code for token in ('"-e"', '"."', '"--no-deps"'))
    assert "validate_runtime_versions" in code

    install_lines = [line for line in code.splitlines() if '"pip", "install"' in line]
    assert install_lines
    assert all(
        "requirements-rlmf-colab.txt" in line or '"-e", ".", "--no-deps"' in line
        for line in install_lines
    )


def test_modes_gpu_and_artifact_roots_are_explicit():
    code = _all_code()

    assert 'RUN_MODE = os.environ.get("RUN_MODE", "smoke")' in code
    assert '"smoke": "configs/rlmf_qwen06b_smoke.json"' in code
    assert '"pilot": "configs/rlmf_qwen06b_confirmatory.json"' in code
    assert '"confirmatory": "configs/rlmf_qwen06b_confirmatory.json"' in code
    assert "CONFIG_PATH = MODE_CONFIGS[RUN_MODE]" in code
    assert code.count("CONFIG_PATH =") == 1
    assert "MIN_GPU_MEMORY_GB = 14" in code
    assert "torch.cuda.is_available()" in code
    assert "total_memory" in code
    assert "/content/rlmf-scratch" in code
    assert "ARTIFACT_ROOT = SCRATCH_ROOT / \"artifacts\"" in code
    assert 'USE_DRIVE = os.environ.get("USE_DRIVE", "0") == "1"' in code
    assert "drive.mount" in code


def test_data_training_and_future_evaluation_are_delegated_in_registered_order():
    code = _all_code()

    assert "test_rlmf_preregistration.py" in code
    assert "verify_endpoint" in code
    assert "prepare-data" in code
    assert "--resume" in code

    pre_sft = code.index('"--stage", "pre-sft"')
    standard = code.index('"--arm", "standard"')
    rlmf = code.index('"--arm", "rlmf"')
    validation = code.index('"--split", "validation"')
    test = code.index('"--split", "test"')
    assert pre_sft < standard < rlmf < validation < test
    assert "rlmf-generate-rollouts" in code
    assert "designated-plus-20" in code


def test_checkpoint_exports_are_atomic_optional_and_non_overwriting():
    code = _all_code()

    assert "rlmf-export-checkpoint" in code
    assert "tempfile" in code
    assert "os.replace" in code
    assert "FileExistsError" in code
    assert "DRIVE_EXPORT_ROOT" in code
    assert "tempfile.TemporaryDirectory" in code
    assert 'archive_name = f"{config.study_id}' in code
    assert ".tar" in code
    assert '"--artifact-root", str(ARTIFACT_ROOT)' in code
    assert '"--output", str(local_archive)' in code
    assert '"--output", str(DRIVE_EXPORT_ROOT)' not in code
    assert "if USE_DRIVE:" in code


def test_final_cell_prints_machine_readable_noninterpretive_summary():
    notebook = _notebook()
    final = "".join(notebook["cells"][-1]["source"])

    assert notebook["cells"][-1]["cell_type"] == "code"
    assert "json.dumps" in final
    assert "sort_keys=True" in final
    assert '"status"' in final
    assert '"checkpoint_hashes"' in final
    assert not any(word in final.lower() for word in ("conclusion", "improved", "significant"))
