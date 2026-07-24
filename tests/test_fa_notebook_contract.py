from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLAB = ROOT / "notebooks" / "06_familiarity_answerability_colab.ipynb"
ANALYSIS = ROOT / "notebooks" / "07_familiarity_answerability_analysis.ipynb"
CIRCUITS = ROOT / "notebooks" / "08_familiarity_answerability_circuits.ipynb"
CORE_LOCK = ROOT / "requirements" / "fa-core.lock"
CIRCUIT_LOCK = ROOT / "requirements" / "fa-circuits.lock"
RUNBOOK = ROOT / "docs" / "familiarity_answerability_runbook.md"
CLAIMS = ROOT / "docs" / "familiarity_answerability_claims.md"
COLAB_SCREENING = ROOT / "src" / "trajectory_extractor" / "fa_colab_screening.py"


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(cell["cell_type"] in {"markdown", "code"} for cell in notebook["cells"])
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_colab_notebook_is_orchestration_only_with_preflight_drive_and_resume():
    source = _source(COLAB)
    preflight = (
        ROOT / "src" / "trajectory_extractor" / "fa_colab_preflight.py"
    ).read_text(encoding="utf-8")
    for command in (
        "fa-run-colab-screening",
        "fa-audit-manifest",
        "fa-materialize-probe-rows",
        "fa-fit-probes",
        "fa-seal-selection",
        "fa-evaluate-probe-test",
        "fa-build-report",
    ):
        assert command in source
    assert "--resume" in source
    assert "drive.mount" in source
    assert "torch.cuda.get_device_properties" in preflight
    assert "shutil.disk_usage" in source
    assert "HF_TOKEN" in source
    assert 'userdata.get("HF_TOKEN")' in source
    assert 'VENV = Path("/content/fa-venv")' in source
    assert 'VENV_PYTHON = VENV / "bin/python"' in source
    assert '[sys.executable, "-m", "venv", str(VENV)]' in source
    assert '[str(VENV_PYTHON), "-m", "pip", "install"' in source
    assert '[str(VENV_PYTHON), "-m", "pip", "check"]' in source
    assert "No broken requirements found." in source
    assert "fa-colab-preflight" in source
    assert 'command = [str(VENV_PYTHON), "-m", "trajectory_extractor.cli"' in source
    assert "import torch" not in source
    assert "import transformers" not in source
    assert "import accelerate" not in source
    assert "from trajectory_extractor" not in source
    assert "LogisticRegression(" not in source
    assert "np.linalg" not in source
    assert "output_hidden_states=True" not in source


def test_colab_notebook_runs_the_frozen_source_v5_screening_before_protected_studies():
    source = _source(COLAB)
    implementation = COLAB_SCREENING.read_text(encoding="utf-8")
    for split in (
        "mechanism_train",
        "locked_validation",
        "behavior_test",
        "probe_test",
        "intervention_test",
    ):
        assert f'SourceV5Split("{split}"' in implementation
        assert "candidate_entities_{self.split}_v1.json" in implementation
        assert "screening_questions_{self.split}_v1.json" in implementation
        assert "synthetic_candidates_{self.split}_v1.json" in implementation
        actual = (
            ROOT
            / "data"
            / "fa"
            / "confirmatory_source_v5"
            / f"synthetic_candidates_{split}_v1.json"
        )
        assert actual.is_file()
        missing = (
            ROOT
            / "data"
            / "fa"
            / "confirmatory_source_v5"
            / f"synthetic_entities_{split}_v1.json"
        )
        assert not missing.exists()
    assert 'Path("data/fa/confirmatory_source_v5")' in implementation
    assert 'SOURCE_INTEGRITY_PATH = SOURCE_ROOT / "source_integrity_v1.json"' in (
        implementation
    )
    assert 'ROOT = str(REPO)' in source
    assert "execution_identity.json" in implementation
    assert "FA_LAUNCH_MANIFEST" in source
    assert "fa-study-launch.json" in source
    assert "runtime_observation" in source
    assert '"diff", "--quiet"' in source
    assert '"ls-files", "--others", "--exclude-standard"' in source
    assert 'path.startswith("runs/familiarity_answerability/")' in source
    assert "fa-run-colab-screening" in source
    assert "1152" in source
    assert "244" in source
    assert source.index("fa-run-colab-screening") < source.index(
        "fa-materialize-probe-rows"
    )
    stop = source.index("STOP_AFTER_SCREENING_ASSEMBLY")
    assert stop < source.index("fa-audit-manifest")
    assert source.index("fa-run-colab-screening") < stop


def test_analysis_notebook_consumes_sealed_artifacts_and_never_trains():
    source = _source(ANALYSIS)
    assert "fa-build-report" in source
    assert "MANIFEST.json" in source
    assert "verify_release_bundle" in source
    assert "fit(" not in source
    assert "generate(" not in source


def test_circuit_notebook_is_explicitly_optional_and_gate_checked():
    source = _source(CIRCUITS)
    assert "OPTIONAL" in source
    assert "circuit fidelity gate" in source.lower()
    assert "F1 plus F2A" in source
    assert "cannot rescue" in source
    assert "fa-select-circuit-cases" in source
    assert "fa-audit-circuit-fidelity" in source
    assert "not implemented" in source
    assert "subprocess.run" not in source


def test_dependency_profiles_are_separate_and_immutably_pinned():
    core = CORE_LOCK.read_text(encoding="utf-8")
    circuits = CIRCUIT_LOCK.read_text(encoding="utf-8")
    assert "circuit-tracer" not in core
    assert "circuit-tracer" in circuits
    for text in (core, circuits):
        requirements = [
            line
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert requirements
        assert all("==" in line or "@ git+https://" in line for line in requirements)
        assert "@main" not in text
        assert not re.search(r">=|~=|\*", text)


def test_runbook_and_claims_keep_execution_and_claim_boundaries_explicit():
    runbook = RUNBOOK.read_text(encoding="utf-8")
    claims = CLAIMS.read_text(encoding="utf-8")
    for phrase in (
        "8 GB",
        "Google Colab",
        "resume",
        "behavior_test",
        "probe_test",
        "intervention_test",
        "human audit",
    ):
        assert phrase.lower() in runbook.lower()
    for phrase in (
        "behavioral interaction",
        "pre-output prediction",
        "local causal",
        "not proof that the model uses the signal",
        "negative result",
    ):
        assert phrase.lower() in claims.lower()
