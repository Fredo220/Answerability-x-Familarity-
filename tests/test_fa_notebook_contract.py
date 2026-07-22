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


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(cell["cell_type"] in {"markdown", "code"} for cell in notebook["cells"])
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_colab_notebook_is_orchestration_only_with_preflight_drive_and_resume():
    source = _source(COLAB)
    for command in (
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
    assert "torch.cuda.get_device_properties" in source
    assert "shutil.disk_usage" in source
    assert "HF_TOKEN" in source
    assert "LogisticRegression(" not in source
    assert "np.linalg" not in source
    assert "output_hidden_states=True" not in source


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
