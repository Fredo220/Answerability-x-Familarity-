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
        "fa-run-screening",
        "fa-screen-entities",
        "fa-assemble-screened-matches",
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
    assert 'userdata.get("HF_TOKEN")' in source
    assert source.index('"pip", "install"') < source.index("import torch")
    assert '"torch" not in sys.modules' in source
    assert "repo_import_path = str(REPO.resolve())" in source
    assert "sys.path.insert(0, repo_import_path)" in source
    assert source.index("sys.path.insert(0, repo_import_path)") < source.index(
        "from trajectory_extractor.fa_artifacts import FAArtifactStore"
    )
    assert "lock_requirements" in source
    assert "metadata.version(requirement.name)" in source
    assert "lock_mismatches" in source
    assert "locked_conflicts" in source
    assert "Colab-preinstalled packages outside fa-core.lock" in source
    assert 'torch.__version__.split("+")[0] == "2.7.1"' in source
    assert 'transformers.__version__ == "4.57.1"' in source
    assert 'accelerate.__version__ == "1.12.0"' in source
    assert "LogisticRegression(" not in source
    assert "np.linalg" not in source
    assert "output_hidden_states=True" not in source


def test_colab_notebook_runs_the_frozen_source_v5_screening_before_protected_studies():
    source = _source(COLAB)
    for split in (
        "mechanism_train",
        "locked_validation",
        "behavior_test",
        "probe_test",
        "intervention_test",
    ):
        assert f"candidate_entities_{split}_v1.json" in source
        assert f"screening_questions_{split}_v1.json" in source
        assert f"synthetic_candidates_{split}_v1.json" in source
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
    assert "data/fa/confirmatory_source_v5/source_integrity_v1.json" in source
    assert 'ROOT = str(REPO)' in source
    assert "execution_identity.json" in source
    assert "FA_LAUNCH_MANIFEST" in source
    assert "fa-study-launch.json" in source
    assert "runtime_observation" in source
    assert '"diff", "--quiet"' in source
    assert '"ls-files", "--others", "--exclude-standard"' in source
    assert 'path.startswith("runs/familiarity_answerability/")' in source
    assert "checkpoint_split" in source
    assert "restore_split_checkpoint" in source
    assert "ColabSplitCheckpointStore" in source
    assert "verify_shard" in source
    assert "1152" in source
    assert "244" in source
    assert source.index("fa-run-screening") < source.index("fa-materialize-probe-rows")
    stop = source.index("STOP_AFTER_SCREENING_ASSEMBLY")
    assert stop < source.index("fa-audit-manifest")
    assert source.index('checkpoint_split(split, "completion")') < source.index(
        '"fa-screen-entities"'
    )


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
