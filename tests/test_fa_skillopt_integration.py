from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.prepare_fa_skillopt_workspace import prepare
from tools.validate_fa_skillopt import validate, validate_staging


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "skillopt" / "skillopt.lock.json"
TASKS = ROOT / "skillopt" / "fa_research_workflow_tasks_v1.json"
TARGET = ROOT / ".agents" / "skills" / "fa-research-workflow" / "SKILL.md"
RUNNER = ROOT / "tools" / "run_fa_skillopt.sh"
VALIDATOR = ROOT / "tools" / "validate_fa_skillopt.py"


def test_skillopt_integration_is_pinned_reviewed_and_split():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))

    assert lock == {
        "repository": "https://github.com/microsoft/SkillOpt.git",
        "commit": "e7014cd18a18e11e6f6c10b897f7a009960d2e1b",
        "integration": "skillopt-sleep",
        "codex_model": "gpt-5.4-mini",
    }
    assert tasks["format"] == "skillopt_sleep.tasks.v1"
    assert tasks["reviewed"] is True
    assert tasks["target_skill_path"] == (
        ".agents/skills/fa-research-workflow/SKILL.md"
    )
    assert len(tasks["tasks"]) >= 9
    assert {task["split"] for task in tasks["tasks"]} == {"train", "val", "test"}
    assert len({task["id"] for task in tasks["tasks"]}) == len(tasks["tasks"])
    assert all(task["origin"] == "real" for task in tasks["tasks"])
    assert all(task["reference_kind"] == "rule" for task in tasks["tasks"])


def test_skillopt_tasks_do_not_contain_protected_data_or_outcomes():
    text = TASKS.read_text(encoding="utf-8").lower()
    forbidden = (
        "data/fa/confirmatory",
        "behavior_test",
        "probe_test",
        "intervention_test",
        "locked_validation",
        "supported hypothesis",
        "hypothesis confirmed",
        "gate_passed",
        "endpoint result",
    )
    assert not any(value in text for value in forbidden)


def test_target_skill_preserves_research_boundaries():
    text = TARGET.read_text(encoding="utf-8").lower()
    for required in (
        "smallest missing gate",
        "not_evaluable",
        "do not optimize",
        "protected",
        "no auto-adopt",
        "software verification",
    ):
        assert required in text


def test_runner_defaults_to_mock_and_blocks_automatic_adoption():
    text = RUNNER.read_text(encoding="utf-8")
    assert 'ACTION="${1:-smoke}"' in text
    assert "--backend mock" in text
    assert "--auto-adopt" not in text
    assert "SKILLOPT_COMMIT" in text
    assert "validate_fa_skillopt.py" in text
    assert "codex_preflight" in text
    assert "project memory" in text
    assert 'echo "Automatic adoption is disabled.' in text
    assert "evaluate-test" in text
    assert 'project "$SAFE_ROOT"' in text


def test_validator_accepts_committed_integration():
    completed = subprocess.run(
        ["python3", str(VALIDATOR), "--project", str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "SkillOpt integration valid" in completed.stdout


def test_validator_rejects_protected_task_text(tmp_path: Path):
    project = tmp_path / "project"
    shutil.copytree(ROOT / "skillopt", project / "skillopt")
    shutil.copytree(ROOT / ".agents", project / ".agents")
    path = project / "skillopt" / "fa_research_workflow_tasks_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tasks"][0]["context_excerpt"] = "data/fa/confirmatory/private.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="protected or outcome text"):
        validate(project)


def test_validator_rejects_memory_edits_before_adoption(tmp_path: Path):
    safe = tmp_path / "safe"
    staging = safe / ".skillopt-sleep" / "staging" / "run"
    staging.mkdir(parents=True)
    (staging / "manifest.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "has_skill": True,
                "has_memory": True,
                "live_skill_path": str(safe / TARGET.relative_to(ROOT)),
            }
        ),
        encoding="utf-8",
    )
    (staging / "proposed_SKILL.md").write_text(
        TARGET.read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="memory edits"):
        validate_staging(ROOT, staging)


def test_validator_rejects_changes_to_immutable_boundaries(tmp_path: Path):
    safe = tmp_path / "safe"
    staging = safe / ".skillopt-sleep" / "staging" / "run"
    staging.mkdir(parents=True)
    proposed = TARGET.read_text(encoding="utf-8").replace(
        "Never use SkillOpt to inspect", "Use SkillOpt to inspect"
    )
    (staging / "proposed_SKILL.md").write_text(proposed, encoding="utf-8")
    (staging / "manifest.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "has_skill": True,
                "has_memory": False,
                "live_skill_path": str(safe / TARGET.relative_to(ROOT)),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="immutable section"):
        validate_staging(ROOT, staging)


def test_sanitized_workspace_contains_only_allowlisted_inputs(tmp_path: Path):
    project = tmp_path / "project"
    shutil.copytree(ROOT / "skillopt", project / "skillopt")
    shutil.copytree(ROOT / ".agents", project / ".agents")
    (project / "protected-result.json").write_text("secret", encoding="utf-8")

    safe = prepare(project, tmp_path / "sanitized", tmp_path)
    files = {
        path.relative_to(safe).as_posix()
        for path in safe.rglob("*")
        if path.is_file()
    }
    assert files == {
        ".agents/skills/fa-research-workflow/SKILL.md",
        "README.md",
        "skillopt/fa_research_workflow_tasks_v1.json",
    }
    assert not any(path.is_symlink() for path in safe.rglob("*"))


def test_sanitized_workspace_must_stay_outside_project(tmp_path: Path):
    project = tmp_path / "project"
    shutil.copytree(ROOT / "skillopt", project / "skillopt")
    shutil.copytree(ROOT / ".agents", project / ".agents")

    with pytest.raises(ValueError, match="outside the source project"):
        prepare(project, project / ".skillopt-workspace", project)


def test_sanitized_workspace_cannot_delete_outside_allowed_root(tmp_path: Path):
    project = tmp_path / "project"
    shutil.copytree(ROOT / "skillopt", project / "skillopt")
    shutil.copytree(ROOT / ".agents", project / ".agents")
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="child of the allowed root"):
        prepare(project, victim, tmp_path / "approved")
    assert marker.read_text(encoding="utf-8") == "keep"


def test_sanitized_workspace_refuses_to_replace_existing_state(tmp_path: Path):
    project = tmp_path / "project"
    shutil.copytree(ROOT / "skillopt", project / "skillopt")
    shutil.copytree(ROOT / ".agents", project / ".agents")
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "staged-proposal.json"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="new and empty"):
        prepare(project, destination, tmp_path)
    assert marker.read_text(encoding="utf-8") == "keep"
