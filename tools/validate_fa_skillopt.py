#!/usr/bin/env python3
"""Validate the bounded SkillOpt development integration."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


KNOWN_CHECKS = {
    "contains",
    "max_chars",
    "min_chars",
    "regex",
    "section_present",
}
FORBIDDEN_TASK_TEXT = (
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
TARGET_PATH = ".agents/skills/fa-research-workflow/SKILL.md"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate(project: Path) -> None:
    project = project.resolve()
    lock = _load(project / "skillopt" / "skillopt.lock.json")
    if lock != {
        "repository": "https://github.com/microsoft/SkillOpt.git",
        "commit": "e7014cd",
        "integration": "skillopt-sleep",
    }:
        raise ValueError("SkillOpt lock does not match the reviewed upstream pin")

    tasks_path = project / "skillopt" / "fa_research_workflow_tasks_v1.json"
    payload = _load(tasks_path)
    if payload.get("format") != "skillopt_sleep.tasks.v1":
        raise ValueError("unsupported SkillOpt task format")
    if payload.get("reviewed") is not True:
        raise ValueError("real backends require an explicitly reviewed task file")
    if payload.get("target_skill_path") != TARGET_PATH:
        raise ValueError("task file points at an unexpected target skill")

    serialized = tasks_path.read_text(encoding="utf-8").lower()
    leaked = [term for term in FORBIDDEN_TASK_TEXT if term in serialized]
    if leaked:
        raise ValueError(f"task file contains protected or outcome text: {leaked}")

    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) < 9:
        raise ValueError("task file needs at least nine reviewed tasks")
    identifiers: set[str] = set()
    split_counts = {"train": 0, "val": 0, "test": 0}
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("every task must be an object")
        identifier = str(task.get("id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError("task identifiers must be nonempty and unique")
        identifiers.add(identifier)
        split = task.get("split")
        if split not in split_counts:
            raise ValueError(f"task {identifier} has an invalid split")
        split_counts[split] += 1
        if task.get("origin") != "real":
            raise ValueError(f"task {identifier} is not a reviewed real task")
        if task.get("reference_kind") != "rule":
            raise ValueError(f"task {identifier} must use deterministic rule scoring")
        checks = task.get("judge", {}).get("checks", [])
        if not checks:
            raise ValueError(f"task {identifier} has no checks")
        for check in checks:
            operation = check.get("op")
            argument = check.get("arg")
            if operation not in KNOWN_CHECKS:
                raise ValueError(f"task {identifier} uses unsafe check {operation!r}")
            if operation == "regex":
                re.compile(str(argument))
            if operation in {"max_chars", "min_chars"}:
                if isinstance(argument, bool) or not isinstance(argument, int):
                    raise ValueError(f"task {identifier} has a noninteger bound")
    if min(split_counts.values()) < 3:
        raise ValueError("train, validation, and test each need at least three tasks")

    skill = (project / TARGET_PATH).read_text(encoding="utf-8").lower()
    required = (
        "smallest missing gate",
        "not_evaluable",
        "do not optimize",
        "protected",
        "no auto-adopt",
        "software verification",
    )
    missing = [term for term in required if term not in skill]
    if missing:
        raise ValueError(f"target skill is missing research boundaries: {missing}")


def validate_staging(project: Path, staging: Path) -> None:
    project = project.resolve()
    staging = staging.resolve()
    manifest = _load(staging / "manifest.json")
    expected_skill = (project / TARGET_PATH).resolve()
    if manifest.get("accepted") is not True:
        raise ValueError("only an accepted SkillOpt proposal may be adopted")
    if manifest.get("has_skill") is not True:
        raise ValueError("proposal does not contain a target-skill improvement")
    if manifest.get("has_memory") is not False:
        raise ValueError("project-memory edits are outside this integration")
    if Path(str(manifest.get("live_skill_path", ""))).resolve() != expected_skill:
        raise ValueError("proposal targets an unexpected live skill")
    proposed = (staging / "proposed_SKILL.md").read_text(encoding="utf-8").lower()
    required = (
        "smallest missing gate",
        "not_evaluable",
        "do not optimize",
        "protected",
        "no auto-adopt",
        "software verification",
    )
    missing = [term for term in required if term not in proposed]
    if missing:
        raise ValueError(f"proposal weakens research boundaries: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--staging", type=Path)
    args = parser.parse_args()
    validate(args.project)
    if args.staging is not None:
        validate_staging(args.project, args.staging)
    print("SkillOpt integration valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
