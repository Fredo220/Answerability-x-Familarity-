#!/usr/bin/env python3
"""Evaluate one staged SkillOpt proposal on the reserved test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from skillopt_sleep.backend import get_backend
from skillopt_sleep.replay import aggregate_scores, replay_one
from skillopt_sleep.tasks_file import load_tasks_file


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--codex-path", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-tasks-sha256", required=True)
    parser.add_argument("--skillopt-commit", required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    staging = args.staging.resolve()
    skill_path = workspace / ".agents/skills/fa-research-workflow/SKILL.md"
    tasks_path = workspace / "skillopt/fa_research_workflow_tasks_v1.json"
    candidate_path = staging / "proposed_SKILL.md"
    if sha256(skill_path) != args.expected_baseline_sha256:
        raise ValueError("baseline skill differs from the source-repository snapshot")
    if sha256(tasks_path) != args.expected_tasks_sha256:
        raise ValueError("task file differs from the reviewed source-repository snapshot")
    tasks, _ = load_tasks_file(str(tasks_path))
    test_tasks = [task for task in tasks if task.split == "test"]
    if len(test_tasks) < 3:
        raise ValueError("reserved test split must contain at least three tasks")

    backend = get_backend(
        "codex", model=args.model, codex_path=args.codex_path, project_dir=str(workspace)
    )
    def evaluate(skill: str):
        results = []
        for task in test_tasks:
            result = replay_one(backend, task, skill, "")
            error = getattr(backend, "last_call_error", "")
            if error or not result.response.strip():
                raise RuntimeError(
                    f"Codex backend failed on {task.id}: {error or 'empty response'}"
                )
            results.append((task, result))
        return results

    baseline = evaluate(skill_path.read_text(encoding="utf-8"))
    candidate = evaluate(candidate_path.read_text(encoding="utf-8"))
    baseline_hard, baseline_soft = aggregate_scores(baseline)
    candidate_hard, candidate_soft = aggregate_scores(candidate)
    passed = candidate_hard >= baseline_hard and candidate_soft >= baseline_soft
    payload = {
        "format": "fa.skillopt_test.v1",
        "passed": passed,
        "baseline": {"hard": baseline_hard, "soft": baseline_soft},
        "candidate": {"hard": candidate_hard, "soft": candidate_soft},
        "baseline_sha256": sha256(skill_path),
        "candidate_sha256": sha256(candidate_path),
        "tasks_sha256": sha256(tasks_path),
        "model": args.model,
        "skillopt_commit": args.skillopt_commit,
        "codex_version": subprocess.run(
            [args.codex_path, "--version"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "manifest_sha256": sha256(staging / "manifest.json"),
        "validation_report_sha256": sha256(staging / "report.json"),
        "tasks": [
            {
                "id": task.id,
                "baseline_hard": base.hard,
                "baseline_soft": base.soft,
                "candidate_hard": cand.hard,
                "candidate_soft": cand.soft,
            }
            for (task, base), (_, cand) in zip(baseline, candidate, strict=True)
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
