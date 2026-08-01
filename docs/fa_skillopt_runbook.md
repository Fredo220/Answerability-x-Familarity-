# SkillOpt Development Runbook

## Purpose

SkillOpt improves the agent instruction used to operate this repository. It is
not part of the Familiarity-vs.-Answerability estimator, model, dataset, or
scientific evidence. The integration is restricted to nine reviewed workflow
tasks derived from recurring engineering failures.

The committed task file contains no protected prompts, labels, generations,
activations, endpoint results, or hypothesis outcomes. Train, validation, and
test tasks are explicit. Rule-based checks run locally.

## Pin and Targets

- Upstream: `https://github.com/microsoft/SkillOpt.git`
- Commit: `e7014cd`
- Integration: `skillopt-sleep`
- Target: `.agents/skills/fa-research-workflow/SKILL.md`
- Tasks: `skillopt/fa_research_workflow_tasks_v1.json`

The target skill is the only file SkillOpt may propose changing. Project
memory evolution is not used. Automatic adoption is unavailable.

## Commands

Validate the committed boundary:

```bash
bash tools/run_fa_skillopt.sh validate
```

Run the deterministic, zero-provider plumbing smoke:

```bash
bash tools/run_fa_skillopt.sh smoke
```

Run a real Codex-backed evaluation without staging changes:

```bash
bash tools/run_fa_skillopt.sh dry-run
```

Generate a gated proposal:

```bash
bash tools/run_fa_skillopt.sh run
```

Read `.skillopt-sleep/staging/<run>/report.md`. Confirm that validation improves,
the test result does not regress, and the edit does not weaken research
boundaries. Then run repository tests. Adoption requires a separate explicit
command:

```bash
FA_SKILLOPT_APPROVE_ADOPT=YES bash tools/run_fa_skillopt.sh adopt
```

## Acceptance Criteria

Adopt only when all conditions hold:

1. The proposal passes SkillOpt's held-out validation gate.
2. Test-task performance does not regress.
3. The exact edit is manually reviewed.
4. `python tools/validate_fa_skillopt.py` still passes.
5. Relevant repository tests pass.
6. No frozen protocol, threshold, dataset, result, or claim changes.

A rejected or no-change proposal is a valid outcome. It means the current skill
was not improved by this task set; it does not justify weakening the gate.
