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
- Commit: `e7014cd18a18e11e6f6c10b897f7a009960d2e1b`
- Integration: `skillopt-sleep`
- Codex evaluation model: `gpt-5.4-mini`
- Target: `.agents/skills/fa-research-workflow/SKILL.md`
- Tasks: `skillopt/fa_research_workflow_tasks_v1.json`

The target skill is the only file SkillOpt may propose changing. Before any
external call, the wrapper creates a new standalone temporary workspace containing
only the reviewed task file, target skill, a generated README, and sandbox runtime
files. The Codex subprocess is additionally wrapped in a macOS sandbox profile that
denies reads from the source repository. Project-memory evolution and automatic
adoption are unavailable.

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

Read the staged path printed by `run` (normally under
`${TMPDIR:-/tmp}/fa-skillopt-workspace/.skillopt-sleep/staging/`). If validation
accepted the proposal, evaluate the untouched test split:

```bash
bash tools/run_fa_skillopt.sh evaluate-test /absolute/path/printed/by/run
```

The command writes a SHA-bound `test_evaluation.json` beside the proposal and
fails on any hard- or soft-score regression. It still does not modify the live
skill. Review the exact diff, apply it manually, and run repository tests in a
normal commit. `bash tools/run_fa_skillopt.sh adopt` intentionally refuses.

## Acceptance Criteria

Apply a proposal manually only when all conditions hold:

1. The proposal passes SkillOpt's held-out validation gate.
2. The reserved test evaluation exists, matches the proposal hash, and passes.
3. The exact edit is manually reviewed.
4. `python tools/validate_fa_skillopt.py` still passes.
5. Relevant repository tests pass.
6. No frozen protocol, threshold, dataset, result, or claim changes.

A rejected or no-change proposal is a valid outcome. It means the current skill
was not improved by this task set; it does not justify weakening the gate.
