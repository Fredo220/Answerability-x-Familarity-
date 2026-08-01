---
name: fa-research-workflow
description: Execute and debug the Familiarity-vs-Answerability study without reopening frozen choices, losing valid work, or overstating evidence.
---

# Familiarity-vs-Answerability Research Workflow

## Continue From Verified State

Read the active runbook, amendment, configuration, Git state, and latest
machine-readable artifact. Identify the last verified gate and continue from
the smallest missing gate. Reuse valid shards and manifests. Preserve a failed
command and diagnose whether it is an input, implementation, infrastructure,
feasibility, or empirical failure before changing anything.

For an interrupted run with matching hashes and execution identity, preserve
the completed shards and resume the same sealed command. Do not restart the
study merely because a runtime disconnected.

## Evidence Language

Use the strongest justified label only:

- `implemented`: code exists;
- `software_verified`: deterministic tests and audits pass;
- `feasible`: runtime and corpus gates pass;
- `empirically_supported` or `not_supported`: a frozen endpoint was evaluated;
- `not_evaluable`: a prerequisite gate failed, so the hypothesis was not tested.

Software verification is not empirical evidence. Preserve null, negative, and
`not_evaluable` results rather than converting them into infrastructure claims.

## Scientific Boundaries

Do not optimize data, thresholds, exclusions, parsers, or claims after seeing
the outcome. Repairs based on open development evidence require a documented
amendment and fresh evaluation units. Never use SkillOpt to inspect, summarize,
score, or tune against protected prompts, labels, activations, generations, or
endpoint results.

Keep model generation on the registered Colab path. Keep deterministic tests,
audits, artifact verification, and reporting local where possible. Preserve
model, tokenizer, template, parser, configuration, commit, and artifact hashes.

## SkillOpt Adoption Boundary

SkillOpt is a development aid, not part of the scientific estimator. Run it
only on the committed open-development workflow tasks. Require held-out
improvement, inspect the staged report and exact edit, and rerun repository
tests before adoption. There is no auto-adopt: applying a proposal always
requires explicit human approval.
