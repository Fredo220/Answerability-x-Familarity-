# Familiarity vs. Answerability in Language Models

Research infrastructure for a preregistered study of whether entity familiarity
changes answer attempts when relation-specific evidence is present or absent.
The confirmatory target is the pinned `google/gemma-2-2b-it` revision; the
Qwen3-0.6B profile is engineering rehearsal only.

This repository currently contains the study implementation, not a confirmatory
empirical result. The minimum Fellowship artifact has two stages:

1. **F1, behavioral interaction:** test familiarity by answerability with
   matched screened-real and synthetic entities.
2. **F2A, mechanistic pilot:** test whether prompt-end internal activations add
   held-out predictive information under reciprocal condition transfer and
   registered nulls.

F2B activation interchange is gated and currently deferred. Attribution graphs
are optional, exploratory, and cannot rescue a failed F1 or F2A result. The
study makes no claim of a universal hallucination mechanism or generalization
beyond the registered task.

## Study Documents

- [Preregistration](docs/familiarity_answerability_preregistration.md)
- [Protocol amendment](docs/familiarity_answerability_protocol_amendment_2026-07-22.md)
- [Confirmatory corpus reserve amendment](docs/amendments/2026-07-24-fa-confirmatory-corpus-reserves.md)
- [Human naturalness protocol](docs/fa_naturalness_rating_protocol.md)
- [Current confirmatory execution status](docs/confirmatory_execution_status_2026-07-24.md)
- [Execution plan](docs/superpowers/plans/2026-07-22-familiarity-answerability-implementation.md)
- [Runbook](docs/familiarity_answerability_runbook.md)
- [Claim ladder](docs/familiarity_answerability_claims.md)

## Compute Boundary

- The 8 GB local machine runs tests, audits, probe analysis, reporting, and
  release verification.
- A resumable Google Colab GPU run performs confirmatory Gemma generation and
  activation extraction.
- The human naturalness audit remains a required external input.

Use Python 3.12 in this worktree:

```bash
../../.venv/bin/python -m pytest -q
```

The Colab workflow is in
[`notebooks/06_familiarity_answerability_colab.ipynb`](notebooks/06_familiarity_answerability_colab.ipynb).
Scientific logic remains in tested Python modules; notebooks only orchestrate
resumable CLI transactions.

## Integrity Boundary

`behavior_test`, `probe_test`, and `intervention_test` are separate one-use,
hash-bound endpoints. F2A feature, layer, hyperparameter, and null selections
are frozen on `mechanism_train` and `locked_validation` before `probe_test` is
unlocked. Reports are regenerated from canonical closed endpoint evidence.

Negative and `not_evaluable` outcomes are publishable results. Thresholds and
claims must not be changed after protected outcomes are opened.

## Legacy Code

Older research modules remain in the repository for provenance and regression
coverage. They are not dependencies of the active Familiarity-vs-Answerability
study and are not part of its confirmatory claims.
