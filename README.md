# Familiarity vs. Answerability in Language Models

Research infrastructure for a preregistered study of whether entity familiarity
changes answer attempts when relation-specific evidence is present or absent.
The confirmatory target is the pinned `google/gemma-2-2b-it` revision; the
Qwen3-0.6B profile is engineering rehearsal only.

The frozen Source-v5 confirmatory corpus is `not_evaluable`: corrected pinned
Gemma screening qualified 31 creative works, 25 organizations, 19 people, and
4 places, while the registered construction gate required 20 per domain. No
human packet or protected F1/F2A endpoint was opened. This is a failed
prerequisite, not evidence for or against the research hypothesis.

Source-v6 R9 subsequently passed its model-blind corpus audits but failed the
registered development instrument-readiness gate. Its immutable result and
screening rows are committed under [`docs/results`](docs/results). R10 is a
registered, one-shot held-out instrument follow-up on the previously unopened
`construction_validation` split. It changes only a narrowly specified
occupation-answer normalization rule and does not change the research
hypothesis, entity pool, protected endpoints, or confirmatory thresholds.

The planned Fellowship artifact had two stages:

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
- [Gemma double-BOS implementation correction](docs/amendments/2026-07-24-fa-gemma-double-bos-implementation-correction.md)
- [Human naturalness protocol](docs/fa_naturalness_rating_protocol.md)
- [Current confirmatory execution status](docs/confirmatory_execution_status_2026-07-24.md)
- [Execution plan](docs/superpowers/plans/2026-07-22-familiarity-answerability-implementation.md)
- [Runbook](docs/familiarity_answerability_runbook.md)
- [R10 Colab runbook](docs/fa_source_v6_r10_runbook.md)
- [R10 preregistered amendment](docs/amendments/2026-07-26-fa-source-v6-r10-heldout-instrument-validation.md)
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

The currently registered R10 execution is pinned to commit
`e2968387167350913b624393b4e4b28ed86d491d`. Add `HF_TOKEN` through Colab
Secrets; never paste or commit access tokens.

## Integrity Boundary

`behavior_test`, `probe_test`, and `intervention_test` are separate one-use,
hash-bound endpoints. F2A feature, layer, hyperparameter, and null selections
are frozen on `mechanism_train` and `locked_validation` before `probe_test` is
unlocked. Reports are regenerated from canonical closed endpoint evidence.

Negative and `not_evaluable` outcomes are publishable results. Thresholds and
claims must not be changed after protected outcomes are opened.
