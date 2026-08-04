# Same-String Answerability Causal Pilot Implementation Plan

> **Execution mode:** use test-driven development, bounded subagent review, and
> validation-gated SkillOpt process review. Preserve every closed study.

**Goal:** Test whether the v3 training-only answerability direction causally
changes Gemma 2 2B's code-versus-`UNKNOWN` response margin on fresh Same-String
units.

**Architecture:** Add a small causal-study module beside the v3 representation
module, a real Hugging Face Gemma residual-hook adapter, deterministic artifact
and analysis modules, and a resumable Colab notebook. Reuse hashing, tokenizer,
anchor, activation, and checkpoint helpers. Do not modify v2/v3 results.

## Global Constraints

- The design spec is the source of truth for identities, splits, layers,
  strengths, controls, statistics, and claims.
- Existing v3 test results are never inputs to fitting or selection.
- Test files are written before implementation for every behavior change.
- Live model outputs are evidence only after identity and audit gates pass.
- Free Colab interruption must resume the same run, not create a replacement.
- SkillOpt proposals remain staged and cannot change scientific boundaries.

## Task 1: Corpus, Direction, and Selection Contracts

**Create:**

- `src/trajectory_extractor/fa_answerability_causal.py`
- `tests/test_fa_answerability_causal.py`
- `configs/familiarity_answerability_causal_pilot_v1.json`

Implement deterministic fresh 2x2 units, split isolation, tokenizer-aware pair
audits, train-only direction extraction from v3 artifacts, fixed validation
selection, typed hashes, and serialization. Tests must fail first for tampered
labels, reused v3 test identities, incomplete units, token mismatch, test-fitted
directions, and non-determinism.

## Task 2: Gemma Intervention Adapter

**Create:**

- `src/trajectory_extractor/fa_answerability_causal_runtime.py`
- `tests/test_fa_answerability_causal_runtime.py`

Implement a prefill-only residual hook for Gemma decoder layers, exact anchor
resolution, candidate sequence scoring, short deterministic generation, and an
audit proving that exactly one registered site changed while prompt/prefix bytes
remained identical. Use fake decoder modules for local tests and one tiny live
smoke in Colab before protected execution.

## Task 3: Controls and Sealed Analysis

**Create:**

- `src/trajectory_extractor/fa_answerability_causal_analysis.py`
- `tests/test_fa_answerability_causal_analysis.py`

Implement all registered controls, complete-unit summaries, fixed bootstrap and
sign-flip tests, preservation checks, typed `causally_supported`,
`not_supported`, and `not_evaluable` decisions, and one-use test evaluation.
Tests must cover each support clause, strongest-control comparison, random
family aggregation, malformed outputs, partial resumes, and endpoint reuse.

## Task 4: CLI and Free-Colab Workflow

**Modify:**

- `src/trajectory_extractor/fa_cli.py`
- `tests/test_fa_cli.py`

**Create:**

- `notebooks/fa_answerability_causal_pilot_colab.ipynb`
- `docs/fa_answerability_causal_pilot_runbook.md`

Add commands to prepare/freeze, select on validation, run one resumable shard,
audit completion, evaluate tests once, and render a report. The notebook pulls
a pinned Git commit, checks `HF_TOKEN`, mounts optional Drive checkpoints,
runs an unprotected smoke, and then executes/resumes the registered shards.

## Task 5: Process Optimization and Verification

Run focused tests after every task, then the complete suite and Ruff. Run the
available SkillOpt smoke first; if a real runner is available, stage a
development-only proposal against the research workflow skill and evaluate it
on the reserved workflow test split. Review proposals manually and adopt only
changes that improve the held-out workflow score without touching scientific
artifacts or protocol boundaries.

## Task 6: Live Execution and Publication

Run the free-Colab smoke, validation selection, seal, and both causal test
splits with atomic resume. Publish:

- frozen corpus, direction, selection, and provenance manifests;
- all primary and control outcomes;
- resource/runtime accounting;
- result JSON, Markdown report, and one concise README update;
- explicit null, failure, and claim-boundary statements.

If live compute cannot be completed in the current session, publish only an
execution-ready, software-verified package and state exactly which empirical
gates remain unrun. Do not describe readiness as a causal result.
