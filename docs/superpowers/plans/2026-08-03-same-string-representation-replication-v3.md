# Same-String Representation Replication v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute this plan task by task.

**Goal:** Replace the unstable four-unit representation pilot with a sealed 80-unit replication that tests whether fixed Gemma 2 2B activations add held-out answerability information beyond strong prompt-surface baselines.

**Architecture:** Add a v3-only corpus and analysis module while reusing the existing pinned tokenizer, Gemma runner, selected-position activation hooks, deterministic activation shards, and hash helpers. Keep all v2 code and artifacts unchanged. Fit every preprocessing step on the training split and evaluate `entity_test` and `template_test` separately.

**Tech Stack:** Python 3.12, NumPy, scikit-learn, PyTorch, Transformers, pytest.

## Global Constraints

- Study identity, sample sizes, templates, layers, anchors, seeds, statistics, and support rule come from the approved v3 design spec.
- Rendered chat prompts must contain one BOS token and use `add_special_tokens=False` in the existing anchor resolver.
- Test outcomes are opened once. No corpus, feature, threshold, layer, or exclusion may change afterward.
- SkillOpt is a process aid only; no SkillOpt optimization result may be reported without an installed runner and held-out workflow benchmark.
- A positive result is model-specific, correlational decodability evidence. A null or failed gate is published as such.

## Task 1: Deterministic Corpus and Leakage Audit

**Files:**
- Create: `src/trajectory_extractor/fa_same_string_replication_v3.py`
- Create: `tests/test_fa_same_string_replication_v3.py`

1. Write failing tests for exact split counts, complete 2x2 units, disjoint identities, exclusive template families, exact within-factor token multisets, deterministic hashes, single-BOS rendering, and tamper rejection.
2. Implement the smallest immutable prompt record, six fixed template families, deterministic 80-unit allocation, exact binding swaps, pinned-tokenizer audit, and JSONL/manifest serialization.
3. Verify focused tests and a real cached-tokenizer corpus audit.

## Task 2: Surface and Activation Evaluation

**Files:**
- Create: `src/trajectory_extractor/fa_same_string_replication_v3_analysis.py`
- Create: `tests/test_fa_same_string_replication_v3_analysis.py`

1. Write failing tests proving train-only TF-IDF/scaler/PCA fitting, group-preserving evaluation, separate test splits, fixed layers, deterministic bootstraps/permutations, and negative-control reporting.
2. Implement the registered character and token TF-IDF surface model, activation-only model, combined model, prediction export, calibration metrics, unit bootstrap, stratified label permutations, fixed mean-layer omnibus, and support decision.
3. Implement a pre-outcome simulation-based sensitivity audit with fixed seed and effect grid.
4. Verify focused tests.

## Task 3: Minimal CLI and Reproducible Runner

**Files:**
- Modify: `src/trajectory_extractor/fa_cli.py`
- Modify: `src/trajectory_extractor/fa_runtime.py` only if required by an observed failing test
- Modify: `tests/test_fa_cli.py`
- Create: `configs/familiarity_answerability_same_string_replication_v3.json`

1. Add commands to prepare/audit/freeze the v3 corpus, extract one split with resumable activation shards, and analyze all four frozen splits.
2. Bind outputs to config, corpus, tokenizer, model, implementation, and activation hashes.
3. Run a tiny fake-runner smoke before loading real weights.

## Task 4: Pre-Outcome Freeze and Real Extraction

**Files:**
- Create generated release artifacts under `release/familiarity_answerability/representation_replication_v3/`

1. Run the full cached Gemma tokenizer audit.
2. Run and freeze the minimum-detectable-effect report before activation extraction.
3. Extract `representation_train`, `representation_validation`, `entity_test`, and `template_test` in separate resumable shards at layers `0,6,12,18,25`.
4. Verify every activation manifest and hash before analysis.

## Task 5: One-Shot Analysis and Publication

**Files:**
- Create: `docs/results/same_string_representation_replication_v3.md`
- Create: `docs/results/same_string_representation_replication_v3.json`
- Modify: `README.md`

1. Run the fixed v3 analysis once on both sealed test splits.
2. Export every prediction, layer metric, bootstrap interval, permutation result, negative control, and the typed support decision.
3. Write a concise result report and README update that state sample size, surface comparison, generalization results, limitations, and whether the registered support rule passed.
4. Run focused tests, full tests, Ruff, and final provenance checks.
5. Commit and push the completed evidence package to GitHub `main`.
