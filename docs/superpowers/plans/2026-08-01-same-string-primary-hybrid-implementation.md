# Same-String Primary Hybrid Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a preregistered, low-compute Same-String experiment that isolates contextual familiarization from answerability, then add a gated mechanistic pilot without reopening or reinterpreting R11.

**Architecture:** Reuse the repository's existing four-cell Same-String generator, naturalness workflow, artifact store, protected endpoint leases, H2b estimator, and Gemma runner. Add a separate Same-String-only construction path that does not depend on real-entity familiarity screening or the original factorial power audit. Keep behavior primary; activation readouts and matched activation replacement remain separately gated follow-ups.

**Tech Stack:** Python 3.12, pytest, NumPy/scikit-learn, Hugging Face Transformers, TransformerLens, Gemma 2 2B IT, JSON/JSONL provenance artifacts, Google Colab for model execution.

## Global Constraints

- R11 remains immutable and `not_evaluable`; no R11 artifact, endpoint, or claim may be changed.
- The primary claim concerns contextual familiarization, not pretrained knowledge, truth, intuition, consciousness, or universal hallucination detection.
- Every experimental unit contains exactly four rows: `high_exposure`/`low_exposure` crossed with `target_bound`/`code_absent`.
- The synthetic target string, requested relation, registry task, code vocabulary, and output contract are byte-identical within each unit.
- Primary `behavior_test` contains 48 units and 192 rows.
- Primary support requires H2b point estimate at least `0.05`, predicted-direction crossed-bootstrap 95% interval excluding zero, complete cells, at least `0.95` format validity per cell, and capability preservation.
- Protected endpoints are generated and evaluated once; failures remain visible and are never rescued by a secondary analysis.
- Real-entity R12 is secondary and cannot rescue a null, negative, invalid, or `not_evaluable` Same-String result.
- Use the pinned `google/gemma-2-2b-it` model/tokenizer revision `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` and deterministic greedy decoding.
- No new heavyweight dependency is added to the local core.
- Implement with TDD, surgical changes, and one reviewable commit per task.

## File Map

- `docs/amendments/2026-08-01-fa-same-string-primary.md`: immutable claim and endpoint amendment.
- `configs/familiarity_answerability_same_string_gemma2_2b.json`: isolated run identity with the existing confirmatory split sizes and pins.
- `src/trajectory_extractor/fa_cli.py`: narrow Same-String preparation, construction, and evaluation command wiring.
- `src/trajectory_extractor/fa_data.py`: Same-String-only manifest validation and construction helper.
- `src/trajectory_extractor/fa_features.py`: separately named Same-String probe-row materializer; original F2A exclusion remains unchanged.
- `tests/test_fa_cli.py`, `tests/test_fa_data.py`, `tests/test_fa_features.py`, `tests/test_fa_scoring.py`: red/green coverage for all new behavior.
- `docs/fa_same_string_primary_runbook.md`: local/Colab execution and fail-closed gate order.
- `notebooks/fa_same_string_primary_colab.ipynb`: thin Colab launcher that pulls the pinned Git revision and calls CLI commands.

---

### Task 1: Freeze the Same-String study identity

**Files:**
- Create: `docs/amendments/2026-08-01-fa-same-string-primary.md`
- Create: `configs/familiarity_answerability_same_string_gemma2_2b.json`
- Modify: `tests/test_fa_config.py`

**Interfaces:**
- Consumes: `FAConfig.from_json(path)` and the existing confirmatory validation rules.
- Produces: study id `familiarity-answerability-same-string-gemma2-2b-v1`, run id `same-string-primary-v1`, and an amendment hash used by later sealed artifacts.

- [ ] **Step 1: Write the failing config test**

```python
def test_same_string_confirmatory_config_reuses_registered_model_and_counts():
    config = FAConfig.from_json(
        "configs/familiarity_answerability_same_string_gemma2_2b.json"
    )
    assert config.study_id == "familiarity-answerability-same-string-gemma2-2b-v1"
    assert config.run_id == "same-string-primary-v1"
    assert config.model_revision == "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
    assert config.split_counts["behavior_test"] == 48
    assert config.thresholds["h1_min_interaction"] == 0.05
```

- [ ] **Step 2: Run the test and verify the missing config fails**

Run: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest tests/test_fa_config.py -q`

- [ ] **Step 3: Add the config and amendment**

Copy only registered model, tokenizer, chat-template, generation, split, bootstrap, anchor, and threshold values from `configs/familiarity_answerability_gemma2_2b.json`; change only `study_id` and `run_id`. The amendment must state that R11 is immutable, H2b becomes the primary Same-String estimand, and all mechanistic/causal work is gated.

- [ ] **Step 4: Run focused tests**

Run: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest tests/test_fa_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/familiarity_answerability_same_string_gemma2_2b.json docs/amendments/2026-08-01-fa-same-string-primary.md tests/test_fa_config.py
git commit -m "docs: register same-string primary study"
```

---

### Task 2: Prepare direct Same-String units without familiarity screening

**Files:**
- Modify: `src/trajectory_extractor/fa_cli.py`
- Modify: `tests/test_fa_cli.py`

**Interfaces:**
- Consumes: checked-in candidate and synthetic manifests, `match_synthetic_entities(...)`, `EntityMatch`, `FAArtifactStore`.
- Produces: CLI `fa-prepare-same-string-matches` and a verified `same_string_match_collection` artifact containing deterministic, split-disjoint matches for every registered unit.

- [ ] **Step 1: Write failing CLI tests**

Add tests proving that the command:

```text
fa-prepare-same-string-matches
  --candidate-manifest <path>   # repeat once per split
  --synthetic-manifest <path>   # repeat once per split
  --shard-id same-string-matches-v1
```

selects exactly each configured split count, preserves all four registered domains in deterministic hash order, rejects duplicate QIDs/pair IDs/name families across splits, and never loads a screening-completion artifact.

- [ ] **Step 2: Run tests and verify the command is unknown**

Run: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest tests/test_fa_cli.py -q -k "same_string_matches"`

- [ ] **Step 3: Implement the minimal command**

Add the command to `FA_COMMANDS`, parser registration, and `dispatch_fa`. Reuse the manifest readers and `match_synthetic_entities`; do not create a second matching algorithm. Write one hash-bound `same_string_match_collection` record with config hash, source hashes, selected pair IDs, and split counts.

- [ ] **Step 4: Add fail-closed loader coverage**

Tests must reject altered source hashes, wrong run/config identity, missing split units, duplicate pairs, and cross-split synthetic-name-family reuse.

- [ ] **Step 5: Run focused tests**

Run: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest tests/test_fa_cli.py -q -k "same_string_matches or naturalness"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trajectory_extractor/fa_cli.py tests/test_fa_cli.py
git commit -m "feat: prepare direct same-string match collection"
```

---

### Task 3: Build and seal a Same-String-only manifest

**Files:**
- Modify: `src/trajectory_extractor/fa_data.py`
- Modify: `src/trajectory_extractor/fa_cli.py`
- Modify: `tests/test_fa_data.py`
- Modify: `tests/test_fa_cli.py`

**Interfaces:**
- Consumes: verified `same_string_match_collection`, verified naturalness ratings, `build_same_string_examples(...)`, tokenizer pinning, artifact capabilities.
- Produces: `audit_same_string_dataset(rows, *, tokenizer) -> DatasetAudit`, CLI `fa-build-same-string-confirmatory`, one four-cell-only manifest, namespace capabilities, typed Same-String seal, tokenizer/probe metadata, and a `same_string_confirmatory_index`.

- [ ] **Step 1: Write failing data tests**

Add `build_same_string_manifest(config, rows)` tests that require:

```python
assert {row.block for row in manifest.examples} == {"same_string"}
assert len([r for r in manifest.examples if r.split == "behavior_test"]) == 192
```

Reject non-Same-String rows, incomplete four-cell units, within-unit target-string changes, duplicate example IDs, and protected split overlap.

- [ ] **Step 2: Run tests and verify the helper is missing**

Run: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest tests/test_fa_data.py -q -k "same_string_manifest"`

- [ ] **Step 3: Implement the data helper**

Implement `audit_same_string_dataset(rows, *, tokenizer)` as a dedicated preflight. Reuse the existing Same-String-specific and shared checks for four-cell completeness, target-string identity, code vocabulary, template/entity isolation, rendered-token length, special tokens, and Same-String token budget. Add explicit checks for relation/code leakage and deterministic cell identity. Do not call factorial-only checks, run the factorial power audit, or weaken `audit_dataset(...)`/`build_manifest(...)` used by the original study.

- [ ] **Step 4: Write failing CLI construction tests**

The new command must require a passed pilot gate, verified naturalness ratings, and the direct match collection. It must emit all protected namespace capabilities and must not accept a screened-match artifact or a power-audit flag.

- [ ] **Step 5: Implement the construction path**

Factor shared pin/capability writing only where it removes literal duplication. Keep `_build_manifest(...)` behavior unchanged. Bind the index to the amendment hash, config hash, full manifest hash, naturalness audit hash, tokenizer pin, and each namespace capability hash.

- [ ] **Step 6: Run focused tests**

Run: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest tests/test_fa_data.py tests/test_fa_cli.py -q -k "same_string or naturalness"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/trajectory_extractor/fa_data.py src/trajectory_extractor/fa_cli.py tests/test_fa_data.py tests/test_fa_cli.py
git commit -m "feat: build sealed same-string confirmatory manifest"
```

---

### Task 4: Make H2b the explicit primary behavioral result

**Files:**
- Modify: `src/trajectory_extractor/fa_scoring.py`
- Modify: `src/trajectory_extractor/fa_cli.py`
- Modify: `tests/test_fa_scoring.py`
- Modify: `tests/test_fa_cli.py`

**Interfaces:**
- Consumes: scored Same-String rows, `cross_resample(...)`, `SameStringSealEvidence`, registered thresholds, and the protected generation transaction.
- Produces: `estimate_same_string_behavior(rows) -> SameStringBehaviorMetrics`, `same_string_crossed_bootstrap(rows, replicates, seed) -> SameStringBootstrapDistribution`, `evaluate_same_string_primary(...) -> SameStringPrimaryDecision`, and a `same_string_behavior_result` artifact with primary status `supported`, `not_supported`, `not_evaluable`, or `infrastructure_failure`.

- [ ] **Step 1: Write failing gate tests**

Cover supported, null, wrong-direction, incomplete-cell, low-format-validity, and capability-impairment cases. Assert that support requires all registered conditions and a verified typed seal. The capability-preservation statistic is the high-minus-low exact-target-code rate on `target_bound` rows. Its lower 95% bootstrap bound must exceed the registered noninferiority margin `-0.05`.

- [ ] **Step 2: Run focused scoring tests**

Run: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest tests/test_fa_scoring.py -q -k "same_string"`

- [ ] **Step 3: Implement Same-String-only estimation and bootstrap**

`estimate_same_string_behavior(...)` must reject non-Same-String rows and report the registered difference-in-differences, per-cell attempt/abstention/format/exact-answer rates, complete-unit count, and target-bound capability difference. `same_string_crossed_bootstrap(...)` must call `cross_resample(...)`, retain only complete four-cell draws, and return intervals for both the primary interaction and capability difference. It must not require factorial H1/H2 cells.

- [ ] **Step 4: Implement the complete primary gate**

`evaluate_same_string_primary(...)` returns `not_evaluable` for a missing/invalid typed seal, incomplete registered cells, or no valid bootstrap draws. It returns `not_supported` unless the interaction is at least `0.05`, its lower 95% bound is above zero, every cell has format validity at least `0.95`, and the capability lower bound is greater than `-0.05`. Serialize every estimate, interval, count, rate, and reason; never infer support from H2b alone.

- [ ] **Step 5: Bind protected evaluation to the new index**

Extend `fa-evaluate-behavior-test` only as needed to recognize a verified `same_string_confirmatory_index`. Preserve atomic endpoint unlock/generation/evaluation/close behavior and recovery semantics.

- [ ] **Step 6: Run focused tests**

Run: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest tests/test_fa_scoring.py tests/test_fa_cli.py -q -k "same_string or behavior_test"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/trajectory_extractor/fa_scoring.py src/trajectory_extractor/fa_cli.py tests/test_fa_scoring.py tests/test_fa_cli.py
git commit -m "feat: report same-string primary behavior result"
```

---

### Task 5: Add the reproducible local and Colab execution path

**Files:**
- Create: `docs/fa_same_string_primary_runbook.md`
- Create: `notebooks/fa_same_string_primary_colab.ipynb`
- Modify: `README.md`
- Test: `tests/test_fa_notebooks.py`

**Interfaces:**
- Consumes: Tasks 1-4 CLI commands and the existing `HF_TOKEN`/Git-based Colab workflow.
- Produces: one ordered procedure for match preparation, blind naturalness audit, manifest sealing, smoke, protected behavior evaluation, artifact download, and local report reproduction.

- [ ] **Step 1: Write failing notebook contract tests**

Assert the notebook contains the pinned repository commit variable, installs from the checked-out repository, reads `HF_TOKEN` from the environment, never embeds a token, calls the Same-String CLI commands, uses persistent Drive output, and documents resume behavior.

- [ ] **Step 2: Run the notebook tests**

Run: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest tests/test_fa_notebooks.py -q -k "same_string"`

- [ ] **Step 3: Create the thin notebook and runbook**

The notebook must not duplicate analysis logic. Each model step shells into the repository CLI. The runbook must show expected row counts, expected artifacts, gate order, and the rule that protected generation begins only after human audit and local manifest audit pass.

- [ ] **Step 4: Update README with claim boundaries and entry points**

Link the design, amendment, implementation plan, runbook, and R11 negative result. State that the Same-String result is not yet empirical until the protected run completes.

- [ ] **Step 5: Run documentation/notebook tests**

Run: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest tests/test_fa_notebooks.py tests/test_fa_cli.py -q -k "same_string or notebook"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/fa_same_string_primary_runbook.md notebooks/fa_same_string_primary_colab.ipynb tests/test_fa_notebooks.py
git commit -m "docs: add same-string primary execution path"
```

---

### Task 6: Execute the model-independent preflight and issue human packets

**Files:**
- Generated under ignored artifact root: match collection, rating packet issuance, manifest audits.
- Publish after completion: `docs/results/same_string_primary_preflight.json`
- Publish after completion: `docs/results/same_string_primary_preflight.md`

**Interfaces:**
- Consumes: checked-in source manifests, Task 2 preparation command, existing two-rater naturalness workflow.
- Produces: a complete preflight outcome and, if needed, blinded rating packets for exactly two independent human raters plus a third adjudicator only for disagreements.

- [ ] **Step 1: Run direct match preparation locally**

Use every registered split and verify exact counts: 64/32/48/24/24 units.

- [ ] **Step 2: Run deterministic audits before human work**

Verify four-cell completeness, target-string identity, code/relation leakage, token tolerance, split isolation, hashes, and prompt byte determinism.

- [ ] **Step 3: Issue blinded naturalness packets**

Use `fa-prepare-naturalness-ratings` with the direct match manifest. Do not expose model outputs or endpoint labels to raters.

- [ ] **Step 4: Compile ratings and adjudicate only disagreements**

If external ratings are not yet available, publish status `awaiting_independent_ratings`; do not open a protected endpoint.

- [ ] **Step 5: Publish the preflight report**

Record all counts, exclusions, hash identities, failed checks, and gate state. A failed audit is a reportable result, not a reason to alter a protected split.

- [ ] **Step 6: Commit publishable preflight artifacts**

```bash
git add docs/results/same_string_primary_preflight.json docs/results/same_string_primary_preflight.md
git commit -m "data: publish same-string primary preflight"
```

---

### Task 7: Run the protected behavior endpoint and publish the result

**Files:**
- Generated under artifact root: behavior capability, generations, scored rows, metrics, endpoint receipt.
- Publish: `docs/results/same_string_primary_behavior_result.json`
- Publish: `docs/results/same_string_primary_behavior_result.md`

**Interfaces:**
- Consumes: passed preflight, sealed behavior capability, Gemma runner, primary evaluator.
- Produces: one immutable supported/null/negative/invalid result and exact artifact hashes.

- [ ] **Step 1: Run an unprotected four-unit smoke in a fresh Colab runtime**

Verify pin, chat-template hash, no double BOS, deterministic generation, checkpoint resume, and valid output parsing.

- [ ] **Step 2: Audit the protected manifest locally one final time**

Record the behavior prompt-manifest hash before generation. Do not print protected prompt text.

- [ ] **Step 3: Execute `behavior_test` once**

Run the protected evaluation command with persistent checkpoint storage. Resume the same transaction after infrastructure interruption; never generate a replacement endpoint.

- [ ] **Step 4: Reproduce scoring locally from downloaded artifacts**

Verify generation hash, scored-row hash, crossed-bootstrap seed/replicate count, cell counts, point estimate, interval, format validity, and capability preservation.

- [ ] **Step 5: Publish the behavioral report**

Use only the registered claim labels. Include all cell rates and all failed gate reasons even if the result is null or negative.

- [ ] **Step 6: Commit the result artifacts**

```bash
git add docs/results/same_string_primary_behavior_result.json docs/results/same_string_primary_behavior_result.md
git commit -m "data: publish same-string primary behavior result"
```

---

### Task 8: Add the gated Same-String mechanistic pilot

**Files:**
- Modify: `src/trajectory_extractor/fa_features.py`
- Modify: `src/trajectory_extractor/fa_cli.py`
- Modify: `tests/test_fa_features.py`
- Modify: `tests/test_fa_cli.py`
- Publish when run: `docs/results/same_string_primary_mechanistic_pilot.json`

**Interfaces:**
- Consumes: Same-String activations at `target_intro_end` and `user_prompt_end`, mechanism-train/locked-validation/probe-test capabilities.
- Produces: separately named probe rows and nested surface/static/static+dynamics comparisons; original F2A still rejects Same-String rows.

- [ ] **Step 1: Write failing feature tests**

Add a new `materialize_same_string_probe_rows(...)` path. Assert that it keeps only Same-String rows, labels exposure and answerability separately, preserves unit/template groups, and does not change the original `materialize_probe_rows(...)` rejection.

- [ ] **Step 2: Run the tests and verify the new path is missing**

Run: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest tests/test_fa_features.py -q -k "same_string"`

- [ ] **Step 3: Implement static readouts first**

Train on `mechanism_train`, select layer/regularization on `locked_validation`, seal selection, and evaluate once on `probe_test`. Report exposure, answerability, and unsupported-attempt prediction against surface-only and output-margin baselines. Require reciprocal transfer: train each readout within one level of the other factor and evaluate it on the opposite level, then reverse the direction.

- [ ] **Step 4: Add dynamics only as a nested ablation**

Dynamics supports a claim only if it improves held-out log loss over static features. Include random-label, layer-order, random-direction, norm-matched random-direction, and final-layer-excluded controls.

- [ ] **Step 5: Run focused and broad tests**

Run: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest tests/test_fa_features.py tests/test_fa_cli.py tests/test_fa_scoring.py -q`
Then: `"/Users/friedrichreichelt/Documents/Machanistic Interpretability/.venv/bin/python" -m pytest -q`
If the broad suite enters the known slow legacy test, record the exact completed count and run all changed-module tests to completion; never report an interrupted run as passing.

- [ ] **Step 6: Publish only reached gates and commit**

If behavior is null, label this representation-only and omit prevention claims. If behavior and probe gates pass, activation replacement may be scheduled as a separate follow-up plan rather than added speculatively here.

```bash
git add src/trajectory_extractor/fa_features.py src/trajectory_extractor/fa_cli.py tests/test_fa_features.py tests/test_fa_cli.py docs/results/same_string_primary_mechanistic_pilot.json
git commit -m "feat: add gated same-string mechanistic pilot"
```

---

## Final Verification

- [ ] Search the implementation and reports for prohibited overclaims: `intuition`, `truth detector`, `pretrained familiarity`, `hallucination prevention`, and `confirmed thesis`.
- [ ] Confirm R11 result files and hashes are byte-identical to commit `b742a4d`.
- [ ] Confirm every protected endpoint has exactly one closed transaction or remains unopened.
- [ ] Re-run all changed-module tests, Ruff, and Pyright.
- [ ] Perform a whole-branch research review: estimand fidelity, leakage, provenance, gates, and claim-to-artifact traceability.
- [ ] Push the reviewed branch; do not merge into `main` without explicit user approval.
