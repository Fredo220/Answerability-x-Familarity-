# Familiarity vs. Answerability Runbook

## Scope

This runbook executes the preregistered Familiarity vs. Answerability study. F1
tests the behavioral interaction. F2A tests whether registered internal signals
add pre-output prediction beyond surface and output-aligned baselines. F2B is a
gated local-causal extension. F3 circuit tracing is optional and currently
deferred until a compatible, immutable PLT asset passes its fidelity gate.

The repository currently provides study infrastructure. It does not contain a
confirmatory empirical result. A failed or null result must remain visible.

## Compute Split

- The 8 GB local computer runs unit tests, manifest audits, sealed analysis,
  report generation, and release verification.
- Google Colab runs Gemma 2 2B generation and activation extraction. Use a GPU
  runtime with at least 14 GiB VRAM and 40 GiB free disk.
- The optional circuit profile is isolated from the confirmatory environment.
  It cannot rescue a failed F1 or F2A result.

## Immutable Inputs

1. Use the pinned model, tokenizer, chat-template hash, dataset generator, and
   configuration under `configs/` and `data/fa/source_pins.json`.
2. Install `requirements/fa-core.lock`, then install this package editable with
   `--no-deps`.
3. Record the Git commit, lock-file SHA-256, config hash, model revision,
   tokenizer revision, and every input manifest hash.
4. Never substitute a filename for a verified artifact manifest.

## Local Verification

Use Python 3.12 from the repository virtual environment:

```bash
../../.venv/bin/python -m pytest -q
```

Do not use Python 3.13 for this worktree. Run the smoke profile before opening
any confirmatory endpoint. Entity screening and the human audit of naturalness
are required inputs; the human audit is not replaced by an automatic score.

## Colab Execution

Open `notebooks/06_familiarity_answerability_colab.ipynb`, set `HF_TOKEN`, mount
Google Drive, and run the preflight. The notebook is orchestration-only: all
scientific logic lives in tested Python modules and CLI transactions.

The normal development sequence is:

```bash
feature-dynamics fa-audit-manifest --config CONFIG --root ROOT --manifest MANIFEST
feature-dynamics fa-materialize-probe-rows --config CONFIG --root ROOT --namespace mechanism_train --manifest MECHANISM_PROMPT_MANIFEST --metadata-manifest PROBE_METADATA --shard-id mechanism-0000 --resume
feature-dynamics fa-materialize-probe-rows --config CONFIG --root ROOT --namespace locked_validation --manifest VALIDATION_PROMPT_MANIFEST --metadata-manifest PROBE_METADATA --shard-id validation-0000 --resume
feature-dynamics fa-fit-probes --config CONFIG --root ROOT --train-rows-manifest TRAIN_PROBE_ROWS --validation-rows-manifest VALIDATION_PROBE_ROWS --probe-test-manifest PROBE_TEST_PROMPTS --shard-id selection-0000
feature-dynamics fa-seal-behavior-test --config CONFIG --root ROOT --behavior-test-manifest BEHAVIOR_TEST_PROMPTS
feature-dynamics fa-seal-selection --config CONFIG --root ROOT --selection-manifest F2A_SELECTION --probe-test-manifest PROBE_TEST_PROMPTS
feature-dynamics fa-evaluate-behavior-test --config CONFIG --root ROOT --manifest BEHAVIOR_TEST_PROMPTS --shard-id behavior-0000
feature-dynamics fa-evaluate-probe-test --config CONFIG --root ROOT --selection-manifest F2A_SELECTION --probe-test-manifest PROBE_TEST_PROMPTS --metadata-manifest PROBE_METADATA --shard-id probe-test-0000
feature-dynamics fa-build-report --config CONFIG --root ROOT --behavior-test-manifest BEHAVIOR_TEST_PROMPTS --probe-test-manifest PROBE_TEST_PROMPTS --selection-manifest F2A_SELECTION --output reports/familiarity_answerability.md
```

`fa-materialize-probe-rows` is the production assembly path for training and
locked validation. Protected `probe_test` rows are generated internally by
`fa-evaluate-probe-test` only after the frozen selection unlocks the one-use
endpoint. The materializer generates outputs, extracts all 26 registered layers at the three registered anchors,
computes exact teacher-forced target/`UNKNOWN` scores, and binds registered
metadata plus explicit unsupported-answer outcomes. Activations remain once in
their verified NPZ shard; the compact JSONL evidence references that shard and
is deterministically reconstructed into canonical `ProbeRow` values when
fitting or evaluation begins. The protected prompt capability is sealed without
being opened by selection code; only its task-specific source identities are
exposed during selection.

Only checksum-valid completed shards may resume. A corrupt or schema-invalid
completed artifact fails closed; an interrupted generation attempt may retry
without reusing malformed output.

## Protected Endpoints

`behavior_test`, `probe_test`, and `intervention_test` are one-use protected
endpoints. Generic generation must reject all three namespaces. The behavioral
test performs generation and canonical scoring after unlock. The F2A test
generates and materializes its protected rows after unlock, evaluates the complete
task bundle, and closes the endpoint atomically. A closed probe evaluation is
recoverable but cannot be refit or rerun.

- `behavior_test`: opened only after the preregistered behavioral selection is
  sealed and the external human audit dependency is satisfied.
- `probe_test`: opened only after all F2A feature, layer, null, and threshold
  choices are sealed on training and locked validation data.
- `intervention_test`: opened only after F1 and F2A pass their registered gates
  and F2B direction, layer, alpha, and controls are sealed. The production model
  adapter for this gated phase is not yet implemented; F2B must therefore be
  reported as `skipped`, not simulated.

Never inspect protected generations before the corresponding canonical metrics
artifact has closed the endpoint.

## Analysis and Release

Run `notebooks/07_familiarity_answerability_analysis.ipynb` locally. It verifies
`MANIFEST.json`, recomputes the claim ladder from canonical metrics, and never
fits or generates. Publish hashes, exclusions, invalid outputs, null tests,
negative results, and missing phases. Resume state and every endpoint state are
part of the public provenance. The release must include the frozen F2A selection
shard and sidecar. Publish the top-level SHA-256 outside the bundle, for example
in a signed immutable Git tag and a DOI-backed paper artifact record, and pass it as
`expected_top_level_sha256` to `verify_release_bundle`; verification without an
external hash establishes internal consistency, not authorship or authenticity.

## Stop Conditions

Stop and report `not_evaluable` on pin mismatch, manifest corruption, leakage,
missing human audit, unsupported schema, incomplete protected transaction, or a
failed registered prerequisite gate. Do not loosen a threshold after seeing a
test result.
