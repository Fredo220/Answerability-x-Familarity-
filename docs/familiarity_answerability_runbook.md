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

## Confirmatory Corpus And Human Gate

Run these steps before any confirmatory task prompt is materialized. The source
pool contains 384 real entities: twice the final registered count in every
split and domain. Gemma screening reduces this to a 244-pair human-audit pool
with pre-registered reserves. The human audit then deterministically selects
the final 192 pairs.

Build and cache the source pool:

```bash
PYTHONPATH=src ../../.venv/bin/python tools/build_fa_confirmatory_source.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --output-dir data/fa/confirmatory_source_v5 \
  --split-seed 20260722 \
  --retrieval-date <actual-UTC-retrieval-date> \
  --exclude-candidates data/fa/pilot_inputs/candidates_v4.json
```

The fixed-rank QLever responses and every 50-entity Wikidata label/alias batch
are cached under `data/fa/confirmatory_source_v5/source_cache/`. An interrupted fetch may rerun the
same command without changing the registered selection.
Existing source and synthetic snapshots are no-clobber: an identical rerun
resumes, while any changed payload fails and requires a newly versioned output
directory plus an explicit pre-outcome amendment.

The source command requires accepted Gemma model terms and a valid `HF_TOKEN`.
It applies tokenizer matchability before split assignment, then generates
exactly three final pseudonym reserves per selected source. The synthetic
snapshot and every split file are hashed into `source_integrity_v1.json` before
screening can run. The following command is only an idempotence audit; it must
produce files identical to those already sealed by the source build:

```bash
PYTHONPATH=src ../../.venv/bin/python tools/build_fa_confirmatory_synthetics.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --candidate-manifest data/fa/confirmatory_source_v5/candidate_entities_mechanism_train_v1.json \
  --candidate-manifest data/fa/confirmatory_source_v5/candidate_entities_locked_validation_v1.json \
  --candidate-manifest data/fa/confirmatory_source_v5/candidate_entities_behavior_test_v1.json \
  --candidate-manifest data/fa/confirmatory_source_v5/candidate_entities_probe_test_v1.json \
  --candidate-manifest data/fa/confirmatory_source_v5/candidate_entities_intervention_test_v1.json \
  --output-dir data/fa/confirmatory_source_v5
```

For each split, run `fa-run-screening`, then `fa-screen-entities`, using the
split-specific candidate, question, and synthetic manifests. Screening uses
only the three registered factual questions per source entity. It is source
qualification, not a hypothesis endpoint; no F1/F2A prompt exists yet.

```bash
feature-dynamics fa-run-screening \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --root . \
  --namespace <split> \
  --candidates-manifest <candidate-manifest> \
  --questions-manifest <question-manifest> \
  --source-integrity-manifest data/fa/confirmatory_source_v5/source_integrity_v1.json \
  --shard-id confirmatory-<split>-screening-v1

feature-dynamics fa-screen-entities \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --root . \
  --candidates-manifest <candidate-manifest> \
  --questions-manifest <question-manifest> \
  --screening-manifest <screening-completion.manifest.json> \
  --synthetic-manifest <synthetic-manifest> \
  --source-integrity-manifest data/fa/confirmatory_source_v5/source_integrity_v1.json
```

Combine exactly one verified screened-match shard from each registered split.
The assembler fails unless the collection contains the frozen 244-pair balance:

```bash
feature-dynamics fa-assemble-screened-matches \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --root . \
  --screened-matches-manifest <mechanism-train-screened.manifest.json> \
  --screened-matches-manifest <locked-validation-screened.manifest.json> \
  --screened-matches-manifest <behavior-test-screened.manifest.json> \
  --screened-matches-manifest <probe-test-screened.manifest.json> \
  --screened-matches-manifest <intervention-test-screened.manifest.json> \
  --shard-id confirmatory-screened-collection-v1
```

Issue blinded packets to two distinct real human raters. Raters must follow
`docs/fa_naturalness_rating_protocol.md`; a language model, the researcher, or
duplicated rater identities cannot substitute for this evidence.

```bash
feature-dynamics fa-prepare-naturalness-ratings \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --root . \
  --screened-matches-manifest <screened-match-collection.manifest.json> \
  --output-dir human_ratings/confirmatory_v1 \
  --rater-id <opaque-rater-a-id> \
  --rater-id <opaque-rater-b-id> \
  --shard-id confirmatory-naturalness-issuance-v1
```

Compile both returned CSV files. Provide a third, distinct human adjudicator at
compile time. The command uses that person only if the first two verdicts
disagree:

```bash
feature-dynamics fa-compile-naturalness-ratings \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --root . \
  --screened-matches-manifest <screened-match-collection.manifest.json> \
  --issuance-manifest <packet-issuance.manifest.json> \
  --response <rater-a-response.csv> \
  --response <rater-b-response.csv> \
  --shard-id confirmatory-naturalness-initial-v1 \
  --adjudicator-id <opaque-rater-c-id> \
  --adjudication-output-dir human_ratings/confirmatory_v1/adjudication
```

If the status is `needs_adjudication`, send only the newly issued public packet
to rater C and finalize:

```bash
feature-dynamics fa-finalize-naturalness-adjudication \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --root . \
  --screened-matches-manifest <screened-match-collection.manifest.json> \
  --initial-submission-manifest <initial-submission.manifest.json> \
  --adjudication-issuance-manifest <adjudication-issuance.manifest.json> \
  --adjudication-response <rater-c-response.csv> \
  --shard-id confirmatory-naturalness-final-v1
```

Build the final task capabilities only after the human ratings artifact exists:

```bash
feature-dynamics fa-build-confirmatory \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --root . \
  --matches-manifest <screened-match-collection.manifest.json> \
  --pilot-gate-manifest <passed-pilot-gate.manifest.json> \
  --naturalness-ratings-manifest <naturalness-ratings.manifest.json> \
  --run-registered-power-audit
```

This build is fail-closed if accepted reserves cannot supply every registered
split-domain quota. It creates sealed capabilities; it does not evaluate a
protected endpoint.

## Colab Execution

Open `notebooks/06_familiarity_answerability_colab.ipynb`, set `HF_TOKEN`,
`FA_GIT_COMMIT`, `FA_GIT_BUNDLE`, and `FA_GIT_BUNDLE_SHA256`, mount Google
Drive, and run the preflight. There is no repository remote for this worktree.
Store `HF_TOKEN` in Colab Secrets; do not paste it into a saved code cell.
For the normal run, place the generated `fa-study-launch.json` and its named
Git bundle in `MyDrive/fa-study-checkpoints`; the launch manifest supplies the
three `FA_GIT_*` values. Environment variables remain an explicit override.
The notebook therefore verifies the bundle SHA-256, clones the bundle, checks
out the exact detached commit, and records the bundle, commit, lock, config,
source-integrity, model, tokenizer, template, and GPU identity before model
execution.
The notebook creates `/content/fa-venv` and installs the exact `fa-core.lock`
there. The mutable Colab host kernel imports no project, NumPy, Pandas,
scikit-learn, PyTorch, Transformers, or Accelerate modules. A dedicated
`fa-colab-preflight` transaction verifies every exact lock pin, the project
import, the registered Torch, Transformers, and Accelerate versions, and CUDA
availability inside the virtual environment. `pip check` must report no broken
requirements in that environment. This separation prevents a package downgrade
from leaving already imported binary extensions in an ABI-incompatible state.
All model and artifact work then runs as CLI subprocesses under the same virtual
environment interpreter.
Tracked changes and unexpected untracked files fail closed. Existing
`runs/familiarity_answerability/` artifacts are the sole untracked exception
so an interrupted transaction can resume on the same Colab VM.

The first Colab phase runs the five frozen Source-v5 factual-screening splits
in registered order. It verifies and checkpoints each completed split
separately, then assembles exactly 1,152 screening completions into the
registered 244-pair human-audit pool. A checkpoint is restored only when its
archive SHA-256 and every contained `FAArtifactStore` shard verify. The
successful completion shard is checkpointed before entity selection starts.
Checkpoints are content-addressed and a small `LATEST` pointer is replaced only
after the new archive and metadata both exist, so a failed update does not
overwrite the last valid checkpoint. Transient free-disk and RAM observations
are logged per runtime session rather than included in the immutable execution
identity. Stage-specific record-kind allowlists prevent later protected shards
from entering screening checkpoints. Restore verifies the exact member list,
data hashes, complete manifest hashes, run ID, split, and archive location in a
staging directory before atomically installing the split.

The notebook deliberately stops after assembly. Do not bypass that stop until two
independent human raters have returned the blinded naturalness packets and an
independent adjudicator is available for disagreements.

The notebook is orchestration-only: all scientific logic lives in tested
Python modules and CLI transactions. Screening has split-level checkpointing,
not prompt-level checkpointing. A failed backend call remains an immutable
infrastructure-failure shard and must retry with a new shard ID. If the
largest split repeatedly fails because the free Colab runtime is interrupted,
freeze and test a microshard amendment before inspecting any downstream
result; do not silently change the transaction protocol.

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
