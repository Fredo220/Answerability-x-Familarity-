# Feature Dynamics for Concept Mixing and Jailbreaks

CPU-first research infrastructure for testing whether layerwise representation dynamics add predictive or causal value beyond simple LLM baselines. The two preregistered tracks are concept-binding errors and unsafe jailbreak responses.

The operator residual is Remizov/Chernoff-inspired, not a claimed application of an operator-semigroup theorem to transformers. It keeps that name only if it beats simpler baselines under the frozen criteria in [the preregistration](docs/preregistration.md).

## Frozen Concept Result

The completed synthetic concept run is negative: neither primary nor secondary
exact-error comparisons support the registered dynamics claim. The selected
registered prefix is the latest pre-token state, so it does not demonstrate early
warning or "artificial intuition." See the full frozen-result record and pending
controls in [docs/results.md](docs/results.md).

## Scope

- Target: `meta-llama/Llama-3.2-1B-Instruct`
- Judge: `meta-llama/Llama-Guard-3-1B`, loaded after unloading the target
- Hardware: CPU, batch size 1, 8 GB RAM
- Compute dtype: bfloat16 after a local CPU capability check; persisted activations remain float16
- Extraction: deterministic generation, then one teacher-forced causal replay for all pre-token states
- Artifacts: float16 answer-token x layer residual states under `runs/<run_id>/`
- No SAE claim: PCA is a tractable coordinate system, not a reversal of superposition

## Setup

Use Python 3.12 and configure Hugging Face access for both gated Meta models.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
hf auth login
```

Generate the controlled dataset without downloading a model:

```bash
feature-dynamics generate-concept-data
```

Run the gated local smoke test:

```bash
feature-dynamics smoke-extract --config configs/llama32_1b.json
```

`configs/pipeline_rehearsal_qwen05b.json` is reserved for an ungated engineering
rehearsal. Its artifacts live under `runs_rehearsal/` and are excluded from all
scientific reports and Llama claims.

Validate an official local JailbreakBench export:

```bash
feature-dynamics prepare-jailbreak-data \
  data/external/jailbreakbench/harmful-behaviors.csv \
  data/external/jailbreakbench/benign-behaviors.csv \
  data/external/jailbreakbench/dsn-llama2.json \
  --artifact-commit <JAILBREAKBENCH_ARTIFACTS_COMMIT>
feature-dynamics validate-jailbreak-data data/external/jailbreakbench/study.jsonl
```

The preparation step verifies all official indices and semantic pair keys,
uses `Goal` rather than the short `Behavior` label as the benign prompt, and
writes a checksum manifest next to the frozen study file.

## Fresh Or Replication Runs

Do not use the following commands for the frozen `concept-main` run. It must not
be rerun; its tracked frozen result is [docs/results.md](docs/results.md). Use a
new run ID for a fresh or replication study.

The concept track is executed in four explicit stages:

```bash
RUN_ID=concept-replication-01
TRANSFER_RUN_ID="${RUN_ID}-real-transfer"
INTERVENTION_RUN_ID="${RUN_ID}-intervention"
TRANSFER_DATA="data/external/${RUN_ID}-real-transfer.jsonl"

feature-dynamics extract-concept --run-id "$RUN_ID" --pilot-per-split 10
feature-dynamics extract-concept --run-id "$RUN_ID"
feature-dynamics evaluate-concept --run-id "$RUN_ID"
feature-dynamics evaluate-secondary-concept \
  --config configs/llama32_1b.json \
  --run-id "$RUN_ID" \
  --bootstrap 2000 \
  --endpoint exact_error
feature-dynamics ablate-concept --run-id "$RUN_ID"
feature-dynamics intervene-concept \
  --baseline-run-id "$RUN_ID" \
  --run-id "$INTERVENTION_RUN_ID"
feature-dynamics prepare-circuit-followup --run-id "$RUN_ID"
```

The secondary command evaluates a **metacognitive internal reliability signal**:
a causal risk score derived from a contrastive error direction and its evolution
across answer-token and layer prefixes. "Artificial intuition" is a user-facing
metaphor only. The score does not imply consciousness or ground-truth access, and
it is not a failure proof. The registered question is whether dynamics add held-out
predictive value beyond the same contrastive direction used statically.

The secondary analysis uses a strict holdout: validation receives full prefix
probability surfaces for selection, while test receives only the frozen
selected-prefix probability. Because each prefix cell is an independently fitted
classifier, thresholds do not transfer across cells and crossing diagnostics are
explicitly not interpretable. Its confirmatory p-value is a paired entity-family
permutation test with seed 42 and 2,000 permutations; the cluster bootstrap
supplies confidence intervals only. The validation figure is named
`validation_metacognitive_risk_gap_<endpoint>.png`. A positive result is
provisional until the frozen falsification controls and external transfer are
completed.

For the next run, response length is a required nuisance control. Pre-register
a response-length-only baseline, report length-matched or stratified test
metrics with uncertainty, and use one shared genuinely pre-output prefix for all
examples. Do not aggregate shorter completed responses through their individual
last available token when testing an early-warning claim. The frozen run's
post-hoc audit and its unchanged negative result are recorded in
[docs/results.md](docs/results.md).

The pilot and full command share a run ID. Completed artifacts are verified and
skipped, so the full command resumes rather than repeats the pilot. Extraction
records per-example runtime and prints a rolling ETA.

The source-documented transfer file must contain 200 JSONL rows with `id`,
`subject`, `relation`, `object`, `source_url`, `distractors`, and
`distractor_answers`:

```bash
feature-dynamics prepare-real-transfer --output "$TRANSFER_DATA"
feature-dynamics extract-transfer "$TRANSFER_DATA" \
  --run-id "$TRANSFER_RUN_ID"
feature-dynamics evaluate-transfer \
  --reference-run-id "$RUN_ID" \
  --run-id "$TRANSFER_RUN_ID"
```

Transfer evaluation fits every learned component and selects its prefix on the
synthetic concept train/validation folds. All 200 documented triples are then
treated as external test examples.

The jailbreak track deliberately alternates target and Guard processes:

```bash
feature-dynamics extract-jailbreak data/external/jailbreakbench/study.jsonl \
  --pilot-pairs-per-category 1
feature-dynamics extract-jailbreak data/external/jailbreakbench/study.jsonl
feature-dynamics judge-jailbreak --run-id jailbreak-main
feature-dynamics record-manual-audit data/external/jailbreakbench/audit_completed.csv
feature-dynamics evaluate-jailbreak --run-id jailbreak-main

feature-dynamics prepare-jailbreak-intervention-validation data/external/jailbreakbench/study.jsonl
feature-dynamics judge-jailbreak --run-id jailbreak-intervention-val
feature-dynamics select-jailbreak-intervention
feature-dynamics prepare-jailbreak-intervention-test data/external/jailbreakbench/study.jsonl
feature-dynamics judge-jailbreak --run-id jailbreak-intervention-test
feature-dynamics evaluate-jailbreak-intervention
feature-dynamics report-study
```

`configs/llama32_1b_jailbreak.json` allows 96 response tokens; the 12-token
concept cap is never reused for safety-response judging.

`report-study` writes its generated output to `docs/generated_study_report.md`.
`docs/results.md` is the immutable hand-maintained concept record and must never
be used as report-study output.

## Method families

1. Output log probability and entropy
2. Layerwise static logistic probes
3. Raw velocity, curvature, and directional change
4. Layerwise PCA-32 ridge operator residuals fitted on correct/safe training examples
5. Static plus dynamics combined

All fitted components record their training example IDs. Evaluation builds and
persists AUROC/AUPRC surfaces across causal token/layer prefixes, freezes the
selected-prefix threshold on validation, and reports selected-prefix test
calibration and false-positive rate. It does not report threshold-crossing timing
across prefix cells. Steering compares no intervention,
norm-matched random, shuffled-label, ITI-style always-on, and
operator-residual-triggered intervention.

Concept examples also carry a diagnostic error taxonomy. The preregistered
primary endpoint remains normalized exact-match error; `binding_error` is a
secondary mechanistic endpoint that only fires when the model outputs a known
distractor object.

## Circuit-tracing follow-up

Anthropic Fellows and Decode Research published attribution-graph tooling with
pretrained transcoders for the base `meta-llama/Llama-3.2-1B` checkpoint. It is
applicable as an exploratory causal follow-up to the concept-binding study, not
as a substitute for held-out detection metrics. The case-preparation command
freezes archetypal true positives, false positives, false negatives, and true
negatives using the validation-selected threshold and a deterministic ranking.

The published Llama transcoders target the base checkpoint, whereas this study's
primary target is the Instruct checkpoint. Base-model graphs are therefore an
external mechanistic replication. They are not presented as direct explanations
of Instruct behavior. See [the follow-up protocol](docs/circuit_tracing_followup.md).

## Notebooks

The notebooks consume persisted artifacts; reusable logic stays in `src/trajectory_extractor/`.

- `notebooks/01_sanity_check.ipynb`
- `notebooks/02_concept_mixing.ipynb`
- `notebooks/03_jailbreak.ipynb`
- `notebooks/04_intervention.ipynb`

## Run layout

```text
runs/<run_id>/
  manifest.json
  examples/*.json
  examples/*.npz
  responses/*.json
  labels/
  metrics/
  bootstrap/
  figures/
```

Detection runs use `examples/` and contain float16 activation trajectories.
Intervention sweeps use `responses/` because their endpoint only requires the
generated answer, intervention trigger metadata, and judge label; they do not
repeat activation replay for every hyperparameter candidate.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Unit and integration tests use fake models and local fixtures. The gated-model smoke test is intentionally separate because it requires accepted licenses and model downloads.
