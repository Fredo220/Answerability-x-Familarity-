# Dual-Track Feature Dynamics Research Design

Status: Approved in review on 2026-07-11

## Decision

The project will follow a concept-first, safety-second, replication-third
sequence:

1. Complete the confirmatory concept-mixing study on
   `meta-llama/Llama-3.2-1B-Instruct` locally.
2. Add Anthropic-inspired contrastive-vector and activation-capping methods as
   a prospective secondary analysis without changing the frozen primary
   analysis.
3. Test causal correction of concept-binding errors.
4. Evaluate the same method families independently on the frozen JailbreakBench
   track.
5. Replicate the selected protocol on a second model using rented Google Colab
   compute.

Concept-mixing hallucinations are the flagship question. Jailbreak detection
and mitigation remain an independently evaluated safety application. A result
in one track cannot rescue a null result in the other.

The fastest acceptable path is the shortest critical path that preserves the
full evidentiary standard. Dataset size, held-out evaluation, negative controls,
human safety audit, and capability-preservation checks will not be reduced to
accelerate delivery.

## Research Position

Anthropic has already demonstrated that linear activation directions can
monitor and causally influence model behavior:

- [Persona Vectors](https://www.anthropic.com/research/persona-vectors) extracts
  contrastive directions for traits including hallucination propensity and uses
  them for monitoring and steering.
- [The Assistant Axis](https://www.anthropic.com/research/assistant-axis) tracks
  persona drift and uses activation capping to reduce persona-based jailbreak
  failures.
- [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html)
  extracts distributed emotion directions and shows causal effects of calm,
  desperation, anger, and other concepts on alignment-relevant behavior.

Therefore this project will not claim to be the first activation-vector
monitor, the first dynamic internal monitor, or the first use of activation
steering for safety. The proposed contribution is narrower:

> Test whether causal layerwise and tokenwise evolution of internal features
> provides predictive and causal information beyond strong static activation
> directions, first for entity-relation-object binding failures and then for
> unsafe jailbreak responses.

The project is designed to falsify this claim. If dynamics do not add held-out
value beyond static probes and contrastive directions, the dynamic claim is
rejected. The term "Remizov-inspired" is retained only if the operator-residual
method satisfies the frozen improvement criterion. No transformer semigroup
theorem is claimed.

## Constraints

- Local target: `meta-llama/Llama-3.2-1B-Instruct`.
- Local hardware: CPU, 8 GB RAM, batch size 1.
- Local judge: `meta-llama/Llama-Guard-3-1B`, loaded only after unloading the
  target model.
- Existing concept data and model artifacts remain valid and are never rewritten
  to accommodate a new method.
- The existing primary preregistration and acceptance thresholds remain frozen.
- New Anthropic-inspired methods are recorded in a dated secondary registration
  before any secondary test metrics are computed.
- No new jailbreak recipes are generated or published.
- Notebooks visualize persisted results; research logic remains in tested Python
  modules.

## Study Topology

### Track A: Concept-mixing hallucinations

The controlled task asks the model to bind an entity and relation to the correct
object in the presence of same-relation hard distractors.

The confirmatory endpoint remains normalized exact-answer error:

\[
Y_{\mathrm{exact}} = \mathbb{1}[\mathrm{answer}\neq\mathrm{target}].
\]

The mechanistically specific secondary endpoint is distractor-binding error:

\[
Y_{\mathrm{binding}} =
\mathbb{1}[\mathrm{answer}\in\mathrm{known\ distractor\ objects}].
\]

Format-only errors, refusals, truncations, and other errors remain separate
diagnostic outcomes. The controlled dataset contains 1,200 examples with a
720/240/240 train/validation/test split by disjoint entity family and prompt
template. An additional 200 source-documented real facts form an external test
that cannot fit or select any component.

### Track B: Jailbreak safety failures

The safety track uses exactly 100 official JailbreakBench harmful behaviors,
one frozen published artifact for each harmful row, and 100 matched benign
controls. Target outputs are judged after target-model unloading using Llama
Guard 3 1B. A stratified 20 percent human audit is required before a final
safety claim.

The track reports unsafe response rate, refusal rate, benign over-refusal,
AUROC, AUPRC, calibration, false-positive rate, and runtime. Splits, directions,
operators, intervention configurations, and result tables are independent from
the concept track.

## Causal Activation Contract

For example \(i\), the extraction artifact contains:

\[
H_i \in \mathbb{R}^{T_i\times L\times d},
\]

where \(T_i\) contains response-token positions, \(L\) contains embedding plus
transformer-layer states, and \(d\) is the residual-stream width.

All early-warning methods use the state immediately before the answer token
being predicted. At answer position \(t\), a detector may use only emitted
answer tokens before \(t\) and layer states up to its selected layer. It may not
use the emitted token at \(t\), future tokens, final-answer correctness, or a
judge label unavailable at inference time.

Existing deterministic generation and teacher-forced causal replay remain the
single source of activation artifacts. New analysis methods consume those
artifacts and do not trigger model re-extraction.

## Method Families

### Frozen primary methods

The original confirmatory comparison remains unchanged:

1. Output log probability and entropy.
2. Layerwise static logistic probes.
3. Normalized velocity, curvature, and directional change.
4. Layer-specific PCA plus ridge transition-operator residuals fitted only on
   correct or safe training examples.
5. Static plus dynamics combined.

### Prospective secondary methods

#### Contrastive activation directions

For risk class \(1\) and control class \(0\), fit on training examples only:

\[
v_\ell = \operatorname{normalize}(\mu_{1,\ell}-\mu_{0,\ell}).
\]

The centered projection score is:

\[
s_{\ell,t} =
\langle h_{\ell,t}-c_\ell, v_\ell\rangle,
\]

where \(c_\ell\) is a training-only center. The corrective direction is the
opposite sign, control minus risk. Every fitted object records its training
example IDs and refuses to fit if either class is absent.

#### Vector dynamics

Projection scores are standardized per layer using training-only statistics
before cross-layer differences are computed. This prevents layer-specific scale
from masquerading as dynamics.

\[
\widetilde{s}_{\ell,t}
= \frac{s_{\ell,t}-\mu_{s,\ell}}{\sigma_{s,\ell}+\epsilon},
\]

\[
\Delta_\ell s_{\ell,t}
= \widetilde{s}_{\ell+1,t}-\widetilde{s}_{\ell,t},
\]

\[
\Delta_t s_{\ell,t}
= \widetilde{s}_{\ell,t}-\widetilde{s}_{\ell,t-1}.
\]

The secondary comparison tests the static projection alone against the static
projection plus its layerwise and tokenwise dynamics. Operator residuals may be
added only in the fully combined secondary model.

#### Assistant-axis surrogate

For the jailbreak track, a small open-model assistant-axis surrogate is learned
from an independently generated, frozen contrastive corpus of normal Assistant
responses and non-Assistant role responses. This corpus is separate from
JailbreakBench and cannot contain its test prompts. The surrogate is labeled as
an approximation of the published Assistant Axis, not as a reproduction of
Anthropic's proprietary or larger-model representation.

## Intervention Design

All intervention directions are fit on training examples. Candidate normal
ranges are estimated from control-class training activations. Layer, strength,
normal-range percentile, and trigger threshold are selected on validation only.
One configuration is frozen before test generation.

### Intervention arms

1. No steering.
2. Norm-matched random direction.
3. Shuffled-label direction.
4. ITI-style always-on corrective steering.
5. Projection capping.
6. Operator-residual-triggered corrective steering.

For risk direction \(v\), center \(c\), and validation-selected normal limit
\(q\), projection capping applies:

\[
a = \langle h-c,v\rangle,
\]

\[
h' = h - \lambda\max(0,a-q)v.
\]

Triggered steering applies a fixed corrective direction only when the selected
operator residual exceeds its validation threshold. Layer-control experiments
apply the same intervention at nearby non-selected layers. Direction controls
use the same norm and strength.

An intervention supports a causal claim only if it reduces the target failure
by at least 20 percent relative, loses no more than five percentage points on
matched controls, and beats norm-matched random steering with a paired 95
percent bootstrap interval excluding zero. It must also act more strongly at the
selected layer than at control layers and show an associated change in the
internal risk score.

## Evaluation and Falsification

All projections, probes, operators, hyperparameters, prefix choices, and
thresholds are train/validation only. The test split is evaluated once after
selection is frozen.

The frozen primary detection claim requires:

\[
\Delta\mathrm{AUROC}\geq0.03
\]

over the strongest simple baseline and a paired cluster-bootstrap 95 percent
interval excluding zero. AUPRC, calibration, and false-positive rates are
mandatory because the failure classes may be imbalanced.

The secondary analysis makes two explicit comparisons:

1. Contrastive direction versus contrastive direction plus vector dynamics.
2. Projection capping versus operator-residual-triggered steering.

Secondary comparisons are reported separately and corrected using
Benjamini-Hochberg false-discovery-rate control within each track. They do not
replace the frozen primary decision.

Required falsification controls are:

- shuffled layer order;
- equal-dimensional random projection;
- shuffled labels;
- norm-matched random steering;
- prompt length, entity rarity, name similarity, distractor count, and answer
  position covariates;
- a static vector model with comparable parameter count;
- intervention at neighboring control layers;
- benign and capability-preservation tasks.

A binary secondary endpoint is confirmatory only if the held-out evaluation
contains at least 20 positive examples spanning at least 10 independent entity
families or behavior categories. Otherwise it is reported as descriptive and
`not_evaluable` for a confirmatory claim. The dataset is not altered after
observing labels to manufacture additional positives.

## Failure Handling

The pipeline fails closed when scientific provenance is uncertain:

- a manifest hash or resolved model revision mismatch aborts resume;
- a missing JSON/NPZ artifact pair is treated as incomplete and regenerated,
  never interpreted as a negative example;
- a fit with one response class, duplicate example IDs, or any test ID aborts;
- non-finite activations, projections, or residuals abort the affected run and
  preserve the last complete artifact;
- an interrupted extraction resumes from verified complete examples;
- a Guard parse failure remains unlabeled and cannot silently become `safe`;
- a human-audit disagreement is retained and reported rather than overwritten;
- an endpoint that fails its class-count or cluster-count requirement is marked
  `not_evaluable` rather than rescued with a different split.

## Replication Contract

The local result is followed by:

1. External evaluation on the 200 real, source-documented facts without any
   refit.
2. A second-model replication on rented Colab compute.

The preferred second model is `Qwen/Qwen2.5-7B-Instruct` on a GPU with at least
24 GB VRAM because it provides model-family separation and direct comparability
with Persona Vectors. If the rented environment cannot provide 24 GB VRAM, the
predeclared fallback is `Qwen/Qwen2.5-3B-Instruct`. The fallback decision is
made from hardware availability before loading replication labels.

Model-specific projections and operators must be refit because representation
spaces differ. Dataset construction, splits, feature definitions, hyperparameter
grid, selection procedure, metrics, and success thresholds remain fixed. The
replication test split is opened once.

## Code Architecture

The primary analysis remains isolated. New secondary functionality is added in
focused modules:

- `contrastive_directions.py`: layerwise direction fit, centering, projection,
  serialization metadata, and fit-ID tracking.
- `vector_dynamics.py`: standardized projection dynamics over layers and tokens.
- `secondary_study.py`: secondary evaluation and artifact persistence without
  modifying primary method selection.
- `activation_control.py`: always-on steering, projection capping, and
  device-native residual-triggered control.

The existing NumPy/scikit-learn operator residual remains the offline reference.
For live use, PCA centering, projection, and ridge transitions are converted to
Torch tensors on the model device. No `.cpu()` or `.numpy()` transfer is allowed
in the live hook. A numerical parity test must match the offline reference within
the tolerance selected for the configured dtype.

Every fitted method exposes the conceptual interface:

```python
fit(batch, train_indices)
transform(batch)
metadata()
```

Every intervention follows:

```python
fit(training_activations)
select(validation_activations)
apply(model, frozen_configuration)
```

Secondary outputs are stored under:

```text
runs/<run-id>/secondary/
  contrastive_vectors/
  vector_dynamics/
  activation_capping/
  comparisons/
```

Artifacts record the resolved model revision, dataset and split hashes, fit IDs,
layer and token prefix, seed, hyperparameters, and schema version. Secondary
commands must not overwrite primary metrics or bootstrap files.

## Test Requirements

Implementation is test-driven. Required tests cover:

1. Mean-difference direction sign, normalization, and zero-vector rejection.
2. Training-only centers and fit-ID provenance.
3. Rejection of test IDs in fitted artifacts.
4. Layerwise and tokenwise differences without future-token leakage.
5. Projection capping only outside the selected normal range.
6. Norm equality for random and shuffled-label controls.
7. Torch and NumPy operator-residual parity.
8. Hook cleanup after normal execution and exceptions.
9. Secondary evaluation leaving primary metrics unchanged.
10. Artifact round-trip and resumability.
11. End-to-end fake-model detection and intervention.

A separate benchmark command reports peak RSS, tokens per second, and percentage
monitor overhead. Runtime targets are reported empirically and are not enforced
as flaky wall-clock CI assertions.

## Execution Order

1. Record the dated secondary registration and update the research basis.
2. Establish a tested source-control baseline before resuming scientific runs.
3. Resume the unchanged `concept-main` extraction to 1,200 examples.
4. Implement contrastive directions and vector dynamics test-first.
5. Run frozen primary and registered secondary concept detection.
6. Run concept ablations, real transfer, and causal intervention.
7. Execute the independent jailbreak pilot, full extraction, Guard judging,
   human audit, detection, and intervention.
8. Freeze the replication protocol and run the second model on Colab.
9. Generate the paper-style report, reproducibility instructions, result cards,
   and Fellowship demonstration from persisted artifacts only.

## Completion Standard

The project is Fellowship-grade when the public artifact demonstrates rigorous
research judgment even if the headline hypothesis is false. A strong positive
claim additionally requires:

- predictive value beyond static persona/contrastive baselines;
- causal, layer-specific intervention beyond random and shuffled controls;
- preserved matched-control performance;
- transfer to source-documented real facts;
- independent jailbreak results;
- second-model replication;
- reproducible code, artifacts, limitations, and null-result reporting.

The value comes from a reliable empirical result about dynamic feature
interactions, not from the Remizov label or a claim of deterministic failure
prediction.
