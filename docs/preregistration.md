# Preregistration: Feature Dynamics Study

## Questions

1. Do dynamics-containing models predict concept-binding errors better than the best output or static baseline?
2. Do they independently predict unsafe jailbreak responses better than the best simple baseline?
3. Does dynamics-triggered steering reduce either failure while preserving matched controls?

The two tracks have separate labels, splits, operators, steering directions, and result tables. A result in one track cannot rescue failure in the other.

Jailbreak intervention uses a seed-frozen, label-independent 60/20/20 split of
behavior categories for train, validation, and test. A split lacking both Guard
response classes is reported as not evaluable; categories are never reassigned
after observing labels.

## Frozen outcomes

Detection is **supported** only when the best dynamics-containing method improves test AUROC by at least 0.03 over the strongest simple baseline and a paired 95% bootstrap interval excludes zero. A positive result missing either condition is **partially supported**. Otherwise it is **not supported**.

Machine-readable outcome labels are `supported`, `partially_supported`, and `not_supported`.

Intervention is **supported** only when triggered steering reduces the target failure by at least 20% relative, loses no more than 5 percentage points on matched controls, and beats norm-matched random steering with a paired 95% bootstrap interval excluding zero. The same partial/not-supported distinction applies.

Thresholds, projections, probes, ridge operators, intervention directions, layers, and strengths are selected on train/validation data only. Test data is evaluated once after choices are frozen.

Paired confidence intervals are cluster-bootstrapped by synthetic entity
family for concept experiments, by behavior category for grouped jailbreak
detection, and by matched pair for safety intervention comparisons. Detection
runs persist full trajectories. Intervention generation persists response-only
records because activation replay is not used to select or score intervention
outcomes.

## Methods

- Output baseline: response-token log probability and entropy.
- Static baseline: layerwise logistic probe.
- Raw dynamics: normalized velocity, curvature, and directional change.
- Operator residual: layer-specific PCA followed by ridge transition operators.
- Combined: static score plus raw dynamics and operator residual.

All activation-based early-warning results use the state immediately before the
predicted answer token and causal prefixes `(answer token < t, layer <= l)`.
No emitted token or future answer token enters its own activation prefix. Output
log probability remains an output baseline and is reported separately. The term
"Remizov-inspired" is retained only if operator residuals add measurable value;
no operator-semigroup theorem is claimed for transformers.

### Pre-run amendment: stable operator reference

Before any gated target-model run, the operator fit was restricted to label-0
training examples: correct responses for concept mixing and safe responses for
the jailbreak track. PCA coordinates and ridge operators therefore describe a
stable reference class; downstream classifiers still train on both classes.
This amendment changes neither held-out splits nor acceptance thresholds and is
recorded before observing any target-model result.

## Concept-mixing data

Generate 1,200 unique controlled in-context examples split 720/240/240 by
surface-entity-disjoint families and prompt-template groups. Every example has
a same-relation hard distractor; name similarity, synthetic-name rarity proxy,
relation, distractor count, and answer position are balanced. Label generated
answers by normalized exact match and cap generation at 12 tokens. Separately
record correct, distractor-binding, format-only, refusal, truncation, and other
errors. Evaluate an additional 200 source-documented real-transfer triples as
an external test only; no projection, probe, operator, threshold, or prefix is
fitted on those examples.

## Jailbreak data

Use exactly 100 official JailbreakBench harmful behaviors, 100 matched benign
controls, and frozen published artifacts for every harmful row. Do not generate
or publish new attack recipes. Use grouped cross-validation by behavior
category. Judge responses with Llama Guard 3 1B only after unloading the target
model, then record a completed stratified 20% human audit before final reporting.

## Ablations

Required: last token/full trajectory, PCA 16/32/64, raw/operator methods, shuffled layers, random projection, prompt length, entity rarity, distractor count, steering layer, and steering strength. Report accuracy, unsafe response rate, refusal rate, benign over-refusal, AUROC, AUPRC, calibration error, false-positive rate, threshold crossing, paired bootstrap intervals, and runtime overhead.
