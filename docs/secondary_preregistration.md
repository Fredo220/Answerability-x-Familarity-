# Secondary Registration: Metacognitive Feature-Flow Monitor

Date frozen: 2026-07-11

## Status

This is a prospective secondary analysis registered after the primary protocol
and before any secondary test metric is computed. It does not modify or replace
the frozen primary comparison in `docs/preregistration.md`.

## Hypothesis

A causal internal reliability score built from the layerwise and tokenwise
evolution of a contrastive error direction predicts held-out concept-mixing
errors better than the same contrastive direction used as a static score.
"Artificial intuition" is an explanatory metaphor; the measured object is a
metacognitive internal reliability signal, not consciousness or ground-truth
access.

## Endpoint And Splits

- Primary secondary endpoint: `exact_error`.
- Mechanistic diagnostic endpoint: `binding_error`.
- Direction, center, and standardization statistics: training split only.
- Prefix and threshold selection: validation split only.
- Test evaluation: once after all choices are frozen.
- Cluster unit: `entity_family`.

The endpoint is confirmatory only when the test split has at least 20 positive
examples spanning at least 10 independent entity families. Otherwise the result
is `not_evaluable` and remains descriptive.

## Registered Methods

1. `contrastive_vector`: centered projection onto the normalized training-only
   risk-minus-control activation direction at each layer, standardized with the
   same training-only layer statistics used by method 2.
2. `contrastive_plus_dynamics`: the static projection plus its train-standardized
   layerwise and causal tokenwise first differences.
3. `full_metacognitive_monitor`: method 2 plus output uncertainty, raw hidden-state
   dynamics, and a PCA-ridge operator residual fitted on label-0 training examples.
   This third method is exploratory and cannot replace the registered comparison.

## Registered Comparison

The confirmatory contrast is:

`contrastive_plus_dynamics - contrastive_vector`

measured as paired test AUROC difference with an entity-family cluster bootstrap.
The claim is supported only if all conditions hold:

- AUROC difference is at least 0.03;
- the paired 95% bootstrap interval excludes zero on the positive side;
- Benjamini-Hochberg adjusted p-value is below 0.05;
- endpoint eligibility requirements are satisfied.

The within-track secondary family has two preregistered hypotheses: this
detection comparison and the later capping-versus-triggered-steering comparison.
Until the intervention p-value exists, reserve its slot with p=1.0. This yields
a conservative adjusted detection p-value; the final intervention plan recomputes
Benjamini-Hochberg across both observed p-values.

Report AUPRC, expected calibration error, false-positive rate, selected causal
token/layer prefix, threshold, and earliest positive threshold crossing even when
the claim is unsupported.

## Falsification And Leakage Controls

- No test example may fit a direction, center, scale, operator, prefix, or threshold.
- Variable response length may not reweight an example during direction fitting.
- Changing a future token activation may not change an earlier token score.
- The operator reference class is label 0 only.
- Existing primary artifacts and metrics are read-only.
- Negative and null effects are reported without changing the dataset or threshold.
