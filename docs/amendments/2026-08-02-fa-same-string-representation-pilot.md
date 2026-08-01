# Same-String v2 Representation-Only Pilot

**Frozen:** 2026-08-02, before opening any v2 activation artifact  
**Scope:** exploratory, model-specific, non-causal

## Question

Does Gemma 2 2B encode contextual exposure and answerability differently in
its residual stream, even though the completed behavioral pilot did not show
the hypothesized exposure-driven failure behavior?

## Data and separation

The pilot reuses the immutable v2 prompt capabilities without changing any
prompt, split, label, threshold, or behavioral result:

- `mechanism_train`: 12 complete 2x2 units;
- `locked_validation`: 4 complete 2x2 units;
- `probe_test`: 4 complete 2x2 units.

`mechanism_train` and `locked_validation` form the fixed training set. All
reported performance is computed once on `probe_test`. Every unit contributes
exactly one prompt to each exposure x answerability cell.

## Fixed analysis

- Tasks: binary `exposure` (`low`, `high`) and binary `answerability`
  (`code_absent`, `target_bound`).
- Layers: `0, 6, 12, 18, 25`; no best-layer selection.
- Primary anchors: `target_intro_end` for exposure and `user_prompt_end` for
  answerability.
- Cross-anchor comparison: the other task anchor.
- Models: morphology-only logistic regression, residual-only logistic
  regression, and morphology-plus-residual logistic regression.
- Residual preprocessing: training-only standardization followed by a fixed
  16-component PCA, then `C=1` logistic regression.
- Uncertainty: entity-unit bootstrap interval on the four held-out units.
- Null: 99 fixed within-unit, other-factor-stratified label permutations. Raw,
  max-layer-adjusted, and mean-layer omnibus permutation values are reported.

The morphology baseline excludes exposure, answerability, code-position, and
completion features. It contains only target/distractor/prompt length and case
statistics plus rendered token count.

## Claim boundary

This analysis can show only exploratory held-out decodability in one small
model and one synthetic task. It cannot establish causal use, model reasoning,
general familiarity, hallucination detection, or frontier-model behavior. It
cannot alter, rescue, or reinterpret the closed `not_supported` behavioral
endpoint. No activation patching or attribution-graph claim is part of this
pilot.
