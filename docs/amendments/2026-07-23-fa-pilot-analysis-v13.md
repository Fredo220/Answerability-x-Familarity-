# FA Exploratory Pilot Analysis Amendment V13

Date: 2026-07-23

## Timing And Scope

This analysis specification was frozen while the V12 activation extraction was
still incomplete and before any activation value or activation-derived result
was opened. It governs only the development-only Qwen3-1.7B pilot. It cannot
select or modify confirmatory Gemma layers, anchors, estimators, thresholds, or
claims.

## Inputs

- Prompt capability:
  `runs/familiarity_answerability/smoke-qwen17b-v1/shards/pilot/prompt-capability-30475c149ddb4403.jsonl.manifest.json`
- Prompt subset SHA-256:
  `30475c149ddb440310554164c5a361d4e8de0eca971eec80a54d7e0ce46db646`
- Activation shard:
  `runs/familiarity_answerability/smoke-qwen17b-v1/activations/pilot/pilot-activations-v12-l0-9-18-27.npz.manifest.json`
- Registered layers: `0, 9, 18, 27`
- Factorial rows only: 288 rows from eight entity units
- Same-string rows are excluded from this analysis.

## Targets And Anchors

- Familiarity is binary: `screened_real = 1`, `matched_synthetic = 0`.
- Answerability is three-state:
  `target_bound`, `distractor_bound`, and `code_absent`.
- Familiarity primary anchor: `target_intro_end`.
- Answerability primary anchor: `user_prompt_end`.
- `assistant_prefix_end` is evaluated separately as an output-proximal control.
- Every registered layer is reported. No best-layer-only result is permitted.

## Frozen Cross-Validation

- Use leave-one-entity-unit-out cross-validation, producing eight held-out
  folds.
- Every preprocessing transform and classifier is fit on the seven training
  entity units in that fold only.
- Out-of-fold predictions are concatenated once for each model candidate.
- No row from a held-out entity unit may affect scaling, PCA, fitting, or
  calibration for that fold.

## Frozen Models

1. `majority`: class-prior probabilities learned from each training fold.
2. `surface_morphology`: target, distractor, and prompt character/word/capital
   counts, rendered token count, and template-family one-hot controls;
   standardize on the training fold, then fit logistic regression.
3. `surface_design_oracle`: the complete existing `surface_feature_vector`,
   including entity order and code position, plus template-family one-hot
   controls. Because entity order and code position deterministically reveal
   the registered answerability state, this is reported as a design oracle, not
   a weak predictive baseline.
4. `residual_static`: standardize the selected residual-stream vector on the
   training fold, reduce it with 16-component whitened randomized PCA, then fit
   logistic regression.
5. `morphology_plus_residual`: concatenate the standardized morphology/template
   controls and the 16 whitened PCA components, then fit logistic regression.

All logistic regressions use `C = 1.0`, the `lbfgs` solver, no class weighting,
`max_iter = 2000`, and random seed `20260723`. PCA uses
`svd_solver = randomized`, `whiten = true`, and the same seed. There is no
hyperparameter selection.

## Metrics And Controls

- Familiarity: AUROC, balanced accuracy, and log loss.
- Answerability: macro one-vs-rest AUROC, balanced accuracy, and multiclass log
  loss.
- Familiarity predictions are also reported separately within each
  answerability state.
- Answerability predictions are also reported separately within each target
  familiarity state.
- Worst-condition balanced accuracy is descriptive.
- The output-proximal anchor is a control and cannot establish pre-expressive
  information.

## Permutation Null

- Run 100 deterministic label permutations using seeds
  `2026072300` through `2026072399`.
- Familiarity labels are permuted within each
  `entity_unit_id × answerability` stratum.
- Answerability labels are permuted within each
  `entity_unit_id × target_familiarity` stratum.
- The identical permuted labels are used for every layer and model family for a
  target.
- Report the empirical AUROC tail probability.
- For residual models, additionally report a max-over-four-layers empirical
  tail probability within each target, anchor, and model family.
- The prespecified omnibus is the mean AUROC over all four layers within each
  target, anchor, and residual model family; report its empirical tail
  probability.
- These exploratory permutation summaries are diagnostics, not confirmatory
  significance tests.

## Claim Boundary

Decodability is not a circuit, causal mechanism, or proof that a representation
drives behavior. A positive pilot supports only this model-specific statement:
the selected residual-stream snapshots contain held-out-entity information
about the registered familiarity or answerability label beyond the listed
controls. A negative pilot is reported as a negative result at this model
scale. Confirmatory F2A remains gated on the independent naturalness audit and
the locked Gemma design.
