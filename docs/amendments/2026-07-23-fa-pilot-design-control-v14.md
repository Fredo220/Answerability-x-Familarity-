# FA Pilot Design-Control Amendment V14

Date: 2026-07-23

## Trigger

This amendment was written after opening the immutable V13 pilot metrics.
The candidate named `surface_design_oracle` achieved macro AUROC `0.8333`
rather than perfect answerability classification.

## Diagnosis

The feature set is sufficient to reconstruct the registered answerability
condition: `code_position = absent` identifies `code_absent`, while the
agreement between `entity_order` and a present `code_position` distinguishes
`target_bound` from `distractor_bound`.

However, the frozen V13 candidate applies a linear multinomial logistic
regression to the raw one-hot indicators. The agreement relation is an
interaction and is not represented by a raw linear term. The historical
artifact key therefore overstates what that fitted candidate is.

## Reporting Rule

- The immutable V13 artifact is not changed or rerun.
- Reports refer to its `surface_design_oracle` key as the
  **linear design-feature control**.
- The deterministic construction rule is described separately as a design
  audit, not as a learned baseline.
- No internal-feature result is compared against a claimed perfect learned
  oracle.
- This naming correction cannot change model selection, thresholds, p-values,
  or the confirmatory design.

## Interpretation

The answerability result remains dominated by visible prompt structure. The
frozen `morphology_plus_residual` candidates reduce held-out log loss relative
to the morphology-only baseline at all four registered layers. This is
descriptively consistent with incremental internal answerability information.
However, V13 did not preregister a paired incremental null distribution or
confidence interval for the combined-minus-surface contrast. The incremental
claim is therefore **not evaluable**, rather than supported or refuted, in this
pilot.
