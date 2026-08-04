# Same-String Answerability Causal Replication v2

**Registered:** 2026-08-05, before any v2 model execution or outcome inspection.

## Purpose

The first causal pilot completed, but its label-shuffled control was
algebraically identical to the primary direction. That frozen run remains
`not_evaluable_as_confirmatory_causal_test`. This replication asks the same
narrow question with a corrected, pre-seal-audited control and entirely new
test identities.

## Frozen question

Does the answerability direction fitted only on the closed v3
`representation_train` split cause a bidirectional change in the registered
code-versus-`UNKNOWN` margin on two fresh Same-String test splits, while
beating every registered control and preserving task behavior?

The study does not test general metacognition, truthfulness, hallucination
prevention, or transfer to frontier models.

## What changes from v1

- New synthetic identities start at unit offset 100 and never reuse v1 names,
  codes, example IDs, or unit IDs.
- The template-transfer split uses two new families with a new panel layout and
  previously unused vocabulary: `briefing_panels` and `dispatch_panels`.
- Freshness is checked against the hash-verified, published v1 prompt artifact,
  including example IDs, unit IDs, names, codes, and rendered-prompt hashes.
- The label-shuffled direction swaps the bound/unbound sign in a balanced half
  of training units.
- Layer `18`, multiplier `1.0`, and `user_prompt_end` are locked from the
  invalidated v1 pilot. The new validation units check runtime, format, and
  preservation gates only; they cannot reselect the intervention site.
- Before a seal can be written, the shuffled vector must differ from the
  primary vector, have absolute cosine below `0.95`, and have L2 distance above
  `0.25`.
- The label-shuffle assignment seed changes to `20260805`; corpus identities
  are generated deterministically from the frozen v2 offset and prefix.
  Bootstrap and sign-flip seeds remain `20260804` to preserve direct
  inferential comparability with v1.

Everything else remains fixed: model and tokenizer revisions, train-only
direction source, 12 validation units, 18 units in each test split, output
contract, controls, bootstrap, sign-flip test,
preservation gates, and the requirement that both test splits pass separately.

This file is SHA-256-bound in the registered v2 config. Corpus preparation
fails before writing a seal if its bytes or the published v1 exclusion artifact
do not match their registered identities.

## Controls and support rule

The registered controls remain `no_intervention`, `sign_reversed`,
`label_shuffled_direction`, five norm-matched orthogonal random vectors,
`wrong_anchor`, and `wrong_layer`. The machine result is
`causally_supported` only if both unpooled test splits satisfy every frozen
criterion, including a positive 95% lower bound for the primary-minus-strongest
control contrast. A stronger effect at another layer therefore prevents a
layer-specific causal claim.

Any incomplete schedule, identity mismatch, failed vector-geometry audit, or
runtime-provenance mismatch is `not_evaluable`. A valid negative result is
`not_supported`. Neither status may be tuned or reclassified after outcomes.

## Precision boundary

This is a resource-bounded causal replication with 18 independent units per
test split. The smallest effect of substantive interest is a raw
primary-minus-strongest-control margin contrast of `0.10`. A negative result is
evidence against an effect of that size only when the corresponding confidence
interval excludes `+0.10`; otherwise it is reported as inconclusive rather than
as evidence of no causal role. The two splits are never pooled to manufacture
precision.

## Compute and reporting

Execution uses the free-Colab, resumable, one-receipt-per-unit workflow. All
receipts, failures, hashes, and null results will be published. The v1 release
is immutable and cannot be pooled with v2 to change the confirmatory decision.
