# Familiarity vs. Answerability Preregistration

**Status:** Pre-outcome protocol. No confirmatory endpoint has been opened.

## Scope

This study tests whether target familiarity selectively increases answer attempts when the requested synthetic relation is absent from context. It uses `google/gemma-2-2b-it` at the immutable revision recorded in `data/fa/source_pins.json`. `Qwen/Qwen3-0.6B` is smoke-only and cannot select confirmatory layers, thresholds, templates, or claims.

The confirmatory entity-unit counts are `mechanism_train=64`, `locked_validation=32`, `behavior_test=48`, `probe_test=24`, and `intervention_test=24`. `pilot` and `circuit_dev` are physically separate non-confirmatory namespaces, not confirmatory entity counts.

## Hypotheses

- **H1:** The target-familiarity by answerability difference-in-differences in answer attempts is positive, has a lower 95% bound above zero, a point estimate of at least 0.05, at least 95% format validity in every cell, and passes H2.
- **H2:** For target-bound prompts, matched-synthetic target exact-answer accuracy is non-inferior to screened-real accuracy within a 5 percentage-point margin. Its lower paired-bootstrap bound must exceed -0.05.
- **H2b:** The sealed same-string high-versus-low exposure block has the H1 sign, an interaction of at least 0.05, and a predicted-direction 95% interval excluding zero. It is confirmatory only when its prefix construction is sealed before outcomes are opened.
- **H3:** A familiarity probe at `target_intro_end` generalizes across answerability conditions, held-out identities, and held-out templates.
- **H4:** An answerability probe at `user_prompt_end` generalizes across target familiarity, distractor familiarity, and held-out relation families.
- **H5:** Within evidence-absent prompts, frozen internal features improve held-out log loss over the nested surface-plus-output-margin baseline.
- **H6:** Cross-layer dynamics improve held-out log loss over the nested static-activation model. This is secondary and cannot invalidate H1-H5.
- **H7:** Same-string full activation replacement (`alpha=1`) changes answer attempts in the predicted direction for high-to-low and low-to-high pairs. Contrastive direction steering is secondary.
- **H8:** The selected intervention reduces target-bound exact-answer accuracy by no more than 0.05 and changes unrelated-control refusal and invalid-format rates by no more than 0.03.

## Registered Analysis

The primary behavioral estimand averages the two evidence-absent states (`distractor_bound`, `code_absent`) equally and averages over distractor familiarity. The crossed bootstrap independently resamples entity units and template families with replacement, weighting each row by the product of multiplicities. H2 uses the 2.5th percentile of the paired bootstrap difference as its lower 95% bound.

The power grid crosses absent attempt rates `{0.10, 0.25, 0.50}`, entity ICC `{0.05, 0.15, 0.30}`, template ICC `{0.02, 0.10}`, invalid-format rates `{0.00, 0.05}`, and interactions `{0.00, 0.025, 0.05, 0.075, 0.10}`. It uses 2,000 simulations per point with seed `20260722`.

Residual candidates include all 26 Gemma 2 layers. PCA candidates are `{none, 16, 32, 64}` and L2-logistic `C` candidates are `{0.01, 0.1, 1.0, 10.0}`. Familiarity uses `target_intro_end`; answerability uses `user_prompt_end`; the output-proximal `assistant_prefix_end` is prediction-only. The validation alpha grid `{0.25, 0.50, 1.00, 1.50}` applies only to secondary contrastive-direction steering.

## Endpoint Rules

Endpoint state is exactly `sealed -> unlocked_once -> evaluated -> closed`. Each of `behavior_test`, `probe_test`, and `intervention_test` is independent, opens once, and requires its registered parent manifests. A manifest for one endpoint cannot unlock another. Generic generation can access only `pilot`, `mechanism_train`, `locked_validation`, and `circuit_dev`; test generation and evaluation occur in one leased transaction.

No confirmatory generation starts before source/model/tokenizer/chat-template pins, this preregistration hash, split hashes, the power amendment, and required human-audit endpoints are sealed. Artifacts are immutable and hash-bound under `runs/familiarity_answerability/<run_id>/`; publishable manifests are under `release/familiarity_answerability/`.

## Claim Boundaries

Passing H1 supports a narrow behavioral interaction, not human-like belief or universal familiarity effects. H3-H5 support condition-invariant decodability and incremental pre-output prediction only when their gates pass. H7-H8 support local causal control only in this task. Optional F3 attribution is a fidelity-audited prompt-local circuit hypothesis and cannot rescue a failed F1 or F2A gate. Null results, invalid outputs, missingness, failed fidelity checks, and skipped gated phases remain visible in all reports.
