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
- **H4:** An answerability probe at `user_prompt_end` generalizes across target familiarity, distractor familiarity, held-out entity units, and held-out template families. The fixed design contains one target relation (`archive_code`), so relation-family transfer is not claimed.
- **H5:** Within evidence-absent prompts, frozen internal features improve held-out log loss over the nested surface-plus-output-margin baseline.
- **H6:** Cross-layer dynamics improve held-out log loss over the nested static-activation model. This is secondary and cannot invalidate H1-H5.
- **H7:** Same-string full activation replacement (`alpha=1`) changes answer attempts in the predicted direction for high-to-low and low-to-high pairs. Contrastive direction steering is secondary.
- **H8:** The selected intervention reduces target-bound exact-answer accuracy by no more than 0.05 and changes unrelated-control refusal and invalid-format rates by no more than 0.03.

## Registered Analysis

The primary behavioral estimand averages the two evidence-absent states (`distractor_bound`, `code_absent`) equally and averages over distractor familiarity. The crossed bootstrap independently resamples entity units and template families with replacement, weighting each row by the product of multiplicities. H2 uses the 2.5th percentile of the paired bootstrap difference as its lower 95% bound.

The power grid crosses absent attempt rates `{0.10, 0.25, 0.50}`, entity ICC `{0.05, 0.15, 0.30}`, template ICC `{0.02, 0.10}`, invalid-format rates `{0.00, 0.05}`, and interactions `{0.00, 0.025, 0.05, 0.075, 0.10}`. It uses 2,000 simulations per point with seed `20260722`.

Residual candidates include all 26 Gemma 2 layers. PCA candidates are `{none, 16, 32, 64}` and L2-logistic `C` candidates are `{0.01, 0.1, 1.0, 10.0}`. Familiarity uses `target_intro_end`; answerability uses `user_prompt_end`; the output-proximal `assistant_prefix_end` is prediction-only. The validation alpha grid `{0.25, 0.50, 1.00, 1.50}` applies only to secondary contrastive-direction steering.

## Frozen Implementation Decisions

- The same-string block reuses the 192 split-isolated entity units. Each unit receives one hash-assigned template family and four rows: `high_exposure`/`low_exposure` crossed with `target_bound`/`code_absent`. Template families are balanced within split and domain. The provisional count may increase only through the pre-outcome power amendment.
- Core rows expand each entity unit over every template family registered for its split: three train, three validation, and four test families. Counterbalancing uses a deterministic hash-indexed Latin-square schedule.
- Output normalization is Unicode NFC plus surrounding-whitespace removal. Exact normalized `UNKNOWN` is abstention. A full-string target code, distractor code, or other registered code-shaped token is classified before generic text. A printable nonempty single-line string is `other_non_abstention`; empty, multiline, truncated, nonprintable, or infrastructure-marked output is `invalid_format`.
- Generic generation commands may access only `pilot`, `mechanism_train`, `locked_validation`, and `circuit_dev`. Test commands acquire an endpoint lease and perform generation plus evaluation in one transaction before marking the endpoint evaluated and closed. Every protected split has a separate encrypted-or-capability-scoped manifest; the global index contains IDs and hashes only, never protected prompt text or labels.
- The crossed bootstrap independently resamples entity units and template families with replacement and uses the product of their multiplicities as row weights. H2 uses the 2.5th percentile of the paired bootstrap difference as its lower 95% bound. The registered power grid crosses absent attempt rates `{0.10, 0.25, 0.50}`, entity ICC `{0.05, 0.15, 0.30}`, template ICC `{0.02, 0.10}`, invalid-format rates `{0.00, 0.05}`, and interactions `{0.00, 0.025, 0.05, 0.075, 0.10}` with 2,000 simulations per point and seed `20260722`.
- Same-string activation interchange is full replacement (`alpha = 1`) and has no tuned alpha. The validation alpha grid `{0.25, 0.50, 1.00, 1.50}` applies only to secondary contrastive-direction steering.
- Residual candidates cover all 26 Gemma 2 transformer layers. PCA candidates are `{none, 16, 32, 64}` and L2 logistic `C` candidates are `{0.01, 0.1, 1.0, 10.0}`. Target familiarity uses `target_intro_end`; answerability uses `user_prompt_end`; unsupported-answer prediction also evaluates the output-proximal control but cannot use it for a pre-output claim.
- Official pins are Gemma model/tokenizer `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`, Gemma chat-template SHA-256 `ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6`, Gemma Scope residual SAE repository `google/gemma-scope-2b-pt-res` at `fd571b47c1c64851e9b1989792367b9babb4af63`, and optional circuit-tracer `4bb8c0ea10bde09727e14565ec8469656880da53` (`v0.5.0`). Individual SAE paths are selected from the registered 16K family on training/validation only and still require the instruction-tuned loss-recovery gate.

## Endpoint Rules

Endpoint state is exactly `sealed -> unlocked_once -> evaluated -> closed`. Each of `behavior_test`, `probe_test`, and `intervention_test` is independent, opens once, and requires its registered parent manifests. A manifest for one endpoint cannot unlock another. Generic generation can access only `pilot`, `mechanism_train`, `locked_validation`, and `circuit_dev`; test generation and evaluation occur in one leased transaction.

No confirmatory generation starts before source/model/tokenizer/chat-template pins, this preregistration hash, split hashes, the power amendment, and required human-audit endpoints are sealed. Artifacts are immutable and hash-bound under `runs/familiarity_answerability/<run_id>/`; publishable manifests are under `release/familiarity_answerability/`.

## Claim Boundaries

Passing H1 supports a narrow behavioral interaction, not human-like belief or universal familiarity effects. H3-H5 support condition-invariant decodability and incremental pre-output prediction only when their gates pass. H7-H8 support local causal control only in this task. Optional F3 attribution is a fidelity-audited prompt-local circuit hypothesis and cannot rescue a failed F1 or F2A gate. Null results, invalid outputs, missingness, failed fidelity checks, and skipped gated phases remain visible in all reports.

The confirmatory behavioral endpoint requires 100% completed, non-truncated generation in every registered factorial cell. Completed malformed non-`UNKNOWN` outputs remain answer attempts under the frozen intention-to-treat rule; missing, infrastructure-marked, and truncated generations do not count as attempts and make the endpoint `not_evaluable` until resumed.
