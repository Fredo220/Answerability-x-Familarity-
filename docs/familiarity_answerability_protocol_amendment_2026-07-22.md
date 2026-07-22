# Protocol Amendment: 2026-07-22

```yaml
date: 2026-07-22
pre_outcome: true
affected_endpoints:
  - mechanism_train
  - locked_validation
  - behavior_test
  - probe_test
  - intervention_test
  - pilot
  - circuit_dev
```

## Rationale

The implementation protocol is being frozen before construction or opening of any Familiarity-vs-Answerability outcome endpoint. This amendment resolves the entity-unit allocation and source revisions required for reproducible implementation.

## Amendment

The confirmatory split counts are fixed at `mechanism_train=64`, `locked_validation=32`, `behavior_test=48`, `probe_test=24`, and `intervention_test=24`. `pilot` and `circuit_dev` are non-confirmatory namespace permissions only; they are excluded from confirmatory entity counts and claims.

The same-string block reuses the 192 split-isolated entity units. Each unit receives one hash-assigned template family and four rows: `high_exposure`/`low_exposure` crossed with `target_bound`/`code_absent`; template families are balanced within split and domain. Core rows expand each entity unit over every template family registered for its split: three train, three validation, and four test families. Counterbalancing uses a deterministic hash-indexed Latin-square schedule. The provisional count may increase only through a pre-outcome power amendment.

Output normalization is Unicode NFC plus surrounding-whitespace removal. Exact normalized `UNKNOWN` is abstention. A full-string target code, distractor code, or other registered code-shaped token is classified before generic text. A printable nonempty single-line string is `other_non_abstention`; empty, multiline, truncated, nonprintable, or infrastructure-marked output is `invalid_format`.

Generic generation commands may access only `pilot`, `mechanism_train`, `locked_validation`, and `circuit_dev`. Test commands acquire an endpoint lease and perform generation plus evaluation in one transaction before marking the endpoint evaluated and closed. Every protected split has a separate encrypted-or-capability-scoped manifest; the global index contains IDs and hashes only, never protected prompt text or labels.

The artifact store is an experimental-integrity control, not a host-security boundary. It fails closed on interrupted writes, accidental concurrent writers, unsafe path components, symlinks, hash mismatches, parent-manifest mismatches, and repeated endpoint unlocks. Every consumer re-verifies content-addressed sidecars before use. A malicious process running concurrently with the same operating-system user can rewrite files or ignore cooperative locking and is outside the registered threat model; confirmatory runs therefore require a dedicated local account or isolated Colab runtime with no untrusted concurrent process.

The crossed bootstrap independently resamples entity units and template families with replacement and uses the product of their multiplicities as row weights. H2 uses the 2.5th percentile of the paired bootstrap difference as its lower 95% bound. The power grid crosses absent attempt rates `{0.10, 0.25, 0.50}`, entity ICC `{0.05, 0.15, 0.30}`, template ICC `{0.02, 0.10}`, invalid-format rates `{0.00, 0.05}`, and interactions `{0.00, 0.025, 0.05, 0.075, 0.10}` with 2,000 simulations per point and seed `20260722`.

Same-string activation interchange is full replacement (`alpha = 1`) and has no tuned alpha. The validation alpha grid `{0.25, 0.50, 1.00, 1.50}` applies only to secondary contrastive-direction steering. Residual candidates cover all 26 Gemma 2 transformer layers; PCA candidates are `{none, 16, 32, 64}`; and L2 logistic `C` candidates are `{0.01, 0.1, 1.0, 10.0}`. Target familiarity uses `target_intro_end`; answerability uses `user_prompt_end`; unsupported-answer prediction may evaluate the output-proximal control but cannot use it for a pre-output claim.

The output-aligned control is frozen as an ordered 11-dimensional vector. It is computed at the assistant-prefix position with teacher-forced scoring of exactly two candidate strings: the example's registered target code and the literal `UNKNOWN`. A candidate's sequence log-likelihood is the sum of its conditional token log-probabilities. Normalized probabilities are the two-way softmax over those two sequence log-likelihoods. The control uses no generated completion and no behavioral outcome label. Its coordinates, in order, are:

1. target sequence log-likelihood
2. UNKNOWN sequence log-likelihood
3. target-minus-UNKNOWN log-likelihood
4. maximum registered sequence log-likelihood
5. registered-sequence log-sum-exp
6. normalized target probability
7. normalized UNKNOWN probability
8. binary target-versus-UNKNOWN entropy
9. absolute normalized probability margin
10. signed normalized probability margin
11. maximum normalized registered-sequence probability

These coordinates are an output-proximal control, not a hidden-state mechanism. H3/H4 primary models and standalone internal-feature claims exclude this vector and the `assistant_prefix_end` anchor. H5 is the prespecified nested exception: the identical surface-plus-output-margin control is included on both sides, and only the incremental contribution from registered prompt-end activation features is interpreted. H5 therefore does not treat the output margin itself as mechanistic evidence.

Grouped cross-validation entirely within `mechanism_train` selects PCA dimensionality and logistic `C` for every registered candidate. `locked_validation` never fits a transform, selector, or estimator; it selects only among train-fitted anchor/layer/family candidates and freezes thresholds, direction signs, and any intervention choice. No selected object is refit after validation.

The same four registered domains occur in every confirmatory split with the exact quotas in the Fellowship plan, and domain identity is therefore not a split-leakage key. Entity units, template families, and synthetic-name families remain split-isolated under their registered grouping rules. The fixed task contains one target relation, `archive_code`; `relation_id` is descriptive metadata rather than a split-isolation key, and H4 makes no relation-family transfer claim. F1's confirmatory interaction is pooled over the balanced domains; F2A performance is reported both pooled and by domain. Any additional domain-specific behavioral interaction is exploratory.

Reciprocal cross-condition transfer is exact. The familiarity probe is trained using each of the three answerability states in turn and evaluated on the union of the other two states, producing three rotations. The answerability probe is trained using each target-familiarity condition in turn and evaluated on the other condition, producing two rotations. Candidate selection aggregates only these registered rotations; each rotation and each distractor-familiarity cell is also reported separately.

Confirmatory interval estimation uses exactly 10,000 bootstrap replicates from the loaded confirmatory configuration. The artifact records requested draws, valid draws, discarded draws, seed, and resampling unit; it never reports discarded draws as valid. Full-pipeline label-permutation nulls use exactly the 99 integer seeds `2026072201` through `2026072299`, in ascending order. The canonical compact-JSON encoding of that complete seed list has SHA-256 `7aee4f4ee03201f4a8b7bee296294bc5c6a14a5251dfa71bb8cff15ce3d4e07f`. This list and hash are bound into every selection and test-null artifact.

The frozen dynamics representation uses adjacent layer transitions. For layer `l`, the adjacent difference is `delta_l = h_l - h_(l-1)`. Direction change is the scalar `1 - cosine(delta_(l-1), delta_l)`, computed with a fixed epsilon guard and reported invalid rather than imputed when either norm is non-finite. Static-plus-dynamics concatenates the registered static state, adjacent differences, and these normalized cosine changes before any train-only scaling or PCA. Layer-order nulls permute layer identity before recomputing the complete dynamics representation; dimension-matched random maps preserve input/output shape and are frozen by the same null-seed schedule.

Any H3 or H4 result described as pre-output must exclude transformer layer 25, `assistant_prefix_end`, and the 11-dimensional output-aligned control. H5/H6 activation features must also exclude layer 25 and `assistant_prefix_end`, but their nested models include the same output-aligned control in both baseline and candidate so that the measured contrast isolates the incremental activation contribution. Output-proximal and final-layer-inclusive standalone models remain mandatory comparison baselines and cannot satisfy H3/H4. H6 remains secondary and cannot rescue H3, H4, or H5.

In short, all confirmatory pre-output activation models exclude transformer layer 25 and `assistant_prefix_end`.

The confirmatory model and tokenizer revision is `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`; the Gemma chat-template SHA-256 is `ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6`; Gemma Scope residual SAE repository `google/gemma-scope-2b-pt-res` is pinned at `fd571b47c1c64851e9b1989792367b9babb4af63`; and optional circuit-tracer is pinned at `4bb8c0ea10bde09727e14565ec8469656880da53` (`v0.5.0`). Individual SAE paths are selected from the registered 16K family on training/validation only and still require the instruction-tuned loss-recovery gate. These pins are recorded in `data/fa/source_pins.json`; gated model access has not been tested by this amendment.

No outcomes were generated, inspected, or used to select the counts, pins, thresholds, template families, layers, directions, or claims described here.
