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

The confirmatory model and tokenizer revision is `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`; the Gemma chat-template SHA-256 is `ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6`; Gemma Scope residual SAE repository `google/gemma-scope-2b-pt-res` is pinned at `fd571b47c1c64851e9b1989792367b9babb4af63`; and optional circuit-tracer is pinned at `4bb8c0ea10bde09727e14565ec8469656880da53` (`v0.5.0`). Individual SAE paths are selected from the registered 16K family on training/validation only and still require the instruction-tuned loss-recovery gate. These pins are recorded in `data/fa/source_pins.json`; gated model access has not been tested by this amendment.

No outcomes were generated, inspected, or used to select the counts, pins, thresholds, template families, layers, directions, or claims described here.
