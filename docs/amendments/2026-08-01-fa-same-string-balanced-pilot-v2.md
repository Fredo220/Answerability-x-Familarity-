# Same-String Balanced Pilot v2 Amendment

**Date:** 2026-08-01  
**Study ID:** `familiarity-answerability-same-string-feasibility-v2`  
**Run ID:** `same-string-feasibility-v2`  
**Status:** Registered pre-outcome pilot

## Purpose

This amendment registers a smaller, balanced Same-String behavioral pilot
after the v1 naturalness gate made the original confirmatory allocation
infeasible. It defines a new study identity and a new deterministic split. It
does not repair, replace, or reinterpret the v1 result.

The behavioral question remains whether unrelated contextual exposure to an
otherwise identical synthetic target string increases answer attempts when the
requested archive code is absent, beyond any corresponding exposure effect
when the code is supplied.

## Immutable v1 record

Study v1 is frozen as `not_evaluable`. Its independent naturalness audit
compiled 452 ratings for 192 pairs and deterministically classified 73 pairs as
accepted and 119 as excluded. Those accepted pairs could not fill the
registered v1 split-by-domain quotas; in particular,
`intervention_test/person` contained zero accepted pairs.

This is an instrument-feasibility outcome, not evidence for or against the
behavioral hypothesis. Before this amendment was written:

- no Same-String v1 target-model output had been generated or inspected;
- no protected behavioral endpoint had been opened;
- no behavioral estimate, bootstrap interval, probe result, or intervention
  result existed.

The v1 source, matching, ratings, exclusions, and status remain immutable. The
v2 allocation is bound to:

- config SHA-256:
  `3cb14e7fbd7df7005231b7dfe2df208d9ec313a64a25099d1e51043c062cc845`;
- match collection SHA-256:
  `28ab2745074e5d68834202328a4c7c10ab3f04ae2b854ff1ccc4267be94b10d4`;
- final naturalness-ratings data SHA-256:
  `78872ca0b1dce1def5ee841556b953ddda0b8d14b286676f2cf1ba2e51b208be`;
- rater A/B/C response SHA-256 values recorded in
  `docs/results/same_string_primary_preflight.json`.

## Eligible pool

Only the 73 pairs accepted by the registered v1 naturalness rule are eligible.
The eligibility decision is binary. Naturalness scores, disagreement patterns,
entity names, original split labels, and any semantic judgment are not used to
rank or allocate pairs.

The eligible counts are:

| Domain | Accepted pairs | Used in v2 |
|---|---:|---:|
| `creative_work` | 22 | 13 |
| `organization` | 13 | 13 |
| `person` | 25 | 13 |
| `place` | 13 | 13 |
| **Total** | **73** | **52** |

The 21 accepted but unallocated pairs are unused reserves. They may not replace
a selected pair after model outcomes are opened.

## Deterministic balanced allocation

Pairs are allocated using only their registered domain and immutable pair ID.
For every eligible pair, compute:

```text
allocation_hash = SHA256(UTF8(domain + "\0" + pair_id))
```

Within each domain, sort ascending by `(allocation_hash, pair_id)` and assign:

| Within-domain rank | v2 split | Pairs per domain | Total units |
|---:|---|---:|---:|
| 1-8 | `behavior_test` | 8 | 32 |
| 9-11 | `mechanism_train` | 3 | 12 |
| 12 | `locked_validation` | 1 | 4 |
| 13 | `probe_test` | 1 | 4 |
| - | **Total** | **13** | **52** |

Each unit retains the registered four-cell Same-String construction, producing
128 behavioral rows, 48 mechanism-training rows, 16 locked-validation rows,
and 16 probe-test rows. Pair IDs, target strings, synthetic names, prompts, and
ratings are not edited during reallocation.

The allocation and its hashes must be generated, audited, and sealed before
any target-model generation. Any mismatch in the eligible-pair hash, allocation
formula, split counts, domain counts, or four-cell completeness is a hard stop.

The pre-outcome local replay fixed the following expected identities:

- accepted-pair IDs SHA-256:
  `9028075b881bc4dfef0fbc933cca3741fb6e6ce2c6b52c5021ed021988bc6ac2`;
- allocation-policy SHA-256:
  `88e83a0b252e6a43d4d66c8650ec0d64d3548ddc224f7dbab528b89761af3023`;
- allocation SHA-256:
  `baca8a6b275a8718c5700c98f8044cffb7dea8d39aeb2bad3b39c52db7095365`;
- v2 matches SHA-256:
  `44808c789f5361a85350aca0583044692bd69c61c62165ba22c778769a268de5`.

## Registered behavioral pilot

The `behavior_test` split is the sole primary endpoint. The outcome remains
`answer_attempt`: exact normalized `UNKNOWN` is abstention and every other
completed response is an attempt. The primary difference-in-differences is:

```text
[attempt(high exposure, code absent) - attempt(low exposure, code absent)]
-
[attempt(high exposure, target bound) - attempt(low exposure, target bound)]
```

The existing Same-String behavioral support rules remain frozen:

- point estimate at least `0.05`;
- predicted-direction crossed-bootstrap 95% interval excludes zero;
- all four cells are complete for every included unit;
- valid output format is at least `0.95` in every cell;
- the target-bound capability-preservation interval does not cross the
  registered `-0.05` non-inferiority margin.

The crossed bootstrap resamples complete entity units and template families,
never individual prompt rows. The smaller sample size must be reflected in the
reported interval and a pre-outcome power or minimum-detectable-effect audit.
Power calculations may qualify the strength of the pilot but may not change
the endpoint, allocation, thresholds, or observed-result interpretation.

This is a registered pilot on one small open-weight model. Even a supported
result is not evidence for general pretrained familiarity, general
hallucination detection, model cognition in general, or behavior in frontier
models.

## Mechanistic scope

`mechanism_train`, `locked_validation`, and `probe_test` support only an
exploratory mechanistic pilot. Any probe, layer-dynamics, or representation
result must be labeled exploratory and reported with its small split sizes.

Mechanistic results cannot alter, rescue, or reinterpret the behavioral
decision. Study v2 contains no intervention split, no activation patching gate,
and no causal claim.

## One-shot protected endpoint

The protected `behavior_test` endpoint may be opened exactly once, and only
after all of the following are sealed and verified:

1. the v1 accepted-pair identity and naturalness artifact hashes;
2. the deterministic v2 allocation and complete four-cell manifest;
3. the pinned model, tokenizer, chat template, generation settings, config,
   repository commit, and analysis implementation;
4. a passed model-independent prompt audit;
5. a passed unprotected runtime smoke.

After unlocking, generation and evaluation must use the same endpoint identity,
manifest, and shard ID. An interrupted infrastructure run may resume from its
last hash-verified transaction, but no replacement endpoint, alternate split,
new seed, or substitute run may be created after outcomes are visible.

## Decision and claim policy

The primary behavioral result receives exactly one of these statuses:

- `supported`: the endpoint is evaluable and every registered behavioral
  support condition above passes;
- `not_supported`: the endpoint is evaluable but one or more support conditions
  fail, including a null, negative, imprecise, format-invalid, or
  capability-impairing result;
- `not_evaluable`: integrity, completeness, provenance, sealing, or scoring
  requirements prevent the registered endpoint from producing a valid
  decision.

A recoverable runtime interruption is recorded separately as
`infrastructure_failure` and may resume only under the same sealed identity. It
does not justify a new endpoint or a scientific claim.

All point estimates, crossed-bootstrap intervals, four cell rates, complete
unit counts, format-validity rates, capability-preservation results, exclusions,
gate decisions, and artifact hashes must be published regardless of status.
No exploratory mechanistic finding may upgrade `not_supported` or
`not_evaluable` behavioral evidence.
