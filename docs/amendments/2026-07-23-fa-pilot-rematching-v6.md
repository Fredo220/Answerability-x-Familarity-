# FA Pilot Rematching Amendment V6

Date: 2026-07-23

## Scope

This amendment applies only to the development-only Qwen3-1.7B pilot. It does
not change the research question, screening answers, domain quotas, outcome
metrics, confirmatory thresholds, or protected endpoints.

## Trigger

The V4 entity screen passed and the V5 pilot builder produced the registered
288 factorial rows plus 32 same-string rows. Before any pilot generation was
run, the dataset audit found that four real/synthetic pairs differed by one
token between the high- and low-exposure versions of the same-string control.
The original matcher checked the registered sentence frame but not the exact
repeated neutral-exposure prefix.

## Frozen Information

- The immutable V4 screening-completion shard is reused.
- The immutable V4 passed screening audit is reused.
- The eight selected real entities remain fixed.
- No model output from the 320-row pilot exists and no outcome was inspected.

## Registered Correction

Synthetic matching must now satisfy token-count equality in both:

1. the pre-existing registered sentence frame; and
2. the exact registered same-string exposure prefix through the `Task:`
   boundary.

The deterministic matching implementation and its registered constants are
hashed as `matching_policy_sha256`. New match shards include the first twelve
hexadecimal characters of that hash in their shard ID and bind the complete
hash in lineage. Consumers reject artifacts produced by a different matching
policy.

## Decision Rule

Only deterministic rematching is rerun. Model screening is not rerun. The pilot
may proceed to generation only if the rebuilt 320-row manifest passes every
registered construction audit, including `same_string_token_budget`.

## Claim Boundary

This is a pre-outcome construction correction. It provides no evidence for H1,
F2A, familiarity, answerability, hallucination, or any mechanistic claim.
