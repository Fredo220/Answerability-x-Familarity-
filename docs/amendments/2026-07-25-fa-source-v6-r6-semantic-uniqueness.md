# Source-v6 R6 Semantic Uniqueness Amendment

**Date:** 2026-07-25
**Scope:** Open instrument development only
**Status:** To be Git-bound before R6 construction or model output

## Reason for revision

R5 passed formal construction but failed its independent pre-model semantic
audit. No R5 model output exists. R6 changes only source determinacy and scoring
validity; it does not change the research hypothesis, model, generation
configuration, split sizes, qualification rule, or numerical readiness gates.

## Registered changes

1. Place candidates must have an English label that is globally unique among
   Wikidata entities returned by the registered label check.
2. Creative works must have an English title unique among film entities.
3. Organizations must have exactly one P17 value, exactly one P159 value, and
   exactly one P571 value.
4. Organization P159 values must be human settlements (`Q486972` hierarchy),
   excluding building-level headquarters.
5. Alphabetic accepted aliases shorter than four alphanumeric characters are
   removed. Canonical labels and longer aliases remain eligible.
6. Every materialized R4 and R5 candidate QID is excluded because those
   candidates were inspected during open development.
7. Source/frame revisions advance to R6.

R6 preserves the 4,000 default and 6,000 Place query limits, source ordering,
tokenizer matching, split seed `20260725`, 24-per-domain split sizes,
two-of-three qualification threshold, domain-by-relation readiness matrix,
and pinned Gemma configuration. The human packet is fixed at at most four
model-scored errors plus two model-scored successes per domain with seed
`20260725`; outcome strata are blinded. If a domain contains fewer than four
errors, every observed error in that domain is audited. Human and model
correctness must agree on every sampled item.

## Stop Rule

R6 receives one construction attempt under these rules. Fewer than 48 complete
records in any domain fails R6. Any later query, wording, alias, source-type,
threshold, or exclusion-policy change requires R7. A fresh independent
pre-model semantic audit must report zero blockers before Gemma runs.

## Claim Boundary

R6 remains instrument development. Passing R6 construction and screening gates
establishes instrument readiness only. It cannot confirm
Familiarity-by-Answerability; that requires a separately frozen confirmatory
corpus and protected analysis.
