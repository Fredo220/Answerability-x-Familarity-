# Source-v6 R8 Local Disambiguation Amendment

**Date:** 2026-07-26
**Scope:** Open instrument development only
**Status:** To be Git-bound before R8 preflight, construction, or model output

## Reason for revision

R7 failed its registered syntax-only QLever preflight twice with HTTP 502 on
the first `LIMIT 1` Person query. No candidate bindings, yields, model outputs,
or research endpoints were inspected. The likely cause is the unbounded global
case-insensitive label and alternative-label joins. R8 changes only where
surface disambiguation is computed.

## Registered change

1. Remove global label and alternative-label scans from QLever.
2. Preserve R7's exact-one non-deprecated relation values and the end-time
   exclusions for current-location relations.
3. Fetch current English labels without an R6 seed cache.
4. Compute normalized candidate-name and candidate-to-answer conflicts over
   the complete eligible record pool before ranked selection. Reject every
   side of a conflict, including same-domain and self-conflicts. The exhaustive
   pre-model semantic audit covers remaining global ambiguity.
5. Preserve canonical-only entity answers, the period variant, rejection of
   alphabetic canonical labels shorter than four characters, and year parsing.
6. Require the independent pre-model semantic audit to inspect every one of
   the 192 materialized candidates and every one of their 576 registered
   question/ground-truth/accepted-surface triples. Any ambiguity, unsafe
   accepted surface, missing ordinary correct surface, granularity error,
   temporal error, or source error blocks R8 before Gemma. The audit manifest
   must hash-bind one machine-readable result per registered question and
   hash-bind complete candidate and question coverage.
7. Source and frame revisions advance to R8. Success thresholds, model,
   questions, order, limits, split sizes, seed, and downstream analysis remain
   unchanged.

## Feasibility and stop rule

After this amendment is Git-bound, one `LIMIT 1` syntax-only preflight may
validate HTTP success and binding columns without retaining identities, ranks,
or yields. R8 then receives one construction attempt with query limits `4000`
and `6000`, `48` complete candidates per domain, `24` per domain and split,
and seed `20260725`.

Any later change to query semantics, relations, wording, surfaces, source
types, thresholds, limits, ordering, audit coverage, or exclusions requires
R9. A zero-blocker exhaustive semantic audit and an independent structural and
provenance audit are mandatory before any Gemma execution.

## Claim boundary

R8 remains instrument development. Passing every R8 gate establishes
instrument readiness only. It cannot confirm Familiarity-by-Answerability;
that requires the separately frozen, disjoint confirmatory corpus and protected
analysis.
