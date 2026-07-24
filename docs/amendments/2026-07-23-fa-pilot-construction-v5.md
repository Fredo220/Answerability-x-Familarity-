# Familiarity vs. Answerability Pilot Construction Amendment v5

**Dated:** 2026-07-23
**Scope:** Development-only local smoke pilot
**Confirmatory impact:** None

## Frozen v4 screening outcome

The sealed v4 screening shard contains 72 immutable completions and passed the
registered exact-domain-balance gate. The selected pilot entities are:

- people: `Albert Einstein`, `Nelson Mandela`;
- places: `New York City`, `Los Angeles`;
- organizations: `United Nations`, `NASA`;
- creative works: `Hamlet`, `Moana`.

The screening completion SHA-256 is
`9287c555c9b596358301c81e5d35da2db79104cd19889f6f585bc254007370e0`.
The screened-match SHA-256 is
`0c99f182720738643f9c370a505d3615ebd66904300390fe555fea372c7e1376`.

## Pre-generation construction failure

The first prompt-capability artifact contained all 288 registered factorial
rows but no same-string rows. Its construction audit therefore failed only
`same_string_token_budget`; every other registered construction check passed.
No model generation was run from this incomplete capability.

The cause was an implementation error in `fa-build-pilot`: the tested
same-string builder was called only for confirmatory configurations even though
the development protocol requires a small same-string block in the pilot.

## Frozen correction before pilot outcomes

1. `fa-build-pilot` now appends the same four-row contextual-familiarization
   block per selected entity unit that `fa-build-confirmatory` uses.
2. The corrected pilot contains 288 factorial rows and 32 same-string rows,
   for 320 rows total.
3. Screening inputs, selected pairs, tokenizer matches, prompt templates,
   generation settings, behavioral thresholds, and confirmatory contracts are
   unchanged.
4. The incomplete 288-row artifact remains immutable and is retained as a
   failed construction artifact.
5. A new content-addressed prompt capability is required and must pass every
   registered construction audit before generation begins.

## Stop rule

If the corrected capability fails any construction audit, pilot generation
does not begin. No audit check may be disabled after observing the failure.
