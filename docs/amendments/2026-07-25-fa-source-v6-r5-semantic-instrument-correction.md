# Source-v6 R5 Semantic Instrument Correction

**Date:** 2026-07-25
**Scope:** Open instrument development only
**Status:** Development protocol fixed before model output

This open-development amendment was authored before the R5 build but was not
cryptographically commit-bound before construction. The first immutable Git
binding therefore covers the completed R5 source frame, its independent
pre-model audit, the registered screening criteria, and all executable code
before any Gemma output. R5 is an instrument-development result, not a
confirmatory hypothesis test. A later confirmatory corpus must be committed
before its protected endpoints are opened.

## Reason for revision

R4 passed formal source checks but failed an independent pre-model semantic
audit. No R4 model output exists. R5 corrects only the observed instrument
defects; it does not change the research hypothesis or numerical gates.

## Registered changes

1. The P131 prompt explicitly requests the direct Wikidata P131
   administrative territorial entity rather than an unspecified
   administrative region.
2. The Organization query excludes entities in the administrative territorial
   entity hierarchy (`Q56061`).
3. The ten QIDs in the frozen R4 semantic-exclusion manifest are excluded
   before R5 selection.
4. The source/frame revisions advance to R5.

R5 preserves the model, tokenizer, chat template, decoding, relation set,
4,000/6,000 query limits, source ordering, tokenizer matching, split seed,
24-per-domain split sizes, two-of-three qualification threshold, and the full
domain-by-relation readiness matrix.

## Stop rule

R5 receives one construction attempt using the registered limits and
exclusions. Fewer than 48 complete records in any domain fails R5. Any further
query, wording, alias, source-type, or threshold change requires R6. The
independent semantic source audit must pass before Gemma is run.

The registered human error audit uses four model-scored errors per domain,
selected with seed `20260725`. Two independent initial raters label each item;
a distinct adjudicator resolves disagreements only. Any adjudicated
`ambiguous_ground_truth`, `incomplete_alias_set`, `wrong_granularity`,
`parser_failure`, `source_error`, or `other` item fails the instrument gate.

## Claim boundary

R5 remains instrument development. Passing construction and screening gates
cannot confirm Familiarity-by-Answerability; it only permits a separately
frozen confirmatory corpus to be built later.
