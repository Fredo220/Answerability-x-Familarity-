# Source-v6 R7 Canonical Exact-Value Amendment

**Date:** 2026-07-26
**Scope:** Open instrument development only
**Status:** To be Git-bound before R7 construction or model output

## Reason for revision

R6 passed structural and provenance checks but failed its independent
pre-model semantic audit. No R6 model output exists. R7 changes only the
determinacy of source facts and accepted answer surfaces. It does not change
the hypothesis, model, generation settings, split sizes, qualification rule,
readiness thresholds, or downstream analysis.

## Registered changes

1. Every registered relation must have exactly one distinct non-deprecated
   Wikidata statement value. Current-location relations (`P17` and `P131` for
   Places; `P17` and `P159` for Organizations) additionally exclude statements
   carrying an end-time qualifier.
2. Entity-valued answers accept only the canonical English Wikidata label and
   its terminal-period variant. Property aliases and abbreviations are not
   accepted. Alphabetic canonical labels shorter than four alphanumeric
   characters are rejected. Year handling is unchanged.
3. Candidate and entity-answer English labels must be unique under
   case-insensitive comparison across Wikidata labels and must not equal an
   alternative label of another entity.
4. Candidate-to-candidate and candidate-to-answer conflicts are computed over
   the complete eligible pool before ranked selection. Every side of a
   conflict is rejected, including self-collisions. Answer-to-answer overlap
   remains allowed because multiple questions may share one factual answer.
5. Every R6 materialized candidate QID and every QID named by the R6 semantic
   audit is excluded from R7.
6. Source and frame revisions advance to R7.

The exact parser is deliberately conservative. Its false-negative risk is
measured by the frozen post-model manual audit: any sampled semantically
correct answer rejected because of an incomplete accepted-surface set fails
the instrument.

## Preserved design

R7 preserves the default query limit of `4000`, Place query limit of `6000`,
source ordering, tokenizer-matchability checks, split seed `20260725`,
`48` complete candidates per domain, `24` candidates per domain and split,
the two-of-three qualification threshold, the domain-by-relation readiness
matrix, the manual-audit design, and the pinned `google/gemma-2-2b-it`
configuration.

R7 does not reuse R6 entity-label payloads. Labels are fetched under the R7
source construction so the queried disambiguation check and materialized
surfaces refer to the same source revision.

## Stop rule

After this amendment is Git-bound, one syntax-only QLever preflight may execute
each frozen query with `LIMIT 1`. The preflight records only HTTP success and
binding shape; it must not retain or inspect candidate identities, ranks, or
domain yields. It is not a construction attempt.

R7 receives one construction attempt under these rules. Fewer than 48 complete
records in any domain fails R7. Any subsequent change to query semantics,
relations, wording, aliases, source types, thresholds, query limits, ordering,
or exclusion policy requires R8. A fresh independent pre-model audit must
inspect the materialized R7 instrument and report zero blockers before any
Gemma execution.

## Claim boundary

R7 remains open instrument development. Passing construction, semantic audit,
Gemma screening, manual audit, and construction validation establishes
instrument readiness only. It cannot confirm Familiarity-by-Answerability.
Confirmation requires a separately frozen, disjoint corpus and protected
analysis under the existing preregistered endpoints.
