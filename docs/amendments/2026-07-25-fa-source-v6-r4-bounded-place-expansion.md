# Source-v6 R4 Bounded Place-Pool Expansion

**Date:** 2026-07-25
**Scope:** Open instrument development only
**Status:** Frozen before R4 source construction or model output

## Motivation

R3 retained 44 complete, tokenizer-compatible Place records after its
pre-model source-quality filters. The registered frame requires 48 records to
create two domain-balanced development splits. R3 therefore failed closed
before split materialization and before any Gemma output.

This is a source-frame feasibility shortfall, not evidence for or against the
Familiarity-by-Answerability hypothesis.

## Single registered change

R4 preserves:

- all R3 Place eligibility and ambiguity filters;
- all question templates and answer-alias rules;
- the model, tokenizer, chat template, and generation configuration;
- the split seed and 24 candidates per domain per split;
- the two-of-three entity qualification threshold;
- every numerical development-readiness criterion;
- exclusion of every Source-v5, R1, and R2 QID.

R4 changes only the bounded QLever retrieval window:

- Person, Organization, and Creative Work remain at 4,000 ranked rows.
- Place increases from 4,000 to 6,000 ranked rows.

The per-domain limits, query text, raw response hashes, entity-cache seed hash,
source code hashes, and selected records are bound into the immutable R4 frame.
The R3 entity cache may be used only as a hashed fetch cache; records are
recomputed under the R4 code and missing entities are fetched from Wikidata.

R4 receives exactly one construction attempt with the 4,000/6,000 query
limits. If any domain produces fewer than 48 complete records, R4 fails. A
larger retrieval window or any changed eligibility rule requires a separately
registered R5 revision.

## Fail-closed sequence

1. R4 construction must produce exactly 48 complete records per domain.
2. Materialization must produce exactly 24 records per domain in each open
   development split, with no cross-split or prior-study QID overlap.
3. A pre-model source audit must reject unresolved ground-truth ambiguity,
   entity-answer alias collisions, or prompt defects.
4. Only then may Gemma screen `instrument_development`.
5. The development error packet must be reviewed by two independent raters;
   disagreements require a third independent adjudicator.
6. R4 advances only if the frozen criteria in
   `configs/fa_source_v6_r4_success_criteria.json` pass.
7. Only after the code gate and human audit pass may the one-time
   `construction_validation` split be opened.

## Claim boundary

R4 can establish only that an open development instrument has adequate and
auditable yield. It cannot confirm Familiarity, Answerability, hallucination,
intuition, a causal mechanism, or a Fellowship-level scientific result.
