# Source-v6 R9 Audit-Qualified Development Amendment

**Date:** 2026-07-25
**Scope:** Open instrument development only
**Status:** To be Git-bound before R9 derivation or model output

## Reason for revision

R8 passed construction and independent structural audit, but its exhaustive
pre-model semantic audit blocked 251 of 576 questions. No prompt was sent to
Gemma and no Familiarity-by-Answerability endpoint was inspected. The blockers
were instrument defects: ambiguous entity labels, incomplete ordinary answer
surfaces, and a small number of source, temporal, or granularity concerns.

R9 is a development-only derivation from this model-blind instrument audit. It
does not repair or replace R8, and it is not a confirmatory corpus.

## Frozen R8 lineage

- construction commit:
  `0aa6f0ef0798073a0a464b3639479d41083d77ac`
- source frame SHA-256:
  `2b790d4607a52cc0bd5ebe9d40d577bb11c833a1366e5c31985ae4538a4d5450`
- source integrity SHA-256:
  `71510bce1a103351226eaf6450f2226336bf1e99534a9a1f025b2c076a22ff5b`
- semantic audit SHA-256:
  `e61d1ce59e13cc25f6b2b2b51764c5d05ae864321228eb6e0d65022087d8db22`
- semantic audit items SHA-256:
  `fd1cf8f6099c3f224d2a8f2e124f00594703f9c56040cb95389f44a36f9325c8`
- structured correction manifest SHA-256:
  `20feb70c4042301aa733472faf5b5863f5e6fe7f2098a01f3b77342ee9a47409`
- canonical correction-items SHA-256:
  `29323754adfed2487dd47e8d1efe8fd5f054a604eeeffb5968b60ac9169fe891`

## Registered derivation

1. A structured correction manifest is written by the R8 semantic auditor. It
   explicitly lists exact safe ordinary answer surfaces per question and
   hash-binds the R8 audit items. R9 code may not parse audit prose.
2. Exclude an entire candidate if any of its three questions has a blocker
   other than `ordinary_surface_missing`.
3. Apply only the exact registered correction surfaces. Do not generate
   spelling, punctuation, abbreviation, or transliteration variants.
4. Before selection, validate all remaining candidates together. Reject every
   side of normalized candidate/answer collisions, duplicate candidate names,
   reserved-output collisions, pseudonym collisions, and unsafe or ambiguous
   correction surfaces.
5. Fail closed unless at least 24 candidates remain in every domain.
6. Select exactly 24 candidates per domain by ascending:

   ```text
   SHA256("20260725:" + domain + ":" + qid), qid
   ```

7. Balance corrected and uncorrected candidates between the two development
   splits, then assign exactly 12 candidates per domain per split. The
   tie-breaker remains the registered SHA ordering.
8. Record one include/exclude decision per R8 QID, all input hashes, the
   correction-manifest hash, the exact construction commit, pinned model and
   tokenizer identity, and all materialization-code hashes in an immutable R9
   derivation manifest.
9. A new independent structural audit and a new auditor's exhaustive semantic
   audit of all 96 candidates and 288 questions are mandatory before Gemma.
10. Write a confirmatory-compatible exclusion manifest containing every R8
    candidate QID and every earlier registered exclusion. Future confirmatory
    corpus construction must consume this manifest.

## Development gate

For `instrument_development`:

- 48 candidates and 144 prompts;
- qualification requires at least two correct answers out of three;
- at least 8 qualified candidates per domain;
- at least 8 correct outputs per registered domain-relation cell;
- Place `P17` and `P30` require at least 9 correct outputs.

These are feasibility thresholds, not a statistical confirmation of the
research hypothesis. The same frozen thresholds are applied once to
`construction_validation` after instrument freeze.

## Stop and claim rules

Any change to correction surfaces, exclusion logic, collision policy, seed,
selection, split balance, questions, model, scoring, or thresholds requires
R10. Any R9 audit blocker stops R9 before Gemma.

All R8 and R9 QIDs are excluded from future confirmatory corpora. R9 may support
only this claim:

> The instrument is reproducible and feasible on a model-blind,
> semantically audited development corpus.

R9 cannot confirm Familiarity-by-Answerability.
