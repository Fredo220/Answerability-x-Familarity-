# Confirmatory Execution Status

**Study:** Familiarity versus Answerability
**Date:** 2026-07-24
**Claim status:** No confirmatory empirical result exists

## Completed

- Preserved the frozen `fa-confirmatory-wikidata-v2` source snapshot after an
  audit found that multi-release works could accept a later year for a
  first-release question.
- Materialized the corrected, superseding
  `fa-confirmatory-wikidata-v3` source snapshot without overwriting v2.
- Materialized 384 split-isolated candidate entities and 1,152 registered
  screening questions.
- Excluded every pilot QID and six coarse-type-ambiguous QIDs.
- Preserved all non-deprecated values for multivalued entity properties and
  retained only the earliest non-deprecated positive year for year-valued
  screening questions.
- Verified exact split/domain quotas, global QID/name/ID uniqueness, and exactly
  three source-bound questions per entity.
- Sealed every materialized file and raw QLever/Wikidata cache file in
  `data/fa/confirmatory_source_v3/source_integrity_v1.json`.
- Source-integrity artifact SHA-256:
  `98d140b0f39a6f8bd2db8a1b861f5bb33ac91e3a88eac786f5ec828158f43964`.
- Completed the tokenizer-only v3 feasibility audit. Only 167/384 candidates
  admitted complete three-pseudonym reserves, including 10/96 places in the
  assigned pool and 18/150 places across the complete cached source frame.
- Froze the v4 source amendment before any Gemma screening completion was
  opened. It raises the fixed maximum source rank to 1,200, applies exact
  matchability before split assignment, and narrows the estimand to
  Gemma-tokenizer-matchable entity names.
- Built and audited source v4, then rejected it before screening when the
  separate idempotence replay found 96 order-dependent pseudonym rows.
- Froze source v5 and pseudonym-generator v3 to canonicalize candidate order
  before global collision handling. All scientific endpoints, quotas, source
  ranks, model pins, and claim boundaries remain unchanged.
- Materialized and independently audited source v5:
  384 globally unique entities, 1,152 source-bound screening questions, and
  1,152 globally unique pseudonym reserves.
- Verified exact split and domain quotas, three questions and three reserves
  per entity, every registered source-integrity hash, and absence of real-name
  versus pseudonym collisions.
- Replayed pseudonym construction with two different manifest orders. All five
  split files were byte-identical and the normalized snapshot semantics were
  identical.
- Source-v5 integrity artifact SHA-256:
  `ac5c7c812482d38af397c1ba5a21355c82ca665d652159eb71c856fa87595bf5`.
- Complete matchability counts before selecting the first 96 per domain were:
  creative works 448, organizations 537, people 484, and places 108.
- Added fail-closed CLI checks requiring the exact 2x source pool and the
  active frozen integrity artifact before confirmatory screening.
- Verified the confirmatory corpus, rating, and collection paths with
  `103 passed`.
- Rebuilt the pinned Graphify architecture graph and report.

## Tokenizer Access

Hugging Face authentication, repository access, the exact pinned revision, and
the protected Gemma tokenizer were verified successfully.

## Current Construction Gate

The v3 corpus cannot satisfy the registered domain quotas under the frozen
exact-token controls. Source v4 met the quotas but is superseded because its
pseudonym output was order-dependent. Source v5 has now passed the fixed-rank,
domain-balance, source-integrity, and order-invariance gates. Pinned Gemma
factual screening is the next unopened gate. Human packet issuance remains
closed until screening and deterministic reserve assembly pass. The model and
tokenizer revision remain:

```text
google/gemma-2-2b-it
299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
```

No alternate model or mutable revision may replace this pin.

## Human Gate

Human packets cannot be issued before pinned Gemma screening and deterministic
matching produce the verified 244-pair collection.

- Two distinct real people must independently rate every issued pair.
- A third distinct person rates only disagreements emitted by the compiler.
- AI agents, the researcher, copied identities, and empty templates are not
  human evidence.

## Protected Endpoints

No `behavior_test`, `probe_test`, or `intervention_test` endpoint was opened.
After valid Gemma authentication and human ratings, follow
`docs/familiarity_answerability_runbook.md` without changing thresholds,
selection rules, model pins, or split assignments.

## Verification Boundary

The focused confirmatory suite completed with `103 passed`. A repository-wide
run covering 1,040 tests from several independent research tracks was stopped
after 19 minutes at `423 passed`; it was not a completed full-suite run and is
not reported as one.
