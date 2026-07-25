# Source-v6 R5 Pre-Model Semantic Audit

**Date:** 2026-07-25
**Scope:** Open instrument development only
**Result:** Blocked before model execution

## Method

An independent reviewer audited all 48 selected Place entities and all 144
Place prompts across both R5 splits. The reviewer also audited a deterministic
SHA-256-sorted sample of eight entities per split for each non-Place domain
(48 entities and 144 prompts). The audit checked source reconstruction,
entity-label ambiguity, question determinacy, relation granularity, accepted
aliases, and split/exclusion integrity.

The reviewed source-integrity SHA-256 was:

```text
e6b41fcee8383ebb0050bbedd3533b9a1b61e3583cdebae53d69e0e1ccbe8133
```

## Passing Checks

- 192 candidates reconstructed from the pinned source records.
- No source-to-manifest mismatch was found.
- No cross-split QID overlap was found.
- All source, snapshot, and exclusion hashes matched.

## Blocking Findings

1. Nineteen Place labels were shared by another geographic entity, so bare-name
   prompts did not uniquely identify the intended P131 target.
2. `Beauty and the Beast` (`Q19946102`) and `Robin Hood` (`Q223559`) were
   ambiguous creative-work titles.
3. AMD (`Q128896`) and Hyundai Motor Company (`Q55931`) had multiple
   incompatible targets for a singular question.
4. Short aliases such as `BY`, `VAN`, `YE`, `IL`, `EL`, `GR`, and `NA` could
   produce false-positive exact matches.
5. Some headquarters values were buildings while the prompt asked for a place,
   producing inconsistent answer granularity.

## Decision

R5 is not model-ready. No Gemma output was generated and no protected
confirmatory endpoint was opened. The findings motivate a separately
registered R6 instrument revision; they do not provide evidence for or against
the Familiarity-by-Answerability hypothesis.
