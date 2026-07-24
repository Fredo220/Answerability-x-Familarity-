# Confirmatory Token-Matchability Amendment

**Date:** 2026-07-24
**Scope:** Familiarity versus Answerability only
**Status:** Frozen before Gemma screening completions, human ratings, task
generation, or protected endpoint outcomes are opened

**Superseded for execution:** The tokenizer-only v3 audit showed that this
post-split pool cannot satisfy the registered domain quotas. Source v4 applies
the same exact controls before split assignment under the separately frozen
`2026-07-24-fa-confirmatory-source-v4-matchability-frame.md` amendment. This
document remains an immutable record of the trigger and failed v3 design.

## Trigger

The first real tokenizer-only construction run showed that some highly
token-compact real names cannot receive three novel pseudonyms under both
registered exact-token controls. `Oppenheimer`, for example, is one token
after a space in the sentence frame but three tokens without a leading space
at the start of the same-string prefix. None of 5,000 deterministic
random-character proposals matched both counts.

A vocabulary-word fallback would reproduce the counts only by selecting
strings already represented as whole leading-space tokens. Treating those
strings as unknown synthetic entities would weaken the familiarity contrast.
The matching tolerances are therefore not relaxed and tokenizer-vocabulary
words are not relabeled as synthetic merely to fill the pool.

## Registered Resolution

The generator still attempts to create three unique exact-compatible
pseudonyms for every source candidate. Partial reserves are discarded.
Candidates without a complete three-name reserve are recorded in the
tokenizer-only synthetic snapshot and are ineligible for the later
domain-balanced familiarity selection.

This matchability filter uses only frozen source strings, the pinned tokenizer,
and the unchanged surface predicate. It is applied before the two-of-three
factual familiarity result can enter domain-balanced selection. If the
intersection of factual qualification and exact matchability cannot fill a
registered split/domain audit quota, the study stops as `not_evaluable`.
Source order, quotas, reserve counts, and outcome thresholds are not changed.

The human naturalness audit remains mandatory for every issued pair.

## Frozen Lineage

- Preregistration SHA-256:
  `1bc81440b507cc30ac899962c8ca3121718870e22a62fc0988fa9f7b8a8ccdf7`
- Source v3 integrity SHA-256:
  `98d140b0f39a6f8bd2db8a1b861f5bb33ac91e3a88eac786f5ec828158f43964`
- Matching-policy v4 SHA-256:
  `33f4ae8a46ced71dcd3abb0489e0eeb6d0126150e4829af908ba7a789a3629f0`
- Gemma/tokenizer revision:
  `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`
- Chat-template SHA-256:
  `ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6`
- Confirmatory outcomes inspected: `false`
- Human ratings issued: `false`
- Protected endpoints opened: `false`

## Claim Boundary

This is a sampling restriction, not evidence. Any final report must disclose
that exact tokenizer matching excludes some morphologically token-compact
familiar entities and limits the population to source names admitting a
novel, exact-compatible pseudonym.
