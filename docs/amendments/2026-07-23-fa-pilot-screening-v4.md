# Familiarity vs. Answerability Pilot Screening Amendment v4

**Dated:** 2026-07-23
**Scope:** Development-only local smoke pilot
**Confirmatory impact:** None

## Frozen v3 outcome

The sealed v3 Qwen3-1.7B screening run completed with 60 immutable
completions. The registered two-of-three gate qualified at least two candidates
in the person, place, and organization strata, but only one candidate in the
creative-work stratum:

- `pilot-work-hamlet`: three of three;
- `pilot-work-mona-lisa`: one of three;
- `pilot-work-starry-night`: zero of three;
- `pilot-work-girl-pearl-earring`: zero of three; and
- `pilot-work-guernica`: one of three.

The exact-domain-balance gate therefore stopped the pilot. This is a negative
development result about candidate feasibility, not evidence for or against the
Familiarity-vs.-Answerability hypotheses.

The retrospective machine-readable audit is sealed at
`runs/familiarity_answerability/smoke-qwen17b-v1/shards/pilot/screening-audit-entity-screening-20260723-v3-0002.jsonl.manifest.json`
with data SHA-256
`28a0b7462d61b8f8efb909a6ec86d1a20d9e2d5936d1a5d51ec6b944f82a684e`.

## Changes frozen before v4 outcomes

1. The v3 candidates, questions, aliases, ordering, parser, threshold, model
   revision, chat template, generation settings, and artifacts remain
   unchanged.
2. The v4 manifests are append-only copies of v3. They add four independently
   sourced, high-familiarity creative works in this deterministic order:
   `Jaws`, `Mulan`, `Moana`, and `Skyfall`.
3. Every added work has exactly three direct factual questions, an immutable
   Wikidata QID, and an authoritative first-party source.
4. One additional synthetic title, `Velora`, is added. Under the pinned
   Qwen3-1.7B tokenizer it has the same framed token count as the four added
   titles and satisfies the frozen word-count, capitalization, character
   tolerance, and character-inventory constraints.
5. The four added candidates are reserve candidates only. They are introduced
   because v3 stopped specifically in the creative-work stratum. No other
   stratum is expanded.
6. A complete new sealed screening shard is required. No v3 completion is
   copied into or reused by the v4 artifact.
7. Selection remains the first two qualifying candidates in original manifest
   order within every domain. Because the v3 prefix is preserved, Hamlet remains
   the first qualified creative work; the first passing appended reserve becomes
   the second.
8. The v4 completion artifact remains sealed until the registered
   `fa-screen-entities` gate has run.

## Frozen execution contract

- Config: `configs/familiarity_answerability_qwen17b_smoke.json`
- Config canonical SHA-256:
  `c8df1f013c2cc7281c6d29584e6f1bbcaabc5798d6ed88b6fe1007dbd4c2ce3c`
- Candidate semantic SHA-256:
  `91fae378d19410d9216c54aa2043ad3a247b3d072d79306d8fb7e22d91dedd3f`
- Question semantic SHA-256:
  `60c7d7ab04d493bb511dd101df59d58b1476243e569376641505b65e9870428d`
- Synthetic-candidate semantic SHA-256:
  `02d2b27b3fe098a7844da9a51b02fee66c35c8d7737a7790b66e793059290763`
- Screening-parser SHA-256:
  `38fa1a9d7960b9ac72b6f7f68d6ef58caf14730ae9de6de3614da3e0e8d87be7`
- Model revision:
  `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Chat-template SHA-256:
  `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`
- Expected completions: `72`
- Planned first attempt: `entity-screening-20260723-v4-0001`
- Later attempt IDs may be used only after an infrastructure failure. Model
  outputs from a completed attempt are never reused in another attempt.

## Stop rule

If v4 has fewer than two qualifying candidates in any registered domain, the
local pilot stops again. The threshold, aliases, parser, quota, and candidate
ordering will not be relaxed after opening the sealed v4 completions. Any
further expansion requires another amendment and a new shard ID.
