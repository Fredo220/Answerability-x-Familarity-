# Familiarity vs. Answerability Pilot Screening Amendment v3

**Dated:** 2026-07-23
**Scope:** Development-only local smoke pilot
**Confirmatory impact:** None

## Frozen prior outcomes

The immutable Qwen3-0.6B v1 and v2 screening artifacts remain negative
development results. Under the exact alias gate, v2 did not qualify the eight
required pilot entities.

A non-sealed Qwen3-1.7B feasibility process was then run on the v2 questions.
It was interrupted before completion and cannot be scored as a study artifact.
The observed prefix of outputs showed:

- substantially better high-salience factual recall than Qwen3-0.6B;
- repeated leading-colon formatting despite the short-answer instruction; and
- CPU feasibility, but latency unsuitable for confirmatory execution.

These observations motivated the changes below. They are not evidence for the
Familiarity-vs.-Answerability hypotheses.

## Changes frozen before v3 outcomes

1. The active smoke runner is `Qwen/Qwen3-1.7B` at immutable revision
   `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
2. Its exact chat-template SHA-256 is
   `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
3. Greedy generation, the two-of-three qualification rule, tokenizer matching,
   the required eight pilot pairs, and all confirmatory contracts are unchanged.
4. Short-answer extraction is deterministic and model-independent:
   - remove a completed `</think>` envelope;
   - retain the final nonempty line;
   - if that line contains a colon, retain only the text after its final colon;
   - remove one matching pair of surrounding single or double quotes;
   - otherwise preserve the answer text, including punctuation.
5. The candidate pool may be expanded with independently sourced,
   high-familiarity entities. Candidate facts, aliases, sources, pseudonyms, and
   their deterministic ordering must be checked in before the v3 shard is run.
6. Qualification selects the first two passing candidates in deterministic
   manifest order within each registered domain: person, place, organization,
   and creative work. No alias, parser rule, source fact, candidate ordering, or
   domain quota may be changed after inspecting the sealed v3 completions.
7. The first v3 execution attempt was interrupted before any completion artifact
   was written, after a lineage review found that the candidate-manifest hash
   canonicalized away row order. Because row order determines reserve selection,
   the hash is now explicitly order-sensitive. The sealed run therefore starts
   with a new shard ID.
8. The second v3 execution attempt was externally interrupted during model
   generation before any completion artifact was written. Its shard ID is
   retired without interpretation; the next execution uses a new shard ID with
   identical frozen inputs and settings.

## Stop rule

If any domain has fewer than two qualifying candidates, the local pilot stops.
The threshold and domain quota will not be relaxed. Any later pool expansion
requires a new dated amendment and a new shard ID.
