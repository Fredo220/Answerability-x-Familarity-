# Confirmatory Source v5 Order-Invariance Correction

**Date:** 2026-07-24
**Scope:** Familiarity versus Answerability only
**Status:** Frozen before Gemma screening completions, human ratings, task
generation, activation extraction, or protected endpoint outcomes are opened

## Trigger

Source v4 successfully produced the registered 384-entity source pool, 1,152
screening questions, and three pseudonym reserves per source. Its independent
integrity audit passed. The separately registered idempotence audit then
replayed pseudonym construction with the candidate-manifest order printed in
the runbook. The replay changed 96 of 1,152 pseudonym rows and correctly
refused to overwrite the sealed v4 files.

The root cause was a global collision set in pseudonym construction. Each
entity's proposals were deterministic, but a colliding proposal was assigned
to whichever entity appeared first in the caller-provided manifest sequence.
Therefore the final pseudonym files were not invariant to a semantically
irrelevant input ordering.

No Gemma factual-screening completion, task output, activation, human rating,
or protected endpoint result was inspected before this correction.

## Frozen Correction

1. Source v4 is retained as a superseded, pre-outcome construction artifact.
   It must not be used for screening or inference.
2. The executable source revision is
   `fa-confirmatory-wikidata-v5`.
3. The executable pseudonym revision is
   `fa-confirmatory-pseudonyms-v3`.
4. Before global pseudonym collision handling, source candidates are sorted by
   `(coarse_type, qid, casefolded source name)`.
5. Pseudonym generation and every materialized split file must be byte
   invariant to the order of candidate-manifest arguments.
6. Source v5 is written to a new no-clobber directory,
   `data/fa/confirmatory_source_v5/`.
7. The fixed source rank, Wikidata queries, eligibility rules, exact surface
   predicate, three-reserve requirement, split quotas, model and tokenizer
   revisions, screening rules, human gate, endpoints, and claim boundary are
   unchanged.
8. If source v5 has fewer than 96 complete matches in any domain, the study is
   `not_evaluable`; v4 yield cannot be used to relax this gate.

## Frozen Lineage

- Preregistration SHA-256:
  `1bc81440b507cc30ac899962c8ca3121718870e22a62fc0988fa9f7b8a8ccdf7`
- Matching-policy SHA-256:
  `88e86649af3edfbd87dd6042e8ab4c3df33f461cb0c051b112c8a2359e379f61`
- Source matchability-policy SHA-256:
  `4bc62dc5bcee6f6c81ea55b4ba07dcd561a111348898723f97b48072685fe61b`
- Gemma/tokenizer revision:
  `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`
- Chat-template SHA-256:
  `ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6`
- Confirmatory Gemma screening completions inspected: `false`
- Human ratings issued: `false`
- Confirmatory task outputs or activations inspected: `false`
- Protected endpoints opened: `false`

## Required Verification

Before screening:

1. run the source-v5 builder to completion;
2. verify all registered source, split, domain, question, and reserve counts;
3. verify every integrity hash and global uniqueness constraint;
4. replay pseudonym construction with at least two different manifest orders;
5. require byte-identical split files and snapshot semantics;
6. commit the sealed source-v5 materialized artifacts without raw caches.

## Claim Boundary

The v4 and v5 builds are corpus-construction evidence only. They provide no
evidence for a Familiarity effect, an Answerability effect, or a mechanism.
The confirmatory estimand remains limited to Gemma-2-tokenizer-matchable entity
names sampled from the frozen Wikidata frame.
