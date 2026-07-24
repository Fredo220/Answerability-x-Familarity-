# FA Pilot Synthetic-Reserve Amendment V7

Date: 2026-07-23

## Scope

This amendment applies only to deterministic development-pilot matching after
the V4 screening audit and before any 320-row pilot generation.

## Trigger

The V6 matcher correctly rejected the V4 synthetic pool:

- `NASA` had no synthetic organization satisfying both registered tokenizer
  frames.
- `Hamlet` and `Moana` shared only one eligible synthetic title, `Velora`, so a
  one-to-one assignment was impossible.

No behavioral or activation outcome was generated or inspected.

## Append-Only Reserve Pool

`data/fa/pilot_inputs/synthetic_candidates_v5.json` preserves every V4 row in
its original order and appends:

- organization acronyms: `MER`, `SEN`, and `TER`;
- creative-work titles: `Selora`, `Thalen`, and `Eloria`.

All six additions were selected using only the pinned Qwen3-1.7B tokenizer and
the registered type, split, word-count, capitalization, character-distance,
character-inventory, sentence-frame, and same-string-frame constraints.

The extra candidates are reserves. Deterministic bipartite matching selects the
first complete one-to-one assignment under the versioned V6 matching policy.
The independent blinded naturalness audit remains mandatory and can exclude
these pairs. No rating is inferred from tokenizer compatibility.

## Decision Rule

The immutable V4 screening completion is reused. Screening scores and the
domain-balanced selection are deterministically recomputed under the unchanged
candidate, question, parser, and threshold rules. Because screening-audit
lineage also binds the synthetic manifest, the recomputation is written under a
new lineage-derived audit ID rather than overwriting the V4 audit. Only this
audit recomputation, matching, and prompt construction are rerun. Generation
remains prohibited until all 320 prompt rows pass every registered dataset
audit.

## Claim Boundary

Tokenizer matchability is a construction property, not evidence that the
synthetic names are natural, unfamiliar to the model, or behaviorally
equivalent to the real entities.
