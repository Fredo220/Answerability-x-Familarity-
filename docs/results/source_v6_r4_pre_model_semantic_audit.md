# Source-v6 R4 Pre-Model Semantic Audit

**Date:** 2026-07-25
**Decision:** Failed before model access
**Claim scope:** Open instrument development only

R4 passed its formal construction checks: 192 unique candidates, 576 questions,
24 candidates per domain and split, zero cross-split QID overlap, zero overlap
with Source-v5/R1/R2, and valid file/provenance hashes.

An independent pre-model review nevertheless found construct-invalid items:

- all 48 P131 prompts left the intended direct administrative level implicit;
- five Place labels had real same-name referents;
- one Place had substantively ambiguous continent ground truth;
- two Organization rows were administrative territorial entities;
- one headquarters answer set mixed geographic granularities;
- one director answer set omitted the common correct surface form.

No Gemma output was generated for R4. The ten item-specific failures are
recorded in
`data/fa/development_source_v6_r4/pre_model_semantic_exclusions_v1.json`.
R5 is a separately registered instrument revision and may use these
development-only audit findings.

R4 provides no evidence for or against Familiarity, Answerability,
hallucination, or any mechanistic hypothesis.
