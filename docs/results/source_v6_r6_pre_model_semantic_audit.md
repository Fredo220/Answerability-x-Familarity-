# Source-v6 R6 Pre-Model Semantic Audit

**Status:** Blocked before model execution

R6 was materialized successfully and passed its independent structural and
provenance audit. Its source-integrity SHA-256 is
`eb1d27fbfe0323c2190f2c5baf9f6f0410fa9a1a3fede49ac925e3fe973d6965`.

An independent semantic audit then reviewed every Place candidate and eight
deterministically SHA-256-sorted candidates per non-Place domain and split.
It found eight blocking defect classes:

1. unresolved short-label scoring policy;
2. candidate labels colliding with answer labels;
3. relations with multiple valid ground-truth values;
4. people with multiple valid occupations;
5. ambiguous candidate labels;
6. unsafe answer aliases and abbreviations;
7. inconsistent geographic granularity; and
8. historical-versus-current ambiguity.

The machine-readable audit is
`data/fa/development_source_v6_r6/pre_model_semantic_audit_v1.json`.
The affected QIDs are frozen in
`data/fa/development_source_v6_r6/r7_semantic_blocker_exclusions_v1.json`.

No R6 prompt was sent to Gemma and no model output or Familiarity-by-
Answerability endpoint was inspected. R6 therefore failed as a measurement
instrument; it provides no evidence for or against the research hypothesis.
The registered R6 stop rule requires any correction to advance to R7.
