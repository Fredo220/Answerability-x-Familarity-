# Source-v6 R8 Outcome

**Status:** Blocked before model execution

R8 was constructed from Git-bound commit
`0aa6f0ef0798073a0a464b3639479d41083d77ac`. The syntax-only QLever
preflight passed, and the registered construction produced exactly 48 source
records per domain. Deterministic materialization produced 192 candidates and
576 questions across two balanced development splits.

An independent code-path structural and provenance audit passed. It replayed
the frame, cache, exclusion, split, manifest, and semantic hashes without
calling the production audit helper. Its machine-readable artifact is:

```text
data/fa/development_source_v6_r8/structural_provenance_audit_v1.json
```

The exhaustive pre-model semantic audit then inspected all 192 candidates and
all 576 questions. It passed 325 questions and blocked 251. The blocker classes
were:

- unresolved entity-label ambiguity: 162 questions;
- missing ordinary correct answer surfaces: 118 questions;
- historical source uncertainty: 4 questions;
- granularity or source errors: 3 questions; and
- temporal or geopolitical uncertainty: 2 questions.

Counts overlap because one question may have more than one blocker. The
machine-readable artifacts are:

```text
data/fa/development_source_v6_r8/pre_model_semantic_audit_items_v1.jsonl
data/fa/development_source_v6_r8/pre_model_semantic_audit_v1.json
```

No R8 prompt was sent to Gemma and no Familiarity-by-Answerability endpoint was
inspected. R8 therefore provides evidence about instrument validity only. It
provides no evidence for or against the research hypothesis and cannot be
repaired in place.
