# Pilot Build Resume Amendment

**Dated:** 2026-07-23
**Scope:** Infrastructure-only, after the registered v4 screening gate passed
**Outcome impact:** None

The first `fa-build-pilot` invocation encountered a `FileExistsError` for a
valid deterministic tokenizer-pin artifact that already existed under the same
content-derived shard ID. The failure occurred before generation or behavioral
scoring.

The implementation now routes the deterministic tokenizer pin, probe metadata,
and prompt capability through the pre-existing fail-closed single-record resume
helper. Existing artifacts are reused only when record kind, canonical payload,
row count, and complete lineage match exactly. Any mismatch still stops the
build.

This amendment changes no candidate, match, prompt, template, generation,
metric, threshold, endpoint, or claim. It only makes an interrupted or
duplicated deterministic build transaction idempotent.
