# Confirmatory Source v3 Amendment

> Historical source correction only. The tokenizer-only matchability audit
> later showed that v3 cannot fill the registered domain quotas. Execution is
> governed by the v4 matchability-frame amendment without overwriting v3.

**Date:** 2026-07-24
**Scope:** Familiarity versus Answerability only
**Status:** Frozen before Gemma screening completions, human ratings, task
generation, or protected endpoint outcomes are opened

## Reason

The v2 source builder preserved every non-deprecated year attached to a
Wikidata property. That was too permissive for registered questions asking
when an organization was founded or when a creative work was **first**
released. A response naming a later release year could therefore have been
scored as familiar even though it did not answer the question as written.

## Correction

For year-valued screening fields, v3 accepts only the earliest
non-deprecated positive year and its period-terminated form. Multivalued
entity-valued fields continue to preserve all non-deprecated labels and
aliases.

The source queries, retrieval date, QID exclusions, split seed, split/domain
quotas, screening threshold, model and tokenizer revision, prompt wording,
human-rating rule, hypotheses, and protected endpoints are unchanged.

The v2 snapshot and integrity manifest remain immutable. v3 is materialized
in a new directory with a new source revision and integrity manifest.

## Screening Namespace Clarification

Split-specific factual screening is a source-qualification stage. Its
artifacts may retain the registered split label for lineage, including
`behavior_test`, `probe_test`, and `intervention_test`, but this does not open
the corresponding study endpoint. At screening time no synthetic registry
task prompt, behavioral outcome, activation endpoint, intervention, or
confirmatory report exists. The one-use endpoint guards continue to apply to
the later task evaluations.

## Access Record

Before this amendment was written, the Hugging Face CLI identity and access
to the exact pinned revision
`google/gemma-2-2b-it@299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`
were verified. The protected tokenizer loaded successfully and reproduced
the registered chat-template hash. No model screening completion or study
outcome was generated.

## Claim Boundary

This correction creates no empirical evidence. It prevents an avoidable
false-positive familiarity label in the pre-outcome qualification stage.
