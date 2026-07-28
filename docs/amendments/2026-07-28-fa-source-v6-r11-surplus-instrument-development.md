# Source-v6 R11 Surplus Instrument Development

**Date:** 2026-07-28
**Scope:** Open instrument development only
**Status:** Registered before R11 model outputs

## Trigger and preserved evidence

R10 was a one-shot held-out instrument-readiness test. The reported run failed
its frozen gate: the entity-level counts were below eight for `creative_work`
and `person`, and the `P131` relation had no exact successes. R10 is not
reinterpreted or rescued.

The canonical R10 archive is not currently committed to this repository.
Before R11 is frozen for validation or publicly reported, its execution
identity, item rows, yield summary, gate result, member hashes, model revision,
and code commit must be ingested and hash-bound. A screenshot or prose summary
is not a substitute. This dependency does not prevent open R11 instrument
development.

R11 reuses the existing pinned Gemma runner, exact-answer parser, batching,
checkpointing, provenance, human-audit, F1, and F2A infrastructure. It does not
restart the research project and does not open any protected endpoint.

## Fixed construct

The familiarity label remains:

```text
at least two exact successes among three frozen relation questions
```

The threshold is not lowered. R11 changes the instrument construction because
R10 showed that the previous relation set and small candidate pool were not a
reliable way to obtain enough qualified entities.

R11 cannot test the Familiarity-by-Answerability hypothesis. It can only show
that a candidate instrument is feasible.

The R11 `place` stratum is restricted to countries. R10 demonstrated that the
city-specific direct-administrative-parent question was not a functioning
exact-answer instrument. Countries provide five stable, source-auditable
relations: capital, continent, official language, currency, and highest
natural point. This scope change is registered before R11 outputs and must be
reported as a limitation; it is not generalized back to all places.

The broad Wikidata `organization` superclass returned countries through its
subclass graph and no complete five-relation candidates. Before any R11 model
output was generated, the R11 organization sampling population was therefore
narrowed to business enterprises (`Q4830453`) and the rank query was
deduplicated. The resulting claim is limited to this registered organization
subpopulation.

The model-blind source audit also found that raw Wikidata aliases admitted
codes and historical values, while `P47` did not consistently denote the land
border asked by the prompt. Before model execution, R11 was repaired to use
primary English labels, preferred or non-ended values for current relations,
and `P610` instead of `P47`. Earlier generated source revisions remain failed
development artifacts and are not screened.

## One broad development revision

R11 evaluates a predeclared bank of five candidate relations per domain on an
open `instrument_development` split with 32 candidates per domain. Candidate
facts, answer aliases, prompt wording, and source provenance must pass a
model-blind structural and semantic audit before Gemma is run.

Every selected candidate must have all five registered facts before model
execution. Missing rows are a source-construction failure, not a model failure,
and may not be silently removed after screening.

The relation bank and sample sizes are stored in
`configs/fa_source_v6_r11_development.json`. They may not change after the
first R11 model completion is opened. Missing or ambiguous facts are removed
before model execution and recorded; model failures may not cause aliases to
be added.

For every domain, all three-relation combinations are evaluated. The selected
combination is determined by this fixed order:

1. greatest number of entities passing the two-of-three screen;
2. greatest minimum success count among the three relations;
3. greatest total success count;
4. lexicographically smallest relation-ID triplet.

Every combination and its score is retained. This makes the adaptation visible
and reproducible rather than hand-selecting a favorable triplet.

## Frozen validation

After selection, the exact triplet, prompts, aliases, parser, model identity,
and two-of-three threshold are frozen. They are then evaluated once on a fresh
`construction_validation` split with 16 candidates per domain.

Development and validation must have no shared QIDs or entity IDs. Validation
passes only if:

- every domain has at least eight two-of-three-qualified entities;
- every selected relation has at least one exact success;
- the registered human scoring audit passes.

Failure is reported as `not_evaluable`. No relation, threshold, alias, or
candidate may be changed against that validation split.

## Confirmatory boundary

A passing R11 validation permits construction of a new entity-disjoint
confirmatory corpus. It does not confirm Familiarity versus Answerability.
Only the later frozen F1 interaction can provide behavioral evidence, and F2A
remains a separately gated mechanistic pilot.

All R8-R10 failures, R11 combination scores, excluded facts, model outputs,
human audits, and validation outcomes remain public negative or feasibility
evidence.
