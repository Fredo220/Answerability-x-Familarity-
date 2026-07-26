# Source-v6 R10 Held-Out Instrument Validation

**Date:** 2026-07-26  
**Scope:** Open instrument development only  
**Status:** Must be Git-bound before opening R9 `construction_validation`

## Trigger

R9 completed its registered `instrument_development` run and failed the
instrument-readiness gate. The failure does not test Familiarity by
Answerability. It showed that the strict surface scorer undercounted valid
occupation answers and that relation-level quotas were stricter than the
registered entity-level familiarity definition.

R9 remains immutable and `not_evaluable`.

## Unchanged evidence

R10 reuses the frozen R9 corpus without adding, removing, reordering, or
reselecting candidates:

- source revision: `fa-development-source-v6-r9`;
- model and tokenizer: pinned `google/gemma-2-2b-it`;
- follow-up split: the previously unopened R9 `construction_validation`;
- 48 candidates, 12 per domain, and 144 prompts;
- qualification: at least two of three correct facts;
- existing structural and exhaustive semantic audits;
- existing checkpoint, execution-identity, and artifact-hash machinery.

The prior R9 gate SHA-256 is
`68eb8fb85ead2b805521e057b6aff36fbd80e227a13ad495f8a81ad6284e5b5d`.

## Minimal scoring correction

The answer parser may remove the modifiers `professional`, `former`, or
`former professional` only when the remaining text is already an exact
registered alias. No semantic model, fuzzy match, substring match, or
post-output alias addition is allowed.

Example:

```text
Professional footballer -> footballer
```

is accepted only when `footballer` was registered before model execution.
`Professional athlete` remains incorrect when `athlete` is not registered.

## Readiness gate

The primary readiness construct is the preregistered entity-level familiarity
screen: at least two correct answers out of three and at least eight qualified
entities per domain.

The complete domain-relation matrix remains reported. Each relation must
produce at least one exact success to catch broken prompts or parsers, but it
is no longer required to independently qualify eight entities. Relation is a
fixed analysis factor in the later study; it is not the familiarity label.

If the gate passes, two independent human raters must audit the registered
sample, with an independent adjudicator for disagreements, before any
protected endpoint is opened.

## Stop rule

The R10 follow-up is run once. After this amendment is committed, no candidate,
question, alias, parser rule, threshold, or model identity may change in
response to its outputs. A failed gate is reported as another instrument
failure. A passing gate permits instrument freeze and the existing
confirmatory pipeline; it does not confirm the research hypothesis.
