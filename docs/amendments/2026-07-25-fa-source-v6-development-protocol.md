# Source-v6 Development Protocol

**Date:** 2026-07-25
**Scope:** Familiarity versus Answerability only
**Status:** Open instrument-development protocol, revision 2

## Objective

Develop and independently validate a domain-aware procedure that can construct
an untouched future confirmatory corpus with adequate familiarity-screening
yield. This protocol does not test H1-H8.

## Scientific Lineage

- Source-v5 remains unchanged and permanently `not_evaluable`.
- The invalid double-BOS execution remains an implementation artifact.
- The corrected Source-v5 execution remains a valid feasibility failure.
- Source-v6 entities are development data and are permanently excluded from the
  future Source-v7 confirmatory frame.

## Splits

Source-v6 uses only:

| Split | Candidates per domain | Status |
|---|---:|---|
| `instrument_development` | 24 | open for instrument development |
| `construction_validation` | 24 | held until instrument freeze |

Across four domains this yields 192 entities and 576 registered screening
questions. Raw source pools may be domain-specific; the materialized development
splits remain balanced.

Revision 2 reduces only the open feasibility sample after revision 1 showed
that the frozen v3 pseudonym generator admits 56 globally unique complete place
reserves. Compatibility criteria and the future Source-v7 requirement are not
relaxed. The construction-validation result remains unopened.

## Frozen Runtime Invariants

- model: `google/gemma-2-2b-it`
- model/tokenizer revision:
  `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`
- rendered chat prompts use `add_special_tokens=False`
- deterministic decoding with `temperature=0`
- `max_new_tokens=16`
- exactly three screening questions per entity
- exact normalized answer or registered-alias matching
- entity qualification threshold: at least two correct answers out of three
- canonical ordering before global pseudonym collision handling
- immutable JSON artifacts with SHA-256 provenance

## Instrument Development

Only `instrument_development` may be used to compare declared alternatives for:

- Wikidata relations and unambiguous question wording;
- accepted aliases and answer granularity;
- deterministic source-rank frames;
- exclusion of ambiguous or erroneous ground truth;
- domain-specific raw-pool sizes.

Every change increments the instrument revision and preserves prior results.
Before opening `construction_validation`, freeze:

- source query and relation set;
- question templates, aliases, and normalization;
- qualification threshold;
- pseudonym and matchability policy;
- sampling order and validation sample size;
- validation success criteria.

## Leakage Rules

- Source-v5 QIDs are excluded from Source-v6.
- Source-v6 QIDs are excluded from Source-v7.
- Source-v6 uses no protected split names.
- Source-v6 cannot select layers, probes, interventions, templates, or H1-H8
  thresholds.
- Construction-validation results cannot be used to tune the same revision.
- Source-v7 is separately preregistered before screening or protected outcomes.

## Required Audits

The implementation must report:

- source integrity and exclusion lineage;
- deterministic split assignment and order invariance;
- exact domain, entity, and question counts;
- tokenizer and pseudonym matchability;
- entity scores `0/3`, `1/3`, `2/3`, and `3/3`;
- qualification yield by domain and relation with Wilson intervals;
- item-level success rates and all exclusion reasons;
- a deterministic manual error-audit packet.

The registered error taxonomy is:

- `entity_unknown`
- `relation_unknown`
- `ambiguous_ground_truth`
- `incomplete_alias_set`
- `wrong_granularity`
- `parser_failure`
- `model_format_failure`
- `source_error`
- `other`

## Human Instrument Audit

Two independent human raters assess a deterministic sample for question
clarity, ground-truth correctness, alias completeness, answer granularity,
domain plausibility, and real/pseudonym naturalness. A third independent rater
adjudicates only registered disagreements. These ratings validate the
instrument; they do not replace the future confirmatory naturalness gate.

## Validation Decision

Construction-validation criteria must be frozen before its results are opened.
A failed validation is reported as a failed instrument revision. Any retuning
requires a new revision and a new untouched validation split.

## Permitted Claim

> We developed and independently validated a domain-aware corpus-construction
> and familiarity-screening procedure for a future confirmatory
> Familiarity-by-Answerability study.

No Familiarity, Answerability, hallucination, intuition, or mechanism claim is
permitted from Source-v6 alone.
