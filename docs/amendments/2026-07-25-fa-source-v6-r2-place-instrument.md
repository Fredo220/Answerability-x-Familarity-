# Source-v6 R2 Place-Instrument Amendment

**Date:** 2026-07-25
**Scope:** Open instrument development only
**Status:** Failed pre-model source audit; no R2 model output produced

## Reason for the revision

Source-v6 R1 completed the pinned Gemma screening on all 96
`instrument_development` entities. The overall qualification counts were:

| Domain | Qualified / 24 |
|---|---:|
| creative work | 21 |
| organization | 19 |
| person | 17 |
| place | 4 |

The Place shortfall was concentrated in the registered P131 administrative
region question (4/24) and P421 time-zone question (2/24). Item review found
that P131 often encoded a finer administrative level than a reasonable answer,
while P421 sometimes exposed historical or overly broad Wikidata statements.
These are instrument-validity failures, not evidence about the
Familiarity-by-Answerability hypothesis.

Re-scoring the complete R1 outputs with the conservative R2 answer parser
changed exactly one Place item and increased Place qualification from 4/24 to
5/24. Parser tolerance alone therefore cannot repair the construct-validity
problem.

R1 remains immutable and failed. R2 is a new open-development instrument
revision. No `construction_validation` result or H1-H8 endpoint was opened.

## R2 Place instrument

All non-Place domains, their relations, prompts, qualification threshold,
model, decoding configuration, split sizes, and split seed remain unchanged.
Place uses:

1. **P17 country**: current non-ended country statement.
2. **P131 administrative region**: current non-ended administrative-region
   statement, with wording that explicitly asks for the current region.
3. **P30 continent**: derived from the P17 country.

The source query retains only places with exactly one non-ended P17 statement,
exactly one non-ended P131 value typed as an administrative territorial entity,
and a country with exactly one P30 value. This makes each registered answer
single-valued before sampling. P421 is removed. No entity is selected from its
R1 model outcome; every R1 QID is excluded and R2 applies one deterministic
query and source-rank rule to the remaining open candidate frame.

The already retrieved R1 Wikidata label/alias cache may seed R2. Its SHA-256
is part of the R2 design and provenance; only QIDs absent from that immutable
cache are fetched again. This changes network work, not candidate order,
relations, aliases, or model outcomes.

P17 and P30 are correlated. R2 therefore remains only a familiarity-screening
instrument. The two-of-three entity qualification threshold is unchanged, and
relation-level yield is reported so that P131 cannot be hidden by the two
easier geographic relations.

## R2 answer parser

The development-only parser is versioned and hashed into the execution
identity. It may:

- remove outer whitespace;
- use content after a final `</think>` marker;
- use the final non-empty answer line;
- remove only `Answer:`, `Final:`, or `Final answer:` prefixes;
- remove one matching pair of outer quotation marks;
- accept `X (Y)` only when both X and Y are registered aliases.

It does not use fuzzy matching, substring matching, global accent folding,
semantic equivalence, administrative-suffix stripping, or model-assisted
grading. Raw and parsed completions are both retained.

## Open-development decision rule

R2 may advance to a separately frozen `construction_validation` run only if
all of the following hold on the 24 entities per domain in
`instrument_development`:

- at least 16 of 24 entities qualify in every domain;
- every registered relation has at least 16 correct answers out of 24;
- Place P17 and P30 each have at least 18 correct answers out of 24;
- the run has complete provenance, 96 unique entities, 288 unique questions,
  and no missing model output;
- a deterministic error packet is created for later independent human review.

The thresholds assess whether a future confirmatory corpus can be constructed
with a practical reserve. They are not hypothesis endpoints.

If any criterion fails, R2 is retained as another failed instrument revision.
The same output cannot be used to change R2 and then be called validation.

## Claim boundary

Passing R2 would establish only that the revised screening instrument has
adequate open-development yield. It would not establish that familiarity
causes answerability, hallucination, confidence, intuition, or any internal
mechanism. Those claims require a fresh Source-v7 confirmatory corpus,
independent ratings, protected behavioral analysis, and the registered
mechanistic follow-up.

## Pre-model audit outcome

Before Colab screening, a human-style source audit found blocking Place
construct errors: homonymous city labels, P131 values equal to P17, and P131
aliases equal to the queried entity label. R2 was therefore stopped before any
Gemma output. Source-v6 R3 addresses those source-only defects with
deterministic query and alias-overlap filters.
