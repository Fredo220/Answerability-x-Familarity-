# Familiarity versus Answerability

## Purpose

This repository tests whether recognition of an entity changes a language
model's willingness to answer when relation-specific evidence is absent.
Familiarity and answerability are separate experimental factors.

## Vocabulary

- **Source-v5**: the frozen confirmatory corpus whose corrected Gemma screening
  failed its preregistered domain quota. Its status is permanently
  `not_evaluable`.
- **Source-v6**: an open development corpus used only to develop and validate
  the source-construction and familiarity-screening instrument.
- **Source-v7**: the future untouched confirmatory corpus. It may be constructed
  only after the Source-v6 instrument and success criteria are frozen.
- **instrument development**: the Source-v6 split whose results may be inspected
  while improving a declared instrument revision.
- **construction validation**: a disjoint Source-v6 split opened once for a
  frozen instrument revision. It cannot be recycled for tuning.
- **familiarity screen**: three unrelated factual-recall questions scored using
  exact normalized answers and registered aliases. Qualification remains at
  least two correct answers out of three.
- **protected endpoint**: any behavioral, probe, or intervention result used to
  test H1-H8. Source-v6 cannot open or select these endpoints.

## Claim Boundary

Source-v6 may support a claim that a domain-aware construction and screening
procedure was developed and validated. It cannot support a Familiarity effect,
an Answerability effect, or a mechanistic claim.

Green tests establish software behavior, not scientific findings.
