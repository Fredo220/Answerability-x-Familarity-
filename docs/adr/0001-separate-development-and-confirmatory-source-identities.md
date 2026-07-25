# ADR-0001: Separate Development and Confirmatory Source Identities

**Status:** Accepted
**Date:** 2026-07-25

## Context

The corrected pinned-Gemma Source-v5 screening qualified 31 creative works, 25
organizations, 19 people, and 4 places. The frozen design required 20 per
domain. Source-v5 therefore stopped before human ratings or protected H1-H8
endpoints and is permanently `not_evaluable`.

Those results revealed a corpus-feasibility problem, especially for places.
They may inform a new development study but cannot be used to repair Source-v5.

## Decision

Use three distinct scientific identities:

1. Source-v5 remains an immutable failed confirmatory feasibility attempt.
2. Source-v6 is open instrument development with disjoint
   `instrument_development` and `construction_validation` splits.
3. Source-v7 will be a separately preregistered, untouched confirmatory corpus.

Source-v6 has its own module, namespace, IDs, manifests, hashes, and reports. It
does not call the Source-v5 Colab orchestrator and does not use protected split
names. All Source-v5 and Source-v6 QIDs are exclusions for Source-v7.

## Consequences

- More than 90 percent of the existing source, tokenizer, screening, artifact,
  and analysis infrastructure remains reusable.
- Instrument changes are allowed only on `instrument_development` and create a
  new revision.
- `construction_validation` is opened once after the instrument and success
  criteria are frozen.
- A failed construction validation is reported. It is not silently retuned.
- Source-v6 results cannot test or rescue H1-H8.
