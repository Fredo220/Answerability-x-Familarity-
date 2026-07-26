# Source-v6 R9.1 Alias-Completeness Amendment

Date: 2026-07-26

## Trigger

The first pre-model R9 semantic audit inspected all 288 registered questions
and blocked model execution. It found no entity-identity, source-value,
relation, temporal, granularity, prompt, or collision error. It found 19
selected questions whose accepted aliases still omitted an ordinary surface:
18 omitted `US`, and one omitted `New York`.

The blocked corpus and audit are retained under
`data/fa/development_source_v6_r9_blocked_attempt1/`.

## Model-blind repair

No model output has been generated or inspected for R9. The correction manifest
is completed with one deterministic rule applied to all 118 frozen R8
`ordinary_surface_missing` rows, not only the 19 selected failures:

> Add every ordinary surface explicitly named in the frozen R8 blocker
> evidence when it is absent from that correction row.

This adds 48 `US` surfaces and three `New York` surfaces. It does not change
candidate eligibility, the SHA selection seed, the selected QIDs, split
assignment, prompts, factual values, success thresholds, or claim scope.

## Required rerun

R9 must be materialized again from a clean commit. The structural auditor must
replay all decisions, and a fresh exhaustive semantic audit must pass with zero
blockers before Gemma execution. The prior blocked attempt remains evidence and
must not be overwritten.

R9 remains development-only and can establish instrument readiness only.
