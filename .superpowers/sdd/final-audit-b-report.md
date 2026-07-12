# Final Audit B

Date: 2026-07-12

## Scope

Documentation-only audit of the final concept-replication workflow and frozen
result boundary. No source code under `src/` and no `runs/` artifacts were
modified by this audit.

## Findings

1. Fresh concept commands now isolate all derived namespaces. They define
   `RUN_ID=concept-replication-01`,
   `TRANSFER_RUN_ID="${RUN_ID}-real-transfer"`, and
   `INTERVENTION_RUN_ID="${RUN_ID}-intervention"`. Both transfer commands use
   `TRANSFER_RUN_ID`; concept intervention uses `INTERVENTION_RUN_ID` and its
   explicit baseline ID. The examples do not rely on `real-transfer` or
   `concept-intervention` defaults.
2. `docs/results.md` and `docs/execution_plan.md` now identify the live frozen
   secondary artifacts as historical artifacts created by
   `04568b9f1c1629ac7f08323b1c0602843fe91f48`, before current provenance and
   completion sealing. They contain no `analysis_id`, no
   `analysis_provenance`, and no completion marker. They are not retrofitted.
   The current legacy metrics guard, tracked artifact hash, and tracked frozen
   result report protect the existing record without representing it as a modern
   provenance-sealed analysis.
3. `report-study` is no longer documented as writing to `docs/results.md`.
   The current CLI default is the safe generated path
   `docs/generated_study_report.md`, and the command rejects the repository's
   immutable `docs/results.md` path. The README invokes the default directly.
4. The negative result and timing limitation remain intact: both registered
   `exact_error` comparisons are `not_supported`, and the selected token
   4/layer 16 prefix is the late, latest pre-token prefix rather than evidence
   of early warning.

## Verification

- `git diff --check` completed successfully.
- `.venv/bin/python -m pytest -q` completed successfully after the audit commit:
  `129 passed in 13.13s`.

## Files Owned By This Audit

- `README.md`
- `docs/results.md`
- `docs/execution_plan.md`
- `.superpowers/sdd/final-audit-b-report.md`
