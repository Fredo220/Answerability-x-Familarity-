# Final Audit A Report

## Scope

Audited and implemented only the secondary endpoint artifact store, its CLI
integration, related tests, and this report. Primary scoring and primary artifact
code were not changed.

## Implemented

1. Secondary analysis provenance now hashes
   `docs/secondary_preregistration.md`. The CLI integration test asserts the exact
   path and SHA-256 digest.
2. Secondary JSON and NPZ writes are no-clobber publications using a same-directory
   temporary file, file `fsync`, atomic hard link, and directory `fsync`.
3. Completion markers bind the deterministic secondary-relative paths and SHA-256
   hashes of all six CLI endpoint artifacts: contrastive vectors, vector dynamics,
   predictions, metrics, method comparison figure, and risk-gap figure.
4. Completion creation accepts only the exact endpoint artifact set under the
   expected `<root>/<run_id>/secondary` namespace. The marker remains exclusive and
   is published after every bound artifact.
5. `verify_completion(run_id, endpoint)` reloads the marker, recomputes every bound
   artifact hash, and revalidates the metrics analysis ID against the canonical
   provenance fingerprint. Tests cover tampering with each of the six artifacts and
   provenance tampering paired with a rewritten marker hash.
6. Legacy metrics and completion markers still block reruns. The CLI test confirms
   that primary run files remain byte-for-byte unchanged.
7. `report-study` now defaults to `docs/generated_study_report.md` and rejects any
   output path resolving to the repository's `docs/results.md`.

## TDD Evidence

- Provenance/report RED: 5 intended failures for the old preregistration path, old
  report default, and missing protected-path rejection.
- Provenance/report GREEN: 5 passed.
- Endpoint RED: 10 intended failures for replacement writes and the missing
  completion manifest/verification API.
- Endpoint focused GREEN: 15 passed.
- Related regression suite: 44 passed.
- Full repository suite: 130 passed in 13.09 seconds.
- `git diff --check` reported no whitespace errors for the owned files.

## Files

- `src/trajectory_extractor/cli.py`
- `src/trajectory_extractor/secondary_artifacts.py`
- `tests/test_secondary_artifacts.py`
- `tests/test_secondary_cli.py`
- `.superpowers/sdd/final-audit-a-report.md`

Concurrent changes to README, docs, results, and another audit report were left
untouched and are not part of this implementation.
