# Task 4 Marker Fix Report

## Scope

Closed the remaining Task 4 P1 finding from
`task-4-final-rereview.md`. The change is limited to `cli.py` and
`test_rlmf_cli.py`; Task 6 and `progress.md` were not changed.

## Regression

Added `test_locked_record_rejects_changed_frozen_alias_endpoint_marker`.
The test completes development, builds locked, seals rater A, rater B, and
adjudication endpoints, then changes only `prepare-data.complete.json`'s
`created_at`. It asserts that `aliases.jsonl` remains byte-identical and that
the mutated prepare-data endpoint still verifies before locked recording
rejects the changed marker hash.

## Fix

- Locked and later audit recording recompute the current prepare-data marker
  hash and compare it with sample metadata.
- Locked recording also requires that hash to equal the development
  `proxy_freeze` alias marker hash.
- Sample and final audit endpoint parents bind `alias_endpoint_marker` in
  addition to the alias artifact hash.
- Existing full proxy-freeze equality remains enforced for locked recording.

## Evidence

### Red

Command:

`.venv/bin/python -m pytest tests/test_rlmf_cli.py::test_locked_record_rejects_changed_frozen_alias_endpoint_marker -q`

Result before the fix: `Failed: DID NOT RAISE ValueError`; the mutated marker
was accepted and locked recording returned `passed: true`.

### Green

The same regression after the fix: `1 passed in 3.42s`.

Focused Task 4 suite:

`.venv/bin/python -m pytest tests/test_rlmf_format.py tests/test_rlmf_cli.py tests/test_rlmf_artifacts.py -q`

Result: `56 passed in 16.06s`.

Full suite, run once after focused green:

`.venv/bin/python -m pytest -q -rs`

Result: `284 passed, 1 skipped in 27.52s`. The skip is the existing CUDA-only
test because CUDA is unavailable.

`git diff --check` passed before commit.

## Commit

`9819472f0ffb44badfea7194a501e110870ee5dd` - `fix: bind frozen alias endpoint marker`
