# Task 6 Fix Report

## Evidence

- Red: `.venv/bin/python -m pytest tests/test_rlmf_advantage.py -q` failed with `10 failed, 11 passed, 1 skipped in 3.36s`. The new finite-overflow cases failed with `DID NOT RAISE`, confirming the regressions exercised the missing checks.
- Green: `.venv/bin/python -m pytest tests/test_rlmf_advantage.py -q` completed with `21 passed, 1 skipped in 3.43s`. The skipped case is the CUDA device-preservation test on a host without CUDA.
- Full suite: `.venv/bin/python -m pytest -q` completed with `274 passed, 3 failed, 1 skipped in 15.32s`. The three failures are pre-existing concurrent Task 4 CLI failures in `tests/test_rlmf_cli.py`, all caused by `NameError: _build_rlmf_judge_audit` in the concurrently modified `src/trajectory_extractor/cli.py`.
- Hygiene: `git diff --check` and `git diff --cached --check` completed without output for the Task 6 changes.

## Change

- Standard GRPO and RLMF now reject non-finite reductions, centering, scaling, branch results, and final advantages with `ValueError`.
- RLMF now rejects finite Python `k` values that are not representable in the reward dtype.
- Grouped computation inherits the same fail-closed behavior through both arm helpers.
- Ordinary parity behavior, detached outputs, and dtype/device preservation remain covered by the existing tests.

## Commit

Code commit: `da3da84` (`fix: reject nonfinite RLMF advantages`).
