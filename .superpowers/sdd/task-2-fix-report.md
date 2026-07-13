# Task 2 Fix Report

## Files Changed

- `src/trajectory_extractor/rlmf_types.py`
- `src/trajectory_extractor/rlmf_artifacts.py`
- `tests/test_rlmf_types.py`
- `tests/test_rlmf_artifacts.py`

The smoke and confirmatory JSON files were reviewed and already matched the
now-explicit frozen contracts, so their bytes did not change.

## Contract Change

`RLMFArtifactStore.complete_endpoint(study_id, endpoint, config, paths)` now
requires a canonical `RLMFConfig` instead of a caller-provided `config_hash`.
The store writes or validates the canonical
`runs/rlmf/<study_id>/metadata/config.json` artifact, computes the config hash
itself, records the artifact path in the completion marker, and recomputes the
hash during verification.

## Red Evidence

- Added config-artifact binding, valid-length hash rewrite, frozen-field,
  direct-construction aliasing, namespace-symlink, and boolean-numeric
  regressions before production changes.
- `.venv/bin/python -m pytest tests/test_rlmf_types.py tests/test_rlmf_artifacts.py -q`
  initially reported `53 failed, 22 passed`.
- The added verification regression for a symlinked `metadata` section then
  failed independently with `1 failed` before its verification-path fix.

## Green Evidence

- `.venv/bin/python -m pytest tests/test_rlmf_types.py tests/test_rlmf_artifacts.py -q`
  passed: `76 passed in 2.44s`.
- `.venv/bin/python -m pytest -q` passed: `220 passed in 13.01s`.
- `git diff --check` passed before the fix commit.

## Commit

Fix commit: `6360bd2740f7b34f14302ea95b5b21f3e9afaa16`
(`fix: harden RLMF study records`)

## Remaining Risks

- The immutable marker and canonical-config checks detect accidental or
  post-completion content changes, but they do not provide cryptographic
  protection against an actor who can deliberately rewrite both files with the
  same local filesystem permissions.
- Namespace checks reject symlinked `runs`, `rlmf`, study, section, and
  verification metadata components. Protection against a hostile concurrent
  process swapping path components after validation would require
  descriptor-relative no-follow writes, which is outside this Task 2 contract.
