# Task 5 Second Re-review Fix Report

## Scope

Closed the second Task 5 re-review findings without changing cMFG*, registered
seeds, joint-draw propagation, leave-one-out confidence, or the existing
shared-arm reward configuration.

## Corrections

- Task 4 canonicalizes audit bootstrap inputs by stratum and stable audit/source
  IDs before its first registered RNG draw. The uncertainty object is identical
  for every permutation of the same audit rows.
- `BehavioralEvaluationRecord` now requires the exact five-field SHA-256
  provenance schema for designated and auxiliary bundles, alias and judge
  evidence, and config. Its raw designated text must reparse exactly to the
  retained `ParsedRLMFOutput`.
- `judge_bias_adjusted_delta` accepts only typed behavioral records. It validates
  all retained raw pairs first, requires both arms in every registered seed,
  excludes only pairs with a malformed or incomplete arm, and returns structured
  pair counts/IDs plus `not_evaluable` results when necessary.
- Percentile intervals now retain the exact 2.5th and 97.5th quantiles even when
  the point estimate falls outside them. Interval schema validation now requires
  only finite ordered bounds.
- The local exact-two-tag parser/reward is explicitly documented as a deliberate
  shared-arm safety/reproducibility divergence, not upstream parity. Upstream
  formatter/extractor symbols, paths, and complete-file hashes are pinned in
  `UPSTREAM.json`; soft-format fixtures remain upstream-derived and frozen.

## TDD Evidence

RED command:

```bash
.venv/bin/python -m pytest -q \
  tests/test_rlmf_format.py::test_task4_confusion_bootstrap_is_invariant_to_audit_row_permutations \
  tests/test_rlmf_format.py::test_task4_bootstrap_interval_keeps_exact_percentiles_when_estimate_is_outside \
  tests/test_rlmf_types.py::test_behavioral_evaluation_record_requires_exact_hash_provenance_and_raw_parse_binding \
  tests/test_rlmf_metrics.py::test_judge_bias_adjustment_consumes_whole_joint_draws_and_returns_registered_bias_quantity \
  tests/test_rlmf_metrics.py::test_judge_bias_adjustment_excludes_only_malformed_complete_pairs_and_reports_not_evaluable \
  tests/test_rlmf_metrics.py::test_task5_percentile_interval_does_not_widen_to_include_point_estimate
```

Result: `5 failed, 1 passed in 8.83s`.

Focused GREEN command:

```bash
.venv/bin/python -m pytest -q tests/test_rlmf_format.py tests/test_rlmf_types.py tests/test_rlmf_metrics.py
```

Result: `105 passed in 12.21s`.

Full suite:

```bash
.venv/bin/python -m pytest -q -rs
```

Result: `304 passed, 1 skipped`. The existing skip is the CUDA-device advantage
test because CUDA is unavailable. Collection confirmed `305 tests`.

Static checks passed: `git diff --check`, `.venv/bin/python -m compileall -q src
tests`, and `.venv/bin/python -m json.tool third_party/rlmf/UPSTREAM.json`.

## Remaining Limitations

Inference remains conditional on fixed seeds `11`, `22`, and `33`; it does not
generalize to unseen training seeds. Judge-bias adjustment remains conditional on
the audited proxy model and requires identifiable confusion parameters.
