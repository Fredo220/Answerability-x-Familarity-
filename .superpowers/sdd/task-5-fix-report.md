# Task 5 Review Fix Report

## Scope

Closed the Task 5 review findings without changing the registered cMFG
equal-mass/axis-width calculation, leave-one-out semantics, exact 20-auxiliary
target, tie-preserving sensitivity, or exact shared-level common-support
sensitivity.

## Corrections

- Task 4 schema-v2 now stores all 2,000 aligned joint bootstrap draws for every
  arm-by-judgment sensitivity and specificity value. Task 5 rejects artifacts
  without the complete joint field and contains no marginal interpolation path.
- `judge_bias_adjusted_delta` consumes each joint draw once with one shared
  paired prompt-cluster resample. It returns typed intervals for proxy
  `delta_cMFG_star`, adjusted `delta_cMFG_star`, and
  `absolute_differential_bias = abs(adjusted_delta - proxy_delta)`.
- Task 10 can apply the registered gate directly as
  `absolute_differential_bias.upper < 0.015`; the adjusted effect is not named or
  treated as a bias bound.
- `BehavioralEvaluationRecord` is frozen, deep-freezes provenance, binds one
  designated response to 20 distinct auxiliary IDs and labels, and explicitly
  represents malformed designated outputs.
- `calibration_metrics` retains every row identity for format reporting, uses
  only explicitly valid complete cases for calibration endpoints, and returns
  `not_evaluable/no_valid_complete_cases` when no complete case exists.
- Aggregate bootstrap calls require exactly `(11, 22, 33)`. Explicit one-seed
  calls are allowed for the three registered per-seed/descriptive analyses;
  two-seed and wrong-three-seed calls fail. Judge-bias propagation always
  requires all three.
- `UPSTREAM.json` now pins the exact upstream cMFG path and complete-file SHA-256.
  Frozen fixtures cover nested, duplicate, unmatched, empty, extra-text,
  out-of-range, and nonregistered-confidence format cases without importing the
  heavy upstream module during normal tests.

## TDD Evidence

Focused RED command:

```bash
.venv/bin/python -m pytest -q \
  tests/test_rlmf_format.py::test_task4_emits_confusion_uncertainty_but_defers_endpoint_bias_propagation \
  tests/test_rlmf_types.py::test_behavioral_evaluation_record_is_immutable_provenanced_and_allows_malformed_designated \
  tests/test_rlmf_types.py::test_behavioral_evaluation_record_requires_designated_plus_twenty_distinct_auxiliaries \
  tests/test_rlmf_metrics.py::test_calibration_metrics_emit_primary_and_sensitivities_without_dropping_rows \
  tests/test_rlmf_metrics.py::test_calibration_metrics_returns_machine_readable_not_evaluable_without_complete_cases \
  tests/test_rlmf_metrics.py::test_fixed_seed_bootstrap_resamples_paired_prompts_within_seed_and_is_reproducible \
  tests/test_rlmf_metrics.py::test_fixed_seed_bootstrap_preserves_duplicate_cluster_multiplicity \
  tests/test_rlmf_metrics.py::test_judge_bias_adjustment_consumes_whole_joint_draws_and_returns_registered_bias_quantity
```

Result: `7 failed, 1 passed in 5.28s`. Failures exposed the absent joint field,
typed record, malformed policy, seed validation, and structured joint-draw
inference. The already-correct multiplicity behavior passed.

Focused GREEN results:

- Targeted review tests: `8 passed in 8.69s`.
- Complete RLMF format/metrics/types files: `100 passed in 6.89s`.

Full Python 3.12 suite:

```bash
.venv/bin/python -m pytest -q -rs
```

Result: `299 passed, 1 skipped in 28.68s`. The existing skip is the CUDA-only
advantage test because CUDA is unavailable.

Static checks:

- `.venv/bin/python -m compileall -q src tests`: passed.
- `python3 -m json.tool third_party/rlmf/UPSTREAM.json`: passed.
- `git diff --check`: passed before the full suite.

## Residual Scientific Limitation

Inference is conditional on the three registered seeds and does not support
generalization to unseen training seeds. Confusion adjustment also relies on the
audited proxy model and is identifiable only when sensitivity plus specificity
exceeds one; Task 5 fails closed otherwise.
