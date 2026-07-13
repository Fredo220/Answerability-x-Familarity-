# Task 4 Scientific Trust-Boundary Fix Report

## Commits

- `4eace0d` - `fix: seal RLMF judge audit workflow`
  - Includes code, regression tests, the minimal artifact-store provenance extension,
    and the Task 4/5/10 plan-contract clarification.

## Red Evidence

- Initial focused regression run:
  - Command: `.venv/bin/python -m pytest tests/test_rlmf_format.py tests/test_rlmf_cli.py tests/test_rlmf_artifacts.py -q`
  - Result: collection failed because `estimate_arm_confusion_uncertainty` did not
    exist.
- CLI and artifact regressions before implementation:
  - Command: `.venv/bin/python -m pytest tests/test_rlmf_cli.py tests/test_rlmf_artifacts.py -q`
  - Result: `5 failed, 18 passed`.
  - Demonstrated that unsealed candidates were accepted, endpoint markers lacked
    timestamps, and additional sealed parent hashes were unsupported.
- Extension authorization regression:
  - Command: `.venv/bin/python -m pytest tests/test_rlmf_cli.py::test_cli_seals_independent_rating_sources_and_appends_test_extensions -q`
  - Result: failed because a 1,250-row build did not require a sealed Task 5/10
    extension request.
- Public decision regression:
  - The same focused CLI test failed because the test-audit command still emitted
    `passed: true` before endpoint-specific propagation.

## Green Evidence

- Final focused command:
  - `.venv/bin/python -m pytest tests/test_rlmf_format.py tests/test_rlmf_cli.py tests/test_rlmf_artifacts.py -q`
  - Result: `49 passed in 4.57s`.
- Final full-suite command:
  - `.venv/bin/python -m pytest -q`
  - Result: `277 passed, 1 skipped in 15.27s`.
- Static and hygiene checks:
  - `.venv/bin/python -m compileall -q src tests` passed.
  - `git diff --check` and `git diff --cached --check` produced no errors.

## Finding Closure

1. Task 4 now emits arm-specific sensitivity/specificity Wilson intervals only.
   `bound_differential_judge_bias` fails closed and directs endpoint propagation to
   Task 5/10. The CLI emits neither `bounded` nor a confirmatory `passed` result.
2. Test audits use size-specific artifacts. Each extension requires the previous
   verified evidence endpoint and a sealed Task 5/10 extension request, preserves
   prior rows and judgments, appends exactly 250 rows, and reuses byte-identical
   partial artifacts on retry.
3. Every build verifies its candidate endpoint and frozen alias endpoint. Phase
   eligibility is enforced as development `pre_sft`/`rl_train`, locked
   `validation`, and test `test`; test candidates bind the passing locked marker.
4. Locked and test audits both apply kappa, ambiguity, sensitivity, and specificity
   gates. Failure returns nonzero and publishes no pass endpoint. Test evidence
   returns nonzero while endpoint propagation is pending, and Task 4 never writes
   the final `test_judge_audit` endpoint.
5. Proxy labels are always recomputed from the frozen alias/equivalence logic.
   Contradictory supplied labels are rejected. Aliases remain in the private ledger
   and are absent from rater payloads.
6. Immutable metadata binds parser version/source hash, normalization version,
   alias hash, registered seed, exact candidate endpoint/marker hash, phase, size,
   parent sample hash, and extension-request marker. Completion markers bind that
   metadata and its scientific parents. The CLI seed is fixed at `20260713`.
7. Development strata are `split x judgment_type x proxy_label` over shared
   pre-treatment material. Locked and test retain
   `arm x judgment_type x proxy_label` strata.
8. Missing, null, blank, and duplicate source IDs are rejected. Audit IDs derive
   from phase plus stable source ID, and hash ranking makes selection independent
   of input order.
9. Alias containers must be non-string sequences of strings. Normalization uses
   NFKC and casefold, removes punctuation but preserves symbols, and removes
   English articles in every token position.
10. The record command now consumes a manifest for separately hashed rater-A,
    rater-B, and adjudication JSONL files. Identities are distinct, timestamps are
    timezone-aware, rating IDs exactly match pending rows, adjudication contains
    exactly disagreements and follows both ratings, and all sources are copied,
    sealed, persisted, and bound to the endpoint.

## Contract Changes

- Task 4 owns parser/normalization freeze, sealed sampling, independent human-source
  provenance, reliability gates, and arm-specific confusion uncertainty.
- Task 5/10 owns propagation through sealed behavioral records and
  `delta_cMFG_star`, the `<0.015` decision, extension requests, the 2,000-label
  `not_evaluable` decision, and publication of the final test-audit endpoint.
- A 1,250/1,500/1,750/2,000 build requires a verified size-specific extension
  request from Task 5/10 bound to the preceding evidence endpoint.
- Development uses shared pre-treatment split strata rather than invented arms.
- Independent ratings and adjudication are three sealed inputs referenced by one
  manifest; the former combined manual-row schema is no longer accepted.

## Remaining Risks

- Task 5/10 endpoint-specific confusion propagation and final audit publication are
  intentionally not implemented in Task 4. Test evidence therefore exits with
  `endpoint_propagation_required` until that work exists.
- Future candidate and rollout producers must publish the verified endpoint names
  and parent hashes consumed here; no production candidate artifacts were migrated
  or generated in this task.
- Source hashes and manifests provide tamper evidence, not cryptographic identity
  signatures. Human identities and timestamps remain operator attestations.
- The legacy `bound_differential_judge_bias` symbol remains as a fail-closed
  compatibility surface; callers must migrate to Task 5/10 endpoint propagation.
