# Task 4 Final P1 Fix Report

## Scope

Closed the three P1 findings in `task-4-rereview.md` without changing Task 6
files or `progress.md`:

1. population-weighted confusion uncertainty for proxy-stratified audits;
2. immutable development-to-locked ordering and proxy freeze;
3. separately sealed rater A, rater B, and adjudication endpoints.

The final `delta_cMFG_star` propagation and `<0.015` decision remain owned by
Task 5/10. Task 4 still publishes only size-specific test evidence and exits
nonzero while endpoint propagation is pending.

## Contract Additions

### Weighted sampling and uncertainty

- Every `AuditRow` now stores the immutable eligible population count for its
  registered sampling stratum.
- Every sample metadata artifact stores all registered strata with population
  count, cumulative sample count, and the exact reduced rational inclusion
  probability.
- Sensitivity and specificity use inverse-inclusion/post-stratification weights
  based on the full proxy stratum sample count. Raw balanced-audit ratios are not
  used for arm/judgment confusion metrics.
- `estimate_arm_confusion_uncertainty` emits schema version 2 with the complete
  sampling design, deterministic stratified-bootstrap method, 2,000 replicates,
  registered RNG seed, and weighted arm/judgment sensitivity and specificity
  intervals.
- The reviewer probe now yields sensitivity `0.5714285714` at 10% eligible
  proxy-positive prevalence and `0.9795918367` at 80%, with identical audited
  within-stratum outcomes.

### Development freeze and locked ordering

- Final development recording publishes and re-verifies the immutable
  `development_judge_audit` endpoint.
- Locked candidate endpoints must bind the development marker hash and have a
  completion-marker timestamp strictly after the development endpoint.
- Locked build and locked record both reverify that dependency and bind it in
  their own sample/final endpoint parents.
- The canonical proxy freeze contains parser version, parser source hash,
  normalization version, alias artifact hash, alias endpoint marker hash, and
  relevant `proxy_*` candidate-parent provenance. Locked build rejects any
  mismatch with the development freeze.
- Test-phase locked verification now rejects legacy locked endpoints that do not
  bind and postdate the development endpoint.

### Independently sealed human inputs

- `rlmf-seal-judge-rating` seals rater A or rater B separately against the sample
  marker, exact pending IDs, resolved source path, source hash, and identity.
- The two rater endpoints must have distinct identities, resolved paths, source
  hashes, endpoint markers, and marker hashes.
- `rlmf-seal-judge-adjudication` requires both verified rater endpoints, accepts
  exactly their disagreement IDs, and binds both marker hashes.
- Sequencing uses endpoint completion-marker timestamps: each rater postdates the
  sample marker, and adjudication postdates both rater markers. Mutable sample
  metadata timestamps and operator-supplied rating timestamps are not clocks.
- The preserved `rlmf-record-judge-audit` final command accepts only a schema-v2
  manifest of the three endpoint names and marker hashes. It reads only verified
  sealed copies and binds all three endpoint marker hashes in the final endpoint.
- CLI help exposes both explicit seal subcommands.

## Red Evidence

- Weighted-stratification cycle:
  - Command: `.venv/bin/python -m pytest tests/test_rlmf_format.py -q`
  - Result: `2 failed, 25 passed in 3.19s`.
  - Failures showed the missing schema-v2 design envelope and missing immutable
    stratum population count on `AuditRow`.
- Trust-chain and staged-source cycle:
  - Command: `.venv/bin/python -m pytest tests/test_rlmf_cli.py -q`
  - Result: `5 failed, 2 passed in 3.83s`.
  - Failures showed absent seal subcommands and locked build accepting candidates
    without `development_judge_audit`.

## Green Evidence

- Focused Task 4 suite:
  - Command: `.venv/bin/python -m pytest tests/test_rlmf_format.py tests/test_rlmf_cli.py tests/test_rlmf_artifacts.py -q`
  - Result: `55 passed in 16.38s`.
- Full suite, run once after focused green:
  - Command: `.venv/bin/python -m pytest -q -rs`
  - Result: `283 passed, 1 skipped in 27.08s`.
  - The single skip is the existing CUDA-only advantage test because CUDA is
    unavailable.
- Static and hygiene checks:
  - `.venv/bin/python -m compileall -q src tests` passed.
  - `git diff --check` passed.

## Commits

- `027687c feat: add audited low-compute RLMF judge`
- `4eace0d fix: seal RLMF judge audit workflow`
- `2128eff docs: record Task 4 trust-boundary fix evidence`
- `fix: complete RLMF audit trust chain` (the commit containing this report)

## Residual Limits

- Task 5/10 must verify the schema-v2 sampling design unchanged and propagate the
  weighted uncertainty through sealed behavioral records and `delta_cMFG_star`.
  Task 4 intentionally cannot publish the final gate or `test_judge_audit`.
- Population counts are exact relative to the verified eligible candidate
  endpoint; exhaustiveness of that upstream candidate population remains a
  producer responsibility.
- The deterministic stratified bootstrap is a design-preserving nonparametric
  interval. As usual, a stratum with no observed outcome variation can produce a
  degenerate bootstrap interval.
- Completion markers, source hashes, resolved paths, identities, and timestamps
  provide application-level tamper evidence and ordering, not signatures,
  authenticated human identity, or an independently attested clock.
- Deliberately rejecting byte-identical rater sources can reject a legitimate
  coincidentally identical file; this is the practical independence constraint
  required by the rereview.
