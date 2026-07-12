# Response-Length Follow-Up B

Date: 2026-07-12

## Scope

Documentation-only post-hoc audit note. No source, test, or frozen run artifact
was modified.

## Findings

1. The frozen `exact_error` test set has 240 examples. Error rates by generated
   response-token length are 6/6 at length 2, 19/19 at length 3, 11/148 at
   length 4, and 3/67 at length 5.
2. Negative response length alone has AUROC 0.8392014287536678. Candidate-risk
   and static-baseline correlations with response length are -0.7160313591464588
   and -0.6467679666859326 respectively.
3. The selected token 4 aggregates every example through its own last available
   token. Shorter completed responses therefore do not share a uniform pre-output
   time point with longer responses.
4. This is an explicitly post-hoc confound analysis. It preserves the frozen
   `not_supported` result and adds required next-run controls: a length-only
   baseline, length-matched or stratified metrics with uncertainty, and a shared
   genuinely pre-output prefix.

## Verification

- The documented worktree-local read-only command, prefixed with
  `PYTHONPATH=src .venv/bin/python`, reproduced all recorded rates, AUROCs, and
  correlations from `runs/concept-main/secondary/comparisons/predictions_exact_error.npz`
  and the corresponding frozen example records.
- `git diff --check` completed successfully.
- The quoted SHA-256 manifest verification previously returned `OK` for all 12
  legacy secondary artifacts.

## Files Owned By This Follow-Up

- `README.md`
- `docs/results.md`
- `docs/execution_plan.md`
- `docs/legacy_secondary_sha256.txt`
- `.superpowers/sdd/final-archive-b-report.md`
- `.superpowers/sdd/final-response-length-b-report.md`
