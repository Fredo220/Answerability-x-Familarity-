# Task 4 Report

## Evidence

- Red: `.venv/bin/python -m pytest tests/test_rlmf_format.py tests/test_rlmf_cli.py -q` failed during collection with `ModuleNotFoundError: trajectory_extractor.rlmf_format`, before the implementation existed.
- Green: `.venv/bin/python -m pytest tests/test_rlmf_format.py tests/test_rlmf_cli.py -q` completed with `21 passed in 3.75s` after implementing the strict parsers, alias judge, audit strata, CLI, and audit decisions.
- Full suite: `.venv/bin/python -m pytest -q` completed with `246 passed in 15.43s`.
- Hygiene: `git diff --check` completed without output.

## Commit

Code commit: `027687c` (`feat: add audited low-compute RLMF judge`).

## Assumptions

- Audit candidates are sealed JSONL records at `runs/rlmf/<study_id>/evaluation/audit_candidates_<phase>.jsonl`; each record supplies the private arm/proxy stratum and the rater-facing question/answer fields.
- Development candidates identify `split` as either `pre_sft` or `rl_train`. Locked and test candidate eligibility is supplied by their sealed evaluation bundles.
- The two registered judgment types are `correctness` and `equivalence`. Test strata are balanced as closely as possible for 1,250 and 1,750 rows, where exact equality across eight strata is arithmetically impossible.

## Residual Risks

- The test-audit bound is a deterministic, stratified proxy-label bias envelope. Task 5 must consume this uncertainty when calculating the endpoint-specific `delta_cMFG_star` adjustment; this task deliberately does not calculate behavioral metrics.
- Test generation itself must check the `locked_judge_audit` completion marker. This task writes that hard-gate marker only after a passing locked audit; the later rollout command is responsible for refusing generation when it is absent.
