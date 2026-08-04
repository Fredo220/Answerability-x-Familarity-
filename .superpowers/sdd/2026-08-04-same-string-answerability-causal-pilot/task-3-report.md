# Task 3 Report: Controls and Sealed Analysis

## Status

Implemented and software-verified the Task 3 analysis layer for
`same-string-answerability-causal-pilot-v1`. The implementation operates only
on typed, caller-supplied score and audit rows. No model was loaded, no causal
validation result was evaluated, and no causal-test or closed-study outcome
artifact was opened, created, or modified.

## Added

- `src/trajectory_extractor/fa_answerability_causal_analysis.py`
  - immutable typed records for baseline, primary, and control margin scores;
    response classes; manipulation checks; preservation results; and execution
    identity hashes;
  - a hash-bound pre-outcome seal binding the selected direction, corpus,
    selection, model/tokenizer, runtime, output contract, layer, multiplier,
    anchor, and five random-control seeds;
  - deterministic schedules for no intervention, sign reversal,
    unit-label-shuffled direction, five-member orthogonal random family,
    wrong anchor, and deterministic farthest registered layer;
  - exact complete-unit effects averaged over exposure, fixed 10,000-draw
    unit bootstrap intervals, fixed 9,999-draw one-sided sign-flip values,
    paired strongest-control contrasts, and length-normalized sensitivity
    output;
  - fail-closed `not_evaluable` decisions for malformed, incomplete, pooled,
    split-mismatched, or identity-mismatched inputs;
  - support decisions limited to `causally_supported`, `not_supported`, or
    `not_evaluable`, with machine-readable reasons for every failed rule;
  - an unpooled two-split wrapper that permits `causally_supported` only when
    both required causal test splits independently pass;
  - a small atomic receipt store that allows only same-seal partial resumes and
    refuses completed endpoints or mismatched-seal resumes.
- `tests/test_fa_answerability_causal_analysis.py`
  - synthetic-only positive, null, wrong-direction, split, factorial,
    schedule, sign, hash, output-format, preservation, copying, random-family,
    control-tie, and endpoint-resume coverage.

## TDD Evidence

The analysis test module was written before the implementation. Its first
focused run failed at collection with:

```text
ModuleNotFoundError: No module named
'trajectory_extractor.fa_answerability_causal_analysis'
```

The final signed-schedule regression was also added before its implementation;
it first failed because `PrimaryScore` did not yet accept a `sign` field.

Final verification passed:

```text
.venv/bin/python -m pytest \
  tests/test_fa_answerability_causal_analysis.py \
  tests/test_fa_answerability_causal.py \
  tests/test_fa_answerability_causal_runtime.py -q
35 passed in 3.83s

.venv/bin/python -m py_compile \
  src/trajectory_extractor/fa_answerability_causal_analysis.py
```

`git diff --check` also passed. Ruff was not available in the project virtual
environment, so no Ruff result is claimed.

## Evidence Boundary

The passing tests establish `software_verified` behavior against synthetic
in-memory margins and hashes. They do not establish live Gemma compatibility,
runtime feasibility, an intervention effect, or causal support. A future
runtime integration must seal the selected validation artifact before causal
test rows exist and provide the complete typed schedule to this module.

## Concerns

- The Task 2 Gemma adapter has not yet been connected to this typed analysis
  input. That integration remains subject to the registered live smoke and
  protected-endpoint procedure.
- No lint result is available because `.venv/bin/ruff` is absent.
