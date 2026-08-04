# Task 1 Report: Corpus, Direction, and Selection Contracts

## Status

Implemented the software-only contracts for
`same-string-answerability-causal-pilot-v1`. No model was loaded, no hook was
implemented, no causal validation output was evaluated, and no causal-test
output was opened.

## Added

- `src/trajectory_extractor/fa_answerability_causal.py`
  - immutable causal prompt, corpus, audit, direction, validation-candidate,
    and selection types;
  - deterministic 48-unit / 192-prompt fresh 2x2 corpus;
  - 12 `causal_validation`, 18 `causal_entity_test`, and 18
    `causal_template_test` units;
  - tokenizer replay plus pairwise token-multiset, unit-constant, output
    contract, split, template, and v3 test-identity audits;
  - canonical JSON, SHA-256-bound corpus, direction-bundle, and selection
    manifest serialization;
  - direction extraction restricted to v3 `representation_train` rows at
    `user_prompt_end`, with registered layers `0, 6, 12, 18, 25` and median
    positive paired-projection scales;
  - validation-only candidate selection with registered multipliers
    `0.25`, `0.5`, and `1.0`, complete-unit checks, provenance-hash matching,
    the 5% invalid-output and bound-accuracy-drop gates, and the required
    multiplier-then-layer tie break.
- `tests/test_fa_answerability_causal.py`
  - deterministic corpus construction and serialization;
  - tampered answerability label, incomplete 2x2 unit, tokenizer mismatch,
    and reused-v3-identity audit coverage;
  - train-only direction fitting, v3 test-split rejection, and incomplete
    v3 training-unit rejection;
  - deterministic validation selection and incomplete-validation-unit
    rejection.
- `configs/familiarity_answerability_causal_pilot_v1.json`
  - pins the Gemma model/tokenizer revision and chat-template hash already
    used by the v3 study;
  - records all Task 1 corpus, direction, validation, generation, and future
    statistical constants.

## TDD Evidence

The test module was created before the production module. The first focused
run failed during collection with:

```text
ModuleNotFoundError: No module named 'trajectory_extractor.fa_answerability_causal'
```

After implementation, the focused suite passed:

```text
5 passed in 0.66s
```

The focused suite plus the existing v3 corpus and representation suites then
passed:

```text
14 passed in 9.81s
```

`ruff` is not installed in the project `.venv`, so no Ruff result is claimed.
A broad-suite attempt did not return a terminal result in this execution
environment and is likewise not claimed as verification.

## Protected-Evidence Boundary

The new corpus uses `CausaNNNN` names, `ZNNNN` codes, and causal unit IDs.
It treats v3 `entity_test` and `template_test` names, codes, IDs, and prompt
IDs as exclusion identities. Direction fitting accepts only the v3 training
split and requires its activation mapping to match that split exactly.

Task 2 remains responsible for the live prefill hook, scoring, generation,
and intervention audit. Task 3 remains responsible for controls, statistics,
and any causal-support decision. This task establishes no empirical or causal
result.
