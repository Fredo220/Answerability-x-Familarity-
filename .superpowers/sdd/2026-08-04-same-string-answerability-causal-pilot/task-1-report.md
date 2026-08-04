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

## Fix Round 1

An independent review identified four P1 failures in the initial Task 1
contracts. The public contracts were tightened as follows:

1. `select_causal_intervention` now requires the exact registered 15-candidate
   grid: five layers times three multipliers. A duplicate, omitted, or added
   layer/multiplier pair fails before selection.
2. `CausalExpectedProvenance` binds the sealed causal-corpus hash, direction
   bundle hash, per-layer direction hashes, model hash, and tokenizer hash.
   Every candidate must match all of them; the candidate direction hash must
   specifically match the expected hash for its layer.
3. The public direction-fitting path now accepts only
   `VerifiedV3TrainingInputs`, produced by
   `load_v3_training_direction_inputs`. The loader verifies the v3 prompt
   manifest and prompt-file hash, the activation sidecar/index/NPZ hashes,
   complete `representation_train` identity matching, model and tokenizer
   provenance, chat-template hash, and prompt-token/hash identity. The raw
   array fitter is private and the resulting `DirectionBundle` serializes this
   verified source provenance.
4. Causal corpus construction, auditing, and verification require the v3
   exclusions. The audit now rejects missing v3 test exclusions and checks
   target/distractor names and archive codes are disjoint across every causal
   split. It requires all 20 entity-test and 20 template-test v3 units, with
   all four prompts per unit, before accepting the exclusion inventory.

### Fix Verification

Regression tests were added before implementation and initially failed during
collection because `CausalExpectedProvenance` did not exist. After the fixes:

```text
5 passed in 1.23s
14 passed in 10.30s
```

The exclusion-completeness regression subsequently failed on a deliberately
removed v3 entity-test prompt and passed after the complete-inventory gate was
added (`5 passed in 1.23s`).

The 14-test command covered the causal contracts plus the existing v3 corpus
and representation suites. The source-provenance test loads the released v3
training activation manifest and verifies its typed records; no model forward
pass, causal validation result, or causal-test output was run or opened.

## Fix Round 2

Two further P1 review findings required public-boundary changes:

1. `fit_train_only_directions` now discards the supplied loaded records for
   fitting purposes and immediately calls `load_v3_training_direction_inputs`
   again using the bound v3 prompt-manifest path, activation-manifest path,
   and expected model/tokenizer pins. This re-verifies the v3 prompt manifest
   and prompt-file bytes, activation manifest/index/NPZ bytes, prompt-token
   identities, and model/tokenizer provenance immediately before extracting
   the direction. Forged fields on a directly constructed
   `VerifiedV3TrainingInputs` object cannot enter the resulting bundle.
2. The public causal corpus construction, audit, and verification APIs now
   require `v3_manifest_path`, not a sequence of v3 prompt objects. The
   manifest loader verifies the safe `prompts_file` binding, `prompts_sha256`,
   canonical manifest self-hash, study ID, row and unit counts, split counts,
   and typed prompt records before deriving v3 exclusion identities.

The new regressions initially failed because construction and audit did not
accept a bound manifest path and fitting preserved a forged activation-index
hash. After the fixes, the focused causal suite passed (`5 passed in 2.59s`).

## Fix Round 2 Interface

Added `LabelShuffledDirectionArtifact` plus builder and verifier APIs for the
Task 3 label-shuffled control. The artifact records the exact ordered
unit-level permutation, selected layer, recomputed normalized vector, complete
v3 training provenance, and an artifact hash. Verification reloads the bound
v3 prompt and activation artifacts through `load_v3_training_direction_inputs`
and recomputes the vector before accepting it. This is infrastructure only: no
causal validation or test output was opened.
