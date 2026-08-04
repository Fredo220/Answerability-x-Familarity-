# Task 2 Report: Gemma Intervention Adapter

## Status

Implemented and software-verified the Task 2 runtime adapter for
`same-string-answerability-causal-pilot-v1`. No pinned model weights were
downloaded, no causal validation output was evaluated, and no causal-test
output or closed-study artifact was opened or modified.

## Added

- `src/trajectory_extractor/fa_answerability_causal_runtime.py`
  - exact replay of an already rendered causal prompt using
    `add_special_tokens=False`, with rendered-byte hash, token-ID,
    chat-template replay, offset, and structured-span checks;
  - temporary additive decoder-layer hook at one
    `[batch=0, position, hidden]` residual site;
  - tensor, tuple, list, and model-output container preservation;
  - write-time-only vector casting to the hidden tensor dtype and device;
  - immutable audit records binding prompt/prefix identity, layer, position,
    source and represented vector hashes, represented dtype/device, one hook
    call, one modified site, and completed hook cleanup;
  - one-shot teacher-forced sequence log-probability scoring for the correct
    archive code and `UNKNOWN`, with EOS appended to both candidates;
  - raw summed and length-normalized candidate scores and margins;
  - short deterministic generation using one hooked prefill followed by an
    unhooked manual greedy cache loop.
- `tests/test_fa_answerability_causal_runtime.py`
  - Gemma-shaped fake decoder modules with no model download;
  - tensor, tuple, list, and model-output return coverage;
  - invalid position, hidden shape, and vector dimension rejection;
  - write-time dtype casting and input-tensor immutability;
  - hook cleanup after success and decoder exceptions;
  - rendered hash and token mismatch rejection;
  - closed-form deterministic teacher-forced score checks;
  - zero-vector equivalence to the unhooked baseline;
  - deterministic generation and prefill-only hook-call auditing.

## TDD Evidence

The runtime test module was created before the production module. The first
focused run failed during collection with:

```text
ModuleNotFoundError: No module named 'trajectory_extractor.fa_answerability_causal_runtime'
```

The first implementation run produced eight passes and three failures because
the existing activation anchor resolver intentionally accepts only the older
`factorial` and `same_string` blocks, while Task 1 registers
`same_string_causal`. The runtime was narrowed to a causal-specific exact
resolver rather than changing or misrepresenting the older helper semantics.

The final focused command passed:

```text
.venv/bin/python -m pytest tests/test_fa_answerability_causal_runtime.py -q
11 passed in 1.18s
```

An adjacent pre-report run also passed 87 Task 1, runtime, activation, and
generation tests. A repository-wide run was explicitly stopped to keep Task 2
scope bounded; it had reached 595 passing tests without a reported failure.
Ruff is not installed in the project `.venv`, so no Ruff result is claimed.

## Evidence Boundary

These results establish `software_verified` runtime behavior against fake
Gemma-shaped modules only. They do not establish feasibility with the pinned
quantized model, an empirical intervention effect, or causal support. The
registered Colab smoke remains responsible for pinned Gemma/tokenizer loading,
real prompt execution, cache compatibility, and resource validation before any
protected execution.

## Concerns

- The pinned `google/gemma-2-2b-it` weights and tokenizer were not run locally;
  live-model compatibility remains gated on the required Colab smoke.
- Ruff could not be run because it is absent from the checked-in environment.
