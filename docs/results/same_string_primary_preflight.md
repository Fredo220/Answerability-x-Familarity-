# Same-String Primary Preflight

**Status:** `naturalness_passed_awaiting_runtime_smoke`
**Execution commit:** `151f818cb51b7782ad4b32c36b9cebfd06a7a499`  
**Protected behavior endpoint:** closed  
**Empirical model claim:** none

## Result

The deterministic construction produced the complete registered corpus and
passed every model-independent prompt audit.

| Split | Units | Prompt rows | Units per domain |
|---|---:|---:|---:|
| `mechanism_train` | 64 | 256 | 16 |
| `locked_validation` | 32 | 128 | 8 |
| `behavior_test` | 48 | 192 | 12 |
| `probe_test` | 24 | 96 | 6 |
| `intervention_test` | 24 | 96 | 6 |
| **Total** | **192** | **768** | - |

The passed checks were four-cell completeness, target-string identity, code
vocabulary, template isolation, entity isolation, rendered-token equality,
special-token equality, Same-String token budget, relation/code leakage, and
deterministic cell identity. The machine-readable report records every source,
selection, policy, packet, model, tokenizer, and config hash.

## Pre-outcome repair

The first local preflight at commit
`40f010964076d7f59e188f86c3d3ce71c7c2ad39` failed two checks because one
selected real title contained the task word `code`. No model output or
protected endpoint had been opened, and its human packets were not
distributed.

The registered amendment now applies one symmetric preselection rule to all
real and synthetic names: names containing prohibited task vocabulary are not
eligible, and the next deterministic reserve fills the unchanged domain
quota. No source record, sample size, threshold, estimand, or endpoint was
edited. The repeated preflight then passed all checks.

## Human gate

The independent human gate passed. Raters A and B completed all 192 blinded
pairs. Their registered binary decisions differed for 68 pairs, so the sealed
workflow issued only those disagreements to an independent rater C. The final
artifact contains 452 ratings: 384 initial ratings and 68 adjudications.

The compiler verified packet identities, blinded item mappings, unchanged
stimuli, rating ranges, distinct rater identities, independence attestations,
and the registered naturalness thresholds. The final ratings artifact has data
SHA-256 `78872ca0b1dce1def5ee841556b953ddda0b8d14b286676f2cf1ba2e51b208be`.

During finalization, a code defect was found in which a disagreement-only
issuance was compared against the full match-set hash. A regression test first
reproduced the defect; the fix validates the issuance against exactly the
registered disagreement subset. No rating, stimulus, threshold, exclusion, or
protected endpoint was changed.

The next gate is the unprotected runtime smoke. The confirmatory index,
protected Gemma evaluation, and mechanistic follow-up remain closed. Passing
the human gate is dataset-quality evidence, not evidence for the behavioral
hypothesis.

## Machine-readable record

See [`same_string_primary_preflight.json`](same_string_primary_preflight.json).
