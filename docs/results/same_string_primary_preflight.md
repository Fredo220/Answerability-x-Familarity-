# Same-String Primary Preflight

**Status:** `awaiting_independent_ratings`  
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

Two independent blinded packets are sealed under the ignored artifact root:

- `runs/same-string-primary-preflight-v2/rater-packets/public/rater-a-packet.json`
- `runs/same-string-primary-preflight-v2/rater-packets/public/rater-a-response.csv`
- `runs/same-string-primary-preflight-v2/rater-packets/public/rater-b-packet.json`
- `runs/same-string-primary-preflight-v2/rater-packets/public/rater-b-response.csv`

Each rater receives only their own packet and response template and follows
[`fa_naturalness_rating_protocol.md`](../fa_naturalness_rating_protocol.md).
The raters must work independently, must not use search or a language model,
and must attest independence in every row. Only disagreements may be sent to a
third independent adjudicator.

Until the two response files compile successfully, the runtime smoke,
confirmatory index, protected Gemma evaluation, and mechanistic follow-up
remain blocked. This preflight is dataset-quality evidence, not evidence for
the behavioral hypothesis.

## Machine-readable record

See [`same_string_primary_preflight.json`](same_string_primary_preflight.json).
