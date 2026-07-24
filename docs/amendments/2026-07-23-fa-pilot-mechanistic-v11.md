# FA Exploratory Mechanistic Pilot Amendment V11

Date: 2026-07-23

## Scope

This amendment registers a small, explicitly exploratory activation study
after V10 passed the development feasibility gate. It cannot provide
confirmatory F2A evidence and cannot select Gemma layers, thresholds,
regularization, or claims.

## Question

Do four evenly spaced Qwen3-1.7B residual-stream snapshots contain
development-only evidence that target familiarity and prompt answerability are
separately decodable at their registered pre-output anchors?

## Frozen Extraction

- Prompt capability:
  `runs/familiarity_answerability/smoke-qwen17b-v1/shards/pilot/prompt-capability-30475c149ddb4403.jsonl.manifest.json`
- Full manifest SHA-256:
  `caa1adcdcdbcc8a3fa95006f3beaedea4c97c458b1dbca191c9084f2b9cc972e`
- Passed pilot gate:
  `runs/familiarity_answerability/smoke-qwen17b-v1/shards/pilot/pilot-gate-07ad00e2a8fd60be.jsonl.manifest.json`
- Activation shard ID: `pilot-activations-v11-l0-9-18-27`
- Layer IDs: `0, 9, 18, 27`
- Anchors: `target_intro_end`, `user_prompt_end`, and
  `assistant_prefix_end`
- Model/backend: pinned `Qwen/Qwen3-1.7B` on complete-model `mps:0`

The layer IDs were chosen before activation extraction as approximately even
depth coverage of the model's 28 transformer layers. They were not selected
from behavioral cell effects or activation outcomes.

## Analysis Boundary

- Familiarity is evaluated primarily at `target_intro_end`.
- Answerability is evaluated primarily at `user_prompt_end`.
- `assistant_prefix_end` is an output-proximal control only.
- Evaluation must keep entity units grouped across folds.
- Surface-only and shuffled-label baselines are mandatory.
- Every layer is reported; no best-layer result can be presented alone.
- Any accuracy or AUROC is exploratory and model-specific.

The pilot may test whether the machinery is capable of producing a useful
mechanistic work sample. Confirmatory F2A still requires the locked Gemma
design, independent naturalness evidence, untouched test endpoints, and the
registered full-layer analysis.
