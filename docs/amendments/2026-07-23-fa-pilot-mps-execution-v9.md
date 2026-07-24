# FA Pilot Uniform-MPS Execution Amendment V9

Date: 2026-07-23

## Scope

This amendment supersedes V8 only for development-pilot execution hardware and
generation shard identity. The frozen 320 prompts, model revision, tokenizer,
chat template, decoding parameters, parser, gates, and claim boundaries are
unchanged.

## Trigger

The V8 process completed three immutable checkpoints under Accelerate
CPU/disk-offload. Timing-only inspection showed 101 to 119 seconds per prompt,
which implied an avoidable multi-hour local run. No output text, answer code,
condition-level behavior, score, or scientific endpoint was inspected. The V8
artifacts remain immutable and are excluded from analysis.

A non-study prompt established that the same pinned model fits wholly on Apple
MPS and that warmed-up inference is materially faster. Because hardware can
change floating-point execution, V9 does not reuse the three V8 outputs.

## Frozen V9 Execution

- Generation shard ID: `pilot-generation-20260723-v9-mps-0001`
- Device: complete model on `mps:0`
- Prompt manifest:
  `runs/familiarity_answerability/smoke-qwen17b-v1/shards/pilot/prompt-capability-a9466bce5b3ef466.jsonl.manifest.json`
- Full manifest SHA-256:
  `4c044139b5ce661f0e95b3215cf21560c8bee7ab60fcdf90e0daf7ccd4fd5430`
- Prompt count: `320`
- Model: `Qwen/Qwen3-1.7B`
- Model and tokenizer revision:
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Decoding: greedy, `do_sample=false`, `temperature=0.0`,
  `max_new_tokens=16`
- Resume: immutable per-prompt checkpoints, as specified in V8

All 320 analytical outputs must come from this V9 shard. V8 is an execution
feasibility record only and cannot be pooled, substituted, or scored.

## Claim Boundary

V9 remains a development pilot. Its results can justify continuation or design
repair, but cannot establish a confirmatory Familiarity effect, Answerability
effect, interaction, or mechanism.
