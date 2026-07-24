# FA Pilot Generation and Resume Amendment V8

Date: 2026-07-23

## Scope

This amendment registers the first behavioral generation for the
Familiarity-vs-Answerability development pilot. It was written after the
320-row construction audit passed and before any output from this manifest was
generated or inspected.

## Frozen Inputs

- Run: `smoke-qwen17b-v1`
- Namespace: `pilot`
- Generation shard ID: `pilot-generation-20260723-v8-0001`
- Prompt capability:
  `runs/familiarity_answerability/smoke-qwen17b-v1/shards/pilot/prompt-capability-a9466bce5b3ef466.jsonl.manifest.json`
- Full manifest SHA-256:
  `4c044139b5ce661f0e95b3215cf21560c8bee7ab60fcdf90e0daf7ccd4fd5430`
- Prompt count: `320`
- Model: `Qwen/Qwen3-1.7B`
- Model and tokenizer revision:
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Chat-template SHA-256:
  `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`
- Tokenizer-pin SHA-256:
  `06f0c2bc68fd95467ddf94625ae4be420eca391e74c80640c0285b4556738457`
- Decoding: greedy, `do_sample=false`, `temperature=0.0`,
  `max_new_tokens=16`

## Interruption and Resume Policy

Generation runs with `--resume`. Each completed prompt is written immediately
as a one-row immutable `generation_checkpoint` artifact with the same exact
source, model, tokenizer, template, config, and decoding lineage as the final
shard. A resumed process may reuse only a checksum-valid completed checkpoint
whose request identity exactly matches the frozen prompt.

Infrastructure failures remain immutable and are never counted as completed
work. A retry receives a new sidecar rather than replacing prior evidence.
Only after all prompts are represented does the runner write the ordinary
`generation` aggregate in original manifest order. This changes durability,
not prompts, decoding, scoring, inclusion rules, thresholds, or scientific
claims.

## Gate

The generated outputs are development-only. They may be scored only with the
registered parser. Pilot continuation still requires the registered
construction checks and independent naturalness review. No pilot result can be
reported as confirmatory evidence.

## Claim Boundary

Passing software, provenance, and construction audits establishes execution
readiness only. It is not evidence for a Familiarity effect, an Answerability
effect, an interaction, or a mechanism.
