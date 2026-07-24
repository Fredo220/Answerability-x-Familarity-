# FA Pilot Exact-Output Contract Amendment V10

Date: 2026-07-23

## Scope

This development-only amendment repairs the output instruction after the first
complete V9 feasibility run failed the registered pilot gate. It does not
change the scientific factors, entities, matched pairs, registry codes,
templates, decoding parameters, parser, thresholds, or gates.

## Observed V9 Failure

The immutable V9 gate was `blocked`:

- target-bound exact-code accuracy: `0.008928571428571428`;
- absent-condition answer-attempt rate: `1.0`;
- format validity exceeded the registered five-percent invalid ceiling in
  every factorial cell.

Development-output inspection showed a measurement-contract failure. Qwen
often stated the correct code or explained that no code was provided, but
wrapped that decision in prose or Markdown. The fixed 16-token limit then
sometimes truncated before the requested code or `UNKNOWN`. The exact parser
correctly marked those outputs invalid.

This result is not interpreted as evidence for or against Familiarity,
Answerability, their interaction, or an internal mechanism.

## Registered Repair

Every template now ends with the same text:

> Final answer: output only the five-character archive code or UNKNOWN. Do not
> explain or use Markdown.

The suffix is condition-invariant and appears after the registered target
introduction and query spans. The rebuilt 320-row manifest passed all twelve
construction audits, including token-length, lexical-multiset,
counterbalancing, special-token, and same-string checks.

## Frozen V10 Execution

- Prompt capability:
  `runs/familiarity_answerability/smoke-qwen17b-v1/shards/pilot/prompt-capability-30475c149ddb4403.jsonl.manifest.json`
- Full manifest SHA-256:
  `caa1adcdcdbcc8a3fa95006f3beaedea4c97c458b1dbca191c9084f2b9cc972e`
- Generation shard ID:
  `pilot-generation-20260723-v10-format-0001`
- Model/backend: pinned `Qwen/Qwen3-1.7B` on complete-model `mps:0`
- Decoding: unchanged greedy generation with `max_new_tokens=16`
- Resume: unchanged immutable per-prompt checkpoints

V9 remains an immutable failed pilot and cannot be pooled with V10. V10 uses
the original registered pilot gate without alteration. If V10 fails, there is
no further prompt optimization on these pairs; the failure is reported and
confirmatory construction remains blocked.

## Claim Boundary

V10 can establish feasibility only. Even a passed V10 gate cannot provide
confirmatory support for the study hypotheses.
