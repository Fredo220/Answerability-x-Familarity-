# FA Qwen Anchor-Adapter Amendment V12

Date: 2026-07-23

## Trigger

The registered V11 extraction failed before writing an activation artifact.
Qwen's tokenized `apply_chat_template` returned a Hugging Face
`BatchEncoding` mapping, while the anchor normalizer accepted only a bare token
sequence. No activation value was written or inspected.

## Repair

The normalizer now extracts `input_ids` from a mapping before applying the
existing nonempty-integer-sequence validation. A regression test covers the
mapping form, and a real tokenizer-only preflight verified the three anchor
indices without loading or running model weights.

All V11 scientific choices remain unchanged. The replacement activation shard
ID is `pilot-activations-v12-l0-9-18-27`. This amendment changes only adapter
compatibility; it does not change prompts, layers, anchors, labels, analysis,
or claims.
