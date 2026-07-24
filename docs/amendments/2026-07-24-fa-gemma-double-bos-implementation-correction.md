# Gemma Double-BOS Implementation Correction

**Date:** 2026-07-24
**Scope:** Familiarity versus Answerability only
**Status:** Frozen after source-qualification failure and before any protected
F1/F2A endpoint was constructed or opened

## Trigger

The first real pinned-Gemma Source-v5 screening run at Git commit
`39832250653b431ee93e0b12c97089de91e9e554` stopped during
`mechanism_train` source qualification because the registered audited-pair
quota could not be filled. The stopped audit reported qualified counts of 31
creative works, 24 organizations, 19 people, and 4 places. This was a
source-qualification artifact only: no archive task, behavioral endpoint,
activation endpoint, intervention, or confirmatory report existed.

Review of the actual model input then found an implementation error. The
pinned Gemma chat template renders a prompt beginning with the BOS token. The
runtime passed that rendered text back to the tokenizer without disabling
automatic special-token insertion. The resulting input began with two BOS
tokens:

```text
rendered text:  <bos><start_of_turn>user...
default IDs:    [2, 2, 106, ...]
intended IDs:   [2, 106, ...]
```

The chat-template hash verified the rendered text but did not detect the
second BOS token introduced during the later tokenization call.

## Registered Correction

`HFModelRunner.generate` must tokenize already-rendered chat prompts with
`add_special_tokens=False`. A regression test must fail under the old call
and pass only when the tokenizer receives that explicit argument.

No other execution rule changes:

- Source-v5 entities, questions, pseudonyms, hashes, and split assignments are
  unchanged.
- The pinned Gemma model, tokenizer revision, and chat-template hash are
  unchanged.
- Greedy generation, `max_new_tokens`, the exact parser, accepted aliases, and
  the two-of-three qualification threshold are unchanged.
- Domain quotas, reserve counts, human-rating rules, hypotheses, and protected
  endpoints are unchanged.

The malformed run and its checksum-verified checkpoint remain preserved as a
failed implementation artifact. They are not reused as model evidence. The
corrected run uses a new Git commit identity in a fully reset Colab runtime.
Local screening artifacts are additionally bound to their Git commit and
configuration before any resume decision, so a reused runtime cannot silently
accept a completion from another execution.

## Stop Rule

If the corrected run still cannot fill every registered split-domain quota,
the confirmatory corpus is `not_evaluable`. No prompt, parser, alias, threshold,
quota, or candidate-order change may be made to rescue the result.

## Claim Boundary

This correction creates no evidence for familiarity, answerability, internal
features, or causal mechanisms. It only restores the exact token sequence
defined by the already pinned Gemma chat template before repeating the
source-qualification stage.
