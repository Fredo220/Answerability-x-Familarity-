# Causal Replication v2: Post-run Audit

**Study:** `same-string-answerability-causal-replication-v2`
**Model:** `google/gemma-2-2b-it` at revision
`299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`
**Frozen decision:** `not_supported`

## What Was Audited

The downloaded Colab archive was inspected without changing its scientific
artifacts or reopening either protected endpoint.

- ZIP SHA-256:
  `c2e711527d64a3a9d7aa4fcfde14a55b0638e1cb88a311fc16920e9e3dacaff0`
- 432 expected JSON receipts and 432 receipt locks were present.
- Every receipt lock and embedded receipt hash verified.
- All receipts had status `completed`; no scheduled shard was missing.
- The corpus, direction bundle, validation selection, evaluation seal, model,
  tokenizer, runtime, request, and implementation identities agreed across the
  evidence tree.
- The corrected label-shuffled direction was rebuilt from the published v3
  training activations and matched the sealed vector exactly. Its cosine with
  the primary direction was `0.02033`, with L2 distance `1.39976`; it was not
  the degenerate control found in v1.
- A fresh statistical replay returned the same split decisions and numerical
  result as the exported endpoint.

The Colab seal stores its original absolute `/content/...` source paths. For
the local replay, those paths were mapped to the byte-identical published v3
files. The bound hashes and rebuilt shuffled vector were verified before the
statistical replay. No released artifact was rewritten for this mapping.

## Frozen Result

| Split | Primary effect | 95% CI | Strongest control | Primary minus control, 95% CI | Decision |
|---|---:|---:|---:|---:|---|
| Unseen entities | 0.1997 | [0.1927, 0.2067] | Wrong layer: 0.3448 | -0.1450 [-0.2563, -0.0327] | `not_supported` |
| Unseen templates | 0.1705 | [0.1608, 0.1804] | Wrong anchor: 0.0027 | 0.1678 [0.1569, 0.1787] | `causally_supported` |

Both primary effects had fixed sign-flip `p = 0.0001`. The corresponding
length-normalized sensitivity effects were `0.1016` and `0.0941`.

The preregistration required both unpooled test splits to pass. The overall
decision is therefore `not_supported`. The entity split provides direct
evidence against the registered layer-specific contrast: the confidence
interval excludes the prespecified `+0.10` effect in favor of the primary
intervention relative to its strongest control.

## Behavior and Preservation

- The primary intervention changed `0/72` greedy outputs on the entity split
  and `1/72` on the template split.
- All 72 positive-direction bound-prompt preservation checks retained the
  correct archive code.
- All 144 unrelated-task checks in the primary receipts were preserved.
- Format, cross-unit-copying, and preservation gates passed.

The intervention therefore changed the registered code-versus-`UNKNOWN`
probability margin much more consistently than it changed the emitted answer.

## Claim Boundary

The v2 result supports a narrow observation: a training-derived answerability
direction can steer Gemma's response margin on fresh controlled prompts. It
does not establish a robust layer-specific answerability mechanism because
the wrong-layer control dominated on one split. It also does not establish
general metacognition, hallucination prevention, natural-question transfer,
or transfer to larger models.
