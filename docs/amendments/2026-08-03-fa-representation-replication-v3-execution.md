# Same-String Representation Replication v3 Execution Record

**Date:** 2026-08-03
**Study:** `same-string-representation-replication-v3`

This record discloses two implementation details without changing the frozen
corpus, labels, splits, layers, endpoints, thresholds, or support rule.

## Batched extraction

The initial local extraction processed one prompt per forward pass and was
stopped before any split shard was published because projected runtime exceeded
one hour. The same frozen prompts were then grouped by equal rendered-token
length and processed in batches of four. Each example retained its own anchor
indices and activation hash. The model, tokenizer, layers, prompts, and analysis
were unchanged. The final result lineage records the batch size and extraction
implementation hash.

The shared activation artifact format stores the existing
`assistant_prefix_end` position in addition to the two v3 analysis anchors for
backward compatibility. V3 analysis reads only `target_intro_end` and
`user_prompt_end`; no assistant-prefix value enters model fitting or selection.

## Surface-count compliance correction

After the first fixed analysis completed, code review found that the TF-IDF
surface model included registered length features but omitted four registered
count columns. The omitted columns were then added exactly as specified. An
audit showed that they were invariant across all 320 prompts:

- full prompt: target `4`, distractor `2`, code `1`, property `1`;
- early prefix: target `2`, distractor `1`, code `0`, property `0`.

They therefore carry no label information after training-only scaling. The
correction changed neither the typed decision nor any endpoint rounded to four
decimal places:

- `entity_test`: log-loss improvement `0.4717`, AUROC improvement `0.4050`,
  permutation `p=0.001`;
- `template_test`: log-loss improvement `0.3029`, AUROC improvement `0.3831`,
  permutation `p=0.001`.

The pre-correction analysis hash was
`30811e7c090388a04348126a9c5028825dbaf80f5aa7d42210e2cf7cd2679444`.
The compliant final analysis hash is
`d69ea385d3850d49e540489f7df19cdc073032f4a35d78011ef17cd094358dd3`.
The pre-correction output is superseded and is not used as evidence.
