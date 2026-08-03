# Same-String Representation Replication v3

**Decision:** `supported`

This fixed analysis tests whether Gemma 2 2B residual activations add held-out answerability information beyond registered TF-IDF surface baselines.

| Test split | Units | Mean paired log-loss improvement | 95% CI | Mean AUROC improvement | Permutation p | Supported |
|---|---:|---:|---:|---:|---:|---|
| `entity_test` | 20 | 0.4717 | [0.3593, 0.5474] | 0.4050 | 0.0010 | true |
| `template_test` | 20 | 0.3029 | [0.0884, 0.4749] | 0.3831 | 0.0010 | true |

## Claim boundary

A positive decision supports only model-specific, correlational held-out decodability on this controlled task. It does not establish causality, general metacognition, truth detection, or hallucination prevention.

The registered surface comparator is a bag-of-ngrams TF-IDF model, not a symbolic binding parser. Answerability is explicitly encoded by the prompt sequence, so this result concerns internal representation decodability rather than information absent from the input.
