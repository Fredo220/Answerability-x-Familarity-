# Same-String v2 Representation-Only Pilot

**Status:** complete and exploratory
**Model:** `google/gemma-2-2b-it` at revision
`299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`
**Analysis identity:** `d904abe2942c30aea1616c082aa6d9e03cfe86f7fbe4d8a66c019e5395448d3e`

## Question

Does the model represent contextual exposure and answerability differently
internally, even though that separation did not appear as the hypothesized
failure behavior in the small behavioral pilot?

## Design

The analysis was frozen before activations were opened. It used the unchanged
Same-String v2 prompts: 12 `mechanism_train` units and four
`locked_validation` units for fitting, then four held-out `probe_test` units.
Each unit contained all four exposure x answerability cells, for 80 prompts in
total and 16 held-out test prompts.

Five fixed layers (`0, 6, 12, 18, 25`) were reported without best-layer
selection. Linear probes were compared with a morphology-only baseline, 99
within-unit stratified label permutations, and entity-unit bootstrap intervals.

## Results

### Exposure

At `target_intro_end`, residual-only probes decoded high versus low exposure
with test AUROC `1.00` and balanced accuracy `1.00` at every fixed layer. The
max-layer-adjusted permutation value and mean-layer omnibus value were both
`0.01`, the smallest possible value with 99 permutations. The morphology-only
baseline had AUROC `0.1875` and balanced accuracy `0.3125`.

At `user_prompt_end`, exposure remained decodable at several layers but was not
uniform: residual-only AUROC ranged from `0.5781` to `1.00`.

### Answerability

At `target_intro_end`, answerability was at chance at every fixed layer
(AUROC and balanced accuracy `0.50`). This anchor occurs before the relevant
archive-code evidence has been fully presented.

At `user_prompt_end`, residual-only answerability AUROC ranged from `0.9219` to
`1.00`; mean-layer AUROC was `0.9781`, with max-layer-adjusted and mean-layer
permutation values of `0.01`.

However, the morphology-only baseline already achieved AUROC `1.00` and
balanced accuracy `0.9375` for answerability. The present pilot therefore does
not show that internal activations add answerability information beyond prompt
surface cues.

## Interpretation

The result is consistent with a position-dependent separation:
exposure is available at the target-introduction anchor, while answerability is
absent there and becomes decodable only after the evidence-bearing prompt has
been processed. Exposure decoding also substantially exceeded the registered
morphology baseline in this sample.

This is not evidence of a causal mechanism, metacognition, reasoning, general
familiarity, or hallucination detection. Only four independent test units were
available. Perfect bootstrap intervals reflect perfect performance on those
four units, not certainty about a wider population. The analysis does not
alter or rescue the closed `not_supported` behavioral result.

## Reproduction artifacts

- [Frozen analysis amendment](../amendments/2026-08-02-fa-same-string-representation-pilot.md)
- [Raw metrics](../../release/familiarity_answerability/representation_pilot_v2/same-string-v2-representation-pilot-metrics.jsonl)
- [Metrics manifest](../../release/familiarity_answerability/representation_pilot_v2/same-string-v2-representation-pilot-metrics.jsonl.manifest.json)
- [Held-out predictions](../../release/familiarity_answerability/representation_pilot_v2/same-string-v2-representation-pilot-predictions.jsonl)
- [Predictions manifest](../../release/familiarity_answerability/representation_pilot_v2/same-string-v2-representation-pilot-predictions.jsonl.manifest.json)

Raw activation arrays are not committed because they add approximately 58 MB
of model-derived data. Their content hashes, request hashes, prompt hashes, and
analysis implementation hash are retained in the released manifests; they can
be regenerated from the pinned model and public prompt snapshot.
