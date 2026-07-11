# Exploratory Circuit-Tracing Follow-up

## Status and role

This phase is an exploratory mechanistic validation. It cannot change the
preregistered detection or intervention outcome. Cases are selected only after
the primary held-out predictions have been persisted, using a deterministic
rule implemented by `prepare-circuit-followup`.

Anthropic's open-source circuit-tracing project supports attribution graphs and
feature interventions on Llama 3.2 1B. Its published Llama transcoders target
the base checkpoint. The primary study uses Llama 3.2 1B Instruct. The two must
therefore be treated as different model checkpoints.

## Frozen case selection

For the selected dynamics-containing detector, use its validation-frozen
threshold and held-out test predictions. Select up to five examples from each:

1. true positive;
2. false positive;
3. false negative;
4. true negative.

Within each stratum, rank by absolute distance from the frozen threshold and
break ties by example ID. This selects archetypal cases without manual
cherry-picking. Both exact-error and distractor-binding endpoints may be
prepared, but their outputs remain separate.

## Base-model replication

Run the selected concept prompts on `meta-llama/Llama-3.2-1B` with the published
Llama transcoders. Re-label every base-model response against the same supplied
fact table. A case is mechanistically comparable only if the base model exhibits
the same correct or distractor-binding outcome as the Instruct checkpoint.

For comparable cases:

1. attribute the correct-object and strongest distractor-object logits;
2. identify feature paths carrying target entity, queried relation, correct
   object, and competing object evidence;
3. record transcoder error nodes and graph completeness rather than hiding them;
4. intervene on candidate paths and measure the change in correct-minus-distractor
   logit difference;
5. compare against equal-count, norm-matched random feature interventions.

Report the fraction of cases with a reproducible feature-level explanation,
causal logit-difference change, graph size, unexplained-error contribution, and
inter-annotator agreement for feature annotations.

## Instruct-checkpoint fidelity gate

Do not apply base transcoders to the Instruct checkpoint as if they were valid by
construction. Such transfer is permitted only as a separately labelled analysis
after measuring replacement fidelity on held-out benign prompts, including
next-token top-1 agreement, logit KL divergence, and task-accuracy preservation.
Failure of this gate means no feature-level claim is made about the Instruct
checkpoint.

## Hardware boundary

The public Llama cross-layer and per-layer transcoder repositories are tens of
gigabytes, larger than the local 8-GB memory budget even before loading the base
model and attribution graph. Full circuit extraction therefore belongs on a GPU
machine or hosted Neuronpedia workflow. The local CPU study remains responsible
for data generation, causal trajectory extraction, held-out evaluation, and
deterministic case selection.

## Jailbreak boundary

The base Llama checkpoint has no equivalent chat-safety policy, so its circuits
cannot directly explain jailbreak success in the Instruct checkpoint. The
jailbreak track receives circuit tracing only if checkpoint-compatible
transcoders pass the same fidelity gate. Until then, it uses causal prefix
detection and activation steering only.
