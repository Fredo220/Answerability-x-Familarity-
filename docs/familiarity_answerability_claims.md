# Familiarity vs. Answerability Claim Ladder

The study asks whether entity familiarity changes answer behavior when
relation-specific evidence is absent, and whether internal states add useful
pre-output information. The project does not claim a universal causal account
of hallucination or generalization beyond the registered task.

## F1: Behavioral Interaction

Allowed only after the registered F1 gate passes:

> Under the tested prompts and model revision, familiarity had a preregistered
> behavioral interaction with answerability while accuracy, format validity,
> and naturalness controls passed. Outputs were scored with the registered
> deterministic exact-string protocol; no model judge was used.

Do not shorten this to "familiar models hallucinate" or claim a universal
causal mechanism. If the interaction, non-inferiority, audit, or validity gate
fails, report a negative result or `not_evaluable` as appropriate.

## F2A: Pre-output Prediction

Allowed only when the registered H3-H5 gate passes on the one-use `probe_test`
endpoint: condition-invariant decodability, incremental pre-output prediction
beyond the registered surface/static/output-aligned controls, and all required
J-space and full-selection nulls. H6 asks whether dynamics add value beyond the
registered static model; it is secondary and cannot invalidate a passing H3-H5
result.

> Registered internal features contained incremental pre-output prediction of
> the study endpoint under the tested distribution.

This is decodability evidence, not proof that the model uses the signal. A probe
must not be described as a mechanism without the registered causal evidence.
For H5, "pre-output" refers only to the incremental prompt-end activation
features: the nested baseline and candidate both contain the same registered
output-margin control.

## F2B: Local Causal Evidence

Allowed only after F1 and F2A pass and a prefill-only intervention changes the
actual answer-versus-abstain behavior in the registered direction while
preserving answerable-case performance. The prompt and assistant-prefix bytes
must remain identical; wrong-layer, wrong-anchor, orthogonal, shuffled,
sign-reversed, norm-matched, cross-entity, and reverse-direction controls must
fail to explain the effect.

> A sealed intervention provided local causal evidence that the registered
> activation state influenced answer-versus-abstain behavior at this layer and
> prompt position.

Local causal evidence is not a claim about global truthfulness, broad factual
accuracy, safety, or a unique circuit.

## F3: Prompt-local Graph Hypotheses

F3 is optional and deferred until a compatible pinned replacement model passes
all circuit-fidelity gates. Attribution graphs may generate prompt-local
hypotheses only. They cannot rescue F1, F2A, or F2B and are not causal evidence
without an intervention in the original model.

## Null and Negative Results

A negative result is a valid study outcome. Publish it when familiarity does
not interact with answerability, when internal dynamics add no held-out signal,
when a simpler baseline matches the proposed method, when a null distribution
contains the observed statistic, or when steering harms answer quality. Missing
prerequisites are `not_evaluable`, not evidence for or against the hypothesis.
