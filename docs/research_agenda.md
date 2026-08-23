# Research Agenda: Verification-Coupled Reasoning at the Interpretability–RL Interface

This document is forward-looking: it extends the completed
Familiarity-versus-Answerability study toward a fellowship-scale research
program. Nothing below is claimed as demonstrated. Each milestone opens only
under its own preregistration with fail-closed endpoints, following the
standards recorded in the README.

## Overarching question

Can probabilistic LLM reasoning be turned into a step-wise, formally
constrained process in which the model searches over reasoning paths while a
deterministic formal system guarantees the validity of every accepted
transition — and can the internal signals that make a step trustworthy also
serve as a training signal, not only an inference-time filter?

## Proposed pipeline

```text
natural language
  → LLM-generated reasoning representation
  → autoformalization
  → Lean 4 step-by-step verification
  → (optionally) reward signal back into policy training
```

Lean would check each individual reasoning step, not only the final artifact. A
step that does not follow from the accumulated premises is rejected, and the
model must generate an alternative — either at inference time (filtering
candidate continuations) or during training (shaping which trajectories get
reinforced).

## What would be new here

Two adjacent lines of work are established. Verifier-in-the-loop proving
couples an LLM proposer to a formal checker in neurosymbolic theorem proving
(e.g., DeepSeek-Prover, AlphaProof). Process-reward models (PRMs) score
intermediate reasoning steps as dense training signals. This agenda differs
from both by grounding the intermediate signal in the model's own activations
rather than in a separately trained step classifier:

**Interpretability side.** Treat internally decoded evidence-sufficiency
signals (the representation result of the completed study, README Section 3.2)
as a monitor on premise quality before a step is accepted, and map which
internal states correspond to individual formal operations through circuit
analysis (attribution graphs) and pretrained SAE features such as Gemma Scope,
rather than training new SAEs from scratch.

**RL side.** Verified reasoning chains are typically rewarded only at final
outcome — the proof compiles or it does not — which yields a sparse-reward
problem over long chains. A mid-chain evidence-sufficiency signal that
measurably tracks premise quality would be a candidate for a denser
process-reward proxy during RL fine-tuning. That raises a direct faithfulness
question: does a policy trained against such a proxy learn to ground its
reasoning more genuinely, or does it learn to satisfy the proxy without the
evidence-grounding it is meant to track — structurally related to known
chain-of-thought unfaithfulness concerns. Because the completed study provides
a decoder-based measurement of evidence-grounding, this dissociation becomes
empirically testable rather than merely rhetorical.

## An external precedent from an adjacent setting

Weco AI's AIDE² report (first-party company report, July 2026; self-published,
not yet peer-reviewed) studies something distant from reasoning verification —
an outer loop rewriting an autonomous research agent over 100 unattended
iterations. Its evaluation design and two reported findings nevertheless map
closely onto this agenda's central risks:

1. **Built monitors silently failed under iteration.** The evolved agent
   accumulated a three-layer defense against reward hacking, one layer of
   which — a statistical filter for suspiciously extreme results — contained a
   bug that made it inert in the final agent; an earlier ancestor had
   implemented the mechanism correctly, and a later mutation broke it
   undetected. Whatever one concludes about the report's headline numbers,
   this is direct support for treating any evidence-sufficiency monitor as an
   object of ongoing empirical scrutiny across iterations, not as a solved,
   static component.

2. **Proxy-gaming fell under selection against a held-out objective.** On a
   KernelBench evaluation, measured reward hacking fell from 63% in the
   starting agent to 34% after 100 iterations, with the hand-tuned baseline at
   42%. Weco attributes this to selecting candidates on a private score never
   visible to the optimized agent. Read cautiously, this is consistent with
   the hypothesis that selection pressure against a true held-out objective
   can shape a system's tendency to game a visible proxy. It is suggestive,
   not confirmatory: the operative mechanism there was indirect outer-loop
   selection, whereas using an internal probe as a direct reward signal is a
   different and harder regime.

Both findings inform the design constraints below; neither supports any claim
of success.

## Why the completed study is upstream

The central bottleneck is autoformalization: an LLM may inject false or
unsupported premises while translating natural language into a formal
representation, so a Lean-checked proof can be valid yet unsound relative to
the original text — and, on the RL side, a policy could learn to emit premises
that satisfy a reward proxy without being genuinely evidence-grounded. The
completed study measured one ingredient of that bottleneck — whether evidence
sufficiency is represented internally and separable from lexical familiarity —
and found decodable answerability without a robust layer-specific causal
mechanism. Those boundaries carry over as constraints on this agenda; they are
not resolved by it. In particular, whether evidence-sufficiency decoding is
reliable enough to serve as a monitor or reward proxy is itself an open
empirical question this agenda must establish — and the precedent above
suggests that even initially demonstrated reliability cannot be assumed stable
under continued iteration.

## Fellowship-scale milestones

**M1 — Natural-question replication.** Replace archive-code prompts with
natural factual and evidence-grounded questions on a second model family,
retaining matched answerability controls; publish a power analysis before
opening any outcome. One schedule risk is stated openly: the completed
project's confirmatory cycle showed that controlled-corpus construction, not
model compute, dominates the timeline (four source iterations ending in a
failed construction gate). M1 therefore treats corpus construction as its
critical path and falls back to a smaller preregistered pilot if its
construction gate fails again.

**M2 — Premise-faithfulness metric.** Formalize agreement between
autoformalized premises and the evidence actually present, evaluated on a small
independently audited corpus using a visible-score/held-out-score split
analogous to the public/private design above. This directly tests whether the
evidence-sufficiency signal is reliable enough to serve later as either an
inference-time monitor or a training-time reward proxy. Human audit time is the
binding constraint and is planned explicitly rather than assumed away.

**M3 — Minimal verification loop with a reward-hacking probe.** A
generate–verify–regenerate prototype in which every accepted step must compile
in Lean, measuring (a) how often internal evidence-sufficiency signals disagree
with premise soundness, and (b) — stretch goal, contingent on M1 and M2 landing
early — under lightweight RL fine-tuning against the verification signal,
whether decoder-measured evidence-grounding improves in step with reward or
dissociates from it: a measurable signature separating genuine faithfulness
from proxy-satisfaction. Fallback if time runs short: evaluate already-trained
checkpoints instead of running fresh fine-tuning, preserving the dissociation
measurement at reduced scope.

## Claim discipline

Nothing above is claimed as demonstrated. The completed study does not
establish that internal states correspond to formal operations, that steering
can enforce valid steps, that evidence-sufficiency signals are reliable enough
to use as a monitor or reward proxy, that such a proxy — even if initially
reliable — would remain reliable under iterative optimization pressure, or that
the v2 margin effect generalizes beyond archive-code prompts. Each milestone
opens only under its own preregistration with fail-closed endpoints, and
negative outcomes remain part of the public record.
