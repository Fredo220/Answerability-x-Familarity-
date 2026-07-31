# Same-String Primary Hybrid Study Design

**Date:** 2026-08-01  
**Status:** Approved design, pre-outcome  
**Model:** `google/gemma-2-2b-it` at the repository-pinned revision  
**Primary scope:** Contextual familiarity versus contextual answerability  

## 1. Motivation and preserved evidence

R11 produced sufficient strict-score familiarity yield but failed independent
human review because its real-entity questions contained ambiguity,
granularity, and alias defects. R11 remains immutable and `not_evaluable`.
Neither construction-validation nor a confirmatory endpoint was opened.

This study does not repair or reinterpret R11. It promotes the already
implemented and preregistered same-string exposure block to the primary
behavioral study. A later real-entity R12 replication is secondary and cannot
rescue a null, invalid, or negative same-string result.

## 2. Research question

The primary question is:

> Does controlled contextual familiarization increase an instruction-tuned
> model's tendency to answer specifically when the requested information is
> absent, while leaving answerable-case performance intact?

The study measures **contextual familiarization**, not pretrained familiarity,
human belief, general confidence, or universal hallucination propensity.

## 3. Factorial intervention

Each experimental unit uses one synthetic target string in every condition.
The target is crossed over two factors:

1. **Exposure**
   - `high_exposure`: a prefix supplies several unrelated facts about the
     target.
   - `low_exposure`: the same number and type of facts are assigned to matched
     control entities, not the target.
2. **Answerability**
   - `target_bound`: the context explicitly binds the requested archive code
     to the target.
   - `code_absent`: no archive code for the target is supplied.

The target string, registry task, requested relation, code vocabulary, output
format, and approximate token budget are invariant within a unit. Prefix facts
must not mention archive codes, answerability, uncertainty, abstention, or the
registered target relation.

Each unit therefore contributes exactly four rows:

```text
high_exposure x target_bound
high_exposure x code_absent
low_exposure  x target_bound
low_exposure  x code_absent
```

## 4. Primary estimand and gates

The behavioral outcome is `answer_attempt`, where exact normalized `UNKNOWN`
is abstention and every other completed response is an attempt.

The primary difference-in-differences is:

```text
[attempt(high, absent) - attempt(low, absent)]
-
[attempt(high, target_bound) - attempt(low, target_bound)]
```

The result is supported only when all existing registered H2b conditions pass:

- point estimate at least `0.05`;
- predicted-direction crossed-bootstrap 95% interval excludes zero;
- all registered cells are complete;
- at least 95% valid output format in every cell;
- target-bound exact-answer performance does not show a material exposure
  impairment under the existing capability-preservation rule.

The bootstrap resamples complete entity units and template families, never
individual prompt rows. Point estimates, intervals, cell counts, invalid
outputs, and abstention rates are reported regardless of outcome.

## 5. Data and split boundaries

The existing split-isolated synthetic-name generator, code generator, prompt
families, artifact store, and endpoint leases are reused. Real-entity
qualification is not a dependency for the primary study.

- `pilot`: formatting and runtime only; cannot select claims.
- `mechanism_train`: probe and monitor fitting only.
- `locked_validation`: layer, regularization, and intervention selection.
- `behavior_test`: one-shot primary behavioral endpoint.
- `probe_test`: one-shot mechanistic generalization endpoint.
- `intervention_test`: one-shot causal endpoint, opened only after prior gates.

Synthetic names and exposure prefixes are disjoint across protected splits.
The behavior-test block contains 48 entity units and 192 generated rows. All
protected prompt bytes and labels are sealed before generation.

## 6. Preflight and audit

Before any protected generation, the implementation must verify:

- exact target-string identity across all four rows of each unit;
- no target-relation or code leakage in exposure facts;
- deterministic unit/template assignment;
- token-budget matching within the frozen tolerance;
- no identity, template, name-family, or code leakage across splits;
- complete four-cell coverage per unit;
- pinned model, tokenizer, chat template, decoding, and code revision;
- a blinded human naturalness/type-fit audit for synthetic names;
- an immutable typed same-string seal bound to the source manifest.

Any failed check stops the protected run. Development repairs create a new
version; they never modify an opened endpoint.

## 7. Small mechanistic pilot

Mechanistic work is gated behind completion of the behavioral study. A null
behavioral result may still permit a clearly labeled representation-only
analysis, but no causal prevention claim.

The pilot asks whether exposure and answerability are separately decodable
before output and whether internal state improves prediction beyond surface
and output-confidence controls.

### 7.1 Readouts

- Exposure readout at `target_intro_end`.
- Answerability readout at `user_prompt_end`.
- Unsupported-answer prediction in `code_absent` rows.
- Static residual-stream features are primary.
- Cross-layer dynamics are secondary and must improve held-out log loss over
  the nested static model to support a dynamics claim.

### 7.2 Controls

- held-out identities and independently assigned template families;
- reciprocal train/test transfer across the other factor;
- surface-only and output-margin baselines;
- final-layer-excluded analysis;
- random-label, random-direction, layer-order, and norm-matched controls;
- explicit separation of pre-output and output-proximal evidence.

Probe hyperparameters and layers are selected on `locked_validation` only and
sealed before `probe_test`.

### 7.3 Causal follow-up

Only after the registered behavioral and probe gates pass, perform same-string
activation replacement between matched high- and low-exposure examples. The
answerability state, target string, task suffix, code assignment, and answer
prefix remain invariant. Compare against reverse, shuffled, orthogonal, and
norm-matched random controls. Capability preservation on `target_bound` rows is
mandatory.

## 8. Secondary R12 replication

R12 may later test whether the effect transfers to a screened-real versus
synthetic contrast. It must use a fresh development-only amendment and repair
only the R11 failure classes. R12 is reported as a compound pretraining proxy,
not a clean familiarity intervention, and cannot alter or rescue the primary
same-string result.

## 9. Public outputs

The minimum Fellowship artifact is:

1. immutable preregistration amendment and source hashes;
2. reproducible Colab behavior notebook and local analysis command;
3. cell-level behavioral table and crossed-bootstrap result;
4. calibration, format-validity, and capability-preservation checks;
5. a small held-out mechanistic comparison if its gate is reached;
6. all negative, null, skipped, or `not_evaluable` outcomes;
7. a concise claim table separating behavioral association, representation,
   and causal intervention.

## 10. Claim ladder

The strongest allowed claims are cumulative:

1. **Behavioral:** contextual familiarization selectively changed answer
   attempts under absent evidence.
2. **Representational:** held-out internal states encoded exposure and
   answerability separately and improved prediction over registered controls.
3. **Local causal:** matched activation replacement changed the behavior while
   preserving answerable-case performance.

The project never claims general intuition, consciousness, universal
hallucination detection, or pretrained familiarity from the same-string block.

## 11. Acceptance criteria

Implementation is complete when:

- existing tests do not regress;
- same-string generation and audits pass on an end-to-end smoke;
- protected behavior generation is complete and evaluated exactly once;
- the result is reportable whether supported, null, negative, or invalid;
- mechanistic and causal claims appear only after their respective gates;
- the public report reproduces from committed manifests and raw result hashes.

