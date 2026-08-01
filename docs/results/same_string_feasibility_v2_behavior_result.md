# Same-String Balanced Pilot v2: Behavioral Result

- **Study:** `familiarity-answerability-same-string-feasibility-v2`
- **Model:** `google/gemma-2-2b-it` at revision `299a8560...`
- **Status:** `not_supported`
- **Endpoint:** evaluable and permanently closed
- **Complete units:** 32 across four registered domains

## Registered result

The preregistered exposure-by-answerability interaction was `-0.09375`. Its
crossed-bootstrap 95% confidence interval was `[-0.4000, 0.1818]`. The point
estimate was below the registered minimum of `0.05`, and the interval did not
exclude zero in the predicted positive direction.

The registered Same-String hypothesis is therefore **not supported in this
pilot**. This is not evidence that contextual familiarity can never matter. It
means this specific manipulation, estimator, sample, and small model did not
provide the required evidence.

| Exposure | Answerability | Attempt rate | Abstention rate | Exact target | Valid format |
|---|---|---:|---:|---:|---:|
| High | Code absent | 0.0625 | 0.9375 | 0.0000 | 1.0000 |
| Low | Code absent | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| High | Target bound | 0.9063 | 0.0938 | 0.9063 | 1.0000 |
| Low | Target bound | 0.7500 | 0.2500 | 0.7500 | 1.0000 |

The answerability manipulation behaved as intended: the model usually
returned the supplied code and usually abstained when the code was absent.
However, the high-minus-low exposure difference was larger in the answerable
condition than in the unanswerable condition, yielding the negative registered
interaction.

Numerically, exposure increased attempts by `0.0625` when the code was absent,
but by `0.15625` when the code was supplied. The registered difference between
those two exposure effects was therefore `0.0625 - 0.15625 = -0.09375`.

## Secondary registered gate

The high-minus-low exposure difference in the target-bound cells was `0.15625`,
with a 95% interval of `[-0.0769, 0.4333]`. Although the point estimate was
positive, its lower bound crossed the registered `-0.05` non-inferiority
margin. The capability-preservation gate therefore also failed.

All 128 prompt rows completed. Every cell had 32 observations, 100% completion,
and 100% format validity. No draw was discarded from the 10,000-draw crossed
bootstrap.

## Protocol deviation

The v2 amendment required a pre-outcome power or minimum-detectable-effect
audit. No corresponding artifact is present in the verified snapshot. This
omission is not repaired with a post-outcome calculation.

The generated endpoint still receives the machine-evaluated decision
`not_supported`: the amendment stated that the power calculation could qualify
the strength of the pilot but could not alter its endpoint, thresholds, or
observed-result interpretation, and it was not listed among the hard-stop
requirements for opening the endpoint. The omission nevertheless weakens the
study. The wide observed interval and small sample require treating v2 as an
imprecise pilot rather than a definitive null result.

## Decision

The registered gate failed for three reasons:

1. `interaction_point_estimate_below_minimum`
2. `interaction_interval_not_positive`
3. `capability_noninferiority_lower_bound`

The gated mechanistic pilot was not run. Under the amendment, an exploratory
activation result cannot rescue or reinterpret a failed behavioral endpoint.

## Claim boundary

This result supports only the following statement:

> In a balanced 32-unit Same-String pilot on the pinned Gemma 2 2B checkpoint,
> unrelated contextual exposure did not produce the preregistered positive
> exposure-by-answerability interaction in answer attempts.

It does not establish a general null effect for familiarity, answerability,
hallucination, larger models, natural knowledge, or other prompt designs.

## Why the hypothesis may not have been supported

Only the first observation below is a direct result. The remaining points are
possible explanations that this experiment did not identify causally.

1. **The code-absent condition was near the floor.** The model abstained on
   97% of code-absent prompts overall, leaving little room for exposure to
   increase unsupported attempts.
2. **Attempt rates were higher under exposure when usable evidence existed.**
   The larger exposure difference occurred in the target-bound condition. One
   possible explanation is that the extra context supported use of available
   evidence rather than indiscriminate answering, but this mechanism was not
   tested.
3. **The manipulation was narrow.** It measured short-term contextual exposure
   to a synthetic target string, not broad familiarity learned during
   pretraining.
4. **The pilot was small and imprecise.** With 32 units, the confidence interval
   still includes moderate effects in either direction.
5. **Model and task scope were limited.** A synthetic archive-code task on one
   2B checkpoint may not generalize to natural factual questions or larger
   models.

None of these explanations can be selected after the fact as the cause of the
result. They are hypotheses for a new experiment.

## Recommended next step

Do not rerun or tune this protected endpoint. Preserve v2 as the registered
negative pilot and create a new study with unseen units:

1. Calibrate prompts only on open development data so code-absent attempt rates
   are not pinned near zero and target-bound rates are not pinned near one.
2. Run and publish a power and minimum-detectable-effect simulation before any
   protected model output, then freeze the sample size.
3. Keep the Same-String 2x2 estimand, but add a registered manipulation check
   showing that high exposure changed the intended familiarity proxy.
4. Replicate the frozen design on the current checkpoint and at least one
   larger or independent open-weight model.
5. Gate any activation analysis on the new behavioral result, or register it
   separately as a manipulation-check study rather than using it to rescue a
   null behavior result.

This sequence tests whether the unsupported result arose from the phenomenon,
the floor effect, low power, or model/task specificity without rewriting v2.

## Reproducibility

The endpoint was unlocked once, evaluated once, and closed. The final snapshot
was restored locally and its content-addressed members and endpoint state were
verified.

- Runtime code commit: `08085e77e153a5314e9ac7698ff9999a7bea8bbc`
- Launcher commit: `df405c6dd073d721ef5a09ad834137e778a2fe90`
- Amendment SHA-256: `f6e1c26c1e3cac853c06b9d97eef559ea1143ff5253af81c7d9a0e1e0ec26d0c`
- Evidence SHA-256: `0b53629677888308f19f4ef183e260d9affd82e0b427459172c96250d8b05173`
- Metrics SHA-256: `beba778e3cdc7f36d7e33132ea6d06f63f59b7f9bb6336b7a88b46d4338e41b6`
- Snapshot SHA-256: `58f1f069cb6a1906ff17a0282805f859675ae80b0f707fc0f768fc7a956178e3`

The compact machine-readable record is
[`same_string_feasibility_v2_behavior_result.json`](same_string_feasibility_v2_behavior_result.json).
The full content-addressed snapshot is published as
[`fa-58f1...178e3.zip`](../../release/familiarity_answerability/fa-58f1f069cb6a1906ff17a0282805f859675ae80b0f707fc0f768fc7a956178e3.zip).
