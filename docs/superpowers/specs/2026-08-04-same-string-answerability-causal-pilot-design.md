# Same-String Answerability Causal Pilot Design

**Date:** 2026-08-04  
**Status:** approved for implementation; no causal-validation or causal-test outcomes opened  
**Study identity:** `same-string-answerability-causal-pilot-v1`  
**Scope:** task-specific causal intervention in `google/gemma-2-2b-it`

## Objective

Test whether the answerability-related residual direction identified from the
completed Same-String representation replication has a local causal influence
on Gemma's decision to answer or abstain.

The narrow causal claim is:

> On this controlled task and pinned model, adding or subtracting a direction
> learned from answerable versus unanswerable training prompts changes the
> model's `archive code` versus `UNKNOWN` response margin in the predicted
> direction on fresh entities and prompt templates.

This study does not reopen or rescue the negative Same-String behavioral pilot.
It does not test the original familiarity interaction, general metacognition,
truthfulness, hidden knowledge, or hallucination prevention.

## Evidence Boundary

The completed v3 representation study is used only as follows:

- `representation_train` activations fit answerability directions and their
  scale;
- `representation_validation` may verify the deterministic direction builder;
- the opened `entity_test` and `template_test` outcomes are excluded from every
  causal design, selection, and analysis step.

The causal study introduces fresh, split-disjoint names, codes, templates, and
unit IDs. No causal-test output may be opened before direction construction,
the validation search space, the selected intervention, all controls, and the
analysis rules are hash-bound.

## Fresh Factorial Corpus

Create 48 independent units and 192 prompts:

| Split | Units | Prompts | Purpose |
|---|---:|---:|---|
| `causal_validation` | 12 | 48 | Select one layer and intervention strength |
| `causal_entity_test` | 18 | 72 | New entities, familiar v3 template families |
| `causal_template_test` | 18 | 72 | New entities and two new template families |

Each unit crosses:

- exposure: `low_exposure` versus `high_exposure`;
- answerability: `target_unbound` versus `target_bound`.

Every prompt asks for the target's archive code and ends with this fixed output
contract:

```text
Reply with only the archive code. If the target's archive code is not stated,
reply UNKNOWN.
```

Within each exposure pair, answerability changes only through the same
target/distractor binding swap used in v3. Within each answerability pair,
exposure changes only through the unrelated-fact binding swap. Token counts,
normalized token multisets, entities, code, neutral property, question, and
output contract are identical within the registered pairs under the pinned
Gemma tokenizer.

## Direction Construction

For each registered layer `l` in `{0, 6, 12, 18, 25}` at
`user_prompt_end`, construct a training-only raw residual direction:

```text
d_l = mean(h_bound - h_unbound), paired within unit and exposure
```

The direction is averaged across complete units, normalized to unit L2 norm,
and oriented so a larger projection means `target_bound`. Its natural scale is
the median positive paired projection gap in `representation_train`:

```text
g_l = median((h_bound - h_unbound) dot d_l)
```

The intervention is additive and prefill-only:

```text
h[l, user_prompt_end] <- h[l, user_prompt_end] + sign * multiplier * g_l * d_l
```

Full activation replacement is excluded because it can transfer donor-specific
content. No online gradients, ODE solver, SAE, or test-fitted direction is used.

## Validation-Only Selection

Search the fixed grid on `causal_validation` only:

- layers: `0, 6, 12, 18, 25`;
- positive multipliers: `0.25, 0.5, 1.0`;
- anchor: `user_prompt_end` only.

For each candidate, add the direction on unbound prompts and subtract it on
bound prompts. Select the candidate with the largest mean bidirectional margin
effect, subject to:

- finite output for every validation unit;
- no prompt, prefix, model, tokenizer, or direction hash mismatch;
- invalid generated-output rate at most `5%`;
- no more than a `5` percentage-point loss in unsteered bound-answer accuracy
  under the non-adversarial positive-direction preservation check.

Ties are resolved by smaller multiplier, then earlier layer. The selected
candidate and hashes are sealed before either causal test split is evaluated.

## Primary Outcome

For prompt `x`, define the teacher-forced sequence log-probability margin:

```text
M(x) = log P(correct archive code | x) - log P(UNKNOWN | x)
```

Both candidate strings are scored with the same terminal token handling. Raw
summed sequence log probability is primary; length-normalized values are
reported as a sensitivity analysis.

For each complete unit, average over exposure and define:

```text
E_unbound = M(unbound, +d) - M(unbound, no intervention)
E_bound   = M(bound, no intervention) - M(bound, -d)
E_unit    = 0.5 * (E_unbound + E_bound)
```

Positive values indicate that adding the direction makes an unanswerable
prompt more answer-favoring and subtracting it makes an answerable prompt more
abstention-favoring.

Report both test splits separately. Do not pool them to rescue a failed split.
Use complete units for 10,000 fixed-seed bootstrap draws and 9,999 fixed-seed
sign-flip permutations.

## Controls

Run the same signed and norm-matched intervention schedule for:

1. `no_intervention`;
2. `sign_reversed`;
3. `label_shuffled_direction`, fit from a frozen unit-level permutation of the
   training labels;
4. `norm_matched_random`, using five frozen orthogonalized random directions;
5. `wrong_anchor`, applying the primary vector at `target_intro_end`;
6. `wrong_layer`, applying it at the deterministic farthest registered layer.

Also report:

- exact generated response class: correct code, `UNKNOWN`, other code, invalid;
- manipulation-check change in projection onto the frozen direction;
- any output of a code belonging to another unit;
- format validity;
- positive-direction performance on bound prompts;
- a small unrelated-task preservation set.

The primary intervention must outperform the strongest matched negative
control in a paired unit bootstrap contrast. A random-control family is
summarized as a family, not cherry-picked by seed.

## Support Rule

The result is `causally_supported` only when all conditions hold separately on
both `causal_entity_test` and `causal_template_test`:

1. mean `E_unit` is positive and its 95% unit-bootstrap lower bound is above
   zero;
2. both `E_unbound` and `E_bound` have positive point estimates;
3. the fixed sign-flip permutation value is at most `0.05`;
4. the paired lower bound for primary minus strongest negative control is above
   zero;
5. no cross-unit code-copying occurs;
6. format and preservation gates pass.

A completed run that misses any support criterion is `not_supported`. Identity,
construction, intervention-audit, or incomplete-run failures are
`not_evaluable`, not null results.

## Compute and Resume Strategy

The live path is designed for free-tier Colab:

- 4-bit inference when the available GPU supports the registered loader;
- batch size 1 and short deterministic completions;
- no gradients;
- one model load per session;
- checkpoint after every unit/control combination;
- atomic, hash-verified resume from Google Drive or downloaded artifacts;
- local CPU-only construction, audit, statistics, plots, and reporting.

The causal test is expected to require multiple free Colab sessions. Runtime
availability is infrastructure, not evidence; interrupted work resumes under
the same hashes rather than creating a new study.

## SkillOpt Boundary

SkillOpt may review development-only workflow instructions and tests. It may
not read protected causal-test outputs, alter this specification, select a
layer or strength, change controls or thresholds, or auto-adopt a proposal.
Any real SkillOpt run is staged, evaluated on a reserved workflow test set, and
manually reviewed. SkillOpt output is never scientific evidence.

## Publication and Claim Policy

Publish the corpus and identity manifests, frozen direction/selection
artifacts, all intervention/control outputs, nulls, failures, exact software
revision, and resource accounting.

A positive result supports only local causal influence of the selected
residual direction on the controlled response margin. It does not establish a
unique mechanism, general answerability, metacognition, truth detection,
hallucination prevention, or transfer to larger models.

