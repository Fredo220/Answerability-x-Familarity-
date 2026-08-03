# Same-String Representation Replication v3 Design

**Date:** 2026-08-03
**Status:** approved design; no v3 activations or outcomes opened
**Study identity:** `same-string-representation-replication-v3`
**Scope:** representation-only, model-specific, non-causal

## Objective

Test whether Gemma 2 2B contains held-out activation information about
contextual exposure and answerability that generalizes across new entity units
and prompt families, beyond information available to strong prompt-surface
baselines.

This is a new replication. It does not modify, pool with, rescue, or reinterpret
the completed Same-String v2 behavioral result or its exploratory `N=4`
representation pilot.

## Threats Addressed

The design directly addresses two limitations of v2:

1. Four independent test units made test metrics highly unstable.
2. Synthetic prompt structure could let a probe learn surface regularities
   instead of a representation that transfers to new wording.

No finite experiment removes every alternative explanation. The purpose of v3
is to make these two explanations substantially harder while keeping the study
feasible on free-tier Colab and an 8 GB analysis machine.

## Factorial Unit

Each independent unit contains one target nonce name, one distractor nonce
name, one archive code, one neutral property, and four prompts crossing:

- exposure: `low` versus `high`;
- answerability: `target_unbound` versus `target_bound`.

The question always asks for the target's archive code. Within a fixed exposure
condition, answerability is changed only by swapping entity-attribute bindings:

```text
target_bound:
  Aster has archive code C7. Beryl has property silver.

target_unbound:
  Beryl has archive code C7. Aster has property silver.
```

The two prompts must have identical normalized token multisets, lengths, target
counts, distractor counts, code counts, and property counts under the pinned
Gemma tokenizer. Statement order is deterministically counterbalanced.

Exposure uses the same principle: the same unrelated facts and entity names
appear in both cells, but their target/distractor bindings are swapped. Holding
answerability fixed, the exposure pair must also have identical normalized
token multisets and factor-irrelevant counts.

## Corpus and Splits

The corpus contains 80 independent complete 2x2 units and 320 prompts:

| Split | Units | Prompts | Purpose |
|---|---:|---:|---|
| `representation_train` | 32 | 128 | Fit scalers, PCA, and probes |
| `representation_validation` | 8 | 32 | Software and calibration audit only |
| `entity_test` | 20 | 80 | New entities with previously seen templates |
| `template_test` | 20 | 80 | New entities with two entirely unseen templates |

Six template families are written and audited before corpus construction. Four
are available to training, validation, and `entity_test`; two are exclusive to
`template_test`. Entity names, codes, and unit IDs never cross splits.

The allocation is deterministic from a fixed seed and SHA-256 ordering. Test
rows and labels are sealed before model execution. No unit may move between
splits after any activation is opened.

## Model and Activations

- Model: `google/gemma-2-2b-it`.
- Model and tokenizer revision:
  `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`.
- Deterministic inference, batch size 1, no gradients.
- Fixed residual-stream layers: `0, 6, 12, 18, 25`.
- Fixed anchors: `target_intro_end` and `user_prompt_end`.
- The rendered chat template is tokenized with `add_special_tokens=False` to
  prevent the previously observed double-BOS defect.

Only the fixed anchors and layers are retained. Full-token hidden-state caches
are not persisted.

## Models Compared

All preprocessing is fitted on `representation_train` only.

### Surface baseline

One fixed logistic model combines:

- character `char_wb` TF-IDF n-grams of lengths 3 to 5, capped at 4,096
  training-derived features;
- whitespace-token TF-IDF unigrams and bigrams, capped at 4,096
  training-derived features;
- prompt/token length and registered count features.

Vocabulary, inverse-document-frequency weights, scaling, and coefficients are
learned from training prompts only. Both TF-IDF blocks use `min_df=2`,
`sublinear_tf=True`, and no lowercasing. The logistic model uses `C=1`, L2
regularization, no class weighting, and `max_iter=2000`.

### Activation baseline

For each fixed anchor and layer:

1. standardize residual vectors using training statistics;
2. fit a training-only PCA with at most 16 components;
3. fit `C=1` logistic regression.

### Combined model

Concatenate the fixed surface representation with the training-only activation
projection and fit the same fixed logistic regression. No layer, anchor,
regularization value, or feature family is selected from test performance.

## Registered Questions

### Primary question

At `user_prompt_end`, does the combined model improve answerability prediction
over the surface model on both held-out test splits?

Primary quantities:

- the mean across the five fixed layers of the unit-averaged paired log-loss
  difference, `surface - combined`;
- the mean across the five fixed layers of the AUROC difference,
  `combined - surface`.

Support requires, separately on `entity_test` and `template_test`:

1. positive mean paired log-loss improvement;
2. a unit-bootstrap 95% interval whose lower bound is above zero;
3. non-negative AUROC difference;
4. a within-unit permutation value at most `0.05` for the fixed mean-layer
   statistic.

### Exposure diagnostic

Exposure decoding at `target_intro_end` is reported with the same baselines and
uncertainty, but it is secondary because exposure is explicitly expressed in
the prompt context.

### Temporal negative control

Answerability at `target_intro_end` should remain near chance. Material
decodability there is treated as evidence of leakage, anchor failure, or a
construction artifact rather than as support.

## Statistical Procedure

- Independent resampling unit: complete 2x2 entity unit.
- Report `entity_test` and `template_test` separately; do not merge them to
  rescue a failed split.
- Use 10,000 fixed-seed unit bootstrap draws for final intervals.
- Use exactly 999 fixed-seed, other-factor-stratified label permutations.
- Report raw layer values, max-layer-adjusted values, and a fixed mean-layer
  omnibus statistic.
- Report balanced accuracy, AUROC, log loss, calibration slope/intercept, and
  all held-out predictions.
- Run and freeze a simulation-based minimum-detectable-effect audit before any
  v3 activation extraction. The audit may qualify interpretation but may not
  change sample size, endpoints, or thresholds after outcomes are opened.

## Construction and Leakage Gates

The corpus is not eligible for model execution unless all checks pass:

- exactly 80 complete units and 320 unique examples;
- exact split and template-family counts;
- no entity, code, or unit overlap across splits;
- complete four-cell structure per unit;
- identical within-pair normalized token multisets and token counts;
- fixed target, distractor, code, property, and question within each unit;
- answerability differs only through the registered binding swap;
- exposure differs only through the registered unrelated-fact binding swap;
- no archive-code leakage in unrelated exposure facts;
- exact pinned tokenizer, chat-template hash, and single BOS token;
- deterministic reconstruction reproduces all prompt and manifest hashes.

Any failed gate stops the run. It cannot be repaired after v3 outcomes are
opened; a corrected corpus would require a new study identity.

## Execution and Compute

Corpus construction, audits, probe fitting, statistics, and reporting run
locally. The 320 short forward passes may run locally with the cached model or
on free-tier Colab. Extraction is checkpointed by split and resumes only under
the same model, code, manifest, and request hashes.

The implementation reuses the existing Gemma loader, activation anchors,
artifact manifests, deterministic hashing, and CLI patterns. It does not add a
new training framework, SAE, attribution graph, ODE solver, or online gradient
path.

## SkillOpt Process Boundary

SkillOpt is used only as a validation-gated process aid for agent instructions:
past execution failures can become held-out workflow checks, and proposed skill
edits must outperform the current instructions before adoption. SkillOpt output
is not experimental evidence and cannot modify the frozen v3 corpus, endpoints,
labels, thresholds, or result interpretation.

The local SkillOpt runner is currently absent. Until it is installed and a
reviewed task split is available, the study uses the same validation-gated
principle without claiming that an optimization run occurred. Installing or
running SkillOpt is not a prerequisite for v3 evidence.

## Claim Policy

A positive result would support only this statement:

> On this controlled Gemma 2 2B task, fixed residual-stream features improved
> held-out answerability prediction beyond registered surface baselines across
> new entities and unseen prompt templates.

It would not establish a causal mechanism, general metacognition, reasoning,
truth detection, hallucination prevention, or transfer to frontier models. A
null result, failed construction gate, leakage signal, or incomplete run is
published under its actual status.

## Acceptance Criteria

The implementation is complete only when:

1. focused and repository regression tests pass;
2. the corpus and pre-outcome MDE audit are hash-bound before activations;
3. all four split extractions complete under the pinned identity;
4. test metrics are computed once and released with predictions and manifests;
5. README and result documents state the effective sample sizes, surface
   baseline comparison, both generalization splits, nulls, and limitations.
