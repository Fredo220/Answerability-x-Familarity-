# RLMF Resource-Scaled Reproduction Preregistration

Date frozen: 2026-07-13

## Scope and provenance

This is a resource-scaled reproduction of the separately queried RLMF intervention, not an exact replication. The source method is RLMF at repository commit `a087e7a1e49f52aaa701add19cd80699b709fdef`. The base model is `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`; PopQA is `akariasai/PopQA` at `5cf59972d88d4aaaa7781ac91b83d053563d8268`.

Study 0 is immutable: its artifacts, registrations, results, and namespaces are not rerun, rewritten, or used to tune Studies 1 or 2. This study measures faithful uncertainty expression and tests an internal metacognitive signal. Quantitative claims do not call the measured object intuition. No jailbreak claim is part of this study.

The frozen study ID is `rlmf-qwen06b-v1`. It uses subject-and-answer-disjoint PopQA groups: 256 pre-SFT, 256 RL train, 128 validation, and 256 test rows (896 total), split seed `20260713`. Artifacts bind configuration hash, source commit, model and dataset revisions, seed, arm, checkpoint hash, and parent hashes. Writes are atomic with immutable completion markers; a corrected analysis receives a new study ID.

## Frozen treatment and estimand

Confirmatory seeds are exactly `11`, `22`, and `33`. The estimand is the finite-three-seed paired mean over these registered seeds, not inference to a seed population. Seeds `44`, `55`, and `66` require preregistration before use and are reported as a separate extension. A one-seed run is infrastructure-only and supports no scientific claim.

Both arms share the pre-SFT checkpoint, prompts, output schema, rewards, data order, rollout count, optimizer settings, LoRA targets, and seed. Their only confirmatory difference is the RLMF advantage formula. Training uses group size four, leave-one-out group consistency, LoRA rank eight, 200 steps, and a separately seeded online metacognition query per completion. Each evaluation row has one designated response and 20 independent auxiliaries. Confidence and metascore use only the registered values 0.0 through 1.0 in 0.1 increments. Labels are computed only from persisted completions; missing completions are never regenerated during evaluation.

The answer completion contains exactly one short answer and one confidence tag. The separate metacognition query is sourced byte-for-byte from the pinned upstream template recorded in `third_party/rlmf/UPSTREAM.json`; it is not included in the rewarded answer completion. RLMF alone uses the separately queried metascore to scale above-mean faithfulness advantages. It is never an additive reward.

The frozen training proxy uses leave-one-out group agreement. Confirmatory evaluation instead uses `g_eval`, the fraction of 20 independent auxiliaries alias-equivalent to the designated answer, and `faithfulness_accuracy = 1 - abs(designated_confidence - g_eval)`. The designated response is excluded from `g_eval`. The primary behavioral endpoint is the upstream cMFG* implementation, with tie-preserving and common-confidence-support sensitivities reported.

## Confirmatory gates

### Study 1: behavioral A/B test

`delta_cMFG_star = cMFG_star(rlmf) - cMFG_star(standard_grpo)`.

Study 1 is supported only when every condition holds:

- The finite-three-seed mean satisfies `delta_cMFG_star >= 0.03`.
- All three per-seed effects are positive.
- The finite-seed paired prompt-cluster bootstrap 95% lower bound is greater than zero, holding seeds fixed and resampling prompts within seed.
- Mean observed-accuracy difference is at least `-0.02`, and its fixed-seed paired prompt-cluster 95% lower bound is greater than `-0.05`.
- Valid format is at least `0.95` in both arms.
- All three paired seeds complete and pass artifact verification.
- The locked validation judge audit passes before test generation.
- Before aggregate test metrics are opened, the blinded test audit's 95% upper confidence limit on absolute arm-differential judge bias is below `0.015`.

Accuracy, intrinsic- and expressed-confidence Brier scores, ECE, absolute expression gap, answer coverage, and format validity are secondary and cannot rescue a failed primary endpoint.

### Study 2A: mechanistic bridge

Study 2A runs only when Study 1 is supported. The locked primary Study 2 anchor is `pre_confidence`: the final token of the unique teacher-forced prefix ending in `</sentence>\n<confidence>`. The locked primary metric is test MAE incremental dynamics gain at that anchor:

`gain_arm = MAE(static_probe) - MAE(static_plus_dynamics_probe)`

`did_gain = gain_rlmf - gain_standard`

Study 2A is supported only when every condition holds:

- The finite-three-seed paired mean satisfies `did_gain >= 0.02` MAE.
- All three per-seed `did_gain` values are positive.
- Its finite-seed paired prompt-cluster bootstrap 95% lower bound is greater than zero.
- `static_plus_dynamics` beats the locked surface-only baseline in the RLMF arm.
- The result is not explained by the locked surface baseline, answer identity, or correctness.
- All three paired seeds are evaluable.

The sole contrast is RLMF-minus-standard incremental dynamics gain over static features at `pre_confidence`. `prompt_end`, R2, Spearman, individual layers, and pre-SFT contrasts are secondary and Holm-corrected. If Study 1 fails, at most one clearly descriptive null-diagnosis seed may be extracted; no mechanistic-support claim is allowed.

### Study 2B: causal confidence-expression intervention

Study 2B runs only when Studies 1 and 2A are supported. With the answer prefix byte-identical and frozen, it patches a validation-selected, probe-aligned `pre_confidence` component from the same-example RLMF state into the standard-GRPO state. It must beat orthogonal, shuffled, norm-matched random, sign-reversed, and earlier-anchor controls.

Study 2B is supported only when every condition holds:

- The validation-selected same-example patch reduces held-out absolute expression error by at least `0.03`.
- All three per-seed error differences are negative.
- The finite-seed paired prompt-cluster bootstrap 95% upper bound for the error difference is below zero.
- It beats every negative control after Holm correction at family-wise `alpha=0.05`.
- The registered dose direction is monotonic on validation, and reverse RLMF-to-standard intervention moves confidence expression oppositely on test.
- Format validity remains at least `0.95`.
- The frozen answer prefix is byte-identical across patched and unpatched conditions.

This can support causal sufficiency of a probe-aligned component for confidence-token expression only. It does not establish mediation, newly created metacognition, hallucination prevention, truth access, consciousness, or general model safety.

## Analysis discipline and stop rules

Train, validation, and test subject groups are disjoint. Test IDs cannot affect prompts, parser rules, aliases, thresholds, selected layers, alpha values, or report wording. The proxy judge may be revised only through the development audit; after parser and source hash freeze, the locked validation audit is a go/no-go gate. Test audit disagreement can estimate measurement bias but cannot change aliases, normalization, prompts, parsers, thresholds, or report wording.

No test-driven change is permitted: no test completion, aggregate, or metric may alter the frozen configuration, prompt, parser, judge, split, threshold, layer, alpha, advantage formula, or report wording. A failed confirmatory gate is reported as `not_supported` and stops the dependent causal stage; descriptive analyses remain explicitly exploratory. Colab interruption resumes only from the last verified checkpoint and does not alter hyperparameters. No online backpropagation, full Jacobians, ODE solvers, matrix exponentials, all-layer retention, or simultaneous loading of multiple 0.6B checkpoints occurs on the Mac.
