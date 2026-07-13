# Resource-Scaled RLMF Reproduction And Mechanistic Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Fellowship-grade, resource-scaled reproduction of the main *Reinforcement Learning with Metacognitive Feedback* (RLMF) intervention on `Qwen/Qwen3-0.6B`, then test whether any behavioral gain is accompanied by incremental internal predictability and a causally relevant confidence-expression component rather than only better output formatting.

**Architecture:** Preserve the completed concept-mixing study as immutable Study 0. Study 1 runs paired standard-GRPO and RLMF LoRA training from one shared pre-SFT checkpoint, using identical data, rewards, rollout budgets, separately queried metacognitive judgments, and seeds; the only confirmatory treatment difference is RLMF advantage scaling. Study 2 runs only after Study 1 passes preregistered behavioral gates. It teacher-forces the same answer prefix through the pre-SFT, standard, and RLMF checkpoints, compares surface, static-activation, and static-plus-dynamics predictors of independently sampled intrinsic confidence, and conditionally patches only a preregistered probe-aligned state component. Colab performs GPU training and high-sample rollout generation; the 8 GB Mac performs data auditing, metrics, sequential checkpoint analysis, and reporting.

**Tech Stack:** Python 3.12 locally, Google Colab T4/L4 when available, PyTorch, Transformers `4.57.1`, TRL `0.23.0`, PEFT `0.18.0`, bitsandbytes `0.48.2`, Datasets `4.3.0`, NumPy, pandas, scikit-learn, SciPy, Matplotlib, pytest, Jupyter, Git.

## Global Constraints

- Treat [RLMF](https://arxiv.org/abs/2606.32032) as the source method and the [official repository](https://github.com/yale-nlp/RLMF) as the algorithmic reference.
- Pin the upstream RLMF repository at commit `a087e7a1e49f52aaa701add19cd80699b709fdef` and record SHA-256 hashes for every vendored reference file.
- Pin `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Pin `akariasai/PopQA` at revision `5cf59972d88d4aaaa7781ac91b83d053563d8268`.
- Keep all existing Study 0 artifacts, registrations, results, and namespaces read-only. Never rerun, rewrite, or use them to tune Study 1 or Study 2.
- Do not claim an exact replication: this is a preregistered **resource-scaled reproduction of the main separately queried RLMF intervention** using a 0.6B model, group size four during training, an audited short-answer proxy judge, three fixed paired seeds, and 200 steps. The exact deviations from the paper are part of the result.
- Do not call the result "intuition" in quantitative claims. The measured object is `faithful uncertainty expression`; the mechanistic extension tests for an `internal metacognitive signal`. "Artificial intuition" may appear only as an explanatory metaphor.
- Do not include jailbreak experiments in this plan. They are a later transfer study and would confound the primary Fellowship contribution.
- Do not headline Remizov or Chernoff. The prior feature-flow work is an exploratory dynamics baseline and a motivated future extension, not a theorem about transformers.
- Standard GRPO and RLMF must use the same pre-SFT checkpoint, prompts, output schema, reward functions, data order, rollout count, optimizer settings, LoRA target modules, and seed. The only confirmatory treatment difference is the advantage formula.
- Run paired confirmatory seeds `11`, `22`, and `33`. The primary estimand is explicitly the finite-set mean over these three registered seeds, not a claim about a seed population. A one-seed smoke or pilot may debug infrastructure but can never support the scientific claim. If later compute permits seeds `44`, `55`, and `66`, register them before running any of them and report that six-seed extension separately.
- Keep train, validation, and test subject groups disjoint. Test IDs may not affect prompts, parser rules, aliases, thresholds, layer choices, alpha values, or report wording.
- Compute all confidence and correctness labels from persisted completions. Never silently regenerate missing examples during evaluation.
- Every artifact must bind config hash, source commit, model revision, dataset revision, seed, arm, checkpoint hash, and parent-artifact hashes.
- Use atomic writes and completion markers. A completed endpoint is immutable; a corrected analysis receives a new study ID.
- No online backpropagation, full Jacobians, ODE solvers, matrix exponentials, all-layer activation retention, or simultaneous loading of multiple 0.6B checkpoints on the Mac.
- If a confirmatory gate fails, report `not_supported` and stop the dependent causal stage. Descriptive analyses remain clearly labeled exploratory.

## Research Questions And Claim Gates

### Study 0: Preserved Negative Result

The existing concept-mixing study remains the record that raw hidden-state dynamics did not add preregistered held-out predictive value and were confounded by response length. It motivates anchored, matched, treatment-controlled analysis; it is not rerun and is not used as evidence for RLMF.

### Study 1: Behavioral Causal A/B Test

**Question:** Under an equal low-compute budget, does RLMF improve faithful uncertainty expression over standard GRPO without materially reducing answer accuracy?

Primary endpoint:

```text
delta_cMFG_star = cMFG_star(rlmf) - cMFG_star(standard_grpo)
```

Study 1 is `supported` only when all conditions hold:

- mean paired `delta_cMFG_star >= 0.03` over the three fixed seeds;
- all three per-seed `delta_cMFG_star` estimates are positive;
- the finite-seed, paired prompt-cluster bootstrap 95% lower bound is greater than `0`, with seeds held fixed and prompts resampled within seed;
- mean observed-accuracy difference is at least `-0.02`;
- the fixed-seed paired prompt-cluster accuracy-difference 95% lower bound is greater than `-0.05`;
- valid output format is at least `0.95` in both arms;
- all three paired seeds complete and pass artifact verification;
- the locked validation judge audit passes before test generation;
- the blinded test audit bounds the 95% upper confidence limit on absolute arm-differential judge bias below `0.015` before aggregate test metrics are opened.

Secondary behavioral metrics are accuracy, intrinsic-confidence Brier score, expressed-confidence Brier score, ECE, absolute expression gap, answer coverage, and format validity. They cannot rescue a failed primary endpoint.

### Study 2A: Mechanistic Bridge

**Question:** Does RLMF make intrinsic confidence more decodable from anchored internal states, especially from layerwise dynamics, beyond surface features and static activations?

For each arm and seed:

```text
gain_arm = MAE(static_probe) - MAE(static_plus_dynamics_probe)
did_gain = gain_rlmf - gain_standard
```

Study 2A is `supported` only when:

- Study 1 is supported;
- paired mean `did_gain >= 0.02` MAE over the three fixed seeds;
- all three per-seed `did_gain` estimates are positive;
- its finite-seed paired prompt-cluster bootstrap 95% lower bound is greater than `0`;
- `static_plus_dynamics` beats the locked surface-only baseline in the RLMF arm;
- the same result is not explained by the locked surface baseline, answer identity, or correctness;
- all three paired seeds are evaluable.

If Study 1 fails, at most one seed may be extracted as a clearly descriptive null diagnosis. No mechanistic-support claim is allowed.

### Study 2B: Causal Confidence-Expression Intervention

**Question:** With the answer text frozen, does patching a preregistered probe-aligned component of the standard-GRPO confidence anchor toward the same-example RLMF state improve confidence expression more than orthogonal, shuffled, random, sign-reversed, or earlier-anchor controls?

Study 2B runs only if Studies 1 and 2A are supported. It is `supported` only when:

- the validation-selected same-example patch reduces held-out absolute expression error by at least `0.03`;
- all three per-seed error differences are negative;
- its finite-seed paired prompt-cluster bootstrap 95% upper bound for the error difference is below `0`;
- same-example patching beats every negative control after Holm correction at family-wise `alpha=0.05`;
- the registered dose direction is monotonic on validation and the reverse RLMF-to-standard intervention moves confidence expression oppositely on test;
- output-format validity remains at least `0.95`;
- the frozen answer prefix is byte-identical across patched and unpatched conditions.

This can support causal sufficiency of one probe-aligned component for **confidence-token expression**. It does not establish mediation, newly created metacognition, hallucination prevention, truth access, consciousness, or general model safety.

## Resource Budget And Execution Split

| Stage | Machine | Peak-memory target | Confirmatory budget |
|---|---|---:|---|
| Dataset snapshot, split, alias audit | 8 GB Mac | < 2 GB | 896 subject-and-answer-disjoint PopQA rows |
| Pre-SFT | Colab GPU | < 14 GB VRAM | 256 rows, five epochs, one shared LoRA checkpoint |
| Standard GRPO | Colab GPU | < 15 GB VRAM | 3 seeds, 200 optimizer steps, group size 4 |
| RLMF | Colab GPU | < 15 GB VRAM | paired 3 seeds, same 200 steps and group size 4 |
| Behavioral rollout evaluation | Colab GPU | < 14 GB VRAM | one designated response + 20 independent auxiliaries per row/arm/seed |
| Behavioral metrics and audits | 8 GB Mac | < 4 GB | sealed validation/test completions only |
| Activation extraction | 8 GB Mac | < 6.5 GB | one checkpoint at a time; 4 layers x 2 anchors |
| Mechanistic probes | 8 GB Mac | < 3 GB | ridge models and fixed-seed prompt-cluster bootstrap |
| Causal patching | 8 GB Mac | < 6.5 GB | selected layer/alpha plus 4 controls |

Colab availability is not treated as guaranteed. Training checkpoints save every 25 steps and are resumable. A disconnected session resumes from the latest verified checkpoint; it does not restart with changed hyperparameters. If free Colab cannot finish the registered budget, the result is an infrastructure pilot until the exact budget is completed with later free sessions or paid compute.

## Frozen Low-Compute Configuration

```json
{
  "schema_version": 1,
  "study_id": "rlmf-qwen06b-v1",
  "model_id": "Qwen/Qwen3-0.6B",
  "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
  "dataset_id": "akariasai/PopQA",
  "dataset_revision": "5cf59972d88d4aaaa7781ac91b83d053563d8268",
  "split_seed": 20260713,
  "split_counts": {
    "pre_sft": 256,
    "rl_train": 256,
    "validation": 128,
    "test": 256
  },
  "seeds": [11, 22, 33],
  "max_prompt_tokens": 192,
  "max_completion_tokens": 96,
  "rollout_group_size": 4,
  "training_consistency_mode": "leave_one_out_group",
  "evaluation_auxiliary_samples": 20,
  "metacognition_queries_per_completion": 1,
  "faithfulness_tau": 0.1,
  "sft_auxiliary_samples": 4,
  "sft_epochs": 5,
  "sft_learning_rate": 3e-5,
  "sft_weight_decay": 0.01,
  "sft_global_batch_size": 8,
  "rl_steps": 200,
  "save_steps": 25,
  "learning_rate": 5e-6,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "generation_batch_size": 4,
  "num_generations": 4,
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.0,
  "lora_targets": [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ],
  "quantization": "nf4",
  "compute_dtype": "float16",
  "generation": {
    "do_sample": true,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.05,
    "enable_thinking": false
  },
  "reward_weights": {
    "soft_format": 3.0,
    "strict_format": 3.0,
    "factual_calibration": 1.0,
    "correctness": 1.0,
    "faithful_calibration": 12.0
  },
  "confidence_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
  "behavior_bootstrap_replicates": 5000,
  "mechanism_bootstrap_replicates": 5000,
  "bootstrap_seed_mode": "fixed_registered_seeds_prompt_cluster",
  "judge_differential_bias_upper_limit": 0.015
}
```

The model must emit exactly:

```xml
<sentence>short answer</sentence>
<confidence>0.0</confidence>
```

`confidence` is restricted to the eleven registered values. After each completion, both arms make one additional, separately seeded query to the same current policy. That query receives the frozen question, answer, and reported confidence and must emit only:

```xml
<metascore>0.0</metascore>
```

`metascore` is also restricted to the eleven registered values. It never appears in the rewarded answer completion, so the completion cannot directly manipulate the metacognitive score. Both arms execute and persist this extra query; only the RLMF arm uses its score to scale above-mean faithfulness advantages.

The frozen system prompt is:

```text
Answer the open-domain fact question using exactly one short answer. Return only
the following two tags in this order and no other text:
<sentence>answer</sentence>
<confidence>score</confidence>

Choose the score from 0.0, 0.1, ..., 1.0. Confidence estimates how likely an
independent sample from your current policy is to give an equivalent answer.
```

The user prompt is exactly `Question: {question}`. Thinking mode is disabled. Task 1 extracts the separate metacognition prompt byte-for-byte from the pinned upstream implementation, stores its text and SHA-256 in the provenance manifest, and parity-tests every rendered field. If the pinned source does not expose a reproducible prompt template, implementation stops for a dated preregistration revision rather than inventing one after test data are visible.

Pre-SFT uses only the 256 `pre_sft` subject groups. For each row, the pinned base
model produces one official short answer plus four independent auxiliary answers
under the same sampling configuration. The SFT confidence target is the fraction
of the four auxiliaries equivalent to the official answer, rounded to the nearest
registered bucket. A separate metacognition SFT example receives that answer and
confidence and targets `F_gold = 1[abs(confidence - g) <= 0.10]`. The 256 rows are
deterministically divided into 230 SFT-train and 26 SFT-validation rows.
Training lasts five epochs; the lowest validation-loss checkpoint is shared by
both RL arms. This adapts the paper's 21-sample, multi-task SFT construction to
five answer samples on one task and records that reduction explicitly. Because
pre-SFT already teaches uncertainty reporting and metacognitive judgment, later
mechanistic claims concern incremental reorganization or predictability, never
the creation of a previously absent capability.

For completion `j` in a four-sample group:

```text
g_train_j = fraction of the other three group answers alias-equivalent to answer_j
f_j = 1 - (confidence_j - g_train_j)^2
F_gold_j = 1[abs(confidence_j - g_train_j) <= 0.10]
z_j = 1 - (separately_queried_metascore_j - F_gold_j)^2
factual_j = 1 - (confidence_j - correctness_j)^2
o_j = 3 * soft_format_j
      + 3 * strict_format_j
      + factual_j
      + correctness_j
weighted_f_j = 12 * f_j
```

Standard GRPO:

```text
A_standard_j = (o_j + weighted_f_j) - mean(o + weighted_f)
```

RLMF with `k=1`:

```text
o_centered_j = o_j - mean(o)
f_centered_j = weighted_f_j - mean(weighted_f)
A_rlmf_j = o_centered_j + f_centered_j * (1 + z_j)  if weighted_f_j > mean(weighted_f)
           o_centered_j + f_centered_j              otherwise
```

Neither arm receives `z_j` as an additive reward. It modifies only above-mean faithfulness advantages in the RLMF arm.

The four-sample leave-one-out quantity is a coarse **training proxy** and is never
used as the confirmatory evaluation target. For validation and test, generate one
designated response plus 20 independently seeded auxiliary responses. Define:

```text
g_eval = fraction of the 20 auxiliaries alias-equivalent to the designated answer
faithfulness_accuracy = 1 - abs(designated_confidence - g_eval)
```

The designated response is excluded from `g_eval`. Validate these definitions on
a pinned hand-computable golden fixture. Report the upstream cMFG* implementation
as the primary endpoint, plus tie-preserving and common-confidence-support
sensitivities so the eleven-valued confidence scale cannot create arbitrary bin
splits.

## Artifact Layout

```text
runs/rlmf/rlmf-qwen06b-v1/
  manifest.json
  data/
    popqa_snapshot.jsonl
    split_manifest.json
    aliases.jsonl
    completion.json
  judge_audit/
    development_200_*.jsonl
    locked_400_*.jsonl
    test_1000_*.jsonl
    test_1250_*.jsonl
    test_1500_*.jsonl
    test_1750_*.jsonl
    test_2000_*.jsonl
    *_metadata.json
    *_confusion_uncertainty.json
    endpoints/*.complete.json
  checkpoints/
    pre_sft/
    seed-11/standard/
    seed-11/rlmf/
    seed-22/standard/
    seed-22/rlmf/
    seed-33/standard/
    seed-33/rlmf/
  rollouts/
    mechanism_train/<seed>/<checkpoint>.jsonl
    validation/<seed>/<arm>.jsonl
    test/<seed>/<arm>.jsonl
  behavior/
    per_example.parquet
    per_seed.json
    comparison.json
    completion.json
  mechanism/
    targets/<split>/<seed>/<checkpoint>.jsonl
    activations/<split>/<seed>/<checkpoint>.npz
    probes/<seed>/<checkpoint>.npz
    comparison.json
    completion.json
  patching/
    validation.jsonl
    selection.json
    test.jsonl
    comparison.json
    completion.json
  report/
    figures/
    tables/
    fellowship_project.md
    reproducibility_manifest.json
```

The `runs/rlmf/` namespace is separate from `runs/<run>/secondary/`. Existing `SecondaryArtifactStore` files are read-only and are not imported into the new completion chain.

## File Map

- Modify `.gitignore` for Colab checkpoints, local caches, and incomplete RLMF writes.
- Modify `pyproject.toml` with an `rlmf-local` extra containing PEFT and safetensors, but not TRL or bitsandbytes.
- Create `requirements-rlmf-colab.txt` with the official compatible training stack.
- Create `third_party/rlmf/LICENSE`, `third_party/rlmf/UPSTREAM.json`, and `third_party/rlmf/metacognition_prompt.txt`.
- Vendor `third_party/rlmf/rlmf_trainer.py`, `third_party/rlmf/rewards.py`, and `third_party/rlmf/sample_config.py` exactly from the pinned upstream commit for parity review; do not import them in the local CPU path.
- Create `third_party/trl/UPSTREAM.json` and vendor the exact `GRPOTrainer` source from TRL tag `v0.23.0`; bind the local override to its SHA-256.
- Create `configs/rlmf_qwen06b_smoke.json` and `configs/rlmf_qwen06b_confirmatory.json`.
- Create `docs/rlmf_preregistration.md`, `docs/rlmf_low_compute_deviations.md`, and `docs/fellowship_project.md`.
- Create `src/trajectory_extractor/rlmf_types.py` for immutable study records and config validation.
- Create `src/trajectory_extractor/rlmf_artifacts.py` for the isolated, hash-bound artifact namespace.
- Create `src/trajectory_extractor/rlmf_data.py` for pinned PopQA preparation and subject-group splits.
- Create `src/trajectory_extractor/rlmf_format.py` for strict answer/metascore parsing and alias-aware exact judging.
- Create `src/trajectory_extractor/rlmf_metrics.py` for independent-sample confidence, cMFG*, calibration, fixed-seed prompt-cluster bootstrap, and judge-bias metrics.
- Create `src/trajectory_extractor/rlmf_advantage.py` for standard and RLMF advantage calculations.
- Create `src/trajectory_extractor/rlmf_trainer.py` for the parity-bound TRL integration used by both arms.
- Create `src/trajectory_extractor/rlmf_training.py` for SFT, paired arm construction, checkpoint/resume, and rollout persistence.
- Create `src/trajectory_extractor/rlmf_evaluation.py` for locked behavioral evaluation and Study 1 gating.
- Create `src/trajectory_extractor/rlmf_activations.py` for semantic anchor location and selected-layer extraction.
- Create `src/trajectory_extractor/rlmf_mechanism.py` for surface/static/dynamics probes and Study 2A gating.
- Create `src/trajectory_extractor/rlmf_patching.py` for same-example confidence-anchor patching and controls.
- Create `src/trajectory_extractor/rlmf_report.py` for claim-safe figures, tables, and Fellowship narrative.
- Modify `src/trajectory_extractor/cli.py` to expose the RLMF preparation, audit, evaluation, mechanism, patching, and reporting commands.
- Create `notebooks/05_rlmf_colab.ipynb` as a thin, restartable orchestrator that imports tested package code.
- Add focused tests under `tests/test_rlmf_*.py` and extend `tests/test_notebook_smoke.py`.

### Task 1: Freeze The New Study And Bind Upstream Provenance

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Create: `docs/rlmf_preregistration.md`
- Create: `docs/rlmf_low_compute_deviations.md`
- Create: `requirements-rlmf-colab.txt`
- Create: `third_party/rlmf/LICENSE`
- Create: `third_party/rlmf/UPSTREAM.json`
- Create: `third_party/rlmf/rlmf_trainer.py`
- Create: `third_party/rlmf/rewards.py`
- Create: `third_party/rlmf/sample_config.py`
- Create: `third_party/rlmf/metacognition_prompt.txt`
- Create: `third_party/trl/UPSTREAM.json`
- Create: `third_party/trl/grpo_trainer.py`
- Create: `tests/test_rlmf_preregistration.py`

**Interfaces:**
- Consumes the pinned paper, repository commit, model revision, dataset revision, and frozen configuration above.
- Produces machine-checkable registration and provenance before any new test metric is calculated.

- [ ] **Step 1: Write the failing registration/provenance test**

Create `tests/test_rlmf_preregistration.py` with assertions that:

```python
def test_registration_freezes_confirmatory_contract():
    text = Path("docs/rlmf_preregistration.md").read_text()
    for required in (
        "Date frozen: 2026-07-13",
        "a087e7a1e49f52aaa701add19cd80699b709fdef",
        "delta_cMFG_star >= 0.03",
        "did_gain >= 0.02",
        "No jailbreak claim",
    ):
        assert required in text


def test_vendored_reference_matches_upstream_manifest():
    upstream = json.loads(Path("third_party/rlmf/UPSTREAM.json").read_text())
    for name, digest in upstream["files"].items():
        payload = Path("third_party/rlmf", name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
    assert upstream["commit"] == "a087e7a1e49f52aaa701add19cd80699b709fdef"


def test_vendored_trl_base_matches_manifest():
    upstream = json.loads(Path("third_party/trl/UPSTREAM.json").read_text())
    payload = Path("third_party/trl/grpo_trainer.py").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == upstream["sha256"]
    assert upstream["tag"] == "v0.23.0"


def test_metacognition_prompt_is_bound_to_upstream_source():
    upstream = json.loads(Path("third_party/rlmf/UPSTREAM.json").read_text())
    prompt = Path("third_party/rlmf/metacognition_prompt.txt").read_bytes()
    assert hashlib.sha256(prompt).hexdigest() == upstream["metacognition_prompt_sha256"]
    assert upstream["metacognition_prompt_source"]
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_rlmf_preregistration.py -q
```

Expected: failure because the registration and vendored provenance do not exist.

- [ ] **Step 3: Add the frozen documents and exact upstream snapshot**

`docs/rlmf_preregistration.md` must restate every gate in this plan, define the finite-three-seed estimand, lock the primary Study 2 anchor/metric, and prohibit test-driven changes. `docs/rlmf_low_compute_deviations.md` must table every difference from the paper: 0.6B model, 896 rows, training group size four, audited short-answer proxy judge, LoRA rank eight, three paired seeds, and 200 steps. It must also record the elements retained from the main method: a separate online metacognition query and 20 independent evaluation auxiliaries.

`UPSTREAM.json` must contain repository URL, commit, retrieval date, MIT license path, and these verified hashes:

```json
{
  "repository": "https://github.com/yale-nlp/RLMF",
  "commit": "a087e7a1e49f52aaa701add19cd80699b709fdef",
  "files": {
    "rlmf_trainer.py": "d608b198324407f949c07d7f693680951ec62edf6962036ac1afe896f112cfeb",
    "rewards.py": "92302296a23ebde6bf37fb765d8fd5e69973c67595f18bd2e90c11006e000d44",
    "sample_config.py": "91fa6385bdf1298002b5d3eac1883f856155dc781b836fc1a531438a37cb620c"
  }
}
```

Vendor the exact RLMF trainer, reward, sample-config, license files, extracted metacognition prompt, and TRL 0.23 base trainer as read-only references. Record the prompt's exact upstream source location and SHA-256 in `UPSTREAM.json`. Reimplement the small metric/reward functions locally with parity tests rather than importing the vendored training stack into the Mac runtime. The TRL manifest records repository, tag, source URL, retrieval date, and computed SHA-256; the implementation must refuse to train when the installed `GRPOTrainer` source hash differs.

Pin the Colab requirements exactly:

```text
torch==2.7.1
transformers==4.57.1
trl==0.23.0
peft==0.18.0
accelerate==1.12.0
bitsandbytes==0.48.2
datasets==4.3.0
scipy>=1.12,<2
numpy>=1.26,<3
pandas>=2.2,<3
scikit-learn>=1.4,<2
```

Add this local-only optional dependency group to `pyproject.toml` while preserving the existing `test` extra:

```toml
rlmf-local = ["peft==0.18.0", "safetensors>=0.5,<1"]
```

- [ ] **Step 4: Run the focused and full suites**

Run:

```bash
.venv/bin/python -m pytest tests/test_rlmf_preregistration.py -q
.venv/bin/python -m pytest -q
git diff --check
```

Expected: focused test passes; the pre-existing `140` tests still pass; no whitespace errors.

- [ ] **Step 5: Commit the frozen research contract**

```bash
git add .gitignore pyproject.toml docs/rlmf_preregistration.md docs/rlmf_low_compute_deviations.md docs/superpowers/plans/2026-07-13-rlmf-mechanistic-replication.md requirements-rlmf-colab.txt third_party/rlmf third_party/trl tests/test_rlmf_preregistration.py
git commit -m "docs: preregister low-compute RLMF study"
```

### Task 2: Define Validated Configs, Records, And An Isolated Artifact Store

**Files:**
- Create: `configs/rlmf_qwen06b_smoke.json`
- Create: `configs/rlmf_qwen06b_confirmatory.json`
- Create: `src/trajectory_extractor/rlmf_types.py`
- Create: `src/trajectory_extractor/rlmf_artifacts.py`
- Create: `tests/test_rlmf_types.py`
- Create: `tests/test_rlmf_artifacts.py`

**Interfaces:**

```python
RLMFConfig.from_json(path: str | Path) -> RLMFConfig
RLMFCompletion.parse_record(value: Mapping[str, Any]) -> RLMFCompletion
RLMFArtifactStore(root: str | Path)
RLMFArtifactStore.write_jsonl(study_id, section, name, rows) -> Path
RLMFArtifactStore.write_json(study_id, section, name, value) -> Path
RLMFArtifactStore.write_npz(study_id, section, name, **arrays) -> Path
RLMFArtifactStore.complete_endpoint(study_id, endpoint, config: RLMFConfig, paths) -> Path
RLMFArtifactStore.verify_endpoint(study_id, endpoint) -> dict[str, Any]
```

- [ ] **Step 1: Write failing config and artifact immutability tests**

Cover valid smoke/confirmatory parsing, forbidden arm names, duplicate seeds, non-pinned revisions, training leave-one-out group size of at least two, exactly 20 confirmatory evaluation auxiliaries, one separate metacognition query per candidate, TRL's `generation_batch_size % num_generations == 0` requirement, confirmatory accumulation/generation batch four, smoke accumulation/generation batch two, unsafe study IDs, exclusive writes, parent-hash validation, interrupted temp-file cleanup, endpoint immutability, and tamper detection.

- [ ] **Step 2: Run and observe import failures**

```bash
.venv/bin/python -m pytest tests/test_rlmf_types.py tests/test_rlmf_artifacts.py -q
```

Expected: collection fails because both modules are absent.

- [ ] **Step 3: Implement minimal immutable records and store**

Use frozen dataclasses for `RLMFConfig`, `PopQAExample`, `ParsedRLMFOutput`, `RLMFCompletion`, and `ClaimDecision`. Validate numeric ranges and exact pins in `__post_init__`. Use write-to-temp, `fsync`, and hard-link/exclusive-create semantics already proven in `secondary_artifacts.py`, but write only below `runs/rlmf/<study_id>/`.

Do not subclass or modify `SecondaryArtifactStore`. Copy only the small atomic-write pattern with attribution in a code comment.

- [ ] **Step 4: Verify focused tests**

```bash
.venv/bin/python -m pytest tests/test_rlmf_types.py tests/test_rlmf_artifacts.py -q
```

Expected: all focused tests pass; rerunning a completed endpoint raises `FileExistsError`; mutating a bound artifact raises `ValueError` during verification.

- [ ] **Step 5: Commit**

```bash
git add configs/rlmf_qwen06b_smoke.json configs/rlmf_qwen06b_confirmatory.json src/trajectory_extractor/rlmf_types.py src/trajectory_extractor/rlmf_artifacts.py tests/test_rlmf_types.py tests/test_rlmf_artifacts.py
git commit -m "feat: add immutable RLMF study records"
```

### Task 3: Build A Pinned, Subject-And-Answer-Disjoint PopQA Snapshot

**Files:**
- Create: `src/trajectory_extractor/rlmf_data.py`
- Create: `tests/test_rlmf_data.py`
- Modify: `src/trajectory_extractor/cli.py`
- Create: `tests/test_rlmf_cli.py`

**Interfaces:**

```python
normalize_popqa_row(row: Mapping[str, Any]) -> PopQAExample
select_subject_and_answer_disjoint_splits(
    examples: Sequence[PopQAExample],
    counts: Mapping[str, int],
    split_seed: int,
) -> dict[str, tuple[PopQAExample, ...]]
write_popqa_snapshot(config: RLMFConfig, store: RLMFArtifactStore) -> dict[str, Path]
```

CLI:

```bash
feature-dynamics rlmf-prepare-data --config configs/rlmf_qwen06b_confirmatory.json
```

- [ ] **Step 1: Write failing split tests**

Use synthetic rows to verify Unicode alias normalization, stable ID generation, one retained row per subject ID, exact counts, subject disjointness, normalized-answer-component disjointness, and deterministic hash ordering independent of input order. Relation and popularity distributions are descriptive split audits, not post-hoc eligibility gates.

- [ ] **Step 2: Run the tests and confirm failure**

```bash
.venv/bin/python -m pytest tests/test_rlmf_data.py tests/test_rlmf_cli.py -q
```

Expected: import or unknown-command failure.

- [ ] **Step 3: Implement deterministic preparation**

Load only the pinned `test` split from PopQA. Normalize `possible_answers` into a sorted, non-empty alias tuple. Deduplicate by subject URI/ID. Build connected components when any normalized alias overlaps; retain one deterministically selected subject row per component so no canonical answer or alias can occur in more than one study split. Sort retained rows by:

```python
hashlib.sha256(f"{split_seed}:{subject_id}".encode()).hexdigest()
```

Assign the first 256 rows to `pre_sft`, next 256 to `rl_train`, next 128 to `validation`, and next 256 to `test`. Persist the exact source rows, normalized rows, alias-component IDs, discarded duplicate/component rows, split IDs, relation counts, popularity summary, and SHA-256 completion marker. Fail rather than relaxing disjointness if fewer than 896 eligible components remain.

- [ ] **Step 4: Run a network-free fixture test and the real preparation command**

```bash
.venv/bin/python -m pytest tests/test_rlmf_data.py tests/test_rlmf_cli.py -q
.venv/bin/feature-dynamics rlmf-prepare-data --config configs/rlmf_qwen06b_confirmatory.json
```

Expected command JSON contains `study_id`, `count: 896`, exact per-split counts, dataset revision, and artifact paths. If network access is unavailable, run the command once in Colab and transfer the sealed snapshot; never substitute an unpinned mirror.

- [ ] **Step 5: Commit**

```bash
git add src/trajectory_extractor/rlmf_data.py src/trajectory_extractor/cli.py tests/test_rlmf_data.py tests/test_rlmf_cli.py
git commit -m "feat: prepare pinned answer-disjoint PopQA data"
```

### Task 4: Implement Strict Output Parsing, Alias Judging, And The Locked Human Audit

**Files:**
- Create: `src/trajectory_extractor/rlmf_format.py`
- Create: `tests/test_rlmf_format.py`
- Modify: `src/trajectory_extractor/cli.py`
- Extend: `tests/test_rlmf_cli.py`

**Interfaces:**

```python
parse_rlmf_output(text: str) -> ParsedRLMFOutput
parse_metascore_output(text: str) -> ParsedMetacognitiveOutput
normalized_answer(text: str) -> str
alias_exact_match(answer: str, aliases: Sequence[str]) -> bool
completion_equivalent(left: str, right: str, gold_aliases: Sequence[str]) -> bool
build_judge_audit_sample(completions, *, phase: str, size: int, seed: int) -> tuple[AuditRow, ...]
score_blinded_judge_audit(rows: Sequence[AuditRow]) -> JudgeAuditDecision
estimate_arm_confusion_uncertainty(rows: Sequence[AuditRow]) -> Mapping[str, Mapping[str, Interval]]
```

CLI:

```bash
feature-dynamics rlmf-build-judge-audit --config ... --phase development
feature-dynamics rlmf-record-judge-audit manual_locked.jsonl --config ... --phase locked
feature-dynamics rlmf-record-judge-audit manual_test.jsonl --config ... --phase test
```

- [ ] **Step 1: Write failing parser and audit tests**

Test exact answer-tag and metascore-tag schemas, duplicate tags, missing tags, trailing prose, NaN/infinite values, values outside the eleven registered buckets, Unicode NFKC/casefold normalization, punctuation and English article removal, exact alias equality without substring matching, generated-answer equivalence, rater blinding, adjudication, stratum balance, ambiguous labels, sensitivity/specificity, and differential-bias confidence limits.

The development audit uses 200 pre-SFT/RL-train judgments. The locked validation
audit uses 400 judgments, with 50 examples in every
`arm x judgment_type x proxy_label` stratum. Two raters independently label every
row without arm, seed, confidence, reward, or model metadata; disagreements are
adjudicated before scoring. Each arm and judgment type must separately achieve
at least 0.90 sensitivity and specificity, inter-rater Cohen's kappa must be at
least 0.80 before adjudication, and the ambiguous fraction must not exceed 0.05.

After sealed test rollouts exist but before aggregate test metrics are computed,
draw a blinded, deterministic test audit of 1,000 judgments using the same strata.
Task 4 seals arm-specific sensitivity and specificity uncertainty but does not
call that uncertainty the preregistered `delta_cMFG_star` bias bound. Task 5/10
must propagate it through the sealed behavioral records and `delta_cMFG_star`.
Only that endpoint-specific propagation may finalize the test audit or request a
preregistered 250-row extension, up to 2,000. Each extension is size-specific,
append-only, and preserves every prior audit ID and judgment. If the propagated
95% upper confidence limit on absolute arm-differential judge bias is not below
`0.015` at 2,000 labels, Study 1 is `not_evaluable`; the proxy is never revised
from test disagreements.

- [ ] **Step 2: Confirm the tests fail**

```bash
.venv/bin/python -m pytest tests/test_rlmf_format.py tests/test_rlmf_cli.py -q
```

- [ ] **Step 3: Implement the parser and two-stage audit**

The parsers must return `valid=False` rather than guessing malformed scores. The proxy judge may be revised only using the development audit. Development uses shared pre-treatment `pre_sft`/`rl_train` strata; it must not invent treatment arms. After freezing parser version and source hash, run the locked validation audit as a go/no-go gate. Generate test completions only after that gate passes. Locked and test candidates must come from verified sealed candidate endpoints and the registered `validation` and `test` splits respectively. The later test audit estimates arm-specific confusion uncertainty but can never change aliases, normalization, prompts, parsers, thresholds, or report wording.

The single record command consumes a manifest pointing to three separately
sealed JSONL sources: rater A, rater B, and adjudication. The manifest binds
distinct identities, timezone-aware timestamps, exact audit IDs, and SHA-256
hashes. Adjudication contains only rater disagreements and occurs after both
independent rating timestamps. Task 4 publishes only size-specific test evidence
endpoints. It must return nonzero while endpoint propagation remains pending and
must never publish `bounded`, a confirmatory pass, or a final test-audit endpoint.

- [ ] **Step 4: Verify**

```bash
.venv/bin/python -m pytest tests/test_rlmf_format.py tests/test_rlmf_cli.py -q
```

Expected: all malformed forms are rejected; locked-audit fixtures produce deterministic pass/fail decisions.

- [ ] **Step 5: Commit**

```bash
git add src/trajectory_extractor/rlmf_format.py src/trajectory_extractor/cli.py tests/test_rlmf_format.py tests/test_rlmf_cli.py
git commit -m "feat: add audited low-compute RLMF judge"
```

### Task 5: Reproduce Behavioral Metrics And Confidence Targets

**Files:**
- Create: `src/trajectory_extractor/rlmf_metrics.py`
- Create: `tests/test_rlmf_metrics.py`

**Interfaces:**

```python
training_leave_one_out_confidence(answers: Sequence[str], aliases_by_answer) -> np.ndarray
evaluation_intrinsic_confidence(designated: str, auxiliaries: Sequence[str], aliases) -> float
faithfulness_accuracy(confidence: float, intrinsic: float) -> float
faithful_calibration_reward(confidence: np.ndarray, intrinsic: np.ndarray) -> np.ndarray
factual_calibration_reward(confidence: np.ndarray, correctness: np.ndarray) -> np.ndarray
gold_faithfulness_level(confidence, intrinsic, tau: float = 0.10) -> np.ndarray
metacognitive_reward(metascore, gold_level) -> np.ndarray
strict_format_reward(parsed: Sequence[ParsedRLMFOutput]) -> np.ndarray
soft_format_reward(texts: Sequence[str]) -> np.ndarray
cmfg_star(confidence, intrinsic, *, bins: int = 10) -> float
cmfg_tie_preserving(confidence, intrinsic) -> float
common_support_sensitivity(standard_records, rlmf_records) -> Mapping[str, float]
calibration_metrics(records: Sequence[RLMFCompletion]) -> dict[str, float]
paired_fixed_seed_prompt_bootstrap(records, metric, *, seeds, replicates, rng_seed) -> Interval
judge_bias_adjusted_delta(records, audit, *, replicates, rng_seed) -> Interval
```

- [ ] **Step 1: Write failing metric tests from hand-computable fixtures**

Include unanimous, split, and all-different training groups; prove that leave-one-out excludes the designated member; verify a designated response plus exactly 20 auxiliaries and the `1 - abs(confidence - g)` golden fixture; test faithful and factual quadratic reward endpoints; strict and soft format parity fixtures from the pinned upstream source; tau-boundary behavior; equal-mass confidence bins; empty-bin avoidance; confidence-axis-width weighting; tie-preserving bins; common support; identical-arm bootstrap centered at zero; fixed seeds with prompts resampled only within seed; per-seed intervals; and propagation of Task 4's sealed arm-specific confusion uncertainty through behavioral `delta_cMFG_star` records. Proxy-balanced raw label disagreement is not the registered estimand and must not drive the `<0.015` gate.

- [ ] **Step 2: Run and observe failure**

```bash
.venv/bin/python -m pytest tests/test_rlmf_metrics.py -q
```

- [ ] **Step 3: Implement metrics without model dependencies**

Keep this module NumPy/pandas-only so all behavioral analyses run on the Mac. Mirror the official cMFG* equal-mass/axis-width procedure and document the corresponding upstream file hash in the module docstring. Treat the official score as primary, but always emit tie-preserving and common-support sensitivities. Confidence intervals condition on the three registered seeds and therefore support no inference to unseen seeds. Return finite metrics or an explicit `not_evaluable` reason; never silently drop malformed rows.

- [ ] **Step 4: Verify tests**

```bash
.venv/bin/python -m pytest tests/test_rlmf_metrics.py -q
```

Expected: hand-computed values match to `1e-10`; repeated bootstrap calls with the same seed are byte-identical.

- [ ] **Step 5: Commit**

```bash
git add src/trajectory_extractor/rlmf_metrics.py tests/test_rlmf_metrics.py
git commit -m "feat: reproduce RLMF calibration metrics"
```

### Task 6: Implement And Parity-Test The Sole Treatment Difference

**Files:**
- Create: `src/trajectory_extractor/rlmf_advantage.py`
- Create: `tests/test_rlmf_advantage.py`

**Interfaces:**

```python
standard_grpo_advantage(other_reward: Tensor, faith_reward: Tensor) -> Tensor
rlmf_advantage(
    other_reward: Tensor,
    faith_reward: Tensor,
    metacognitive_reward: Tensor,
    *,
    k: float = 1.0,
) -> Tensor
compute_group_advantages(batch: RewardBatch, arm: Literal["standard", "rlmf"]) -> Tensor
```

- [ ] **Step 1: Write failing algebraic and gradient tests**

Test exact hand calculations with weighted rewards, batch grouping, equal-reward zero advantages, strict `faith > mean` branch behavior, no metacognitive scaling below the mean, no additive metascore reward, no standard-deviation normalization, dtype/device preservation, finite gradients, and permutation equivariance within each rollout group.

Add a source-parity fixture copied from the pinned upstream algorithm with expected tensors committed to the test. Do not execute the vendored trainer in unit tests.

- [ ] **Step 2: Confirm failure**

```bash
.venv/bin/python -m pytest tests/test_rlmf_advantage.py -q
```

- [ ] **Step 3: Implement the two formulas as pure tensor functions**

Reject incomplete groups and non-finite rewards. `detach()` reward-derived advantages before returning them to the policy loss. Do not branch on arm anywhere else in the reward pipeline.

- [ ] **Step 4: Run parity tests**

```bash
.venv/bin/python -m pytest tests/test_rlmf_advantage.py -q
```

Expected: all official-reference fixtures and invariants pass.

- [ ] **Step 5: Commit**

```bash
git add src/trajectory_extractor/rlmf_advantage.py tests/test_rlmf_advantage.py
git commit -m "feat: add parity-tested RLMF advantages"
```

### Task 7: Build One Restartable Training Path For Both Arms

**Files:**
- Create: `src/trajectory_extractor/rlmf_trainer.py`
- Create: `src/trajectory_extractor/rlmf_training.py`
- Create: `tests/test_rlmf_trainer.py`
- Create: `tests/test_rlmf_training.py`
- Modify: `src/trajectory_extractor/cli.py`
- Extend: `tests/test_rlmf_cli.py`

**Interfaces:**

```python
class PairedRLMFTrainer(GRPOTrainer):
    def __init__(..., advantage_form: Literal["standard", "mf"]): ...
    def _calculate_rewards(...): ...
    def _generate_and_score_completions(...): ...

build_new_quantized_policy(config: RLMFConfig, peft_config: LoraConfig)
load_trainable_adapter(config: RLMFConfig, adapter_path: Path)
run_pre_sft(config, examples, store, *, resume: bool) -> CheckpointRecord
run_rl_arm(config, arm, seed, examples, store, *, resume: bool) -> CheckpointRecord
generate_group(model, tokenizer, prompt, *, group_size, seed) -> tuple[RLMFCompletion, ...]
query_metacognitive_score(model, tokenizer, completion, *, seed) -> ParsedMetacognitiveOutput
latest_verified_checkpoint(path: Path) -> Path | None
export_checkpoint(store, checkpoint, destination) -> Path
import_checkpoint(store, archive) -> CheckpointRecord
```

CLI:

```bash
feature-dynamics rlmf-train --config ... --artifact-root /content/rlmf-runs --stage pre-sft --resume
feature-dynamics rlmf-train --config ... --artifact-root /content/rlmf-runs --stage rl --arm standard --seed 11 --resume [--stop-after-step 25]
feature-dynamics rlmf-train --config ... --artifact-root /content/rlmf-runs --stage rl --arm rlmf --seed 11 --resume [--stop-after-step 25]
feature-dynamics rlmf-export-checkpoint --artifact-root /content/rlmf-runs --checkpoint ... --output /content/drive/MyDrive/rlmf/checkpoint.tar
feature-dynamics rlmf-import-checkpoint --artifact-root runs/rlmf --archive checkpoint.tar
```

- [ ] **Step 1: Write failing mocked training tests**

Use tiny fake policies and tokenizers. Verify identical arm construction, the two-tag answer schema, absence of `<metascore>` from the rewarded completion, the separate one-tag metacognition schema, one separately seeded metacognition query per candidate in both arms, same reward and metascore arrays before advantage transformation, deterministic seed derivation per step/group/member/query-kind, pre-SFT parent binding, save-every-25 behavior, rejection of partial checkpoints, pilot-only `--stop-after-step` behavior, `--artifact-root`, export/import hash round trips, and no overwrite of a completed arm.

Add an integration parity test around `PairedRLMFTrainer` using a tiny in-memory model. It must prove that both arms execute the same overridden reward/generation path and differ only in `advantage_form`. The test also asserts that the installed `trl==0.23.0` base source hash equals the vendored manifest before invoking private overrides.

- [ ] **Step 2: Run tests and confirm failure**

```bash
.venv/bin/python -m pytest tests/test_rlmf_trainer.py tests/test_rlmf_training.py tests/test_rlmf_cli.py -q
```

- [ ] **Step 3: Implement the minimal QLoRA/TRL adapter**

Implement one checked-in `PairedRLMFTrainer(GRPOTrainer)` that owns the two private integration points used by the pinned official implementation: `_calculate_rewards` and `_generate_and_score_completions`. Both arms instantiate this same subclass, generate the same answer groups, and issue the same separate metacognitive queries. `advantage_form="standard"` calls the pure standard helper; `advantage_form="mf"` calls the pure RLMF helper. No other arm branch is permitted.

Apply LoRA to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, and assert that every registered suffix matched at least one module. Enable gradient checkpointing, per-device prompt batch size one, group size four, generation batch size four, gradient accumulation four, NF4 double quantization, FP16 compute, and deterministic seeds. Leave `steps_per_generation` unset. All base, SFT, training, validation, and test sampling uses the frozen Qwen non-thinking generation settings from the config. Use common random numbers across arms: derive answer-member seeds from SHA-256 of `(study_id, seed, step, example_id, group_member, "answer")` and metacognition-query seeds from the same tuple ending in `"metacognition"`; `arm` is deliberately excluded. Persist both raw outputs before parsing.

For a new adapter, pass the quantized base and one `LoraConfig` to TRL. For pre-SFT initialization or exact resume, use `PeftModel.from_pretrained(..., is_trainable=True)` and do not pass a second PEFT config. A complete resumable checkpoint must bind adapter weights/config, `optimizer.pt`, `scheduler.pt`, `trainer_state.json`, Python/NumPy/Torch/CUDA RNG state, global and micro-step, sampler cursor, and any custom generation-buffer state. Resume only via `trainer.train(resume_from_checkpoint=verified_path)`; loading adapter weights alone is inference or fresh-arm initialization, never resume.

Pre-SFT uses rank 8/alpha 16 on the same targets, five epochs, global batch size eight, AdamW, weight decay 0.01, cosine schedule, and learning rate `3e-5`. It mixes the frozen answer/confidence examples with the separate metacognition-query examples; it never emits all three tags in one completion. The best of five epoch checkpoints is selected by loss on the fixed 26-row SFT validation subset. Both RL arms load the exact same verified adapter checkpoint.

The training module must fail fast if runtime library versions differ from `requirements-rlmf-colab.txt`. It must log peak VRAM, wall time, examples seen, optimizer steps, checkpoint hash, and package versions.

- [ ] **Step 4: Run mocked tests and an optional tiny-model smoke**

```bash
.venv/bin/python -m pytest tests/test_rlmf_trainer.py tests/test_rlmf_training.py tests/test_rlmf_cli.py -q
```

Expected: mocked suite passes without downloading a model. The real 0.6B smoke is deferred to Task 9 on Colab.

- [ ] **Step 5: Commit**

```bash
git add src/trajectory_extractor/rlmf_trainer.py src/trajectory_extractor/rlmf_training.py src/trajectory_extractor/cli.py tests/test_rlmf_trainer.py tests/test_rlmf_training.py tests/test_rlmf_cli.py
git commit -m "feat: add paired restartable RLMF training"
```

### Task 8: Create A Thin, Restartable Colab Notebook

**Files:**
- Create: `notebooks/05_rlmf_colab.ipynb`
- Modify: `tests/test_notebook_smoke.py`
- Create: `tests/test_rlmf_notebook_contract.py`

**Notebook contract:**

1. Mount Drive only when `USE_DRIVE=1`.
2. Clone or upload the exact project commit.
3. Install `requirements-rlmf-colab.txt`, run `pip install -e . --no-deps`, and print versions.
4. Verify GPU name and VRAM; stop below 14 GB.
5. Verify sealed data and upstream manifests.
6. Run pre-SFT once.
7. Run paired arms in order `standard`, then `rlmf`, for one selected seed.
8. Generate designated-plus-20 validation/test rollouts only after the checkpoint completes and only in the registered audit order.
9. Train in local Colab scratch storage, then export one manifest-bound archive to Drive after each checkpoint.
10. Support rerun without overwriting completed artifacts.

- [ ] **Step 1: Write a failing notebook contract test**

Assert that every code cell delegates to package or CLI functions, no cell defines training math, no cell contains unpinned dependency installation, the project is installed editable with `--no-deps`, artifact roots are explicit, Drive receives only exported archives, and smoke mode is controlled by one configuration path.

- [ ] **Step 2: Confirm failure**

```bash
.venv/bin/python -m pytest tests/test_notebook_smoke.py tests/test_rlmf_notebook_contract.py -q
```

- [ ] **Step 3: Generate the notebook using the repository's notebook helper pattern**

The notebook should be inspectable but contain no hidden scientific logic. Use Markdown cells to distinguish smoke, pilot, and confirmatory runs. The final cell prints a machine-readable completion summary, not an interpretive conclusion.

- [ ] **Step 4: Execute notebook structural tests**

```bash
.venv/bin/python -m pytest tests/test_notebook_smoke.py tests/test_rlmf_notebook_contract.py -q
```

Expected: notebook parses, imports the package, and passes the thin-orchestrator contract without requiring a GPU.

- [ ] **Step 5: Commit**

```bash
git add notebooks/05_rlmf_colab.ipynb tests/test_notebook_smoke.py tests/test_rlmf_notebook_contract.py
git commit -m "feat: add restartable RLMF Colab notebook"
```

### Task 9: Run Smoke And Pilot Gates Before Confirmatory Compute

**Files:**
- Create: `docs/rlmf_runbook.md`
- Modify only if a software defect is found: RLMF source/tests from Tasks 2-8

- [ ] **Step 1: Run the smoke configuration on one Colab GPU**

Smoke budget: 8 pre-SFT rows, 8 RL rows, 2 validation rows, group size 2, generation batch size 2, gradient accumulation 2, one optimizer step per arm, seed 11. Run:

```bash
feature-dynamics rlmf-train --config configs/rlmf_qwen06b_smoke.json --stage pre-sft --resume
feature-dynamics rlmf-train --config configs/rlmf_qwen06b_smoke.json --stage rl --arm standard --seed 11 --resume
feature-dynamics rlmf-train --config configs/rlmf_qwen06b_smoke.json --stage rl --arm rlmf --seed 11 --resume
```

Expected: both arms complete, loss and advantages are finite, peak VRAM is below 15 GB, and parser validity is nonzero. Scientific metrics are ignored.

- [ ] **Step 2: Run a 25-step infrastructure pilot**

Use the confirmatory config with `--stop-after-step 25`, seed 11, and only the RL-train split. This flag may shorten a run but must not alter any hyperparameter.

Record wall time per optimizer step, generation throughput, peak VRAM, checkpoint size, Drive copy time, and estimated time for all six confirmatory arms.

- [ ] **Step 3: Apply the infrastructure gate**

Proceed only if:

- no OOM occurs;
- projected wall time is finite and documented;
- checkpoint resume reproduces step 25 state and next-step loss within deterministic tolerance;
- standard and RLMF answer groups, separate metacognition outputs, and pre-advantage reward tensors are identical on a shared fixture;
- all manifests verify after round-trip transfer to the Mac.

Fix only demonstrated software defects. Any hyperparameter change requires a new dated preregistration version before confirmatory training and invalidates the current study ID.

- [ ] **Step 4: Document the exact run procedure**

`docs/rlmf_runbook.md` must list session order, resume commands, artifact transfer, hash verification, expected storage, and what to do after disconnection.

- [ ] **Step 5: Commit code fixes and the runbook before confirmatory runs**

```bash
git add docs/rlmf_runbook.md src tests
git commit -m "docs: freeze verified RLMF execution runbook"
```

### Task 10: Generate Locked Rollouts And Evaluate Study 1

**Files:**
- Create: `src/trajectory_extractor/rlmf_evaluation.py`
- Create: `tests/test_rlmf_evaluation.py`
- Modify: `src/trajectory_extractor/cli.py`
- Extend: `tests/test_rlmf_cli.py`

**Interfaces:**

```python
generate_evaluation_bundle(config, arm, seed, split, store, *, auxiliary_count: int) -> Path
evaluate_behavior(study_id: str, store: RLMFArtifactStore) -> ClaimDecision
apply_behavior_gate(metrics: Mapping[str, Any]) -> ClaimDecision
```

CLI:

```bash
feature-dynamics rlmf-generate-rollouts --config ... --arm standard --seed 11 --split validation
feature-dynamics rlmf-generate-rollouts --config ... --arm rlmf --seed 11 --split validation
feature-dynamics rlmf-generate-rollouts --config ... --arm standard --seed 11 --split mechanism_train
feature-dynamics rlmf-evaluate-behavior --config ...
```

- [ ] **Step 1: Write failing evaluation and gate tests**

Test missing-seed rejection, paired-ID equality, duplicate completion rejection, exactly one designated plus 20 independent auxiliaries, exclusion of the designated response from `g_eval`, sealed `mechanism_train` rollouts, locked-validation-audit prerequisite, blinded-test-audit ordering, malformed-output inclusion, fixed-seed prompt-cluster bootstrap, three same-sign seed effects, judge-bias propagation and its `0.015` upper-limit gate, each individual gate boundary, `supported`, `not_supported`, and `not_evaluable` outcomes.

- [ ] **Step 2: Confirm failure**

```bash
.venv/bin/python -m pytest tests/test_rlmf_evaluation.py tests/test_rlmf_cli.py -q
```

- [ ] **Step 3: Implement locked evaluation**

Generate and seal development rollouts from pre-SFT/RL-train without using test IDs, freeze the proxy, then generate validation bundles for all paired seeds. Every evaluation bundle contains one designated answer/confidence response and 20 independently seeded answer-only auxiliaries. Run the locked 400-item validation audit and verify its completion marker. The test-rollout command must refuse to run before that marker exists and passes.

Only then generate and seal test bundles once. Before aggregate test metrics are calculated, create the blinded 1,000-row test audit and collect two independent ratings plus later disagreement-only adjudication through the sealed-source manifest. Task 4 publishes arm-specific confusion uncertainty, not a `delta_cMFG_star` gate. This task joins that uncertainty to the sealed behavioral records, propagates it through `delta_cMFG_star`, and exclusively decides whether the `<0.015` upper-limit gate passes, a 250-row append is required, or the study is `not_evaluable` at 2,000. An extension-required command exits nonzero and writes a sealed request binding the current size-specific evidence endpoint; the next Task 4 audit appends only the requested 250 stable IDs. A final `test_judge_audit` endpoint is published only after reliability gates pass and endpoint-specific propagation yields a final evaluable result. Evaluation joins arms by `(seed, example_id)` for designated responses and validates auxiliary member IDs within each bundle. Emit per-example, per-seed, fixed-seed aggregate, tie-preserving sensitivity, common-support sensitivity, unadjusted, and judge-bias-adjusted tables.

Only after the machine-derived Study 1 decision is `supported`, generate and seal one designated-plus-20 `mechanism_train` bundle for every final arm/seed checkpoint. Task 11 adds the corresponding pre-SFT bundles and teacher-forced activation artifacts. These sealed bundles are the only legal probe-training targets in Study 2.

The report object must include every gate value and a machine-derived decision string. Human prose may not override it.

- [ ] **Step 4: Run focused tests, then the confirmatory command after all six arms exist**

```bash
.venv/bin/python -m pytest tests/test_rlmf_evaluation.py tests/test_rlmf_cli.py -q
.venv/bin/feature-dynamics rlmf-evaluate-behavior --config configs/rlmf_qwen06b_confirmatory.json
```

Expected command output contains `decision`, `delta_cMFG_star`, fixed-seed prompt-cluster interval, all three seed effects and intervals, judge-bias-adjusted interval, format validity, accuracy difference, both cMFG sensitivities, and completion-marker path.

- [ ] **Step 5: Commit analysis code, not generated runs**

```bash
git add src/trajectory_extractor/rlmf_evaluation.py src/trajectory_extractor/cli.py tests/test_rlmf_evaluation.py tests/test_rlmf_cli.py
git commit -m "feat: add gated RLMF behavioral evaluation"
```

### Task 11: Extract Semantic Anchors Without Retaining Full Hidden States

**Files:**
- Create: `src/trajectory_extractor/rlmf_activations.py`
- Create: `tests/test_rlmf_activations.py`
- Modify: `src/trajectory_extractor/cli.py`
- Extend: `tests/test_rlmf_cli.py`

**Interfaces:**

```python
resolve_decoder_layers(model) -> Sequence[nn.Module]
locate_unique_anchor(input_ids, anchor_ids, *, name: str) -> int
selected_layer_indices(num_layers: int) -> tuple[int, int, int, int]
extract_anchored_states(
    model,
    tokenizer,
    prompt: str,
    teacher_forced_answer: str,
    layers: Sequence[int],
) -> AnchoredStates
```

CLI:

```bash
feature-dynamics rlmf-prepare-mechanism-targets --config ... --seed 11 --split mechanism_train
feature-dynamics rlmf-extract-activations --config ... --checkpoint pre_sft --seed 11 --split mechanism_train
feature-dynamics rlmf-extract-activations --config ... --checkpoint standard --seed 11 --split mechanism_train
feature-dynamics rlmf-extract-activations --config ... --checkpoint rlmf --seed 11 --split mechanism_train
```

Anchors:

- `prompt_end`: last non-padding prompt token;
- `pre_confidence`: final token of the unique teacher-forced prefix ending in `</sentence>\n<confidence>`.

- [ ] **Step 1: Write failing anchor and memory-contract tests**

Use tokenizer/model stubs to test unique subsequence matching, missing/duplicate anchors, left/right padding, 25/50/75/100-percent layer selection without duplicates, byte-identical teacher-forced answers across checkpoints, last-token-only storage, hook removal after exceptions, model-device retention, and absence of `.cpu()` calls inside layer hooks.

- [ ] **Step 2: Confirm failure**

```bash
.venv/bin/python -m pytest tests/test_rlmf_activations.py tests/test_rlmf_cli.py -q
```

- [ ] **Step 3: Implement sequential extraction**

For each seed/example, freeze the standard arm's designated answer as the teacher-forced answer for **all three checkpoints**: shared pre-SFT, standard, and RLMF. Compute checkpoint-specific `g` against that same answer using the already sealed 20 auxiliary responses; generate a sealed pre-SFT auxiliary bundle because Study 1 evaluation does not otherwise require one.

Load one base model plus one LoRA adapter, process batch size one under `torch.inference_mode()`, gather only eight vectors per example, cast them to float32 after leaving the hook, write one chunk atomically, and unload the model before switching checkpoints. Persist the frozen answer bytes, correctness, hashed answer-character features, answer token count, mean teacher-forced answer logprob, mean entropy, relation, popularity, and the independently sampled checkpoint-specific `g` beside the anchors.

Refuse to run unless Study 1's completion marker verifies. If Study 1 is unsupported, require explicit `--descriptive-seed 11` and watermark the artifact `confirmatory=false`. The confirmatory `pre_confidence` analysis uses the paired complete-case intersection of examples valid for pre-SFT, standard, and RLMF for the same seed. Persist every excluded ID and exclusion reason, require at least 95% paired coverage, and retain `prompt_end` only as a multiplicity-corrected sensitivity analysis. Never extract from freshly generated or unsealed `rl_train` data.

- [ ] **Step 4: Verify tests and a two-example local smoke**

```bash
.venv/bin/python -m pytest tests/test_rlmf_activations.py tests/test_rlmf_cli.py -q
.venv/bin/feature-dynamics rlmf-prepare-mechanism-targets --config configs/rlmf_qwen06b_smoke.json --seed 11 --split validation --limit 2
.venv/bin/feature-dynamics rlmf-extract-activations --config configs/rlmf_qwen06b_smoke.json --checkpoint standard --seed 11 --split validation --limit 2
```

Expected smoke: one `.npz` chunk per checkpoint, each shape `[2, 2, 4, hidden_size]`, identical answer hashes across checkpoints, finite values, and recorded peak RSS below the registered Mac limit.

- [ ] **Step 5: Commit**

```bash
git add src/trajectory_extractor/rlmf_activations.py src/trajectory_extractor/cli.py tests/test_rlmf_activations.py tests/test_rlmf_cli.py
git commit -m "feat: extract anchored RLMF activations"
```

### Task 12: Test Whether Dynamics Add An Internal Metacognitive Signal

**Files:**
- Create: `src/trajectory_extractor/rlmf_mechanism.py`
- Create: `tests/test_rlmf_mechanism.py`
- Modify: `src/trajectory_extractor/cli.py`
- Extend: `tests/test_rlmf_cli.py`

**Interfaces:**

```python
surface_features(records) -> np.ndarray
static_activation_features(states) -> np.ndarray
layer_dynamics_features(states) -> np.ndarray
fit_mechanism_probe(X_train, y_train, *, alpha_grid) -> FittedProbe
evaluate_mechanism(study_id, store) -> ClaimDecision
apply_mechanism_gate(metrics) -> ClaimDecision
```

- [ ] **Step 1: Write failing leakage, ablation, and DiD tests**

Test train-only pooled standardization/PCA/Ridge fitting, identical feature capacity across checkpoints, validation-only alpha selection, subject-and-answer-disjoint splits, paired three-checkpoint complete-case construction, 95% coverage gate, paired test evaluation, fixed-seed prompt-cluster inference, continuous independent-sample target `g`, primary `pre_confidence` MAE, answer-identity/correctness controls, prompt-end multiplicity correction, R2/Spearman sensitivities, layer-order permutation ablation, random-label null, static feature nesting, and exact DiD gate boundaries.

- [ ] **Step 2: Confirm failure**

```bash
.venv/bin/python -m pytest tests/test_rlmf_mechanism.py tests/test_rlmf_cli.py -q
```

- [ ] **Step 3: Implement three locked probe families**

1. `surface`: frozen-answer token count, hashed answer-character features, correctness, mean answer logprob, mean entropy, one-hot relation, and standardized popularity.
2. `static`: surface plus anchored activation PCA coordinates.
3. `static_plus_dynamics`: static plus adjacent selected-layer differences, norms, cosine changes, and a train-only ridge transition residual.

Fit one pooled, train-only PCA basis with fixed dimensionality across pre-SFT, standard, and RLMF states. Fit probes on sealed `mechanism_train`, select Ridge alpha from `[0.01, 0.1, 1.0, 10.0, 100.0]` on validation, and evaluate test once. The sole confirmatory contrast is test MAE at `pre_confidence`: RLMF-minus-standard difference in incremental dynamics gain over static features. Report pre-SFT as a baseline for whether the signal was already present; do not call a larger RLMF signal newly created. `prompt_end`, R2, Spearman, individual layers, and pre-SFT contrasts are secondary and Holm-corrected as one family.

The operator residual is described as a feature-flow residual. Do not label it a Remizov error bound.

- [ ] **Step 4: Verify and run after all activation endpoints complete**

```bash
.venv/bin/python -m pytest tests/test_rlmf_mechanism.py tests/test_rlmf_cli.py -q
.venv/bin/feature-dynamics rlmf-evaluate-mechanism --config configs/rlmf_qwen06b_confirmatory.json
```

Expected output includes every checkpoint's held-out metrics, all three per-seed effects and prompt-cluster intervals, the primary fixed-seed DiD interval, answer/correctness controls, corrected sensitivities, null controls, and a machine-derived Study 2A decision.

- [ ] **Step 5: Commit**

```bash
git add src/trajectory_extractor/rlmf_mechanism.py src/trajectory_extractor/cli.py tests/test_rlmf_mechanism.py tests/test_rlmf_cli.py
git commit -m "feat: evaluate internal metacognitive dynamics"
```

### Task 13: Add A Gated Same-Example Confidence-Expression Intervention

**Files:**
- Create: `src/trajectory_extractor/rlmf_patching.py`
- Create: `tests/test_rlmf_patching.py`
- Modify: `src/trajectory_extractor/cli.py`
- Extend: `tests/test_rlmf_cli.py`

**Interfaces:**

```python
probe_aligned_component(source, target, direction) -> Tensor
patch_hidden_state(hidden, component, *, alpha: float) -> Tensor
materialize_source_vectors(config, records, selection, store) -> Path
select_patch(validation_records, layers, alphas) -> PatchSelection
run_patch_condition(model, tokenizer, record, selection, condition) -> PatchedCompletion
evaluate_patching(study_id, store) -> ClaimDecision
```

Registered validation grid:

```text
layers = [25%, 50%, 75%]
alpha = [0.25, 0.50, 1.00]
```

Registered test conditions:

```text
no_patch
same_example_probe_aligned
orthogonal_same_example
shuffled_probe_aligned
norm_matched_random
sign_reversed_same_example
earlier_anchor_probe_aligned
reverse_rlmf_to_standard
```

- [ ] **Step 1: Write failing patch semantics and control tests**

Verify exact projection and tensor formulas, train-only direction fitting, one-token/one-layer application, two-pass source materialization, source-model unloading before target-model loading, initial full-prefix forward patching, hook cleanup, identical frozen answer bytes, deterministic shuffle and random controls, norm-matched orthogonal controls, sign reversal, earlier-anchor control, reverse-direction intervention, validation-only selection, dose-response reporting, test-once immutability, Holm correction, and gate refusal when Study 1 or 2A is unsupported.

- [ ] **Step 2: Confirm failure**

```bash
.venv/bin/python -m pytest tests/test_rlmf_patching.py tests/test_rlmf_cli.py -q
```

- [ ] **Step 3: Implement constrained patching**

On `mechanism_train`, fit one hidden-space Ridge direction `u_l` per candidate
layer using only RLMF `pre_confidence` states and independent `g` targets. The
validation selection chooses one layer and alpha. For the same prompt and frozen
standard-answer prefix, define:

```text
h_standard = standard state at pre_confidence
h_rlmf = RLMF state on the identical teacher-forced prefix
delta = h_rlmf - h_standard
component = u * dot(delta, u) / (dot(u, u) + epsilon)
h_patched = h_standard + alpha * component
```

Use a strict two-pass implementation suitable for the 8 GB Mac. First load the
RLMF checkpoint, materialize only the selected-layer source vectors for all
records, seal them, and unload the model. Then load the standard checkpoint,
teacher-force the same prefix, apply the component once during the initial
full-prefix forward pass, and generate only the `<confidence>` suffix. Never load
both checkpoints simultaneously and never retain full-layer tensors.

Accuracy is fixed by construction and must not be presented as improved. Measure
absolute confidence-expression error and cMFG* on the fixed answers. The primary
claim requires the same-example component to beat every control after Holm
correction and show the registered dose direction; reverse RLMF-to-standard
patching must move the expression effect oppositely. Even then, describe the
result as causal sufficiency of a probe-aligned component for confidence-token
generation, not mediation or a complete metacognitive mechanism.

- [ ] **Step 4: Verify and run validation before test**

```bash
.venv/bin/python -m pytest tests/test_rlmf_patching.py tests/test_rlmf_cli.py -q
.venv/bin/feature-dynamics rlmf-select-patch --config configs/rlmf_qwen06b_confirmatory.json
.venv/bin/feature-dynamics rlmf-evaluate-patch --config configs/rlmf_qwen06b_confirmatory.json
```

Expected: `select-patch` writes one immutable layer/alpha choice from validation. `evaluate-patch` refuses a second test execution and emits the corrected control comparisons and Study 2B decision.

- [ ] **Step 5: Commit**

```bash
git add src/trajectory_extractor/rlmf_patching.py src/trajectory_extractor/cli.py tests/test_rlmf_patching.py tests/test_rlmf_cli.py
git commit -m "feat: add gated confidence-state patching"
```

### Task 14: Build The Fellowship-Grade Report Without Claim Inflation

**Files:**
- Create: `src/trajectory_extractor/rlmf_report.py`
- Create: `tests/test_rlmf_report.py`
- Create: `docs/fellowship_project.md`
- Modify: `README.md`
- Modify: `docs/references.md`
- Modify: `src/trajectory_extractor/cli.py`

**Interfaces:**

```python
build_rlmf_report(study_id: str, store: RLMFArtifactStore, output: Path) -> Path
claim_vocabulary(decisions: Mapping[str, ClaimDecision]) -> Mapping[str, str]
```

CLI:

```bash
feature-dynamics rlmf-report --config configs/rlmf_qwen06b_confirmatory.json --output docs/fellowship_project.md
```

- [ ] **Step 1: Write failing report-contract tests**

Assert that reports always include hardware, deviations, frozen Study 0 null, development/locked-validation/blinded-test judge audits, every preregistered metric, per-seed and fixed-seed prompt-cluster intervals, negative controls, runtime/RAM/VRAM, limitations, artifact hashes, the analysis-source Git commit, and exact reproduction commands.

Add forbidden unsupported phrases:

```text
exact intuition
solves hallucinations
prevents jailbreaks
Remizov theorem for transformers
replicated the paper exactly
```

The report may use `causal treatment effect` for Study 1 only when arm-parity and behavioral completion markers verify. It may use `causal state intervention` only when the patching completion marker verifies and Study 2B is supported. It may never use `causal mediator`.

- [ ] **Step 2: Confirm failure**

```bash
.venv/bin/python -m pytest tests/test_rlmf_report.py -q
```

- [ ] **Step 3: Implement figures and narrative from sealed metrics only**

Required figures:

1. paired per-seed cMFG* and accuracy;
2. reliability diagrams for standard and RLMF;
3. surface/static/dynamics probe comparison;
4. patch condition effect with corrected intervals, only if Study 2B ran;
5. compute and memory budget table.

`docs/fellowship_project.md` must lead with the scientific question and the causal A/B design, not the application goal. It must distinguish:

- resource-scaled reproduction evidence;
- original mechanistic extension;
- unsupported or unrun stages;
- the future live-monitor proposal.

- [ ] **Step 4: Verify and render**

```bash
.venv/bin/python -m pytest tests/test_rlmf_report.py -q
.venv/bin/feature-dynamics rlmf-report --config configs/rlmf_qwen06b_confirmatory.json --output docs/fellowship_project.md
```

Expected: report is deterministic from sealed artifacts and exits nonzero if any cited metric lacks a verified completion chain.

- [ ] **Step 5: Commit**

```bash
git add src/trajectory_extractor/rlmf_report.py src/trajectory_extractor/cli.py tests/test_rlmf_report.py docs/fellowship_project.md README.md docs/references.md
git commit -m "docs: publish gated RLMF research report"
```

### Task 15: Final Reproduction Audit And Release Bundle

**Files:**
- Create: `scripts/verify_rlmf_release.py`
- Create: `tests/test_rlmf_release.py`
- Create: `docs/rlmf_reproduction_checklist.md`
- Modify: `README.md`

- [ ] **Step 1: Write a failing release-verifier test**

The verifier must reject missing source pins, dirty generated report inputs, unverified endpoint markers, mismatched paired IDs, duplicate test execution markers, absent development/validation/test judge audits, a failed judge-bias bound, absent null controls, seed-population language unsupported by the fixed-three-seed design, or a report whose recorded analysis-source commit is not an ancestor of the release commit. It must also reject changes under `src/`, `configs/`, `requirements-rlmf-colab.txt`, or `third_party/rlmf/` between the recorded analysis-source commit and the release commit. Documentation-only release commits are allowed, avoiding a circular self-hash requirement.

- [ ] **Step 2: Confirm failure**

```bash
.venv/bin/python -m pytest tests/test_rlmf_release.py -q
```

- [ ] **Step 3: Implement the read-only release verifier**

The script prints one JSON object with:

```json
{
  "source_pins_valid": true,
  "study0_unchanged": true,
  "judge_audit_valid": true,
  "behavior_complete": true,
  "mechanism_status": "supported|not_supported|not_run",
  "patching_status": "supported|not_supported|not_run",
  "report_reproducible": true,
  "release_ready": true
}
```

It never writes or repairs artifacts.

- [ ] **Step 4: Run all verification commands**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_rlmf_release.py --config configs/rlmf_qwen06b_confirmatory.json
git diff --check
git status --short
```

Expected:

- complete test suite passes;
- release verifier reports `release_ready: true` for a finished study, regardless of whether the scientific decision is supported or null;
- no whitespace errors;
- only intentionally uncommitted generated run artifacts, if any, remain outside Git.

- [ ] **Step 5: Independent scientific and engineering review**

Dispatch two read-only reviewers:

1. scientific reviewer: leakage, estimand, judge validity, gates, multiplicity, and claim language;
2. engineering reviewer: resume semantics, parity, artifact immutability, memory limits, and reproduction commands.

Resolve every high-severity finding with a failing regression test before changing code. Record accepted and rejected findings in `docs/rlmf_reproduction_checklist.md`.

- [ ] **Step 6: Commit the release audit**

```bash
git add scripts/verify_rlmf_release.py tests/test_rlmf_release.py docs/rlmf_reproduction_checklist.md README.md
git commit -m "chore: add RLMF release verification"
```

## Execution Order And Stop Rules

```text
Tasks 1-8
  -> smoke and infrastructure pilot (Task 9)
  -> pre-SFT once
  -> paired standard/RLMF arms for seeds 11, 22, 33
  -> development audit and frozen proxy
  -> locked validation audit
  -> sealed test rollouts and blinded test audit
  -> Study 1 evaluation
       if not_supported: report null; optional one-seed descriptive mechanism only; stop
       if supported: extract all paired mechanistic artifacts
  -> Study 2A evaluation
       if not_supported: report behavior-only result; stop
       if supported: select patch on validation and run Study 2B test once
  -> deterministic report and release audit
```

No result is wasted:

- Study 1 null: a resource-scaled boundary result with audited judge and paired treatment control.
- Study 1 positive, Study 2A null: RLMF improves output behavior without evidence for the tested internal dynamics mechanism.
- Study 2A positive, Study 2B null: internal decodability rises, but the selected state difference is not causally sufficient.
- All positive: evidence on the three registered seeds that RLMF improves faithful uncertainty expression, increases held-out predictability from the tested internal dynamics beyond locked controls, and that a constrained probe-aligned component is causally sufficient to change confidence-token expression.

## Fellowship Deliverable Standard

The project is ready to present when it contains:

- a frozen preregistration and explicit deviations table;
- exact upstream/model/data pins;
- a tested, resumable single-GPU training path;
- three paired seeds and sealed per-example outputs;
- passed development, locked-validation, and blinded-test human audits with bounded differential judge bias;
- strong behavioral baselines, per-seed estimates, and fixed-seed prompt-cluster intervals;
- a mechanistic extension that is gated rather than retrofitted;
- causal controls if intervention runs;
- complete compute/RAM/VRAM accounting;
- a deterministic report that remains credible under a null result;
- one-command release verification.

The headline application sentence should be:

> We reproduced the core RLMF treatment at resource scale on a 0.6B open model, then tested whether any calibration gain adds held-out predictability in answer-controlled internal dynamics and whether a probe-aligned component causally affects confidence-token expression.

The previous feature-flow experiment appears as methodological motivation and a negative control history. The live per-token "artificial intuition" monitor remains a clearly separated next experiment, justified only if the anchored dynamics and causal-patching stages support it.
