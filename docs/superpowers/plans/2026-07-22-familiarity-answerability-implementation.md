# Familiarity vs. Answerability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, leakage-resistant pipeline for the preregistered Familiarity-vs-Answerability behavioral study (F1), mechanistic decoding study (F2A), gated causal intervention study (F2B), and optional circuit follow-up (F3).

**Architecture:** Add an isolated `fa_*` study stack beside the legacy concept-mixing and RLMF stacks. Reuse only generic numerical and Hugging Face techniques; do not reuse legacy run state, endpoint stores, study objects, or decoding hooks. Scientific logic lives in tested Python modules, while notebooks only orchestrate resumable commands.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face Transformers, NumPy, pandas, scikit-learn, Matplotlib, optional SAELens/Gemma Scope, optional `decoderesearch/circuit-tracer`, pytest.

## Global Constraints

- Preserve the baseline of `362 passed, 3 skipped`.
- Keep the old RLMF and metacognitive-feature-flow namespaces read-only; no `fa_*` module may import `rlmf_*`, `secondary_study`, `study`, or `intervention_study`.
- Runtime artifacts live under `runs/familiarity_answerability/<run_id>/`; publishable manifests live under `release/familiarity_answerability/`.
- Keep `pilot`, `mechanism_train`, `locked_validation`, `behavior_test`, `probe_test`, `intervention_test`, and `circuit_dev` physically and logically separate.
- Endpoint state is exactly `sealed -> unlocked_once -> evaluated -> closed`; each test endpoint can be opened once and only with its registered parent manifests.
- `behavior_test`, `probe_test`, and `intervention_test` are independent endpoints. A manifest for one must never unlock another.
- Deterministic example IDs are SHA-256 hashes of canonical example content. Generated artifacts are immutable, hash-bound, and no-clobber.
- F1 plus F2A is the minimum Fellowship artifact. F2B is gated by F1/F2A; F3 is optional and cannot rescue a failed earlier gate.
- `google/gemma-2-2b-it` is confirmatory. `Qwen/Qwen3-0.6B` is smoke-only and cannot select confirmatory layers, thresholds, templates, or claims.
- No confirmatory generation starts before exact source/model/tokenizer/chat-template pins, preregistration hash, split hashes, power amendment, and required human-audit endpoints are sealed.
- No custom SAE training. Gemma Scope is optional and enters confirmatory comparisons only after the registered instruction-tuned loss-recovery gate.
- Interventions are prefill-only and position-addressed; all hooks must be disabled during autoregressive decoding.
- Reports recompute gates from canonical metrics and always expose null results, invalid outputs, missingness, failed fidelity checks, and skipped gated phases.

## Frozen Implementation Decisions

- The same-string block reuses the 192 split-isolated entity units. Each unit receives one hash-assigned template family and four rows: `high_exposure`/`low_exposure` crossed with `target_bound`/`code_absent`. Template families are balanced within split and domain. The provisional count may increase only through the pre-outcome power amendment.
- Core rows expand each entity unit over every template family registered for its split: three train, three validation, and four test families. Counterbalancing uses a deterministic hash-indexed Latin-square schedule.
- Output normalization is Unicode NFC plus surrounding-whitespace removal. Exact normalized `UNKNOWN` is abstention. A full-string target code, distractor code, or other registered code-shaped token is classified before generic text. A printable nonempty single-line string is `other_non_abstention`; empty, multiline, truncated, nonprintable, or infrastructure-marked output is `invalid_format`.
- Generic generation commands may access only `pilot`, `mechanism_train`, `locked_validation`, and `circuit_dev`. Test commands acquire an endpoint lease and perform generation plus evaluation in one transaction before marking the endpoint evaluated and closed.
- Every protected split has a separate encrypted-or-capability-scoped manifest; the global index contains IDs and hashes only, never protected prompt text or labels.
- The crossed bootstrap independently resamples entity units and template families with replacement and uses the product of their multiplicities as row weights. H2 uses the 2.5th percentile of the paired bootstrap difference as its lower 95% bound.
- The registered power grid crosses absent attempt rates `{0.10, 0.25, 0.50}`, entity ICC `{0.05, 0.15, 0.30}`, template ICC `{0.02, 0.10}`, invalid-format rates `{0.00, 0.05}`, and interactions `{0.00, 0.025, 0.05, 0.075, 0.10}` with 2,000 simulations per point and seed `20260722`.
- Same-string activation interchange is full replacement (`alpha = 1`) and has no tuned alpha. The validation alpha grid `{0.25, 0.50, 1.00, 1.50}` applies only to secondary contrastive-direction steering.
- Residual candidates cover all 26 Gemma 2 transformer layers. PCA candidates are `{none, 16, 32, 64}` and L2 logistic `C` candidates are `{0.01, 0.1, 1.0, 10.0}`. Target familiarity uses `target_intro_end`; answerability uses `user_prompt_end`; unsupported-answer prediction also evaluates the output-proximal control but cannot use it for a pre-output claim.
- Official revision pins recorded before implementation are Gemma model/tokenizer `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`, Gemma Scope residual SAE repository `fd571b47c1c64851e9b1989792367b9babb4af63`, and optional circuit-tracer `4bb8c0ea10bde09727e14565ec8469656880da53` (`v0.5.0`). Individual SAE paths are selected from the registered 16K family on training/validation only and still require the instruction-tuned loss-recovery gate.

---

## File Map

| Path | Responsibility |
|---|---|
| `src/trajectory_extractor/fa_config.py` | Immutable config, split/endpoint constants, canonical hashes |
| `src/trajectory_extractor/fa_artifacts.py` | No-clobber shards, lineage, endpoint state, release checksums |
| `src/trajectory_extractor/fa_entities.py` | Screening, matching, naturalness/reserve audit records |
| `src/trajectory_extractor/fa_data.py` | 2 x 2 x 3 construction, same-string block, split and lexical audits |
| `src/trajectory_extractor/fa_scoring.py` | Parser, outcome taxonomy, F1 estimands, crossed bootstrap, gates |
| `src/trajectory_extractor/fa_runtime.py` | Injectable model runner and resumable generation orchestration |
| `src/trajectory_extractor/fa_activations.py` | Anchor resolution and selected-position activation extraction |
| `src/trajectory_extractor/fa_probes.py` | Surface/output/residual/SAE/dynamics models, selection, nulls, OOD |
| `src/trajectory_extractor/fa_interventions.py` | Same-string patching, steering, controls, capability gates |
| `src/trajectory_extractor/fa_circuits.py` | Optional adapter, case selection, replacement/perturbation fidelity |
| `src/trajectory_extractor/fa_report.py` | Claims, tables, figures, release bundle |
| `src/trajectory_extractor/fa_cli.py` | Parser registration and command handlers |
| `src/trajectory_extractor/cli.py` | Two-line registration/dispatch integration only |

## Task 1: Frozen Configuration and CLI Boundary

**Files:**
- Create: `src/trajectory_extractor/fa_config.py`
- Create: `src/trajectory_extractor/fa_cli.py`
- Create: `configs/familiarity_answerability_gemma2_2b.json`
- Create: `configs/familiarity_answerability_qwen06b_smoke.json`
- Create: `docs/familiarity_answerability_preregistration.md`
- Create: `docs/familiarity_answerability_protocol_amendment_2026-07-22.md`
- Create: `data/fa/source_pins.json`
- Modify: `src/trajectory_extractor/cli.py`
- Test: `tests/test_fa_config.py`
- Test: `tests/test_fa_cli.py`

**Interfaces:**
- Produces `FAConfig.from_json(path)`, `FAConfig.canonical_bytes`, `FAConfig.config_hash`.
- Produces `register_fa_subcommands(subparsers)` and `dispatch_fa(args) -> int | None`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_confirmatory_config_is_canonical_and_pinned():
    config = FAConfig.from_json(CONFIRMATORY_CONFIG)
    assert config.profile == "confirmatory"
    assert config.model_id == "google/gemma-2-2b-it"
    assert len(config.model_revision) == 40
    assert config.split_counts == {
        "mechanism_train": 64,
        "locked_validation": 32,
        "behavior_test": 48,
        "probe_test": 24,
        "intervention_test": 24,
    }
    assert len(config.config_hash) == 64

def test_config_rejects_mutable_revision_and_endpoint_overlap(tmp_path):
    payload = json.loads(CONFIRMATORY_CONFIG.read_text())
    payload["model_revision"] = "main"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="immutable revision"):
        FAConfig.from_json(path)
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_fa_config.py -q`

Expected: import failure for `trajectory_extractor.fa_config`.

- [ ] **Step 3: Implement the frozen dataclass and canonical serialization**

```python
@dataclass(frozen=True)
class FAConfig:
    schema_version: int
    profile: str
    study_id: str
    run_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    chat_template_sha256: str
    split_seed: int
    split_counts: Mapping[str, int]
    generation: Mapping[str, Any]
    bootstrap_replicates: int
    bootstrap_seed: int
    thresholds: Mapping[str, float]
    anchors: tuple[str, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "FAConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("FA config must be a JSON object")
        config = cls(**value)
        config.validate()
        return config

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()
```

Validation must reject mutable revisions, unknown split names, duplicate endpoint counts, nonpositive counts, non-finite thresholds, nonregistered anchors, and smoke configurations that label Qwen confirmatory.

- [ ] **Step 4: Add thin CLI routing**

`cli.main()` calls `register_fa_subcommands(subparsers)` before parsing and returns `dispatch_fa(args)` when it returns a non-`None` result. Legacy command bodies remain unchanged.

- [ ] **Step 5: Freeze the pre-outcome protocol artifacts**

The preregistration copies H1-H8, the registered grids, endpoint rules, and claim boundaries from the approved research plan plus Frozen Implementation Decisions. The amendment records the date, rationale, affected endpoints, and `pre_outcome: true`. `source_pins.json` records the three official immutable revisions and the API retrieval date without claiming that gated model access has been tested.

- [ ] **Step 6: Verify GREEN and regression safety**

Run: `.venv/bin/python -m pytest tests/test_fa_config.py tests/test_fa_cli.py tests/test_rlmf_cli.py -q`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/trajectory_extractor/fa_config.py src/trajectory_extractor/fa_cli.py src/trajectory_extractor/cli.py configs/familiarity_answerability_*.json docs/familiarity_answerability_preregistration.md docs/familiarity_answerability_protocol_amendment_2026-07-22.md data/fa/source_pins.json tests/test_fa_config.py tests/test_fa_cli.py
git commit -m "feat: add familiarity answerability config boundary"
```

## Task 2: Sealed Artifact Store and Endpoint State Machine

**Files:**
- Create: `src/trajectory_extractor/fa_artifacts.py`
- Test: `tests/test_fa_artifacts.py`

**Interfaces:**
- Produces `SealedShard`, `UnlockReceipt`, and `FAArtifactStore`.
- Consumes only `FAConfig`; no legacy store/config types.

- [ ] **Step 1: Write failing store tests**

```python
def test_completed_shard_is_no_clobber_and_hash_verified(tmp_path, config):
    store = FAArtifactStore(tmp_path)
    sealed = store.write_completed_shard(
        config.run_id, "pilot", "0001", [{"example_id": "a"}], lineage(config)
    )
    assert store.verify_shard(sealed.manifest_path).sha256 == sealed.sha256
    with pytest.raises(FileExistsError):
        store.write_completed_shard(
            config.run_id, "pilot", "0001", [{"example_id": "b"}], lineage(config)
        )

def test_probe_endpoint_rejects_behavior_parent_and_second_unlock(tmp_path, config):
    store = prepared_store(tmp_path, config)
    with pytest.raises(ValueError, match="probe selection"):
        store.unlock_endpoint("probe_test", prereg_hash(), behavior_selection_hash())
    receipt = store.unlock_endpoint("probe_test", prereg_hash(), probe_selection_hash())
    with pytest.raises(ValueError, match="already unlocked"):
        store.unlock_endpoint("probe_test", prereg_hash(), probe_selection_hash())
    assert receipt.state == "unlocked_once"
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_fa_artifacts.py -q`

- [ ] **Step 3: Implement immutable shards**

```python
@dataclass(frozen=True)
class SealedShard:
    namespace: str
    shard_id: str
    data_path: Path
    manifest_path: Path
    sha256: str
    row_count: int

@dataclass(frozen=True)
class UnlockReceipt:
    endpoint: str
    lease_id: str
    state: str
    preregistration_hash: str
    selection_manifest_hash: str
```

`FAArtifactStore` implements the exact public methods `write_completed_shard(run_id, namespace, shard_id, rows, lineage)`, `verify_shard(manifest_path)`, `seal_endpoint(endpoint, artifacts, parents)`, `unlock_endpoint(endpoint, preregistration_hash, selection_manifest_hash)`, `mark_evaluated(receipt, metrics_path)`, and `close_endpoint(endpoint)` with the return types declared in the Interfaces block.

Writes use a same-directory temporary file, `flush`, `fsync`, SHA-256, exclusive publish, sidecar manifest, and directory `fsync`. Resume scans only verified sidecars. Readers enforce namespace-to-endpoint mapping.

- [ ] **Step 4: Add tampering, interruption, traversal, symlink, and cross-endpoint tests**

The tests inject write/fsync failures and assert no completed marker exists; they also mutate data bytes and require verification failure.

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_fa_artifacts.py tests/test_rlmf_artifacts.py tests/test_secondary_artifacts.py -q`

- [ ] **Step 6: Commit**

```bash
git add src/trajectory_extractor/fa_artifacts.py tests/test_fa_artifacts.py
git commit -m "feat: add sealed familiarity answerability artifacts"
```

## Task 3: Entity Screening, Matching, and Human-Audit Contracts

**Files:**
- Create: `src/trajectory_extractor/fa_entities.py`
- Create: `data/fa/schemas/candidate_entity.schema.json`
- Create: `data/fa/schemas/screening_question.schema.json`
- Create: `data/fa/schemas/synthetic_match.schema.json`
- Create: `data/fa/schemas/naturalness_rating.schema.json`
- Test: `tests/test_fa_entities.py`

**Interfaces:**
- Produces `CandidateEntity`, `ScreeningResult`, `EntityMatch`, `NaturalnessRating`.
- Produces `score_screening`, `match_synthetic_entities`, `audit_naturalness_manifest`.

- [ ] **Step 1: Write failing screening and matching tests**

```python
def test_screening_requires_two_of_three_alias_correct_answers():
    result = score_screening(candidate(), ["Paris", "France", "wrong"])
    assert result.qualifies is True
    assert result.recall_score == pytest.approx(2 / 3)

def test_matching_enforces_token_and_surface_constraints(fake_tokenizer):
    match = match_synthetic_entities([real_entity()], synthetic_pool(), fake_tokenizer)[0]
    assert match.real_token_count == match.synthetic_token_count
    assert match.real_word_count == match.synthetic_word_count
    assert match.capitalization_pattern_equal
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_fa_entities.py -q`

- [ ] **Step 3: Implement immutable records and exact alias scoring**

The implementation exposes these exact typed call signatures: `score_screening(candidate, completions) -> ScreeningResult`, `match_synthetic_entities(real_entities, synthetic_candidates, tokenizer) -> tuple[EntityMatch, ...]`, and `audit_naturalness_manifest(matches, ratings) -> NaturalnessAudit`.

Matching uses exact token count, word count, capitalization pattern, coarse type, registered character-length tolerance, deterministic tie-breaking, and one-to-one assignment. Naturalness requires two independent raters and a third rater only for registered disagreements.

- [ ] **Step 4: Add schema validation and split-safe reserve-list tests**

Reject missing QIDs/source provenance, duplicate names, match reuse, rater identity reuse, and reserve entries crossing split boundaries.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_fa_entities.py -q`

```bash
git add src/trajectory_extractor/fa_entities.py data/fa/schemas tests/test_fa_entities.py
git commit -m "feat: add entity screening and matching contracts"
```

## Task 4: Deterministic Factorial Dataset and Same-String Replication

**Files:**
- Create: `src/trajectory_extractor/fa_data.py`
- Test: `tests/test_fa_data.py`

**Interfaces:**
- Consumes audited `EntityMatch` records and `FAConfig`.
- Produces `FAExample`, `FAManifest`, `build_factorial_examples`, `build_same_string_examples`, and `audit_dataset`.

- [ ] **Step 1: Write failing balance and lexical-control tests**

```python
def test_each_entity_unit_has_exact_two_by_two_by_three_balance():
    rows = build_factorial_examples(config(), [entity_unit()])
    assert len(rows) == 12 * config().template_repetitions
    assert Counter((r.target_familiarity, r.distractor_familiarity, r.answerability) for r in rows) == expected_cells()

def test_target_and_distractor_bound_pairs_preserve_lexical_multiset(tokenizer):
    target, distractor = paired_answerability_rows()
    assert Counter(tokenizer.encode(target.user_text)) == Counter(tokenizer.encode(distractor.user_text))
    assert target.registry_code == distractor.registry_code

def test_same_string_pair_keeps_target_and_token_budget_fixed():
    low, high = same_string_pair()
    assert low.target_text == high.target_text
    assert low.rendered_token_count == high.rendered_token_count
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_fa_data.py -q`

- [ ] **Step 3: Implement canonical examples**

```python
@dataclass(frozen=True)
class FAExample:
    example_id: str
    entity_unit_id: str
    split: str
    template_family: str
    target_familiarity: str
    distractor_familiarity: str
    answerability: str
    target_text: str
    distractor_text: str
    registry_code: str
    expected_output: str
    user_text: str
    canonical_payload_sha256: str
```

IDs are recomputed from canonical content and exclude runtime paths. Code assignment is collision-free and balanced. Split assignment is grouped by full entity unit and template family.

- [ ] **Step 4: Implement all audits**

Audit independent target/distractor variation, entity order, query role, relation order, code position, code vocabulary, template overlap, entity overlap, rendered token length, special-token sequence, and same-string token budget.

- [ ] **Step 5: Add conservative cluster-power simulation**

`simulate_interaction_power(design, effect_grid, within_entity_correlations, seed) -> PowerAudit` uses the exact registered grid in Frozen Implementation Decisions and stores every cell's estimated power and Monte Carlo standard error.

The confirmatory builder refuses to run until the 5-point interaction has at least 0.80 estimated power on every registered conservative grid point or a dated amendment updates sample size.

- [ ] **Step 6: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_fa_data.py -q`

```bash
git add src/trajectory_extractor/fa_data.py tests/test_fa_data.py
git commit -m "feat: build controlled familiarity answerability dataset"
```

## Task 5: Strict Scoring, F1 Estimands, and Behavioral Gates

**Files:**
- Create: `src/trajectory_extractor/fa_scoring.py`
- Test: `tests/test_fa_scoring.py`

**Interfaces:**
- Produces `OutcomeClass`, `ScoredResponse`, `score_response`, `estimate_behavior`, `crossed_bootstrap`, and `behavioral_gate`.

- [ ] **Step 1: Write failing parser and taxonomy tests**

```python
@pytest.mark.parametrize("text", ["UNKNOWN", " UNKNOWN\n"])
def test_exact_normalized_unknown_is_only_non_attempt(text):
    scored = score_response(absent_example(), text)
    assert scored.answer_attempt == 0
    assert scored.outcome == OutcomeClass.ABSTENTION

@pytest.mark.parametrize("text", ["I do not know", "unknown.", "", "K7M2 because..."])
def test_every_other_completion_is_intention_to_treat_attempt(text):
    assert score_response(absent_example(), text).answer_attempt == 1
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_fa_scoring.py -q`

- [ ] **Step 3: Implement six-class exact scoring**

Classes are `abstention`, `exact_target_code`, `distractor_code_copy`, `novel_code_assertion`, `other_non_abstention`, and `invalid_format`. Invalid format remains in the denominator and counts as an attempt.

- [ ] **Step 4: Implement registered estimands**

```python
def estimate_behavior(rows: Sequence[ScoredResponse]) -> BehavioralMetrics:
    cell_rates = weighted_cell_rates(rows)
    absent_real = 0.5 * (
        cell_rates[("screened_real", "distractor_bound")]
        + cell_rates[("screened_real", "code_absent")]
    )
    absent_synthetic = 0.5 * (
        cell_rates[("matched_synthetic", "distractor_bound")]
        + cell_rates[("matched_synthetic", "code_absent")]
    )
    interaction = (absent_real - absent_synthetic) - (
        cell_rates[("screened_real", "target_bound")]
        - cell_rates[("matched_synthetic", "target_bound")]
    )
    return BehavioralMetrics.from_cell_rates(cell_rates, interaction)

def crossed_bootstrap(
    rows: Sequence[ScoredResponse], replicates: int, seed: int
) -> BootstrapDistribution:
    rng = np.random.default_rng(seed)
    samples = [
        estimate_behavior(cross_resample(rows, rng)).interaction
        for _ in range(replicates)
    ]
    return BootstrapDistribution.from_samples(samples)
```

Bootstrap replicates resample entity units and template families, never rows. Report completion by cell and return `not_evaluable` below 95%.

- [ ] **Step 5: Add deterministic CI, sensitivity, and gate tests**

Tests distinguish H1, H2, and H2b; H2b cannot rescue H1. Sensitivities retain denominators and invalid-format counts.

- [ ] **Step 6: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_fa_scoring.py -q`

```bash
git add src/trajectory_extractor/fa_scoring.py tests/test_fa_scoring.py
git commit -m "feat: score familiarity answerability behavior"
```

## Task 6: Resumable Generation and Core CLI Commands

**Files:**
- Create: `src/trajectory_extractor/fa_runtime.py`
- Modify: `src/trajectory_extractor/fa_cli.py`
- Test: `tests/test_fa_runtime.py`
- Modify: `tests/test_fa_cli.py`

**Interfaces:**
- Produces `ModelRunner` protocol, `HFModelRunner`, `run_generation_shard`, and `resume_generation`.
- Adds commands through `fa-score-behavior`.

- [ ] **Step 1: Write failing fake-runner and resume tests**

```python
class FakeRunner:
    def generate(self, prompts, generation):
        return ["UNKNOWN" if "not stated" in p else "K7M2" for p in prompts]

def test_resume_skips_only_verified_completed_shards(tmp_path):
    first = run_generation_shard(FakeRunner(), pilot_manifest(), store(tmp_path), "0001")
    second = run_generation_shard(FakeRunner(), pilot_manifest(), store(tmp_path), "0001")
    assert first == second
    assert FakeRunner.calls == 1
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_fa_runtime.py tests/test_fa_cli.py -q`

- [ ] **Step 3: Implement deterministic and sampled generation records**

Every row records example/config/model/tokenizer/chat-template hashes, rendered prompt hash, generation parameters, raw completion, completion status, exception class, wall time, and peak memory. Infrastructure failures remain retryable; model outputs are never deleted.

- [ ] **Step 4: Register and implement commands**

Implement `fa-screen-entities`, `fa-build-pilot`, `fa-build-confirmatory`, `fa-audit-manifest`, `fa-run-generation`, and `fa-score-behavior`. All handlers accept `--config`, `--root`, and explicit input manifests. Test commands cannot infer or glob another endpoint path.

- [ ] **Step 5: Verify pilot gates and CLI JSON contracts**

The pilot stops confirmatory construction when target-bound accuracy is below 70%, invalid output exceeds 5% in any cell, or absent answering is at a registered floor/ceiling.

- [ ] **Step 6: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_fa_runtime.py tests/test_fa_cli.py -q`

```bash
git add src/trajectory_extractor/fa_runtime.py src/trajectory_extractor/fa_cli.py tests/test_fa_runtime.py tests/test_fa_cli.py
git commit -m "feat: add resumable familiarity answerability generation"
```

## Task 7: Registered Activation Anchors and Resume

**Files:**
- Create: `src/trajectory_extractor/fa_activations.py`
- Modify: `src/trajectory_extractor/fa_cli.py`
- Test: `tests/test_fa_activations.py`

**Interfaces:**
- Produces `AnchorRecord`, `ActivationRecord`, `resolve_registered_anchors`, and `extract_registered_anchors`.

- [ ] **Step 1: Write failing anchor tests**

```python
def test_anchor_resolution_handles_repeated_target_and_assistant_prefix(fake_tokenizer):
    record = resolve_registered_anchors(example_with_repeated_target(), fake_tokenizer)
    assert record.target_intro_end < record.user_prompt_end < record.assistant_prefix_end
    assert record.assistant_prefix_end == len(record.input_ids) - 1
    assert record.rendered_prompt_sha256 == sha256(record.rendered_bytes)

def test_activation_extraction_stores_only_registered_positions(fake_runner):
    result = extract_registered_anchors(fake_runner, example(), registered_layers=(4, 12, 20))
    assert result.activations.shape == (3, 3, fake_runner.hidden_size)
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_fa_activations.py -q`

- [ ] **Step 3: Implement token provenance and selected-position extraction**

Store rendered UTF-8 bytes, token IDs, special-token mask, offset mapping when available, exact anchor indices, layer IDs, dtype, and activation hash. Fail closed on ambiguous target occurrence rather than choosing a token silently.

- [ ] **Step 4: Stream shards without retaining all hidden states**

The production runner registers hooks only on selected layers, copies registered positions to CPU, releases each prompt tensor, and writes a verified NPZ shard plus JSONL index.

- [ ] **Step 5: Add `fa-extract-activations` command and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_fa_activations.py tests/test_fa_cli.py -q`

- [ ] **Step 6: Commit**

```bash
git add src/trajectory_extractor/fa_activations.py src/trajectory_extractor/fa_cli.py tests/test_fa_activations.py tests/test_fa_cli.py
git commit -m "feat: extract registered familiarity answerability anchors"
```

## Task 8: F2A Probes, SAE Gate, Dynamics, Nulls, and OOD

**Files:**
- Create: `src/trajectory_extractor/fa_probes.py`
- Modify: `src/trajectory_extractor/fa_cli.py`
- Test: `tests/test_fa_probes.py`

**Interfaces:**
- Produces `SelectionManifest`, `ProbeResult`, `fit_selection`, `evaluate_probe_test_once`, and `audit_sae_transfer`.

- [ ] **Step 1: Write failing leakage and selection tests**

```python
def test_transform_and_estimator_fit_only_on_mechanism_train(tracked_estimator):
    fit_selection(train_rows(), validation_rows(), estimators=[tracked_estimator])
    assert tracked_estimator.fit_ids == set(train_ids())
    assert not tracked_estimator.fit_ids & set(validation_ids() + probe_test_ids())

def test_selection_is_frozen_before_probe_test_and_never_refit(tmp_path):
    selection = fit_and_seal_selection(train(), validation(), store(tmp_path), config())
    result = evaluate_probe_test_once(selection, unlock_probe_test(tmp_path), probe_test())
    assert result.selection_hash == selection.sha256
    assert result.refit_performed is False
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_fa_probes.py -q`

- [ ] **Step 3: Implement registered baselines**

Implement `surface`, `output_margin`, `residual_static`, `sae_1_sparse`, `sae_small_sparse`, `static_plus_dynamics`, and `final_layer_excluded`. Fit scalers/PCA/selectors on `mechanism_train` only. Select layer, anchor, estimator, regularization, and threshold on `locked_validation` only.

- [ ] **Step 4: Implement H3-H6 metrics**

Report AUROC, balanced accuracy, log loss, calibration, per-condition worst case, held-out entity/template/relation transfer, H3/H4 Holm correction, H5 nested log-loss improvement of at least 2%, and H6 improvement of at least 1%.

- [ ] **Step 5: Implement full-selection nulls**

Label permutation reruns anchor/layer/feature/estimator selection. Layer-order and random-map nulls preserve shape and norms. Output-aligned controls use the exact registered 11-dimensional output subspace.

- [ ] **Step 6: Implement SAE transfer gate**

```python
def audit_sae_transfer(original_loss, reconstructed_loss, ablated_loss, finite_fraction):
    recovery = (ablated_loss - reconstructed_loss) / (ablated_loss - original_loss)
    return SAEGate(passed=finite_fraction >= 0.95 and recovery >= 0.70, recovery=recovery)
```

Failed SAE transfer is reported and nonblocking; residual baselines remain required.

- [ ] **Step 7: Add selection/evaluation commands and verify GREEN**

Commands: `fa-fit-probes`, `fa-seal-selection`, `fa-unlock-endpoint`, `fa-evaluate-behavior-test`, and `fa-evaluate-probe-test`.

Run: `.venv/bin/python -m pytest tests/test_fa_probes.py tests/test_fa_artifacts.py tests/test_fa_cli.py -q`

- [ ] **Step 8: Commit**

```bash
git add src/trajectory_extractor/fa_probes.py src/trajectory_extractor/fa_cli.py tests/test_fa_probes.py tests/test_fa_cli.py
git commit -m "feat: add familiarity answerability mechanistic decoding"
```

## Task 9: F2B Prefill-Only Causal Intervention

**Files:**
- Create: `src/trajectory_extractor/fa_interventions.py`
- Modify: `src/trajectory_extractor/fa_cli.py`
- Test: `tests/test_fa_interventions.py`

**Interfaces:**
- Produces `PatchSpec`, `InterventionSelection`, `run_prefill_patch`, `select_intervention`, and `evaluate_intervention_test_once`.

- [ ] **Step 1: Write failing prefill and control tests**

```python
def test_patch_changes_only_registered_prefill_position(fake_model):
    outcome = run_prefill_patch(fake_model, paired_examples(), patch_spec())
    assert outcome.changed_positions == {(patch_spec().layer, patch_spec().position)}
    assert outcome.decode_hook_calls == 0

def test_required_controls_are_norm_matched():
    controls = build_controls(direction(), examples(), seed=20260722)
    assert set(controls) == {
        "orthogonal", "shuffled", "norm_matched_random", "sign_reversed",
        "wrong_anchor", "reverse_direction", "cross_entity"
    }
    assert all(np.isclose(np.linalg.norm(v), np.linalg.norm(direction())) for v in controls.values())
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_fa_interventions.py -q`

- [ ] **Step 3: Implement same-string activation interchange**

Patch only the selected layer/anchor during prefill. The answer prefix before intervention is byte-identical. Hooks unregister before `generate()` decoding begins. Same-example high-to-low and low-to-high are primary; direction steering is secondary.

- [ ] **Step 4: Implement validation-only selection and H7/H8 gates**

Layer, anchor, direction, and alpha are selected on locked validation and sealed before `intervention_test`. Require bidirectional average effect of at least 5 percentage points with intervals excluding zero, at least 2 points beyond random/cross-entity controls, target-bound accuracy loss no more than 5 points, and unrelated refusal/invalid-format shifts no more than 3 points.

- [ ] **Step 5: Add commands and verify GREEN**

Commands: `fa-run-interventions` and `fa-evaluate-intervention-test`.

Run: `.venv/bin/python -m pytest tests/test_fa_interventions.py tests/test_fa_cli.py -q`

- [ ] **Step 6: Commit**

```bash
git add src/trajectory_extractor/fa_interventions.py src/trajectory_extractor/fa_cli.py tests/test_fa_interventions.py tests/test_fa_cli.py
git commit -m "feat: add gated familiarity answerability patching"
```

## Task 10: Optional F3 Circuit Adapter and Fidelity Audit

**Files:**
- Create: `src/trajectory_extractor/fa_circuits.py`
- Modify: `src/trajectory_extractor/fa_cli.py`
- Test: `tests/test_fa_circuits.py`

**Interfaces:**
- Produces `CircuitTracerAdapter` protocol, `select_circuit_cases`, `audit_circuit_fidelity`, and `CircuitGate`.

- [ ] **Step 1: Write failing deterministic selection and fidelity tests**

```python
def test_case_selection_uses_dedicated_circuit_dev_only():
    cases = select_circuit_cases(circuit_dev_rows(), max_quartets=12)
    assert all(case.split == "circuit_dev" for case in cases)
    assert not {c.example_id for c in cases} & protected_test_ids()

def test_fidelity_gate_requires_distribution_and_perturbation_agreement():
    gate = audit_circuit_fidelity(metrics(distribution=.81, perturbation=.61, sign=.76))
    assert gate.passed
    assert not audit_circuit_fidelity(metrics(distribution=.81, perturbation=.59, sign=.76)).passed
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_fa_circuits.py -q`

- [ ] **Step 3: Implement adapter and raw-attempt ledger**

The optional dependency is imported only inside the production adapter. Every graph attempt records success/failure, parameters, case hash, replacement distribution, error-node metrics, perturbation effects, and artifact hashes.

- [ ] **Step 4: Implement registered gates and constrained claim**

Require proxy Spearman >= 0.80, next-token distribution Spearman >= 0.80, node-perturbation Spearman >= 0.60, and sign concordance >= 0.75. Graphs remain prompt-local hypotheses and require original-model intervention support.

- [ ] **Step 5: Add commands and verify GREEN**

Commands: `fa-select-circuit-cases` and `fa-audit-circuit-fidelity`.

Run: `.venv/bin/python -m pytest tests/test_fa_circuits.py tests/test_fa_cli.py -q`

- [ ] **Step 6: Commit**

```bash
git add src/trajectory_extractor/fa_circuits.py src/trajectory_extractor/fa_cli.py tests/test_fa_circuits.py tests/test_fa_cli.py
git commit -m "feat: add optional circuit fidelity followup"
```

## Task 11: Claims, Figures, Release Builder, and Negative Results

**Files:**
- Create: `src/trajectory_extractor/fa_report.py`
- Modify: `src/trajectory_extractor/fa_cli.py`
- Test: `tests/test_fa_report.py`

**Interfaces:**
- Produces `recompute_claim_ladder`, `build_release_bundle`, `build_report`, and four registered figure builders.

- [ ] **Step 1: Write failing claim-recomputation tests**

```python
def test_report_never_trusts_stored_supported_boolean(tmp_path):
    metrics = canonical_metrics(h1_interval=(-.02, .04), stored_supported=True)
    report = build_report(metrics, output=tmp_path / "report.md")
    assert "H1 not supported" in report.read_text()

def test_failed_or_missing_f2b_is_visible_not_omitted(tmp_path):
    report = build_report(core_metrics(), f2b=None, output=tmp_path / "report.md")
    assert "F2B: skipped" in report.read_text()
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_fa_report.py -q`

- [ ] **Step 3: Implement canonical claim ladder and tables**

Recompute behavioral interaction, non-inferiority, decodability, incremental prediction, dynamics, intervention, and circuit claims from metric values and registered thresholds. Include denominator/missingness, all invalid classes, null distributions, OOD rows, SAE failure, and graph failures.

- [ ] **Step 4: Implement release checksums**

The release builder copies only an explicit allowlist, verifies each source artifact against lineage, writes `MANIFEST.json`, and writes a deterministic top-level SHA-256 manifest. Large external shards are represented by immutable retrieval records.

- [ ] **Step 5: Add `fa-build-report` and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_fa_report.py tests/test_fa_cli.py -q`

- [ ] **Step 6: Commit**

```bash
git add src/trajectory_extractor/fa_report.py src/trajectory_extractor/fa_cli.py tests/test_fa_report.py tests/test_fa_cli.py
git commit -m "feat: report familiarity answerability evidence"
```

## Task 12: Colab Orchestration, Dependency Locks, Research Documents, and Full Verification

**Files:**
- Create: `docs/familiarity_answerability_runbook.md`
- Create: `docs/familiarity_answerability_claims.md`
- Create: `requirements/fa-core.lock`
- Create: `requirements/fa-circuits.lock`
- Create: `notebooks/06_familiarity_answerability_colab.ipynb`
- Create: `notebooks/07_familiarity_answerability_analysis.ipynb`
- Create: `notebooks/08_familiarity_answerability_circuits.ipynb`
- Create: `tests/test_fa_notebook_contract.py`
- Create: `tests/test_fa_end_to_end.py`
- Modify: `README.md`

**Interfaces:**
- Notebooks invoke only tested CLI commands.
- The deterministic fake-runner test exercises F1, F2A, F2B gating, reporting, interruption, resume, and release creation.

- [ ] **Step 1: Write failing notebook and end-to-end contract tests**

```python
def test_colab_notebook_has_preflight_drive_resume_and_no_scientific_logic():
    notebook = json.loads(COLAB_NOTEBOOK.read_text())
    source = "\n".join(cell_source(notebook))
    assert "fa-audit-manifest" in source
    assert "fa-run-generation" in source
    assert "fa-extract-activations" in source
    assert "--resume" in source
    assert "LogisticRegression(" not in source

def test_fake_end_to_end_builds_auditable_core_release(tmp_path):
    result = run_fake_core_pipeline(tmp_path)
    assert result.behavior_endpoint.state == "closed"
    assert result.probe_endpoint.state == "closed"
    assert (result.release / "MANIFEST.json").is_file()
    assert verify_release(result.release)
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_fa_notebook_contract.py tests/test_fa_end_to_end.py -q`

- [ ] **Step 3: Create orchestration-only notebooks**

The Colab notebook checks Hugging Face authentication/license, GPU type and memory, free RAM/disk, pinned revisions, dependency hashes, Drive destination, and existing verified shards. It stops cleanly on preflight failure and resumes shard-by-shard. The analysis notebook runs locally from sealed artifacts. The circuit notebook is clearly optional and gate-checked.

- [ ] **Step 4: Freeze documents and dependency profiles**

The preregistration mirrors the approved hypotheses and thresholds. The runbook contains exact local/Colab commands, crash recovery, human-audit instructions, and endpoint-opening procedure. The claims document maps every allowed sentence to a machine gate. Core dependencies exclude `circuit-tracer`; optional dependencies pin its release/commit and model assets.

- [ ] **Step 5: Run focused end-to-end verification**

Run: `.venv/bin/python -m pytest tests/test_fa_*.py -q`

Expected: all FA tests pass, including interruption/resume and release checksum verification.

- [ ] **Step 6: Run complete regression suite**

Run: `.venv/bin/python -m pytest -q`

Expected: no failures and at least the original `362 passed, 3 skipped` plus all FA tests.

- [ ] **Step 7: Run static repository checks**

Run: `git diff --check`

Run: `rg -n "TODO|TBD|placeholder|artificial intuition|exact jailbreak|human-like" src/trajectory_extractor/fa_* docs/familiarity_answerability_* notebooks/0[678]_*.ipynb`

Expected: no implementation placeholders or unsupported claim language.

- [ ] **Step 8: Commit**

```bash
git add README.md docs/familiarity_answerability_runbook.md docs/familiarity_answerability_claims.md requirements/fa-*.lock notebooks/06_familiarity_answerability_colab.ipynb notebooks/07_familiarity_answerability_analysis.ipynb notebooks/08_familiarity_answerability_circuits.ipynb tests/test_fa_notebook_contract.py tests/test_fa_end_to_end.py
git commit -m "docs: package familiarity answerability fellowship study"
```

## Final Review

- [ ] Generate a whole-branch review package from merge base `15aedd4` through `HEAD`.
- [ ] Dispatch an independent scientific/spec reviewer and an independent code-quality reviewer.
- [ ] Fix all Critical and Important findings with focused tests and re-review.
- [ ] Run `.venv/bin/python -m pytest -q` and `git diff --check` again.
- [ ] Record what is implemented versus what still requires live Gemma, human-audit, Colab, or circuit execution. Do not describe infrastructure completion as an experimental result.
