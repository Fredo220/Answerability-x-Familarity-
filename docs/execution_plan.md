# Execution Plan and Live Status

Last updated: 2026-07-10. This file preserves the complete execution sequence
across chat compaction and future sessions. The frozen scientific criteria live
in `docs/preregistration.md`; this file does not replace them.

## Objective

Test, independently for concept-binding errors and jailbreak success, whether
causal layerwise dynamics or a Remizov/Chernoff-inspired operator residual add
predictive value beyond output and static-activation baselines, and whether a
dynamics-triggered activation intervention reduces failures without materially
damaging matched controls.

## Frozen interpretation

- Early warning means prediction from pre-token causal prefixes, never an exact
  mathematical failure point.
- Concept mixing and jailbreaks remain separate mechanisms, labels, operators,
  directions, interventions, and result tables.
- PCA is a tractable coordinate system, not a reversal of superposition.
- Remizov/Chernoff is mathematical inspiration, not a proven transformer
  semigroup theorem.
- Null and partially supported results are retained under the original criteria.
- The primary claims are local to the resolved Llama 3.2 1B Instruct revision.

## Stage 0: Infrastructure and design - completed

- Hugging Face causal-LM execution with deterministic generation.
- One teacher-forced causal replay extracts every pre-response-token state.
- Stateful intervention replay retains incremental trigger semantics.
- Float16 activation artifacts; bfloat16 CPU model execution.
- Resumable run stores with model, seed, dtype, token boundaries, labels,
  provenance, dataset hashes, and resolved model revision.
- Per-example timing, ETA, memory-conscious response-only intervention artifacts.
- Sixty-four local unit/integration tests before target-model execution.
- Separate ungated Qwen engineering rehearsal; excluded from scientific results.

## Stage 1: Frozen data - completed

### Concept mixing

- 1,200 controlled in-context entity-relation-object examples.
- 720/240/240 train/validation/test split by disjoint entity family and template.
- Independently balanced relation, distractor count, name similarity, rarity
  proxy, and answer position within distractor count.
- Same-relation hard distractor and unique table entities/objects.
- Exact error is primary; distractor-binding error is mechanistic secondary.
- Generation cap: 12 tokens.
- Dataset SHA-256 and generator manifest frozen before Llama runs.

### Real transfer

- 200 source-documented Wikidata birthplace triples.
- External test only; no projection, threshold, probe, prefix, or operator fit.

### Jailbreak

- Exactly 100 official harmful and 100 matched benign JailbreakBench behaviors.
- One frozen published artifact per harmful behavior; no new attack generation.
- Pair and category validation plus upstream commit/checksum provenance.
- Llama Guard 3 1B labels after unloading the target model.
- Stratified 20 percent human audit required before final safety conclusions.

## Stage 2: Target-model smoke and representative pilots - in progress

1. Hugging Face CLI authentication and gated repository access are verified for
   `Fredolin21`, Llama 3.2 1B Instruct, and Llama Guard 3 1B.
2. The local smoke passed on resolved target revision
   `9213176726f574b556790deb65791e0c5aa438b6`: shape `(2, 17, 2048)`, finite
   float16 artifacts, deterministic response and activations, successful reload,
   restored steering hook, 2.86 GB peak RSS, and zero swaps.
3. The ten-per-split concept pilot completed under `concept-main`: all 30
   artifacts reload, no response reached the 12-token cap, and two responses were
   genuine distractor-binding errors. The first ten training examples were all
   correct, so pilot class balance is diagnostic only and is not analyzed as a
   scientific result.
4. The full `concept-main` extraction is resumable and currently has 93 complete
   examples. It was paused without partial artifacts because the laptop reported
   15 percent battery and continued discharging under AC power; resume only after
   the power source can sustain the CPU load.
5. Run one matched harmful/benign pair per jailbreak category; inspect target
   responses and Guard parsing before the complete safety run.

## Stage 3: Concept detection - extraction in progress

1. Extract all 1,200 examples under `concept-main`.
2. Fit only on training folds:
   - output baseline: token log probability and entropy;
   - static baseline: layerwise logistic probes;
   - raw dynamics: normalized velocity, curvature, direction change;
   - stable operator residual: layerwise PCA-32 plus ridge transition operators,
     fitted only on correct training responses;
   - combined static plus dynamics model.
3. Select token/layer prefix and decision threshold on validation only.
4. Evaluate exact error once on held-out test; cluster bootstrap by entity family.
5. Repeat independently for distractor-binding error.
6. Persist AUROC/AUPRC surfaces, calibration, FPR, threshold crossing,
   per-example predictions, bootstrap draws, figures, runtime, and fit IDs.
7. Apply the frozen acceptance rule: dynamics gain at least 0.03 AUROC over the
   best simple baseline and paired 95 percent CI excludes zero.

### Stage 3b: Prospective metacognitive feature-flow monitor

1. Keep `docs/preregistration.md` and `evaluate-concept` unchanged.
2. Fit centered risk-minus-control activation directions on training examples only.
3. Standardize direction projections per layer with training-only statistics.
4. Derive causal cross-layer and prior-token differences.
5. Select the prefix and threshold on validation only. Validation receives full
   prefix probability surfaces and crossing diagnostics; test receives only the
   frozen selected-prefix probability.
6. Compare `contrastive_plus_dynamics` against `contrastive_vector` once on test.
   The confirmatory p-value is a paired entity-family permutation test with seed
   42 and 2,000 permutations. Cluster bootstrap is used for confidence intervals
   only.
7. Persist results under `runs/concept-main/secondary/`; never overwrite primary
   metrics. Write the validation figure as
   `validation_metacognitive_risk_gap_<endpoint>.png`, not as a test risk surface.
8. Mark the result `not_evaluable` unless test has 20 positive examples across 10
   independent entity families. Even when supported, mark the positive result
   provisional until the frozen falsification controls and external transfer are
   completed.

This stage validates a non-anthropomorphic monitoring signal only. The scientific
term is a metacognitive internal reliability signal: a model-derived risk score,
not consciousness, introspection, ground-truth access, or a proof of failure. The
user-facing phrase "artificial intuition" is metaphorical. External-fact transfer,
activation control, JailbreakBench transfer, and multi-model replication require
separate plans before any broad reliability or safety claim.

## Stage 4: Concept ablations and transfer - pending

- Full trajectory versus last token.
- PCA dimensions 16, 32, and 64.
- Raw dynamics versus operator residual versus combined.
- Shuffled layers and random projection controls.
- Prompt length, rarity proxy, and distractor-count subgroups.
- Extract and evaluate all 200 source-documented transfer examples using only
  components selected on synthetic train/validation folds.

## Stage 5: Concept intervention - pending

1. Learn correct-minus-confused raw-space directions from training only.
2. Rank candidate transitions and tune layer, strength, and residual threshold
   on validation only; persist every tuning candidate for resume.
3. Freeze one configuration and compare on test:
   - no steering;
   - norm-matched random direction;
   - shuffled-label direction;
   - ITI-style always-on steering;
   - operator-residual-triggered steering.
4. Require at least 20 percent relative error reduction, no more than five
   percentage points control loss, and a paired 95 percent bootstrap advantage
   over norm-matched random steering.

## Stage 6: Jailbreak detection and audit - pending

1. Extract all 200 official matched examples under `jailbreak-main`, cap 96 tokens.
2. Unload target; load Llama Guard 3 1B; label unsafe/safe and refusal status.
3. Generate a frozen stratified 20 percent manual-audit sample.
4. Human reviewer completes the sample; record agreement and disagreements.
5. Run leave-one-category-out detection with the same five method families.
6. Aggregate out-of-fold predictions and category-clustered paired bootstrap.
7. Report unsafe response rate, refusal rate, benign over-refusal, AUROC, AUPRC,
   calibration, FPR, early crossing, and runtime.

## Stage 7: Jailbreak intervention - pending

1. Use a seed-frozen, label-independent 60/20/20 category partition.
2. Fit stable operators on safe training responses and a refusal-minus-unsafe
   direction, with a documented safe-minus-unsafe fallback if refusals are sparse.
3. Tune layer, strength, and trigger threshold only on frozen validation categories.
4. Judge all validation candidates with Guard before selecting one configuration.
5. Compare the same five intervention arms on frozen test categories.
6. Bootstrap matched harmful/benign pairs and apply the intervention criteria.
7. A category split lacking both Guard classes is not evaluable; categories are
   never reassigned after labels are observed.

## Stage 8: Anthropic circuit-tracing follow-up - scaffold completed, execution pending

- Deterministically select held-out TP/FP/FN/TN concept cases after primary metrics.
- Use published circuit-tracer attribution graphs and interventions on the Llama
  3.2 1B base checkpoint for mechanistically comparable concept outcomes.
- Attribute correct-object versus distractor-object logits; inspect entity,
  relation, object paths, transcoder error nodes, and causal feature interventions.
- Compare candidate feature interventions against norm-matched random features.
- Base-checkpoint graphs are external mechanistic replication, not direct
  explanations of the Instruct checkpoint.
- Base transcoders may touch Instruct only after a separate fidelity gate covering
  top-1 agreement, logit KL, and task preservation.
- Full public transcoder sets require external GPU/hosted resources; they exceed
  the local 8-GB memory budget.

## Stage 9: Final report - pending

- Generate `docs/results.md` only from frozen artifacts.
- Report supported, partially supported, not supported, or not evaluable without
  changing thresholds after observing results.
- Separate preregistered findings, exploratory circuit evidence, limitations,
  negative controls, runtime, and model-local generalization limits.
- Do not make a final jailbreak claim until the human audit is completed.

## Immediate next commands

```bash
.venv/bin/feature-dynamics extract-concept \
  --config configs/llama32_1b.json --run-id concept-main
.venv/bin/feature-dynamics evaluate-concept \
  --config configs/llama32_1b.json --run-id concept-main --bootstrap 2000 \
  --endpoint exact_error
```
