# Execution Plan and Live Status

Last updated: 2026-07-12. This file preserves the complete execution sequence
across chat compaction and future sessions. The frozen scientific criteria live
in `docs/preregistration.md`; this file does not replace them.

## Objective

Test, independently for concept-binding errors and jailbreak success, whether
causal layerwise dynamics or a Remizov/Chernoff-inspired operator residual add
predictive value beyond output and static-activation baselines, and whether a
dynamics-triggered activation intervention reduces failures without materially
damaging matched controls.

## Frozen interpretation

- Early-warning timing is not inferred from cross-cell threshold crossings; no
  such claim is reported without a preregistered shared calibrated score.
- Concept mixing and jailbreaks remain separate mechanisms, labels, operators,
  directions, interventions, and result tables.
- PCA is a tractable coordinate system, not a reversal of superposition.
- Remizov/Chernoff is mathematical inspiration, not a proven transformer
  semigroup theorem.
- Null and partially supported results are retained under the original criteria.
- The primary claims are local to the resolved Llama 3.2 1B Instruct revision.

## Historical Secondary Artifact Boundary

The live frozen secondary artifacts were produced by
`04568b9f1c1629ac7f08323b1c0602843fe91f48` before the current provenance and
completion protocol. They contain no `analysis_id`, `analysis_provenance`, or
completion marker. They remain frozen legacy evidence: the current legacy
metrics guard blocks a rerun, while the tracked artifact hash and
[frozen result report](results.md) preserve the record. They are not retrofitted
or represented as modern provenance-sealed artifacts.

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

## Stage 2: Target-model smoke and representative pilots - concept completed; safety pilot pending

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
4. The full `concept-main` extraction completed with 1,200 exact artifact pairs.
   Its frozen analysis is a negative concept result; it must not be re-extracted
   or rerun to reinterpret the registered endpoint. See [the frozen results](results.md).
5. Run one matched harmful/benign pair per jailbreak category; inspect target
   responses and Guard parsing before the complete safety run.

## Stage 3: Concept detection - complete negative run

1. The 1,200 frozen `concept-main` artifact pairs were evaluated once under the
   registered train/validation/test protocol.
2. Primary `exact_error` is `not_supported`: the registered AUROC delta is
   -0.03610154356423012 with 95% CI
   [-0.05645103380084038, -0.006502990610897541].
3. Secondary `exact_error` is `not_supported`; `binding_error` is
   `not_evaluable` because it has 13 positives across 8 clusters.
4. The validation probability surface is retained for prefix selection. Its cells
   are independently fitted classifiers, so threshold-crossing timing is not
   interpretable and is not reported as an early warning diagnostic.
5. The complete result record, hashes, selected-prefix limitation, and remaining
   controls are in [docs/results.md](results.md).

### Stage 3b: Prospective metacognitive feature-flow monitor - complete negative run

1. `contrastive_plus_dynamics` was compared with `contrastive_vector` once on the
   frozen test fold using the registered paired entity-family permutation test
   (seed 42, 2,000 permutations) and cluster bootstrap confidence interval.
2. The secondary endpoint is `not_supported`; the result is synthetic-only and
   does not establish an early warning, intuition, transfer, or safety signal.
3. The selected registered prefix, token 4/layer 16, is the latest pre-token
   prefix and has observed all but the final response token. It is not early.
4. No cross-cell threshold-crossing claim is retained because the validation
   surface consists of independently fitted prefix classifiers. See
   [docs/results.md](results.md) for the frozen numeric record and pending
   controls.

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
   selected-prefix calibration, FPR, and runtime. Do not report cross-cell
   threshold-crossing timing without a preregistered shared calibrated score.

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

## Stage 9: Final report - concept result completed; safety report pending

- [docs/results.md](results.md) records the frozen negative concept result without
  changing thresholds after observing results.
- The report separates the registered findings from the exploratory full monitor,
  limitations, and remaining falsification, transfer, intervention, and safety
  controls.
- Do not make a final jailbreak claim until the human audit is completed.

## Immediate next work

Do not rerun the frozen concept extraction or registered evaluations. Remaining
work is the pending concept falsification and transfer controls, intervention
study, and the separate jailbreak extraction, audit, detection, and intervention
stages.
