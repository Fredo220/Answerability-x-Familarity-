# Graph Report - .  (2026-07-23)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1684 nodes · 7002 edges · 37 communities (34 shown, 3 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 1118 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `61839216`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- FAExample
- fa_features.py
- FAArtifactStore
- test_fa_cli.py
- fa_scoring.py
- fa_runtime.py
- test_fa_interventions.py
- test_fa_probes.py
- SelectionManifest
- fa_cli.py
- fa_interventions.py
- fa_probes.py
- test_fa_activations.py
- test_fa_report.py
- fa_entities.py
- fa_report.py
- ValueError
- AppliedIntervention
- Any
- evaluate_f2a_gates
- CandidateEntity
- build_release_bundle
- Any
- NullSelectionResult
- UnlockReceipt
- PatchAudit
- UnrelatedCapabilityPrompt
- FAConfig
- ProbeSourceIdentity
- write_activation_shard
- _HFSelectedCapture
- RawInterventionOutcome
- test_fa_notebook_contract.py
- TrackingRunner
- _Capture
- CircuitGateEvidence

## God Nodes (most connected - your core abstractions)
1. `FAArtifactStore` - 235 edges
2. `FAConfig` - 96 edges
3. `FAExample` - 72 edges
4. `SelectionManifest` - 72 edges
5. `ProbeRow` - 70 edges
6. `ProbeBundleResult` - 51 edges
7. `SameStringSealEvidence` - 51 edges
8. `BehavioralMetrics` - 50 edges
9. `_rows()` - 50 edges
10. `_behavior()` - 50 edges

## Surprising Connections (you probably didn't know these)
- `test_qwen_smoke_rendering_disables_thinking_without_affecting_other_models()` --indirect_call--> `HFModelRunner`  [INFERRED]
  tests/test_fa_runtime.py → src/trajectory_extractor/fa_runtime.py
- `_Capture` --uses--> `HFSelectedPositionRunner`  [INFERRED]
  tests/test_fa_activations.py → src/trajectory_extractor/fa_activations.py
- `RecordingLayer` --uses--> `HFSelectedPositionRunner`  [INFERRED]
  tests/test_fa_activations.py → src/trajectory_extractor/fa_activations.py
- `TrackingRunner` --uses--> `HFSelectedPositionRunner`  [INFERRED]
  tests/test_fa_activations.py → src/trajectory_extractor/fa_activations.py
- `FakeRunner` --uses--> `UnlockReceipt`  [INFERRED]
  tests/test_fa_cli.py → src/trajectory_extractor/fa_artifacts.py

## Import Cycles
- None detected.

## Communities (37 total, 3 thin omitted)

### Community 0 - "FAExample"
Cohesion: 0.05
Nodes (124): _allocate_codes(), _assign_distractor_units(), audit_dataset(), _balanced_same_string_families(), _build_core_row(), build_factorial_examples(), build_manifest(), build_same_string_examples() (+116 more)

### Community 1 - "fa_features.py"
Cohesion: 0.06
Nodes (74): ActivationRecord, The three registered positions for selected layers of one prompt only., build_probe_row(), build_probe_rows(), _canonical_bytes(), ExactSequenceScorer, FeatureEvidence, _freeze() (+66 more)

### Community 2 - "FAArtifactStore"
Cohesion: 0.07
Nodes (64): _before_final_open(), _canonical_json(), ClosedEndpointMetrics, _endpoint(), FAArtifactStore, _file_identity(), _FileIdentity, _fsync_descriptor() (+56 more)

### Community 3 - "test_fa_cli.py"
Cohesion: 0.05
Nodes (75): _argument_error(), _build_manifest(), _confirmatory_index_record(), register_fa_subcommands(), _write_probe_metadata(), _write_tokenizer_pin(), Path, _SubParsersAction (+67 more)

### Community 4 - "fa_scoring.py"
Cohesion: 0.06
Nodes (61): Cell, Enum, behavioral_gate(), _cell_record(), _cell_summaries(), _contains_infrastructure_marker(), cross_resample(), crossed_bootstrap() (+53 more)

### Community 5 - "fa_runtime.py"
Cohesion: 0.08
Nodes (59): _freeze_json_value(), Any, _reject_duplicates(), _thaw_json_value(), _validate_revision(), _CandidateDataError, _completed_record_matches_request(), _example_hash() (+51 more)

### Community 6 - "test_fa_interventions.py"
Cohesion: 0.06
Nodes (61): _array_hash(), _bootstrap_summary(), _candidate(), _canonical_hash(), _control_source(), _difference_in_means(), _evaluate(), _Executor (+53 more)

### Community 7 - "test_fa_probes.py"
Cohesion: 0.09
Nodes (68): audit_sae_transfer(), evaluate_probe_bundle_once(), f2a_selection_bundle_hash(), _bundle_authorization(), _evaluate(), _null_result(), _production_cv_fixture(), _prompt_capability_record() (+60 more)

### Community 8 - "SelectionManifest"
Cohesion: 0.11
Nodes (42): ConfirmatoryInterventionExecutor, ExecutedPatchEvidence, ExecutedUnrelatedEvidence, GateArtifactEvidence, InterventionCandidate, InterventionMetrics, InterventionPair, InterventionSelection (+34 more)

### Community 9 - "fa_cli.py"
Cohesion: 0.14
Nodes (56): _artifact_path_from_record(), _design_sha256(), _evaluate_behavior_test(), _file_sha256(), _group_ratings(), _json_safe(), _load_f2a_selection_bundle(), _load_manifest() (+48 more)

### Community 10 - "fa_interventions.py"
Cohesion: 0.10
Nodes (47): _artifact_run_id(), _behavioral_metrics(), behavioral_metrics_from_record(), _bind_activations_to_manifest(), _bind_pairs_to_endpoint(), _bind_unrelated_prompts(), _bootstrap_distribution(), bootstrap_distribution_from_record() (+39 more)

### Community 11 - "fa_probes.py"
Cohesion: 0.12
Nodes (54): PCA, _auroc_interval(), _best_log_loss(), _calculate_probe_result(), _calibration_error(), _class_probabilities(), compute_classification_metrics(), _crossed_bootstrap_interval() (+46 more)

### Community 12 - "test_fa_activations.py"
Cohesion: 0.09
Nodes (44): extract_registered_anchors(), HFSelectedPositionRunner, Model adapter that captures residual outputs only at requested positions., Run one prompt while capturing only registered positions and layers.      The ru, AddedSpecialMaskTokenizer, ContextualBoundaryTokenizer, ContextualNoOffsetTokenizer, example() (+36 more)

### Community 13 - "test_fa_report.py"
Cohesion: 0.13
Nodes (57): build_registered_figures(), Recompute every allowed claim from metrics, never stored booleans., Build registered figures directly from canonical typed evidence., recompute_claim_ladder(), _behavior(), _binary(), _canonical_f1_producer_payload(), _canonical_f2a_producer_bundle() (+49 more)

### Community 14 - "fa_entities.py"
Cohesion: 0.09
Nodes (44): _allowed_character_inventory(), _capitalization_pattern(), _deterministic_assignment(), _integer(), _make_match(), match_synthetic_entities(), _nonempty_text(), _normal_form() (+36 more)

### Community 15 - "fa_report.py"
Cohesion: 0.09
Nodes (48): _behavior_h1(), _behavior_h2(), _behavior_h2b(), build_report(), _canonical_intervention_gates(), _circuit_claim(), ClaimDecision, _f1_bootstrap_provenance_reasons() (+40 more)

### Community 16 - "ValueError"
Cohesion: 0.11
Nodes (34): _activation_hash_payload(), _activation_index_row(), _anchor_provenance_payload(), _anchor_record_from_index(), AnchorRecord, _canonical_example_content(), _canonical_json(), _canonical_json_line() (+26 more)

### Community 17 - "AppliedIntervention"
Cohesion: 0.12
Nodes (18): _applied_intervention(), AppliedIntervention, _array_bytes(), build_controls(), _control_difference_in_means(), _finite_vector(), _match_norm(), _mean_interval() (+10 more)

### Community 18 - "Any"
Cohesion: 0.11
Nodes (40): _before_source_open(), _canonical_json(), _closed_endpoint_phase_hash(), _compact_json(), load_closed_f1_evidence(), load_closed_f2a_evidence(), _normalize_retrieval_records(), _open_directory_descriptor() (+32 more)

### Community 19 - "evaluate_f2a_gates"
Cohesion: 0.08
Nodes (25): LogisticRegression, _candidate_rank(), _decoding_gate(), evaluate_f2a_gates(), F2AGates, FrozenProbeModel, _full_selection_null_p(), GateCriterion (+17 more)

### Community 20 - "CandidateEntity"
Cohesion: 0.18
Nodes (28): _candidate_manifest_sha256(), _config_runtime_hashes(), _load_verified_screening_completions(), _run_screening(), _screen_entities(), _screening_answer_text(), _screening_parser_sha256(), _screening_question_manifest_sha256() (+20 more)

### Community 21 - "build_release_bundle"
Cohesion: 0.09
Nodes (32): _before_release_publish(), build_release_bundle(), _create_staging_directory_at(), _directory_open_flags(), _open_or_create_parent_descriptor_at(), Test seam immediately before the exclusive directory publication., Build a no-clobber release from hashed files and verified FA artifacts., _remove_tree_at() (+24 more)

### Community 22 - "Any"
Cohesion: 0.20
Nodes (25): _binary_metrics_from_record(), _bootstrap_interval_from_record(), _candidate_score_from_record(), _criterion_from_record(), _distractor_cell_from_record(), _f2a_gates_from_record(), _freeze_mapping(), _freeze_value() (+17 more)

### Community 23 - "NullSelectionResult"
Cohesion: 0.10
Nodes (15): _canonical_json(), _canonical_source_identities(), _digest(), _frozen_null_test_transform(), NullSelectionResult, Path, Read task-specific source identities from the sealed prompt manifest., Rerun the complete train/validation selection path for every registered null. (+7 more)

### Community 24 - "UnlockReceipt"
Cohesion: 0.07
Nodes (19): UnlockReceipt, BinaryMetrics, CandidateScore, compute_binary_metrics(), CrossConditionRotationResult, CrossConditionTransferSummary, DistractorFamiliarityCellResult, FrozenTransferRotation (+11 more)

### Community 25 - "PatchAudit"
Cohesion: 0.12
Nodes (11): _deep_freeze(), _nonempty(), _parse_decoded_patch(), PatchAudit, Adapter-observed evidence for one prefill-only intervention., Freeze the deterministic validation-selected primary intervention., _revision(), select_intervention() (+3 more)

### Community 26 - "UnrelatedCapabilityPrompt"
Cohesion: 0.15
Nodes (6): InterventionPrompt, _prefix_evidence_sha256(), A separately sealed H8 prompt, never a relabelled same-string row., One immutable prompt and its selected activation for intervention., UnrelatedCapabilityPrompt, _verify_prompt_evidence()

### Community 27 - "FAConfig"
Cohesion: 0.13
Nodes (24): _extract_activations(), _parse_layer_ids(), Fill every registered split/domain quota from accepted pairs in hash order., Replay immutable exact-sequence evidence without loading the model again., _RecordedOutputScorer, _registered_extraction_layers(), _run_generation(), _select_confirmatory_matches() (+16 more)

### Community 28 - "ProbeSourceIdentity"
Cohesion: 0.14
Nodes (21): Namespace, _audit_manifest(), _build_evidence_report(), dispatch_fa(), _evaluate_probe_test(), _fit_probes(), _load_probe_rows_manifest(), _load_prompt_task_source_identities() (+13 more)

### Community 29 - "write_activation_shard"
Cohesion: 0.27
Nodes (15): ActivationShard, load_activation_records(), Path, Paths and hashes for one verified immutable activation shard., Extract and publish a deterministic NPZ plus a verified JSONL index.      Each p, Verify both shard files, every indexed array, and their activation hashes., Verify an immutable activation shard and reconstruct its typed rows.      The ma, _read_index() (+7 more)

### Community 31 - "_HFSelectedCapture"
Cohesion: 0.24
Nodes (4): _HFSelectedCapture, AbstractContextManager, _run_selected(), _transformer_layers()

### Community 32 - "RawInterventionOutcome"
Cohesion: 0.36
Nodes (6): _derive_intervention_metrics(), _h7_crossed_bootstrap(), _oriented_effects(), One executed observation with labels derived only from decoded text., RawInterventionOutcome, _worst_directional_change()

### Community 33 - "test_fa_notebook_contract.py"
Cohesion: 0.36
Nodes (5): Path, _source(), test_analysis_notebook_consumes_sealed_artifacts_and_never_trains(), test_circuit_notebook_is_explicitly_optional_and_gate_checked(), test_colab_notebook_is_orchestration_only_with_preflight_drive_and_resume()

## Knowledge Gaps
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FAArtifactStore` connect `FAArtifactStore` to `test_fa_cli.py`, `fa_runtime.py`, `test_fa_interventions.py`, `test_fa_probes.py`, `SelectionManifest`, `fa_cli.py`, `fa_interventions.py`, `fa_probes.py`, `fa_report.py`, `AppliedIntervention`, `Any`, `evaluate_f2a_gates`, `CandidateEntity`, `build_release_bundle`, `Any`, `NullSelectionResult`, `UnlockReceipt`, `PatchAudit`, `UnrelatedCapabilityPrompt`, `FAConfig`, `ProbeSourceIdentity`, `RawInterventionOutcome`, `CircuitGateEvidence`?**
  _High betweenness centrality (0.174) - this node is a cross-community bridge._
- **Why does `HFSelectedPositionRunner` connect `test_fa_activations.py` to `TrackingRunner`, `_Capture`, `fa_cli.py`, `ValueError`, `FAConfig`, `_HFSelectedCapture`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `FAConfig` connect `FAConfig` to `FAExample`, `fa_features.py`, `FAArtifactStore`, `test_fa_cli.py`, `fa_scoring.py`, `fa_runtime.py`, `fa_cli.py`, `CandidateEntity`, `ProbeSourceIdentity`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Are the 376 inferred relationships involving `ValueError` (e.g. with `.__post_init__()` and `_anchor_record_from_index()`) actually correct?**
  _`ValueError` has 376 INFERRED edges - model-reasoned connections that need verification._
- **Are the 71 inferred relationships involving `FAArtifactStore` (e.g. with `_RecordedOutputScorer` and `VerifiedF2ASelectionBundle`) actually correct?**
  _`FAArtifactStore` has 71 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `FAConfig` (e.g. with `_RecordedOutputScorer` and `VerifiedF2ASelectionBundle`) actually correct?**
  _`FAConfig` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `FAExample` (e.g. with `_RecordedOutputScorer` and `VerifiedF2ASelectionBundle`) actually correct?**
  _`FAExample` has 20 INFERRED edges - model-reasoned connections that need verification._