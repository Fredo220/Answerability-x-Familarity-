# Graph Report - fa-source-v6-development  (2026-07-26)

## Corpus Check
- 51 files · ~146,165 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2258 nodes · 9241 edges · 59 communities (57 shown, 2 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 1300 edges (avg confidence: 0.63)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2dc9dd5f`
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
- test_fa_features.py
- _HFSelectedCapture
- RawInterventionOutcome
- test_fa_notebook_contract.py
- TrackingRunner
- _Capture
- ColabSplitCheckpointStore
- CircuitGateEvidence
- test_fa_graphify_contract.py
- build_release_bundle
- test_fa_development_source.py
- FAExample
- simulate_interaction_power
- SourceRecord
- test_fa_confirmatory_source.py
- NullSelectionResult
- output_feature_vector
- build_manifest
- _HFSelectedCapture
- HFTeacherForcedScorer
- _assemble_screened_matches
- build_factorial_examples
- test_fa_naturalness.py
- test_fa_notebook_contract.py
- FAManifest
- _screening_reserve_per_domain
- TrackingRunner
- _Capture

## God Nodes (most connected - your core abstractions)
1. `FAArtifactStore` - 262 edges
2. `FAConfig` - 132 edges
3. `FAExample` - 77 edges
4. `SelectionManifest` - 72 edges
5. `ProbeRow` - 70 edges
6. `EntityMatch` - 65 edges
7. `CandidateEntity` - 62 edges
8. `ProbeBundleResult` - 51 edges
9. `SameStringSealEvidence` - 51 edges
10. `BehavioralMetrics` - 50 edges

## Surprising Connections (you probably didn't know these)
- `_Capture` --uses--> `HFSelectedPositionRunner`  [INFERRED]
  tests/test_fa_activations.py → src/trajectory_extractor/fa_activations.py
- `FakeRunner` --uses--> `HFSelectedPositionRunner`  [INFERRED]
  tests/test_fa_activations.py → src/trajectory_extractor/fa_activations.py
- `NeverRunRunner` --uses--> `HFSelectedPositionRunner`  [INFERRED]
  tests/test_fa_activations.py → src/trajectory_extractor/fa_activations.py
- `RecordingLayer` --uses--> `HFSelectedPositionRunner`  [INFERRED]
  tests/test_fa_activations.py → src/trajectory_extractor/fa_activations.py
- `TrackingRunner` --uses--> `HFSelectedPositionRunner`  [INFERRED]
  tests/test_fa_activations.py → src/trajectory_extractor/fa_activations.py

## Import Cycles
- None detected.

## Communities (59 total, 2 thin omitted)

### Community 0 - "FAExample"
Cohesion: 0.07
Nodes (64): _before_final_open(), _canonical_json(), ClosedEndpointMetrics, _endpoint(), FAArtifactStore, _file_identity(), _FileIdentity, _fsync_descriptor() (+56 more)

### Community 1 - "fa_features.py"
Cohesion: 0.04
Nodes (87): _argument_error(), _build_manifest(), _confirmatory_index_record(), _design_sha256(), _prepare_power_audit(), Fill every registered split/domain quota from accepted pairs in hash order., register_fa_subcommands(), _screening_required_count() (+79 more)

### Community 2 - "FAArtifactStore"
Cohesion: 0.08
Nodes (56): ArgumentParser, _emit(), main(), _parser(), Any, Minimal Colab entrypoint that avoids importing unrelated study stacks., Any, Path (+48 more)

### Community 3 - "test_fa_cli.py"
Cohesion: 0.08
Nodes (21): ConfirmatoryInterventionExecutor, ExecutedUnrelatedEvidence, GateArtifactEvidence, InterventionCandidate, InterventionMetrics, Adapter that executes one preregistered raw intervention trial., Validation-frozen construct and generic-confidence tolerances., Deeply frozen summary derived from validation or raw test outcomes. (+13 more)

### Community 4 - "fa_scoring.py"
Cohesion: 0.10
Nodes (78): _answer_key(), _canonical_bytes(), _canonical_sha256(), _current_git_commit(), development_screening_parser_sha256(), evaluate_instrument_readiness(), _item_row(), _jsonl_bytes() (+70 more)

### Community 5 - "fa_runtime.py"
Cohesion: 0.08
Nodes (40): Cell, Enum, PatchOutcome, PatchRow, ReadoutSnapshot, _ScoringExample, behavioral_gate(), BehavioralMetrics (+32 more)

### Community 6 - "test_fa_interventions.py"
Cohesion: 0.09
Nodes (56): _applied_intervention(), AppliedIntervention, _artifact_run_id(), _behavioral_metrics(), behavioral_metrics_from_record(), _bind_activations_to_manifest(), _bind_pairs_to_endpoint(), _bind_unrelated_prompts() (+48 more)

### Community 7 - "test_fa_probes.py"
Cohesion: 0.09
Nodes (62): _CandidateDataError, _completed_record_matches_request(), _example_hash(), _expected_completed_record(), _generation_identity(), _generation_record(), HFModelRunner, _json_value() (+54 more)

### Community 8 - "SelectionManifest"
Cohesion: 0.07
Nodes (57): _array_hash(), _bootstrap_summary(), _candidate(), _canonical_f2a_metrics_row(), _canonical_hash(), _control_source(), _difference_in_means(), _evaluate() (+49 more)

### Community 9 - "fa_cli.py"
Cohesion: 0.08
Nodes (66): _before_release_publish(), _before_source_open(), _canonical_intervention_gates(), _compact_json(), _create_staging_directory_at(), _directory_open_flags(), _finite_number(), _format_probe_metrics() (+58 more)

### Community 10 - "fa_interventions.py"
Cohesion: 0.09
Nodes (66): audit_sae_transfer(), f2a_selection_bundle_hash(), _bundle_authorization(), _evaluate(), _null_result(), _production_cv_fixture(), _prompt_capability_record(), _rows() (+58 more)

### Community 11 - "fa_probes.py"
Cohesion: 0.11
Nodes (60): _analyze_pilot_activations(), _assemble_screened_matches(), _audit_confirmatory_match_pool(), _audit_manifest(), _build_evidence_report(), dispatch_fa(), _evaluate_behavior_test(), _evaluate_probe_test() (+52 more)

### Community 12 - "test_fa_activations.py"
Cohesion: 0.12
Nodes (54): PCA, _auroc_interval(), _best_log_loss(), _calculate_probe_result(), _calibration_error(), _class_probabilities(), compute_classification_metrics(), _crossed_bootstrap_interval() (+46 more)

### Community 13 - "test_fa_report.py"
Cohesion: 0.11
Nodes (56): _compile_naturalness_ratings(), _finalize_naturalness_adjudication(), _validate_revision(), audit_naturalness_manifest(), naturalness_rating_passes(), NaturalnessRating, Require independent blinded ratings and registered third-rater adjudication., Return the preregistered per-rater naturalness verdict. (+48 more)

### Community 14 - "fa_entities.py"
Cohesion: 0.14
Nodes (55): build_registered_figures(), Recompute every allowed claim from metrics, never stored booleans., Build registered figures directly from canonical typed evidence., recompute_claim_ladder(), _behavior(), _binary(), _capture_figure_semantics(), _cells() (+47 more)

### Community 15 - "fa_report.py"
Cohesion: 0.11
Nodes (45): _activation_hash_payload(), _activation_index_row(), ActivationShard, _anchor_provenance_payload(), _anchor_record_from_index(), AnchorRecord, _canonical_example_content(), _canonical_json() (+37 more)

### Community 16 - "ValueError"
Cohesion: 0.11
Nodes (34): _candidate_manifest_sha256(), _config_runtime_hashes(), _load_verified_screening_completions(), _matchable_screening_candidates(), _matching_policy_sha256(), _prompt_subset_sha256(), Keep candidates with the complete registered exact-surface reserve., _record_value() (+26 more)

### Community 17 - "AppliedIntervention"
Cohesion: 0.13
Nodes (41): _allocate_codes(), audit_dataset(), _build_same_string_row(), _check_code_position(), _check_code_vocabulary(), _check_counterbalance(), _check_entity_isolation(), _check_factorial_balance() (+33 more)

### Community 18 - "Any"
Cohesion: 0.06
Nodes (27): LogisticRegression, _candidate_rank(), _decoding_gate(), evaluate_f2a_gates(), F2AGates, FrozenProbeModel, _full_selection_null_p(), GateCriterion (+19 more)

### Community 19 - "evaluate_f2a_gates"
Cohesion: 0.13
Nodes (38): MonkeyPatch, _atomic_write(), _canonical_bytes(), _decode_object(), DevelopmentCheckpointMirror, _load_verified_metadata(), _member_sha256_map(), _publish_immutable_bytes() (+30 more)

### Community 20 - "CandidateEntity"
Cohesion: 0.17
Nodes (30): cross_resample(), crossed_bootstrap(), estimate_behavior(), One raw response with scoring inputs retained for reproducible analysis., Calculate frozen ITT estimands without excluding invalid outcomes., Cross-resample entity units and template families, never individual rows., Apply product multiplicities from independent entity and template draws., _scored_rows() (+22 more)

### Community 21 - "build_release_bundle"
Cohesion: 0.13
Nodes (39): _behavior_code_hashes(), build_development_domain_query(), build_development_frame(), _canonical_sha256(), _design_payload(), _load_excluded_qids(), main(), Any (+31 more)

### Community 22 - "Any"
Cohesion: 0.06
Nodes (76): ActivationRecord, The three registered positions for selected layers of one prompt only., Replay immutable exact-sequence evidence without loading the model again., _RecordedOutputScorer, build_probe_row(), build_probe_rows(), _canonical_bytes(), ExactSequenceScorer (+68 more)

### Community 23 - "NullSelectionResult"
Cohesion: 0.10
Nodes (21): _canonical_json(), _canonical_source_identities(), _digest(), evaluate_probe_bundle_once(), _frozen_null_test_transform(), ProbeSourceIdentity, ProbeTestAuthorization, _published_probe_bundle_metrics() (+13 more)

### Community 24 - "UnlockReceipt"
Cohesion: 0.16
Nodes (37): build_same_string_examples(), Build the sealed four-row contextual-familiarization block per unit., accented_entity_unit(), ChatTemplateTokenizer, config(), entity_unit(), FakeTokenizer, full_confirmatory_design() (+29 more)

### Community 25 - "PatchAudit"
Cohesion: 0.11
Nodes (28): _behavior_h1(), _behavior_h2(), _behavior_h2b(), build_report(), _circuit_claim(), CircuitGateEvidence, ClaimDecision, _f1_bootstrap_provenance_reasons() (+20 more)

### Community 26 - "UnrelatedCapabilityPrompt"
Cohesion: 0.09
Nodes (14): UnlockReceipt, BinaryMetrics, CandidateScore, compute_binary_metrics(), CrossConditionRotationResult, CrossConditionTransferSummary, DistractorFamiliarityCellResult, _freeze_mapping() (+6 more)

### Community 27 - "FAConfig"
Cohesion: 0.15
Nodes (28): analyze_pilot_rows(), _attach_layer_permutation_summaries(), build_pilot_analysis_rows(), _CandidateResult, _canonical_json(), _evaluate_candidate(), _fit_oof_probabilities(), _metric_record() (+20 more)

### Community 28 - "ProbeSourceIdentity"
Cohesion: 0.16
Nodes (30): Resolve registered anchors against the exact rendered chat-template bytes., resolve_registered_anchors(), example(), FakeRunner, NeverRunRunner, test_activation_extraction_rejects_nonregistered_layer_sequences(), test_activation_extraction_stores_only_registered_positions_and_selected_layers(), test_activation_resume_fails_closed_on_npz_or_index_tampering() (+22 more)

### Community 29 - "write_activation_shard"
Cohesion: 0.17
Nodes (28): Request, build_source_records_from_ranked_values(), _claim_datavalue(), _claim_entity_qids(), _claim_value_aliases(), _dedupe_aliases(), _eligible_label(), _english_label() (+20 more)

### Community 30 - "test_fa_features.py"
Cohesion: 0.33
Nodes (25): _artifact_path_from_record(), _group_ratings(), _load_f2a_selection_bundle(), _load_manifest(), _load_naturalness_matches(), _load_naturalness_packet_issuance(), _load_naturalness_submission(), _load_probe_metadata_manifest() (+17 more)

### Community 31 - "_HFSelectedCapture"
Cohesion: 0.11
Nodes (19): extract_registered_anchors(), HFSelectedPositionRunner, Model adapter that captures residual outputs only at requested positions., Run one prompt while capturing only registered positions and layers.      The ru, AddedSpecialMaskTokenizer, ContextualBoundaryTokenizer, ContextualNoOffsetTokenizer, ExtractOnlyRunner (+11 more)

### Community 32 - "RawInterventionOutcome"
Cohesion: 0.16
Nodes (24): _deterministic_assignment(), match_synthetic_entities(), A deterministic pseudonym candidate reserved for exactly one split., Build a deterministic, split-isolated, one-to-one eligible matching., SyntheticCandidate, test_screening_selection_fails_closed_on_domain_shortage_and_unknown_domain(), candidate(), entity_match() (+16 more)

### Community 33 - "test_fa_notebook_contract.py"
Cohesion: 0.16
Nodes (23): DevelopmentManifests, audit_development_source(), build_development_source_records_from_ranked_values(), _canonical_value_label_keys(), DevelopmentSourceDesign, _is_safe_development_alias(), _load_candidate_qids(), _load_source_frame() (+15 more)

### Community 34 - "TrackingRunner"
Cohesion: 0.24
Nodes (22): _binary_metrics_from_record(), _bootstrap_interval_from_record(), _candidate_score_from_record(), _criterion_from_record(), _distractor_cell_from_record(), _f2a_gates_from_record(), _frozen_model_from_record(), _frozen_rotation_from_record() (+14 more)

### Community 35 - "_Capture"
Cohesion: 0.15
Nodes (9): _deep_freeze(), ExecutedPatchEvidence, _nonempty(), _parse_decoded_patch(), PatchAudit, Adapter-observed evidence for one prefill-only intervention., Adapter output before any confirmatory label is derived., _revision() (+1 more)

### Community 36 - "ColabSplitCheckpointStore"
Cohesion: 0.15
Nodes (25): _canonical_bytes(), ColabSplitCheckpointStore, _copy_content_addressed(), Any, Path, Content-addressed Colab checkpoints for confirmatory screening shards., Persist immutable split artifacts without replacing prior checkpoints., _sha256_file() (+17 more)

### Community 37 - "CircuitGateEvidence"
Cohesion: 0.19
Nodes (21): _allowed_character_inventory(), _capitalization_pattern(), _encoded_length(), _integer(), _make_match(), _nonempty_text(), _normal_form(), _pair_id() (+13 more)

### Community 38 - "test_fa_graphify_contract.py"
Cohesion: 0.16
Nodes (13): _PatchSession, PatchSpec, PrefillPatchRunner, Protocol, Apply a full replacement at one prefill site, then decode with no hook., run_prefill_patch(), _pair(), _spec() (+5 more)

### Community 39 - "build_release_bundle"
Cohesion: 0.13
Nodes (24): build_release_bundle(), Build a no-clobber release from hashed files and verified FA artifacts., _rename_directory_noreplace(), _closed_core_store(), Path, _release_probe_rows(), _release_selections(), _report_input_metadata() (+16 more)

### Community 40 - "test_fa_development_source.py"
Cohesion: 0.15
Nodes (28): assign_development_pools(), build_manual_error_audit_packet(), compile_manual_error_audit(), materialize_development_manifests(), Assign balanced, disjoint, input-order-invariant development pools., Create development-only candidate and screening-question manifests., Create a deterministic, stratified packet without model outcome labels., Require two initial raters and a distinct adjudicator for disagreements. (+20 more)

### Community 41 - "FAExample"
Cohesion: 0.21
Nodes (20): generate_synthetic_candidates(), generate_synthetic_manifests(), main(), Any, Path, Load all split manifests, generate candidates, and verify global matching., Generate multiple unique compatible pseudonyms for every source entity., _sha256_file() (+12 more)

### Community 42 - "simulate_interaction_power"
Cohesion: 0.20
Nodes (14): _calibrated_logit_intercept(), _cluster_correction(), _joint_logit_random_effect_variances(), _logistic_normal_mean(), _logit_random_effect_sd(), _prepare_power_design(), ndarray, Run the preregistered conservative crossed-cluster interaction simulation. (+6 more)

### Community 43 - "SourceRecord"
Cohesion: 0.15
Nodes (20): assign_split_pools(), audit_materialized_source(), exclude_cross_domain_source_collisions(), filter_matchable_source_records(), materialize_manifests(), Select the first tokenizer-matchable records before split assignment., Hash the exact tokenizer-only policy used before split assignment., Remove every QID or normalized label represented in multiple domains. (+12 more)

### Community 44 - "test_fa_confirmatory_source.py"
Cohesion: 0.23
Nodes (18): build_domain_query(), build_source_records(), parse_qlever_candidates(), RankedSource, Return the frozen QID-only query for one registered entity domain., Parse a QLever result into unique ordered QIDs and three source values., Build eligible records in source rank order without assigning outcomes., _entity() (+10 more)

### Community 45 - "NullSelectionResult"
Cohesion: 0.16
Nodes (16): CandidateEntity, order_screening_questions(), Join exactly three ordered, alias-consistent questions to every candidate., A checked-in real entity with its three exact screening answer sets., One provenance-bound forced-answer question for entity recall screening., ScreeningQuestion, test_candidate_manifest_hash_binds_selection_order(), test_confirmatory_screening_selection_retains_registered_reserves() (+8 more)

### Community 46 - "output_feature_vector"
Cohesion: 0.50
Nodes (8): Random, _apply_case(), _capitalization(), _generated_word(), _normal_form(), _pseudonym(), _punctuation_shaped_word(), Pinned-tokenizer pseudonym construction for the confirmatory FA corpus.

### Community 47 - "build_manifest"
Cohesion: 0.18
Nodes (17): build_manifest(), FAManifest, _is_complete_confirmatory_design(), _manifest_sha256(), _power_audit_payload(), PowerAudit, Seal deterministic rows and fail closed for every confirmatory manifest., _recompute_confirmatory_power_audit() (+9 more)

### Community 48 - "_HFSelectedCapture"
Cohesion: 0.24
Nodes (4): _HFSelectedCapture, AbstractContextManager, _run_selected(), _transformer_layers()

### Community 49 - "HFTeacherForcedScorer"
Cohesion: 0.39
Nodes (8): _canonical_json(), _closed_endpoint_phase_hash(), _canonical_f1_producer_payload(), _canonical_f2a_producer_bundle(), _release_unscored_nulls(), test_closed_f2a_phase_rejects_reduced_noncanonical_mapping(), test_closed_phase_hashes_require_complete_canonical_producer_records(), test_generated_report_hash_binds_every_supplied_phase_and_input_bundle()

### Community 50 - "_assemble_screened_matches"
Cohesion: 0.47
Nodes (4): _freeze_json_value(), Any, _reject_duplicates(), _thaw_json_value()

### Community 51 - "build_factorial_examples"
Cohesion: 0.12
Nodes (21): VerifiedF2ASelectionBundle, VerifiedPromptManifest, _assign_distractor_units(), _balanced_same_string_families(), _build_core_row(), build_factorial_examples(), DatasetAudit, _entity_id() (+13 more)

### Community 52 - "test_fa_naturalness.py"
Cohesion: 0.44
Nodes (9): _fill_response(), _match(), _matches(), Path, test_compiler_uses_sealed_issuance_after_packet_file_is_modified(), test_compiler_writes_verifiable_ratings_artifact(), test_disagreement_requires_and_accepts_independent_adjudication(), test_rating_packets_are_deterministic_blinded_and_counterbalanced() (+1 more)

### Community 53 - "test_fa_notebook_contract.py"
Cohesion: 0.29
Nodes (6): Path, _source(), test_analysis_notebook_consumes_sealed_artifacts_and_never_trains(), test_circuit_notebook_is_explicitly_optional_and_gate_checked(), test_colab_notebook_is_orchestration_only_with_preflight_drive_and_resume(), test_colab_notebook_runs_the_frozen_source_v5_screening_before_protected_studies()

### Community 54 - "FAManifest"
Cohesion: 0.07
Nodes (30): _array_bytes(), build_controls(), _control_difference_in_means(), _finite_vector(), _hash(), InterventionPrompt, InterventionSelection, _load_f2a_selections() (+22 more)

### Community 55 - "_screening_reserve_per_domain"
Cohesion: 0.67
Nodes (3): Return the pre-outcome reserve quota registered for one screening split., _screening_reserve_per_domain(), test_confirmatory_reserve_table_is_frozen_per_split()

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FAArtifactStore` connect `FAExample` to `fa_features.py`, `FAArtifactStore`, `test_fa_cli.py`, `fa_runtime.py`, `test_fa_interventions.py`, `test_fa_probes.py`, `SelectionManifest`, `fa_cli.py`, `fa_interventions.py`, `fa_probes.py`, `test_fa_activations.py`, `test_fa_report.py`, `ValueError`, `Any`, `Any`, `NullSelectionResult`, `PatchAudit`, `UnrelatedCapabilityPrompt`, `test_fa_features.py`, `_Capture`, `ColabSplitCheckpointStore`, `test_fa_graphify_contract.py`, `build_release_bundle`, `build_factorial_examples`, `test_fa_naturalness.py`, `FAManifest`?**
  _High betweenness centrality (0.133) - this node is a cross-community bridge._
- **Why does `FAConfig` connect `fa_probes.py` to `FAExample`, `fa_features.py`, `FAArtifactStore`, `fa_scoring.py`, `test_fa_probes.py`, `test_fa_report.py`, `ValueError`, `AppliedIntervention`, `CandidateEntity`, `build_release_bundle`, `Any`, `UnlockReceipt`, `write_activation_shard`, `test_fa_features.py`, `FAExample`, `SourceRecord`, `test_fa_confirmatory_source.py`, `output_feature_vector`, `build_manifest`, `_assemble_screened_matches`, `build_factorial_examples`, `_screening_reserve_per_domain`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Why does `HFSelectedPositionRunner` connect `_HFSelectedCapture` to `fa_probes.py`, `ProbeSourceIdentity`, `fa_report.py`, `_HFSelectedCapture`, `build_factorial_examples`, `Any`, `TrackingRunner`, `_Capture`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 502 inferred relationships involving `ValueError` (e.g. with `.__post_init__()` and `_anchor_record_from_index()`) actually correct?**
  _`ValueError` has 502 INFERRED edges - model-reasoned connections that need verification._
- **Are the 73 inferred relationships involving `FAArtifactStore` (e.g. with `_RecordedOutputScorer` and `VerifiedF2ASelectionBundle`) actually correct?**
  _`FAArtifactStore` has 73 INFERRED edges - model-reasoned connections that need verification._
- **Are the 38 inferred relationships involving `FAConfig` (e.g. with `_RecordedOutputScorer` and `VerifiedF2ASelectionBundle`) actually correct?**
  _`FAConfig` has 38 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `FAExample` (e.g. with `_RecordedOutputScorer` and `VerifiedF2ASelectionBundle`) actually correct?**
  _`FAExample` has 23 INFERRED edges - model-reasoned connections that need verification._