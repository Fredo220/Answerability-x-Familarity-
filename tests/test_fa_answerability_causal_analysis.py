from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trajectory_extractor.fa_answerability_causal import (
    CAUSAL_VALIDATION_LAYERS,
    CausalExpectedProvenance,
    ValidationSelection,
)
from trajectory_extractor.fa_answerability_causal_analysis import (
    BOOTSTRAP_DRAWS,
    PERMUTATION_DRAWS,
    CAUSAL_CONTROLS,
    CausalAnalysisStore,
    CausalDecision,
    CausalEvidence,
    ControlScore,
    ExecutionAuditHashes,
    GenerationClass,
    GenerationResult,
    ManipulationCheck,
    PreservationResult,
    PrimaryScore,
    BaselineScore,
    analyze_causal_evidence,
    analyze_causal_study,
    deterministic_farthest_layer,
    seal_causal_evaluation,
)


HASH = "a" * 64
SPLIT = "causal_entity_test"
EXPECTED_UNITS = {
    "causal_entity_test": tuple(f"causal-v1-unit-{index:03d}" for index in range(12, 30)),
    "causal_template_test": tuple(f"causal-v1-unit-{index:03d}" for index in range(30, 48)),
}
PRIMARY_VECTOR = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
SHUFFLED_VECTOR = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
RANDOM_VECTORS = tuple(
    tuple(1.0 if index == member + 1 else 0.0 for index in range(6))
    for member in range(5)
)


def _selection() -> ValidationSelection:
    payload = {
        "layer_id": 6,
        "multiplier": 0.5,
        "mean_bidirectional_effect": 0.1,
        "direction_sha256": "b" * 64,
        "corpus_sha256": "c" * 64,
        "direction_bundle_sha256": "d" * 64,
        "model_sha256": "e" * 64,
        "tokenizer_sha256": "f" * 64,
    }
    from trajectory_extractor.fa_answerability_causal import _sha256

    return ValidationSelection(selection_sha256=_sha256(payload), **payload)


def _provenance(selection: ValidationSelection) -> CausalExpectedProvenance:
    return CausalExpectedProvenance(
        corpus_sha256=selection.corpus_sha256,
        direction_bundle_sha256=selection.direction_bundle_sha256,
        direction_hashes={
            layer: (selection.direction_sha256 if layer == selection.layer_id else f"{index + 1:x}" * 64)
            for index, layer in enumerate(CAUSAL_VALIDATION_LAYERS)
        },
        model_sha256=selection.model_sha256,
        tokenizer_sha256=selection.tokenizer_sha256,
    )


def _seal(*, runtime_sha256="1" * 64):
    return seal_causal_evaluation(
        selection=_selection(),
        expected_provenance=_provenance(_selection()),
        runtime_sha256=runtime_sha256,
        output_contract_sha256="2" * 64,
        random_seeds=(11, 12, 13, 14, 15),
        expected_unit_ids_by_split=EXPECTED_UNITS,
        primary_vector=PRIMARY_VECTOR,
        shuffled_vector=SHUFFLED_VECTOR,
        frozen_label_permutation_sha256="3" * 64,
        random_vectors=RANDOM_VECTORS,
    )


def _generation(*, copied: bool = False, valid: bool = True) -> GenerationResult:
    return GenerationResult(
        response_class=GenerationClass.CORRECT_CODE if valid else GenerationClass.INVALID,
        format_valid=valid,
        copied_from_other_unit=copied,
    )


def _baseline(unit_id: str, exposure: str, answerability: str, seal) -> BaselineScore:
    margin = 0.4 if answerability == "target_bound" else -0.4
    return BaselineScore(
        unit_id=unit_id,
        split=SPLIT,
        exposure=exposure,
        answerability=answerability,
        raw_margin=margin,
        length_normalized_margin=margin / 2,
        generation=_generation(),
        audit=ExecutionAuditHashes.for_control(seal, "no_intervention"),
    )


def _intervention_score(kind, unit_id, exposure, answerability, effect, seal, member=None):
    baseline = _baseline(unit_id, exposure, answerability, seal)
    sign = 0 if kind == "no_intervention" else (1 if answerability == "target_unbound" else -1)
    if kind == "sign_reversed":
        sign *= -1
    signed_effect = sign * effect
    score_type = PrimaryScore if kind == "primary" else ControlScore
    keyword = {"control": kind} if kind != "primary" else {}
    return score_type(
        unit_id=unit_id,
        split=SPLIT,
        exposure=exposure,
        answerability=answerability,
        raw_margin=baseline.raw_margin + signed_effect,
        length_normalized_margin=(baseline.raw_margin + signed_effect) / 2,
        generation=_generation(),
        audit=ExecutionAuditHashes.for_control(seal, kind, member=member),
        sign=sign,
        control_member=member,
        **keyword,
    )


def _evidence(*, primary_effect=0.4, control_effect=0.05) -> CausalEvidence:
    seal = _seal()
    baselines = []
    primary = []
    controls = []
    manipulation = []
    for unit_id in EXPECTED_UNITS[SPLIT]:
        for exposure in ("low_exposure", "high_exposure"):
            for answerability in ("target_unbound", "target_bound"):
                baselines.append(_baseline(unit_id, exposure, answerability, seal))
                primary.append(
                    _intervention_score(
                        "primary", unit_id, exposure, answerability, primary_effect, seal
                    )
                )
                for control in CAUSAL_CONTROLS:
                    control_scale = {
                        "no_intervention": 0.2,
                        "sign_reversed": 0.3,
                        "label_shuffled_direction": 0.4,
                        "norm_matched_random": 0.5,
                        "wrong_anchor": 0.6,
                        "wrong_layer": 0.7,
                    }[control]
                    if control == "norm_matched_random":
                        for random_member in range(5):
                            controls.append(
                                _intervention_score(
                                    control,
                                    unit_id,
                                    exposure,
                                    answerability,
                                    control_effect * control_scale,
                                    seal,
                                    member=random_member,
                                )
                            )
                    else:
                        controls.append(
                            _intervention_score(
                                control,
                                unit_id,
                                exposure,
                                answerability,
                                control_effect * control_scale,
                                seal,
                            )
                        )
        manipulation.append(
            ManipulationCheck(unit_id=unit_id, split=SPLIT, primary_projection_delta=0.2)
        )
    return CausalEvidence(
        split=SPLIT,
        seal=seal,
        baselines=tuple(baselines),
        primary_scores=tuple(primary),
        control_scores=tuple(controls),
        manipulation_checks=tuple(manipulation),
        preservation=PreservationResult(
            split=SPLIT,
            bound_accuracy_drop=0.0,
            unrelated_task_preserved=True,
        ),
    )


def test_positive_synthetic_fixture_satisfies_the_complete_support_rule():
    result = analyze_causal_evidence(_evidence())

    assert isinstance(result, CausalDecision)
    assert result.status == "causally_supported"
    assert result.reasons == ()
    assert result.bootstrap.draws == BOOTSTRAP_DRAWS == 10_000
    assert result.sign_flip.draws == PERMUTATION_DRAWS == 9_999
    assert result.length_normalized_sensitivity.mean_effect > 0


@pytest.mark.parametrize("split", ["causal_validation", "pooled"])
def test_missing_or_pooled_test_split_is_not_evaluable(split):
    evidence = _evidence()
    result = analyze_causal_evidence(replace(evidence, split=split))

    assert result.status == "not_evaluable"
    assert "test_split_required" in result.reasons


def test_study_decision_requires_both_unpooled_test_splits():
    result = analyze_causal_study({SPLIT: _evidence()})

    assert result.status == "not_evaluable"
    assert "both_test_splits_required" in result.reasons

    result = analyze_causal_study(
        {"causal_entity_test": _evidence(), "causal_template_test": _evidence()}
    )
    assert result.status == "not_evaluable"
    assert "split_key_mismatch" in result.reasons


def test_incomplete_factorial_unit_or_control_schedule_is_not_evaluable():
    evidence = _evidence()
    result = analyze_causal_evidence(
        replace(evidence, baselines=evidence.baselines[1:])
    )

    assert result.status == "not_evaluable"
    assert "expected_prompt_count_mismatch" in result.reasons

    result = analyze_causal_evidence(
        replace(evidence, control_scores=evidence.control_scores[:-1])
    )
    assert result.status == "not_evaluable"
    assert "incomplete_control_schedule" in result.reasons


def test_seal_requires_exact_registered_units_and_all_72_prompt_rows():
    evidence = _evidence()
    missing_unit = EXPECTED_UNITS[SPLIT][-1]
    subset = tuple(row for row in evidence.baselines if row.unit_id != missing_unit)
    result = analyze_causal_evidence(replace(evidence, baselines=subset))
    assert result.status == "not_evaluable"
    assert "expected_unit_ids_mismatch" in result.reasons

    substituted = replace(evidence.baselines[0], unit_id="causal-v1-unit-999")
    result = analyze_causal_evidence(
        replace(evidence, baselines=(substituted, *evidence.baselines[1:]))
    )
    assert result.status == "not_evaluable"
    assert "expected_unit_ids_mismatch" in result.reasons


def test_baselines_bind_the_same_execution_identity_as_intervention_rows():
    evidence = _evidence()
    changed = replace(evidence.baselines[0].audit, runtime_sha256="9" * 64)
    result = analyze_causal_evidence(
        replace(evidence, baselines=(replace(evidence.baselines[0], audit=changed), *evidence.baselines[1:]))
    )

    assert result.status == "not_evaluable"
    assert "execution_identity_mismatch" in result.reasons


def test_control_vector_artifacts_are_hash_bound_and_geometrically_verified():
    seal = _seal()
    assert set(seal.control_vector_artifact_hashes) == {
        "primary",
        "no_intervention",
        "sign_reversed",
        "label_shuffled_direction",
        "wrong_anchor",
        "wrong_layer",
        *(f"norm_matched_random:{index}" for index in range(5)),
    }
    assert seal.frozen_label_permutation_sha256 == "3" * 64
    for vector in seal.random_vectors:
        assert np.isclose(np.linalg.norm(vector), np.linalg.norm(seal.primary_vector))
        assert np.isclose(np.dot(vector, seal.primary_vector), 0.0, atol=1e-12)

    with pytest.raises(ValueError, match="norm-matched"):
        seal_causal_evaluation(
            selection=_selection(),
            expected_provenance=_provenance(_selection()),
            runtime_sha256="1" * 64,
            output_contract_sha256="2" * 64,
            random_seeds=(11, 12, 13, 14, 15),
            expected_unit_ids_by_split=EXPECTED_UNITS,
            primary_vector=PRIMARY_VECTOR,
            shuffled_vector=SHUFFLED_VECTOR,
            frozen_label_permutation_sha256="3" * 64,
            random_vectors=((2.0, 0.0, 0.0, 0.0, 0.0, 0.0), *RANDOM_VECTORS[1:]),
        )

    evidence = _evidence()
    changed = replace(
        evidence.control_scores[0].audit, control_vector_artifact_sha256="9" * 64
    )
    result = analyze_causal_evidence(
        replace(
            evidence,
            control_scores=(replace(evidence.control_scores[0], audit=changed), *evidence.control_scores[1:]),
        )
    )
    assert result.status == "not_evaluable"
    assert "execution_identity_mismatch" in result.reasons


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_sha256", "9" * 64),
        ("direction_sha256", "9" * 64),
        ("layer_id", 12),
        ("multiplier", 1.0),
        ("anchor", "target_intro_end"),
        ("random_seeds", (21, 22, 23, 24, 25)),
        ("output_contract_sha256", "9" * 64),
    ],
)
def test_changed_execution_identity_is_not_evaluable(field, value):
    evidence = _evidence()
    changed_audit = replace(evidence.primary_scores[0].audit, **{field: value})
    changed_primary = replace(evidence.primary_scores[0], audit=changed_audit)

    result = analyze_causal_evidence(
        replace(evidence, primary_scores=(changed_primary, *evidence.primary_scores[1:]))
    )

    assert result.status == "not_evaluable"
    assert "execution_identity_mismatch" in result.reasons


def test_primary_null_or_wrong_direction_is_not_supported():
    assert analyze_causal_evidence(_evidence(primary_effect=0.0)).status == "not_supported"
    assert analyze_causal_evidence(_evidence(primary_effect=-0.2)).status == "not_supported"


def test_signed_primary_and_control_schedule_is_fail_closed():
    evidence = _evidence()
    wrong_sign = replace(evidence.primary_scores[0], sign=-1)

    result = analyze_causal_evidence(
        replace(evidence, primary_scores=(wrong_sign, *evidence.primary_scores[1:]))
    )

    assert result.status == "not_evaluable"
    assert "intervention_sign_mismatch" in result.reasons


def test_tied_strongest_control_is_not_supported():
    result = analyze_causal_evidence(_evidence(control_effect=0.05))
    assert result.status == "causally_supported"

    evidence = _evidence()
    tied = []
    for row in evidence.control_scores:
        if row.control == "wrong_layer":
            delta = -0.005 if row.answerability == "target_unbound" else 0.005
            tied.append(replace(row, raw_margin=row.raw_margin + delta))
        else:
            tied.append(row)
    result = analyze_causal_evidence(replace(evidence, control_scores=tuple(tied)))
    assert result.status == "not_supported"
    assert "strongest_control_tie" in result.reasons


def test_sign_flip_format_preservation_and_copying_fail_support():
    assert "sign_flip_p_above_0_05" in analyze_causal_evidence(
        replace(
            _evidence(primary_effect=0.0),
            primary_scores=tuple(
                replace(row, raw_margin=row.raw_margin + (0.001 if row.unit_id == EXPECTED_UNITS[SPLIT][0] else 0.0))
                for row in _evidence(primary_effect=0.0).primary_scores
            ),
        )
    ).reasons

    evidence = _evidence()
    invalid = replace(evidence.primary_scores[0], generation=_generation(valid=False))
    result = analyze_causal_evidence(
        replace(evidence, primary_scores=(invalid, *evidence.primary_scores[1:]))
    )
    assert "format_gate_failed" in result.reasons

    copied = replace(evidence.primary_scores[0], generation=_generation(copied=True))
    result = analyze_causal_evidence(
        replace(evidence, primary_scores=(copied, *evidence.primary_scores[1:]))
    )
    assert "cross_unit_code_copying" in result.reasons

    result = analyze_causal_evidence(
        replace(evidence, preservation=replace(evidence.preservation, unrelated_task_preserved=False))
    )
    assert "preservation_failed" in result.reasons


def test_random_controls_are_aggregated_as_a_family_not_a_best_seed():
    evidence = _evidence()
    changed = []
    for row in evidence.control_scores:
        if row.control == "norm_matched_random" and row.control_member == 0:
            changed.append(replace(row, raw_margin=row.raw_margin + 0.3))
        else:
            changed.append(row)
    result = analyze_causal_evidence(replace(evidence, control_scores=tuple(changed)))

    assert result.control_effects["norm_matched_random"].member_count == 5
    assert result.control_effects["norm_matched_random"].mean_effect < 0.2
    assert result.strongest_control != "norm_matched_random"


def test_store_allows_same_hash_partial_resume_and_rejects_completed_or_mismatched_reuse(tmp_path):
    evidence = _evidence()
    store = CausalAnalysisStore(tmp_path)
    receipt = store.begin_or_resume(evidence.seal, SPLIT)
    resumed = store.begin_or_resume(evidence.seal, SPLIT)
    assert receipt == resumed

    completed = store.complete(evidence)
    assert completed.status == "causally_supported"
    with pytest.raises(ValueError, match="completed"):
        store.begin_or_resume(evidence.seal, SPLIT)

    different = _seal(runtime_sha256="9" * 64)
    with pytest.raises(ValueError, match="hash"):
        store.begin_or_resume(different, SPLIT)


def test_store_uses_an_atomic_exclusive_lease_for_competing_callers(tmp_path):
    evidence = _evidence()
    store = CausalAnalysisStore(tmp_path)
    with store.acquire(evidence.seal, SPLIT) as lease:
        assert lease.closed is False
        with pytest.raises(RuntimeError, match="lease"):
            store.acquire(evidence.seal, SPLIT)
    assert lease.closed is True
    with store.acquire(evidence.seal, SPLIT):
        pass


def test_deterministic_farthest_registered_layer_has_a_stable_tie_break():
    assert deterministic_farthest_layer(12) == 25
    assert deterministic_farthest_layer(18) == 0
