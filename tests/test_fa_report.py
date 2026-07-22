from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest

import trajectory_extractor.fa_report as report_module
import trajectory_extractor.fa_interventions as interventions_module
import trajectory_extractor.fa_probes as probes_module
from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_config import CONFIRMATORY_THRESHOLDS
from trajectory_extractor.fa_interventions import (
    REQUIRED_CAUSAL_CONTROLS,
    InterventionMetrics,
    InterventionTestResult,
    ReadoutConstraints,
)
from trajectory_extractor.fa_probes import (
    BinaryMetrics,
    CrossConditionRotationResult,
    CrossedBootstrapInterval,
    CrossConditionTransferSummary,
    DEFAULT_FULL_SELECTION_NULL_SEED_HASH,
    DEFAULT_FULL_SELECTION_NULL_SEEDS,
    DistractorFamiliarityCellResult,
    F2AGates,
    GateCriterion,
    HypothesisGate,
    NullSelectionResult,
    ProbeResult,
    ProbeRowIdentity,
    SAEGate,
    TARGET_FAMILIARITY_CONDITIONS,
    TransferConditionResult,
    evaluate_f2a_gates,
)
from trajectory_extractor.fa_report import (
    CircuitFailure,
    CircuitGateEvidence,
    F1Evidence,
    F2AEvidence,
    build_registered_figures,
    build_release_bundle,
    build_report,
    recompute_claim_ladder,
    verify_release_bundle,
)
from trajectory_extractor.fa_scoring import (
    BehavioralGate,
    BehavioralMetrics,
    BootstrapDistribution,
    GateDecision,
    PercentileInterval,
    SameStringSealEvidence,
)


REGISTERED_DOMAINS = ("person", "place", "organization", "creative_work")


def _cells() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (target, distractor, answerability)
        for target in ("screened_real", "matched_synthetic")
        for distractor in ("screened_real", "matched_synthetic")
        for answerability in ("target_bound", "distractor_bound", "code_absent")
    )


def _behavior(
    *,
    interval=(0.06, 0.12),
    interaction=0.08,
    gate_status="not_supported",
    requested_draws=10_000,
    valid_draws=10_000,
    discarded_draws=0,
    bootstrap_seed=20260722,
    bootstrap_alpha=0.05,
    typed_same_string_seal=True,
):
    cells = _cells()
    metrics = BehavioralMetrics(
        "evaluable",
        (),
        {cell: 0.5 for cell in cells},
        {cell: 1.0 for cell in cells},
        {cell: 0.99 for cell in cells},
        {cell: 2 for cell in cells},
        {cell: 0 for cell in cells},
        interaction,
        -0.01,
        0.08,
        {},
    )
    bootstrap = BootstrapDistribution(
        (interaction,) * valid_draws,
        (-0.01,) * valid_draws,
        (0.08,) * valid_draws,
        PercentileInterval(interaction, *interval),
        PercentileInterval(-0.01, -0.04, 0.01),
        PercentileInterval(0.08, 0.06, 0.12),
        (len(cells) * 2,),
        bootstrap_seed,
        requested_draws=requested_draws,
        valid_draws=valid_draws,
        discarded_draws=discarded_draws,
        resampling_unit=("entity_unit_id", "template_family"),
        alpha=bootstrap_alpha,
    )
    seal = (
        SameStringSealEvidence.from_registered_block(
            source_manifest_sha256="b" * 64,
            example_ids=("same-string-1", "same-string-2"),
        )
        if typed_same_string_seal
        else None
    )
    gate = BehavioralGate(
        gate_status,
        GateDecision(gate_status, ("stored decision must not be trusted",)),
        GateDecision("not_supported", ()),
        GateDecision("not_supported", ()),
        CONFIRMATORY_THRESHOLDS,
        True,
        "a" * 64,
        "b" * 64,
        same_string_seal=seal,
    )
    return F1Evidence(metrics, bootstrap, gate)


def _binary(*, invalid=0) -> BinaryMetrics:
    return BinaryMetrics(
        "evaluable", (), 12 + invalid, 12, 0, invalid, 6, 6, 0.70, 0.60, 0.50,
        0.02, 0.5, (0, 1), ((0, 6), (1, 6)),
    )


def _transfer(task: str, metrics: BinaryMetrics) -> CrossConditionTransferSummary:
    rotations = []
    for train_condition, test_conditions in probes_module._registered_rotation_specs(task):
        condition_results = tuple(
            TransferConditionResult(
                test_condition,
                metrics,
                tuple(
                    DistractorFamiliarityCellResult(condition, metrics)
                    for condition in TARGET_FAMILIARITY_CONDITIONS
                ),
            )
            for test_condition in test_conditions
        )
        rotations.append(
            CrossConditionRotationResult(
                task,
                train_condition,
                test_conditions,
                metrics,
                condition_results,
            )
        )
    return CrossConditionTransferSummary(task, tuple(rotations))


def _scored_null(
    task: str,
    kind: str,
    seed: int,
    *,
    auroc: float = 0.40,
    h6_improvement: float | None = None,
    seed_list_sha256: str = DEFAULT_FULL_SELECTION_NULL_SEED_HASH,
) -> NullSelectionResult:
    identity = ProbeRowIdentity(f"{task}-id", "e" * 64)
    config = {
        "kind": kind,
        "seed": seed,
        "seed_list_sha256": seed_list_sha256,
        "registered_seed_count": len(DEFAULT_FULL_SELECTION_NULL_SEEDS),
    }
    null_metrics = _binary().__class__(
        "evaluable", (), 12, 12, 0, 0, 6, 6, auroc, 0.60, 0.50,
        0.02, 0.5, (0, 1), ((0, 6), (1, 6)),
    )
    instance = object.__new__(NullSelectionResult)
    values = {
        "kind": kind,
        "seed": seed,
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "selection": SimpleNamespace(
            task=task,
            sha256=hashlib.sha256(f"selection:{task}:{kind}:{seed}".encode()).hexdigest(),
            to_record=lambda: {
                "fixture": f"selection:{task}:{kind}:{seed}",
            },
            null_provenance={
                "kind": kind,
                "seed": seed,
                "seed_list_sha256": seed_list_sha256,
                "registered_seed_count": len(DEFAULT_FULL_SELECTION_NULL_SEEDS),
            },
        ),
        "max_norm_error": 0.0,
        "test_source_identities": (identity,),
        "test_transform": {"registered": True},
        "test_ids": (identity.example_id,),
        "test_row_sha256s": (identity.probe_row_sha256,),
        "test_metrics": null_metrics,
        "test_model_metrics": {},
        "test_cross_condition_transfer": (
            _transfer(task, null_metrics)
            if task in {"familiarity", "answerability"}
            else None
        ),
        "test_relative_h5_log_loss_improvement": 0.0,
        "test_relative_h6_log_loss_improvement": h6_improvement,
    }
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance


def _probe(
    task: str,
    *,
    invalid: int = 0,
    null_results: tuple[NullSelectionResult, ...] = (),
) -> ProbeResult:
    metrics = _binary(invalid=invalid)
    hypothesis = {"familiarity": "H3", "answerability": "H4", "unsupported_answer": "H5"}[task]
    primary = HypothesisGate(hypothesis, (GateCriterion("typed metric", 1.0, 0.0, ">"),))
    identity = ProbeRowIdentity(f"{task}-id", "e" * 64)
    model_metrics = {
        "surface": metrics,
        "surface_output": metrics,
        "surface_output_static": metrics,
        "surface_output_static_dynamics": metrics,
        "residual_static": metrics,
    }
    return ProbeResult(
        schema_version=3,
        task=task,
        selection_hash="c" * 64,
        authorization_sha256="d" * 64,
        endpoint_input_sha256="f" * 64,
        endpoint_input_identities_sha256=probes_module._identity_digest((identity,)),
        test_ids=(f"{task}-id",),
        test_row_sha256s=("e" * 64,),
        selected_feature_family="residual_static",
        selected_anchor=(
            "target_intro_end" if task == "familiarity" else "user_prompt_end"
        ),
        selected_layer=24,
        claim_scope="pre_output",
        selected_model_sha256="9" * 64,
        metrics=metrics,
        model_metrics=model_metrics,
        per_condition={"held_out": metrics},
        worst_condition=metrics,
        ood_transfer={
            "entity": {"held_out": metrics},
            "template": {"held_out": metrics},
            "relation": {"held_out": metrics},
            "domain": {"held_out": metrics},
        },
        worst_ood_transfer={
            "entity": metrics,
            "template": metrics,
            "relation": metrics,
            "domain": metrics,
        },
        cross_condition_transfer=(
            _transfer(task, metrics)
            if task in {"familiarity", "answerability"}
            else None
        ),
        relative_h5_log_loss_improvement=(
            0.03 if task == "unsupported_answer" else None
        ),
        relative_h6_log_loss_improvement=(
            0.015 if task == "unsupported_answer" else None
        ),
        crossed_auroc_95=CrossedBootstrapInterval(0.55, 0.78),
        h5_absolute_log_loss_difference_95=(
            CrossedBootstrapInterval(0.01, 0.05)
            if task == "unsupported_answer"
            else None
        ),
        h6_absolute_log_loss_difference_95=(
            CrossedBootstrapInterval(0.005, 0.03)
            if task == "unsupported_answer"
            else None
        ),
        primary_gate=primary,
        null_results=null_results,
    )


def _f2a(*, sae_failed=False, include_sae=True) -> F2AEvidence:
    familiarity = _probe(
        "familiarity",
        null_results=tuple(
            _scored_null("familiarity", "label_permutation", seed)
            for seed in DEFAULT_FULL_SELECTION_NULL_SEEDS
        ),
    )
    answerability = _probe(
        "answerability",
        invalid=2,
        null_results=tuple(
            _scored_null("answerability", "label_permutation", seed)
            for seed in DEFAULT_FULL_SELECTION_NULL_SEEDS
        ),
    )
    unsupported = _probe(
        "unsupported_answer",
        null_results=(
            *(
                _scored_null(
                    "unsupported_answer",
                    "layer_order",
                    seed,
                    h6_improvement=0.001,
                )
                for seed in DEFAULT_FULL_SELECTION_NULL_SEEDS
            ),
            *(
                _scored_null(
                    "unsupported_answer",
                    "random_map",
                    seed,
                    h6_improvement=0.002,
                )
                for seed in DEFAULT_FULL_SELECTION_NULL_SEEDS
            ),
        ),
    )
    gates = evaluate_f2a_gates(familiarity, answerability, unsupported)
    sae = SAEGate(1.0, 1.9 if sae_failed else 1.2, 2.0, 0.90 if sae_failed else 1.0, 0.10 if sae_failed else 0.80, ("finite fraction is below 0.95", "loss recovery is below 0.70") if sae_failed else ())
    return F2AEvidence(
        familiarity,
        answerability,
        unsupported,
        gates,
        {"sae_1_sparse": sae} if include_sae else {},
    )


def _with_f2a_nulls(
    evidence: F2AEvidence,
    task: str,
    null_results: tuple[NullSelectionResult, ...],
) -> F2AEvidence:
    results = {
        "familiarity": evidence.familiarity,
        "answerability": evidence.answerability,
        "unsupported_answer": evidence.unsupported_answer,
    }
    results[task] = replace(results[task], null_results=null_results)
    gates = evaluate_f2a_gates(
        results["familiarity"],
        results["answerability"],
        results["unsupported_answer"],
    )
    return F2AEvidence(
        results["familiarity"],
        results["answerability"],
        results["unsupported_answer"],
        gates,
        evidence.sae_gates,
    )


def _complete_bootstrap() -> dict:
    return {
        "method": "crossed_entity_unit_template_family_bootstrap",
        "seed": 20260722,
        "replicates": 10_000,
        "requested_draws": 10_000,
        "valid_draws": 10_000,
        "discarded_draws": 0,
        "resampling_unit": ["entity_unit_id", "template_family"],
        "alpha": 0.05,
        "directions": {
            "high_to_low": {
                "point_estimate": 0.08,
                "raw_interval": (0.01, 0.14),
                "raw_p": 0.01,
                "entities": ["entity-a", "entity-b"],
                "template_families": ["family-a", "family-b"],
                "holm_interval": (0.02, 0.14),
                "holm_adjusted_p": 0.02,
            },
            "low_to_high": {
                "point_estimate": 0.08,
                "raw_interval": (0.01, 0.15),
                "raw_p": 0.01,
                "entities": ["entity-c", "entity-d"],
                "template_families": ["family-a", "family-b"],
                "holm_interval": (0.01, 0.15),
                "holm_adjusted_p": 0.02,
            },
        },
    }


def _f2b(
    *,
    completed=1.0,
    random_effect=0.0,
    cross_entity_effect=0.0,
    control_effects=None,
    bootstrap_summary=None,
    metric_overrides=None,
) -> InterventionTestResult:
    if control_effects is None:
        controls = {name: (0.0, 0.0) for name in REQUIRED_CAUSAL_CONTROLS}
        controls["norm_matched_random"] = (random_effect, random_effect)
        controls["cross_entity"] = (cross_entity_effect, cross_entity_effect)
    else:
        controls = dict(control_effects)
    values = {
        "high_to_low_effect": 0.08,
        "high_to_low_interval": (0.02, 0.14),
        "low_to_high_effect": 0.08,
        "low_to_high_interval": (0.01, 0.15),
        "control_effects": controls,
        "target_bound_accuracy_change": -0.02,
        "unrelated_refusal_change": 0.01,
        "unrelated_invalid_format_change": 0.0,
        "unrelated_refusal_change_by_direction": {
            "high_to_low": 0.01,
            "low_to_high": 0.01,
        },
        "unrelated_invalid_format_change_by_direction": {
            "high_to_low": 0.0,
            "low_to_high": 0.0,
        },
        "familiarity_readout_effect": 0.05,
        "answerability_max_abs_change": 0.01,
        "entity_type_max_abs_change": 0.01,
        "generic_confidence_max_abs_change": 0.01,
        "readout_constraints": ReadoutConstraints(0.02, 0.02, 0.02, 0.02),
        "observed_domains": ("person", "place", "organization", "creative_work"),
        "passing_domains": ("person", "place", "organization"),
        "completed_fraction": completed,
        "bootstrap_summary": (
            _complete_bootstrap() if bootstrap_summary is None else bootstrap_summary
        ),
    }
    values.update(metric_overrides or {})
    metrics = InterventionMetrics(**values)
    provisional = object.__new__(InterventionTestResult)
    object.__setattr__(provisional, "selection_sha256", "f" * 64)
    object.__setattr__(provisional, "preregistration_sha256", "a" * 64)
    object.__setattr__(provisional, "example_ids", ("a", "b"))
    object.__setattr__(provisional, "metrics", metrics)
    object.__setattr__(provisional, "h7_passed", metrics.h7_passed)
    object.__setattr__(provisional, "h8_passed", metrics.h8_passed)
    object.__setattr__(provisional, "refit_performed", False)
    digest = interventions_module._result_sha256(provisional)
    return InterventionTestResult("f" * 64, "a" * 64, ("a", "b"), metrics, metrics.h7_passed, metrics.h8_passed, False, digest)


def _circuit(
    intervention: InterventionTestResult,
    *,
    attempted: int = 8,
    successful: int = 7,
    original_model_supported: bool = True,
) -> CircuitGateEvidence:
    failures = tuple(
        CircuitFailure(
            f"case-{index}",
            "graph_build",
            "trace_failed",
            hashlib.sha256(f"failure-{index}".encode()).hexdigest(),
        )
        for index in range(attempted - successful)
    )
    return CircuitGateEvidence(
        "1" * 64,
        "a" * 64,
        "2" * 64,
        "3" * 64,
        intervention.result_sha256,
        0.81,
        0.82,
        0.61,
        0.76,
        0.40,
        attempted,
        successful,
        failures,
        original_model_supported,
    )


def _closed_core_store(root: Path):
    behavior = _behavior()
    f2a = _f2a()
    store = FAArtifactStore(root)
    endpoint_manifests = {}
    allowlist = {}
    for index, endpoint in enumerate(("behavior_test", "probe_test")):
        selection = str(index + 3) * 64
        input_shard = store.write_completed_shard(
            "release-run",
            endpoint,
            "inputs",
            ({"endpoint": endpoint, "row": 1},),
            {"config_sha256": "1" * 64},
        )
        store.seal_endpoint(
            endpoint,
            (input_shard,),
            {"preregistration": "b" * 64, "selection_manifest": selection},
        )
        receipt = store.unlock_endpoint(endpoint, "b" * 64, selection)
        if endpoint == "behavior_test":
            metric_record = {
                "kind": "metrics",
                "phase": "F1",
                "metrics": behavior.metrics.to_record(),
                "bootstrap": behavior.bootstrap.to_record(),
                "gate": behavior.gate.to_record(),
            }
        else:
            metric_record = {
                "kind": "metrics",
                "metric_type": "f2a_bundle",
                "result": {
                    "results": {
                        "familiarity": f2a.familiarity.to_record(),
                        "answerability": f2a.answerability.to_record(),
                        "unsupported_answer": f2a.unsupported_answer.to_record(),
                    },
                    "gates": f2a.gates.to_record(),
                },
            }
        metrics = store.write_completed_shard(
            "release-run",
            endpoint,
            "metrics",
            (metric_record,),
            {"input_sha256": input_shard.sha256},
            record_kind="metrics",
        )
        store.mark_evaluated(receipt, metrics.data_path)
        store.close_endpoint(endpoint)
        endpoint_manifests[endpoint] = input_shard.manifest_path
        for shard in (input_shard, metrics):
            allowlist[shard.data_path.relative_to(store.root).as_posix()] = shard.sha256
            allowlist[shard.manifest_path.relative_to(store.root).as_posix()] = hashlib.sha256(
                shard.manifest_path.read_bytes()
            ).hexdigest()
    report = build_report(
        behavior=behavior,
        f2a=f2a,
        f2b=None,
        circuit=None,
        output=store.root / "report.md",
    )
    allowlist["report.md"] = hashlib.sha256(report.read_bytes()).hexdigest()
    return store, endpoint_manifests, allowlist


def _capture_figure_semantics(monkeypatch):
    captured = {}

    def capture(fig, path, plt):
        axis = fig.axes[0]
        captured[path.name] = {
            "title": axis.get_title(),
            "ylabel": axis.get_ylabel(),
            "tick_labels": tuple(label.get_text() for label in axis.get_xticklabels()),
            "bar_heights": tuple(float(patch.get_height()) for patch in axis.patches),
            "texts": tuple(text.get_text() for text in axis.texts),
        }
        plt.close(fig)
        return path

    monkeypatch.setattr(report_module, "_save_figure", capture)
    return captured


def _report_input_metadata(path: Path) -> dict:
    prefix = "<!-- fa-report-inputs:"
    line = next(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    )
    assert line.endswith(" -->")
    return json.loads(line[len(prefix) : -4])


def test_report_recomputes_claims_and_never_trusts_stored_supported(tmp_path):
    report = build_report(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=None,
        circuit=None,
        output=tmp_path / "report.md",
    )
    text = report.read_text(encoding="utf-8")

    assert "H1: supported" in text
    assert "H2b: supported" in text
    assert "F2B: skipped" in text
    assert "F3: skipped" in text
    assert "Invalid outputs: 0" in text
    assert "F2A invalid rows: 2" in text


def test_generated_report_hash_binds_every_supplied_phase_and_input_bundle(tmp_path):
    behavior = _behavior()
    f2a = _f2a()
    f2b = _f2b()
    circuit = _circuit(f2b)

    report = build_report(
        behavior=behavior,
        f2a=f2a,
        f2b=f2b,
        circuit=circuit,
        output=tmp_path / "report.md",
    )
    metadata = _report_input_metadata(report)
    f2a_core = {
        "familiarity": f2a.familiarity.to_record(),
        "answerability": f2a.answerability.to_record(),
        "unsupported_answer": f2a.unsupported_answer.to_record(),
        "gates": f2a.gates.to_record(),
    }
    expected_hashes = {
        "F1": behavior.sha256,
        "F2A": hashlib.sha256(report_module._canonical_json(f2a_core)).hexdigest(),
        "F2B": f2b.result_sha256,
        "F3": circuit.sha256,
    }
    bundle_record = {
        "schema_version": 1,
        "phase_evidence_sha256": expected_hashes,
    }

    assert metadata["schema_version"] == 1
    assert metadata["phase_evidence_sha256"] == expected_hashes
    assert metadata["report_input_bundle_sha256"] == hashlib.sha256(
        report_module._canonical_json(bundle_record)
    ).hexdigest()


@pytest.mark.parametrize(
    "behavior",
    (
        pytest.param(
            _behavior(requested_draws=101, valid_draws=101),
            id="small-bootstrap",
        ),
        pytest.param(_behavior(bootstrap_seed=17), id="unregistered-seed"),
        pytest.param(_behavior(bootstrap_alpha=0.10), id="unregistered-alpha"),
    ),
)
def test_f1_claims_require_registered_confirmatory_bootstrap_provenance(behavior):
    claims = recompute_claim_ladder(
        behavior=behavior,
        f2a=None,
        f2b=None,
        circuit=None,
    )

    assert claims["H1"].status == "not_evaluable"
    assert claims["H2"].status == "not_evaluable"
    assert claims["H2b"].status == "not_evaluable"
    assert any("bootstrap" in reason for reason in claims["H1"].reasons)


def test_h2b_requires_typed_same_string_seal_not_legacy_boolean():
    claims = recompute_claim_ladder(
        behavior=_behavior(typed_same_string_seal=False),
        f2a=None,
        f2b=None,
        circuit=None,
    )

    assert claims["H1"].status == "supported"
    assert claims["H2b"].status == "not_evaluable"
    assert any(
        "typed same-string seal" in reason for reason in claims["H2b"].reasons
    )


def test_report_marks_absent_f2b_controls_and_bootstrap_not_run(tmp_path):
    report = build_report(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=None,
        circuit=None,
        output=tmp_path / "report.md",
    )

    text = report.read_text(encoding="utf-8")
    assert "F2B controls: skipped/not_run" in text
    assert "F2B crossed bootstrap: skipped/not_run" in text


def test_report_publishes_f1_and_f2a_confirmatory_provenance(tmp_path):
    behavior = _behavior()
    report = build_report(
        behavior=behavior,
        f2a=_f2a(),
        f2b=None,
        circuit=None,
        output=tmp_path / "report.md",
    ).read_text(encoding="utf-8")

    assert "F1 bootstrap requested_draws: 10000" in report
    assert "F1 bootstrap valid_draws: 10000" in report
    assert "F1 bootstrap discarded_draws: 0" in report
    assert 'F1 bootstrap resampling_unit: ["entity_unit_id","template_family"]' in report
    assert "F1 bootstrap seed: 20260722" in report
    assert "F1 bootstrap alpha: 0.05" in report
    assert f"F1 same-string seal_sha256: {behavior.gate.same_string_seal.sha256}" in report
    for task, kind in (
        ("familiarity", "label_permutation"),
        ("answerability", "label_permutation"),
        ("unsupported_answer", "layer_order"),
        ("unsupported_answer", "random_map"),
    ):
        assert (
            f"F2A null provenance {task}/{kind}: count=99, "
            f"seed_list_sha256={DEFAULT_FULL_SELECTION_NULL_SEED_HASH}"
        ) in report


def test_report_marks_sae_not_run_when_f2a_is_absent(tmp_path):
    report = build_report(
        behavior=_behavior(),
        f2a=None,
        f2b=None,
        circuit=None,
        output=tmp_path / "report.md",
    )

    assert "SAE analysis: skipped/not_run" in report.read_text(encoding="utf-8")


def test_claim_ladder_fails_closed_on_missing_or_nonfinite_metrics():
    claims = recompute_claim_ladder(behavior=None, f2a=None, f2b=None, circuit=None)
    assert claims["H1"].status == "not_evaluable"
    assert claims["H5"].status == "not_evaluable"
    assert claims["H7"].status == "skipped"

    with pytest.raises(ValueError, match="canonical typed"):
        recompute_claim_ladder(
            behavior={"interaction": 0.2}, f2a=_f2a(), f2b=None, circuit=None
        )


def test_causal_and_circuit_claims_are_gated_and_narrow():
    f2b = _f2b()
    circuit = _circuit(f2b, attempted=8, successful=3)
    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=f2b,
        circuit=circuit,
    )

    assert claims["H1"].status == "supported"
    assert claims["H7"].status == "supported"
    assert claims["F3"].status == "supported_prompt_local_hypothesis"
    assert "universal" not in claims["F3"].claim.lower()


def test_intervention_recomputes_exact_h7_h8_criteria():
    claims = recompute_claim_ladder(behavior=_behavior(), f2a=_f2a(), f2b=_f2b(completed=0.94), circuit=None)
    assert claims["H7"].status == "not_supported"
    assert claims["H8"].status == "not_supported"
    claims = recompute_claim_ladder(behavior=_behavior(), f2a=_f2a(), f2b=_f2b(random_effect=0.07), circuit=None)
    assert claims["H7"].status == "not_supported"


def test_intervention_claim_uses_all_canonical_controls_and_crossed_bootstrap():
    controls = {name: (0.0, 0.0) for name in REQUIRED_CAUSAL_CONTROLS}
    controls["orthogonal"] = (0.07, 0.07)
    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(control_effects=controls),
        circuit=None,
    )
    assert claims["H7"].status == "not_supported"


def test_intervention_draw_audit_supports_and_is_published_verbatim(tmp_path):
    result = _f2b()
    claims = recompute_claim_ladder(
        behavior=_behavior(), f2a=_f2a(), f2b=result, circuit=None
    )
    assert claims["H7"].status == "supported"
    assert claims["H8"].status == "supported"

    report = build_report(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=result,
        circuit=None,
        output=tmp_path / "report.md",
    ).read_text(encoding="utf-8")
    assert "F2B bootstrap requested_draws: 10000" in report
    assert "F2B bootstrap valid_draws: 10000" in report
    assert "F2B bootstrap discarded_draws: 0" in report
    assert (
        'F2B bootstrap resampling_unit: ["entity_unit_id","template_family"]'
        in report
    )
    assert "F2B bootstrap seed: 20260722" in report
    assert "F2B bootstrap alpha: 0.05" in report
    assert "F2B bootstrap high_to_low:" in report
    assert "F2B bootstrap low_to_high:" in report


def test_intervention_bootstrap_accepts_honest_discarded_draw_accounting():
    bootstrap = _complete_bootstrap()
    bootstrap.update({"valid_draws": 9_999, "discarded_draws": 1})

    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(bootstrap_summary=bootstrap),
        circuit=None,
    )

    assert claims["H7"].status == "supported"
    assert claims["H8"].status == "supported"


@pytest.mark.parametrize(
    ("field", "reason"),
    (
        ("method", "method"),
        ("seed", "seed"),
        ("replicates", "replicate count"),
        ("requested_draws", "requested draws"),
        ("valid_draws", "valid draws"),
        ("discarded_draws", "discarded draws"),
        ("resampling_unit", "resampling unit"),
        ("alpha", "alpha"),
        ("directions", "directions"),
    ),
)
def test_intervention_bootstrap_requires_every_root_audit_field(field, reason):
    bootstrap = _complete_bootstrap()
    bootstrap.pop(field)
    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(bootstrap_summary=bootstrap),
        circuit=None,
    )

    assert claims["H7"].status == "not_evaluable"
    assert claims["H8"].status == "not_evaluable"
    assert reason in " ".join(claims["H7"].reasons)


@pytest.mark.parametrize(
    ("updates", "reason"),
    (
        ({"method": "row_bootstrap"}, "method"),
        ({"seed": 1}, "seed"),
        ({"replicates": 9_999}, "replicate count"),
        ({"requested_draws": 9_999}, "requested draws"),
        ({"valid_draws": 0, "discarded_draws": 10_000}, "valid draws"),
        ({"valid_draws": 9_999, "discarded_draws": 0}, "draw accounting"),
        ({"valid_draws": 10_000, "discarded_draws": 1}, "draw accounting"),
        ({"discarded_draws": -1}, "discarded draws"),
        ({"resampling_unit": ["row"]}, "resampling unit"),
        ({"alpha": 0.10}, "alpha"),
    ),
)
def test_intervention_bootstrap_rejects_every_invalid_draw_audit(updates, reason):
    bootstrap = _complete_bootstrap()
    bootstrap.update(updates)
    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(bootstrap_summary=bootstrap),
        circuit=None,
    )

    assert claims["H7"].status == "not_evaluable"
    assert reason in " ".join(claims["H7"].reasons)


@pytest.mark.parametrize("direction", ("high_to_low", "low_to_high"))
def test_intervention_bootstrap_requires_both_direction_summaries(direction):
    bootstrap = _complete_bootstrap()
    bootstrap["directions"].pop(direction)
    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(bootstrap_summary=bootstrap),
        circuit=None,
    )

    assert claims["H7"].status == "not_evaluable"
    assert "directions" in " ".join(claims["H7"].reasons)


@pytest.mark.parametrize("direction", ("high_to_low", "low_to_high"))
@pytest.mark.parametrize(
    "field",
    (
        "point_estimate",
        "raw_interval",
        "raw_p",
        "entities",
        "template_families",
        "holm_interval",
        "holm_adjusted_p",
    ),
)
def test_intervention_bootstrap_requires_every_direction_field(direction, field):
    bootstrap = _complete_bootstrap()
    bootstrap["directions"][direction].pop(field)
    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(bootstrap_summary=bootstrap),
        circuit=None,
    )

    assert claims["H7"].status == "not_evaluable"
    reasons = " ".join(claims["H7"].reasons)
    assert direction in reasons and field in reasons


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("point_estimate", 0.09, "point estimate"),
        ("raw_interval", [0.2, 0.1], "raw interval"),
        ("raw_p", 1.1, "raw p"),
        ("entities", [], "entities"),
        ("template_families", [], "template families"),
        ("holm_interval", [0.03, 0.14], "Holm interval"),
        ("holm_adjusted_p", 0.03, "Holm-adjusted p-value"),
    ),
)
def test_intervention_bootstrap_rejects_invalid_direction_evidence(field, value, reason):
    bootstrap = _complete_bootstrap()
    bootstrap["directions"]["high_to_low"][field] = value
    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(bootstrap_summary=bootstrap),
        circuit=None,
    )

    assert claims["H7"].status == "not_evaluable"
    assert reason in " ".join(claims["H7"].reasons)


@pytest.mark.parametrize("missing_domain", REGISTERED_DOMAINS)
def test_h7_requires_each_of_the_four_registered_domains(missing_domain):
    observed = tuple(domain for domain in REGISTERED_DOMAINS if domain != missing_domain)
    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(
            metric_overrides={
                "observed_domains": observed,
                "passing_domains": observed,
            }
        ),
        circuit=None,
    )

    assert claims["H7"].status == "not_supported"
    assert "all four registered domains" in " ".join(claims["H7"].reasons)


@pytest.mark.parametrize("passing_domains", tuple(combinations(REGISTERED_DOMAINS, 2)))
def test_h7_requires_at_least_three_passing_domains(passing_domains):
    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(metric_overrides={"passing_domains": passing_domains}),
        circuit=None,
    )

    assert claims["H7"].status == "not_supported"
    assert "at least three" in " ".join(claims["H7"].reasons)


@pytest.mark.parametrize(
    ("field", "boundary", "violation", "reason"),
    (
        ("familiarity_readout_effect", 0.02, 0.019, "familiarity readout"),
        ("answerability_max_abs_change", 0.02, 0.021, "answerability readout"),
        ("entity_type_max_abs_change", 0.02, 0.021, "entity-type readout"),
        (
            "generic_confidence_max_abs_change",
            0.02,
            0.021,
            "generic-confidence readout",
        ),
    ),
)
def test_h7_enforces_each_readout_tolerance_at_the_registered_boundary(
    field, boundary, violation, reason
):
    passing = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(metric_overrides={field: boundary}),
        circuit=None,
    )
    failing = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(metric_overrides={field: violation}),
        circuit=None,
    )

    assert passing["H7"].status == "supported"
    assert failing["H7"].status == "not_supported"
    assert reason in " ".join(failing["H7"].reasons)


@pytest.mark.parametrize("control", REQUIRED_CAUSAL_CONTROLS)
def test_h7_requires_and_evaluates_every_registered_control(control):
    missing = {name: (0.0, 0.0) for name in REQUIRED_CAUSAL_CONTROLS if name != control}
    missing_claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(control_effects=missing),
        circuit=None,
    )
    assert missing_claims["H7"].status == "not_evaluable"

    strong = {name: (0.0, 0.0) for name in REQUIRED_CAUSAL_CONTROLS}
    strong[control] = (0.07, 0.07)
    strong_claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(control_effects=strong),
        circuit=None,
    )
    assert strong_claims["H7"].status == "not_supported"
    assert control in " ".join(strong_claims["H7"].reasons)

    crossed_bootstrap = _complete_bootstrap()
    crossed_bootstrap["directions"]["high_to_low"]["raw_p"] = 0.06
    crossed_bootstrap["directions"]["high_to_low"]["holm_adjusted_p"] = 0.06
    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(bootstrap_summary=crossed_bootstrap),
        circuit=None,
    )
    assert claims["H7"].status == "not_supported"


def test_report_exposes_h2b_invalid_ood_and_sae_failure(tmp_path):
    report = build_report(behavior=_behavior(), f2a=_f2a(sae_failed=True), f2b=None, circuit=None, output=tmp_path / "report.md")
    text = report.read_text(encoding="utf-8")
    assert "H2b: supported" in text
    assert "F2A OOD entity" in text
    assert "SAE sae_1_sparse: failed" in text


def test_release_builder_copies_only_hash_allowlist_and_is_verifiable(tmp_path):
    source = tmp_path / "source"
    store, endpoint_manifests, allowlist = _closed_core_store(source)
    allowed = source / "report.md"
    secret = source / "protected.jsonl"
    secret.write_text('{"label":1}\n', encoding="utf-8")

    release = build_release_bundle(
        source_root=source,
        output=tmp_path / "release",
        allowlist=allowlist,
        artifact_store=store,
        core_endpoint_manifests=endpoint_manifests,
        config_hash="a" * 64,
        preregistration_hash="b" * 64,
    )

    assert (release / "report.md").read_bytes() == allowed.read_bytes()
    assert not (release / "protected.jsonl").exists()
    assert verify_release_bundle(release)
    manifest = json.loads((release / "MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == set(allowlist)
    assert set(manifest["core_endpoints"]) == {"behavior_test", "probe_test"}
    assert len(manifest["top_level_sha256"]) == 64


def test_release_rejects_report_not_bound_to_closed_endpoint_evidence(tmp_path):
    source = tmp_path / "source"
    store, endpoint_manifests, allowlist = _closed_core_store(source)
    report = build_report(
        behavior=_behavior(interaction=0.09),
        f2a=_f2a(),
        f2b=None,
        circuit=None,
        output=source / "report.md",
    )
    allowlist["report.md"] = hashlib.sha256(report.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="report.*closed endpoint"):
        build_release_bundle(
            source_root=source,
            output=tmp_path / "release",
            allowlist=allowlist,
            artifact_store=store,
            core_endpoint_manifests=endpoint_manifests,
            config_hash="a" * 64,
            preregistration_hash="b" * 64,
        )


def test_release_rejects_arbitrary_report_text_even_with_valid_closed_metrics(tmp_path):
    source = tmp_path / "source"
    store, endpoint_manifests, allowlist = _closed_core_store(source)
    report = source / "report.md"
    report.write_text("unrelated report text\n", encoding="utf-8")
    allowlist["report.md"] = hashlib.sha256(report.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="hash-bound generated report"):
        build_release_bundle(
            source_root=source,
            output=tmp_path / "release",
            allowlist=allowlist,
            artifact_store=store,
            core_endpoint_manifests=endpoint_manifests,
            config_hash="a" * 64,
            preregistration_hash="b" * 64,
        )


def test_release_rejects_tampered_report_input_bundle_hash(tmp_path):
    source = tmp_path / "source"
    store, endpoint_manifests, allowlist = _closed_core_store(source)
    report = source / "report.md"
    text = report.read_text(encoding="utf-8")
    metadata = _report_input_metadata(report)
    text = text.replace(
        metadata["report_input_bundle_sha256"],
        "0" * 64,
        1,
    )
    report.write_text(text, encoding="utf-8")
    allowlist["report.md"] = hashlib.sha256(report.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="bundle hash"):
        build_release_bundle(
            source_root=source,
            output=tmp_path / "release",
            allowlist=allowlist,
            artifact_store=store,
            core_endpoint_manifests=endpoint_manifests,
            config_hash="a" * 64,
            preregistration_hash="b" * 64,
        )


def test_release_builder_rejects_hash_mismatch_traversal_and_symlink(tmp_path):
    source = tmp_path / "source"
    store, endpoint_manifests, core_allowlist = _closed_core_store(source)
    artifact = source / "artifact.txt"
    artifact.write_text("value", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        build_release_bundle(
            source_root=source,
            output=tmp_path / "bad-hash",
            allowlist={**core_allowlist, "artifact.txt": "0" * 64},
            artifact_store=store,
            core_endpoint_manifests=endpoint_manifests,
            config_hash="a" * 64,
            preregistration_hash="b" * 64,
        )
    assert not (tmp_path / "bad-hash").exists()
    with pytest.raises(ValueError, match="relative"):
        build_release_bundle(
            source_root=source,
            output=tmp_path / "traversal",
            allowlist={
                **core_allowlist,
                "../artifact.txt": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            },
            artifact_store=store,
            core_endpoint_manifests=endpoint_manifests,
            config_hash="a" * 64,
            preregistration_hash="b" * 64,
        )
    link = source / "link.txt"
    link.symlink_to(artifact)
    with pytest.raises(ValueError, match="regular"):
        build_release_bundle(
            source_root=source,
            output=tmp_path / "link-release",
            allowlist={
                **core_allowlist,
                "link.txt": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            },
            artifact_store=store,
            core_endpoint_manifests=endpoint_manifests,
            config_hash="a" * 64,
            preregistration_hash="b" * 64,
        )
    assert not (tmp_path / "link-release").exists()


def test_release_verifier_rejects_unmanifested_symlink_directory(tmp_path):
    source = tmp_path / "source"
    store, endpoint_manifests, allowlist = _closed_core_store(source)
    artifact = source / "artifact.txt"
    artifact.write_text("value", encoding="utf-8")
    allowlist["artifact.txt"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    release = build_release_bundle(
        source_root=source,
        output=tmp_path / "release",
        allowlist=allowlist,
        artifact_store=store,
        core_endpoint_manifests=endpoint_manifests,
        config_hash="a" * 64,
        preregistration_hash="b" * 64,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (release / "unmanifested").symlink_to(outside, target_is_directory=True)

    assert not verify_release_bundle(release)


def test_release_builder_verifies_store_shards_and_rejects_open_endpoint(tmp_path):
    store = FAArtifactStore(tmp_path / "source")
    endpoint_manifests = {}
    allowlist = {}
    for index, endpoint in enumerate(("behavior_test", "probe_test")):
        selection = str(index + 3) * 64
        shard = store.write_completed_shard(
            "run", endpoint, "inputs", ({"row": 1},), {"source": "test"}
        )
        store.seal_endpoint(
            endpoint,
            (shard,),
            {"preregistration": "b" * 64, "selection_manifest": selection},
        )
        endpoint_manifests[endpoint] = shard.manifest_path
        allowlist[shard.data_path.relative_to(store.root).as_posix()] = shard.sha256
        allowlist[shard.manifest_path.relative_to(store.root).as_posix()] = hashlib.sha256(
            shard.manifest_path.read_bytes()
        ).hexdigest()
    with pytest.raises(ValueError, match="closed"):
        build_release_bundle(
            source_root=store.root,
            output=tmp_path / "release",
            allowlist=allowlist,
            artifact_store=store,
            core_endpoint_manifests=endpoint_manifests,
            config_hash="a" * 64,
            preregistration_hash="b" * 64,
        )


def test_release_builder_requires_each_data_shard_and_sidecar_pair(tmp_path):
    store, endpoint_manifests, allowlist = _closed_core_store(tmp_path / "source")
    omitted_sidecar = next(name for name in allowlist if name.endswith(".manifest.json"))
    incomplete = dict(allowlist)
    incomplete.pop(omitted_sidecar)

    with pytest.raises(ValueError, match="together"):
        build_release_bundle(
            source_root=store.root,
            output=tmp_path / "release",
            allowlist=incomplete,
            artifact_store=store,
            core_endpoint_manifests=endpoint_manifests,
            config_hash="a" * 64,
            preregistration_hash="b" * 64,
        )


def test_release_builder_rejects_symlink_swap_after_validation(tmp_path, monkeypatch):
    source = tmp_path / "source"
    store, endpoint_manifests, allowlist = _closed_core_store(source)
    artifact = source / "artifact.txt"
    artifact.write_text("trusted", encoding="utf-8")
    allowlist["artifact.txt"] = hashlib.sha256(b"trusted").hexdigest()
    outside = tmp_path / "outside.txt"
    outside.write_text("untrusted", encoding="utf-8")
    def swap(path):
        if path == artifact:
            artifact.unlink()
            artifact.symlink_to(outside)
    monkeypatch.setattr(report_module, "_before_source_open", swap)
    with pytest.raises(ValueError, match="regular"):
        build_release_bundle(
            source_root=source,
            output=tmp_path / "release",
            allowlist=allowlist,
            artifact_store=store,
            core_endpoint_manifests=endpoint_manifests,
            config_hash="a" * 64,
            preregistration_hash="b" * 64,
        )
    assert not (tmp_path / "release").exists()


def test_release_publication_does_not_clobber_a_destination_race(tmp_path, monkeypatch):
    source = tmp_path / "source"
    store, endpoint_manifests, allowlist = _closed_core_store(source)
    artifact = source / "artifact.txt"
    artifact.write_text("trusted", encoding="utf-8")
    allowlist["artifact.txt"] = hashlib.sha256(b"trusted").hexdigest()
    output = tmp_path / "release"

    def create_racing_destination(_):
        output.mkdir()
        (output / "sentinel.txt").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(report_module, "_before_release_publish", create_racing_destination, raising=False)
    with pytest.raises(FileExistsError, match="already exists"):
        build_release_bundle(
            source_root=source,
            output=output,
            allowlist=allowlist,
            artifact_store=store,
            core_endpoint_manifests=endpoint_manifests,
            config_hash="a" * 64,
            preregistration_hash="b" * 64,
        )
    assert (output / "sentinel.txt").read_text(encoding="utf-8") == "keep"


def test_release_publication_rejects_an_intermediate_symlink_swap(tmp_path, monkeypatch):
    source = tmp_path / "source"
    store, endpoint_manifests, allowlist = _closed_core_store(source)
    artifact = source / "artifact.txt"
    artifact.write_text("trusted", encoding="utf-8")
    allowlist["artifact.txt"] = hashlib.sha256(b"trusted").hexdigest()
    parent = tmp_path / "publish" / "nested"
    parent.mkdir(parents=True)
    output = parent / "release"
    detached = tmp_path / "detached"
    outside = tmp_path / "outside"
    outside.mkdir()

    def swap_intermediate_parent(_):
        parent.rename(detached)
        parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(report_module, "_before_release_publish", swap_intermediate_parent, raising=False)
    with pytest.raises(ValueError, match="output parent changed"):
        build_release_bundle(
            source_root=source,
            output=output,
            allowlist=allowlist,
            artifact_store=store,
            core_endpoint_manifests=endpoint_manifests,
            config_hash="a" * 64,
            preregistration_hash="b" * 64,
        )
    assert not (outside / "release").exists()
    assert not (detached / "release").exists()


def test_registered_figures_are_built_from_canonical_typed_evidence(tmp_path):
    paths = build_registered_figures(
        _behavior(),
        _f2a(),
        tmp_path / "figures",
        intervention=_f2b(),
    )

    assert tuple(path.name for path in paths) == (
        "figure_1_behavior.png",
        "figure_2_layer_probes.png",
        "figure_3_baselines.png",
        "figure_4_causal_intervention.png",
        "appendix_h6_nulls.png",
    )
    assert all(path.is_file() and path.stat().st_size > 1000 for path in paths)


def test_figure_4_plots_canonical_intervention_and_every_control(monkeypatch, tmp_path):
    captured = _capture_figure_semantics(monkeypatch)
    controls = {
        name: ((index + 1) / 1_000, (index + 10) / 1_000)
        for index, name in enumerate(REQUIRED_CAUSAL_CONTROLS)
    }
    result = _f2b(control_effects=controls)

    build_registered_figures(
        _behavior(),
        _f2a(),
        tmp_path / "figures",
        intervention=result,
    )

    causal = captured["figure_4_causal_intervention.png"]
    labels = ("primary", *sorted(REQUIRED_CAUSAL_CONTROLS))
    assert causal["title"] == "Registered causal intervention and controls"
    assert causal["ylabel"] == "Oriented answer-attempt effect"
    assert causal["tick_labels"] == labels
    assert causal["bar_heights"][: len(labels)] == pytest.approx(
        (
            result.metrics.high_to_low_effect,
            *(result.metrics.control_effects[name][0] for name in labels[1:]),
        )
    )
    assert causal["bar_heights"][len(labels) :] == pytest.approx(
        (
            result.metrics.low_to_high_effect,
            *(result.metrics.control_effects[name][1] for name in labels[1:]),
        )
    )
    assert captured["appendix_h6_nulls.png"]["title"].startswith("Appendix: H6")


def test_figure_4_explicitly_reports_f2b_not_run(monkeypatch, tmp_path):
    captured = _capture_figure_semantics(monkeypatch)

    build_registered_figures(_behavior(), _f2a(), tmp_path / "figures")

    causal = captured["figure_4_causal_intervention.png"]
    assert causal["title"] == "Registered causal intervention: not run"
    assert causal["bar_heights"] == ()
    assert any("F2B not run" in text for text in causal["texts"])


def test_figure_4_explicitly_reports_prerequisite_gate_failure(monkeypatch, tmp_path):
    captured = _capture_figure_semantics(monkeypatch)

    build_registered_figures(
        _behavior(interaction=0.01, interval=(-0.01, 0.03)),
        _f2a(),
        tmp_path / "figures",
        intervention=_f2b(),
    )

    causal = captured["figure_4_causal_intervention.png"]
    assert causal["title"] == "Registered causal intervention: not gated"
    assert causal["bar_heights"] == ()
    assert any("F1/H3/H4" in text for text in causal["texts"])


def test_figure_4_explicitly_reports_incomplete_f2b_evidence(monkeypatch, tmp_path):
    captured = _capture_figure_semantics(monkeypatch)

    build_registered_figures(
        _behavior(),
        _f2a(),
        tmp_path / "figures",
        intervention=_f2b(bootstrap_summary={}),
    )

    causal = captured["figure_4_causal_intervention.png"]
    assert causal["title"] == "Registered causal intervention: not evaluable"
    assert causal["bar_heights"] == ()
    assert any("bootstrap" in text for text in causal["texts"])


def test_registered_figures_reject_untyped_intervention(tmp_path):
    with pytest.raises(ValueError, match="typed InterventionTestResult"):
        build_registered_figures(
            _behavior(),
            _f2a(),
            tmp_path / "figures",
            intervention={"high_to_low_effect": 0.08},
        )


def test_red_empty_intervention_bootstrap_can_never_support_h7_or_h8():
    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=_f2b(bootstrap_summary={}),
        circuit=None,
    )

    assert claims["H7"].status in {"not_evaluable", "not_supported"}
    assert claims["H8"].status in {"not_evaluable", "not_supported"}


def test_red_f2a_rejects_stored_gates_not_recomputed_from_probe_results():
    familiarity = _probe("familiarity")
    answerability = _probe("answerability")
    unsupported = _probe("unsupported_answer")
    forged = F2AGates(
        familiarity.sha256,
        answerability.sha256,
        unsupported.sha256,
        {"H3": 0.01, "H4": 0.01},
        HypothesisGate("H3", (GateCriterion("forged", 1.0, 0.0, ">"),)),
        HypothesisGate("H4", (GateCriterion("forged", 1.0, 0.0, ">"),)),
        HypothesisGate("H5", (GateCriterion("forged", 1.0, 0.0, ">"),)),
        HypothesisGate("H6", (GateCriterion("forged", 1.0, 0.0, ">"),)),
    )

    with pytest.raises(ValueError, match="canonical F2A"):
        F2AEvidence(familiarity, answerability, unsupported, forged, {})


@pytest.mark.parametrize(
    ("task", "kind"),
    (
        ("familiarity", "label_permutation"),
        ("answerability", "label_permutation"),
        ("unsupported_answer", "layer_order"),
        ("unsupported_answer", "random_map"),
    ),
)
def test_f2a_claims_require_every_registered_null_seed(task, kind):
    evidence = _f2a()
    result = getattr(evidence, task)
    removed = False
    nulls = []
    for null in result.null_results:
        if not removed and null.kind == kind:
            removed = True
            continue
        nulls.append(null)
    mutated = _with_f2a_nulls(evidence, task, tuple(nulls))

    claims = recompute_claim_ladder(
        behavior=_behavior(), f2a=mutated, f2b=None, circuit=None
    )

    for hypothesis in ("H3", "H4", "H5", "H6"):
        assert claims[hypothesis].status == "not_evaluable"
        assert any("null" in reason for reason in claims[hypothesis].reasons)


@pytest.mark.parametrize("mutation", ("duplicate", "unregistered", "wrong_hash"))
def test_f2a_claims_reject_mutated_registered_null_provenance(mutation):
    evidence = _f2a()
    result = evidence.familiarity
    nulls = list(result.null_results)
    if mutation == "duplicate":
        nulls[-1] = nulls[0]
    elif mutation == "unregistered":
        nulls[0] = _scored_null("familiarity", "label_permutation", 17)
    else:
        nulls[0] = _scored_null(
            "familiarity",
            "label_permutation",
            DEFAULT_FULL_SELECTION_NULL_SEEDS[0],
            seed_list_sha256="9" * 64,
        )
    mutated = _with_f2a_nulls(evidence, "familiarity", tuple(nulls))

    claims = recompute_claim_ladder(
        behavior=_behavior(), f2a=mutated, f2b=None, circuit=None
    )

    assert claims["H3"].status == "not_evaluable"
    assert any("null" in reason for reason in claims["H3"].reasons)


def test_red_circuit_claim_requires_yield_failures_and_original_model_support():
    f2b = _f2b()
    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=_f2a(),
        f2b=f2b,
        circuit=_circuit(
            f2b, attempted=0, successful=0, original_model_supported=False
        ),
    )

    assert claims["F3"].status != "supported_prompt_local_hypothesis"


def test_f3_requires_complete_f2a_core_gate_including_h5():
    evidence = _f2a()
    unsupported = replace(
        evidence.unsupported_answer,
        relative_h5_log_loss_improvement=0.0,
    )
    gates = evaluate_f2a_gates(
        evidence.familiarity, evidence.answerability, unsupported
    )
    f2a = F2AEvidence(
        evidence.familiarity,
        evidence.answerability,
        unsupported,
        gates,
        evidence.sae_gates,
    )
    f2b = _f2b()

    claims = recompute_claim_ladder(
        behavior=_behavior(),
        f2a=f2a,
        f2b=f2b,
        circuit=_circuit(f2b),
    )

    assert claims["H3"].status == "supported"
    assert claims["H4"].status == "supported"
    assert claims["H5"].status == "not_supported"
    assert claims["F3"].status == "not_supported"
    assert any("H5" in reason for reason in claims["F3"].reasons)


def test_circuit_evidence_must_be_typed_and_bound_to_intervention_result():
    f2b = _f2b()
    with pytest.raises(ValueError, match="typed CircuitGateEvidence"):
        recompute_claim_ladder(
            behavior=_behavior(),
            f2a=_f2a(),
            f2b=f2b,
            circuit={"attempted": 8},
        )
    mismatched = CircuitGateEvidence(
        "1" * 64,
        "a" * 64,
        "2" * 64,
        "3" * 64,
        "9" * 64,
        0.81,
        0.82,
        0.61,
        0.76,
        0.40,
        1,
        1,
        (),
        True,
    )
    with pytest.raises(ValueError, match="not bound"):
        recompute_claim_ladder(
            behavior=_behavior(), f2a=_f2a(), f2b=f2b, circuit=mismatched
        )


def test_red_report_lists_every_negative_null_and_failure_class(tmp_path):
    f2b = _f2b()
    report = build_report(
        behavior=_behavior(),
        f2a=_f2a(include_sae=False),
        f2b=f2b,
        circuit=_circuit(
            f2b, attempted=1, successful=0, original_model_supported=False
        ),
        output=tmp_path / "report.md",
    )
    text = report.read_text(encoding="utf-8")

    assert "F2A null familiarity label_permutation seed=2026072201:" in text
    assert "F2A OOD entity (familiarity):" in text and "denominator=" in text
    assert "SAE analysis: skipped/not_run" in text
    assert "F2B control orthogonal:" in text
    assert "F2B bootstrap high_to_low:" in text
    assert "Circuit failure case-0: trace_failed" in text


def test_red_report_only_release_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    report = source / "report.md"
    report.write_text("unbound report\n", encoding="utf-8")

    with pytest.raises(ValueError, match="FAArtifactStore"):
        build_release_bundle(
            source_root=source,
            output=tmp_path / "release",
            allowlist={"report.md": hashlib.sha256(report.read_bytes()).hexdigest()},
            config_hash="a" * 64,
            preregistration_hash="b" * 64,
        )


def test_red_registered_figures_reject_free_unbound_mapping(tmp_path):
    free_mapping = {
        "behavior": {"conditions": ["a"], "attempt_rates": [0.5]},
        "layers": [0],
        "familiarity_auroc": [0.6],
        "answerability_auroc": [0.6],
        "baseline_names": ["surface"],
        "baseline_log_loss": [0.6],
        "observed_increment": 0.01,
        "null_increments": [0.0],
    }

    with pytest.raises(ValueError, match="canonical typed evidence"):
        build_registered_figures(free_mapping, tmp_path / "figures")
