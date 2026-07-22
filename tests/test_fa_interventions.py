from __future__ import annotations

import hashlib
import json
from contextlib import AbstractContextManager
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

import trajectory_extractor.fa_interventions as interventions_module
from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_config import (
    CONFIRMATORY_CHAT_TEMPLATE_SHA256,
    CONFIRMATORY_MODEL_ID,
    CONFIRMATORY_MODEL_REVISION,
    CONFIRMATORY_THRESHOLDS,
)
from trajectory_extractor.fa_interventions import (
    AppliedIntervention,
    ConfirmatoryInterventionExecutor,
    ConfirmatoryPinBundle,
    ExecutedPatchEvidence,
    ExecutedUnrelatedEvidence,
    GateArtifactEvidence,
    InterventionCandidate,
    InterventionMetrics,
    InterventionPair,
    InterventionPrompt,
    PatchAudit,
    PatchSpec,
    RawInterventionOutcome,
    ReadoutConstraints,
    REQUIRED_CAUSAL_CONTROLS,
    UnrelatedCapabilityPrompt,
    build_controls,
    evaluate_intervention_test_once,
    run_prefill_patch,
    select_intervention,
    verify_gate_result_artifact,
    verify_validation_control_artifact,
)
from trajectory_extractor.fa_probes import F2AGates, GateCriterion, HypothesisGate
from trajectory_extractor.fa_scoring import (
    BehavioralMetrics,
    BootstrapDistribution,
    PercentileInterval,
    behavioral_gate,
)


MODEL_REVISION = CONFIRMATORY_MODEL_REVISION
TOKENIZER_REVISION = CONFIRMATORY_MODEL_REVISION
CHAT_TEMPLATE_SHA256 = CONFIRMATORY_CHAT_TEMPLATE_SHA256
CONFIG_SHA256 = "76f557db589863ab217f963ce5020b4a57e88774582e1e3d3bd58600143103fa"
SOURCE_PINS_SHA256 = "af9f6a042168c4715958d7d376c1eaedc4e643c56eb6da34ff47574799bc33f9"
PREREGISTRATION_SHA256 = "d" * 64
PROBE_SELECTION_SHA256 = "9" * 64
REGISTERED_DOMAINS = ("person", "place", "organization", "creative_work")
READOUT_CONSTRAINTS = ReadoutConstraints(
    familiarity_min_effect=0.10,
    answerability_max_abs_change=0.05,
    entity_type_max_abs_change=0.05,
    generic_confidence_max_abs_change=0.02,
)
PINS = ConfirmatoryPinBundle(
    model_id=CONFIRMATORY_MODEL_ID,
    model_revision=MODEL_REVISION,
    tokenizer_revision=TOKENIZER_REVISION,
    chat_template_sha256=CHAT_TEMPLATE_SHA256,
    config_sha256=CONFIG_SHA256,
    source_pins_sha256=SOURCE_PINS_SHA256,
)


def _canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_hash(values):
    array = np.asarray(values, dtype=np.float64)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\n" + array.tobytes(order="C")).hexdigest()


class _PatchSession(AbstractContextManager):
    def __init__(self, runner, intervention):
        self.runner = runner
        self.intervention = intervention

    def __enter__(self):
        assert not self.runner.patch_active
        self.runner.patch_active = True
        return self

    def prefill(self, input_ids):
        assert self.runner.patch_active
        self.runner.last_input_ids = tuple(input_ids)
        intervention = self.intervention
        self.runner.changed.append(
            (intervention.layer, intervention.position, intervention.replacement.copy())
        )
        if self.runner.extra_site is not None:
            self.runner.changed.append((*self.runner.extra_site, intervention.replacement.copy()))
        return tuple(input_ids)

    def __exit__(self, exc_type, exc, tb):
        self.runner.patch_active = False
        return False


class _FakeRunner:
    def __init__(
        self,
        *,
        decode_patch=False,
        extra_site=None,
        prefix_token_ids=(20,),
        forged_vector=False,
    ):
        self.model_revision = MODEL_REVISION
        self.patch_active = False
        self.decode_hook_calls = 0
        self.decode_patch = decode_patch
        self.extra_site = extra_site
        self.prefix_token_ids = tuple(prefix_token_ids)
        self.forged_vector = forged_vector
        self.changed = []
        self.last_input_ids = ()
        self.intervention = None

    def prefill_patch(self, *, intervention):
        assert isinstance(intervention, AppliedIntervention)
        self.intervention = intervention
        return _PatchSession(self, intervention)

    def decode(self, prefill_state):
        if self.decode_patch:
            self.decode_hook_calls += 1
        return {"token_ids": tuple(prefill_state) + (99,), "nested": {"tokens": [99]}}

    def observed_patch_audit(self):
        intervention = self.intervention
        assert intervention is not None
        return {
            "modified_sites": tuple((layer, position) for layer, position, _ in self.changed),
            "decode_hook_calls": self.decode_hook_calls,
            "input_ids": self.last_input_ids,
            "assistant_prefix_token_ids": self.prefix_token_ids,
            "rendered_prefix_utf8_sha256": hashlib.sha256(
                _prompt_rendered_prefix("low_exposure")
            ).hexdigest(),
            "tokenizer_revision": TOKENIZER_REVISION,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "applied_intervention_sha256": intervention.sha256,
            "applied_vector_sha256": "0" * 64
            if self.forged_vector
            else intervention.vector_sha256,
            "replacement_sha256": intervention.replacement_sha256,
            "source_evidence_sha256": intervention.source_evidence_sha256,
            "source_activation_sha256": intervention.source_activation_sha256,
            "source_example_id": intervention.source_example_id,
            "destination_example_id": intervention.destination_example_id,
            "destination_entity_unit_id": intervention.destination_entity_unit_id,
            "destination_evidence_sha256": intervention.destination_evidence_sha256,
        }


def _shared_query_suffix():
    return (
        b" Task: Target: Atlas; relation: archive_code; "
        b"reply with exactly one registry code or UNKNOWN.</chat>"
    )


def _prompt_rendered_prefix(exposure="low_exposure"):
    context = (
        b"<chat>Atlas appears in four neutral facts."
        if exposure == "high_exposure"
        else b"<chat>Orion appears in four neutral facts."
    )
    return context + _shared_query_suffix()


def _prompt(
    exposure: str,
    *,
    split: str = "locked_validation",
    entity: str = "entity-a",
    domain: str = "person",
    answerability: str = "code_absent",
    template_family: str = "family-a",
    activation_manifest_sha256: str = "b" * 64,
    activation_scale: float = 1.0,
):
    activation = (
        np.array([3.0, 4.0, 5.0])
        if exposure == "high_exposure"
        else np.array([1.0, 2.0, 3.0])
    ) * activation_scale
    return InterventionPrompt(
        example_id=f"{split}-{entity}-{answerability}-{exposure}",
        entity_unit_id=entity,
        split=split,
        exposure=exposure,
        answerability=answerability,
        template_family=template_family,
        domain=domain,
        target_string="Atlas",
        relation_id="archive_code",
        entity_type=domain,
        output_instruction="reply with exactly one registry code or UNKNOWN",
        registry_code=f"CODE_{entity.upper()}",
        target_familiarity="screened_real",
        distractor_familiarity="matched_synthetic",
        model_revision=MODEL_REVISION,
        input_ids=(
            (10 if exposure == "high_exposure" else 12),
            (11 if exposure == "high_exposure" else 13),
            30,
            31,
            20,
        ),
        assistant_prefix_token_ids=(20,),
        rendered_prefix_utf8=_prompt_rendered_prefix(exposure),
        shared_query_suffix_token_ids=(30, 31, 20),
        shared_query_suffix_utf8=_shared_query_suffix(),
        shared_anchor_offsets={"target_intro_end": 1, "user_prompt_end": 2},
        tokenizer_revision=TOKENIZER_REVISION,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
        anchor_positions={"target_intro_end": 3, "user_prompt_end": 4},
        activation_layer=12,
        activation_anchor="target_intro_end",
        activation=activation,
        activation_sha256=None,
        activation_manifest_sha256=activation_manifest_sha256,
    )


def _pair(
    *,
    split: str = "locked_validation",
    entity: str = "entity-a",
    domain: str = "person",
    answerability: str = "code_absent",
    template_family: str = "family-a",
    activation_manifest_sha256: str = "b" * 64,
    activation_scale: float = 1.0,
):
    common = {
        "split": split,
        "entity": entity,
        "domain": domain,
        "answerability": answerability,
        "template_family": template_family,
        "activation_manifest_sha256": activation_manifest_sha256,
        "activation_scale": activation_scale,
    }
    return InterventionPair(
        high=_prompt("high_exposure", **common),
        low=_prompt("low_exposure", **common),
    )


def _spec(direction: str = "high_to_low"):
    return PatchSpec(
        layer=12,
        anchor="target_intro_end",
        direction=direction,
        mode="full_replacement",
        alpha=1.0,
        model_revision=MODEL_REVISION,
        activation_manifest_sha256="b" * 64,
    )


def _bootstrap_summary():
    return {
        "method": "crossed_entity_unit_template_family_bootstrap",
        "seed": 20260722,
        "replicates": 10000,
        "requested_draws": 10000,
        "valid_draws": 10000,
        "discarded_draws": 0,
        "resampling_unit": ["entity_unit_id", "template_family"],
        "alpha": 0.05,
        "directions": {
            "high_to_low": {
                "point_estimate": 0.08,
                "raw_interval": [0.02, 0.14],
                "raw_p": 0.005,
                "entities": ["entity-a"],
                "template_families": ["family-a"],
                "holm_interval": [0.02, 0.14],
                "holm_adjusted_p": 0.01,
            },
            "low_to_high": {
                "point_estimate": 0.08,
                "raw_interval": [0.01, 0.15],
                "raw_p": 0.01,
                "entities": ["entity-a"],
                "template_families": ["family-a"],
                "holm_interval": [0.01, 0.15],
                "holm_adjusted_p": 0.02,
            },
        },
    }


def _metrics(
    effect: float = 0.08,
    *,
    interval=None,
    controls=None,
    bootstrap=True,
    observed_domains=REGISTERED_DOMAINS,
    passing_domains=REGISTERED_DOMAINS,
    generic_confidence_change=0.01,
    target_bound_accuracy_change=-0.02,
    target_bound_accuracy_by_direction=None,
    unrelated_refusal_change=0.01,
    unrelated_invalid_format_change=0.0,
    unrelated_refusal_by_direction=None,
    unrelated_invalid_by_direction=None,
):
    evidence = {name: (0.0, 0.0) for name in REQUIRED_CAUSAL_CONTROLS}
    if controls is not None:
        evidence = controls
    return InterventionMetrics(
        high_to_low_effect=effect,
        high_to_low_interval=interval or (0.02, 0.14),
        low_to_high_effect=effect,
        low_to_high_interval=(0.01, 0.15),
        control_effects=evidence,
        target_bound_accuracy_change=target_bound_accuracy_change,
        unrelated_refusal_change=unrelated_refusal_change,
        unrelated_invalid_format_change=unrelated_invalid_format_change,
        familiarity_readout_effect=0.30,
        answerability_max_abs_change=0.0,
        entity_type_max_abs_change=0.0,
        generic_confidence_max_abs_change=generic_confidence_change,
        readout_constraints=READOUT_CONSTRAINTS,
        observed_domains=tuple(observed_domains),
        passing_domains=tuple(passing_domains),
        completed_fraction=1.0,
        bootstrap_summary=_bootstrap_summary() if bootstrap else {},
        unrelated_refusal_change_by_direction=(
            unrelated_refusal_by_direction
            or {"high_to_low": 0.01, "low_to_high": 0.01}
        ),
        unrelated_invalid_format_change_by_direction=(
            unrelated_invalid_by_direction
            or {"high_to_low": 0.0, "low_to_high": 0.0}
        ),
        target_bound_accuracy_change_by_direction=(
            target_bound_accuracy_by_direction
            or {"high_to_low": -0.02, "low_to_high": -0.02}
        ),
    )


def _candidate(layer: int = 12, *, metrics=None, source_split="locked_validation"):
    return InterventionCandidate(
        layer=layer,
        anchor="target_intro_end",
        method="full_replacement",
        alpha=1.0,
        source_split=source_split,
        metrics=metrics or _metrics(),
        direction_sha256="c" * 64,
    )


def _f1_metrics_row(endpoint_input_sha256, *, supported=True):
    cell = ("screened_real", "matched_synthetic", "code_absent")
    interaction = 0.08 if supported else 0.01
    metrics = BehavioralMetrics(
        status="evaluable",
        reasons=(),
        cell_rates={cell: 0.5},
        completion_by_cell={cell: 1.0},
        format_validity_by_cell={cell: 1.0},
        denominators={cell: 16},
        invalid_format_counts={cell: 0},
        interaction=interaction,
        h2_accuracy_difference=0.0,
        h2b_interaction=interaction,
        sensitivities={},
    )
    bootstrap = BootstrapDistribution(
        interaction_samples=(interaction,) * 50,
        h2_accuracy_difference_samples=(0.0,) * 50,
        h2b_interaction_samples=(interaction,) * 50,
        interaction_interval=PercentileInterval(
            interaction, 0.02 if supported else -0.02, 0.14
        ),
        h2_accuracy_difference_interval=PercentileInterval(0.0, -0.01, 0.01),
        h2b_interaction_interval=PercentileInterval(
            interaction, 0.01 if supported else -0.02, 0.15
        ),
        weighted_denominators=(16,),
        seed=20260722,
    )
    gate = behavioral_gate(
        metrics,
        bootstrap,
        thresholds=CONFIRMATORY_THRESHOLDS,
        same_string_sealed=True,
        config_hash=CONFIG_SHA256,
        manifest_hash=endpoint_input_sha256,
    )
    evidence = {
        "metrics": metrics.to_record(),
        "bootstrap": bootstrap.to_record(),
        "gate": gate.to_record(),
        "scored_rows": [],
    }
    return {
        "kind": "metrics",
        "phase": "F1",
        **evidence,
        "evidence_sha256": _canonical_hash(evidence),
    }


def _hypothesis_gate(hypothesis, *, supported=True):
    return HypothesisGate(
        hypothesis,
        (GateCriterion(f"{hypothesis.lower()}_criterion", 1.0 if supported else 0.0, 0.5, ">="),),
    )


def _probe_result_record(task, hypothesis, endpoint_input_sha256, *, supported=True):
    gate = _hypothesis_gate(hypothesis, supported=supported)
    record = {
        "schema_version": 3,
        "task": task,
        "selection_hash": _canonical_hash({"task": task}),
        "authorization_sha256": "a" * 64,
        "endpoint_input_sha256": endpoint_input_sha256,
        "endpoint_input_identities_sha256": "b" * 64,
        "endpoint_source_identities_sha256": "b" * 64,
        "test_ids": [f"{task}-example"],
        "test_row_sha256s": [_canonical_hash({"row": task})],
        "selected_feature_family": "static",
        "selected_model_scope": {
            "feature_family": "static",
            "anchor": "user_prompt_end",
            "layer": 12,
            "claim_scope": "pre_output",
            "selected_model_sha256": _canonical_hash({"model": task}),
        },
        "metrics": {"status": "evaluable"},
        "model_metrics": {},
        "per_condition": {},
        "worst_condition": None,
        "ood_transfer": {},
        "worst_ood_transfer": {},
        "cross_condition_transfer": None,
        "relative_h5_log_loss_improvement": 0.10,
        "relative_h6_log_loss_improvement": 0.02,
        "crossed_auroc_95": None,
        "h5_absolute_log_loss_difference_95": None,
        "h6_absolute_log_loss_difference_95": None,
        "primary_gate": gate.to_record(),
        "null_results": [],
        "refit_performed": False,
    }
    return record


def _f2a_metrics_row(endpoint_input_sha256, *, supported=True):
    results = {
        "familiarity": _probe_result_record(
            "familiarity", "H3", endpoint_input_sha256, supported=supported
        ),
        "answerability": _probe_result_record(
            "answerability", "H4", endpoint_input_sha256, supported=supported
        ),
        "unsupported_answer": _probe_result_record(
            "unsupported_answer", "H5", endpoint_input_sha256, supported=supported
        ),
    }
    gates = F2AGates(
        familiarity_result_sha256=_canonical_hash(results["familiarity"]),
        answerability_result_sha256=_canonical_hash(results["answerability"]),
        unsupported_result_sha256=_canonical_hash(results["unsupported_answer"]),
        holm_adjusted_p={"H3": 0.01, "H4": 0.02},
        h3=_hypothesis_gate("H3", supported=supported),
        h4=_hypothesis_gate("H4", supported=supported),
        h5=_hypothesis_gate("H5", supported=supported),
        h6=_hypothesis_gate("H6", supported=True),
    )
    result = {
        "schema_version": 1,
        "selection_bundle_hash": PROBE_SELECTION_SHA256,
        "authorization_sha256": "c" * 64,
        "endpoint_input_sha256": endpoint_input_sha256,
        "endpoint_input_identities_sha256": "b" * 64,
        "endpoint_source_identities_sha256": "b" * 64,
        "results": results,
        "gates": gates.to_record(),
        "refit_performed": False,
    }
    return {"kind": "metrics", "metric_type": "f2a_bundle", "result": result}


def _gate_evidence(tmp_path, phase, *, status="supported"):
    supported = status == "supported"
    root = tmp_path / f"gate-evidence-{status}"
    store = FAArtifactStore(root)
    endpoint = "behavior_test" if phase == "F1" else "probe_test"
    input_manifest = (
        root
        / "runs"
        / "familiarity_answerability"
        / "gate-run"
        / "shards"
        / endpoint
        / "inputs.jsonl.manifest.json"
    )
    if not input_manifest.exists():
        endpoint_input = store.write_completed_shard(
            "gate-run",
            endpoint,
            "inputs",
            [{"kind": "endpoint_input", "endpoint": endpoint}],
            {"config_sha256": CONFIG_SHA256},
        )
        selection_hash = (
            "8" * 64 if phase == "F1" else PROBE_SELECTION_SHA256
        )
        store.seal_endpoint(
            endpoint,
            [endpoint_input],
            {
                "preregistration": PREREGISTRATION_SHA256,
                "selection_manifest": selection_hash,
            },
        )
        receipt = store.unlock_endpoint(
            endpoint, PREREGISTRATION_SHA256, selection_hash
        )
        row = (
            _f1_metrics_row(endpoint_input.sha256, supported=supported)
            if phase == "F1"
            else _f2a_metrics_row(endpoint_input.sha256, supported=supported)
        )
        lineage = (
            {
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "selection_sha256": selection_hash,
                "prompt_manifest_sha256": endpoint_input.sha256,
                "evidence_sha256": row["evidence_sha256"],
                "config_sha256": CONFIG_SHA256,
            }
            if phase == "F1"
            else {
                "selection_manifest": selection_hash,
                "authorization": row["result"]["authorization_sha256"],
                "endpoint_input_sha256": endpoint_input.sha256,
                "endpoint_source_identities_sha256": row["result"][
                    "endpoint_input_identities_sha256"
                ],
            }
        )
        metrics = store.write_completed_shard(
            "gate-run",
            endpoint,
            "metrics",
            [row],
            lineage,
            record_kind="metrics",
        )
        store.mark_evaluated(receipt, metrics.data_path)
        store.close_endpoint(endpoint)
    return verify_gate_result_artifact(store, input_manifest, phase=phase)


CONTROL_FIT_METHOD = "difference_in_means_with_median_norm_v1"
CONTROL_PERMUTATION_SEED = 20260722
CONTROL_DIRECTION_DERIVATION = {
    "shuffled_direction": "mean_activation_by_fixed_permuted_familiarity_label",
    "answerability_direction": "mean_activation_answerable_minus_unanswerable",
    "activation_norm_direction": "mean_activation_above_median_norm_minus_at_or_below",
}


def _difference_in_means(matrix, labels):
    labels = np.asarray(labels, dtype=np.int64)
    return matrix[labels == 1].mean(axis=0) - matrix[labels == 0].mean(axis=0)


def _write_control_source(tmp_path, *, vector_override=None, shard_id="f2b-controls"):
    store = FAArtifactStore(tmp_path / "control-source")
    raw = (
        ("control-0", 0, 0, [0.2, 1.0, 0.1]),
        ("control-1", 0, 1, [1.4, 0.3, 0.2]),
        ("control-2", 1, 0, [0.4, 1.8, 1.1]),
        ("control-3", 1, 1, [1.8, 0.8, 1.3]),
        ("control-4", 0, 0, [0.1, 0.4, 1.7]),
        ("control-5", 0, 1, [1.2, 0.2, 1.9]),
        ("control-6", 1, 0, [0.8, 1.5, 2.1]),
        ("control-7", 1, 1, [2.2, 1.1, 2.4]),
    )
    input_rows = []
    activation_rows = []
    for example_id, familiarity, answerability, activation in raw:
        input_payload = {
            "example_id": example_id,
            "familiarity_label": familiarity,
            "answerability_label": answerability,
            "registered_input": f"sealed validation input for {example_id}",
        }
        input_row = {
            "kind": "f2b_control_input",
            "row_id": f"input-{example_id}",
            **input_payload,
            "input_sha256": _canonical_hash(input_payload),
        }
        input_rows.append(input_row)
        activation_rows.append(
            {
                "kind": "f2b_control_activation",
                "row_id": f"activation-{example_id}",
                "example_id": example_id,
                "input_row_sha256": _canonical_hash(input_row),
                "activation": activation,
                "activation_sha256": _array_hash(activation),
            }
        )
    source_lineage = {
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "probe_selection_sha256": PROBE_SELECTION_SHA256,
    }
    try:
        inputs = store.write_completed_shard(
            "control-source",
            "locked_validation",
            "f2b-control-inputs",
            input_rows,
            source_lineage,
            record_kind="f2b_control_input",
        )
    except FileExistsError:
        inputs = store.verify_shard(
            store.root
            / "runs/familiarity_answerability/control-source/shards/locked_validation/"
            "f2b-control-inputs.jsonl.manifest.json"
        )
    activation_lineage = {
        **source_lineage,
        "input_artifact_sha256": inputs.sha256,
        "input_manifest_sha256": hashlib.sha256(
            inputs.manifest_path.read_bytes()
        ).hexdigest(),
    }
    try:
        activations = store.write_completed_shard(
            "control-source",
            "locked_validation",
            "f2b-control-activations",
            activation_rows,
            activation_lineage,
            record_kind="f2b_control_activation",
        )
    except FileExistsError:
        activations = store.verify_shard(
            store.root
            / "runs/familiarity_answerability/control-source/shards/locked_validation/"
            "f2b-control-activations.jsonl.manifest.json"
        )

    ordered_inputs = sorted(input_rows, key=lambda row: row["example_id"])
    ordered_activations = sorted(activation_rows, key=lambda row: row["example_id"])
    matrix = np.asarray(
        [row["activation"] for row in ordered_activations], dtype=np.float64
    )
    familiarity = np.asarray(
        [row["familiarity_label"] for row in ordered_inputs], dtype=np.int64
    )
    answerability = np.asarray(
        [row["answerability_label"] for row in ordered_inputs], dtype=np.int64
    )
    permutation = np.random.default_rng(CONTROL_PERMUTATION_SEED).permutation(
        len(ordered_inputs)
    )
    norm_labels = (np.linalg.norm(matrix, axis=1) > np.median(np.linalg.norm(matrix, axis=1))).astype(
        np.int64
    )
    vectors = {
        "shuffled_direction": _difference_in_means(
            matrix, familiarity[permutation]
        ).tolist(),
        "answerability_direction": _difference_in_means(
            matrix, answerability
        ).tolist(),
        "activation_norm_direction": _difference_in_means(
            matrix, norm_labels
        ).tolist(),
    }
    if vector_override is not None:
        vectors.update(vector_override)
    input_row_sha256s = [_canonical_hash(row) for row in ordered_inputs]
    activation_row_sha256s = [_canonical_hash(row) for row in ordered_activations]
    input_row_ids = [row["row_id"] for row in ordered_inputs]
    activation_row_ids = [row["row_id"] for row in ordered_activations]
    labels = [
        {
            "example_id": row["example_id"],
            "familiarity_label": row["familiarity_label"],
            "answerability_label": row["answerability_label"],
        }
        for row in ordered_inputs
    ]
    row = {
        "kind": "f2b_control_source",
        "source_split": "locked_validation",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "probe_selection_sha256": PROBE_SELECTION_SHA256,
        "input_manifest_path": str(inputs.manifest_path.relative_to(store.root)),
        "input_artifact_sha256": inputs.sha256,
        "input_manifest_sha256": hashlib.sha256(inputs.manifest_path.read_bytes()).hexdigest(),
        "activation_manifest_path": str(
            activations.manifest_path.relative_to(store.root)
        ),
        "activation_artifact_sha256": activations.sha256,
        "activation_manifest_sha256": hashlib.sha256(
            activations.manifest_path.read_bytes()
        ).hexdigest(),
        "example_ids": [row["example_id"] for row in ordered_inputs],
        "input_row_ids": input_row_ids,
        "activation_row_ids": activation_row_ids,
        "input_row_ids_sha256": _canonical_hash(input_row_ids),
        "activation_row_ids_sha256": _canonical_hash(activation_row_ids),
        "input_row_sha256s": input_row_sha256s,
        "activation_row_sha256s": activation_row_sha256s,
        "input_rows_sha256": _canonical_hash(input_row_sha256s),
        "activation_rows_sha256": _canonical_hash(activation_row_sha256s),
        "label_sha256": _canonical_hash(labels),
        "fit_method": CONTROL_FIT_METHOD,
        "permutation_seed": CONTROL_PERMUTATION_SEED,
        "permutation": permutation.tolist(),
        "permutation_sha256": _canonical_hash(permutation.tolist()),
        "direction_derivation": CONTROL_DIRECTION_DERIVATION,
        "direction_derivation_sha256": _canonical_hash(
            CONTROL_DIRECTION_DERIVATION
        ),
        **vectors,
        **{f"{name}_sha256": _array_hash(value) for name, value in vectors.items()},
    }
    lineage_keys = (
        "preregistration_sha256",
        "probe_selection_sha256",
        "input_artifact_sha256",
        "input_manifest_sha256",
        "activation_artifact_sha256",
        "activation_manifest_sha256",
        "input_row_ids_sha256",
        "activation_row_ids_sha256",
        "input_rows_sha256",
        "activation_rows_sha256",
        "label_sha256",
        "fit_method",
        "permutation_seed",
        "permutation_sha256",
        "direction_derivation_sha256",
        "shuffled_direction_sha256",
        "answerability_direction_sha256",
        "activation_norm_direction_sha256",
    )
    control_lineage = {name: row[name] for name in lineage_keys}
    shard = store.write_completed_shard(
        "control-source",
        "locked_validation",
        shard_id,
        [row],
        control_lineage,
        record_kind="f2b_control_source",
    )
    return store, shard


def _control_source(tmp_path):
    try:
        store, shard = _write_control_source(tmp_path)
        manifest_path = shard.manifest_path
    except FileExistsError:
        store = FAArtifactStore(tmp_path / "control-source")
        manifest_path = (
            store.root
            / "runs/familiarity_answerability/control-source/shards/locked_validation/"
            "f2b-controls.jsonl.manifest.json"
        )
    return verify_validation_control_artifact(
        store,
        manifest_path,
        preregistration_sha256=PREREGISTRATION_SHA256,
        probe_selection_sha256=PROBE_SELECTION_SHA256,
    )


def _selection(tmp_path):
    return select_intervention(
        (_candidate(),),
        preregistration_sha256=PREREGISTRATION_SHA256,
        probe_selection_sha256=PROBE_SELECTION_SHA256,
        confirmatory_pins=PINS,
        f1_evidence=_gate_evidence(tmp_path, "F1"),
        f2a_evidence=_gate_evidence(tmp_path, "F2A"),
        control_source=_control_source(tmp_path),
    )


def _unrelated_prompt(prompt_id, task, token):
    rendered = f"<chat>{prompt_id}</chat>".encode("utf-8")
    return UnrelatedCapabilityPrompt(
        prompt_id=prompt_id,
        task=task,
        input_ids=(token, token + 1, token + 2),
        assistant_prefix_token_ids=(token + 2,),
        rendered_prefix_utf8=rendered,
        anchor_positions={"target_intro_end": 1, "user_prompt_end": 2},
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
    )


def _sealed_intervention_endpoint(tmp_path, selection, *, selection_hash=None):
    store = FAArtifactStore(tmp_path / "endpoint")
    provisional = tuple(
        _pair(
            split="intervention_test",
            entity=f"entity-{index}",
            domain=domain,
            template_family=f"family-{index}",
            activation_scale=float(index),
        )
        for index, domain in enumerate(REGISTERED_DOMAINS, start=1)
    ) + (
        _pair(
            split="intervention_test",
            entity="entity-target",
            domain="person",
            answerability="target_bound",
            template_family="family-target",
            activation_scale=5.0,
        ),
    )
    activation_rows = [
        {
            "example_id": prompt.example_id,
            "activation_layer": prompt.activation_layer,
            "activation_anchor": prompt.activation_anchor,
            "activation_sha256": prompt.activation_sha256,
            "model_revision": prompt.model_revision,
            "tokenizer_revision": prompt.tokenizer_revision,
            "chat_template_sha256": prompt.chat_template_sha256,
        }
        for pair in provisional
        for prompt in (pair.high, pair.low)
    ]
    lineage = {
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "confirmatory_pins": PINS.canonical_payload(),
        "confirmatory_pins_sha256": PINS.sha256,
    }
    activations = store.write_completed_shard(
        "intervention-run",
        "intervention_test",
        "activation-manifest",
        activation_rows,
        lineage,
    )
    pairs = tuple(
        replace(
            pair,
            high=replace(pair.high, activation_manifest_sha256=activations.sha256),
            low=replace(pair.low, activation_manifest_sha256=activations.sha256),
        )
        for pair in provisional
    )
    prompts = store.write_completed_shard(
        "intervention-run",
        "intervention_test",
        "prompt-capability",
        [
            {"kind": "intervention_prompt", "prompt": prompt.canonical_payload()}
            for pair in pairs
            for prompt in (pair.high, pair.low)
        ],
        lineage,
    )
    unrelated_prompts = (
        _unrelated_prompt("unrelated-factual-1", "unrelated_factual", 40),
        _unrelated_prompt(
            "unrelated-instruction-1", "unrelated_instruction_following", 50
        ),
    )
    unrelated = store.write_completed_shard(
        "intervention-run",
        "intervention_test",
        "unrelated-capability",
        [
            {"kind": "unrelated_capability_prompt", "prompt": prompt.canonical_payload()}
            for prompt in unrelated_prompts
        ],
        lineage,
    )
    store.seal_endpoint(
        "intervention_test",
        [prompts, activations, unrelated],
        {
            "preregistration": PREREGISTRATION_SHA256,
            "selection_manifest": selection_hash or selection.sha256,
        },
    )
    return store, prompts, activations, unrelated, pairs, unrelated_prompts


class _Executor:
    def __init__(
        self,
        *,
        missing_control=None,
        decode_patch=False,
        extra_site=False,
        forged_provenance=False,
        prefix_mismatch=False,
        forged_vector=False,
        forged_intervention=False,
        strong_control=None,
        injected_labels=False,
        generic_confidence_change=0.01,
        unrelated_directional_harm=False,
        target_bound_directional_harm=False,
    ):
        self.missing_control = missing_control
        self.decode_patch = decode_patch
        self.extra_site = extra_site
        self.forged_provenance = forged_provenance
        self.prefix_mismatch = prefix_mismatch
        self.forged_vector = forged_vector
        self.forged_intervention = forged_intervention
        self.strong_control = strong_control
        self.injected_labels = injected_labels
        self.generic_confidence_change = generic_confidence_change
        self.unrelated_directional_harm = unrelated_directional_harm
        self.target_bound_directional_harm = target_bound_directional_harm
        self.calls = []
        self.unrelated_calls = []
        self.audits = []

    def _provenance(self, intervention, confirmatory_pins):
        return {
            "activation_manifest_sha256": "0" * 64
            if self.forged_provenance
            else intervention.activation_manifest_sha256,
            "model_id": confirmatory_pins.model_id,
            "model_revision": confirmatory_pins.model_revision,
            "tokenizer_revision": confirmatory_pins.tokenizer_revision,
            "chat_template_sha256": confirmatory_pins.chat_template_sha256,
            "config_sha256": confirmatory_pins.config_sha256,
            "source_pins_sha256": confirmatory_pins.source_pins_sha256,
            "confirmatory_pins_sha256": confirmatory_pins.sha256,
            "applied_intervention_sha256": intervention.sha256,
        }

    def _audit(self, prompt, intervention):
        sites = (
            ((intervention.layer, intervention.position),)
            if intervention.apply_patch
            else ()
        )
        if self.extra_site:
            sites = (*sites, (intervention.layer + 2, intervention.position))
        audit = PatchAudit(
            modified_sites=sites,
            decode_hook_calls=1 if self.decode_patch else 0,
            input_ids=prompt.input_ids,
            assistant_prefix_token_ids=(999,)
            if self.prefix_mismatch
            else prompt.assistant_prefix_token_ids,
            rendered_prefix_utf8_sha256=hashlib.sha256(
                prompt.rendered_prefix_utf8
            ).hexdigest(),
            tokenizer_revision=prompt.tokenizer_revision,
            chat_template_sha256=prompt.chat_template_sha256,
            applied_intervention_sha256=intervention.sha256,
            applied_vector_sha256="0" * 64
            if self.forged_vector
            else intervention.vector_sha256,
            replacement_sha256=intervention.replacement_sha256,
            source_evidence_sha256=intervention.source_evidence_sha256,
            source_activation_sha256=intervention.source_activation_sha256,
            source_example_id=intervention.source_example_id,
            destination_example_id=intervention.destination_example_id,
            destination_entity_unit_id=intervention.destination_entity_unit_id,
            destination_evidence_sha256=intervention.destination_evidence_sha256,
        )
        self.audits.append(audit)
        return audit

    def _decoded(self, pair, intervention):
        destination = pair.low if intervention.direction == "high_to_low" else pair.high
        if destination.answerability == "target_bound":
            baseline_text = destination.registry_code
            patched_text = destination.registry_code
            if self.target_bound_directional_harm:
                if intervention.direction == "high_to_low":
                    patched_text = "UNKNOWN"
                else:
                    baseline_text = "UNKNOWN"
        else:
            baseline_text = (
                "UNKNOWN" if intervention.direction == "high_to_low" else destination.registry_code
            )
            active = intervention.control_name == "primary" or (
                intervention.control_name == self.strong_control
            )
            if active:
                patched_text = (
                    destination.registry_code
                    if intervention.direction == "high_to_low"
                    else "UNKNOWN"
                )
            else:
                patched_text = baseline_text
        baseline_familiarity = 0.2 if destination.exposure == "low_exposure" else 0.8
        familiarity_change = 0.0
        if intervention.control_name == "primary" and destination.answerability == "code_absent":
            familiarity_change = 0.30 if intervention.direction == "high_to_low" else -0.30
        baseline_readouts = {
            "familiarity": baseline_familiarity,
            "answerability": 0.25,
            "entity_type": 0.60,
            "generic_confidence": 0.50,
        }
        patched_readouts = {
            **baseline_readouts,
            "familiarity": baseline_familiarity + familiarity_change,
            "generic_confidence": 0.50
            + (
                self.generic_confidence_change
                if intervention.control_name == "primary"
                else 0.0
            ),
        }
        decoded = {
            "baseline_text": baseline_text,
            "patched_text": patched_text,
            "baseline_truncated": False,
            "patched_truncated": False,
            "baseline_infrastructure_marked": False,
            "patched_infrastructure_marked": False,
            "baseline_readouts": baseline_readouts,
            "patched_readouts": patched_readouts,
        }
        if self.injected_labels:
            decoded["patched_answer_attempt"] = False
        return decoded

    def execute(
        self,
        *,
        pair,
        intervention,
        activation_manifest_sha256,
        confirmatory_pins,
    ):
        assert intervention.activation_manifest_sha256 == activation_manifest_sha256
        self.calls.append((pair.high.entity_unit_id, intervention))
        if intervention.control_name == self.missing_control:
            return None
        destination = pair.low if intervention.direction == "high_to_low" else pair.high
        return ExecutedPatchEvidence(
            applied_intervention_sha256="0" * 64
            if self.forged_intervention
            else intervention.sha256,
            audit=self._audit(destination, intervention),
            provenance=self._provenance(intervention, confirmatory_pins),
            decoded=self._decoded(pair, intervention),
        )

    def execute_unrelated(
        self,
        *,
        prompt,
        intervention,
        activation_manifest_sha256,
        confirmatory_pins,
    ):
        assert intervention.activation_manifest_sha256 == activation_manifest_sha256
        self.unrelated_calls.append((prompt.prompt_id, intervention))
        readouts = {
            "familiarity": 0.4,
            "answerability": 0.7,
            "entity_type": 0.5,
            "generic_confidence": 0.5,
        }
        baseline_text = "OK"
        patched_text = "OK"
        if self.unrelated_directional_harm:
            if intervention.direction == "high_to_low":
                patched_text = "I cannot answer that request."
            else:
                baseline_text = "I cannot answer that request."
        return ExecutedUnrelatedEvidence(
            applied_intervention_sha256=intervention.sha256,
            audit=self._audit(prompt, intervention),
            provenance=self._provenance(intervention, confirmatory_pins),
            decoded={
                "baseline_text": baseline_text,
                "patched_text": patched_text,
                "baseline_truncated": False,
                "patched_truncated": False,
                "baseline_infrastructure_marked": False,
                "patched_infrastructure_marked": False,
                "baseline_readouts": readouts,
                "patched_readouts": readouts,
            },
        )


def _evaluate(tmp_path, *, executor=None, pairs_transform=None):
    selection = _selection(tmp_path)
    store, prompts, activations, unrelated, pairs, unrelated_prompts = (
        _sealed_intervention_endpoint(tmp_path, selection)
    )
    if pairs_transform is not None:
        pairs = pairs_transform(pairs)
    adapter = executor or _Executor()
    result = evaluate_intervention_test_once(
        selection,
        store,
        endpoint_manifest_path=prompts.manifest_path,
        activation_manifest_path=activations.manifest_path,
        unrelated_manifest_path=unrelated.manifest_path,
        test_pairs=pairs,
        unrelated_prompts=unrelated_prompts,
        executor=adapter,
        confirmatory_pins=PINS,
    )
    return result, store, prompts, activations, unrelated, pairs, unrelated_prompts, adapter


def test_patch_changes_only_adapter_observed_prefill_position_and_closes_before_decode():
    runner = _FakeRunner()
    outcome = run_prefill_patch(
        runner,
        (_pair(),),
        _spec(),
        verified_activation_manifest_sha256="b" * 64,
    )
    assert outcome.changed_positions == {(12, 3)}
    assert outcome.decode_hook_calls == 0
    assert runner.patch_active is False
    assert len(outcome.rows) == 1


@pytest.mark.parametrize(
    ("runner", "message"),
    (
        (_FakeRunner(decode_patch=True), "decode-time"),
        (_FakeRunner(extra_site=(14, 3)), "modified site"),
        (_FakeRunner(prefix_token_ids=(999,)), "prefix"),
        (_FakeRunner(forged_vector=True), "bind the applied intervention"),
    ),
)
def test_patch_fails_closed_on_adversarial_prefill_audit(runner, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        run_prefill_patch(
            runner,
            (_pair(),),
            _spec(),
            verified_activation_manifest_sha256="b" * 64,
        )


def test_patch_requires_same_construct_and_verified_activation_provenance():
    pair = _pair()
    with pytest.raises(ValueError, match="answerability"):
        run_prefill_patch(
            _FakeRunner(),
            (replace(pair, low=replace(pair.low, answerability="target_bound")),),
            _spec(),
            verified_activation_manifest_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="verified activation manifest"):
        run_prefill_patch(
            _FakeRunner(),
            (pair,),
            _spec(),
            verified_activation_manifest_sha256="0" * 64,
        )


def test_build_controls_is_deterministic_norm_matched_and_used_by_execution(tmp_path):
    direction = np.array([1.0, -2.0, 3.0, -4.0])
    controls = build_controls(
        direction,
        shuffled_direction=np.array([4.0, 3.0, 2.0, 1.0]),
        answerability_direction=np.array([1.0, 0.0, 0.0, 0.0]),
        activation_norm_direction=np.array([0.0, 1.0, 0.0, 0.0]),
        cross_entity_direction=np.array([2.0, 2.0, -1.0, 0.5]),
        seed=20260722,
    )
    assert {
        "orthogonal",
        "shuffled",
        "norm_matched_random",
        "sign_reversed",
        "cross_entity",
    }.issubset(controls)
    assert all(
        np.isclose(np.linalg.norm(value), np.linalg.norm(direction))
        for value in controls.values()
    )

    _, _, _, _, _, _, _, executor = _evaluate(tmp_path)
    cross_calls = [
        (destination_entity, intervention)
        for destination_entity, intervention in executor.calls
        if intervention.control_name == "cross_entity"
    ]
    assert cross_calls
    assert all(
        intervention.source_entity_unit_id != destination_entity
        and intervention.source_activation_sha256 is not None
        and intervention.source_kind == "cross_entity_norm_matched_direction"
        and intervention.replacement_sha256 != intervention.source_activation_sha256
        for destination_entity, intervention in cross_calls
    )


def test_selection_uses_verified_gate_artifacts_and_persists_hash_lineage(tmp_path):
    f1 = _gate_evidence(tmp_path, "F1")
    f2a = _gate_evidence(tmp_path, "F2A")
    selection = select_intervention(
        (_candidate(13), _candidate(12)),
        preregistration_sha256=PREREGISTRATION_SHA256,
        probe_selection_sha256=PROBE_SELECTION_SHA256,
        confirmatory_pins=PINS,
        f1_evidence=f1,
        f2a_evidence=f2a,
        control_source=_control_source(tmp_path),
    )
    assert selection.layer == 12
    assert selection.f1_evidence_sha256 == f1.sha256
    assert selection.f2a_result_sha256 == f2a.result_sha256
    assert selection.f1_artifact_sha256 == f1.artifact_sha256
    assert selection.f2a_manifest_sha256 == f2a.manifest_sha256

    rejected = _gate_evidence(tmp_path / "rejected", "F1", status="not_supported")
    with pytest.raises(ValueError, match="F1 and F2A"):
        select_intervention(
            (_candidate(),),
            preregistration_sha256=PREREGISTRATION_SHA256,
            probe_selection_sha256=PROBE_SELECTION_SHA256,
            confirmatory_pins=PINS,
            f1_evidence=rejected,
            f2a_evidence=f2a,
            control_source=_control_source(tmp_path),
        )


def test_gate_verifier_rejects_self_authored_validation_evidence(tmp_path):
    store = FAArtifactStore(tmp_path)
    result = {"status": "supported"}
    result_sha256 = _canonical_hash(result)
    shard = store.write_completed_shard(
        "forged-gate",
        "locked_validation",
        "f1",
        [
            {
                "kind": "metrics",
                "phase": "F1",
                "status": "supported",
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "config_sha256": CONFIG_SHA256,
                "result_sha256": result_sha256,
                "result": result,
            }
        ],
        {
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "config_sha256": CONFIG_SHA256,
            "result_sha256": result_sha256,
        },
        record_kind="metrics",
    )
    with pytest.raises(ValueError, match="closed|behavior_test"):
        verify_gate_result_artifact(store, shard.manifest_path, phase="F1")


def test_gate_verifier_recomputes_f1_and_rejects_forged_supported_gate(tmp_path):
    store = FAArtifactStore(tmp_path)
    endpoint_input = store.write_completed_shard(
        "gate-run",
        "behavior_test",
        "inputs",
        [{"kind": "endpoint_input", "endpoint": "behavior_test"}],
        {"config_sha256": CONFIG_SHA256},
    )
    store.seal_endpoint(
        "behavior_test",
        [endpoint_input],
        {
            "preregistration": PREREGISTRATION_SHA256,
            "selection_manifest": "8" * 64,
        },
    )
    receipt = store.unlock_endpoint(
        "behavior_test", PREREGISTRATION_SHA256, "8" * 64
    )
    row = _f1_metrics_row(endpoint_input.sha256, supported=True)
    row["metrics"]["interaction"] = 0.0
    evidence = {
        name: row[name] for name in ("metrics", "bootstrap", "gate", "scored_rows")
    }
    row["evidence_sha256"] = _canonical_hash(evidence)
    metrics = store.write_completed_shard(
        "gate-run",
        "behavior_test",
        "metrics",
        [row],
        {
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "selection_sha256": "8" * 64,
            "prompt_manifest_sha256": endpoint_input.sha256,
            "evidence_sha256": row["evidence_sha256"],
            "config_sha256": CONFIG_SHA256,
        },
        record_kind="metrics",
    )
    store.mark_evaluated(receipt, metrics.data_path)
    store.close_endpoint("behavior_test")

    with pytest.raises(ValueError, match="canonical|recomputed"):
        verify_gate_result_artifact(store, endpoint_input.manifest_path, phase="F1")


def test_selection_requires_f1_and_f2a_from_same_store_and_run(tmp_path):
    f1 = _gate_evidence(tmp_path / "one", "F1")
    f2a = _gate_evidence(tmp_path / "two", "F2A")
    with pytest.raises(ValueError, match="same artifact store and run"):
        select_intervention(
            (_candidate(),),
            preregistration_sha256=PREREGISTRATION_SHA256,
            probe_selection_sha256=PROBE_SELECTION_SHA256,
            confirmatory_pins=PINS,
            f1_evidence=f1,
            f2a_evidence=f2a,
            control_source=_control_source(tmp_path),
        )


def test_gate_evidence_factory_cannot_be_called_without_verification():
    result_sha256 = _canonical_hash({"status": "supported"})
    with pytest.raises(ValueError, match="only come from artifact verification"):
        GateArtifactEvidence._from_verified_artifact(
            _verification_token=object(),
            phase="F1",
            status="supported",
            preregistration_sha256=PREREGISTRATION_SHA256,
            config_sha256=CONFIG_SHA256,
            result_sha256=result_sha256,
            artifact_sha256="a" * 64,
            manifest_sha256="b" * 64,
            probe_selection_sha256=None,
        )


def test_confirmatory_metrics_are_derived_from_decoded_output(tmp_path):
    result, store, prompts, _, _, pairs, _, _ = _evaluate(tmp_path)
    assert result.refit_performed is False
    assert result.h7_passed is True
    assert result.h8_passed is True
    assert result.metrics.high_to_low_effect == 1.0
    assert result.metrics.low_to_high_effect == 1.0
    assert result.metrics.familiarity_readout_effect == pytest.approx(0.30)
    assert result.metrics.generic_confidence_max_abs_change == pytest.approx(0.01)
    assert set(result.metrics.observed_domains) == set(REGISTERED_DOMAINS)
    assert store.endpoint_state("intervention_test", prompts.manifest_path) == "closed"
    assert result.example_ids == tuple(
        sorted(prompt.example_id for pair in pairs for prompt in (pair.high, pair.low))
    )


def test_executor_cannot_inject_independent_outcome_booleans(tmp_path):
    with pytest.raises(ValueError, match="only registered text and readouts"):
        _evaluate(tmp_path, executor=_Executor(injected_labels=True))
    fields = RawInterventionOutcome.__dataclass_fields__
    initializable = {name for name, value in fields.items() if value.init}
    assert "baseline_answer_attempt" not in initializable
    assert "patched_correct" not in initializable


def test_confirmatory_evaluation_persists_raw_before_bootstrap_metrics(tmp_path):
    result, store, _, _, _, _, _, _ = _evaluate(tmp_path)
    shards = store.resume_verified_shards("intervention-run", "intervention_test")
    raw = next(shard for shard in shards if shard.record_kind == "raw_intervention_outcomes")
    metrics = next(shard for shard in shards if shard.record_kind == "metrics")
    lineage = json.loads(metrics.manifest_path.read_text(encoding="utf-8"))["lineage"]
    assert raw.row_count == (
        len(REGISTERED_DOMAINS) * (1 + len(REQUIRED_CAUSAL_CONTROLS)) * 2
        + 2  # target-bound directions
        + len(REGISTERED_DOMAINS) * 2 * 2  # sources x directions x prompts
    )
    assert lineage["raw_outcomes_manifest_sha256"] == raw.sha256
    assert lineage["f1_evidence_sha256"] == _selection(tmp_path).f1_evidence_sha256
    assert result.metrics.bootstrap_summary["seed"] == 20260722
    assert result.metrics.bootstrap_summary["replicates"] == 10000
    assert all(
        values["holm_adjusted_p"] <= 0.05
        for values in result.metrics.bootstrap_summary["directions"].values()
    )


def test_h7_requires_primary_to_beat_every_negative_control(tmp_path):
    result, *_ = _evaluate(tmp_path, executor=_Executor(strong_control="orthogonal"))
    assert result.metrics.control_effects["orthogonal"] == (1.0, 1.0)
    assert result.h7_passed is False


def test_h7_uses_frozen_margin_only_for_random_and_cross_entity_controls():
    controls = {name: (0.0, 0.0) for name in REQUIRED_CAUSAL_CONTROLS}
    controls["orthogonal"] = (0.07, 0.07)
    controls["shuffled"] = (0.079, 0.079)
    controls["norm_matched_random"] = (0.05, 0.05)
    controls["cross_entity"] = (0.05, 0.05)
    assert _metrics(controls=controls).h7_passed is True

    equal_other = {**controls, "wrong_anchor": (0.08, 0.08)}
    assert _metrics(controls=equal_other).h7_passed is False

    insufficient_registered_margin = {
        **controls,
        "norm_matched_random": (0.061, 0.061),
    }
    assert _metrics(controls=insufficient_registered_margin).h7_passed is False


def test_h7_fails_closed_without_crossed_holm_evidence_or_all_domains():
    assert _metrics(bootstrap=False).h7_passed is False
    assert _metrics(
        observed_domains=REGISTERED_DOMAINS[:3],
        passing_domains=REGISTERED_DOMAINS[:3],
    ).h7_passed is False


def test_generic_confidence_tolerance_is_validation_frozen_and_enforced(tmp_path):
    assert _metrics(generic_confidence_change=0.03).h7_passed is False
    result, *_ = _evaluate(
        tmp_path,
        executor=_Executor(generic_confidence_change=0.03),
    )
    assert result.metrics.generic_confidence_max_abs_change == pytest.approx(0.03)
    assert result.h7_passed is False


def test_unrelated_h8_runs_the_selected_intervention_not_a_no_op(tmp_path):
    result, _, _, _, _, pairs, unrelated_prompts, executor = _evaluate(tmp_path)
    assert result.h8_passed
    assert len(executor.unrelated_calls) == (
        len(REGISTERED_DOMAINS) * 2 * len(unrelated_prompts)
    )
    assert all(
        intervention.apply_patch
        and intervention.control_name == "unrelated"
        and np.linalg.norm(intervention.vector) > 0
        for _, intervention in executor.unrelated_calls
    )
    assert {
        intervention.direction for _, intervention in executor.unrelated_calls
    } == {"high_to_low", "low_to_high"}
    assert {
        intervention.source_entity_unit_id
        for _, intervention in executor.unrelated_calls
    } == {pair.high.entity_unit_id for pair in pairs if pair.high.answerability == "code_absent"}
    assert result.metrics.unrelated_refusal_change_by_direction == {
        "high_to_low": 0.0,
        "low_to_high": 0.0,
    }
    assert result.metrics.unrelated_invalid_format_change_by_direction == {
        "high_to_low": 0.0,
        "low_to_high": 0.0,
    }


def test_h8_rejects_opposite_direction_capability_losses_that_average_to_zero(
    tmp_path,
):
    metrics = _metrics(
        unrelated_refusal_change=0.0,
        unrelated_refusal_by_direction={"high_to_low": 0.20, "low_to_high": -0.20},
    )
    assert metrics.h8_passed is False

    result, *_ = _evaluate(
        tmp_path,
        executor=_Executor(unrelated_directional_harm=True),
    )
    assert result.metrics.unrelated_refusal_change_by_direction == {
        "high_to_low": 1.0,
        "low_to_high": -1.0,
    }
    assert abs(result.metrics.unrelated_refusal_change) == 1.0
    assert result.h8_passed is False


def test_h8_rejects_target_bound_accuracy_loss_hidden_by_opposite_direction(
    tmp_path,
):
    metrics = _metrics(
        target_bound_accuracy_change=0.0,
        target_bound_accuracy_by_direction={
            "high_to_low": -0.10,
            "low_to_high": 0.10,
        },
    )
    assert metrics.h8_passed is False

    result, *_ = _evaluate(
        tmp_path,
        executor=_Executor(target_bound_directional_harm=True),
    )
    assert result.metrics.target_bound_accuracy_change_by_direction == {
        "high_to_low": -1.0,
        "low_to_high": 1.0,
    }
    assert result.metrics.target_bound_accuracy_change == -1.0
    assert result.h8_passed is False


def test_same_string_pair_allows_distinct_familiarization_prefills_but_binds_causal_invariants():
    pair = _pair()
    assert pair.high.input_ids != pair.low.input_ids
    assert pair.high.rendered_prefix_utf8 != pair.low.rendered_prefix_utf8
    assert not np.array_equal(pair.high.activation, pair.low.activation)
    assert pair.high.causal_invariants_sha256 == pair.low.causal_invariants_sha256


@pytest.mark.parametrize(
    "low",
    (
        lambda prompt: replace(prompt, target_string="Different"),
        lambda prompt: replace(prompt, answerability="target_bound"),
        lambda prompt: replace(prompt, relation_id="different_relation"),
        lambda prompt: replace(prompt, registry_code="DIFFERENT_CODE"),
        lambda prompt: replace(prompt, entity_type="different_type"),
        lambda prompt: replace(prompt, output_instruction="different instruction"),
        lambda prompt: replace(
            prompt,
            input_ids=(12, 13, 30, 99, 20),
            shared_query_suffix_token_ids=(30, 99, 20),
        ),
        lambda prompt: replace(
            prompt,
            rendered_prefix_utf8=b"<chat>different suffix",
            shared_query_suffix_utf8=b"different suffix",
        ),
        lambda prompt: replace(prompt, model_revision="b" * 40),
        lambda prompt: replace(prompt, tokenizer_revision="b" * 40),
        lambda prompt: replace(prompt, chat_template_sha256="b" * 64),
        lambda prompt: replace(prompt, activation_anchor="user_prompt_end"),
    ),
)
def test_same_string_pair_rejects_mismatched_registered_causal_invariant(low):
    pair = _pair()
    with pytest.raises(
        ValueError, match="causal invariant|registered site|preserve answerability"
    ):
        replace(pair, low=low(pair.low))


def test_confirmatory_pins_bind_model_config_and_source_pin_artifacts():
    assert PINS.canonical_payload() == {
        "model_id": "google/gemma-2-2b-it",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "source_pins_sha256": SOURCE_PINS_SHA256,
    }


def test_confirmatory_rejects_unrelated_substitution_and_pin_mismatch(tmp_path):
    selection = _selection(tmp_path)
    store, prompts, activations, unrelated, pairs, unrelated_prompts = (
        _sealed_intervention_endpoint(tmp_path, selection)
    )
    substitute = replace(
        unrelated_prompts[0],
        input_ids=(90, 91, 92),
        assistant_prefix_token_ids=(92,),
    )
    with pytest.raises(ValueError, match="unrelated prompts do not exactly match"):
        evaluate_intervention_test_once(
            selection,
            store,
            endpoint_manifest_path=prompts.manifest_path,
            activation_manifest_path=activations.manifest_path,
            unrelated_manifest_path=unrelated.manifest_path,
            test_pairs=pairs,
            unrelated_prompts=(substitute, unrelated_prompts[1]),
            executor=_Executor(),
            confirmatory_pins=PINS,
        )
    with pytest.raises(ValueError, match="confirmatory pins"):
        replace(PINS, model_revision="b" * 40)


def test_confirmatory_rejects_pair_substitution(tmp_path):
    def substitute(pairs):
        replacement = _pair(
            split="intervention_test",
            entity="substitute",
            activation_manifest_sha256=pairs[0].high.activation_manifest_sha256,
        )
        return (replacement, *pairs[1:])

    with pytest.raises(ValueError, match="sealed endpoint"):
        _evaluate(tmp_path, pairs_transform=substitute)


def test_confirmatory_rejects_missing_control_evidence(tmp_path):
    with pytest.raises(ValueError, match="missing executed control evidence"):
        _evaluate(tmp_path, executor=_Executor(missing_control="orthogonal"))


@pytest.mark.parametrize(
    ("executor", "message"),
    (
        (_Executor(decode_patch=True), "decode-time"),
        (_Executor(extra_site=True), "modified site"),
        (_Executor(forged_provenance=True), "activation provenance"),
        (_Executor(prefix_mismatch=True), "prefix"),
        (_Executor(forged_vector=True), "bind the applied intervention"),
        (_Executor(forged_intervention=True), "different intervention"),
    ),
)
def test_confirmatory_rejects_adversarial_execution(tmp_path, executor, message):
    with pytest.raises(ValueError, match=message):
        _evaluate(tmp_path, executor=executor)


def test_intervention_test_endpoint_is_one_use(tmp_path):
    result, store, prompts, activations, unrelated, pairs, unrelated_prompts, _ = _evaluate(
        tmp_path
    )
    assert result.h7_passed
    with pytest.raises(ValueError, match="already closed"):
        evaluate_intervention_test_once(
            _selection(tmp_path),
            store,
            endpoint_manifest_path=prompts.manifest_path,
            activation_manifest_path=activations.manifest_path,
            unrelated_manifest_path=unrelated.manifest_path,
            test_pairs=pairs,
            unrelated_prompts=unrelated_prompts,
            executor=_Executor(),
            confirmatory_pins=PINS,
        )


def test_wrong_endpoint_parent_and_forged_activation_manifest_fail_closed(tmp_path):
    selection = _selection(tmp_path)
    store, prompts, activations, unrelated, pairs, unrelated_prompts = (
        _sealed_intervention_endpoint(tmp_path, selection, selection_hash="0" * 64)
    )
    with pytest.raises(ValueError, match="selection"):
        evaluate_intervention_test_once(
            selection,
            store,
            endpoint_manifest_path=prompts.manifest_path,
            activation_manifest_path=activations.manifest_path,
            unrelated_manifest_path=unrelated.manifest_path,
            test_pairs=pairs,
            unrelated_prompts=unrelated_prompts,
            executor=_Executor(),
            confirmatory_pins=PINS,
        )

    good_store, good_prompts, _, good_unrelated, good_pairs, good_unrelated_prompts = (
        _sealed_intervention_endpoint(tmp_path / "forged", selection)
    )
    forged = good_store.write_completed_shard(
        "intervention-run",
        "intervention_test",
        "forged-activation",
        [{"example_id": "forged"}],
        {"preregistration_sha256": PREREGISTRATION_SHA256},
    )
    with pytest.raises(ValueError, match="sealed endpoint|activation manifest"):
        evaluate_intervention_test_once(
            selection,
            good_store,
            endpoint_manifest_path=good_prompts.manifest_path,
            activation_manifest_path=forged.manifest_path,
            unrelated_manifest_path=good_unrelated.manifest_path,
            test_pairs=good_pairs,
            unrelated_prompts=good_unrelated_prompts,
            executor=_Executor(),
            confirmatory_pins=PINS,
        )


def test_nested_payloads_and_arrays_are_deeply_immutable(tmp_path):
    metrics = _metrics(interval=[0.02, 0.14])
    assert isinstance(metrics.control_effects, MappingProxyType)
    with pytest.raises(TypeError):
        metrics.control_effects["orthogonal"] = (9.0, 9.0)

    runner = _FakeRunner()
    outcome = run_prefill_patch(
        runner,
        (_pair(),),
        _spec(),
        verified_activation_manifest_sha256="b" * 64,
    )
    assert isinstance(outcome.rows[0].decoded, MappingProxyType)
    with pytest.raises(TypeError):
        outcome.rows[0].decoded["nested"]["tokens"] += (100,)

    _, _, _, _, _, _, _, executor = _evaluate(tmp_path)
    intervention = executor.calls[0][1]
    assert intervention.vector.flags.writeable is False
    with pytest.raises(ValueError):
        intervention.vector[0] = 99


def test_public_contracts_expose_typed_interventions_not_caller_booleans():
    execute_parameters = ConfirmatoryInterventionExecutor.execute.__annotations__
    unrelated_parameters = ConfirmatoryInterventionExecutor.execute_unrelated.__annotations__
    selection_parameters = select_intervention.__annotations__
    assert "intervention" in execute_parameters and "control_name" not in execute_parameters
    assert "intervention" in unrelated_parameters
    assert "f1_evidence" in selection_parameters and "f2a_evidence" in selection_parameters
    assert "f1_passed" not in selection_parameters and "f2a_passed" not in selection_parameters


def test_h7_requires_preregistered_answerability_norm_residualized_control():
    assert "answerability_norm_residualized" in REQUIRED_CAUSAL_CONTROLS
    incomplete = {
        name: (0.0, 0.0)
        for name in REQUIRED_CAUSAL_CONTROLS
        if name != "answerability_norm_residualized"
    }
    assert _metrics(controls=incomplete).h7_passed is False


def test_validation_control_source_recomputes_bound_locked_validation_derivations(
    tmp_path,
):
    source = _control_source(tmp_path)

    assert source.source_split == "locked_validation"
    assert source.fit_method == CONTROL_FIT_METHOD
    assert source.permutation_seed == CONTROL_PERMUTATION_SEED
    assert source.example_ids == tuple(f"control-{index}" for index in range(8))
    assert source.input_row_ids == tuple(
        f"input-control-{index}" for index in range(8)
    )
    assert source.activation_row_ids == tuple(
        f"activation-control-{index}" for index in range(8)
    )
    assert len(source.input_row_sha256s) == len(source.example_ids)
    assert len(source.activation_row_sha256s) == len(source.example_ids)
    assert source.input_artifact_sha256
    assert source.activation_artifact_sha256
    assert source.permutation_sha256 == _canonical_hash(
        np.random.default_rng(CONTROL_PERMUTATION_SEED)
        .permutation(8)
        .tolist()
    )
    assert source.direction_derivation_sha256 == _canonical_hash(
        CONTROL_DIRECTION_DERIVATION
    )


def test_validation_control_source_rejects_arbitrary_hash_consistent_named_arrays(
    tmp_path,
):
    store, shard = _write_control_source(
        tmp_path,
        vector_override={"shuffled_direction": [9.0, 8.0, 7.0]},
        shard_id="forged-f2b-controls",
    )

    with pytest.raises(ValueError, match="recomputed|derivation"):
        verify_validation_control_artifact(
            store,
            shard.manifest_path,
            preregistration_sha256=PREREGISTRATION_SHA256,
            probe_selection_sha256=PROBE_SELECTION_SHA256,
        )


def test_shuffled_control_is_frozen_from_locked_validation_and_applied(tmp_path):
    selection = _selection(tmp_path)
    assert selection.control_source.source_split == "locked_validation"
    assert selection.control_source.source_artifact_sha256 == selection.control_source_artifact_sha256

    _, _, _, _, _, _, _, executor = _evaluate(tmp_path)
    shuffled = [
        intervention
        for _, intervention in executor.calls
        if intervention.control_name == "shuffled"
    ]
    assert shuffled
    assert all(
        intervention.control_source_split == "locked_validation"
        and intervention.control_source_artifact_sha256
        == selection.control_source_artifact_sha256
        and intervention.control_source_component_sha256
        == selection.control_source.shuffled_direction_sha256
        and intervention.vector_sha256 == audit.applied_vector_sha256
        for intervention, audit in zip(
            shuffled,
            [
                audit
                for (_, intervention), audit in zip(executor.calls, executor.audits)
                if intervention.control_name == "shuffled"
            ],
            strict=True,
        )
    )


def test_executed_cross_entity_vector_is_norm_matched_to_primary_direction(tmp_path):
    _, _, _, _, _, pairs, _, executor = _evaluate(tmp_path)
    prompt_by_id = {
        prompt.example_id: prompt
        for pair in pairs
        for prompt in (pair.high, pair.low)
    }
    primary = {
        (intervention.destination_example_id, intervention.direction): intervention
        for _, intervention in executor.calls
        if intervention.control_name == "primary"
    }
    cross_entity = [
        intervention
        for _, intervention in executor.calls
        if intervention.control_name == "cross_entity"
    ]

    assert cross_entity
    for intervention in cross_entity:
        matched = primary[
            (intervention.destination_example_id, intervention.direction)
        ]
        destination = prompt_by_id[intervention.destination_example_id]
        assert intervention.source_entity_unit_id != destination.entity_unit_id
        assert np.linalg.norm(intervention.vector) == pytest.approx(
            np.linalg.norm(matched.vector)
        )
        np.testing.assert_allclose(
            intervention.replacement,
            destination.activation + intervention.vector,
        )


def test_site_and_no_intervention_controls_execute_exact_registered_plan(tmp_path):
    selection = _selection(tmp_path)
    _, _, _, _, _, pairs, _, executor = _evaluate(tmp_path)
    prompt_by_id = {
        prompt.example_id: prompt
        for pair in pairs
        for prompt in (pair.high, pair.low)
    }
    executed = list(zip(executor.calls, executor.audits, strict=False))

    no_intervention = [
        (intervention, audit)
        for (_, intervention), audit in executed
        if intervention.control_name == "no_intervention"
    ]
    wrong_layer = [
        (intervention, audit)
        for (_, intervention), audit in executed
        if intervention.control_name == "wrong_layer"
    ]
    wrong_anchor = [
        (intervention, audit)
        for (_, intervention), audit in executed
        if intervention.control_name == "wrong_anchor"
    ]

    assert no_intervention and wrong_layer and wrong_anchor
    for intervention, audit in no_intervention:
        destination = prompt_by_id[intervention.destination_example_id]
        assert intervention.apply_patch is False
        assert audit.modified_sites == ()
        assert np.linalg.norm(intervention.vector) == 0.0
        np.testing.assert_array_equal(intervention.replacement, destination.activation)
    for intervention, audit in wrong_layer:
        assert intervention.layer != selection.layer
        assert intervention.anchor == selection.anchor
        assert audit.modified_sites == ((intervention.layer, intervention.position),)
    for intervention, audit in wrong_anchor:
        destination = prompt_by_id[intervention.destination_example_id]
        assert intervention.layer == selection.layer
        assert intervention.anchor != selection.anchor
        assert intervention.position == destination.anchor_positions[intervention.anchor]
        assert audit.modified_sites == ((intervention.layer, intervention.position),)


def test_bootstrap_records_complete_requested_valid_and_discarded_draw_accounting(
    tmp_path,
):
    result, *_ = _evaluate(tmp_path)
    summary = result.metrics.bootstrap_summary
    assert summary["requested_draws"] == 10_000
    assert type(summary["valid_draws"]) is int and summary["valid_draws"] > 0
    assert type(summary["discarded_draws"]) is int and summary["discarded_draws"] >= 0
    assert summary["valid_draws"] + summary["discarded_draws"] == summary["requested_draws"]
    assert summary["resampling_unit"] == ("entity_unit_id", "template_family")
    assert summary["seed"] == 20260722
    assert summary["alpha"] == pytest.approx(0.05)


def test_evaluated_intervention_endpoint_recovers_without_reexecution(
    tmp_path, monkeypatch
):
    selection = _selection(tmp_path)
    store, prompts, activations, unrelated, pairs, unrelated_prompts = (
        _sealed_intervention_endpoint(tmp_path, selection)
    )
    executor = _Executor()
    original_close = store.close_endpoint

    def interrupt_before_close(endpoint):
        raise RuntimeError(f"interrupted before closing {endpoint}")

    monkeypatch.setattr(store, "close_endpoint", interrupt_before_close)
    with pytest.raises(RuntimeError, match="interrupted before closing"):
        evaluate_intervention_test_once(
            selection,
            store,
            endpoint_manifest_path=prompts.manifest_path,
            activation_manifest_path=activations.manifest_path,
            unrelated_manifest_path=unrelated.manifest_path,
            test_pairs=pairs,
            unrelated_prompts=unrelated_prompts,
            executor=executor,
            confirmatory_pins=PINS,
        )
    assert store.endpoint_state("intervention_test", prompts.manifest_path) == "evaluated"
    calls_before_recovery = (len(executor.calls), len(executor.unrelated_calls))
    metrics_shard = next(
        shard
        for shard in store.resume_verified_shards(
            "intervention-run", "intervention_test"
        )
        if shard.record_kind == "metrics"
    )
    expected_payload = json.loads(
        metrics_shard.data_path.read_text(encoding="utf-8")
    )["result"]

    close_calls = []

    def counted_close(endpoint):
        close_calls.append(endpoint)
        return original_close(endpoint)

    monkeypatch.setattr(store, "close_endpoint", counted_close)
    recovered = evaluate_intervention_test_once(
        selection,
        store,
        endpoint_manifest_path=prompts.manifest_path,
        activation_manifest_path=activations.manifest_path,
        unrelated_manifest_path=unrelated.manifest_path,
        test_pairs=pairs,
        unrelated_prompts=unrelated_prompts,
        executor=executor,
        confirmatory_pins=PINS,
    )
    assert recovered.result_sha256 == expected_payload["result_sha256"]
    assert interventions_module._result_payload(recovered) == expected_payload
    assert (len(executor.calls), len(executor.unrelated_calls)) == calls_before_recovery
    assert close_calls == ["intervention_test"]
    assert store.endpoint_state("intervention_test", prompts.manifest_path) == "closed"


def test_destination_identity_is_bound_into_intervention_and_adapter_audit(tmp_path):
    _, _, _, _, _, _, _, executor = _evaluate(tmp_path)
    assert executor.calls
    for (destination_entity, intervention), audit in zip(
        executor.calls, executor.audits[: len(executor.calls)], strict=True
    ):
        assert intervention.destination_entity_unit_id == destination_entity
        assert intervention.destination_example_id
        assert intervention.destination_evidence_sha256
        assert audit.destination_example_id == intervention.destination_example_id
        assert audit.destination_evidence_sha256 == intervention.destination_evidence_sha256
        assert audit.sha256 == audit.audit_sha256
