"""Claim recomputation and publishable artifacts for the FA study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
import ctypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_config import (
    CONFIRMATORY_BOOTSTRAP_REPLICATES,
    CONFIRMATORY_BOOTSTRAP_SEED,
)
from trajectory_extractor.fa_interventions import (
    REQUIRED_CAUSAL_CONTROLS,
    InterventionMetrics,
    InterventionTestResult,
)
from trajectory_extractor.fa_probes import (
    DEFAULT_FULL_SELECTION_NULL_SEED_HASH,
    DEFAULT_FULL_SELECTION_NULL_SEEDS,
    F2AGates,
    NullSelectionResult,
    ProbeResult,
    SAEGate,
    evaluate_f2a_gates,
)
from trajectory_extractor.fa_scoring import (
    BehavioralGate,
    BehavioralMetrics,
    BootstrapDistribution,
    SameStringSealEvidence,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REGISTERED_DOMAINS = frozenset(
    {"person", "place", "organization", "creative_work"}
)
_H7_BOOTSTRAP_METHOD = "crossed_entity_unit_template_family_bootstrap"
_H7_BOOTSTRAP_SEED = 20260722
_H7_BOOTSTRAP_REPLICATES = 10_000
_H7_ALPHA = 0.05
_H7_DIRECTIONS = frozenset({"high_to_low", "low_to_high"})
_H7_RESAMPLING_UNIT = ("entity_unit_id", "template_family")
_H7_BOOTSTRAP_ROOT_FIELDS = frozenset(
    {
        "method",
        "seed",
        "replicates",
        "requested_draws",
        "valid_draws",
        "discarded_draws",
        "resampling_unit",
        "alpha",
        "directions",
    }
)
_F1_MINIMUM_VALID_DRAWS = 50
_F1_ALPHA = 0.05
_F1_RESAMPLING_UNIT = ("entity_unit_id", "template_family")


@dataclass(frozen=True)
class ClaimDecision:
    status: str
    claim: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class F1Evidence:
    """Immutable F1 records emitted by the behavioral analysis."""

    metrics: BehavioralMetrics
    bootstrap: BootstrapDistribution
    gate: BehavioralGate

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, BehavioralMetrics):
            raise ValueError("F1 evidence requires BehavioralMetrics")
        if not isinstance(self.bootstrap, BootstrapDistribution):
            raise ValueError("F1 evidence requires BootstrapDistribution")
        if not isinstance(self.gate, BehavioralGate):
            raise ValueError("F1 evidence requires BehavioralGate")

    def to_record(self) -> Mapping[str, Any]:
        return {
            "metrics": self.metrics.to_record(),
            "bootstrap": self.bootstrap.to_record(),
            "gate": self.gate.to_record(),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_record())).hexdigest()


@dataclass(frozen=True)
class F2AEvidence:
    """Immutable probe outputs and their joint, typed F2A gate record."""

    familiarity: ProbeResult
    answerability: ProbeResult
    unsupported_answer: ProbeResult
    gates: F2AGates
    sae_gates: Mapping[str, SAEGate]

    def __post_init__(self) -> None:
        results = (self.familiarity, self.answerability, self.unsupported_answer)
        if any(not isinstance(result, ProbeResult) for result in results):
            raise ValueError("F2A evidence requires typed ProbeResult records")
        if tuple(result.task for result in results) != (
            "familiarity",
            "answerability",
            "unsupported_answer",
        ):
            raise ValueError("F2A evidence has incorrect probe tasks")
        if not isinstance(self.gates, F2AGates):
            raise ValueError("F2A evidence requires F2AGates")
        if (
            self.gates.familiarity_result_sha256 != self.familiarity.sha256
            or self.gates.answerability_result_sha256 != self.answerability.sha256
            or self.gates.unsupported_result_sha256 != self.unsupported_answer.sha256
        ):
            raise ValueError("F2A gate record is not bound to the supplied probe results")
        canonical = evaluate_f2a_gates(*results)
        if canonical.sha256 != self.gates.sha256:
            raise ValueError(
                "stored F2A gates do not match canonical F2A gate recomputation"
            )
        if not isinstance(self.sae_gates, Mapping) or any(
            not isinstance(name, str) or not name or not isinstance(gate, SAEGate)
            for name, gate in self.sae_gates.items()
        ):
            raise ValueError("F2A SAE evidence must be named SAEGate records")
        object.__setattr__(self, "sae_gates", dict(sorted(self.sae_gates.items())))

    def to_record(self) -> Mapping[str, Any]:
        return {
            "familiarity": self.familiarity.to_record(),
            "answerability": self.answerability.to_record(),
            "unsupported_answer": self.unsupported_answer.to_record(),
            "gates": self.gates.to_record(),
            "sae_gates": {
                name: gate.to_record() for name, gate in self.sae_gates.items()
            },
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_record())).hexdigest()


@dataclass(frozen=True)
class CircuitFailure:
    """One immutable failed graph attempt included in the circuit denominator."""

    case_id: str
    stage: str
    error_code: str
    detail_sha256: str

    def __post_init__(self) -> None:
        for name in ("case_id", "stage", "error_code"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"circuit failure {name} must be nonempty")
        _sha256_value(self.detail_sha256, "circuit failure detail_sha256")

    def to_record(self) -> Mapping[str, str]:
        return {
            "case_id": self.case_id,
            "stage": self.stage,
            "error_code": self.error_code,
            "detail_sha256": self.detail_sha256,
        }


@dataclass(frozen=True)
class CircuitGateEvidence:
    """Hash-bound prompt-local replacement-model and causal gate evidence."""

    config_sha256: str
    preregistration_sha256: str
    case_manifest_sha256: str
    replacement_model_sha256: str
    intervention_result_sha256: str
    proxy_spearman: float
    distribution_spearman: float
    perturbation_spearman: float
    sign_concordance: float
    error_node_share: float
    attempted: int
    successful: int
    failures: tuple[CircuitFailure, ...]
    original_model_intervention_supported: bool

    def __post_init__(self) -> None:
        for name in (
            "config_sha256",
            "preregistration_sha256",
            "case_manifest_sha256",
            "replacement_model_sha256",
            "intervention_result_sha256",
        ):
            _sha256_value(getattr(self, name), name)
        for name in (
            "proxy_spearman",
            "distribution_spearman",
            "perturbation_spearman",
            "sign_concordance",
        ):
            value = _finite_number(getattr(self, name), name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
        error_share = _finite_number(self.error_node_share, "error_node_share")
        if not 0.0 <= error_share <= 1.0:
            raise ValueError("error_node_share must be in [0, 1]")
        if (
            type(self.attempted) is not int
            or type(self.successful) is not int
            or self.attempted < 0
            or not 0 <= self.successful <= self.attempted
        ):
            raise ValueError("circuit graph counts are invalid")
        failures = tuple(self.failures)
        if any(not isinstance(failure, CircuitFailure) for failure in failures):
            raise ValueError("circuit failures must be typed CircuitFailure records")
        if len(failures) != self.attempted - self.successful:
            raise ValueError("circuit failures must account for every failed graph attempt")
        if len({failure.case_id for failure in failures}) != len(failures):
            raise ValueError("circuit failures contain duplicate case IDs")
        if type(self.original_model_intervention_supported) is not bool:
            raise ValueError("original-model intervention support must be boolean")
        object.__setattr__(self, "failures", failures)

    @property
    def graph_yield(self) -> float | None:
        return None if self.attempted == 0 else self.successful / self.attempted

    def to_record(self) -> Mapping[str, Any]:
        return {
            "config_sha256": self.config_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "case_manifest_sha256": self.case_manifest_sha256,
            "replacement_model_sha256": self.replacement_model_sha256,
            "intervention_result_sha256": self.intervention_result_sha256,
            "proxy_spearman": self.proxy_spearman,
            "distribution_spearman": self.distribution_spearman,
            "perturbation_spearman": self.perturbation_spearman,
            "sign_concordance": self.sign_concordance,
            "error_node_share": self.error_node_share,
            "attempted": self.attempted,
            "successful": self.successful,
            "graph_yield": self.graph_yield,
            "failures": [failure.to_record() for failure in self.failures],
            "original_model_intervention_supported": self.original_model_intervention_supported,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_record())).hexdigest()


def _before_source_open(path: Path) -> None:
    """Test seam after descriptor-safe parent traversal and before leaf open."""

    del path


def _before_release_publish(destination: Path) -> None:
    """Test seam immediately before the exclusive directory publication."""

    del destination


def recompute_claim_ladder(
    *,
    behavior: F1Evidence | None,
    f2a: F2AEvidence | None,
    f2b: InterventionTestResult | None,
    circuit: CircuitGateEvidence | None,
) -> Mapping[str, ClaimDecision]:
    """Recompute every allowed claim from metrics, never stored booleans."""

    if behavior is not None and not isinstance(behavior, F1Evidence):
        raise ValueError("behavior must be canonical typed F1Evidence")
    if f2a is not None and not isinstance(f2a, F2AEvidence):
        raise ValueError("f2a must be canonical typed F2AEvidence")
    if f2b is not None and not isinstance(f2b, InterventionTestResult):
        raise ValueError("f2b must be a canonical typed InterventionTestResult")
    if circuit is not None and not isinstance(circuit, CircuitGateEvidence):
        raise ValueError("circuit evidence must be typed CircuitGateEvidence")
    if circuit is not None and f2b is not None:
        if circuit.intervention_result_sha256 != f2b.result_sha256:
            raise ValueError("circuit evidence is not bound to the supplied intervention result")
        if circuit.preregistration_sha256 != f2b.preregistration_sha256:
            raise ValueError("circuit and intervention preregistration hashes differ")

    h2 = _behavior_h2(behavior)
    h1 = _behavior_h1(behavior, h2)
    h2b = _behavior_h2b(behavior)
    h3 = _probe_claim(f2a, "H3", "familiarity")
    h4 = _probe_claim(f2a, "H4", "answerability")
    h5 = _h5_claim(f2a)
    h6 = _h6_claim(f2a)
    h7, h8 = _intervention_claims(f2b, h1, h3, h4)
    f3 = _circuit_claim(circuit, h3, h4, h5, h7)
    return {
        "H1": h1,
        "H2": h2,
        "H2b": h2b,
        "H3": h3,
        "H4": h4,
        "H5": h5,
        "H6": h6,
        "H7": h7,
        "H8": h8,
        "F3": f3,
    }


def build_report(
    *,
    behavior: F1Evidence | None,
    f2a: F2AEvidence | None,
    f2b: InterventionTestResult | None,
    circuit: CircuitGateEvidence | None,
    output: str | Path,
) -> Path:
    claims = recompute_claim_ladder(
        behavior=behavior,
        f2a=f2a,
        f2b=f2b,
        circuit=circuit,
    )
    report_inputs = _report_input_metadata_record(
        behavior=behavior,
        f2a=f2a,
        f2b=f2b,
        circuit=circuit,
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Familiarity vs. Answerability Evidence Report",
        f"<!-- fa-report-inputs:{_compact_json(report_inputs)} -->",
        "",
        "This report is generated from canonical metrics. It does not treat stored claim booleans as evidence.",
        "",
        "## Claim Ladder",
        "",
    ]
    for name in ("H1", "H2", "H2b", "H3", "H4", "H5", "H6", "H7", "H8", "F3"):
        decision = claims[name]
        lines.append(f"- **{name}: {decision.status}** - {decision.claim}")
        if decision.reasons:
            lines.append(f"  Reasons: {'; '.join(decision.reasons)}")

    lines.extend(["", "## Phase Status", ""])
    lines.append(f"- F1: {_f1_phase_status(behavior)}")
    lines.append(f"- F2A: {_f2a_phase_status(f2a)}")
    lines.append(f"- F2B: {_f2b_phase_status(f2b)}")
    lines.append(f"- F3: {'evaluated' if circuit is not None else 'skipped'}")

    lines.extend(["", "## Denominators And Missingness", ""])
    if behavior is None:
        lines.append("- F1: unavailable")
        lines.append("- Invalid outputs: unavailable")
    else:
        lines.append(f"- F1 denominator: {sum(behavior.metrics.denominators.values())}")
        lines.append(f"- F1 incomplete rows: {_incomplete_behavior_rows(behavior.metrics)}")
        lines.append(f"- Invalid outputs: {sum(behavior.metrics.invalid_format_counts.values())}")
        for field in (
            "requested_draws",
            "valid_draws",
            "discarded_draws",
            "resampling_unit",
            "seed",
            "alpha",
        ):
            lines.append(
                f"- F1 bootstrap {field}: "
                f"{_compact_json(_plain_tree(getattr(behavior.bootstrap, field)))}"
            )
        seal = behavior.gate.same_string_seal
        if isinstance(seal, SameStringSealEvidence):
            lines.append(f"- F1 same-string seal_sha256: {seal.sha256}")
            lines.append(
                "- F1 same-string registered_block_sha256: "
                f"{seal.registered_block_sha256}"
            )
            lines.append(
                "- F1 same-string source_manifest_sha256: "
                f"{seal.source_manifest_sha256}"
            )
        else:
            lines.append("- F1 same-string seal: missing/not_evaluable")
        for cell, count in sorted(behavior.metrics.invalid_format_counts.items()):
            lines.append(f"- F1 invalid_format {':'.join(cell)}: {count}")
    if f2a is None:
        lines.append("- F2A: unavailable")
    else:
        for task, kind, result in (
            ("familiarity", "label_permutation", f2a.familiarity),
            ("answerability", "label_permutation", f2a.answerability),
            ("unsupported_answer", "layer_order", f2a.unsupported_answer),
            ("unsupported_answer", "random_map", f2a.unsupported_answer),
        ):
            nulls = tuple(null for null in result.null_results if null.kind == kind)
            hashes = {
                null.config.get("seed_list_sha256")
                for null in nulls
                if isinstance(null.config, Mapping)
            }
            seed_hash = next(iter(hashes)) if len(hashes) == 1 else "invalid"
            lines.append(
                f"- F2A null provenance {task}/{kind}: count={len(nulls)}, "
                f"seed_list_sha256={seed_hash}"
            )
        for result in (f2a.familiarity, f2a.answerability, f2a.unsupported_answer):
            metrics = result.metrics
            lines.append(
                f"- F2A {result.task}: total={metrics.total}, denominator={metrics.denominator}, missing={metrics.missing}, invalid={metrics.invalid}, classes={_compact_json(metrics.to_record()['class_counts'])}"
            )
        lines.append(f"- F2A invalid rows: {sum(result.metrics.invalid for result in (f2a.familiarity, f2a.answerability, f2a.unsupported_answer))}")

    lines.extend(["", "## Nulls, Ablations, And Failures", ""])
    if f2a is None:
        lines.append("- F2A null distributions: unavailable")
        lines.append("- SAE analysis: skipped/not_run")
    else:
        lines.append(f"- F2A Holm p-values: {_compact_json(dict(f2a.gates.holm_adjusted_p))}")
        for result in (f2a.familiarity, f2a.answerability, f2a.unsupported_answer):
            if not result.null_results:
                lines.append(f"- F2A null {result.task}: none/not_run")
            for null in result.null_results:
                metrics = null.test_metrics
                if metrics is None:
                    lines.append(
                        f"- F2A null {result.task} {null.kind} seed={null.seed}: not_evaluable"
                    )
                else:
                    lines.append(
                        f"- F2A null {result.task} {null.kind} seed={null.seed}: "
                        f"{_format_probe_metrics(metrics)}, "
                        f"h5_improvement={null.test_relative_h5_log_loss_improvement}, "
                        f"h6_improvement={null.test_relative_h6_log_loss_improvement}"
                    )
            for dimension, groups in sorted(result.ood_transfer.items()):
                for group, metrics in sorted(groups.items()):
                    lines.append(
                        f"- F2A OOD {dimension}/{group} ({result.task}): "
                        f"{_format_probe_metrics(metrics)}"
                    )
                worst = result.worst_ood_transfer.get(dimension)
                lines.append(
                    f"- F2A OOD {dimension} ({result.task}): "
                    + (
                        "not_evaluable, denominator=0"
                        if worst is None
                        else f"worst {_format_probe_metrics(worst)}"
                    )
                )
        if not f2a.sae_gates:
            lines.append("- SAE analysis: skipped/not_run")
        else:
            for name, gate in f2a.sae_gates.items():
                lines.append(f"- SAE {name}: {'passed' if gate.passed else 'failed'} ({'; '.join(gate.reasons) or 'all registered criteria satisfied'})")
    if f2b is not None:
        lines.append(f"- F2B completed fraction: {f2b.metrics.completed_fraction:.3f}")
        lines.append(f"- F2B H7 evidence: {claims['H7'].status} ({'; '.join(claims['H7'].reasons) or 'all registered criteria satisfied'})")
        lines.append(f"- F2B H8 evidence: {claims['H8'].status} ({'; '.join(claims['H8'].reasons) or 'all registered criteria satisfied'})")
        for name, effects in sorted(f2b.metrics.control_effects.items()):
            lines.append(
                f"- F2B control {name}: high_to_low={effects[0]:.6g}, "
                f"low_to_high={effects[1]:.6g}"
            )
        bootstrap = _plain_tree(f2b.metrics.bootstrap_summary)
        audit_labels = (
            ("method", "method"),
            ("requested_draws", "requested_draws"),
            ("valid_draws", "valid_draws"),
            ("discarded_draws", "discarded_draws"),
            ("resampling_unit", "resampling_unit"),
            ("seed", "seed"),
            ("alpha", "alpha"),
            ("replicates", "replicates"),
        )
        for field, label in audit_labels:
            value = bootstrap.get(field) if isinstance(bootstrap, dict) else None
            rendered = "missing" if value is None else _compact_json(value)
            lines.append(f"- F2B bootstrap {label}: {rendered}")
        directions = bootstrap.get("directions", {}) if isinstance(bootstrap, dict) else {}
        if not directions:
            lines.append("- F2B crossed bootstrap: missing/not_evaluable")
        else:
            for direction in sorted(directions):
                lines.append(
                    f"- F2B bootstrap {direction}: "
                    f"{_compact_json(directions[direction])}"
                )
    else:
        lines.append("- F2B controls: skipped/not_run")
        lines.append("- F2B crossed bootstrap: skipped/not_run")
    if circuit is not None:
        lines.append(
            f"- Circuit graph yield: {circuit.successful}/{circuit.attempted} "
            f"({circuit.graph_yield if circuit.graph_yield is not None else 'not_evaluable'})"
        )
        lines.append(f"- Circuit error-node share: {circuit.error_node_share}")
        lines.append(
            "- Circuit original-model intervention support: "
            f"{circuit.original_model_intervention_supported}"
        )
        if not circuit.failures:
            lines.append("- Circuit failures: none")
        for failure in circuit.failures:
            lines.append(
                f"- Circuit failure {failure.case_id}: {failure.error_code} "
                f"(stage={failure.stage}, detail_sha256={failure.detail_sha256})"
            )
    else:
        lines.append("- Circuit attempts: not run")

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Supported results are local to the registered model, prompts, anchors, and endpoints. They do not establish human-like intuition, universal truth representations, general metacognition, or jailbreak prevention.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _report_input_metadata_record(
    *,
    behavior: F1Evidence | None,
    f2a: F2AEvidence | None,
    f2b: InterventionTestResult | None,
    circuit: CircuitGateEvidence | None,
) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    if behavior is not None:
        hashes["F1"] = behavior.sha256
    if f2a is not None:
        core = {
            "familiarity": f2a.familiarity.to_record(),
            "answerability": f2a.answerability.to_record(),
            "unsupported_answer": f2a.unsupported_answer.to_record(),
            "gates": f2a.gates.to_record(),
        }
        hashes["F2A"] = hashlib.sha256(_canonical_json(core)).hexdigest()
    if f2b is not None:
        hashes["F2B"] = f2b.result_sha256
    if circuit is not None:
        hashes["F3"] = circuit.sha256
    bundle = {
        "schema_version": 1,
        "phase_evidence_sha256": dict(sorted(hashes.items())),
    }
    return {
        **bundle,
        "report_input_bundle_sha256": hashlib.sha256(
            _canonical_json(bundle)
        ).hexdigest(),
    }


def build_release_bundle(
    *,
    source_root: str | Path,
    output: str | Path,
    allowlist: Mapping[str, str],
    config_hash: str,
    preregistration_hash: str,
    artifact_store: FAArtifactStore | None = None,
    core_endpoint_manifests: Mapping[str, str | Path] | None = None,
    retrieval_records: Sequence[Mapping[str, str]] = (),
) -> Path:
    """Build a no-clobber release from hashed files and verified FA artifacts."""

    source = Path(source_root).absolute()
    destination = Path(output).absolute()
    if not source.is_dir() or source.is_symlink():
        raise ValueError("source_root must be a real directory")
    if not isinstance(allowlist, Mapping) or not allowlist:
        raise ValueError("release allowlist must be nonempty")
    if not isinstance(artifact_store, FAArtifactStore) or artifact_store.root != source:
        raise ValueError("release requires an FAArtifactStore that owns source_root")
    _sha256_value(config_hash, "config_hash")
    _sha256_value(preregistration_hash, "preregistration_hash")
    retrieval = _normalize_retrieval_records(retrieval_records)
    allowlist_names = set(allowlist)
    for name in allowlist_names:
        if name.endswith(".jsonl") and f"{name}.manifest.json" not in allowlist_names:
            raise ValueError("data shards and sidecars must be released together")
        if (
            name.endswith(".jsonl.manifest.json")
            and name.removesuffix(".manifest.json") not in allowlist_names
        ):
            raise ValueError("data shards and sidecars must be released together")
    core_endpoints = _verify_core_endpoint_evidence(
        artifact_store,
        core_endpoint_manifests,
        allowlist,
        preregistration_hash,
    )
    source_descriptor = _open_directory_descriptor(source, "source_root")
    try:
        report_inputs = _verify_report_inputs_at_descriptor(
            source_descriptor, allowlist, core_endpoints
        )
    except BaseException:
        os.close(source_descriptor)
        raise
    destination_parent_descriptor = _open_directory_descriptor(
        destination.parent, "release output parent"
    )
    staging_name, staging_descriptor = _create_staging_directory_at(
        destination_parent_descriptor, destination.name
    )
    published = False
    try:
        files: dict[str, dict[str, Any]] = {}
        source_manifests: dict[str, dict[str, str]] = {}
        for relative_name in sorted(allowlist):
            relative = _safe_relative(relative_name)
            expected = allowlist[relative_name]
            if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
                raise ValueError("allowlist hash must be a lowercase SHA-256")
            source_path = source / relative
            _before_source_open(source_path)
            payload = _read_regular_bytes_at_descriptor(
                source_descriptor, relative, "release source"
            )
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected:
                raise ValueError(f"release source hash mismatch: {relative_name}")
            if relative.name.endswith(".jsonl"):
                manifest_relative = Path(f"{relative.as_posix()}.manifest.json")
                manifest_path = source / manifest_relative
                shard = artifact_store.verify_shard(manifest_path)
                if shard.data_path != source_path or shard.sha256 != actual:
                    raise ValueError("release shard does not match its verified sidecar")
                if shard.namespace in {"behavior_test", "probe_test", "intervention_test"}:
                    if not _protected_shard_is_closed(artifact_store, shard):
                        raise ValueError("protected experimental evidence must have a closed endpoint")
                _before_source_open(manifest_path)
                manifest_payload = _read_regular_bytes_at_descriptor(
                    source_descriptor, manifest_relative, "shard sidecar"
                )
                source_manifests[manifest_relative.as_posix()] = {
                    "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                    "shard_sha256": shard.sha256,
                }
            _write_new_regular_at(staging_descriptor, relative, payload)
            copied = _read_regular_bytes_at_descriptor(
                staging_descriptor, relative, "release copy"
            )
            if hashlib.sha256(copied).hexdigest() != actual:
                raise ValueError(f"release copy hash mismatch: {relative_name}")
            files[relative.as_posix()] = {"sha256": actual, "bytes": len(payload)}
        identity = {
            "schema_version": 4,
            "files": files,
            "source_manifests": source_manifests,
            "core_endpoints": core_endpoints,
            "report_inputs": report_inputs,
            "config_hash": config_hash,
            "preregistration_hash": preregistration_hash,
            "retrieval_records": retrieval,
        }
        top_hash = hashlib.sha256(_canonical_json(identity)).hexdigest()
        manifest = {**identity, "top_level_sha256": top_hash}
        _write_new_regular_at(
            staging_descriptor, Path("MANIFEST.json"), _canonical_json(manifest) + b"\n"
        )
        _write_new_regular_at(
            staging_descriptor,
            Path("MANIFEST.sha256"),
            f"{top_hash}  MANIFEST.json\n".encode("ascii"),
        )
        os.fsync(staging_descriptor)
        _before_release_publish(destination)
        if not _same_directory(destination.parent, destination_parent_descriptor):
            raise ValueError("release output parent changed during publication")
        _rename_directory_noreplace(
            destination_parent_descriptor, staging_name, destination.name
        )
        published = True
        if not _same_directory(destination.parent, destination_parent_descriptor):
            _remove_tree_at(destination_parent_descriptor, destination.name)
            published = False
            raise ValueError("release output parent changed during publication")
    except BaseException:
        if not published:
            _remove_tree_at(destination_parent_descriptor, staging_name)
        raise
    finally:
        os.close(staging_descriptor)
        os.close(destination_parent_descriptor)
        os.close(source_descriptor)
    return destination


def verify_release_bundle(path: str | Path) -> bool:
    root = Path(path).absolute()
    if not root.is_dir() or root.is_symlink():
        return False
    try:
        manifest = json.loads(_read_regular_bytes_at(root, Path("MANIFEST.json"), "manifest"))
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "files",
            "source_manifests",
            "core_endpoints",
            "report_inputs",
            "config_hash",
            "preregistration_hash",
            "retrieval_records",
            "top_level_sha256",
        }:
            return False
        files = manifest["files"]
        if manifest["schema_version"] != 4 or not isinstance(files, dict):
            return False
        _sha256_value(manifest["config_hash"], "config_hash")
        _sha256_value(manifest["preregistration_hash"], "preregistration_hash")
        source_manifests = manifest["source_manifests"]
        if not isinstance(source_manifests, dict):
            return False
        if not _valid_source_manifest_records(source_manifests, files):
            return False
        if not _valid_release_core_endpoint_records(
            manifest["core_endpoints"], files, source_manifests
        ):
            return False
        if not _valid_release_report_inputs(
            root,
            manifest["report_inputs"],
            manifest["core_endpoints"],
            files,
        ):
            return False
        _normalize_retrieval_records(manifest["retrieval_records"])
        identity = {
            "schema_version": 4,
            "files": files,
            "source_manifests": manifest["source_manifests"],
            "core_endpoints": manifest["core_endpoints"],
            "report_inputs": manifest["report_inputs"],
            "config_hash": manifest["config_hash"],
            "preregistration_hash": manifest["preregistration_hash"],
            "retrieval_records": manifest["retrieval_records"],
        }
        top_hash = hashlib.sha256(_canonical_json(identity)).hexdigest()
        if manifest["top_level_sha256"] != top_hash:
            return False
        checksum = _read_regular_bytes_at(root, Path("MANIFEST.sha256"), "manifest checksum").decode("ascii")
        if checksum != f"{top_hash}  MANIFEST.json\n":
            return False
        expected_names = {"MANIFEST.json", "MANIFEST.sha256"}
        for name, metadata in files.items():
            relative = _safe_relative(name)
            payload = _read_regular_bytes_at(root, relative, "release artifact")
            if not isinstance(metadata, dict) or set(metadata) != {"sha256", "bytes"}:
                return False
            if metadata["bytes"] != len(payload):
                return False
            if metadata["sha256"] != hashlib.sha256(payload).hexdigest():
                return False
            expected_names.add(relative.as_posix())
        all_paths = tuple(root.rglob("*"))
        if any(path.is_symlink() for path in all_paths):
            return False
        actual_names = {
            path.relative_to(root).as_posix() for path in all_paths if path.is_file()
        }
        expected_directories = {
            parent.as_posix()
            for name in expected_names
            for parent in Path(name).parents
            if parent != Path(".")
        }
        actual_directories = {
            path.relative_to(root).as_posix() for path in all_paths if path.is_dir()
        }
        return actual_names == expected_names and actual_directories == expected_directories
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
        return False


def build_registered_figures(
    behavior: F1Evidence,
    f2a: F2AEvidence | str | Path,
    output: str | Path | None = None,
    *,
    intervention: InterventionTestResult | None = None,
) -> tuple[Path, ...]:
    """Build registered figures directly from canonical typed evidence."""

    if not isinstance(behavior, F1Evidence) or not isinstance(f2a, F2AEvidence):
        raise ValueError("registered figures require canonical typed evidence")
    if intervention is not None and not isinstance(intervention, InterventionTestResult):
        raise ValueError("intervention must be a typed InterventionTestResult")
    if output is None:
        raise ValueError("registered figures require an output directory")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional reporting dependency
        raise ImportError("matplotlib is required to build registered figures") from error

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    conditions = tuple(
        "/".join(cell) for cell in sorted(behavior.metrics.cell_rates)
    )
    attempt_rates = tuple(
        behavior.metrics.cell_rates[cell]
        for cell in sorted(behavior.metrics.cell_rates)
    )
    if not conditions or any(value is None for value in attempt_rates):
        raise ValueError("canonical behavior evidence lacks registered attempt rates")

    paths: list[Path] = []
    fig, axis = plt.subplots(figsize=(8, 4.8))
    palette = ("#257A6B", "#C34B3F", "#4D6A89", "#D29A32")
    axis.bar(
        range(len(conditions)),
        attempt_rates,
        color=[palette[index % len(palette)] for index in range(len(conditions))],
    )
    axis.set_xticks(range(len(conditions)), conditions, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Answer-attempt rate")
    axis.set_title("Familiarity and answerability behavior")
    paths.append(_save_figure(fig, destination / "figure_1_behavior.png", plt))

    probe_results = (f2a.familiarity, f2a.answerability, f2a.unsupported_answer)
    task_names = tuple(result.task for result in probe_results)
    overall_auroc = tuple(
        float("nan") if result.metrics.auroc is None else result.metrics.auroc
        for result in probe_results
    )
    worst_auroc = tuple(
        float("nan")
        if result.worst_condition is None or result.worst_condition.auroc is None
        else result.worst_condition.auroc
        for result in probe_results
    )
    fig, axis = plt.subplots(figsize=(8, 4.8))
    positions = list(range(len(task_names)))
    axis.bar(
        [value - 0.18 for value in positions],
        overall_auroc,
        width=0.36,
        label="Overall",
        color="#257A6B",
    )
    axis.bar(
        [value + 0.18 for value in positions],
        worst_auroc,
        width=0.36,
        label="Worst condition",
        color="#C34B3F",
    )
    axis.axhline(0.5, color="#666666", linestyle=":", linewidth=1)
    axis.set_ylim(0.45, 1.0)
    axis.set_xticks(positions, task_names, rotation=15, ha="right")
    axis.set_ylabel("Held-out AUROC")
    axis.legend()
    axis.set_title("Registered held-out probe performance")
    paths.append(_save_figure(fig, destination / "figure_2_layer_probes.png", plt))

    baselines = tuple(
        (name, metrics.log_loss)
        for name, metrics in sorted(f2a.unsupported_answer.model_metrics.items())
        if metrics.log_loss is not None
    )
    if not baselines:
        raise ValueError("canonical F2A evidence lacks held-out baseline log losses")
    names = tuple(name for name, _ in baselines)
    losses = tuple(float(loss) for _, loss in baselines)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(range(len(names)), losses, color="#4D6A89")
    axis.set_xticks(range(len(names)), names, rotation=20, ha="right")
    axis.set_ylabel("Held-out log loss")
    axis.set_title("Nested baseline comparison")
    paths.append(_save_figure(fig, destination / "figure_3_baselines.png", plt))

    fig, axis = plt.subplots(figsize=(8, 4.8))
    if intervention is None:
        axis.set_title("Registered causal intervention: not run")
        axis.text(
            0.5,
            0.5,
            "F2B not run: no typed intervention result was supplied",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        axis.set_axis_off()
    else:
        claims = recompute_claim_ladder(
            behavior=behavior,
            f2a=f2a,
            f2b=intervention,
            circuit=None,
        )
        h7 = claims["H7"]
        if h7.status in {"supported", "not_supported"}:
            metrics = intervention.metrics
            control_names = tuple(sorted(metrics.control_effects))
            names = ("primary", *control_names)
            high_effects = (
                metrics.high_to_low_effect,
                *(metrics.control_effects[name][0] for name in control_names),
            )
            low_effects = (
                metrics.low_to_high_effect,
                *(metrics.control_effects[name][1] for name in control_names),
            )
            positions = list(range(len(names)))
            axis.bar(
                [value - 0.18 for value in positions],
                high_effects,
                width=0.36,
                label="High to low",
                color="#257A6B",
            )
            axis.bar(
                [value + 0.18 for value in positions],
                low_effects,
                width=0.36,
                label="Low to high",
                color="#C34B3F",
            )
            axis.axhline(0.0, color="#666666", linestyle=":", linewidth=1)
            axis.set_xticks(positions, names, rotation=25, ha="right")
            axis.set_ylabel("Oriented answer-attempt effect")
            axis.set_title("Registered causal intervention and controls")
            axis.legend()
        elif h7.status == "skipped_by_gate":
            axis.set_title("Registered causal intervention: not gated")
            axis.text(
                0.5,
                0.5,
                "F2B not gated: F1/H3/H4 prerequisite gates did not pass",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            axis.set_axis_off()
        else:
            axis.set_title("Registered causal intervention: not evaluable")
            axis.text(
                0.5,
                0.5,
                "F2B negative/not evaluable: " + "; ".join(h7.reasons),
                transform=axis.transAxes,
                ha="center",
                va="center",
                wrap=True,
            )
            axis.set_axis_off()
    paths.append(
        _save_figure(fig, destination / "figure_4_causal_intervention.png", plt)
    )

    observed = f2a.unsupported_answer.relative_h6_log_loss_improvement
    nulls = tuple(
        float(null.test_relative_h6_log_loss_improvement)
        for null in f2a.unsupported_answer.null_results
        if null.kind in {"layer_order", "random_map"}
        and null.test_relative_h6_log_loss_improvement is not None
    )
    fig, axis = plt.subplots(figsize=(8, 4.8))
    if nulls:
        axis.hist(
            nulls,
            bins=min(20, max(3, len(nulls))),
            color="#8A8F98",
            edgecolor="white",
        )
    else:
        axis.text(
            0.5,
            0.5,
            "Registered dynamics nulls not evaluated",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    if observed is not None:
        axis.axvline(observed, color="#C34B3F", linewidth=2, label="Observed")
    axis.set_xlabel("Incremental improvement")
    axis.set_ylabel("Null count")
    if observed is not None:
        axis.legend()
    axis.set_title("Appendix: H6 dynamics null and ablation distribution")
    paths.append(_save_figure(fig, destination / "appendix_h6_nulls.png", plt))
    return tuple(paths)


def _behavior_h2(evidence: F1Evidence | None) -> ClaimDecision:
    if evidence is None:
        return _not_evaluable("Target-bound non-inferiority was not evaluated")
    provenance_reasons = _f1_bootstrap_provenance_reasons(evidence.bootstrap)
    if provenance_reasons:
        return ClaimDecision(
            "not_evaluable",
            "Target-bound non-inferiority lacks confirmatory bootstrap provenance",
            provenance_reasons,
        )
    if evidence.metrics.status != "evaluable" or evidence.bootstrap.h2_accuracy_difference_interval is None:
        return _not_evaluable("Target-bound non-inferiority was not evaluable")
    lower = evidence.bootstrap.h2_accuracy_difference_interval.lower
    passed = lower > -0.05
    return ClaimDecision(
        "supported" if passed else "not_supported",
        "Target-bound matched-synthetic accuracy is locally non-inferior to screened-real accuracy"
        if passed
        else "Registered target-bound non-inferiority was not established",
        () if passed else ("paired lower bound did not exceed -0.05",),
    )


def _behavior_h1(
    evidence: F1Evidence | None, h2: ClaimDecision
) -> ClaimDecision:
    if evidence is not None:
        provenance_reasons = _f1_bootstrap_provenance_reasons(evidence.bootstrap)
        if provenance_reasons:
            return ClaimDecision(
                "not_evaluable",
                "The behavioral endpoint lacks confirmatory bootstrap provenance",
                provenance_reasons,
            )
    if evidence is None or evidence.metrics.status != "evaluable" or evidence.bootstrap.interaction_interval is None:
        return _not_evaluable("The behavioral endpoint was not evaluated")
    interaction = evidence.metrics.interaction
    if interaction is None:
        return _not_evaluable("The behavioral interaction was not evaluable")
    interval = evidence.bootstrap.interaction_interval
    validity_values = tuple(evidence.metrics.format_validity_by_cell.values())
    reasons = []
    if interaction < 0.05:
        reasons.append("interaction below 0.05")
    if interval.lower <= 0:
        reasons.append("interaction interval includes zero")
    if not validity_values or min(validity_values) < 0.95:
        reasons.append("a cell is below 95% format validity")
    if h2.status != "supported":
        reasons.append("H2 non-inferiority did not pass")
    return ClaimDecision(
        "supported" if not reasons else "not_supported",
        "Target familiarity selectively increased answer attempts under absent evidence in this registered task"
        if not reasons
        else "The registered behavioral familiarity-by-answerability interaction was not established",
        tuple(reasons),
    )


def _behavior_h2b(evidence: F1Evidence | None) -> ClaimDecision:
    if evidence is None:
        return _not_evaluable("The same-string exposure block was not evaluated")
    provenance_reasons = _f1_bootstrap_provenance_reasons(evidence.bootstrap)
    if provenance_reasons:
        return ClaimDecision(
            "not_evaluable",
            "The same-string exposure block lacks confirmatory bootstrap provenance",
            provenance_reasons,
        )
    seal = evidence.gate.same_string_seal
    if not isinstance(seal, SameStringSealEvidence):
        return ClaimDecision(
            "not_evaluable",
            "The same-string exposure block lacks immutable seal evidence",
            ("typed same-string seal evidence is missing",),
        )
    seal_record = evidence.gate.to_record()
    if (
        seal.source_manifest_sha256 != evidence.gate.manifest_hash
        or seal.endpoint != "behavior_test"
        or seal.block != "same_string"
        or seal_record.get("same_string_seal_sha256") != seal.sha256
    ):
        return ClaimDecision(
            "not_evaluable",
            "The same-string exposure block seal is not bound to F1",
            ("typed same-string seal hash or registered block record is invalid",),
        )
    if (
        evidence.metrics.status != "evaluable"
        or evidence.metrics.h2b_interaction is None
        or evidence.bootstrap.h2b_interaction_interval is None
    ):
        return _not_evaluable("The same-string exposure block was not evaluable")
    interval = evidence.bootstrap.h2b_interaction_interval
    reasons = []
    if evidence.metrics.h2b_interaction < 0.05:
        reasons.append("same-string interaction below 0.05")
    if interval.lower <= 0:
        reasons.append("same-string predicted-direction interval includes zero")
    return ClaimDecision(
        "supported" if not reasons else "not_supported",
        "The sealed same-string exposure block showed the registered directional interaction"
        if not reasons
        else "The registered same-string exposure interaction was not established",
        tuple(reasons),
    )


def _f1_bootstrap_provenance_reasons(
    bootstrap: BootstrapDistribution,
) -> tuple[str, ...]:
    reasons = []
    if bootstrap.requested_draws != CONFIRMATORY_BOOTSTRAP_REPLICATES:
        reasons.append("bootstrap requested draws must equal 10000")
    if (
        type(bootstrap.valid_draws) is not int
        or bootstrap.valid_draws < _F1_MINIMUM_VALID_DRAWS
    ):
        reasons.append("bootstrap has insufficient valid draws")
    if type(bootstrap.discarded_draws) is not int or bootstrap.discarded_draws < 0:
        reasons.append("bootstrap discarded draws are invalid")
    if (
        type(bootstrap.requested_draws) is int
        and type(bootstrap.valid_draws) is int
        and type(bootstrap.discarded_draws) is int
        and bootstrap.valid_draws + bootstrap.discarded_draws
        != bootstrap.requested_draws
    ):
        reasons.append("bootstrap draw accounting is inconsistent")
    if bootstrap.resampling_unit != _F1_RESAMPLING_UNIT:
        reasons.append("bootstrap resampling unit does not match the registration")
    if bootstrap.seed != CONFIRMATORY_BOOTSTRAP_SEED:
        reasons.append("bootstrap seed does not match the registration")
    if not math.isclose(bootstrap.alpha, _F1_ALPHA):
        reasons.append("bootstrap alpha does not match the registration")
    return tuple(reasons)


def _probe_claim(
    evidence: F2AEvidence | None, hypothesis: str, concept: str
) -> ClaimDecision:
    if evidence is None:
        return _not_evaluable(f"The {concept} probe endpoint was not evaluated")
    provenance_reasons = _f2a_null_provenance_reasons(evidence)
    if provenance_reasons:
        return ClaimDecision(
            "not_evaluable",
            f"The {concept} probe lacks complete confirmatory null provenance",
            provenance_reasons,
        )
    gate = evidence.gates.h3 if hypothesis == "H3" else evidence.gates.h4
    status, reasons = _typed_gate_status(gate)
    return ClaimDecision(
        status,
        f"{concept.capitalize()} was condition-invariantly decodable on registered holdouts"
        if status == "supported"
        else f"Registered condition-invariant {concept} decodability was not established",
        reasons,
    )


def _h5_claim(evidence: F2AEvidence | None) -> ClaimDecision:
    if evidence is None:
        return _not_evaluable("Incremental pre-output prediction was not evaluated")
    provenance_reasons = _f2a_null_provenance_reasons(evidence)
    if provenance_reasons:
        return ClaimDecision(
            "not_evaluable",
            "Incremental pre-output prediction lacks complete confirmatory null provenance",
            provenance_reasons,
        )
    scope_reasons = _h5_selection_scope_reasons(evidence.unsupported_answer)
    if scope_reasons:
        return ClaimDecision(
            "not_evaluable",
            "Incremental pre-output prediction lacks registered selection-scope evidence",
            scope_reasons,
        )
    status, reasons = _typed_gate_status(evidence.gates.h5)
    return ClaimDecision(
        status,
        "Frozen internal static features improved pre-output unsupported-answer prediction beyond nested controls"
        if status == "supported"
        else "Incremental pre-output prediction beyond nested controls was not established",
        reasons,
    )


def _h6_claim(evidence: F2AEvidence | None) -> ClaimDecision:
    if evidence is None:
        return _not_evaluable("Cross-layer dynamics were not evaluated")
    provenance_reasons = _f2a_null_provenance_reasons(evidence)
    if provenance_reasons:
        return ClaimDecision(
            "not_evaluable",
            "Cross-layer dynamics lack complete confirmatory null provenance",
            provenance_reasons,
        )
    status, reasons = _typed_gate_status(evidence.gates.h6)
    return ClaimDecision(
        status,
        "Cross-layer dynamics added held-out predictive information beyond static internal features"
        if status == "supported"
        else "Incremental predictive value from cross-layer dynamics was not established",
        reasons,
    )


def _f2a_null_provenance_reasons(evidence: F2AEvidence) -> tuple[str, ...]:
    required = (
        ("familiarity", "label_permutation", evidence.familiarity),
        ("answerability", "label_permutation", evidence.answerability),
        ("unsupported_answer", "layer_order", evidence.unsupported_answer),
        ("unsupported_answer", "random_map", evidence.unsupported_answer),
    )
    reasons = []
    for task, kind, result in required:
        nulls = tuple(null for null in result.null_results if null.kind == kind)
        prefix = f"{task}/{kind} null"
        if any(not isinstance(null, NullSelectionResult) for null in nulls):
            reasons.append(f"{prefix} evidence is not typed")
            continue
        seeds = tuple(null.seed for null in nulls)
        if seeds != DEFAULT_FULL_SELECTION_NULL_SEEDS:
            reasons.append(f"{prefix} seeds do not match the exact registered 99-seed list")
        for null in nulls:
            config = null.config
            provenance = getattr(null.selection, "null_provenance", None)
            if (
                not isinstance(config, Mapping)
                or config.get("seed_list_sha256")
                != DEFAULT_FULL_SELECTION_NULL_SEED_HASH
                or config.get("registered_seed_count")
                != len(DEFAULT_FULL_SELECTION_NULL_SEEDS)
            ):
                reasons.append(f"{prefix} config lacks the canonical seed-list hash")
                break
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("seed_list_sha256")
                != DEFAULT_FULL_SELECTION_NULL_SEED_HASH
                or provenance.get("registered_seed_count")
                != len(DEFAULT_FULL_SELECTION_NULL_SEEDS)
                or provenance.get("kind") != kind
                or provenance.get("seed") != null.seed
            ):
                reasons.append(f"{prefix} selection provenance is not canonically bound")
                break
    return tuple(reasons)


def _h5_selection_scope_reasons(result: ProbeResult) -> tuple[str, ...]:
    fields = ("selected_anchor", "selected_layer", "selected_claim_scope")
    if not any(hasattr(result, field) for field in fields):
        return ()
    if not all(hasattr(result, field) for field in fields):
        return ("selection-scope evidence is incomplete",)
    anchor = getattr(result, "selected_anchor")
    layer = getattr(result, "selected_layer")
    claim_scope = getattr(result, "selected_claim_scope")
    reasons = []
    if claim_scope != "pre_output":
        reasons.append("selected claim scope is not pre_output")
    if anchor == "assistant_prefix_end" or not isinstance(anchor, str) or not anchor:
        reasons.append("selected anchor is not a registered pre-output anchor")
    if type(layer) is not int or layer < 0:
        reasons.append("selected layer is invalid")
    return tuple(reasons)


def _canonical_intervention_gates(
    metrics: InterventionMetrics,
) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
    """Recompute H7/H8 from complete registered intervention evidence."""

    if not isinstance(metrics, InterventionMetrics):
        raise ValueError("intervention claims require canonical InterventionMetrics")
    completeness_reasons = _intervention_evidence_completeness_reasons(metrics)
    if completeness_reasons:
        reasons = tuple(completeness_reasons)
        return "not_evaluable", reasons, "not_evaluable", reasons

    h7_reasons: list[str] = []
    if metrics.completed_fraction < 0.95:
        h7_reasons.append("completed fraction is below 0.95")
    if metrics.high_to_low_interval[0] <= 0:
        h7_reasons.append("high-to-low Holm interval includes zero")
    if metrics.low_to_high_interval[0] <= 0:
        h7_reasons.append("low-to-high Holm interval includes zero")
    if metrics.average_effect < 0.05:
        h7_reasons.append("average bidirectional effect is below 0.05")
    for name, effects in sorted(metrics.control_effects.items()):
        if metrics.high_to_low_effect - effects[0] < 0.02:
            h7_reasons.append(f"high-to-low effect does not beat {name} by 0.02")
        if metrics.low_to_high_effect - effects[1] < 0.02:
            h7_reasons.append(f"low-to-high effect does not beat {name} by 0.02")
    if any(domain not in _REGISTERED_DOMAINS for domain in metrics.passing_domains):
        h7_reasons.append("passing domains contain an unregistered domain")
    if set(metrics.observed_domains) != _REGISTERED_DOMAINS:
        h7_reasons.append("not all four registered domains were observed")
    if len(metrics.passing_domains) < 3:
        h7_reasons.append("effects did not reproduce in at least three registered domains")
    constraints = metrics.readout_constraints
    if metrics.familiarity_readout_effect < constraints.familiarity_min_effect:
        h7_reasons.append("familiarity readout did not move by the frozen minimum")
    if metrics.answerability_max_abs_change > constraints.answerability_max_abs_change:
        h7_reasons.append("answerability readout exceeded its frozen tolerance")
    if metrics.entity_type_max_abs_change > constraints.entity_type_max_abs_change:
        h7_reasons.append("entity-type readout exceeded its frozen tolerance")
    if (
        metrics.generic_confidence_max_abs_change
        > constraints.generic_confidence_max_abs_change
    ):
        h7_reasons.append("generic-confidence readout exceeded its frozen tolerance")

    directions = _plain_tree(metrics.bootstrap_summary)["directions"]
    for direction in sorted(_H7_DIRECTIONS):
        values = directions[direction]
        if values["holm_adjusted_p"] > _H7_ALPHA:
            h7_reasons.append(f"{direction} Holm-adjusted p-value exceeds 0.05")
        if values["holm_interval"][0] <= 0:
            h7_reasons.append(f"{direction} Holm interval includes zero")

    h8_reasons: list[str] = []
    if metrics.completed_fraction < 0.95:
        h8_reasons.append("completed fraction is below 0.95")
    if metrics.target_bound_accuracy_change < -0.05:
        h8_reasons.append("target-bound accuracy loss exceeds 0.05")
    for direction, change in metrics.unrelated_refusal_change_by_direction.items():
        if abs(change) > 0.03:
            h8_reasons.append(
                f"{direction} unrelated refusal change exceeds 0.03"
            )
    for direction, change in metrics.unrelated_invalid_format_change_by_direction.items():
        if abs(change) > 0.03:
            h8_reasons.append(
                f"{direction} unrelated invalid-format change exceeds 0.03"
            )
    return (
        "supported" if not h7_reasons else "not_supported",
        tuple(h7_reasons),
        "supported" if not h8_reasons else "not_supported",
        tuple(h8_reasons),
    )


def _intervention_evidence_completeness_reasons(
    metrics: InterventionMetrics,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if set(metrics.control_effects) != set(REQUIRED_CAUSAL_CONTROLS):
        reasons.append("complete registered causal-control evidence is missing")
    for name, values in (
        (
            "unrelated-refusal directional evidence",
            metrics.unrelated_refusal_change_by_direction,
        ),
        (
            "unrelated-invalid-format directional evidence",
            metrics.unrelated_invalid_format_change_by_direction,
        ),
    ):
        if set(values) != _H7_DIRECTIONS:
            reasons.append(f"complete {name} is missing")
    bootstrap = _plain_tree(metrics.bootstrap_summary)
    if not isinstance(bootstrap, dict):
        return (*reasons, "complete registered crossed-bootstrap evidence is missing")

    missing_reasons = {
        "method": "crossed-bootstrap method is missing",
        "seed": "crossed-bootstrap seed is missing",
        "replicates": "crossed-bootstrap replicate count is missing",
        "requested_draws": "crossed-bootstrap requested draws are missing",
        "valid_draws": "crossed-bootstrap valid draws are missing",
        "discarded_draws": "crossed-bootstrap discarded draws are missing",
        "resampling_unit": "crossed-bootstrap resampling unit is missing",
        "alpha": "crossed-bootstrap alpha is missing",
        "directions": "crossed-bootstrap directions are missing",
    }
    for field in sorted(_H7_BOOTSTRAP_ROOT_FIELDS - set(bootstrap)):
        reasons.append(missing_reasons[field])
    unexpected = set(bootstrap) - _H7_BOOTSTRAP_ROOT_FIELDS
    if unexpected:
        reasons.append(
            "crossed-bootstrap evidence has unexpected fields: "
            + ", ".join(sorted(unexpected))
        )

    if bootstrap.get("method") != _H7_BOOTSTRAP_METHOD:
        reasons.append("crossed-bootstrap method does not match the registration")
    if bootstrap.get("seed") != _H7_BOOTSTRAP_SEED:
        reasons.append("crossed-bootstrap seed does not match the registration")
    if bootstrap.get("replicates") != _H7_BOOTSTRAP_REPLICATES:
        reasons.append("crossed-bootstrap replicate count does not match the registration")
    if bootstrap.get("requested_draws") != _H7_BOOTSTRAP_REPLICATES:
        reasons.append("crossed-bootstrap requested draws must equal 10000")

    valid_draws = bootstrap.get("valid_draws")
    discarded_draws = bootstrap.get("discarded_draws")
    requested_draws = bootstrap.get("requested_draws")
    if type(valid_draws) is not int or valid_draws <= 0:
        reasons.append("crossed-bootstrap valid draws must be a positive integer")
    if type(discarded_draws) is not int or discarded_draws < 0:
        reasons.append("crossed-bootstrap discarded draws must be a nonnegative integer")
    if (
        type(requested_draws) is int
        and type(valid_draws) is int
        and type(discarded_draws) is int
        and valid_draws + discarded_draws != requested_draws
    ):
        reasons.append("crossed-bootstrap draw accounting is inconsistent")
    if bootstrap.get("resampling_unit") != list(_H7_RESAMPLING_UNIT):
        reasons.append("crossed-bootstrap resampling unit does not match the registration")

    alpha = bootstrap.get("alpha")
    if (
        type(alpha) not in {int, float}
        or not math.isfinite(alpha)
        or not math.isclose(float(alpha), _H7_ALPHA)
    ):
        reasons.append("crossed-bootstrap alpha does not match the registration")
    directions = bootstrap.get("directions")
    if not isinstance(directions, dict) or set(directions) != _H7_DIRECTIONS:
        reasons.append("crossed-bootstrap directions are incomplete")
        return tuple(reasons)

    expected_fields = {
        "point_estimate",
        "raw_interval",
        "raw_p",
        "entities",
        "template_families",
        "holm_interval",
        "holm_adjusted_p",
    }
    raw_p: dict[str, float] = {}
    for direction in sorted(_H7_DIRECTIONS):
        values = directions[direction]
        if not isinstance(values, dict):
            reasons.append(f"{direction} crossed-bootstrap record is not a mapping")
            continue
        missing_fields = expected_fields - set(values)
        for field in sorted(missing_fields):
            reasons.append(f"{direction} crossed-bootstrap {field} is missing")
        unexpected_fields = set(values) - expected_fields
        if unexpected_fields:
            reasons.append(
                f"{direction} crossed-bootstrap has unexpected fields: "
                + ", ".join(sorted(unexpected_fields))
            )
        if missing_fields or unexpected_fields:
            continue
        try:
            point = _finite_number(values["point_estimate"], f"{direction} point")
            raw_interval = _validated_interval(values["raw_interval"], f"{direction} raw interval")
            holm_interval = _validated_interval(values["holm_interval"], f"{direction} Holm interval")
            raw_p[direction] = _probability(values["raw_p"], f"{direction} raw p")
            _probability(
                values["holm_adjusted_p"], f"{direction} Holm-adjusted p"
            )
            if not _nonempty_string_values(values["entities"]):
                raise ValueError(f"{direction} bootstrap entities are missing")
            if not _nonempty_string_values(values["template_families"]):
                raise ValueError(f"{direction} bootstrap template families are missing")
        except ValueError as error:
            reasons.append(str(error))
            continue
        expected_point = (
            metrics.high_to_low_effect
            if direction == "high_to_low"
            else metrics.low_to_high_effect
        )
        expected_interval = (
            metrics.high_to_low_interval
            if direction == "high_to_low"
            else metrics.low_to_high_interval
        )
        if not math.isclose(point, expected_point, rel_tol=1e-12, abs_tol=1e-12):
            reasons.append(f"{direction} point estimate does not match InterventionMetrics")
        if not _interval_close(holm_interval, expected_interval):
            reasons.append(f"{direction} Holm interval does not match InterventionMetrics")
        if raw_interval[0] > raw_interval[1]:  # defensive; validated above
            reasons.append(f"{direction} raw interval is reversed")

    if set(raw_p) == _H7_DIRECTIONS:
        expected_adjusted = _holm_adjusted(raw_p)
        for direction, expected in expected_adjusted.items():
            observed = float(directions[direction]["holm_adjusted_p"])
            if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                reasons.append(f"{direction} Holm-adjusted p-value is inconsistent")
    return tuple(reasons)


def _intervention_claims(result, h1, h3, h4):
    if result is None:
        skipped = ClaimDecision("skipped", "The gated causal study was not run", ())
        return skipped, skipped
    if any(claim.status != "supported" for claim in (h1, h3, h4)):
        skipped = ClaimDecision(
            "skipped_by_gate",
            "The causal endpoint is not interpretable because prerequisite gates did not pass",
            ("F1/H3/H4 gate failure",),
        )
        return skipped, skipped
    h7_status, h7_reasons, h8_status, h8_reasons = _canonical_intervention_gates(
        result.metrics
    )
    h7 = ClaimDecision(
        h7_status,
        "The frozen prefill intervention had local bidirectional causal relevance in this task"
        if h7_status == "supported"
        else "Local bidirectional causal relevance was not established",
        h7_reasons,
    )
    h8 = ClaimDecision(
        h8_status,
        "The selected local intervention passed registered capability-preservation bounds"
        if h8_status == "supported"
        else "Registered capability-preservation bounds were not met",
        h8_reasons,
    )
    return h7, h8


def _circuit_claim(evidence, h3, h4, h5, h7):
    if evidence is None:
        return ClaimDecision("skipped", "The optional circuit follow-up was not run", ())
    reasons = []
    if h3.status != "supported" or h4.status != "supported":
        reasons.append("H3/H4 mechanistic decoding prerequisites did not pass")
    if h5.status != "supported":
        reasons.append("H5 incremental pre-output prerequisite did not pass")
    for field, threshold in (
        ("proxy_spearman", 0.80),
        ("distribution_spearman", 0.80),
        ("perturbation_spearman", 0.60),
        ("sign_concordance", 0.75),
    ):
        if getattr(evidence, field) < threshold:
            reasons.append(f"{field} below registered threshold")
    if evidence.error_node_share > 0.50:
        reasons.append("error-node share exceeded 0.50")
    if evidence.attempted == 0:
        reasons.append("no graph attempts were executed")
    elif evidence.successful == 0:
        reasons.append("graph yield was zero")
    if not evidence.original_model_intervention_supported:
        reasons.append("original-model intervention support is absent")
    if h7.status != "supported":
        reasons.append("original-model intervention did not support the graph hypothesis")
    return ClaimDecision(
        "supported_prompt_local_hypothesis" if not reasons else "not_supported",
        "The fidelity-audited graph is a prompt-local replacement-model hypothesis consistent with an original-model intervention"
        if not reasons
        else "The optional attribution graph did not meet the prompt-local fidelity claim gate",
        tuple(reasons),
    )


def _not_evaluable(claim: str) -> ClaimDecision:
    return ClaimDecision("not_evaluable", claim, ("required canonical metrics are missing",))


def _typed_gate_status(gate: Any) -> tuple[str, tuple[str, ...]]:
    """Re-evaluate typed gate criteria rather than trusting its status property."""

    values = tuple(criterion.satisfied for criterion in gate.criteria)
    if not values or any(value is None for value in values):
        return "not_evaluable", tuple(gate.reasons)
    return ("supported" if all(values) else "not_supported"), tuple(gate.reasons)


def _incomplete_behavior_rows(metrics: BehavioralMetrics) -> int:
    return sum(
        denominator - round(metrics.completion_by_cell.get(cell, 0.0) * denominator)
        for cell, denominator in metrics.denominators.items()
    )


def _f1_phase_status(evidence: F1Evidence | None) -> str:
    if evidence is None or evidence.metrics.status != "evaluable":
        return "not_evaluable"
    return "evaluated"


def _f2a_phase_status(evidence: F2AEvidence | None) -> str:
    if evidence is None:
        return "not_evaluable"
    statuses = tuple(result.metrics.status for result in (
        evidence.familiarity,
        evidence.answerability,
        evidence.unsupported_answer,
    ))
    return "evaluated" if all(status == "evaluable" for status in statuses) else "not_evaluable"


def _f2b_phase_status(result: InterventionTestResult | None) -> str:
    if result is None:
        return "skipped"
    return "evaluated" if result.metrics.completed_fraction >= 0.95 else "not_evaluable"


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    if name not in metrics:
        raise ValueError(f"required metric is missing: {name}")
    return _finite_number(metrics[name], name)


def _interval(metrics: Mapping[str, Any], name: str) -> tuple[float, float]:
    value = metrics.get(name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value interval")
    interval = (_finite_number(value[0], name), _finite_number(value[1], name))
    if interval[0] > interval[1]:
        raise ValueError(f"{name} bounds are reversed")
    return interval


def _validated_interval(value: Any, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must be a two-value interval")
    interval = (_finite_number(value[0], name), _finite_number(value[1], name))
    if interval[0] > interval[1]:
        raise ValueError(f"{name} bounds are reversed")
    return interval


def _probability(value: Any, name: str) -> float:
    probability = _finite_number(value, name)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return probability


def _nonempty_string_values(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and bool(value)
        and all(isinstance(item, str) and item for item in value)
    )


def _interval_close(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) == 2 and all(
        math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
        for a, b in zip(left, right)
    )


def _holm_adjusted(raw_p: Mapping[str, float]) -> Mapping[str, float]:
    ordered = sorted(raw_p, key=lambda name: (raw_p[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(
            running,
            min(1.0, float(raw_p[name]) * (len(ordered) - index)),
        )
        adjusted[name] = running
    return adjusted


def _plain_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain_tree(child) for child in value]
    if isinstance(value, list):
        return [_plain_tree(child) for child in value]
    return value


def _format_probe_metrics(metrics: Any) -> str:
    return (
        f"status={metrics.status}, total={metrics.total}, "
        f"denominator={metrics.denominator}, missing={metrics.missing}, "
        f"invalid={metrics.invalid}, auroc={metrics.auroc}, "
        f"balanced_accuracy={metrics.balanced_accuracy}, log_loss={metrics.log_loss}"
    )


def _require_finite_tree(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} keys must be strings")
            if key == "stored_supported":
                continue
            _require_finite_tree(child, f"{name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _require_finite_tree(child, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must contain finite numbers")


def _finite_number(value: Any, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _number_sequence(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{name} must be a nonempty sequence")
    return tuple(_finite_number(item, name) for item in value)


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{name} must be a nonempty string sequence")
    return tuple(value)


def _same_length(left: Sequence[Any], right: Sequence[Any], name: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"{name} inputs must have equal length")


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("release path must be a nonempty relative string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError("release paths must be canonical relative paths")
    if relative.parts[0] in {"MANIFEST.json", "MANIFEST.sha256"}:
        raise ValueError("release allowlist cannot replace release manifests")
    return relative


def _sha256_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _normalize_retrieval_records(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("retrieval_records must be a sequence")
    records: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"uri", "sha256"}:
            raise ValueError("retrieval records require exactly uri and sha256")
        uri = item["uri"]
        if not isinstance(uri, str) or not uri.startswith("https://"):
            raise ValueError("retrieval record uri must be immutable HTTPS evidence")
        records.append({"uri": uri, "sha256": _sha256_value(item["sha256"], "retrieval sha256")})
    if len({(item["uri"], item["sha256"]) for item in records}) != len(records):
        raise ValueError("retrieval records must be unique")
    return sorted(records, key=lambda item: (item["uri"], item["sha256"]))


def _protected_shard_is_closed(store: FAArtifactStore, shard: Any) -> bool:
    """Bind a protected input or metric shard to a verified closed endpoint."""

    try:
        relative = shard.data_path.relative_to(
            store.root / "runs" / "familiarity_answerability"
        )
        run_id = relative.parts[0]
        endpoint = shard.namespace
        closed_path = store.root / "runs" / "familiarity_answerability" / run_id / "endpoints" / endpoint / "closed.json"
        evaluated_path = closed_path.with_name("evaluated.json")
        closed, _ = store._read_endpoint_record(closed_path, endpoint, "closed")
        evaluated, evaluated_bytes = store._read_endpoint_record(evaluated_path, endpoint, "evaluated")
        if closed["evaluated_sha256"] != hashlib.sha256(evaluated_bytes).hexdigest():
            return False
        metrics_manifest = store._path_from_root_record(
            evaluated["metrics_manifest_path"], "evaluated metrics manifest"
        )
        metrics = store.verify_shard(metrics_manifest)
        if metrics.sha256 != evaluated["metrics_sha256"]:
            return False
        if shard.manifest_path == metrics.manifest_path:
            return shard.record_kind == "metrics" and shard.sha256 == metrics.sha256
        return store.verify_endpoint_artifact(endpoint, shard.manifest_path) == shard
    except (AttributeError, IndexError, ValueError, OSError):
        return False


def _verify_core_endpoint_evidence(
    store: FAArtifactStore,
    endpoint_manifests: Mapping[str, str | Path],
    allowlist: Mapping[str, str],
    preregistration_hash: str,
) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(endpoint_manifests, Mapping) or set(endpoint_manifests) != {
        "behavior_test",
        "probe_test",
    }:
        raise ValueError(
            "release requires explicit behavior_test and probe_test endpoint manifests"
        )
    records: dict[str, Mapping[str, Any]] = {}
    for endpoint in ("behavior_test", "probe_test"):
        supplied = Path(endpoint_manifests[endpoint])
        manifest_path = supplied if supplied.is_absolute() else store.root / supplied
        input_shard = store.verify_endpoint_artifact(endpoint, manifest_path)
        if store.endpoint_state(endpoint, manifest_path) != "closed":
            raise ValueError(f"core endpoint {endpoint} is not closed")
        sealed_path = store._find_endpoint_path(endpoint, "sealed")
        sealed, _ = store._read_endpoint_record(sealed_path, endpoint, "sealed")
        if sealed["parents"]["preregistration"] != preregistration_hash:
            raise ValueError("core endpoint preregistration hash does not match release")
        evaluated_path = store._find_endpoint_path(endpoint, "evaluated")
        evaluated, evaluated_bytes = store._read_endpoint_record(
            evaluated_path, endpoint, "evaluated"
        )
        closed_path = store._find_endpoint_path(endpoint, "closed")
        closed, closed_bytes = store._read_endpoint_record(
            closed_path, endpoint, "closed"
        )
        if closed["evaluated_sha256"] != hashlib.sha256(evaluated_bytes).hexdigest():
            raise ValueError("core endpoint closed state does not bind evaluated state")
        metrics_manifest = store._path_from_root_record(
            evaluated["metrics_manifest_path"], "core metrics manifest"
        )
        metrics_shard = store.verify_shard(metrics_manifest)
        if (
            metrics_shard.namespace != endpoint
            or metrics_shard.record_kind != "metrics"
            or metrics_shard.sha256 != evaluated["metrics_sha256"]
        ):
            raise ValueError("core endpoint metrics do not verify")
        metrics_payload = store._read_regular_bytes(
            metrics_shard.data_path, "core endpoint metrics"
        )
        try:
            phase, evidence_sha256 = _closed_endpoint_phase_hash(
                endpoint, metrics_payload
            )
        except ValueError as error:
            raise ValueError(
                "report cannot bind to closed endpoint metrics evidence"
            ) from error
        required_files = (
            input_shard.data_path,
            input_shard.manifest_path,
            metrics_shard.data_path,
            metrics_shard.manifest_path,
        )
        for path in required_files:
            relative = path.relative_to(store.root).as_posix()
            expected = allowlist.get(relative)
            if expected is None:
                raise ValueError(
                    f"release allowlist omits required closed {endpoint} evidence: {relative}"
                )
            payload = store._read_regular_bytes(path, "core release evidence")
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"release allowlist hash mismatch for {relative}")
        records[endpoint] = {
            "input_manifest": input_shard.manifest_path.relative_to(store.root).as_posix(),
            "input_sha256": input_shard.sha256,
            "metrics_manifest": metrics_shard.manifest_path.relative_to(store.root).as_posix(),
            "metrics_sha256": metrics_shard.sha256,
            "closed_state_sha256": hashlib.sha256(closed_bytes).hexdigest(),
            "phase": phase,
            "evidence_sha256": evidence_sha256,
        }
    return records


def _closed_endpoint_phase_hash(
    endpoint: str, payload: bytes
) -> tuple[str, str]:
    records = []
    for line in payload.splitlines():
        if not line:
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("closed endpoint metrics must be JSONL") from error
        if not isinstance(record, dict) or _canonical_json(record) != line:
            raise ValueError("closed endpoint metrics must be canonical JSONL")
        records.append(record)
    if len(records) != 1:
        raise ValueError("closed endpoint metrics require exactly one evidence record")
    record = records[0]
    if endpoint == "behavior_test":
        if record.get("kind") != "metrics" or any(
            name not in record for name in ("metrics", "bootstrap", "gate")
        ):
            raise ValueError("behavior metrics lack canonical F1 evidence")
        evidence = {
            "metrics": record["metrics"],
            "bootstrap": record["bootstrap"],
            "gate": record["gate"],
        }
        return "F1", hashlib.sha256(_canonical_json(evidence)).hexdigest()
    if endpoint == "probe_test":
        if record.get("kind") != "metrics":
            raise ValueError("probe metrics lack canonical F2A evidence")
        if record.get("metric_type") == "f2a_bundle":
            bundle = record.get("result")
            if not isinstance(bundle, Mapping):
                raise ValueError("F2A bundle result is missing")
            results = bundle.get("results")
            gates = bundle.get("gates")
            if not isinstance(results, Mapping) or not isinstance(gates, Mapping):
                raise ValueError("F2A bundle core records are missing")
            evidence = {
                "familiarity": results.get("familiarity"),
                "answerability": results.get("answerability"),
                "unsupported_answer": results.get("unsupported_answer"),
                "gates": gates,
            }
        else:
            evidence = {
                "familiarity": record.get("familiarity"),
                "answerability": record.get("answerability"),
                "unsupported_answer": record.get("unsupported_answer"),
                "gates": record.get("gates"),
            }
        if any(not isinstance(value, Mapping) for value in evidence.values()):
            raise ValueError("probe metrics lack complete canonical F2A evidence")
        return "F2A", hashlib.sha256(_canonical_json(evidence)).hexdigest()
    raise ValueError("endpoint has no registered report phase")


def _parse_report_input_metadata(payload: bytes) -> dict[str, Any] | None:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("report metadata must be UTF-8") from error
    prefix = "<!-- fa-report-inputs:"
    matches = [line for line in lines if line.startswith(prefix)]
    if not matches:
        return None
    if len(matches) != 1 or not matches[0].endswith(" -->"):
        raise ValueError("report must contain exactly one canonical input metadata record")
    encoded = matches[0][len(prefix) : -4]
    try:
        metadata = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("report input metadata is invalid JSON") from error
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema_version",
        "phase_evidence_sha256",
        "report_input_bundle_sha256",
    }:
        raise ValueError("report input metadata has an invalid schema")
    if metadata["schema_version"] != 1 or not isinstance(
        metadata["phase_evidence_sha256"], dict
    ):
        raise ValueError("report input metadata has an invalid schema")
    if encoded != _canonical_json(metadata).decode("ascii"):
        raise ValueError("report input metadata must be canonical JSON")
    phases = metadata["phase_evidence_sha256"]
    if not phases or set(phases) - {"F1", "F2A", "F2B", "F3"}:
        raise ValueError("report input phases are invalid")
    for phase, digest in phases.items():
        _sha256_value(digest, f"report {phase} evidence hash")
    bundle = {
        "schema_version": 1,
        "phase_evidence_sha256": phases,
    }
    expected = hashlib.sha256(_canonical_json(bundle)).hexdigest()
    if metadata["report_input_bundle_sha256"] != expected:
        raise ValueError("report input bundle hash does not match its phase evidence")
    return metadata


def _verify_report_inputs_at_descriptor(
    source_descriptor: int,
    allowlist: Mapping[str, str],
    core_endpoints: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    candidates = []
    for name, expected_sha256 in sorted(allowlist.items()):
        relative = _safe_relative(name)
        if relative.suffix.lower() != ".md":
            continue
        payload = _read_regular_bytes_at_descriptor(
            source_descriptor, relative, "release report"
        )
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError(f"release source hash mismatch: {name}")
        metadata = _parse_report_input_metadata(payload)
        if metadata is not None:
            candidates.append((relative.as_posix(), metadata))
    if len(candidates) != 1:
        raise ValueError("release requires exactly one hash-bound generated report")
    report_path, metadata = candidates[0]
    expected_phases = {
        record["phase"]: record["evidence_sha256"]
        for record in core_endpoints.values()
    }
    if metadata["phase_evidence_sha256"] != expected_phases:
        raise ValueError("report evidence does not match released closed endpoints")
    return {"report_path": report_path, **metadata}


def _valid_release_report_inputs(
    root: Path,
    value: Any,
    core_endpoints: Mapping[str, Any],
    files: Mapping[str, Any],
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "report_path",
        "schema_version",
        "phase_evidence_sha256",
        "report_input_bundle_sha256",
    }:
        return False
    report_path = value["report_path"]
    if not isinstance(report_path, str) or report_path not in files:
        return False
    try:
        payload = _read_regular_bytes_at(root, _safe_relative(report_path), "release report")
        metadata = _parse_report_input_metadata(payload)
    except (OSError, ValueError):
        return False
    if metadata is None or {"report_path": report_path, **metadata} != dict(value):
        return False
    expected_phases = {
        record["phase"]: record["evidence_sha256"]
        for record in core_endpoints.values()
    }
    return metadata["phase_evidence_sha256"] == expected_phases


def _valid_source_manifest_records(
    value: Mapping[str, Any], files: Mapping[str, Any]
) -> bool:
    data_names = {name for name in files if name.endswith(".jsonl")}
    expected_manifests = {f"{name}.manifest.json" for name in data_names}
    if set(value) != expected_manifests:
        return False
    for manifest_name, record in value.items():
        if not isinstance(record, Mapping) or set(record) != {"sha256", "shard_sha256"}:
            return False
        data_name = manifest_name.removesuffix(".manifest.json")
        if manifest_name not in files or data_name not in files:
            return False
        if record["sha256"] != files[manifest_name].get("sha256"):
            return False
        if record["shard_sha256"] != files[data_name].get("sha256"):
            return False
    return True


def _valid_release_core_endpoint_records(
    value: Any,
    files: Mapping[str, Any],
    source_manifests: Mapping[str, Any],
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "behavior_test",
        "probe_test",
    }:
        return False
    required = {
        "input_manifest",
        "input_sha256",
        "metrics_manifest",
        "metrics_sha256",
        "closed_state_sha256",
        "phase",
        "evidence_sha256",
    }
    try:
        for endpoint, record in value.items():
            if not isinstance(record, Mapping) or set(record) != required:
                return False
            if not str(record["input_manifest"]).endswith(".jsonl.manifest.json"):
                return False
            if not str(record["metrics_manifest"]).endswith(".jsonl.manifest.json"):
                return False
            for key in ("input_manifest", "metrics_manifest"):
                manifest_name = str(record[key])
                data_name = manifest_name.removesuffix(".manifest.json")
                if manifest_name not in files or data_name not in files:
                    return False
                expected_shard = (
                    record["input_sha256"]
                    if key == "input_manifest"
                    else record["metrics_sha256"]
                )
                if source_manifests.get(manifest_name, {}).get("shard_sha256") != expected_shard:
                    return False
            for name in ("input_sha256", "metrics_sha256", "closed_state_sha256"):
                _sha256_value(record[name], f"{endpoint} {name}")
            expected_phase = "F1" if endpoint == "behavior_test" else "F2A"
            if record["phase"] != expected_phase:
                return False
            _sha256_value(record["evidence_sha256"], f"{endpoint} evidence_sha256")
    except (TypeError, ValueError):
        return False
    return True


def _directory_open_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("descriptor-safe directory traversal is unavailable")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_directory_descriptor(path: Path, label: str) -> int:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    flags = _directory_open_flags()
    try:
        descriptor = os.open("/", flags)
    except OSError as error:
        raise RuntimeError("descriptor-safe directory traversal is unavailable") from error
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(f"{label} must be a real directory") from error
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _same_directory(path: Path, descriptor: int) -> bool:
    try:
        current = _open_directory_descriptor(path, "release output parent")
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        expected = os.fstat(descriptor)
        actual = os.fstat(current)
        return (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
    finally:
        os.close(current)


def _open_parent_descriptor_at(root_descriptor: int, relative: Path) -> tuple[int, str]:
    if not relative.parts:
        raise ValueError("release path must name a file")
    try:
        descriptor = os.dup(root_descriptor)
    except OSError as error:
        raise ValueError("release root must be a real directory") from error
    try:
        for part in relative.parts[:-1]:
            try:
                child = os.open(part, _directory_open_flags(), dir_fd=descriptor)
            except OSError as error:
                raise ValueError("release source path has an unsafe directory") from error
            os.close(descriptor)
            descriptor = child
        return descriptor, relative.parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _open_parent_descriptor(root: Path, relative: Path) -> tuple[int, str]:
    root_descriptor = _open_directory_descriptor(root, "release root")
    try:
        return _open_parent_descriptor_at(root_descriptor, relative)
    finally:
        os.close(root_descriptor)


def _read_regular_bytes_at_descriptor(
    root_descriptor: int, relative: Path, label: str
) -> bytes:
    parent_descriptor, name = _open_parent_descriptor_at(root_descriptor, relative)
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise ValueError(f"{label} must be a regular file") from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"{label} must be a regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _read_regular_bytes_at(root: Path, relative: Path, label: str) -> bytes:
    root_descriptor = _open_directory_descriptor(root, "release root")
    try:
        _before_source_open(root / relative)
        return _read_regular_bytes_at_descriptor(root_descriptor, relative, label)
    finally:
        os.close(root_descriptor)


def _open_or_create_parent_descriptor_at(root_descriptor: int, relative: Path) -> tuple[int, str]:
    if not relative.parts:
        raise ValueError("release path must name a file")
    descriptor = os.dup(root_descriptor)
    try:
        for part in relative.parts[:-1]:
            try:
                os.mkdir(part, 0o755, dir_fd=descriptor)
            except FileExistsError:
                pass
            try:
                child = os.open(part, _directory_open_flags(), dir_fd=descriptor)
            except OSError as error:
                raise ValueError("release staging path has an unsafe directory") from error
            os.close(descriptor)
            descriptor = child
        return descriptor, relative.parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _write_new_regular_at(root_descriptor: int, relative: Path, payload: bytes) -> None:
    parent_descriptor, name = _open_or_create_parent_descriptor_at(root_descriptor, relative)
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd=parent_descriptor,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("could not write release artifact")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _create_staging_directory_at(parent_descriptor: int, destination_name: str) -> tuple[str, int]:
    if not destination_name or destination_name in {".", ".."}:
        raise ValueError("release output must name a directory")
    for _ in range(128):
        name = f".{destination_name}.staging-{secrets.token_hex(16)}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        try:
            return name, os.open(name, _directory_open_flags(), dir_fd=parent_descriptor)
        except BaseException:
            os.rmdir(name, dir_fd=parent_descriptor)
            raise
    raise RuntimeError("could not allocate a private release staging directory")


def _rename_directory_noreplace(
    parent_descriptor: int, source_name: str, destination_name: str
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform == "darwin":
        rename = getattr(libc, "renameatx_np", None)
        if rename is None:
            raise RuntimeError("exclusive directory rename is unavailable")
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(parent_descriptor, source, parent_descriptor, destination, 0x00000004)
    elif sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is not None:
            rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
            rename.restype = ctypes.c_int
            result = rename(parent_descriptor, source, parent_descriptor, destination, 0x00000001)
        else:
            numbers = {"x86_64": 316, "aarch64": 276, "arm64": 276, "i386": 353, "armv7l": 382, "ppc64le": 357}
            number = numbers.get(os.uname().machine)
            if number is None:
                raise RuntimeError("exclusive directory rename is unavailable")
            syscall = libc.syscall
            syscall.restype = ctypes.c_long
            result = syscall(number, parent_descriptor, source, parent_descriptor, destination, 0x00000001)
    else:
        raise RuntimeError("exclusive directory rename is unavailable")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == 17:
        raise FileExistsError(f"release output already exists: {destination_name}")
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _remove_tree_at(parent_descriptor: int, name: str) -> None:
    try:
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_descriptor)
    except FileNotFoundError:
        return
    try:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    _remove_tree_at(descriptor, entry.name)
                else:
                    os.unlink(entry.name, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    os.rmdir(name, dir_fd=parent_descriptor)


def _write_new_regular(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_empty_tree(root: Path) -> None:
    """Best-effort cleanup of a publication that never received a manifest."""

    try:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file() and not path.is_symlink():
                path.unlink()
            elif path.is_dir() and not path.is_symlink():
                path.rmdir()
        root.rmdir()
    except OSError:
        pass


def _require_regular_under(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("release path escapes source root") from error
    current = root
    for part in relative.parts:
        current = current / part
        try:
            status = current.lstat()
        except FileNotFoundError as error:
            raise ValueError("release source does not exist") from error
        if stat.S_ISLNK(status.st_mode):
            raise ValueError("release source cannot be a symlink")
    status = path.stat()
    if not stat.S_ISREG(status.st_mode):
        raise ValueError("release source must be a regular file")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _compact_json(value: Any) -> str:
    return _canonical_json(value).decode("utf-8")


def _save_figure(fig: Any, path: Path, plt: Any) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, metadata={"Software": "fa-report"})
    plt.close(fig)
    return path
