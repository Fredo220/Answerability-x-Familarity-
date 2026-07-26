"""Bounded, resumable Gemma screening for open Source-v6 development splits."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import subprocess
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trajectory_extractor.fa_config import (
    CONFIRMATORY_CHAT_TEMPLATE_SHA256,
    CONFIRMATORY_GENERATION,
    CONFIRMATORY_MODEL_ID,
    CONFIRMATORY_MODEL_REVISION,
    FAConfig,
)
from trajectory_extractor.fa_development_checkpoint import (
    DevelopmentCheckpointMirror,
)
from trajectory_extractor.fa_development_source import (
    DEVELOPMENT_DOMAIN_FIELDS,
    DEVELOPMENT_SPLITS,
    ERROR_TAXONOMY,
    DevelopmentSourceDesign,
    build_manual_error_audit_packet,
    compile_manual_error_audit,
    summarize_screening_yield,
)
from trajectory_extractor.fa_entities import (
    CandidateEntity,
    ScreeningQuestion,
    score_screening,
)
from trajectory_extractor.fa_runtime import HFModelRunner

_ALLOWED_SOURCE_REVISIONS = frozenset(
    {
        "fa-development-source-v6",
        "fa-development-source-v6-r2",
        "fa-development-source-v6-r3",
        "fa-development-source-v6-r4",
        "fa-development-source-v6-r5",
        "fa-development-source-v6-r6",
        "fa-development-source-v6-r7",
    }
)
_INTEGRITY_FILE = "source_integrity_v1.json"
_REGISTERED_ANSWER_PREFIX = re.compile(
    r"^(?:answer|final|final answer)\s*:\s*(.*)$",
    re.IGNORECASE,
)
_PARENTHETICAL_ANSWER = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def run_development_screening(
    config: FAConfig,
    source_root: str | Path,
    split: str,
    output_root: str | Path,
    *,
    batch_size: int = 8,
    runner: Any | None = None,
    freeze_manifest: str | Path | None = None,
    success_criteria: Mapping[str, Any] | None = None,
    pre_model_semantic_audit: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Screen one open Source-v6 split and resume only verified batch shards."""
    if split not in DEVELOPMENT_SPLITS:
        raise ValueError("screening requires a registered development split")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    _verify_config(config)
    source = _load_verified_source(Path(source_root), split)
    resolved_git_commit = git_commit or _current_git_commit()
    _validate_git_commit(resolved_git_commit)
    freeze_sha256 = None
    criteria_sha256 = None
    semantic_audit_sha256 = None
    frozen_criteria: Mapping[str, Any] | None = None
    if split == "instrument_development":
        if success_criteria is None:
            raise ValueError(
                "instrument_development requires registered success criteria"
            )
        if success_criteria.get("source_revision") != source["source_revision"]:
            raise ValueError("success criteria source revision mismatch")
        criteria_sha256 = _canonical_sha256(success_criteria)
        if source["source_revision"] in {
            "fa-development-source-v6-r5",
            "fa-development-source-v6-r6",
            "fa-development-source-v6-r7",
        }:
            if pre_model_semantic_audit is None:
                raise ValueError(
                    "this source revision requires a passing pre-model semantic audit"
                )
            semantic_audit_sha256 = _verify_pre_model_semantic_audit(
                Path(pre_model_semantic_audit),
                source_revision=source["source_revision"],
                source_integrity_sha256=source["integrity_sha256"],
            )
    if split == "construction_validation":
        if freeze_manifest is None:
            raise ValueError(
                "construction_validation requires a frozen instrument manifest"
            )
        frozen = _verify_freeze_manifest(
            Path(freeze_manifest),
            source_revision=source["source_revision"],
            source_integrity_sha256=source["integrity_sha256"],
            config_sha256=config.config_hash,
            git_commit=resolved_git_commit,
        )
        freeze_sha256 = frozen["manifest_sha256"]
        criteria_sha256 = frozen["success_criteria_sha256"]
        frozen_criteria = frozen["success_criteria"]
    candidates = source["candidates"]
    questions = source["questions"]
    prompts = _ordered_prompts(candidates, questions)

    identity = {
        "schema_version": 1,
        "source_revision": source["source_revision"],
        "source_integrity_sha256": source["integrity_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "question_sha256": source["question_sha256"],
        "split": split,
        "candidate_count": len(candidates),
        "prompt_count": len(prompts),
        "batch_size": batch_size,
        "config_sha256": config.config_hash,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "chat_template_sha256": config.chat_template_sha256,
        "generation": dict(config.generation),
        "screening_parser_sha256": development_screening_parser_sha256(),
        "git_commit": resolved_git_commit,
        "freeze_manifest_sha256": freeze_sha256,
        "success_criteria_sha256": criteria_sha256,
        "pre_model_semantic_audit_sha256": semantic_audit_sha256,
    }
    identity_sha256 = _sha256_bytes(_canonical_bytes(identity))
    run_dir = Path(output_root) / split / identity_sha256
    _write_immutable(run_dir / "execution_identity.json", _canonical_bytes(identity))
    checkpoint_mirror = (
        DevelopmentCheckpointMirror(checkpoint_root)
        if checkpoint_root is not None
        else None
    )
    if checkpoint_mirror is not None:
        checkpoint_mirror.restore(run_dir, identity_sha256)

    active_runner = runner
    rows: list[dict[str, Any]] = []
    resumed = 0
    for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
        batch = prompts[start : start + batch_size]
        batch_path = run_dir / f"batch-{batch_index:06d}.jsonl"
        checkpoint = _read_checkpoint(
            batch_path,
            identity_sha256=identity_sha256,
            batch_index=batch_index,
            expected_count=len(batch),
        )
        if checkpoint is not None:
            rows.extend(checkpoint)
            resumed += 1
            continue
        if active_runner is None:
            active_runner = HFModelRunner(config)
        rendered = tuple(
            active_runner.render_prompt(question.prompt) for _, question in batch
        )
        completions = tuple(active_runner.generate(rendered, dict(config.generation)))
        if len(completions) != len(batch) or any(
            not isinstance(value, str) for value in completions
        ):
            raise RuntimeError("model runner returned an invalid completion batch")
        batch_rows = [
            _item_row(candidate, question, completion, split)
            for (candidate, question), completion in zip(
                batch, completions, strict=True
            )
        ]
        _write_checkpoint(
            batch_path,
            batch_rows,
            identity_sha256=identity_sha256,
            batch_index=batch_index,
        )
        if checkpoint_mirror is not None:
            checkpoint_mirror.snapshot(run_dir, identity_sha256)
        rows.extend(batch_rows)

    _verify_complete_rows(rows, prompts, split)
    completions_by_entity = {
        candidate.entity_id: tuple(
            row["parsed_completion"]
            for row in rows
            if row["entity_id"] == candidate.entity_id
        )
        for candidate in candidates
    }
    summary = summarize_screening_yield(candidates, completions_by_entity)
    items_path = run_dir / "screening_items.jsonl"
    summary_path = run_dir / "screening_yield.json"
    _write_immutable(items_path, _jsonl_bytes(rows))
    _write_immutable(summary_path, _canonical_bytes(summary))
    gate_criteria = (
        success_criteria
        if split == "instrument_development"
        else frozen_criteria
    )
    gate_result = (
        evaluate_instrument_readiness(summary, rows, gate_criteria)
        if gate_criteria is not None
        else None
    )
    gate_path = run_dir / (
        "instrument_readiness_gate.json"
        if split == "instrument_development"
        else "construction_validation_gate.json"
    )
    if gate_result is not None:
        _write_immutable(gate_path, _canonical_bytes(gate_result))
    final_checkpoint = (
        checkpoint_mirror.snapshot(run_dir, identity_sha256)
        if checkpoint_mirror is not None
        else None
    )
    result = {
        "status": "completed",
        "split": split,
        "candidate_count": len(candidates),
        "item_count": len(rows),
        "resumed_batch_count": resumed,
        "execution_identity_sha256": identity_sha256,
        "items_path": str(items_path),
        "items_sha256": _sha256_file(items_path),
        "summary_path": str(summary_path),
        "summary_sha256": _sha256_file(summary_path),
        "summary": summary,
    }
    if gate_result is not None:
        result.update(
            {
                "gate_path": str(gate_path),
                "gate_sha256": _sha256_file(gate_path),
                "gate_result": gate_result,
            }
        )
    if final_checkpoint is not None:
        result["checkpoint_metadata"] = str(final_checkpoint)
    return result


def write_instrument_freeze_manifest(
    path: str | Path,
    *,
    source_root: str | Path,
    development_run_dir: str | Path,
    config: FAConfig,
    success_criteria: Mapping[str, Any],
    manual_audit_manifest: str | Path | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Freeze all identities before construction-validation outcomes are opened."""
    _verify_config(config)
    source = _load_verified_source(Path(source_root), "construction_validation")
    development_source = _load_verified_source(
        Path(source_root),
        "instrument_development",
    )
    if (
        development_source["source_revision"] != source["source_revision"]
        or development_source["integrity_sha256"] != source["integrity_sha256"]
    ):
        raise ValueError("development and validation source identities differ")
    resolved_git_commit = git_commit or _current_git_commit()
    _validate_git_commit(resolved_git_commit)
    if not success_criteria:
        raise ValueError("freeze manifest requires explicit success criteria")
    success_criteria_sha256 = _canonical_sha256(success_criteria)
    evidence = _load_development_evidence(
        Path(development_run_dir),
        source=development_source,
        config=config,
        git_commit=resolved_git_commit,
        success_criteria=success_criteria,
        success_criteria_sha256=success_criteria_sha256,
    )
    audit_requirement = success_criteria.get("human_audit")
    audit_required = (
        isinstance(audit_requirement, Mapping)
        and audit_requirement.get("required_before_construction_validation") is True
    )
    if audit_required and manual_audit_manifest is None:
        raise ValueError(
            "instrument freeze requires a completed independent human audit"
        )
    manual_audit = (
        _verify_manual_audit_manifest(
            Path(manual_audit_manifest),
            development_run_dir=Path(development_run_dir),
            source_revision=source["source_revision"],
            development_evidence=evidence,
            audit_requirement=audit_requirement,
        )
        if manual_audit_manifest is not None
        else None
    )
    manifest = {
        "schema_version": 1,
        "kind": "fa_source_v6_instrument_freeze",
        "source_revision": source["source_revision"],
        "source_integrity_sha256": source["integrity_sha256"],
        "config_sha256": config.config_hash,
        "git_commit": resolved_git_commit,
        "success_criteria": dict(success_criteria),
        "success_criteria_sha256": success_criteria_sha256,
        "development_evidence": evidence,
        "manual_audit": manual_audit,
    }
    target = Path(path)
    _write_immutable(target, _canonical_bytes(manifest))
    return {
        "freeze_manifest": str(target),
        "freeze_manifest_sha256": _sha256_file(target),
    }


def _verify_freeze_manifest(
    path: Path,
    *,
    source_revision: str,
    source_integrity_sha256: str,
    config_sha256: str,
    git_commit: str,
) -> dict[str, Any]:
    manifest = _read_json(path)
    expected_identity = {
        "schema_version": 1,
        "kind": "fa_source_v6_instrument_freeze",
        "source_revision": source_revision,
        "source_integrity_sha256": source_integrity_sha256,
        "config_sha256": config_sha256,
        "git_commit": git_commit,
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError("instrument freeze manifest identity mismatch")
    if (
        not isinstance(manifest.get("success_criteria"), dict)
        or not manifest["success_criteria"]
    ):
        raise ValueError("instrument freeze manifest lacks success criteria")
    criteria_sha256 = _canonical_sha256(manifest["success_criteria"])
    if manifest.get("success_criteria_sha256") != criteria_sha256:
        raise ValueError("instrument freeze success criteria hash mismatch")
    evidence = manifest.get("development_evidence")
    expected_evidence = {
        "source_revision": source_revision,
        "source_integrity_sha256": source_integrity_sha256,
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "screening_parser_sha256": development_screening_parser_sha256(),
        "success_criteria_sha256": criteria_sha256,
        "gate_passed": True,
    }
    if not isinstance(evidence, dict) or any(
        evidence.get(key) != value for key, value in expected_evidence.items()
    ):
        raise ValueError("instrument freeze lacks passing development evidence")
    semantic_audit_sha256 = evidence.get("pre_model_semantic_audit_sha256")
    _validate_sha256(semantic_audit_sha256, "pre-model semantic audit")
    audit_requirement = manifest["success_criteria"].get("human_audit")
    if (
        isinstance(audit_requirement, Mapping)
        and audit_requirement.get("required_before_construction_validation") is True
    ):
        manual_audit = manifest.get("manual_audit")
        expected_manual_audit = {
            "source_revision": source_revision,
            "development_execution_identity_sha256": evidence.get(
                "execution_identity_sha256"
            ),
            "items_sha256": evidence.get("items_sha256"),
            "summary_sha256": evidence.get("summary_sha256"),
            "audit_design_sha256": _canonical_sha256(audit_requirement),
            "sample_per_domain": audit_requirement.get("sample_per_domain"),
            "success_sample_per_domain": audit_requirement.get(
                "success_sample_per_domain"
            ),
            "seed": audit_requirement.get("seed"),
            "gate_passed": True,
        }
        if not isinstance(manual_audit, dict) or any(
            manual_audit.get(key) != value
            for key, value in expected_manual_audit.items()
        ):
            raise ValueError("instrument freeze lacks a verified manual audit")
        _validate_sha256(manual_audit.get("manifest_sha256"), "manual audit")
    return {
        "manifest_sha256": _sha256_file(path),
        "success_criteria_sha256": criteria_sha256,
        "success_criteria": manifest["success_criteria"],
    }


def evaluate_instrument_readiness(
    summary: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    success_criteria: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the preregistered open-development readiness gate."""
    if (
        success_criteria.get("schema_version") != 2
        or success_criteria.get("source_revision")
        not in _ALLOWED_SOURCE_REVISIONS
    ):
        raise ValueError("success criteria have an unsupported identity")
    gate = success_criteria.get("development_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("success criteria lack a development gate")
    required_integer_keys = (
        "candidate_count",
        "prompt_count",
        "qualification_threshold",
        "minimum_qualified_per_domain",
    )
    if any(
        type(gate.get(key)) is not int or gate[key] <= 0
        for key in required_integer_keys
    ):
        raise ValueError("development gate requires positive integer thresholds")

    failures = []
    observed = {
        "candidate_count": summary.get("entity_count"),
        "prompt_count": len(items),
        "qualified_by_domain": {
            domain: row.get("qualified_count")
            for domain, row in summary.get("by_domain", {}).items()
            if isinstance(row, Mapping)
        },
        "qualification_threshold": 2,
        "success_by_domain_relation": _success_by_domain_relation(items),
    }
    if observed["candidate_count"] != gate["candidate_count"]:
        failures.append("candidate_count")
    if observed["prompt_count"] != gate["prompt_count"]:
        failures.append("prompt_count")
    if gate["qualification_threshold"] != observed["qualification_threshold"]:
        failures.append("qualification_threshold")

    minimum_domain = gate["minimum_qualified_per_domain"]
    if set(observed["qualified_by_domain"]) != set(
        DEVELOPMENT_DOMAIN_FIELDS
    ) or any(
        type(value) is not int or value < minimum_domain
        for value in observed["qualified_by_domain"].values()
    ):
        failures.append("qualified_by_domain")

    expected_domain_relations = {
        domain: {field.property_id for field in fields}
        for domain, fields in DEVELOPMENT_DOMAIN_FIELDS.items()
    }
    minimum_matrix = gate.get("minimum_success_by_domain_relation")
    matrix_valid = (
        isinstance(minimum_matrix, Mapping)
        and set(minimum_matrix) == set(expected_domain_relations)
        and all(
            isinstance(minimum_matrix.get(domain), Mapping)
            and set(minimum_matrix[domain]) == expected_domain_relations[domain]
            and all(
                type(value) is int and value > 0
                for value in minimum_matrix[domain].values()
            )
            for domain in expected_domain_relations
        )
    )
    if not matrix_valid:
        raise ValueError(
            "development gate requires a complete positive "
            "domain-relation threshold matrix"
        )
    if any(
        observed["success_by_domain_relation"].get(domain, {}).get(relation, 0)
        < threshold
        for domain, relation_thresholds in minimum_matrix.items()
        for relation, threshold in relation_thresholds.items()
    ):
        failures.append("success_by_domain_relation")

    return {
        "gate_passed": not failures,
        "failed_criteria": sorted(failures),
        "observed": observed,
    }


def _success_by_domain_relation(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    successes = {
        domain: {field.property_id: 0 for field in fields}
        for domain, fields in DEVELOPMENT_DOMAIN_FIELDS.items()
    }
    for item in items:
        domain = item.get("domain")
        if domain not in DEVELOPMENT_DOMAIN_FIELDS:
            raise ValueError("development item has an invalid domain")
        fields = DEVELOPMENT_DOMAIN_FIELDS[domain]
        question_id = str(item.get("question_id", ""))
        try:
            question_index = int(question_id.rsplit("-q", 1)[1]) - 1
        except (IndexError, ValueError) as error:
            raise ValueError("development item has an invalid question ID") from error
        if question_index not in range(len(fields)):
            raise ValueError("development item has an invalid question index")
        if item.get("is_correct") is True:
            successes[domain][fields[question_index].property_id] += 1
    return successes


def _load_development_evidence(
    run_dir: Path,
    *,
    source: Mapping[str, Any],
    config: FAConfig,
    git_commit: str,
    success_criteria: Mapping[str, Any],
    success_criteria_sha256: str,
) -> dict[str, Any]:
    if success_criteria.get("source_revision") != source["source_revision"]:
        raise ValueError("success criteria source revision mismatch")
    identity_path = run_dir / "execution_identity.json"
    items_path = run_dir / "screening_items.jsonl"
    summary_path = run_dir / "screening_yield.json"
    identity = _read_json(identity_path)
    expected_identity = {
        "source_revision": source["source_revision"],
        "source_integrity_sha256": source["integrity_sha256"],
        "config_sha256": config.config_hash,
        "split": "instrument_development",
        "git_commit": git_commit,
        "screening_parser_sha256": development_screening_parser_sha256(),
    }
    if not isinstance(identity, dict):
        raise ValueError("development evidence identity mismatch")
    if identity.get("success_criteria_sha256") != success_criteria_sha256:
        raise ValueError("development evidence success criteria hash mismatch")
    semantic_audit_sha256 = identity.get("pre_model_semantic_audit_sha256")
    _validate_sha256(semantic_audit_sha256, "pre-model semantic audit")
    if any(
        identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError("development evidence identity mismatch")
    identity_sha256 = _canonical_sha256(identity)
    if run_dir.name != identity_sha256:
        raise ValueError("development evidence directory identity mismatch")

    try:
        items = [
            json.loads(line)
            for line in items_path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read development evidence items") from error
    prompts = _ordered_prompts(source["candidates"], source["questions"])
    _verify_complete_rows(items, prompts, "instrument_development")
    batch_size = identity.get("batch_size")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("development evidence has an invalid batch size")
    checkpoint_items = []
    identity_sha256 = _canonical_sha256(identity)
    for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
        expected_count = min(batch_size, len(prompts) - start)
        batch_path = run_dir / f"batch-{batch_index:06d}.jsonl"
        checkpoint = _read_checkpoint(
            batch_path,
            identity_sha256=identity_sha256,
            batch_index=batch_index,
            expected_count=expected_count,
        )
        if checkpoint is None:
            raise ValueError("development evidence is missing batch shards")
        checkpoint_items.extend(checkpoint)
    if items != checkpoint_items:
        raise ValueError("development evidence differs from verified batch shards")
    reconstructed_items = []
    for row, (candidate, question) in zip(items, prompts, strict=True):
        completion = row.get("completion")
        if not isinstance(completion, str):
            raise ValueError("development evidence raw completion is invalid")
        reconstructed_items.append(
            _item_row(
                candidate,
                question,
                completion,
                "instrument_development",
            )
        )
    if items != reconstructed_items:
        raise ValueError("development evidence contains derived-field drift")
    completions_by_entity = {
        candidate.entity_id: tuple(
            row["parsed_completion"]
            for row in items
            if row["entity_id"] == candidate.entity_id
        )
        for candidate in source["candidates"]
    }
    recomputed = summarize_screening_yield(
        source["candidates"],
        completions_by_entity,
    )
    summary = _read_json(summary_path)
    if summary != recomputed:
        raise ValueError("development evidence summary does not recompute")
    gate_result = evaluate_instrument_readiness(
        summary,
        items,
        success_criteria,
    )
    if not gate_result["gate_passed"]:
        failed = ", ".join(gate_result["failed_criteria"])
        raise ValueError(f"development gate failed: {failed}")
    return {
        "source_revision": source["source_revision"],
        "source_integrity_sha256": source["integrity_sha256"],
        "config_sha256": config.config_hash,
        "git_commit": git_commit,
        "screening_parser_sha256": development_screening_parser_sha256(),
        "success_criteria_sha256": success_criteria_sha256,
        "pre_model_semantic_audit_sha256": semantic_audit_sha256,
        "execution_identity_sha256": identity_sha256,
        "items_sha256": _sha256_file(items_path),
        "summary_sha256": _sha256_file(summary_path),
        "gate_passed": True,
        "gate_result": gate_result,
    }


def _verify_manual_audit_manifest(
    path: Path,
    *,
    development_run_dir: Path,
    source_revision: str,
    development_evidence: Mapping[str, Any],
    audit_requirement: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("manual audit manifest must be a JSON object")
    identity = _read_json(development_run_dir / "execution_identity.json")
    items = [
        json.loads(line)
        for line in (
            development_run_dir / "screening_items.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    sample_per_domain = manifest.get("sample_per_domain")
    success_sample_per_domain = manifest.get("success_sample_per_domain")
    seed = manifest.get("seed")
    if not isinstance(audit_requirement, Mapping):
        raise ValueError("manual audit requires a registered audit design")
    expected_sample = audit_requirement.get("sample_per_domain")
    expected_success_sample = audit_requirement.get("success_sample_per_domain")
    expected_seed = audit_requirement.get("seed")
    if (
        type(sample_per_domain) is not int
        or sample_per_domain <= 0
        or type(seed) is not int
        or sample_per_domain != expected_sample
        or type(success_sample_per_domain) is not int
        or success_sample_per_domain < 0
        or success_sample_per_domain != expected_success_sample
        or seed != expected_seed
    ):
        raise ValueError(
            "manual audit manifest does not match the registered sampling design"
        )
    packet = build_manual_error_audit_packet(
        items,
        sample_per_domain=sample_per_domain,
        success_sample_per_domain=success_sample_per_domain,
        seed=seed,
    )
    ratings = manifest.get("ratings")
    if not isinstance(ratings, list):
        raise ValueError("manual audit manifest ratings must be an array")
    compiled = compile_manual_error_audit(packet, ratings)
    expected_identity = {
        "schema_version": 1,
        "kind": "fa_source_v6_manual_error_audit",
        "source_revision": source_revision,
        "development_execution_identity_sha256": (
            development_evidence["execution_identity_sha256"]
        ),
        "items_sha256": development_evidence["items_sha256"],
        "summary_sha256": development_evidence["summary_sha256"],
    }
    if any(manifest.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("manual audit manifest identity mismatch")
    if manifest.get("packet") != list(packet) or manifest.get("compiled") != compiled:
        raise ValueError("manual audit manifest does not recompute")
    if not isinstance(identity, dict) or _canonical_sha256(identity) != (
        development_evidence["execution_identity_sha256"]
    ):
        raise ValueError("manual audit references invalid development evidence")
    disallowed = audit_requirement.get("disallowed_error_labels")
    maximum_disallowed = audit_requirement.get("maximum_disallowed_count")
    if (
        not isinstance(disallowed, list)
        or not disallowed
        or any(label not in ERROR_TAXONOMY for label in disallowed)
        or type(maximum_disallowed) is not int
        or maximum_disallowed < 0
    ):
        raise ValueError("manual audit acceptance rule is invalid")
    disallowed_count = sum(
        compiled["decision_counts"].get(label, 0) for label in disallowed
    )
    if disallowed_count > maximum_disallowed:
        raise ValueError("manual audit acceptance gate failed")
    model_score_by_audit_id = {
        row["audit_id"]: bool(item["is_correct"])
        for row in packet
        for item in items
        if item["question_id"] == row["question_id"]
    }
    if len(model_score_by_audit_id) != len(packet):
        raise ValueError("manual audit packet cannot be aligned to model scores")
    scoring_disagreement_count = sum(
        (compiled["decisions"][audit_id] == "no_error")
        != model_score_by_audit_id[audit_id]
        for audit_id in compiled["decisions"]
    )
    maximum_scoring_disagreement = audit_requirement.get(
        "maximum_scoring_disagreement_count"
    )
    if (
        type(maximum_scoring_disagreement) is not int
        or maximum_scoring_disagreement < 0
    ):
        raise ValueError("manual audit scoring agreement rule is invalid")
    if scoring_disagreement_count > maximum_scoring_disagreement:
        raise ValueError("manual audit scoring agreement gate failed")
    return {
        "manifest_sha256": _sha256_file(path),
        "source_revision": source_revision,
        "development_execution_identity_sha256": development_evidence[
            "execution_identity_sha256"
        ],
        "items_sha256": development_evidence["items_sha256"],
        "summary_sha256": development_evidence["summary_sha256"],
        "audit_design_sha256": _canonical_sha256(audit_requirement),
        "sample_per_domain": sample_per_domain,
        "success_sample_per_domain": success_sample_per_domain,
        "seed": seed,
        "item_count": compiled["item_count"],
        "adjudicated_count": compiled["adjudicated_count"],
        "decision_counts": compiled["decision_counts"],
        "disallowed_count": disallowed_count,
        "scoring_disagreement_count": scoring_disagreement_count,
        "gate_passed": True,
    }


def _verify_pre_model_semantic_audit(
    path: Path,
    *,
    source_revision: str,
    source_integrity_sha256: str,
) -> str:
    manifest = _read_json(path)
    expected = {
        "schema_version": 1,
        "kind": "fa_source_v6_pre_model_semantic_audit",
        "source_revision": source_revision,
        "source_integrity_sha256": source_integrity_sha256,
        "status": "passed",
        "blocker_count": 0,
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("pre-model semantic audit identity or result mismatch")
    if not isinstance(manifest.get("auditor_id"), str) or not manifest[
        "auditor_id"
    ].strip():
        raise ValueError("pre-model semantic audit lacks an auditor identity")
    return _sha256_file(path)


def _current_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot resolve git commit for screening identity") from error


def _validate_git_commit(value: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("screening git commit must be a 40-character lowercase SHA")


def _validate_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")


def _verify_clean_checkout(expected_commit: str) -> None:
    _validate_git_commit(expected_commit)
    current = _current_git_commit()
    if current != expected_commit:
        raise ValueError("screening commit does not match the checked-out HEAD")
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot verify clean screening checkout") from error
    if dirty:
        raise ValueError("screening requires a clean git checkout")


def _verify_config(config: FAConfig) -> None:
    expected = (
        config.profile == "confirmatory"
        and config.model_id == CONFIRMATORY_MODEL_ID
        and config.model_revision == CONFIRMATORY_MODEL_REVISION
        and config.tokenizer_revision == CONFIRMATORY_MODEL_REVISION
        and config.chat_template_sha256 == CONFIRMATORY_CHAT_TEMPLATE_SHA256
        and dict(config.generation) == CONFIRMATORY_GENERATION
    )
    if not expected:
        raise ValueError(
            "development screening requires the pinned confirmatory config"
        )


def _load_verified_source(root: Path, selected_split: str) -> dict[str, Any]:
    integrity_path = root / _INTEGRITY_FILE
    integrity = _read_json(integrity_path)
    source_revision = (
        integrity.get("source_revision") if isinstance(integrity, dict) else None
    )
    if (
        not isinstance(integrity, dict)
        or integrity.get("schema_version") != 1
        or source_revision not in _ALLOWED_SOURCE_REVISIONS
    ):
        raise ValueError("Source-v6 integrity identity is invalid")
    snapshot_path = root / str(integrity.get("source_snapshot", ""))
    if _sha256_file(snapshot_path) != integrity.get("source_snapshot_sha256"):
        raise ValueError("Source-v6 snapshot hash mismatch")
    snapshot = _read_json(snapshot_path)
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("source_revision") != source_revision
        or snapshot.get("claim_scope") != "instrument_development_only"
        or snapshot.get("materialized_files") != integrity.get("materialized_files")
    ):
        raise ValueError("Source-v6 snapshot identity mismatch")
    design = DevelopmentSourceDesign(
        revision=str(snapshot["source_revision"]),
        split_seed=int(snapshot["split_seed"]),
        candidates_per_domain_per_split=int(
            snapshot["candidates_per_domain_per_split"]
        ),
        splits=tuple(snapshot["splits"]),
    )
    if tuple(design.splits) != DEVELOPMENT_SPLITS:
        raise ValueError("Source-v6 registered split identity mismatch")
    files = integrity.get("materialized_files", {}).get(selected_split)
    if not isinstance(files, dict):
        raise ValueError("Source-v6 integrity is missing the selected split")
    candidate_path = root / str(files.get("candidate_manifest", ""))
    question_path = root / str(files.get("question_manifest", ""))
    candidate_sha256 = _sha256_file(candidate_path)
    question_sha256 = _sha256_file(question_path)
    if candidate_sha256 != files.get(
        "candidate_sha256"
    ) or question_sha256 != files.get("question_sha256"):
        raise ValueError("Source-v6 materialized file hash mismatch")
    candidates = tuple(CandidateEntity(**row) for row in _read_array(candidate_path))
    questions = tuple(
        ScreeningQuestion(**row) for row in _read_array(question_path)
    )
    expected_candidates = (
        design.candidates_per_domain_per_split * len(DEVELOPMENT_DOMAIN_FIELDS)
    )
    if len(candidates) != expected_candidates or len(questions) != (
        expected_candidates * 3
    ):
        raise ValueError("Source-v6 selected split count mismatch")
    if any(candidate.split != selected_split for candidate in candidates):
        raise ValueError("Source-v6 candidate belongs to a foreign split")
    candidate_ids = [candidate.entity_id for candidate in candidates]
    candidate_qids = [candidate.qid for candidate in candidates]
    if (
        len(set(candidate_ids)) != len(candidate_ids)
        or len(set(candidate_qids)) != len(candidate_qids)
    ):
        raise ValueError("Source-v6 selected split contains duplicate candidates")
    observed_domains = {
        domain: sum(candidate.coarse_type == domain for candidate in candidates)
        for domain in DEVELOPMENT_DOMAIN_FIELDS
    }
    expected_domains = {
        domain: design.candidates_per_domain_per_split
        for domain in DEVELOPMENT_DOMAIN_FIELDS
    }
    if observed_domains != expected_domains:
        raise ValueError("Source-v6 selected split domain balance mismatch")
    _ordered_prompts(candidates, questions)
    return {
        "source_revision": source_revision,
        "candidates": candidates,
        "questions": questions,
        "integrity_sha256": _sha256_file(integrity_path),
        "candidate_sha256": candidate_sha256,
        "question_sha256": question_sha256,
    }


def _ordered_prompts(
    candidates: Sequence[CandidateEntity],
    questions: Sequence[ScreeningQuestion],
) -> tuple[tuple[CandidateEntity, ScreeningQuestion], ...]:
    by_qid: dict[str, list[ScreeningQuestion]] = {}
    question_ids = []
    for question in questions:
        by_qid.setdefault(question.qid, []).append(question)
        question_ids.append(question.question_id)
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Source-v6 questions contain duplicate IDs")
    candidate_qids = {candidate.qid for candidate in candidates}
    if set(by_qid) != candidate_qids:
        raise ValueError("Source-v6 questions contain an orphan or missing candidate")
    ordered = []
    for candidate in candidates:
        candidate_questions = sorted(
            by_qid.get(candidate.qid, ()), key=lambda row: row.question_id
        )
        if len(candidate_questions) != 3:
            raise ValueError("Source-v6 requires exactly three prompts per candidate")
        if tuple(row.accepted_aliases for row in candidate_questions) != tuple(
            candidate.screening_aliases
        ):
            raise ValueError("Source-v6 candidate and question aliases mismatch")
        ordered.extend((candidate, question) for question in candidate_questions)
    if len(ordered) != len(candidates) * 3:
        raise ValueError("Source-v6 prompt count mismatch")
    return tuple(ordered)


def _item_row(
    candidate: CandidateEntity,
    question: ScreeningQuestion,
    completion: str,
    split: str,
) -> dict[str, Any]:
    question_index = int(question.question_id.rsplit("-q", 1)[1]) - 1
    parsed_completion = parse_development_screening_answer(
        completion,
        question.accepted_aliases,
    )
    probe = [""] * 3
    probe[question_index] = parsed_completion
    is_correct = score_screening(candidate, probe).correct_answers[question_index]
    return {
        "kind": "development_screening_item",
        "split": split,
        "entity_id": candidate.entity_id,
        "qid": candidate.qid,
        "domain": candidate.coarse_type,
        "question_id": question.question_id,
        "prompt": question.prompt,
        "accepted_aliases": list(question.accepted_aliases),
        "completion": completion,
        "parsed_completion": parsed_completion,
        "is_correct": is_correct,
    }


def parse_development_screening_answer(
    raw_output: str,
    accepted_aliases: Sequence[str],
) -> str:
    """Remove only registered wrappers before exact alias scoring."""
    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be text")
    aliases = tuple(accepted_aliases)
    if not aliases or any(not isinstance(alias, str) for alias in aliases):
        raise ValueError("accepted_aliases must contain text")

    value = raw_output.strip()
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[1].strip()
    lines = tuple(line.strip() for line in value.splitlines() if line.strip())
    value = lines[-1] if lines else ""

    prefix_match = _REGISTERED_ANSWER_PREFIX.fullmatch(value)
    if prefix_match is not None:
        value = prefix_match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()

    parenthetical = _PARENTHETICAL_ANSWER.fullmatch(value)
    if parenthetical is not None:
        base = parenthetical.group(1).strip()
        qualifier = parenthetical.group(2).strip()
        alias_keys = {_answer_key(alias) for alias in aliases}
        if _answer_key(base) in alias_keys and _answer_key(qualifier) in alias_keys:
            return base
    return value


def development_screening_parser_sha256() -> str:
    """Bind resumable checkpoints to the exact development answer parser."""
    return _canonical_sha256(
        {
            "revision": "fa-development-screening-answer-v2",
            "implementation": inspect.getsource(parse_development_screening_answer),
            "rules": [
                "strip",
                "suffix-after-final-think-close",
                "last-nonempty-line",
                "registered-answer-prefix-only",
                "single-matching-quote-pair",
                "parenthetical-only-when-both-parts-are-registered-aliases",
            ],
        }
    )


def _answer_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _verify_complete_rows(
    rows: Sequence[Mapping[str, Any]],
    prompts: Sequence[tuple[CandidateEntity, ScreeningQuestion]],
    split: str,
) -> None:
    expected = [question.question_id for _, question in prompts]
    observed = [row.get("question_id") for row in rows]
    if observed != expected or len(set(observed)) != len(expected):
        raise ValueError("screening checkpoint identity or count mismatch")
    if any(
        row.get("kind") != "development_screening_item" or row.get("split") != split
        for row in rows
    ):
        raise ValueError("screening checkpoint contains a foreign identity")


def _write_checkpoint(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    identity_sha256: str,
    batch_index: int,
) -> None:
    payload = _jsonl_bytes(rows)
    _write_immutable(path, payload)
    manifest = {
        "schema_version": 1,
        "kind": "development_screening_checkpoint",
        "identity_sha256": identity_sha256,
        "batch_index": batch_index,
        "data_file": path.name,
        "data_sha256": _sha256_bytes(payload),
        "row_count": len(rows),
    }
    _write_immutable(
        path.with_name(f"{path.stem}.manifest.json"),
        _canonical_bytes(manifest),
    )


def _read_checkpoint(
    path: Path,
    *,
    identity_sha256: str,
    batch_index: int,
    expected_count: int,
) -> list[dict[str, Any]] | None:
    manifest_path = path.with_name(f"{path.stem}.manifest.json")
    if not path.exists() and not manifest_path.exists():
        return None
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("screening checkpoint is incomplete")
    manifest = _read_json(manifest_path)
    payload = path.read_bytes()
    expected = {
        "schema_version": 1,
        "kind": "development_screening_checkpoint",
        "identity_sha256": identity_sha256,
        "batch_index": batch_index,
        "data_file": path.name,
        "data_sha256": _sha256_bytes(payload),
        "row_count": expected_count,
    }
    if manifest != expected:
        raise ValueError("screening checkpoint manifest mismatch")
    rows = [json.loads(line) for line in payload.splitlines()]
    if len(rows) != expected_count or _jsonl_bytes(rows) != payload:
        raise ValueError("screening checkpoint payload mismatch")
    return rows


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_array(path: Path) -> list[dict[str, Any]]:
    value = _read_json(path)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path.name} must contain a JSON array of objects")
    return value


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read verified artifact {path}") from error


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as error:
        raise ValueError(f"cannot verify artifact hash: {path}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded Gemma screening for one Source-v6 development split."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split", choices=DEVELOPMENT_SPLITS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--success-criteria", type=Path)
    parser.add_argument("--pre-model-semantic-audit", type=Path)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Drive-compatible mirror root; output-root itself must stay local.",
    )
    parser.add_argument(
        "--git-commit",
        help="Exact source commit for archive-based runtimes without a .git directory.",
    )
    args = parser.parse_args(argv)
    resolved_commit = args.git_commit or _current_git_commit()
    _verify_clean_checkout(resolved_commit)
    success_criteria = (
        _read_json(args.success_criteria)
        if args.success_criteria is not None
        else None
    )
    if success_criteria is not None and not isinstance(success_criteria, dict):
        raise ValueError("success criteria must be a JSON object")
    result = run_development_screening(
        FAConfig.from_json(args.config),
        args.source_root,
        args.split,
        args.output_root,
        batch_size=args.batch_size,
        freeze_manifest=args.freeze_manifest,
        success_criteria=success_criteria,
        pre_model_semantic_audit=args.pre_model_semantic_audit,
        checkpoint_root=args.checkpoint_root,
        git_commit=resolved_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
