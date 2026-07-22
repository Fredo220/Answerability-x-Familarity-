from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from trajectory_extractor.fa_activations import (
    HFSelectedPositionRunner,
    write_activation_shard,
)
from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_config import FAConfig, SMOKE_CHAT_TEMPLATE_SHA256
from trajectory_extractor.fa_data import (
    CONFIRMATORY_POWER_SEED,
    CONFIRMATORY_POWER_SIMULATIONS,
    REGISTERED_POWER_GRID,
    FAExample,
    PowerAudit,
    PowerCell,
    audit_dataset,
    build_factorial_examples,
    build_manifest,
    build_same_string_examples,
    simulate_interaction_power,
)
from trajectory_extractor.fa_entities import (
    CandidateEntity,
    EntityMatch,
    NaturalnessRating,
    SyntheticCandidate,
    audit_naturalness_manifest,
    match_synthetic_entities,
    score_screening,
)
from trajectory_extractor.fa_runtime import (
    HFModelRunner,
    load_pinned_tokenizer,
    run_generation_shard,
    validate_runner_binding,
)
from trajectory_extractor.fa_scoring import (
    behavioral_gate,
    crossed_bootstrap,
    estimate_behavior,
    score_response,
)
from trajectory_extractor.fa_probes import (
    TASKS,
    NullSelectionResult,
    ProbeRow,
    ProbeSourceIdentity,
    ProbeTestAuthorization,
    SelectionManifest,
    evaluate_probe_bundle_once,
    f2a_selection_bundle_hash,
    fit_selection,
    run_full_selection_nulls,
)


FA_COMMANDS = (
    "fa-screen-entities", "fa-build-pilot", "fa-build-confirmatory", "fa-audit-manifest",
    "fa-run-generation", "fa-score-behavior",
    "fa-extract-activations", "fa-fit-probes", "fa-seal-selection", "fa-unlock-endpoint",
    "fa-evaluate-behavior-test", "fa-evaluate-probe-test", "fa-evaluate-intervention-test",
    "fa-run-interventions", "fa-select-circuit-cases", "fa-audit-circuit-fidelity", "fa-build-report",
)
_IMPLEMENTED = frozenset(
    (
        *FA_COMMANDS[:6],
        "fa-extract-activations",
        "fa-fit-probes",
        "fa-seal-selection",
        "fa-unlock-endpoint",
        "fa-evaluate-behavior-test",
        "fa-evaluate-probe-test",
    )
)
_GENERATION_NAMESPACES = ("pilot", "mechanism_train", "locked_validation", "circuit_dev", "behavior_test", "probe_test", "intervention_test")
_PROTECTED = frozenset({"behavior_test", "probe_test", "intervention_test"})
_SMOKE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "familiarity_answerability_qwen06b_smoke.json"
)
_PREREGISTRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "familiarity_answerability_preregistration.md"
)
_TOKENIZER_LOADER = None
_POWER_EXECUTOR = simulate_interaction_power
_SMOKE_CHAT_TEMPLATE_SHA256 = SMOKE_CHAT_TEMPLATE_SHA256
_ACTIVATION_RUNNER_FACTORY = HFSelectedPositionRunner
_ACTIVATION_SHARD_WRITER = write_activation_shard
_BEHAVIOR_BOOTSTRAP = crossed_bootstrap
_BEHAVIOR_GATE = behavioral_gate
_PROBE_SELECTOR = fit_selection
_PROBE_NULL_SELECTOR = run_full_selection_nulls
_PROBE_BUNDLE_EVALUATOR = evaluate_probe_bundle_once


@dataclass(frozen=True)
class VerifiedPromptManifest:
    config_hash: str
    manifest_sha256: str
    full_manifest_sha256: str
    chat_template_sha256: str
    namespace: str
    examples: tuple[FAExample, ...]
    model_sha256: str
    tokenizer_sha256: str
    tokenizer_pin_manifest_path: Path
    tokenizer_pin_sha256: str
    naturalness_audit_manifest_path: Path | None
    naturalness_audit_sha256: str | None
    generation: dict[str, Any]
    shard_manifest_path: Path
    shard_sha256: str


@dataclass(frozen=True)
class VerifiedF2ASelectionBundle:
    selections: Mapping[str, SelectionManifest]
    null_selections: Mapping[str, tuple[NullSelectionResult, ...]]
    selection_bundle_hash: str
    probe_test_prompt_sha256: str
    probe_test_task_identities_sha256: str
    shard_manifest_path: Path
    shard_sha256: str


def register_fa_subcommands(subparsers: argparse._SubParsersAction) -> None:
    parsers: dict[str, argparse.ArgumentParser] = {}
    for command in FA_COMMANDS:
        parser = subparsers.add_parser(command)
        parser.error = lambda message, command=command: _argument_error(command, message)
        parser.add_argument("--config", required=True)
        parser.add_argument("--root", default=".")
        parsers[command] = parser
    parsers["fa-screen-entities"].add_argument("--candidates-manifest", required=True)
    parsers["fa-screen-entities"].add_argument("--screening-manifest", required=True)
    parsers["fa-screen-entities"].add_argument("--synthetic-manifest", required=True)
    parsers["fa-build-pilot"].add_argument("--matches-manifest", required=True)
    parsers["fa-build-confirmatory"].add_argument("--matches-manifest", required=True)
    parsers["fa-build-confirmatory"].add_argument("--pilot-gate-manifest", required=True)
    parsers["fa-build-confirmatory"].add_argument(
        "--naturalness-ratings-manifest", required=True
    )
    power = parsers["fa-build-confirmatory"].add_mutually_exclusive_group(required=True)
    power.add_argument("--power-audit-manifest")
    power.add_argument("--run-registered-power-audit", action="store_true")
    parsers["fa-audit-manifest"].add_argument("--manifest", required=True)
    generation = parsers["fa-run-generation"]
    generation.add_argument("--manifest", required=True)
    generation.add_argument("--shard-id", required=True)
    generation.add_argument("--namespace", choices=_GENERATION_NAMESPACES, required=True)
    generation.add_argument("--resume", action="store_true")
    activations = parsers["fa-extract-activations"]
    activations.add_argument("--manifest", required=True)
    activations.add_argument("--shard-id", required=True)
    activations.add_argument("--namespace", choices=_GENERATION_NAMESPACES, required=True)
    activations.add_argument("--layers")
    activations.add_argument("--resume", action="store_true")
    fit_probes = parsers["fa-fit-probes"]
    fit_probes.add_argument("--train-rows-manifest", required=True)
    fit_probes.add_argument("--validation-rows-manifest", required=True)
    fit_probes.add_argument("--probe-test-manifest", required=True)
    fit_probes.add_argument("--shard-id", required=True)
    seal_selection = parsers["fa-seal-selection"]
    seal_selection.add_argument("--selection-manifest", required=True)
    seal_selection.add_argument("--probe-test-manifest", required=True)
    seal_selection.add_argument("--probe-rows-manifest", required=True)
    behavior_test = parsers["fa-evaluate-behavior-test"]
    behavior_test.add_argument("--manifest", required=True)
    behavior_test.add_argument("--shard-id", required=True)
    probe_test = parsers["fa-evaluate-probe-test"]
    probe_test.add_argument("--selection-manifest", required=True)
    probe_test.add_argument("--probe-test-manifest", required=True)
    probe_test.add_argument("--probe-rows-manifest", required=True)
    score = parsers["fa-score-behavior"]
    score.add_argument("--manifest", required=True)
    score.add_argument("--generation-manifest", required=True)


def dispatch_fa(args: argparse.Namespace) -> int | None:
    command = getattr(args, "command", None)
    if command not in FA_COMMANDS:
        return None
    if command not in _IMPLEMENTED:
        print(
            json.dumps(
                {
                    "command": command,
                    "error": {
                        "message": "FA command is not implemented",
                        "type": "NotImplementedError",
                    },
                    "status": "not_implemented",
                },
                sort_keys=True,
            )
        )
        return 2
    try:
        config = FAConfig.from_json(args.config)
        root = Path(args.root)
        if command == "fa-screen-entities":
            payload = _screen_entities(config, root, args)
        elif command in {"fa-build-pilot", "fa-build-confirmatory"}:
            payload = _build_manifest(config, root, args, confirmatory=command == "fa-build-confirmatory")
        elif command == "fa-audit-manifest":
            payload = _audit_manifest(config, root, args)
        elif command == "fa-run-generation":
            payload = _run_generation(config, root, args)
        elif command == "fa-extract-activations":
            payload = _extract_activations(config, root, args)
        elif command == "fa-fit-probes":
            payload = _fit_probes(config, root, args)
        elif command == "fa-seal-selection":
            payload = _seal_probe_selection(config, root, args)
        elif command == "fa-evaluate-behavior-test":
            payload = _evaluate_behavior_test(config, root, args)
        elif command == "fa-evaluate-probe-test":
            payload = _evaluate_probe_test(config, root, args)
        elif command == "fa-unlock-endpoint":
            payload = _reject_standalone_unlock()
        else:
            payload = _score_behavior(config, root, args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": command,
                    "error": {"message": str(error), "type": type(error).__name__},
                    "status": "error",
                },
                sort_keys=True,
            )
        )
        return 3 if isinstance(error, (ImportError, OSError, RuntimeError)) else 2
    print(json.dumps(_json_safe({"command": command, **payload}), sort_keys=True, allow_nan=False))
    return 3 if payload.get("status") == "infrastructure_failure" else 0


def _reject_standalone_unlock() -> dict[str, Any]:
    raise ValueError(
        "standalone endpoint unlock is disabled; a dedicated protected evaluation "
        "command must acquire and close its lease atomically"
    )


def _argument_error(command: str, message: str) -> None:
    print(
        json.dumps(
            {
                "command": command,
                "error": {"message": message, "type": "ArgumentError"},
                "status": "error",
            },
            sort_keys=True,
        )
    )
    raise SystemExit(2)


def _screen_entities(config: FAConfig, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    candidates = tuple(CandidateEntity(**_without_schema(row)) for row in _read_json_rows(args.candidates_manifest))
    synthetic = tuple(SyntheticCandidate(**_without_schema(row)) for row in _read_json_rows(args.synthetic_manifest))
    completions = _read_json_object(args.screening_manifest)
    qualified = []
    for candidate in candidates:
        result = score_screening(candidate, completions.get(candidate.entity_id, ()))
        if result.qualifies:
            qualified.append(candidate)
    matches = match_synthetic_entities(qualified, synthetic, _WhitespaceTokenizer())
    path = _write_manifest(root, config.run_id, "screened_matches", [asdict(row) for row in matches])
    return {"status": "screened", "manifest": str(path), "count": len(matches)}


def _build_manifest(config: FAConfig, root: Path, args: argparse.Namespace, *, confirmatory: bool) -> dict[str, Any]:
    store = FAArtifactStore(root)
    gate = None
    naturalness_shard = None
    if confirmatory:
        if config.profile != "confirmatory":
            raise ValueError("confirmatory construction requires the confirmatory config")
        smoke_config = FAConfig.from_json(_SMOKE_CONFIG_PATH)
        gate = _load_verified_pilot_gate(
            store, args.pilot_gate_manifest, smoke_config
        )
        if gate.get("status") != "passed":
            raise ValueError("confirmatory construction requires a verified passed pilot gate")
    matches = tuple(EntityMatch(**_without_schema(row)) for row in _read_json_rows(args.matches_manifest))
    if confirmatory:
        ratings, ratings_shard = _load_verified_naturalness_ratings(
            store,
            args.naturalness_ratings_manifest,
            config,
        )
        audit = audit_naturalness_manifest(matches, ratings)
        if not audit.accepted_pair_ids:
            raise ValueError("confirmatory construction requires accepted naturalness pairs")
        accepted = frozenset(audit.accepted_pair_ids)
        audited_matches = matches
        matches = tuple(match for match in matches if match.pair_id in accepted)
        naturalness_shard = _write_naturalness_audit(
            store, config, audited_matches, audit, ratings, ratings_shard
        )
    prepared = load_pinned_tokenizer(config, tokenizer_loader=_TOKENIZER_LOADER)
    factorial_rows = build_factorial_examples(config, matches, tokenizer=prepared.tokenizer)
    rows = factorial_rows
    power_shard = None
    power_audit = None
    if confirmatory:
        rows += build_same_string_examples(config, matches, tokenizer=prepared.tokenizer)
        power_audit, power_shard = _prepare_power_audit(
            store,
            config,
            factorial_rows,
            args.power_audit_manifest,
            run_registered=args.run_registered_power_audit,
        )
    manifest = build_manifest(config, rows, power_audit=power_audit)
    tokenizer_pin = _write_tokenizer_pin(store, config, prepared, manifest.manifest_sha256)
    capabilities = {}
    for namespace in sorted({row.split for row in rows}):
        namespace_rows = tuple(row for row in rows if row.split == namespace)
        capabilities[namespace] = _write_prompt_capability(
            store,
            config,
            manifest.manifest_sha256,
            namespace,
            namespace_rows,
            prepared.chat_template_sha256,
            tokenizer_pin,
            naturalness_shard,
        )

    if not confirmatory:
        pilot = capabilities.get("pilot")
        if pilot is None or len(capabilities) != 1:
            raise ValueError("pilot construction must emit exactly one pilot capability")
        return {
            "status": "built",
            "manifest": str(pilot.manifest_path),
            "count": len(rows),
            "manifest_sha256": manifest.manifest_sha256,
            "tokenizer_pin_manifest": str(tokenizer_pin.manifest_path),
        }

    assert power_shard is not None and gate is not None
    missing_protected = _PROTECTED - capabilities.keys()
    if missing_protected:
        raise ValueError(
            "confirmatory manifest is missing protected endpoint capabilities: "
            + ", ".join(sorted(missing_protected))
        )
    index_row = _confirmatory_index_record(
        store,
        config,
        manifest.manifest_sha256,
        rows,
        capabilities,
        power_shard,
        tokenizer_pin,
        naturalness_shard,
    )
    index = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        f"confirmatory-index-{manifest.manifest_sha256[:16]}",
        [index_row],
        {
            "config_sha256": config.config_hash,
            "source_manifest_sha256": manifest.manifest_sha256,
            "power_audit_sha256": power_shard.sha256,
        },
        record_kind="confirmatory_index",
    )
    return {
        "status": "built",
        "manifest": str(index.manifest_path),
        "count": len(rows),
        "manifest_sha256": manifest.manifest_sha256,
        "power_audit_manifest": str(power_shard.manifest_path),
        "namespace_manifests": {
            namespace: str(shard.manifest_path)
            for namespace, shard in sorted(capabilities.items())
        },
        "tokenizer_pin_manifest": str(tokenizer_pin.manifest_path),
        "naturalness_audit_manifest": str(naturalness_shard.manifest_path),
        "naturalness_audit_sha256": naturalness_shard.sha256,
    }


def _confirmatory_index_record(
    store,
    config,
    full_manifest_sha256,
    rows,
    capabilities,
    power_shard,
    tokenizer_pin,
    naturalness_audit=None,
):
    return {
        "kind": "confirmatory_index",
        "config_sha256": config.config_hash,
        "full_manifest_sha256": full_manifest_sha256,
        "power_audit_manifest": str(power_shard.manifest_path.relative_to(store.root)),
        "power_audit_sha256": power_shard.sha256,
        "tokenizer_pin_manifest": str(tokenizer_pin.manifest_path.relative_to(store.root)),
        "tokenizer_pin_sha256": tokenizer_pin.sha256,
        "naturalness_audit_manifest": (
            None
            if naturalness_audit is None
            else str(naturalness_audit.manifest_path.relative_to(store.root))
        ),
        "naturalness_audit_sha256": (
            None if naturalness_audit is None else naturalness_audit.sha256
        ),
        "capabilities": {
            namespace: {
                "manifest_path": str(shard.manifest_path.relative_to(store.root)),
                "sha256": shard.sha256,
                "example_ids": [row.example_id for row in rows if row.split == namespace],
                "canonical_payload_sha256s": [
                    row.canonical_payload_sha256 for row in rows if row.split == namespace
                ],
            }
            for namespace, shard in sorted(capabilities.items())
        },
    }


def _audit_manifest(config: FAConfig, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(FAArtifactStore(root), args.manifest, config)
    factorial = tuple(row for row in manifest.examples if row.block == "factorial")
    same_string = tuple(row for row in manifest.examples if row.block == "same_string")
    audit = audit_dataset(factorial, same_string, tokenizer=_WhitespaceTokenizer())
    return {"status": "passed" if audit.passed else "failed", "checks": dict(audit.checks), "violations": list(audit.violations)}


def _run_generation(config: FAConfig, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.namespace in _PROTECTED:
        raise ValueError(
            "fa-run-generation is generic-only and cannot evaluate protected test namespaces"
        )
    store = FAArtifactStore(root)
    manifest = _load_manifest(store, args.manifest, config)
    if any(row.split != args.namespace for row in manifest.examples):
        raise ValueError("explicit generation manifest contains another namespace")
    runner = HFModelRunner(config)
    validate_runner_binding(
        runner,
        config,
        expected_chat_template_sha256=manifest.chat_template_sha256,
    )
    shard = run_generation_shard(runner, manifest, store, args.shard_id, config=config, namespace=args.namespace)
    records = _load_verified_generation_sidecar(
        store, shard.manifest_path, manifest, config
    )
    if not all(record["status"] == "completed" for record in records):
        exception_classes = sorted(
            {
                str(record.get("exception_class"))
                for record in records
                if record["status"] == "infrastructure_failure"
            }
        )
        return {
            "status": "infrastructure_failure",
            "error": {
                "message": "generation infrastructure failure; retry the existing artifact transaction",
                "type": ",".join(exception_classes) or "InfrastructureFailure",
            },
            "shard_manifest": str(shard.manifest_path),
            "sha256": shard.sha256,
        }
    return {
        "status": "generated",
        "shard_manifest": str(shard.manifest_path),
        "sha256": shard.sha256,
    }


def _extract_activations(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    if args.namespace in _PROTECTED:
        raise ValueError(
            "fa-extract-activations is generic-only and cannot evaluate protected test namespaces"
        )
    store = FAArtifactStore(root)
    manifest = _load_manifest(store, args.manifest, config)
    if manifest.namespace != args.namespace:
        raise ValueError("explicit activation manifest contains another namespace")
    layers = _registered_extraction_layers(config, args.layers)
    shard_id = _safe_cli_id(args.shard_id, "activation shard-id")
    model_runner = HFModelRunner(config)
    validate_runner_binding(
        model_runner,
        config,
        expected_chat_template_sha256=manifest.chat_template_sha256,
    )
    runner = _ACTIVATION_RUNNER_FACTORY(
        model_runner.model,
        model_runner.tokenizer,
        model_id=config.model_id,
        model_revision=config.model_revision,
        tokenizer_revision=config.tokenizer_revision,
    )
    destination = (
        root.absolute()
        / "runs"
        / "familiarity_answerability"
        / config.run_id
        / "activations"
        / args.namespace
        / f"{shard_id}.npz"
    )
    shard = _ACTIVATION_SHARD_WRITER(
        runner,
        manifest.examples,
        layers,
        destination=destination,
    )
    return {
        "status": "extracted",
        "manifest": str(shard.manifest_path),
        "request_sha256": shard.request_sha256,
        "row_count": shard.row_count,
    }


def _registered_extraction_layers(config: FAConfig, raw: str | None) -> tuple[int, ...]:
    registered = tuple(range(26))
    if config.profile == "confirmatory":
        if raw is not None and _parse_layer_ids(raw) != registered:
            raise ValueError("confirmatory extraction must use all 26 registered layers")
        return registered
    if raw is None:
        raise ValueError("smoke activation extraction requires explicit --layers")
    return _parse_layer_ids(raw)


def _parse_layer_ids(raw: str) -> tuple[int, ...]:
    if not isinstance(raw, str) or not raw:
        raise ValueError("activation layers must be a comma-separated integer sequence")
    try:
        layers = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise ValueError("activation layers must be a comma-separated integer sequence") from error
    if not layers or any(value < 0 for value in layers) or layers != tuple(sorted(set(layers))):
        raise ValueError("activation layers must be unique increasing nonnegative integers")
    return layers


def _safe_cli_id(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError(f"{field} is invalid")
    if value in {".", ".."} or ".." in value:
        raise ValueError(f"{field} is invalid")
    return value


def _fit_probes(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    store = FAArtifactStore(root)
    train_rows, train_shard = _load_probe_rows_manifest(
        store,
        args.train_rows_manifest,
        config,
        expected_namespace="mechanism_train",
    )
    validation_rows, validation_shard = _load_probe_rows_manifest(
        store,
        args.validation_rows_manifest,
        config,
        expected_namespace="locked_validation",
    )
    task_identities, prompt_shard = _load_prompt_task_source_identities(
        store,
        args.probe_test_manifest,
        config,
        expected_namespace="probe_test",
    )
    train_by_task = _probe_rows_by_task(train_rows)
    validation_by_task = _probe_rows_by_task(validation_rows)
    protected_ids = {
        identity.example_id
        for identities in task_identities.values()
        for identity in identities
    }
    selections = {
        task: _PROBE_SELECTOR(
            train_by_task[task],
            validation_by_task[task],
            protected_test_ids=protected_ids,
        )
        for task in TASKS
    }
    nulls: dict[str, tuple[NullSelectionResult, ...]] = {}
    for task in TASKS:
        kwargs: dict[str, Any] = {
            "protected_test_ids": protected_ids,
            "probe_test_source_identities": task_identities[task],
        }
        if config.profile == "smoke":
            kwargs.update(
                seeds=(2026072201,),
                _allow_test_seed_override=True,
            )
        nulls[task] = tuple(
            _PROBE_NULL_SELECTOR(
                train_by_task[task], validation_by_task[task], **kwargs
            )
        )
    selection_bundle_hash = f2a_selection_bundle_hash(selections)
    task_identities_sha256 = _task_source_identities_sha256(task_identities)
    mode = "registered_confirmatory" if config.profile == "confirmatory" else "smoke_rehearsal"
    row = {
        "kind": "selection_manifest",
        "schema_version": 1,
        "config_sha256": config.config_hash,
        "selection_bundle_hash": selection_bundle_hash,
        "probe_test_prompt_sha256": prompt_shard.sha256,
        "probe_test_task_identities_sha256": task_identities_sha256,
        "selection_mode": mode,
        "selections": {
            task: selections[task].to_record() for task in TASKS
        },
        "null_selections": {
            task: [result.to_record() for result in nulls[task]]
            for task in TASKS
        },
    }
    shard = store.write_completed_shard(
        config.run_id,
        "locked_validation",
        _safe_cli_id(args.shard_id, "selection shard-id"),
        (row,),
        {
            "config_sha256": config.config_hash,
            "train_probe_rows_sha256": train_shard.sha256,
            "validation_probe_rows_sha256": validation_shard.sha256,
            "probe_test_prompt_sha256": prompt_shard.sha256,
            "probe_test_task_identities_sha256": task_identities_sha256,
            "selection_bundle_hash": selection_bundle_hash,
            "selection_mode": mode,
        },
        record_kind="selection_manifest",
    )
    return {
        "status": "selected",
        "selection_manifest": str(shard.manifest_path),
        "selection_bundle_hash": selection_bundle_hash,
        "selection_mode": mode,
        "null_count": sum(len(values) for values in nulls.values()),
    }


def _seal_probe_selection(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    store = FAArtifactStore(root)
    selection = _load_f2a_selection_bundle(
        store, args.selection_manifest, config
    )
    task_identities, prompt_shard = _load_prompt_task_source_identities(
        store,
        args.probe_test_manifest,
        config,
        expected_namespace="probe_test",
    )
    if selection.probe_test_prompt_sha256 != prompt_shard.sha256:
        raise ValueError("selection bundle does not bind the probe_test prompt capability")
    if selection.probe_test_task_identities_sha256 != _task_source_identities_sha256(
        task_identities
    ):
        raise ValueError("selection bundle does not bind probe_test task identities")
    probe_rows_shard = _require_verified_shard_kind(
        store,
        args.probe_rows_manifest,
        "probe_rows",
        "verified protected probe rows manifest",
    )
    if probe_rows_shard.namespace != "probe_test":
        raise ValueError("protected probe rows must use the probe_test namespace")
    _verify_artifact_run_id(probe_rows_shard.manifest_path, config.run_id)
    _verify_shard_lineage(
        probe_rows_shard,
        {
            "config_sha256": config.config_hash,
            "prompt_manifest_sha256": prompt_shard.sha256,
            "task_source_identities_sha256": _task_source_identities_sha256(
                task_identities
            ),
        },
    )
    sealed_path = store.seal_endpoint(
        "probe_test",
        (prompt_shard, probe_rows_shard),
        {
            "preregistration": hashlib.sha256(
                _PREREGISTRATION_PATH.read_bytes()
            ).hexdigest(),
            "selection_manifest": selection.selection_bundle_hash,
        },
    )
    return {
        "status": "sealed",
        "endpoint": "probe_test",
        "endpoint_state": "sealed",
        "sealed_state": str(sealed_path),
        "selection_bundle_hash": selection.selection_bundle_hash,
    }


def _evaluate_probe_test(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    store = FAArtifactStore(root)
    selection = _load_f2a_selection_bundle(
        store, args.selection_manifest, config
    )
    task_identities, prompt_shard = _load_prompt_task_source_identities(
        store,
        args.probe_test_manifest,
        config,
        expected_namespace="probe_test",
    )
    if selection.probe_test_prompt_sha256 != prompt_shard.sha256:
        raise ValueError("selection bundle does not bind the probe_test prompt capability")
    if selection.probe_test_task_identities_sha256 != _task_source_identities_sha256(
        task_identities
    ):
        raise ValueError("selection bundle does not bind probe_test task identities")
    store.verify_endpoint_artifact("probe_test", prompt_shard.manifest_path)
    probe_rows_shard = store.verify_endpoint_artifact(
        "probe_test", args.probe_rows_manifest
    )
    if probe_rows_shard.record_kind != "probe_rows":
        raise ValueError("sealed probe test rows have the wrong record kind")
    state = store.endpoint_state("probe_test", prompt_shard.manifest_path)
    if state == "closed":
        closed = store.read_closed_metrics(
            "probe_test", prompt_shard.manifest_path
        )
        return {
            "status": "recovered",
            "endpoint_state": "closed",
            "metrics_manifest": str(closed.metrics_artifact.manifest_path),
            "selection_bundle_hash": selection.selection_bundle_hash,
        }
    receipt = store.unlock_or_resume_endpoint(
        "probe_test", prompt_shard.manifest_path
    )
    authorization = ProbeTestAuthorization.from_unlock_receipt(receipt)
    rows, _ = _load_probe_rows_manifest(
        store,
        probe_rows_shard.manifest_path,
        config,
        expected_namespace="probe_test",
        authorization=authorization,
        expected_prompt_sha256=prompt_shard.sha256,
        expected_task_identities=task_identities,
    )
    rows_by_task = _probe_rows_by_task(rows)
    bundle = _PROBE_BUNDLE_EVALUATOR(
        selection.selections,
        authorization,
        rows_by_task,
        null_selections_by_task=selection.null_selections,
        store=store,
        endpoint_manifest_path=prompt_shard.manifest_path,
    )
    closed = store.read_closed_metrics("probe_test", prompt_shard.manifest_path)
    return {
        "status": "evaluated",
        "endpoint_state": "closed",
        "metrics_manifest": str(closed.metrics_artifact.manifest_path),
        "selection_bundle_hash": selection.selection_bundle_hash,
        "bundle_sha256": bundle.sha256,
        "gate_status": bundle.gates.status,
    }


def _load_probe_rows_manifest(
    store: FAArtifactStore,
    path: str | Path,
    config: FAConfig,
    *,
    expected_namespace: str,
    authorization: ProbeTestAuthorization | None = None,
    expected_prompt_sha256: str | None = None,
    expected_task_identities: Mapping[
        str, Sequence[ProbeSourceIdentity]
    ] | None = None,
) -> tuple[tuple[ProbeRow, ...], Any]:
    if expected_namespace == "probe_test" and not isinstance(
        authorization, ProbeTestAuthorization
    ):
        raise ValueError("protected probe rows require a probe_test authorization")
    shard = _require_verified_shard_kind(
        store, path, "probe_rows", "verified probe rows manifest"
    )
    if shard.namespace != expected_namespace:
        raise ValueError("probe rows artifact uses the wrong namespace")
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    expected_lineage: dict[str, Any] = {"config_sha256": config.config_hash}
    if expected_prompt_sha256 is not None:
        expected_lineage["prompt_manifest_sha256"] = expected_prompt_sha256
    if expected_task_identities is not None:
        expected_lineage["task_source_identities_sha256"] = (
            _task_source_identities_sha256(expected_task_identities)
        )
    _verify_shard_lineage(shard, expected_lineage)
    raw_rows = _read_json_rows(shard.data_path)
    if any(
        set(record) != {"kind", "row"} or record.get("kind") != "probe_rows"
        for record in raw_rows
    ):
        raise ValueError("probe rows artifact has an invalid wrapper schema")
    rows = tuple(ProbeRow.from_record(record["row"]) for record in raw_rows)
    by_task = _probe_rows_by_task(rows)
    if expected_task_identities is not None:
        for task in TASKS:
            observed = tuple(
                sorted(
                    (ProbeSourceIdentity.from_row(row) for row in by_task[task]),
                    key=lambda item: (
                        item.example_id,
                        item.canonical_payload_sha256,
                    ),
                )
            )
            if observed != tuple(expected_task_identities[task]):
                raise ValueError(
                    f"{task} probe rows do not match the sealed task identities"
                )
    return rows, shard


def _probe_rows_by_task(
    rows: Sequence[ProbeRow],
) -> Mapping[str, tuple[ProbeRow, ...]]:
    values = tuple(rows)
    if not values:
        raise ValueError("probe rows artifact is empty")
    grouped = {
        task: tuple(
            sorted(
                (row for row in values if row.task == task),
                key=lambda row: (row.example_id, row.sha256),
            )
        )
        for task in TASKS
    }
    if any(not grouped[task] for task in TASKS):
        raise ValueError("probe rows artifact must contain every registered task")
    if len({(row.task, row.example_id) for row in values}) != len(values):
        raise ValueError("probe rows artifact contains duplicate task/example IDs")
    familiarity = {
        row.example_id: row.source_sha256 for row in grouped["familiarity"]
    }
    answerability = {
        row.example_id: row.source_sha256 for row in grouped["answerability"]
    }
    unsupported = {
        row.example_id: row.source_sha256
        for row in grouped["unsupported_answer"]
    }
    if familiarity != answerability or not set(unsupported.items()).issubset(
        familiarity.items()
    ):
        raise ValueError("probe task rows do not share a coherent source identity set")
    return grouped


def _task_source_identities_sha256(
    values: Mapping[str, Sequence[ProbeSourceIdentity]],
) -> str:
    if set(values) != set(TASKS):
        raise ValueError("task source identities require every registered task")
    return _sha256_json(
        {
            task: [identity.to_record() for identity in values[task]]
            for task in TASKS
        }
    )


def _load_f2a_selection_bundle(
    store: FAArtifactStore,
    path: str | Path,
    config: FAConfig,
) -> VerifiedF2ASelectionBundle:
    shard = _require_verified_shard_kind(
        store, path, "selection_manifest", "verified F2A selection manifest"
    )
    if shard.namespace != "locked_validation":
        raise ValueError("F2A selection manifest must use locked_validation")
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    records = _read_json_rows(shard.data_path)
    expected = {
        "kind",
        "schema_version",
        "config_sha256",
        "selection_bundle_hash",
        "probe_test_prompt_sha256",
        "probe_test_task_identities_sha256",
        "selection_mode",
        "selections",
        "null_selections",
    }
    if len(records) != 1 or set(records[0]) != expected:
        raise ValueError("F2A selection manifest has an invalid schema")
    record = records[0]
    if (
        record["kind"] != "selection_manifest"
        or record["schema_version"] != 1
        or record["config_sha256"] != config.config_hash
    ):
        raise ValueError("F2A selection manifest identity is invalid")
    if not isinstance(record["selections"], dict) or set(
        record["selections"]
    ) != set(TASKS):
        raise ValueError("F2A selection manifest requires every registered task")
    if not isinstance(record["null_selections"], dict) or set(
        record["null_selections"]
    ) != set(TASKS):
        raise ValueError("F2A null selections require every registered task")
    selections = {
        task: SelectionManifest.from_record(record["selections"][task])
        for task in TASKS
    }
    nulls: dict[str, tuple[NullSelectionResult, ...]] = {}
    for task in TASKS:
        raw = record["null_selections"][task]
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"F2A {task} null selections are missing")
        loaded = tuple(NullSelectionResult.from_record(item) for item in raw)
        if any(item.selection.task != task or item.test_metrics is not None for item in loaded):
            raise ValueError("F2A null selections are task-mismatched or already scored")
        nulls[task] = loaded
    bundle_hash = f2a_selection_bundle_hash(selections)
    if record["selection_bundle_hash"] != bundle_hash:
        raise ValueError("F2A selection bundle hash does not verify")
    mode = record["selection_mode"]
    expected_count = 396 if mode == "registered_confirmatory" else 4
    if mode not in {"registered_confirmatory", "smoke_rehearsal"} or any(
        len(nulls[task]) != expected_count for task in TASKS
    ):
        raise ValueError("F2A null selection count does not match its mode")
    prompt_sha256 = record["probe_test_prompt_sha256"]
    _required_sha256(prompt_sha256, "probe_test_prompt_sha256")
    task_identities_sha256 = record["probe_test_task_identities_sha256"]
    _required_sha256(
        task_identities_sha256, "probe_test_task_identities_sha256"
    )
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config.config_hash,
            "probe_test_prompt_sha256": prompt_sha256,
            "probe_test_task_identities_sha256": task_identities_sha256,
            "selection_bundle_hash": bundle_hash,
            "selection_mode": mode,
        },
    )
    return VerifiedF2ASelectionBundle(
        selections=selections,
        null_selections=nulls,
        selection_bundle_hash=bundle_hash,
        probe_test_prompt_sha256=prompt_sha256,
        probe_test_task_identities_sha256=task_identities_sha256,
        shard_manifest_path=shard.manifest_path,
        shard_sha256=shard.sha256,
    )


def _evaluate_behavior_test(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    store = FAArtifactStore(root)
    manifest = _load_manifest(store, args.manifest, config)
    if manifest.namespace != "behavior_test":
        raise ValueError("behavior evaluation requires the protected behavior_test manifest")
    state = store.endpoint_state("behavior_test", manifest.shard_manifest_path)
    if state == "closed":
        raise ValueError("behavior_test is already closed")
    if state == "evaluated":
        store.close_endpoint("behavior_test")
        return {"status": "recovered", "endpoint_state": "closed"}
    receipt = store.unlock_or_resume_endpoint(
        "behavior_test", manifest.shard_manifest_path
    )
    runner = HFModelRunner(config)
    validate_runner_binding(
        runner,
        config,
        expected_chat_template_sha256=manifest.chat_template_sha256,
    )
    generation = run_generation_shard(
        runner,
        manifest,
        store,
        _safe_cli_id(args.shard_id, "behavior shard-id"),
        config=config,
        namespace="behavior_test",
    )
    records = _load_verified_generation_sidecar(
        store, generation.manifest_path, manifest, config
    )
    if not all(record["status"] == "completed" for record in records):
        return {
            "status": "infrastructure_failure",
            "endpoint_state": "unlocked_once",
            "generation_manifest": str(generation.manifest_path),
        }
    scored = _score_rows(records, manifest)
    metrics = estimate_behavior(scored)
    bootstrap = _BEHAVIOR_BOOTSTRAP(
        scored,
        replicates=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )
    gate = _BEHAVIOR_GATE(
        metrics,
        bootstrap,
        thresholds=config.thresholds,
        same_string_sealed=any(row.block == "same_string" for row in manifest.examples),
        config_hash=config.config_hash,
        manifest_hash=manifest.full_manifest_sha256,
    )
    evidence = {
        "metrics": _json_safe(metrics.to_record()),
        "bootstrap": _json_safe(bootstrap.to_record()),
        "gate": _json_safe(gate.to_record()),
        "scored_rows": [_json_safe(row.to_record()) for row in scored],
    }
    row = {
        "kind": "metrics",
        "phase": "F1",
        **evidence,
        "evidence_sha256": _sha256_json(evidence),
    }
    lineage = {
        "config_sha256": config.config_hash,
        "preregistration_sha256": receipt.preregistration_hash,
        "selection_sha256": receipt.selection_manifest_hash,
        "prompt_manifest_sha256": manifest.shard_sha256,
        "generation_manifest_sha256": generation.sha256,
        "evidence_sha256": row["evidence_sha256"],
    }
    metrics_shard = _write_or_resume_metrics(
        store,
        config.run_id,
        "behavior_test",
        f"behavior-metrics-{receipt.lease_id}",
        row,
        lineage,
    )
    store.mark_evaluated(receipt, metrics_shard.data_path)
    store.close_endpoint("behavior_test")
    return {
        "status": "evaluated",
        "endpoint_state": "closed",
        "metrics_manifest": str(metrics_shard.manifest_path),
        "evidence_sha256": row["evidence_sha256"],
    }


def _write_or_resume_metrics(
    store: FAArtifactStore,
    run_id: str,
    namespace: str,
    shard_id: str,
    row: dict[str, Any],
    lineage: dict[str, Any],
):
    try:
        return store.write_completed_shard(
            run_id,
            namespace,
            shard_id,
            (row,),
            lineage,
            record_kind="metrics",
        )
    except FileExistsError:
        candidate = (
            store.root
            / "runs"
            / "familiarity_answerability"
            / run_id
            / "shards"
            / namespace
            / f"{shard_id}.jsonl.manifest.json"
        )
        existing = store.verify_shard(candidate)
        expected = json.dumps(
            row, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        if (
            existing.record_kind != "metrics"
            or existing.row_count != 1
            or existing.sha256 != hashlib.sha256(expected).hexdigest()
        ):
            raise ValueError("existing metrics artifact does not match this evaluation")
        sidecar = _read_json_object(existing.manifest_path)
        if sidecar.get("lineage") != lineage:
            raise ValueError("existing metrics lineage does not match this evaluation")
        return existing


def _score_behavior(config: FAConfig, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    store = FAArtifactStore(root)
    manifest = _load_manifest(store, args.manifest, config)
    generation_shard = _require_generation_sidecar_manifest(
        store, args.generation_manifest
    )
    records = _load_verified_generation_sidecar(
        store, generation_shard.manifest_path, manifest, config
    )
    rows = _score_rows(records, manifest)
    metrics = estimate_behavior(rows)
    gate = _pilot_gate(rows, metrics)
    metrics_record = _json_safe(metrics.to_record())
    evidence_sha256 = _sha256_json(
        {"metrics": metrics_record, "pilot_gate": gate}
    )
    gate_shard = store.write_completed_shard(
        config.run_id,
        generation_shard.namespace,
        f"pilot-gate-{generation_shard.sha256[:16]}",
        [
            {
                "kind": "pilot_gate",
                "config_sha256": config.config_hash,
                "source_manifest_sha256": manifest.manifest_sha256,
                "prompt_manifest": str(
                    manifest.shard_manifest_path.relative_to(store.root)
                ),
                "prompt_manifest_sha256": manifest.shard_sha256,
                "tokenizer_pin_manifest": str(
                    manifest.tokenizer_pin_manifest_path.relative_to(store.root)
                ),
                "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
                "chat_template_sha256": manifest.chat_template_sha256,
                "generation_sidecar_manifest": str(
                    generation_shard.manifest_path.relative_to(store.root)
                ),
                "generation_sidecar_sha256": generation_shard.sha256,
                "pilot_gate": gate,
                "metrics": metrics_record,
                "evidence_sha256": evidence_sha256,
            }
        ],
        {
            "config_sha256": config.config_hash,
            "source_manifest_sha256": manifest.manifest_sha256,
            "generation_sidecar_sha256": generation_shard.sha256,
            "prompt_manifest_sha256": manifest.shard_sha256,
            "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
            "chat_template_sha256": manifest.chat_template_sha256,
        },
        record_kind="pilot_gate",
    )
    return {
        "status": "scored",
        "metrics": metrics_record,
        "pilot_gate": gate,
        "pilot_gate_manifest": str(gate_shard.manifest_path),
    }


def _score_rows(records: tuple[dict[str, Any], ...], manifest: Any):
    examples = {row.example_id: row for row in manifest.examples}
    codes = frozenset(row.registry_code for row in manifest.examples)
    scored = []
    for record in records:
        example_record = record.get("example")
        example_id = record.get("example_id") or (example_record or {}).get("example_id")
        example = examples.get(example_id)
        if example is None:
            raise ValueError("generation record does not bind a manifest example")
        status = record.get("status")
        scored.append(score_response(
            example, record.get("raw_output"), registered_codes=codes or {example.registry_code},
            infrastructure_marked=status != "completed",
        ))
    if not scored:
        raise ValueError("generation manifest contains no records")
    return tuple(scored)


def _pilot_gate(rows: tuple[Any, ...], metrics: Any) -> dict[str, Any]:
    target = [row for row in rows if row.answerability == "target_bound"]
    accuracy = sum(row.outcome.value == "exact_target_code" for row in target) / len(target) if target else 0.0
    reasons = []
    if accuracy < 0.70:
        reasons.append("target_bound_accuracy_below_70_percent")
    for cell, denominator in metrics.denominators.items():
        if denominator and metrics.invalid_format_counts[cell] / denominator > 0.05:
            reasons.append(f"invalid_format_above_5_percent:{':'.join(cell)}")
    absent = [row for row in rows if row.answerability in {"distractor_bound", "code_absent"}]
    if absent:
        rate = sum(row.answer_attempt for row in absent) / len(absent)
        if rate <= 0.05 or rate >= 0.95:
            reasons.append("absent_answer_attempt_at_registered_floor_or_ceiling")
    else:
        rate = None
        reasons.append("absent_answer_attempt_at_registered_floor_or_ceiling")
    return {"status": "passed" if not reasons else "blocked", "reasons": reasons, "target_bound_accuracy": accuracy, "absent_answer_attempt_rate": rate}


def _load_manifest(
    store: FAArtifactStore, path: str | Path, config: FAConfig | None = None
) -> VerifiedPromptManifest:
    shard = _require_verified_shard_kind(
        store, path, "prompt_manifest", "verified prompt manifest"
    )
    rows = _read_json_rows(shard.data_path)
    if len(rows) != 1:
        raise ValueError("prompt manifest must contain exactly one capability record")
    value = rows[0]
    required = {
        "kind",
        "config_hash",
        "full_manifest_sha256",
        "subset_manifest_sha256",
        "chat_template_sha256",
        "namespace",
        "model_sha256",
        "tokenizer_sha256",
        "tokenizer_pin_manifest",
        "tokenizer_pin_sha256",
        "naturalness_audit_manifest",
        "naturalness_audit_sha256",
        "generation",
        "examples",
    }
    if set(value) != required or value.get("kind") != "prompt_manifest":
        raise ValueError("prompt manifest has an invalid schema")
    config_hash = _required_sha256(value.get("config_hash"), "prompt config_hash")
    if config is not None and config_hash != config.config_hash:
        raise ValueError("explicit manifest config hash does not match config")
    namespace = value.get("namespace")
    if namespace != shard.namespace or namespace not in _GENERATION_NAMESPACES:
        raise ValueError("prompt manifest namespace does not match its immutable shard")
    raw_examples = value.get("examples")
    if not isinstance(raw_examples, list) or not raw_examples:
        raise ValueError("prompt manifest must contain canonical examples")
    try:
        examples = tuple(FAExample(**row) for row in raw_examples)
    except (TypeError, ValueError) as error:
        raise ValueError(f"prompt manifest contains an invalid FAExample: {error}") from error
    if any(_record_value(example) != raw for example, raw in zip(examples, raw_examples, strict=True)):
        raise ValueError("prompt manifest FAExample payload is not canonical")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("prompt manifest contains duplicate example IDs")
    if any(example.split != namespace for example in examples):
        raise ValueError("prompt manifest contains another namespace")
    template_hash = _required_sha256(
        value.get("chat_template_sha256"), "prepared chat_template_sha256"
    )
    if config is not None and config.chat_template_sha256 and template_hash != config.chat_template_sha256:
        raise ValueError("manifest chat template pin does not match config")
    full_hash = _required_sha256(
        value.get("full_manifest_sha256"), "full_manifest_sha256"
    )
    if config is None:
        raise ValueError("prompt verification requires an immutable FA config")
    tokenizer_pin_path = _artifact_path_from_record(
        store, value.get("tokenizer_pin_manifest"), "tokenizer pin manifest"
    )
    tokenizer_pin_sha256 = _required_sha256(
        value.get("tokenizer_pin_sha256"), "tokenizer_pin_sha256"
    )
    _load_verified_tokenizer_pin(
        store,
        tokenizer_pin_path,
        config,
        expected_sha256=tokenizer_pin_sha256,
        expected_source_manifest_sha256=full_hash,
        expected_chat_template_sha256=template_hash,
    )
    naturalness_audit_path = None
    naturalness_audit_sha256 = None
    if config.profile == "confirmatory":
        naturalness_audit_path = _artifact_path_from_record(
            store,
            value.get("naturalness_audit_manifest"),
            "naturalness audit manifest",
        )
        naturalness_audit_sha256 = _required_sha256(
            value.get("naturalness_audit_sha256"),
            "naturalness_audit_sha256",
        )
        _load_verified_naturalness_audit(
            store,
            naturalness_audit_path,
            config,
            expected_sha256=naturalness_audit_sha256,
            expected_pair_ids=frozenset(
                example.entity_unit_id for example in examples
            ),
        )
    elif (
        value.get("naturalness_audit_manifest") is not None
        or value.get("naturalness_audit_sha256") is not None
    ):
        raise ValueError("smoke prompt manifests cannot claim a naturalness audit")
    subset_hash = _prompt_subset_sha256(
        config_hash,
        full_hash,
        namespace,
        template_hash,
        tokenizer_pin_sha256,
        naturalness_audit_sha256,
        examples,
    )
    if value.get("subset_manifest_sha256") != subset_hash:
        raise ValueError("prompt subset manifest identity does not verify")
    model_hash = _required_sha256(value.get("model_sha256"), "prompt model_sha256")
    tokenizer_hash = _required_sha256(
        value.get("tokenizer_sha256"), "prompt tokenizer_sha256"
    )
    generation = value.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("prompt generation config is invalid")
    if config is not None:
        expected_model, expected_tokenizer = _config_runtime_hashes(config)
        if (model_hash, tokenizer_hash, generation) != (
            expected_model,
            expected_tokenizer,
            dict(config.generation),
        ):
            raise ValueError("prompt runtime pins do not match config")
    task_source_identities = _task_source_identity_records(examples)
    source_identities = task_source_identities["familiarity"]
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config_hash,
            "source_manifest_sha256": full_hash,
            "subset_manifest_sha256": subset_hash,
            "chat_template_sha256": template_hash,
            "tokenizer_pin_sha256": tokenizer_pin_sha256,
            "naturalness_audit_sha256": naturalness_audit_sha256,
            "source_identities": source_identities,
            "source_identities_sha256": _sha256_json(source_identities),
            "task_source_identities": task_source_identities,
            "task_source_identities_sha256": _sha256_json(
                task_source_identities
            ),
        },
    )
    return VerifiedPromptManifest(
        config_hash=config_hash,
        manifest_sha256=subset_hash,
        full_manifest_sha256=full_hash,
        chat_template_sha256=template_hash,
        namespace=namespace,
        examples=examples,
        model_sha256=model_hash,
        tokenizer_sha256=tokenizer_hash,
        tokenizer_pin_manifest_path=tokenizer_pin_path,
        tokenizer_pin_sha256=tokenizer_pin_sha256,
        naturalness_audit_manifest_path=naturalness_audit_path,
        naturalness_audit_sha256=naturalness_audit_sha256,
        generation=dict(generation),
        shard_manifest_path=shard.manifest_path,
        shard_sha256=shard.sha256,
    )


def _load_prompt_source_identities(
    store: FAArtifactStore,
    path: str | Path,
    config: FAConfig,
    *,
    expected_namespace: str,
) -> tuple[tuple[ProbeSourceIdentity, ...], Any]:
    """Load only pre-outcome IDs and hashes from a prompt capability sidecar."""

    task_identities, shard = _load_prompt_task_source_identities(
        store, path, config, expected_namespace=expected_namespace
    )
    return task_identities["familiarity"], shard


def _load_prompt_task_source_identities(
    store: FAArtifactStore,
    path: str | Path,
    config: FAConfig,
    *,
    expected_namespace: str,
) -> tuple[Mapping[str, tuple[ProbeSourceIdentity, ...]], Any]:
    """Load task-scoped identities without parsing protected prompt contents."""

    shard = _require_verified_shard_kind(
        store, path, "prompt_manifest", "verified prompt identity capability"
    )
    if shard.namespace != expected_namespace:
        raise ValueError("prompt identity capability uses the wrong namespace")
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    sidecar = _read_json_object(shard.manifest_path)
    lineage = sidecar.get("lineage")
    if not isinstance(lineage, dict) or lineage.get("config_sha256") != config.config_hash:
        raise ValueError("prompt identity capability does not bind the config")
    raw = lineage.get("task_source_identities")
    if not isinstance(raw, dict) or set(raw) != set(TASKS):
        raise ValueError("prompt identity capability has no task-scoped identities")
    task_identities: dict[str, tuple[ProbeSourceIdentity, ...]] = {}
    for task in TASKS:
        records = raw.get(task)
        if not isinstance(records, list) or not records:
            raise ValueError(f"prompt identity capability has no {task} identities")
        identities = tuple(ProbeSourceIdentity.from_record(item) for item in records)
        canonical = tuple(
            sorted(
                identities,
                key=lambda item: (item.example_id, item.canonical_payload_sha256),
            )
        )
        if identities != canonical or len(set(identities)) != len(identities):
            raise ValueError(
                "prompt source identities must be unique and canonically ordered"
            )
        task_identities[task] = identities
    canonical_record = {
        task: [item.to_record() for item in task_identities[task]] for task in TASKS
    }
    if lineage.get("task_source_identities_sha256") != _sha256_json(
        canonical_record
    ):
        raise ValueError("prompt task source identity hash does not verify")
    if task_identities["familiarity"] != task_identities["answerability"]:
        raise ValueError("familiarity and answerability identities must match")
    if not set(task_identities["unsupported_answer"]).issubset(
        task_identities["familiarity"]
    ):
        raise ValueError("unsupported-answer identities must be a strict task subset")
    return task_identities, shard


def _task_source_identity_records(
    examples: Sequence[Any],
) -> dict[str, list[dict[str, str]]]:
    ordered = tuple(sorted(examples, key=lambda example: example.example_id))
    source = [
        {
            "example_id": example.example_id,
            "canonical_payload_sha256": example.canonical_payload_sha256,
        }
        for example in ordered
    ]
    return {
        "familiarity": source,
        "answerability": source,
        "unsupported_answer": [
            identity
            for identity, example in zip(source, ordered, strict=True)
            if example.answerability in {"distractor_bound", "code_absent"}
        ],
    }


def _write_prompt_capability(
    store: FAArtifactStore,
    config: FAConfig,
    full_manifest_sha256: str,
    namespace: str,
    examples: tuple[FAExample, ...],
    chat_template_sha256: str,
    tokenizer_pin,
    naturalness_audit=None,
):
    if config.profile == "confirmatory" and naturalness_audit is None:
        raise ValueError("confirmatory prompt capability requires a naturalness audit")
    if config.profile == "smoke" and naturalness_audit is not None:
        raise ValueError("smoke prompt capability cannot claim a naturalness audit")
    ordered = tuple(sorted(examples, key=lambda row: row.example_id))
    template_hash = _required_sha256(
        chat_template_sha256, "prepared chat_template_sha256"
    )
    subset_hash = _prompt_subset_sha256(
        config.config_hash,
        full_manifest_sha256,
        namespace,
        template_hash,
        tokenizer_pin.sha256,
        None if naturalness_audit is None else naturalness_audit.sha256,
        ordered,
    )
    model_hash, tokenizer_hash = _config_runtime_hashes(config)
    task_source_identities = _task_source_identity_records(ordered)
    source_identities = task_source_identities["familiarity"]
    row = {
        "kind": "prompt_manifest",
        "config_hash": config.config_hash,
        "full_manifest_sha256": full_manifest_sha256,
        "subset_manifest_sha256": subset_hash,
        "chat_template_sha256": template_hash,
        "namespace": namespace,
        "model_sha256": model_hash,
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_pin_manifest": str(
            tokenizer_pin.manifest_path.relative_to(store.root)
        ),
        "tokenizer_pin_sha256": tokenizer_pin.sha256,
        "naturalness_audit_manifest": (
            None
            if naturalness_audit is None
            else str(naturalness_audit.manifest_path.relative_to(store.root))
        ),
        "naturalness_audit_sha256": (
            None if naturalness_audit is None else naturalness_audit.sha256
        ),
        "generation": dict(config.generation),
        "examples": [_record_value(example) for example in ordered],
    }
    return store.write_completed_shard(
        config.run_id,
        namespace,
        f"prompt-capability-{subset_hash[:16]}",
        [row],
        {
            "config_sha256": config.config_hash,
            "source_manifest_sha256": full_manifest_sha256,
            "subset_manifest_sha256": subset_hash,
            "chat_template_sha256": template_hash,
            "tokenizer_pin_sha256": tokenizer_pin.sha256,
            "naturalness_audit_sha256": (
                None if naturalness_audit is None else naturalness_audit.sha256
            ),
            "source_identities": source_identities,
            "source_identities_sha256": _sha256_json(source_identities),
            "task_source_identities": task_source_identities,
            "task_source_identities_sha256": _sha256_json(
                task_source_identities
            ),
        },
        record_kind="prompt_manifest",
    )


def _prompt_subset_sha256(
    config_hash,
    full_hash,
    namespace,
    template_hash,
    tokenizer_pin_sha256,
    naturalness_audit_sha256,
    examples,
):
    return _sha256_json(
        {
            "config_hash": config_hash,
            "full_manifest_sha256": full_hash,
            "namespace": namespace,
            "chat_template_sha256": template_hash,
            "tokenizer_pin_sha256": tokenizer_pin_sha256,
            "naturalness_audit_sha256": naturalness_audit_sha256,
            "examples": [_record_value(row) for row in examples],
        }
    )


def _write_naturalness_audit(
    store, config, matches, audit, ratings, ratings_shard
):
    row = {
        "kind": "naturalness_audit",
        "config_sha256": config.config_hash,
        "ratings_manifest": str(ratings_shard.manifest_path.relative_to(store.root)),
        "ratings_sha256": ratings_shard.sha256,
        "matches": [_record_value(value) for value in sorted(matches, key=lambda value: value.pair_id)],
        "ratings": [
            _record_value(value)
            for value in sorted(
                ratings,
                key=lambda value: (value.pair_id, value.round, value.rater_id),
            )
        ],
        "accepted_pair_ids": list(audit.accepted_pair_ids),
        "excluded_pair_ids": list(audit.excluded_pair_ids),
        "third_rater_pair_ids": list(audit.third_rater_pair_ids),
        "decisions": dict(audit.decisions),
    }
    audit_hash = _sha256_json(row)
    return store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        f"naturalness-audit-{audit_hash[:16]}",
        [row],
        {
            "config_sha256": config.config_hash,
            "audit_sha256": audit_hash,
            "ratings_sha256": ratings_shard.sha256,
        },
        record_kind="naturalness_audit",
    )


def _load_verified_naturalness_audit(
    store,
    path,
    config,
    *,
    expected_sha256,
    expected_pair_ids,
):
    shard = _require_verified_shard_kind(
        store, path, "naturalness_audit", "verified naturalness audit manifest"
    )
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    if shard.namespace != "mechanism_train" or shard.sha256 != expected_sha256:
        raise ValueError("naturalness audit identity does not verify")
    rows = _read_json_rows(shard.data_path)
    if len(rows) != 1 or rows[0].get("kind") != "naturalness_audit":
        raise ValueError("naturalness audit has an invalid schema")
    row = rows[0]
    required = {
        "kind",
        "config_sha256",
        "ratings_manifest",
        "ratings_sha256",
        "matches",
        "ratings",
        "accepted_pair_ids",
        "excluded_pair_ids",
        "third_rater_pair_ids",
        "decisions",
    }
    if set(row) != required or row.get("config_sha256") != config.config_hash:
        raise ValueError("naturalness audit has an invalid schema or config")
    ratings_path = _artifact_path_from_record(
        store, row.get("ratings_manifest"), "naturalness ratings manifest"
    )
    expected_ratings_sha256 = _required_sha256(
        row.get("ratings_sha256"), "naturalness ratings sha256"
    )
    verified_ratings, ratings_shard = _load_verified_naturalness_ratings(
        store, ratings_path, config
    )
    if ratings_shard.sha256 != expected_ratings_sha256:
        raise ValueError("naturalness ratings identity does not verify")
    try:
        matches = tuple(EntityMatch(**value) for value in row["matches"])
        ratings = tuple(NaturalnessRating(**value) for value in row["ratings"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("naturalness audit evidence is invalid") from error
    recomputed = audit_naturalness_manifest(matches, ratings)
    if ratings != verified_ratings:
        raise ValueError("naturalness audit ratings differ from their source artifact")
    recorded = {
        "accepted_pair_ids": list(recomputed.accepted_pair_ids),
        "excluded_pair_ids": list(recomputed.excluded_pair_ids),
        "third_rater_pair_ids": list(recomputed.third_rater_pair_ids),
        "decisions": dict(recomputed.decisions),
    }
    if any(row[name] != value for name, value in recorded.items()):
        raise ValueError("naturalness audit does not match deterministic recomputation")
    accepted = frozenset(recomputed.accepted_pair_ids)
    if not expected_pair_ids <= accepted:
        raise ValueError("naturalness audit does not accept every prompt entity unit")
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config.config_hash,
            "audit_sha256": _sha256_json(row),
            "ratings_sha256": expected_ratings_sha256,
        },
    )
    return shard


def _load_verified_naturalness_ratings(store, path, config):
    shard = _require_verified_shard_kind(
        store,
        path,
        "naturalness_ratings",
        "verified naturalness ratings manifest",
    )
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    if shard.namespace != "mechanism_train":
        raise ValueError("naturalness ratings must use mechanism_train namespace")
    rows = _read_json_rows(shard.data_path)
    required = {
        "kind",
        "schema_version",
        "config_sha256",
        "protocol_sha256",
        "blinding_manifest_sha256",
        "assignments",
        "ratings",
    }
    if len(rows) != 1 or set(rows[0]) != required:
        raise ValueError("naturalness ratings have an invalid schema")
    row = rows[0]
    if row.get("kind") != "naturalness_ratings" or row.get("schema_version") != 1:
        raise ValueError("naturalness ratings schema_version must equal one")
    if row.get("config_sha256") != config.config_hash:
        raise ValueError("naturalness ratings config does not match")
    protocol_sha256 = hashlib.sha256(_PREREGISTRATION_PATH.read_bytes()).hexdigest()
    if row.get("protocol_sha256") != protocol_sha256:
        raise ValueError("naturalness ratings protocol hash does not match preregistration")
    assignments = row.get("assignments")
    raw_ratings = row.get("ratings")
    if not isinstance(assignments, list) or not isinstance(raw_ratings, list):
        raise ValueError("naturalness ratings evidence must be lists")
    if row.get("blinding_manifest_sha256") != _sha256_json(assignments):
        raise ValueError("naturalness ratings blinding manifest does not verify")
    assignment_keys = set()
    submissions = set()
    slots_by_pair = {}
    for assignment in assignments:
        if not isinstance(assignment, dict) or set(assignment) != {
            "pair_id",
            "rater_id",
            "blind_slot",
            "submission_sha256",
        }:
            raise ValueError("naturalness rating assignment has an invalid schema")
        key = (assignment["pair_id"], assignment["rater_id"])
        if key in assignment_keys:
            raise ValueError("naturalness rating assignments must be unique")
        assignment_keys.add(key)
        submission = _required_sha256(
            assignment.get("submission_sha256"), "rating submission sha256"
        )
        if submission in submissions:
            raise ValueError("rating submissions must be unique")
        submissions.add(submission)
        slot = assignment.get("blind_slot")
        if slot not in {"slot-a", "slot-b", "adjudicator"}:
            raise ValueError("naturalness rating blind slot is invalid")
        slots_by_pair.setdefault(assignment["pair_id"], set()).add(slot)
    try:
        ratings = tuple(NaturalnessRating(**value) for value in raw_ratings)
    except (TypeError, ValueError) as error:
        raise ValueError(f"naturalness rating schema_version is invalid: {error}") from error
    rating_keys = {(rating.pair_id, rating.rater_id) for rating in ratings}
    if len(rating_keys) != len(ratings) or rating_keys != assignment_keys:
        raise ValueError("naturalness ratings must match blinded assignments exactly")
    for pair_id, pair_ratings in _group_ratings(ratings).items():
        initial = tuple(value for value in pair_ratings if value.round == 1)
        adjudicators = tuple(value for value in pair_ratings if value.round == 2)
        expected_slots = {"slot-a", "slot-b"} | (
            {"adjudicator"} if adjudicators else set()
        )
        if len(initial) != 2 or len(adjudicators) > 1 or slots_by_pair[pair_id] != expected_slots:
            raise ValueError("naturalness ratings do not prove the registered blind assignment")
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config.config_hash,
            "protocol_sha256": protocol_sha256,
            "blinding_manifest_sha256": row["blinding_manifest_sha256"],
        },
    )
    return tuple(sorted(ratings, key=lambda value: (value.pair_id, value.round, value.rater_id))), shard


def _group_ratings(ratings):
    grouped = {}
    for rating in ratings:
        grouped.setdefault(rating.pair_id, []).append(rating)
    return grouped


def _record_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        result = value
    elif hasattr(value, "__dataclass_fields__"):
        result = asdict(value)
    elif hasattr(value, "__dict__"):
        result = dict(value.__dict__)
    else:
        raise ValueError("manifest example must be a serializable record")
    return json.loads(json.dumps(result, sort_keys=True, default=list))


def _write_tokenizer_pin(store, config, prepared, source_manifest_sha256):
    return store.write_completed_shard(
        config.run_id,
        "mechanism_train" if config.profile == "confirmatory" else "pilot",
        f"tokenizer-pin-{prepared.chat_template_sha256[:16]}",
        [
            {
                "kind": "tokenizer_pin",
                "model_id": config.model_id,
                "model_revision": config.model_revision,
                "tokenizer_revision": config.tokenizer_revision,
                "chat_template_sha256": prepared.chat_template_sha256,
                "chat_template_utf8_hex": prepared.chat_template_bytes.hex(),
            }
        ],
        {
            "config_sha256": config.config_hash,
            "source_manifest_sha256": _required_sha256(
                source_manifest_sha256, "source manifest sha256"
            ),
            "chat_template_sha256": prepared.chat_template_sha256,
        },
        record_kind="tokenizer_pin",
    )


def _load_verified_tokenizer_pin(
    store: FAArtifactStore,
    path: str | Path,
    expected_config: FAConfig,
    *,
    expected_sha256: str,
    expected_source_manifest_sha256: str,
    expected_chat_template_sha256: str,
):
    shard = _require_verified_shard_kind(
        store, path, "tokenizer_pin", "verified tokenizer pin manifest"
    )
    _verify_artifact_run_id(shard.manifest_path, expected_config.run_id)
    if shard.sha256 != _required_sha256(expected_sha256, "tokenizer pin sha256"):
        raise ValueError("tokenizer pin shard hash does not verify")
    expected_namespace = (
        "mechanism_train" if expected_config.profile == "confirmatory" else "pilot"
    )
    if shard.namespace != expected_namespace:
        raise ValueError("tokenizer pin namespace does not match the registered config")
    rows = _read_json_rows(shard.data_path)
    required = {
        "kind",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "chat_template_sha256",
        "chat_template_utf8_hex",
    }
    if len(rows) != 1 or set(rows[0]) != required or rows[0].get("kind") != "tokenizer_pin":
        raise ValueError("tokenizer pin has an invalid schema")
    row = rows[0]
    if (
        row.get("model_id"),
        row.get("model_revision"),
        row.get("tokenizer_revision"),
    ) != (
        expected_config.model_id,
        expected_config.model_revision,
        expected_config.tokenizer_revision,
    ):
        raise ValueError("tokenizer pin model identity does not match the registered config")
    claimed_template_hash = _required_sha256(
        row.get("chat_template_sha256"), "tokenizer pin chat_template_sha256"
    )
    template_hex = row.get("chat_template_utf8_hex")
    if not isinstance(template_hex, str):
        raise ValueError("tokenizer pin chat template bytes are invalid")
    try:
        template_bytes = bytes.fromhex(template_hex)
    except ValueError as error:
        raise ValueError("tokenizer pin chat template bytes are invalid") from error
    if template_bytes.hex() != template_hex:
        raise ValueError("tokenizer pin chat template bytes are not canonical")
    if not template_bytes:
        raise ValueError("tokenizer pin chat template bytes must be nonempty")
    if hashlib.sha256(template_bytes).hexdigest() != claimed_template_hash:
        raise ValueError("tokenizer pin chat template bytes do not match the claimed hash")
    if claimed_template_hash != _required_sha256(
        expected_chat_template_sha256, "expected tokenizer pin chat template hash"
    ):
        raise ValueError("tokenizer pin chat template does not match the prompt capability")
    if (
        expected_config.profile == "smoke"
        and claimed_template_hash != _SMOKE_CHAT_TEMPLATE_SHA256
    ):
        raise ValueError(
            "tokenizer pin chat template does not match the registered smoke tokenizer revision"
        )
    if expected_config.chat_template_sha256 and (
        claimed_template_hash != expected_config.chat_template_sha256
    ):
        raise ValueError("tokenizer pin chat template does not match the registered config")
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": expected_config.config_hash,
            "source_manifest_sha256": _required_sha256(
                expected_source_manifest_sha256,
                "tokenizer pin source_manifest_sha256",
            ),
            "chat_template_sha256": claimed_template_hash,
        },
    )
    return shard


def _prepare_power_audit(
    store,
    config,
    factorial_rows,
    explicit_manifest,
    *,
    run_registered,
):
    design_hash = _design_sha256(factorial_rows)
    if explicit_manifest:
        shard = _require_verified_shard_kind(
            store, explicit_manifest, "power_audit", "verified power audit manifest"
        )
        rows = _read_json_rows(shard.data_path)
        if len(rows) != 1 or set(rows[0]) != {"kind", "audit"}:
            raise ValueError("power audit manifest has an invalid schema")
        audit = _power_audit_from_record(rows[0].get("audit"))
        _verify_shard_lineage(
            shard,
            {"config_sha256": config.config_hash, "design_sha256": design_hash},
        )
        _validate_power_audit(audit, design_hash)
        return audit, shard
    if not run_registered:
        raise ValueError("confirmatory build requires an explicit power audit preparation mode")
    audit = _POWER_EXECUTOR(
        factorial_rows,
        REGISTERED_POWER_GRID.interactions,
        {
            "entity_icc": REGISTERED_POWER_GRID.entity_iccs,
            "template_icc": REGISTERED_POWER_GRID.template_iccs,
            "invalid_format_rate": REGISTERED_POWER_GRID.invalid_format_rates,
        },
        CONFIRMATORY_POWER_SEED,
        simulations=CONFIRMATORY_POWER_SIMULATIONS,
    )
    if not isinstance(audit, PowerAudit):
        raise ValueError("registered power executor must return a PowerAudit")
    _validate_power_audit(audit, design_hash)
    audit_hash = _sha256_json(asdict(audit))
    shard = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        f"power-audit-{audit_hash[:16]}",
        [{"kind": "power_audit", "audit": asdict(audit)}],
        {
            "config_sha256": config.config_hash,
            "design_sha256": design_hash,
        },
        record_kind="power_audit",
    )
    return audit, shard


def _load_verified_pilot_gate(
    store: FAArtifactStore,
    path: str | Path,
    expected_config: FAConfig,
) -> dict[str, Any]:
    if expected_config.profile != "smoke":
        raise ValueError("pilot gate identity must be a smoke config")
    shard = _require_verified_shard_kind(
        store, path, "pilot_gate", "verified pilot gate sidecar manifest"
    )
    _verify_artifact_run_id(shard.manifest_path, expected_config.run_id)
    rows = _read_json_rows(shard.data_path)
    required = {
        "kind",
        "config_sha256",
        "source_manifest_sha256",
        "prompt_manifest",
        "prompt_manifest_sha256",
        "tokenizer_pin_manifest",
        "tokenizer_pin_sha256",
        "chat_template_sha256",
        "generation_sidecar_manifest",
        "generation_sidecar_sha256",
        "pilot_gate",
        "metrics",
        "evidence_sha256",
    }
    if len(rows) != 1 or set(rows[0]) != required or rows[0].get("kind") != "pilot_gate":
        raise ValueError("pilot gate tokenizer pin chain has an invalid schema")
    row = rows[0]
    prompt_path = _artifact_path_from_record(
        store, row.get("prompt_manifest"), "pilot prompt manifest"
    )
    try:
        manifest = _load_manifest(store, prompt_path, expected_config)
    except ValueError as error:
        raise ValueError(
            f"pilot gate does not match the exact registered smoke config: {error}"
        ) from error
    _verify_artifact_run_id(manifest.shard_manifest_path, expected_config.run_id)
    if manifest.namespace != "pilot" or manifest.shard_sha256 != row.get("prompt_manifest_sha256"):
        raise ValueError("pilot gate prompt manifest does not verify")
    if not manifest.chat_template_sha256:
        raise ValueError("pilot gate must bind an observed nonempty chat template pin")
    gate_pin_path = _artifact_path_from_record(
        store, row.get("tokenizer_pin_manifest"), "pilot gate tokenizer pin manifest"
    )
    _load_verified_tokenizer_pin(
        store,
        gate_pin_path,
        expected_config,
        expected_sha256=row.get("tokenizer_pin_sha256"),
        expected_source_manifest_sha256=manifest.full_manifest_sha256,
        expected_chat_template_sha256=row.get("chat_template_sha256"),
    )
    if (
        gate_pin_path != manifest.tokenizer_pin_manifest_path
        or row.get("tokenizer_pin_sha256") != manifest.tokenizer_pin_sha256
        or row.get("chat_template_sha256") != manifest.chat_template_sha256
    ):
        raise ValueError("pilot gate does not bind the prompt tokenizer pin and template")
    if (
        row.get("config_sha256") != expected_config.config_hash
        or row.get("source_manifest_sha256") != manifest.manifest_sha256
    ):
        raise ValueError("pilot gate does not match the exact registered smoke config")
    generation_path = _artifact_path_from_record(
        store, row.get("generation_sidecar_manifest"), "generation sidecar manifest"
    )
    generation = _require_generation_sidecar_manifest(store, generation_path)
    _verify_artifact_run_id(generation.manifest_path, expected_config.run_id)
    if generation.sha256 != row.get("generation_sidecar_sha256"):
        raise ValueError("pilot gate generation sidecar hash does not verify")
    if shard.namespace != "pilot" or generation.namespace != "pilot":
        raise ValueError("confirmatory construction requires pilot-namespace gate evidence")
    records = _load_verified_generation_sidecar(
        store, generation.manifest_path, manifest, expected_config
    )
    scored = _score_rows(records, manifest)
    metrics = _json_safe(estimate_behavior(scored).to_record())
    gate = _pilot_gate(scored, estimate_behavior(scored))
    evidence_hash = _sha256_json({"metrics": metrics, "pilot_gate": gate})
    if row.get("metrics") != metrics or row.get("pilot_gate") != gate:
        raise ValueError("stored pilot gate evidence does not match deterministic recomputation")
    if row.get("evidence_sha256") != evidence_hash:
        raise ValueError("pilot gate evidence hash does not verify")
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": expected_config.config_hash,
            "source_manifest_sha256": manifest.manifest_sha256,
            "generation_sidecar_sha256": generation.sha256,
            "prompt_manifest_sha256": manifest.shard_sha256,
            "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
            "chat_template_sha256": manifest.chat_template_sha256,
        },
    )
    return {"status": gate["status"], "evidence_sha256": evidence_hash}


def _require_generation_sidecar_manifest(
    store: FAArtifactStore, path: str | Path, *, label: str = "verified generation sidecar manifest"
):
    return _require_verified_shard_kind(store, path, "generation", label)


def _load_verified_generation_sidecar(store, manifest_path, manifest, config=None):
    shard = _require_generation_sidecar_manifest(store, manifest_path)
    expected_model = manifest.model_sha256
    expected_tokenizer = manifest.tokenizer_sha256
    if config is not None and (expected_model, expected_tokenizer) != _config_runtime_hashes(config):
        raise ValueError("generation runtime pins do not match config")
    expected_lineage = {
        "config_sha256": manifest.config_hash,
        "source_manifest_sha256": manifest.manifest_sha256,
        "model_sha256": expected_model,
        "tokenizer_sha256": expected_tokenizer,
        "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
        "chat_template_sha256": manifest.chat_template_sha256,
    }
    _verify_shard_lineage(shard, expected_lineage)
    examples = {row.example_id: row for row in manifest.examples}
    records = _read_json_rows(shard.data_path)
    expected_multiset = Counter(row.example_id for row in manifest.examples)
    observed_multiset = Counter(record.get("example_id") for record in records)
    if observed_multiset != expected_multiset:
        raise ValueError(
            "verified generation sidecar must contain every expected example exactly once"
        )
    for record in records:
        example = examples.get(record.get("example_id"))
        if example is None:
            raise ValueError("verified generation sidecar contains an unregistered example")
        expected = {
            "config_sha256": manifest.config_hash,
            "example_sha256": example.canonical_payload_sha256,
            "model_sha256": expected_model,
            "tokenizer_sha256": expected_tokenizer,
            "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
            "chat_template_sha256": manifest.chat_template_sha256,
        }
        if any(record.get(name) != value for name, value in expected.items()):
            raise ValueError("verified generation sidecar provenance does not match its manifest")
        if record.get("kind") != "generation":
            raise ValueError("verified generation sidecar has an invalid record kind")
        if record.get("generation") != manifest.generation:
            raise ValueError("verified generation sidecar generation config does not match")
        if record.get("example") != _record_value(example):
            raise ValueError("verified generation sidecar canonical example does not match")
        if record.get("status") not in {"completed", "infrastructure_failure"}:
            raise ValueError("verified generation sidecar has an invalid completion status")
    return records


def _require_verified_shard_kind(store, path, record_kind, label):
    source = Path(path)
    if not source.name.endswith(".jsonl.manifest.json"):
        raise ValueError(f"{label} must name an immutable shard manifest")
    try:
        shard = store.verify_shard(source)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"{label} does not verify: {error}") from error
    if shard.record_kind != record_kind:
        raise ValueError(f"{label} has record kind {shard.record_kind}, expected {record_kind}")
    return shard


def _config_runtime_hashes(config: FAConfig) -> tuple[str, str]:
    return (
        _sha256_json({"model_id": config.model_id, "revision": config.model_revision}),
        _sha256_json(
            {"model_id": config.model_id, "revision": config.tokenizer_revision}
        ),
    )


def _design_sha256(rows) -> str:
    return hashlib.sha256(
        "\n".join(sorted(row.example_id for row in rows)).encode("utf-8")
    ).hexdigest()


def _power_audit_from_record(value: Any) -> PowerAudit:
    if not isinstance(value, dict) or set(value) != {
        "design_sha256",
        "seed",
        "simulations",
        "cells",
        "registered_grid",
    }:
        raise ValueError("power audit record has an invalid schema")
    cells = value.get("cells")
    if not isinstance(cells, list):
        raise ValueError("power audit cells must be a list")
    try:
        return PowerAudit(
            design_sha256=value["design_sha256"],
            seed=value["seed"],
            simulations=value["simulations"],
            cells=tuple(PowerCell(**cell) for cell in cells),
            registered_grid=value["registered_grid"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"power audit record is invalid: {error}") from error


def _validate_power_audit(audit: PowerAudit, design_hash: str) -> None:
    expected = {
        (absent, entity, template, invalid, interaction)
        for absent in REGISTERED_POWER_GRID.absent_attempt_rates
        for entity in REGISTERED_POWER_GRID.entity_iccs
        for template in REGISTERED_POWER_GRID.template_iccs
        for invalid in REGISTERED_POWER_GRID.invalid_format_rates
        for interaction in REGISTERED_POWER_GRID.interactions
    }
    if (
        not audit.registered_grid
        or audit.design_sha256 != design_hash
        or audit.seed != CONFIRMATORY_POWER_SEED
        or audit.simulations != CONFIRMATORY_POWER_SIMULATIONS
        or len(audit.cells) != 180
    ):
        raise ValueError("power audit does not match the exact registered design")
    observed = set()
    for cell in audit.cells:
        values = (
            cell.absent_attempt_rate,
            cell.entity_icc,
            cell.template_icc,
            cell.invalid_format_rate,
            cell.interaction,
            cell.estimated_power,
            cell.monte_carlo_standard_error,
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError("power audit contains non-finite values")
        key = values[:5]
        if key not in expected or key in observed:
            raise ValueError("power audit must contain all 180 unique registered cells")
        observed.add(key)
        expected_mcse = math.sqrt(
            cell.estimated_power * (1.0 - cell.estimated_power) / cell.simulations
        ) if 0.0 <= cell.estimated_power <= 1.0 and cell.simulations > 0 else math.nan
        if (
            cell.simulations != CONFIRMATORY_POWER_SIMULATIONS
            or not 0.0 <= cell.estimated_power <= 1.0
            or not math.isclose(
                cell.monte_carlo_standard_error,
                expected_mcse,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("power audit cell power or MCSE is invalid")
    if observed != expected:
        raise ValueError("power audit must contain all 180 unique registered cells")


def _verify_shard_lineage(shard, expected):
    sidecar = _read_json_object(shard.manifest_path)
    lineage = sidecar.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("verified shard lineage is invalid")
    for name, value in expected.items():
        if lineage.get(name) != value:
            raise ValueError(f"verified shard lineage does not bind {name}")


def _verify_artifact_run_id(manifest_path: Path, expected_run_id: str) -> None:
    sidecar = _read_json_object(manifest_path)
    if sidecar.get("run_id") != expected_run_id:
        raise ValueError("pilot gate artifact does not belong to the registered smoke run")


def _artifact_path_from_record(store, value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{label} path is invalid")
    path = store.root / relative
    try:
        path.relative_to(store.root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes the artifact root") from error
    return path


def _read_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("explicit manifest must be a JSON object")
    return value


def _read_json_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    source = Path(path)
    if source.suffix == ".jsonl":
        values = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    else:
        value = json.loads(source.read_text(encoding="utf-8"))
        values = value.get("rows", value) if isinstance(value, dict) else value
    if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
        raise ValueError("explicit manifest must contain JSON object rows")
    return tuple(values)


def _without_schema(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "schema_version"}


def _write_manifest(root: Path, run_id: str, label: str, value: Any) -> Path:
    directory = root / "fa_manifests" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{label}-{hashlib.sha256(json.dumps(value, sort_keys=True, default=list).encode()).hexdigest()}.json"
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError("immutable manifest path collision")
        return path
    path.write_text(payload, encoding="utf-8")
    return path


def _required_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


class _WhitespaceTokenizer:
    all_special_ids = ()

    def encode(self, text: str, add_special_tokens: bool = False):
        return text.split()

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        return self.encode(messages[0]["content"])
