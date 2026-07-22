from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_data import audit_dataset, build_factorial_examples, build_manifest, build_same_string_examples
from trajectory_extractor.fa_entities import CandidateEntity, EntityMatch, SyntheticCandidate, match_synthetic_entities, score_screening
from trajectory_extractor.fa_runtime import HFModelRunner, run_generation_shard
from trajectory_extractor.fa_scoring import estimate_behavior, score_response


FA_COMMANDS = (
    "fa-screen-entities", "fa-build-pilot", "fa-build-confirmatory", "fa-audit-manifest",
    "fa-run-generation", "fa-score-behavior",
    "fa-extract-activations", "fa-fit-probes", "fa-seal-selection", "fa-unlock-endpoint",
    "fa-evaluate-behavior-test", "fa-evaluate-probe-test", "fa-evaluate-intervention-test",
    "fa-run-interventions", "fa-select-circuit-cases", "fa-audit-circuit-fidelity", "fa-build-report",
)
_IMPLEMENTED = frozenset(FA_COMMANDS[:6])
_GENERATION_NAMESPACES = ("pilot", "mechanism_train", "locked_validation", "circuit_dev", "behavior_test", "probe_test", "intervention_test")
_PROTECTED = frozenset({"behavior_test", "probe_test", "intervention_test"})


def register_fa_subcommands(subparsers: argparse._SubParsersAction) -> None:
    parsers: dict[str, argparse.ArgumentParser] = {}
    for command in FA_COMMANDS:
        parser = subparsers.add_parser(command)
        parser.add_argument("--config", required=True)
        parser.add_argument("--root", default=".")
        parsers[command] = parser
    parsers["fa-screen-entities"].add_argument("--candidates-manifest", required=True)
    parsers["fa-screen-entities"].add_argument("--screening-manifest", required=True)
    parsers["fa-screen-entities"].add_argument("--synthetic-manifest", required=True)
    parsers["fa-build-pilot"].add_argument("--matches-manifest", required=True)
    parsers["fa-build-confirmatory"].add_argument("--matches-manifest", required=True)
    parsers["fa-build-confirmatory"].add_argument("--pilot-gate-manifest", required=True)
    parsers["fa-audit-manifest"].add_argument("--manifest", required=True)
    generation = parsers["fa-run-generation"]
    generation.add_argument("--manifest", required=True)
    generation.add_argument("--shard-id", required=True)
    generation.add_argument("--namespace", choices=_GENERATION_NAMESPACES, required=True)
    generation.add_argument("--endpoint-manifest")
    score = parsers["fa-score-behavior"]
    score.add_argument("--manifest", required=True)
    score.add_argument("--generation-manifest", required=True)


def dispatch_fa(args: argparse.Namespace) -> int | None:
    command = getattr(args, "command", None)
    if command not in FA_COMMANDS:
        return None
    if command not in _IMPLEMENTED:
        print(json.dumps({"command": command, "status": "not_implemented"}))
        return 2
    config = FAConfig.from_json(args.config)
    root = Path(args.root)
    if command == "fa-screen-entities":
        payload = _screen_entities(config, root, args)
    elif command in {"fa-build-pilot", "fa-build-confirmatory"}:
        payload = _build_manifest(config, root, args, confirmatory=command == "fa-build-confirmatory")
    elif command == "fa-audit-manifest":
        payload = _audit_manifest(config, args)
    elif command == "fa-run-generation":
        payload = _run_generation(config, root, args)
    else:
        payload = _score_behavior(config, root, args)
    print(json.dumps(_json_safe({"command": command, **payload}), sort_keys=True, allow_nan=False))
    return 0


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
    if confirmatory:
        gate = _read_json_object(args.pilot_gate_manifest)
        if gate.get("status") != "passed":
            raise ValueError("confirmatory construction requires a passed pilot gate manifest")
    matches = tuple(EntityMatch(**_without_schema(row)) for row in _read_json_rows(args.matches_manifest))
    rows = build_factorial_examples(config, matches, tokenizer=_WhitespaceTokenizer())
    if confirmatory:
        rows += build_same_string_examples(config, matches, tokenizer=_WhitespaceTokenizer())
    manifest = build_manifest(config, rows)
    path = _write_manifest(root, config.run_id, "confirmatory" if confirmatory else "pilot", _manifest_record(manifest))
    return {"status": "built", "manifest": str(path), "count": len(rows), "manifest_sha256": manifest.manifest_sha256}


def _audit_manifest(config: FAConfig, args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.manifest, config)
    factorial = tuple(row for row in manifest.examples if row.block == "factorial")
    same_string = tuple(row for row in manifest.examples if row.block == "same_string")
    audit = audit_dataset(factorial, same_string, tokenizer=_WhitespaceTokenizer())
    return {"status": "passed" if audit.passed else "failed", "checks": dict(audit.checks), "violations": list(audit.violations)}


def _run_generation(config: FAConfig, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.manifest, config)
    if any(row.split != args.namespace for row in manifest.examples):
        raise ValueError("explicit generation manifest contains another namespace")
    endpoint = None
    store = FAArtifactStore(root)
    if args.namespace in _PROTECTED:
        if not args.endpoint_manifest:
            raise ValueError("protected test generation requires an explicit endpoint manifest")
        endpoint = _read_json_object(args.endpoint_manifest)
        if endpoint.get("endpoint") != args.namespace:
            raise ValueError("endpoint manifest does not bind the requested endpoint")
        selection = tuple(store.verify_shard(path) for path in endpoint.get("selection_shard_manifests", ()))
        if not selection:
            raise ValueError("endpoint manifest requires explicit selection shard manifests")
        store.seal_endpoint(args.namespace, selection, _endpoint_parents(endpoint))
        parents = _endpoint_parents(endpoint)
        receipt = store.unlock_endpoint(
            args.namespace, parents["preregistration"], parents["selection_manifest"]
        )
    runner = HFModelRunner(config)
    shard = run_generation_shard(runner, manifest, store, args.shard_id, config=config, namespace=args.namespace)
    result = {"status": "generated", "shard_manifest": str(shard.manifest_path), "sha256": shard.sha256}
    if endpoint is not None:
        metric = _write_protected_metrics(store, config, manifest, shard, args.namespace)
        store.mark_evaluated(receipt, metric.data_path)
        closed = store.close_endpoint(args.namespace)
        result.update({"metrics_manifest": str(metric.manifest_path), "closed_endpoint": str(closed)})
    return result


def _score_behavior(config: FAConfig, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.manifest, config)
    rows = _score_rows(_read_json_rows(args.generation_manifest), manifest)
    metrics = estimate_behavior(rows)
    gate = _pilot_gate(rows, metrics)
    return {"status": "scored", "metrics": _json_safe(metrics.to_record()), "pilot_gate": gate}


def _write_protected_metrics(store: FAArtifactStore, config: FAConfig, manifest: Any, shard: Any, namespace: str):
    scored = _score_rows(_read_json_rows(shard.data_path), manifest)
    return store.write_completed_shard(
        config.run_id, namespace, f"{shard.shard_id}.metrics",
        [{"metrics": _json_safe(estimate_behavior(scored).to_record())}],
        {"config_sha256": config.config_hash, "source_manifest_sha256": manifest.manifest_sha256},
    )


def _score_rows(records: tuple[dict[str, Any], ...], manifest: Any):
    examples = {row.example_id: row for row in manifest.examples}
    codes = frozenset(row.registry_code for row in manifest.examples)
    scored = []
    for record in records:
        example_record = record.get("example")
        example_id = record.get("example_id") or (example_record or {}).get("example_id")
        example = examples.get(example_id)
        if example is None and isinstance(example_record, dict):
            example = SimpleNamespace(**example_record)
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


def _load_manifest(path: str, config: FAConfig):
    value = _read_json_object(path)
    if value.get("config_hash") != config.config_hash:
        raise ValueError("explicit manifest config hash does not match config")
    examples = tuple(SimpleNamespace(**row) for row in value.get("examples", ()))
    if not isinstance(value.get("manifest_sha256"), str):
        raise ValueError("manifest requires manifest_sha256")
    return SimpleNamespace(config_hash=value["config_hash"], manifest_sha256=value["manifest_sha256"], examples=examples)


def _manifest_record(manifest: Any) -> dict[str, Any]:
    return {"config_hash": manifest.config_hash, "manifest_sha256": manifest.manifest_sha256, "examples": [asdict(row) for row in manifest.examples]}


def _endpoint_parents(value: dict[str, Any]) -> dict[str, str]:
    parents = {"preregistration": value.get("preregistration_hash"), "selection_manifest": value.get("selection_manifest_hash")}
    if any(not isinstance(item, str) or len(item) != 64 for item in parents.values()):
        raise ValueError("endpoint manifest requires hash-bound preregistration and selection manifests")
    return parents


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
