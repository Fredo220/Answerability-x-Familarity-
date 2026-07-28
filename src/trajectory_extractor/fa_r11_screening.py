"""Resumable screening for the R11 broad relation-bank instrument."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_development_screening import (
    _answer_key,
    development_screening_parser_sha256,
    parse_development_screening_answer,
)
from trajectory_extractor.fa_runtime import HFModelRunner

_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def run_r11_screening(
    config: FAConfig,
    *,
    source_root: Path,
    split: str,
    output_root: Path,
    pre_model_semantic_audit: Path,
    batch_size: int,
    runner: Any | None = None,
    selection: Mapping[str, Any] | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Run or resume one R11 split with immutable batch checkpoints."""
    if split not in {"instrument_development", "construction_validation"}:
        raise ValueError("R11 screening split is invalid")
    if split == "construction_validation" and selection is None:
        raise ValueError("R11 validation requires a frozen relation selection")
    if split == "instrument_development" and selection is not None:
        raise ValueError("R11 development must screen the full relation bank")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch size must be positive")
    commit = git_commit or _current_git_commit()
    if not _SHA40.fullmatch(commit):
        raise ValueError("git commit must be a full SHA-1")

    manifest = _read_json(source_root / "source_manifest_v1.json")
    rows_path = source_root / str(manifest.get("rows_file", ""))
    if _sha256_file(rows_path) != manifest.get("rows_sha256"):
        raise ValueError("R11 source rows hash mismatch")
    audit = _read_json(pre_model_semantic_audit)
    if (
        audit.get("kind") != "fa_r11_pre_model_semantic_audit"
        or audit.get("status") != "passed"
        or audit.get("blocker_count") != 0
        or audit.get("rows_sha256") != manifest.get("rows_sha256")
    ):
        raise ValueError("R11 pre-model semantic audit is not passing")
    audit_items_path = pre_model_semantic_audit.parent / str(
        audit.get("items_file", "")
    )
    if (
        not audit_items_path.is_file()
        or _sha256_file(audit_items_path) != audit.get("items_sha256")
    ):
        raise ValueError("R11 pre-model audit item hash mismatch")
    audit_items = _read_jsonl(audit_items_path)
    source_rows = _read_jsonl(rows_path)
    expected_audit_keys = {
        (str(row["entity_id"]), str(row["relation_id"])) for row in source_rows
    }
    observed_audit_keys = {
        (str(row.get("entity_id")), str(row.get("relation_id")))
        for row in audit_items
    }
    if (
        audit.get("item_count") != len(audit_items)
        or observed_audit_keys != expected_audit_keys
        or len(observed_audit_keys) != len(audit_items)
        or any(
            row.get("structural_pass") is not True
            or row.get("semantic_pass") is not True
            or not isinstance(row.get("structural_auditor_id"), str)
            or not isinstance(row.get("semantic_auditor_id"), str)
            or not row["structural_auditor_id"]
            or not row["semantic_auditor_id"]
            or row["structural_auditor_id"] == row["semantic_auditor_id"]
            for row in audit_items
        )
    ):
        raise ValueError("R11 pre-model audit coverage or independence failed")
    prompts = [
        row
        for row in source_rows
        if row.get("split") == split
    ]
    if selection is not None:
        frozen_selection = dict(selection)
        selection_sha256 = frozen_selection.pop("selection_sha256", None)
        screening_identity = selection.get("screening_identity")
        expected_screening_identity = {
            "config_sha256": config.config_hash,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "tokenizer_revision": config.tokenizer_revision,
            "chat_template_sha256": config.chat_template_sha256,
            "parser_sha256": development_screening_parser_sha256(),
            "semantic_audit_sha256": _sha256_file(
                pre_model_semantic_audit
            ),
        }
        if (
            selection_sha256 != _canonical_sha256(frozen_selection)
            or selection.get("source_manifest_sha256")
            != _sha256_file(source_root / "source_manifest_v1.json")
            or selection.get("git_commit") != commit
            or screening_identity != expected_screening_identity
        ):
            raise ValueError("R11 frozen selection provenance mismatch")
        selected = selection.get("selected_relations")
        if not isinstance(selected, Mapping):
            raise ValueError("R11 selection lacks selected relations")
        prompts = [
            row
            for row in prompts
            if row.get("relation_id") in selected.get(row.get("domain"), ())
        ]
    prompts = sorted(
        prompts,
        key=lambda row: (
            str(row.get("domain")),
            str(row.get("entity_id")),
            str(row.get("relation_id")),
        ),
    )
    if not prompts:
        raise ValueError("R11 screening has no prompts")

    identity = {
        "schema_version": 1,
        "kind": "fa_r11_screening_execution",
        "source_manifest_sha256": _sha256_file(
            source_root / "source_manifest_v1.json"
        ),
        "rows_sha256": manifest["rows_sha256"],
        "split": split,
        "prompt_count": len(prompts),
        "batch_size": batch_size,
        "config_sha256": config.config_hash,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "chat_template_sha256": config.chat_template_sha256,
        "generation": dict(config.generation),
        "parser_sha256": development_screening_parser_sha256(),
        "semantic_audit_sha256": _sha256_file(pre_model_semantic_audit),
        "selection_sha256": (
            selection.get("selection_sha256") if selection is not None else None
        ),
        "git_commit": commit,
    }
    identity_sha256 = _canonical_sha256(identity)
    run_dir = output_root / split / identity_sha256
    _write_json_immutable(run_dir / "execution_identity.json", identity)

    active_runner = runner
    items = []
    resumed = 0
    for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
        batch = prompts[start : start + batch_size]
        batch_path = run_dir / f"batch-{batch_index:06d}.jsonl"
        if batch_path.exists():
            batch_rows = _read_jsonl(batch_path)
            expected = {
                (str(row["entity_id"]), str(row["relation_id"])) for row in batch
            }
            observed = {
                (str(row.get("entity_id")), str(row.get("relation_id")))
                for row in batch_rows
            }
            if observed != expected:
                raise ValueError("R11 checkpoint batch identity mismatch")
            source_by_key = {
                (str(row["entity_id"]), str(row["relation_id"])): row
                for row in batch
            }
            for checkpoint_row in batch_rows:
                key = (
                    str(checkpoint_row.get("entity_id")),
                    str(checkpoint_row.get("relation_id")),
                )
                completion = checkpoint_row.get("completion")
                if not isinstance(completion, str) or checkpoint_row != _score_item(
                    source_by_key[key],
                    completion,
                ):
                    raise ValueError("R11 checkpoint content verification failed")
            items.extend(batch_rows)
            resumed += 1
            continue
        if active_runner is None:
            active_runner = HFModelRunner(config)
        rendered = [
            active_runner.render_prompt(str(row["prompt"])) for row in batch
        ]
        completions = list(
            active_runner.generate(rendered, dict(config.generation))
        )
        if len(completions) != len(batch):
            raise RuntimeError("R11 model runner returned the wrong item count")
        batch_rows = [
            _score_item(row, completion)
            for row, completion in zip(batch, completions, strict=True)
        ]
        _write_jsonl_immutable(batch_path, batch_rows)
        items.extend(batch_rows)

    item_keys = [
        (str(row["entity_id"]), str(row["relation_id"])) for row in items
    ]
    if len(item_keys) != len(set(item_keys)) or len(items) != len(prompts):
        raise ValueError("R11 output is incomplete or duplicated")
    items = sorted(
        items,
        key=lambda row: (
            str(row["domain"]),
            str(row["entity_id"]),
            str(row["relation_id"]),
        ),
    )
    items_path = run_dir / "screening_items.jsonl"
    _write_jsonl_immutable(items_path, items)
    summary = _summarize(items)
    summary_path = run_dir / "screening_yield.json"
    _write_json_immutable(summary_path, summary)
    return {
        "status": "completed",
        "split": split,
        "execution_identity_sha256": identity_sha256,
        "resumed_batch_count": resumed,
        "items_path": str(items_path),
        "items_sha256": _sha256_file(items_path),
        "summary_path": str(summary_path),
        "summary_sha256": _sha256_file(summary_path),
        "summary": summary,
    }


def _score_item(source: Mapping[str, Any], completion: str) -> dict[str, Any]:
    aliases = tuple(str(value) for value in source["accepted_aliases"])
    parsed = parse_development_screening_answer(
        completion,
        aliases,
        allow_occupation_modifier=(
            source.get("domain") == "person"
            and source.get("relation_id") == "P106"
        ),
    )
    correct = _answer_key(parsed) in {_answer_key(alias) for alias in aliases}
    return {
        **dict(source),
        "kind": "fa_r11_screening_item",
        "completion": completion,
        "parsed_completion": parsed,
        "is_correct": correct,
    }


def _summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_domain_relation: dict[str, dict[str, dict[str, float | int]]] = {}
    for domain in sorted({str(row["domain"]) for row in items}):
        by_relation = {}
        domain_rows = [row for row in items if row["domain"] == domain]
        for relation in sorted({str(row["relation_id"]) for row in domain_rows}):
            relation_rows = [
                row for row in domain_rows if row["relation_id"] == relation
            ]
            successes = sum(row["is_correct"] is True for row in relation_rows)
            by_relation[relation] = {
                "item_count": len(relation_rows),
                "success_count": successes,
                "success_rate": successes / len(relation_rows),
            }
        by_domain_relation[domain] = by_relation
    return {
        "schema_version": 1,
        "item_count": len(items),
        "entity_count": len({str(row["entity_id"]) for row in items}),
        "correct_count": Counter(row["is_correct"] for row in items)[True],
        "by_domain_relation": by_domain_relation,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pre-model-semantic-audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--git-commit")
    args = parser.parse_args(argv)
    selection = _read_json(args.selection) if args.selection else None
    result = run_r11_screening(
        FAConfig.from_json(args.config),
        source_root=args.source_root,
        split=args.split,
        output_root=args.output_root,
        pre_model_semantic_audit=args.pre_model_semantic_audit,
        batch_size=args.batch_size,
        selection=selection,
        git_commit=args.git_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _current_git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact must be an object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL artifact must contain objects")
    return rows


def _write_json_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_bytes_immutable(path, encoded)


def _write_jsonl_immutable(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
        for row in rows
    ).encode("utf-8")
    _write_bytes_immutable(path, encoded)


def _write_bytes_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
