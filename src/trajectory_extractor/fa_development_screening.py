"""Bounded, resumable Gemma screening for open Source-v6 development splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
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
from trajectory_extractor.fa_development_source import (
    DEVELOPMENT_SPLITS,
    DevelopmentSourceDesign,
    audit_development_source,
    summarize_screening_yield,
)
from trajectory_extractor.fa_entities import (
    CandidateEntity,
    ScreeningQuestion,
    score_screening,
)
from trajectory_extractor.fa_runtime import HFModelRunner

_SOURCE_REVISION = "fa-development-source-v6"
_INTEGRITY_FILE = "source_integrity_v1.json"


def run_development_screening(
    config: FAConfig,
    source_root: str | Path,
    split: str,
    output_root: str | Path,
    *,
    batch_size: int = 8,
    runner: Any | None = None,
    freeze_manifest: str | Path | None = None,
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
    if split == "construction_validation":
        if freeze_manifest is None:
            raise ValueError(
                "construction_validation requires a frozen instrument manifest"
            )
        freeze_sha256 = _verify_freeze_manifest(
            Path(freeze_manifest),
            source_integrity_sha256=source["integrity_sha256"],
            config_sha256=config.config_hash,
            git_commit=resolved_git_commit,
        )
    candidates = source["candidates"]
    questions = source["questions"]
    prompts = _ordered_prompts(candidates, questions)

    identity = {
        "schema_version": 1,
        "source_revision": _SOURCE_REVISION,
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
        "git_commit": resolved_git_commit,
        "freeze_manifest_sha256": freeze_sha256,
    }
    identity_sha256 = _sha256_bytes(_canonical_bytes(identity))
    run_dir = Path(output_root) / split / identity_sha256
    _write_immutable(run_dir / "execution_identity.json", _canonical_bytes(identity))

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
        rows.extend(batch_rows)

    _verify_complete_rows(rows, prompts, split)
    completions_by_entity = {
        candidate.entity_id: tuple(
            row["completion"] for row in rows if row["entity_id"] == candidate.entity_id
        )
        for candidate in candidates
    }
    summary = summarize_screening_yield(candidates, completions_by_entity)
    items_path = run_dir / "screening_items.jsonl"
    summary_path = run_dir / "screening_yield.json"
    _write_immutable(items_path, _jsonl_bytes(rows))
    _write_immutable(summary_path, _canonical_bytes(summary))
    return {
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


def write_instrument_freeze_manifest(
    path: str | Path,
    *,
    source_root: str | Path,
    config: FAConfig,
    success_criteria: Mapping[str, Any],
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Freeze all identities before construction-validation outcomes are opened."""
    _verify_config(config)
    source = _load_verified_source(Path(source_root), "construction_validation")
    resolved_git_commit = git_commit or _current_git_commit()
    _validate_git_commit(resolved_git_commit)
    if not success_criteria:
        raise ValueError("freeze manifest requires explicit success criteria")
    manifest = {
        "schema_version": 1,
        "kind": "fa_source_v6_instrument_freeze",
        "source_revision": _SOURCE_REVISION,
        "source_integrity_sha256": source["integrity_sha256"],
        "config_sha256": config.config_hash,
        "git_commit": resolved_git_commit,
        "success_criteria": dict(success_criteria),
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
    source_integrity_sha256: str,
    config_sha256: str,
    git_commit: str,
) -> str:
    manifest = _read_json(path)
    expected_identity = {
        "schema_version": 1,
        "kind": "fa_source_v6_instrument_freeze",
        "source_revision": _SOURCE_REVISION,
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
    if (
        not isinstance(integrity, dict)
        or integrity.get("schema_version") != 1
        or integrity.get("source_revision") != _SOURCE_REVISION
    ):
        raise ValueError("Source-v6 integrity identity is invalid")
    snapshot_path = root / str(integrity.get("source_snapshot", ""))
    if _sha256_file(snapshot_path) != integrity.get("source_snapshot_sha256"):
        raise ValueError("Source-v6 snapshot hash mismatch")
    snapshot = _read_json(snapshot_path)
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("source_revision") != _SOURCE_REVISION
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
    manifests: dict[
        str, tuple[tuple[CandidateEntity, ...], tuple[ScreeningQuestion, ...]]
    ] = {}
    file_hashes: dict[str, dict[str, str]] = {}
    for split in DEVELOPMENT_SPLITS:
        files = integrity.get("materialized_files", {}).get(split)
        if not isinstance(files, dict):
            raise ValueError("Source-v6 integrity is missing a development split")
        candidate_path = root / str(files.get("candidate_manifest", ""))
        question_path = root / str(files.get("question_manifest", ""))
        candidate_sha256 = _sha256_file(candidate_path)
        question_sha256 = _sha256_file(question_path)
        if candidate_sha256 != files.get(
            "candidate_sha256"
        ) or question_sha256 != files.get("question_sha256"):
            raise ValueError("Source-v6 materialized file hash mismatch")
        candidates = tuple(
            CandidateEntity(**row) for row in _read_array(candidate_path)
        )
        questions = tuple(
            ScreeningQuestion(**row) for row in _read_array(question_path)
        )
        manifests[split] = (candidates, questions)
        file_hashes[split] = {
            "candidate_sha256": candidate_sha256,
            "question_sha256": question_sha256,
        }
    excluded_qids = frozenset(str(qid) for qid in snapshot.get("excluded_qids", ()))
    audit = audit_development_source(
        manifests,
        design=design,
        excluded_qids=excluded_qids,
    )
    if audit != snapshot.get("audit"):
        raise ValueError("Source-v6 audit or count mismatch")
    candidates, questions = manifests[selected_split]
    return {
        "candidates": candidates,
        "questions": questions,
        "integrity_sha256": _sha256_file(integrity_path),
        **file_hashes[selected_split],
    }


def _ordered_prompts(
    candidates: Sequence[CandidateEntity],
    questions: Sequence[ScreeningQuestion],
) -> tuple[tuple[CandidateEntity, ScreeningQuestion], ...]:
    by_qid: dict[str, list[ScreeningQuestion]] = {}
    for question in questions:
        by_qid.setdefault(question.qid, []).append(question)
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
    probe = [""] * 3
    probe[question_index] = completion
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
        "is_correct": is_correct,
    }


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
    args = parser.parse_args(argv)
    result = run_development_screening(
        FAConfig.from_json(args.config),
        args.source_root,
        args.split,
        args.output_root,
        batch_size=args.batch_size,
        freeze_manifest=args.freeze_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
