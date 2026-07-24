"""Single-process Colab orchestration for frozen Source-v5 screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_colab_checkpoint import ColabSplitCheckpointStore
from trajectory_extractor.fa_colab_preflight import run_colab_preflight
from trajectory_extractor.fa_config import FAConfig


SOURCE_REVISION = "fa-confirmatory-wikidata-v5"
SOURCE_ROOT = Path("data/fa/confirmatory_source_v5")
SOURCE_INTEGRITY_PATH = SOURCE_ROOT / "source_integrity_v1.json"
LOCK_PATH = Path("requirements/fa-core.lock")
ASSEMBLED_MATCH_COUNT = 244
SCREENING_COMPLETION_COUNT = 1152
SOURCE_CANDIDATE_COUNT = 384
COLLECTION_SHARD_ID = "confirmatory-screened-collection-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SourceV5Split:
    split: str
    candidate_count: int
    completion_count: int
    match_count: int

    @property
    def candidate_path(self) -> Path:
        return SOURCE_ROOT / f"candidate_entities_{self.split}_v1.json"

    @property
    def question_path(self) -> Path:
        return SOURCE_ROOT / f"screening_questions_{self.split}_v1.json"

    @property
    def synthetic_path(self) -> Path:
        return SOURCE_ROOT / f"synthetic_candidates_{self.split}_v1.json"


SOURCE_V5_SPLITS = (
    SourceV5Split("mechanism_train", 128, 384, 80),
    SourceV5Split("locked_validation", 64, 192, 40),
    SourceV5Split("behavior_test", 96, 288, 60),
    SourceV5Split("probe_test", 48, 144, 32),
    SourceV5Split("intervention_test", 48, 144, 32),
)


Transaction = Callable[[FAConfig, Path, argparse.Namespace], dict[str, Any]]


def run_colab_screening(
    config: FAConfig,
    root: str | Path,
    args: argparse.Namespace,
    *,
    run_screening: Transaction,
    screen_entities: Transaction,
    assemble_screened_matches: Transaction,
) -> dict[str, Any]:
    repo_root = Path(root).resolve()
    _verify_registered_design(config)
    execution = _verify_execution_inputs(repo_root, args)
    source_integrity_sha256 = _verify_source_v5(repo_root, config)

    runtime = run_colab_preflight(repo_root, LOCK_PATH)
    if runtime.get("status") != "ready":
        raise RuntimeError("Colab preflight did not return ready status")
    lock_sha256 = _sha256_file(repo_root / LOCK_PATH)
    if runtime.get("lock_sha256") != lock_sha256:
        raise RuntimeError("runtime lock identity changed after preflight")

    identity_dir = Path(args.checkpoint_root).resolve() / execution["git_commit"]
    identity_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "git_commit": execution["git_commit"],
        "bundle_sha256": execution["bundle_sha256"],
        "launch_manifest_sha256": execution["launch_manifest_sha256"],
        "lock_sha256": lock_sha256,
        "config_sha256": config.config_hash,
        "source_integrity_sha256": source_integrity_sha256,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "chat_template_sha256": config.chat_template_sha256,
    }
    identity_path, identity_sha256 = _persist_exact_json(
        identity_dir / "execution_identity.json", identity
    )
    runtime_observation = {
        **runtime,
        "git_commit": execution["git_commit"],
        "config_sha256": config.config_hash,
        "execution_identity_sha256": identity_sha256,
        "source_integrity_sha256": source_integrity_sha256,
    }
    runtime_bytes = _canonical_bytes(runtime_observation)
    runtime_sha256 = hashlib.sha256(runtime_bytes).hexdigest()
    runtime_path, _ = _persist_exact_json(
        identity_dir / f"runtime_observation-{runtime_sha256[:16]}.json",
        runtime_observation,
    )

    checkpoint_store = ColabSplitCheckpointStore(
        repo_root=repo_root,
        checkpoint_root=identity_dir,
        scratch_root=Path(args.scratch_root),
        run_id=config.run_id,
        git_commit=execution["git_commit"],
        config_sha256=config.config_hash,
    )
    artifact_store = FAArtifactStore(repo_root)
    screened_manifests: list[Path] = []
    checkpoint_manifests: dict[str, str] = {}
    restored_splits: list[str] = []

    for spec in SOURCE_V5_SPLITS:
        if checkpoint_store.restore_split_checkpoint(spec.split):
            restored_splits.append(spec.split)
        screened_manifest = checkpoint_store.unique_manifest(
            spec.split, "screened_match"
        )
        if screened_manifest is None:
            completion_manifest = checkpoint_store.successful_completion_manifest(
                spec.split
            )
            if completion_manifest is None:
                generated = run_screening(
                    config,
                    repo_root,
                    argparse.Namespace(
                        candidates_manifest=str(repo_root / spec.candidate_path),
                        questions_manifest=str(repo_root / spec.question_path),
                        shard_id=checkpoint_store.next_screening_shard_id(
                            spec.split
                        ),
                        namespace=spec.split,
                        source_integrity_manifest=str(
                            repo_root / SOURCE_INTEGRITY_PATH
                        ),
                    ),
                )
                completion_manifest = _required_manifest(
                    generated, "shard_manifest"
                )
                completion = _verify_shard(
                    artifact_store,
                    completion_manifest,
                    namespace=spec.split,
                    record_kind="screening_completion",
                    row_count=spec.completion_count,
                )
                if generated.get("status") == "infrastructure_failure":
                    checkpoint = checkpoint_store.checkpoint_split(
                        spec.split, "failure"
                    )
                    return {
                        **generated,
                        "split": spec.split,
                        "shard_manifest": str(completion.manifest_path),
                        "checkpoint_manifest": str(checkpoint),
                        **_execution_payload(
                            identity_path,
                            identity_sha256,
                            runtime_path,
                            runtime_sha256,
                        ),
                    }
                if (
                    generated.get("status") != "generated"
                    or generated.get("count") != spec.completion_count
                ):
                    raise RuntimeError(
                        f"{spec.split} generation returned an invalid result"
                    )
                checkpoint_manifests[spec.split] = str(
                    checkpoint_store.checkpoint_split(
                        spec.split, "completion"
                    )
                )
            _verify_shard(
                artifact_store,
                completion_manifest,
                namespace=spec.split,
                record_kind="screening_completion",
                row_count=spec.completion_count,
            )
            try:
                screened = screen_entities(
                    config,
                    repo_root,
                    argparse.Namespace(
                        candidates_manifest=str(repo_root / spec.candidate_path),
                        questions_manifest=str(repo_root / spec.question_path),
                        screening_manifest=str(completion_manifest),
                        synthetic_manifest=str(repo_root / spec.synthetic_path),
                        source_integrity_manifest=str(
                            repo_root / SOURCE_INTEGRITY_PATH
                        ),
                    ),
                )
                if (
                    screened.get("status") != "screened"
                    or screened.get("count") != spec.match_count
                ):
                    raise RuntimeError(
                        f"{spec.split} entity screening returned an invalid result"
                    )
                audit_manifest = _required_manifest(
                    screened, "audit_manifest"
                )
                _verify_shard(
                    artifact_store,
                    audit_manifest,
                    namespace=spec.split,
                    record_kind="screening_audit",
                )
                screened_manifest = _required_manifest(screened, "manifest")
                _verify_shard(
                    artifact_store,
                    screened_manifest,
                    namespace=spec.split,
                    record_kind="screened_match",
                    row_count=spec.match_count,
                )
            except Exception:
                checkpoint_store.checkpoint_split(spec.split, "completion")
                raise

        _verify_complete_split(
            artifact_store, checkpoint_store, spec, screened_manifest
        )
        checkpoint_manifests[spec.split] = str(
            checkpoint_store.checkpoint_split(spec.split, "screened")
        )
        screened_manifests.append(Path(screened_manifest))

    collection_manifest = checkpoint_store.unique_manifest(
        "mechanism_train", "screened_match_collection"
    )
    matches_sha256: str | None = None
    if collection_manifest is None:
        assembled = assemble_screened_matches(
            config,
            repo_root,
            argparse.Namespace(
                screened_matches_manifest=[
                    str(path) for path in screened_manifests
                ],
                shard_id=COLLECTION_SHARD_ID,
            ),
        )
        if (
            assembled.get("status") != "assembled"
            or assembled.get("count") != ASSEMBLED_MATCH_COUNT
        ):
            raise RuntimeError("screened-match assembly returned an invalid result")
        collection_manifest = _required_manifest(assembled, "manifest")
        matches_sha256 = assembled.get("matches_sha256")
    collection = _verify_shard(
        artifact_store,
        collection_manifest,
        namespace="mechanism_train",
        record_kind="screened_match_collection",
        row_count=ASSEMBLED_MATCH_COUNT,
    )
    if matches_sha256 is None:
        matches_sha256 = _collection_matches_sha256(collection.manifest_path)
    checkpoint_manifests["collection"] = str(
        checkpoint_store.checkpoint_split("mechanism_train", "collection")
    )

    return {
        "status": "assembled",
        "manifest": str(collection.manifest_path),
        "count": ASSEMBLED_MATCH_COUNT,
        "matches_sha256": matches_sha256,
        "source_candidate_count": SOURCE_CANDIDATE_COUNT,
        "screening_completion_count": SCREENING_COMPLETION_COUNT,
        "source_revision": SOURCE_REVISION,
        "source_integrity_sha256": source_integrity_sha256,
        "checkpoint_manifests": checkpoint_manifests,
        "restored_splits": restored_splits,
        "protected_endpoints_accessed": False,
        "stopped_before": "naturalness_and_f1_f2a",
        **_execution_payload(
            identity_path,
            identity_sha256,
            runtime_path,
            runtime_sha256,
        ),
    }


def _verify_registered_design(config: FAConfig) -> None:
    expected_splits = {
        "mechanism_train": 64,
        "locked_validation": 32,
        "behavior_test": 48,
        "probe_test": 24,
        "intervention_test": 24,
    }
    if config.profile != "confirmatory":
        raise ValueError("Colab screening requires the confirmatory config")
    if dict(config.split_counts) != expected_splits:
        raise ValueError("confirmatory split counts differ from Source-v5")
    if sum(spec.candidate_count for spec in SOURCE_V5_SPLITS) != SOURCE_CANDIDATE_COUNT:
        raise RuntimeError("internal Source-v5 candidate counts are invalid")
    if (
        sum(spec.completion_count for spec in SOURCE_V5_SPLITS)
        != SCREENING_COMPLETION_COUNT
    ):
        raise RuntimeError("internal Source-v5 completion counts are invalid")
    if sum(spec.match_count for spec in SOURCE_V5_SPLITS) != ASSEMBLED_MATCH_COUNT:
        raise RuntimeError("internal Source-v5 match counts are invalid")
    for spec in SOURCE_V5_SPLITS:
        if spec.candidate_count != 2 * expected_splits[spec.split]:
            raise RuntimeError("internal Source-v5 source pool is not exactly 2x")


def _verify_execution_inputs(
    repo_root: Path, args: argparse.Namespace
) -> dict[str, str | None]:
    git_commit = str(args.git_commit)
    bundle_sha256 = str(args.bundle_sha256)
    if _COMMIT_PATTERN.fullmatch(git_commit) is None:
        raise ValueError("git commit must be a lowercase 40-character SHA")
    if _SHA256_PATTERN.fullmatch(bundle_sha256) is None:
        raise ValueError("bundle sha256 must be a lowercase 64-character digest")
    bundle_path = Path(args.bundle_path).resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Git bundle does not exist: {bundle_path}")
    if _sha256_file(bundle_path) != bundle_sha256:
        raise ValueError("Git bundle sha256 does not match the frozen identity")

    launch_sha256 = None
    launch_value = getattr(args, "launch_manifest", None)
    if launch_value:
        launch_path = Path(launch_value)
        if not launch_path.is_absolute():
            launch_path = repo_root / launch_path
        launch_path = launch_path.resolve()
        launch = _read_json_object(launch_path)
        required = {
            "schema_version",
            "git_commit",
            "bundle_file",
            "bundle_sha256",
        }
        if (
            set(launch) != required
            or launch.get("schema_version") != 1
            or launch.get("git_commit") != git_commit
            or launch.get("bundle_sha256") != bundle_sha256
            or launch.get("bundle_file") != bundle_path.name
        ):
            raise ValueError("launch manifest does not match execution identity")
        launch_sha256 = _sha256_file(launch_path)

    _verify_frozen_checkout(repo_root, git_commit, bundle_path)
    return {
        "git_commit": git_commit,
        "bundle_sha256": bundle_sha256,
        "launch_manifest_sha256": launch_sha256,
    }


def _verify_frozen_checkout(
    repo_root: Path, git_commit: str, bundle_path: Path
) -> None:
    if _git_output(repo_root, "rev-parse", "HEAD") != git_commit:
        raise RuntimeError("checkout HEAD does not match the frozen commit")
    _git_check(repo_root, "diff", "--quiet")
    _git_check(repo_root, "diff", "--cached", "--quiet")
    untracked = _git_output(
        repo_root, "ls-files", "--others", "--exclude-standard"
    ).splitlines()
    unexpected = [
        path
        for path in untracked
        if not path.startswith("runs/familiarity_answerability/")
    ]
    if unexpected:
        raise RuntimeError(f"unexpected untracked checkout files: {unexpected}")
    _git_check(repo_root, "bundle", "verify", str(bundle_path))
    bundle_heads = _git_output(
        repo_root, "bundle", "list-heads", str(bundle_path)
    ).splitlines()
    head_commits = {
        line.split(maxsplit=1)[0]
        for line in bundle_heads
        if line.strip()
    }
    if git_commit not in head_commits:
        raise RuntimeError("frozen commit is not advertised by the Git bundle")


def _verify_source_v5(repo_root: Path, config: FAConfig) -> str:
    integrity_path = repo_root / SOURCE_INTEGRITY_PATH
    integrity = _read_json_object(integrity_path)
    if (
        integrity.get("schema_version") != 1
        or integrity.get("source_revision") != SOURCE_REVISION
    ):
        raise ValueError("Source-v5 integrity revision is invalid")
    materialized = integrity.get("materialized_files")
    synthetic_files = integrity.get("synthetic_files")
    expected_splits = {spec.split for spec in SOURCE_V5_SPLITS}
    if (
        not isinstance(materialized, Mapping)
        or set(materialized) != expected_splits
        or not isinstance(synthetic_files, Mapping)
        or set(synthetic_files) != expected_splits
    ):
        raise ValueError("Source-v5 integrity must cover exactly five splits")

    for spec in SOURCE_V5_SPLITS:
        source_record = materialized[spec.split]
        synthetic_record = synthetic_files[spec.split]
        if not isinstance(source_record, Mapping) or not isinstance(
            synthetic_record, Mapping
        ):
            raise ValueError("Source-v5 integrity split record is invalid")
        expected_source = {
            "candidate_manifest": str(spec.candidate_path),
            "question_manifest": str(spec.question_path),
        }
        if any(
            source_record.get(key) != value
            for key, value in expected_source.items()
        ) or synthetic_record.get("path") != str(spec.synthetic_path):
            raise ValueError("Source-v5 integrity path changed")
        candidate_path = repo_root / spec.candidate_path
        question_path = repo_root / spec.question_path
        synthetic_path = repo_root / spec.synthetic_path
        if (
            source_record.get("candidate_sha256")
            != _sha256_file(candidate_path)
            or source_record.get("question_sha256")
            != _sha256_file(question_path)
            or synthetic_record.get("sha256") != _sha256_file(synthetic_path)
        ):
            raise ValueError("Source-v5 materialized file hash changed")
        candidates = _read_json_array(candidate_path)
        questions = _read_json_array(question_path)
        synthetics = _read_json_array(synthetic_path)
        if len(candidates) != spec.candidate_count:
            raise ValueError(f"{spec.split} candidate count is not frozen")
        if len(questions) != spec.completion_count:
            raise ValueError(f"{spec.split} question count is not frozen")
        if len(synthetics) != spec.completion_count:
            raise ValueError(f"{spec.split} synthetic count is not frozen")
        if any(row.get("split") != spec.split for row in candidates):
            raise ValueError(f"{spec.split} candidate split identity changed")
        if any(row.get("split") != spec.split for row in synthetics):
            raise ValueError(f"{spec.split} synthetic split identity changed")
        if spec.candidate_count != 2 * config.split_counts[spec.split]:
            raise ValueError(f"{spec.split} source pool is not exactly 2x")

    for path_key, sha_key in (
        ("source_snapshot", "source_snapshot_sha256"),
        ("synthetic_snapshot", "synthetic_snapshot_sha256"),
    ):
        relative = Path(str(integrity.get(path_key, "")))
        path = (repo_root / relative).resolve()
        if (
            relative.is_absolute()
            or not path.is_relative_to(repo_root)
            or not path.is_file()
            or integrity.get(sha_key) != _sha256_file(path)
        ):
            raise ValueError(f"Source-v5 {path_key} does not verify")
    return _sha256_file(integrity_path)


def _verify_complete_split(
    artifact_store: FAArtifactStore,
    checkpoint_store: ColabSplitCheckpointStore,
    spec: SourceV5Split,
    screened_manifest: str | Path,
) -> None:
    completion_manifest = checkpoint_store.successful_completion_manifest(
        spec.split
    )
    if completion_manifest is None:
        raise ValueError(f"{spec.split} has no successful completion artifact")
    _verify_shard(
        artifact_store,
        completion_manifest,
        namespace=spec.split,
        record_kind="screening_completion",
        row_count=spec.completion_count,
    )
    audit_manifest = checkpoint_store.unique_manifest(
        spec.split, "screening_audit"
    )
    if audit_manifest is None:
        raise ValueError(f"{spec.split} has no screening audit artifact")
    _verify_shard(
        artifact_store,
        audit_manifest,
        namespace=spec.split,
        record_kind="screening_audit",
    )
    _verify_shard(
        artifact_store,
        screened_manifest,
        namespace=spec.split,
        record_kind="screened_match",
        row_count=spec.match_count,
    )


def _verify_shard(
    store: FAArtifactStore,
    manifest: str | Path,
    *,
    namespace: str,
    record_kind: str,
    row_count: int | None = None,
):
    shard = store.verify_shard(manifest)
    if shard.namespace != namespace or shard.record_kind != record_kind:
        raise ValueError(f"unexpected {record_kind} artifact identity")
    if row_count is not None and shard.row_count != row_count:
        raise ValueError(f"unexpected {record_kind} artifact row count")
    return shard


def _required_manifest(payload: Mapping[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"transaction omitted {key}")
    return Path(value)


def _collection_matches_sha256(manifest_path: Path) -> str:
    manifest = _read_json_object(manifest_path)
    lineage = manifest.get("lineage")
    value = lineage.get("matches_sha256") if isinstance(lineage, Mapping) else None
    if isinstance(value, str) and _SHA256_PATTERN.fullmatch(value):
        return value
    shard_sha256 = manifest.get("sha256")
    if not isinstance(shard_sha256, str) or _SHA256_PATTERN.fullmatch(
        shard_sha256
    ) is None:
        raise ValueError("collection manifest has no verifiable content hash")
    return shard_sha256


def _execution_payload(
    identity_path: Path,
    identity_sha256: str,
    runtime_path: Path,
    runtime_sha256: str,
) -> dict[str, str]:
    return {
        "execution_identity_path": str(identity_path),
        "execution_identity_sha256": identity_sha256,
        "runtime_observation_path": str(runtime_path),
        "runtime_observation_sha256": runtime_sha256,
    }


def _persist_exact_json(path: Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    payload = _canonical_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"persisted identity changed: {path}")
        return path, digest
    partial = path.with_name(f".{path.name}.{digest[:16]}.partial")
    partial.write_bytes(payload)
    os.replace(partial, path)
    return path, digest


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON object is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON array is unreadable: {path}") from error
    if not isinstance(value, list) or any(
        not isinstance(row, dict) for row in value
    ):
        raise ValueError(f"expected JSON object array: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"file is unreadable: {path}") from error
    return digest.hexdigest()


def _git_output(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {message}")
    return result.stdout.strip()


def _git_check(repo_root: Path, *arguments: str) -> None:
    _git_output(repo_root, *arguments)
