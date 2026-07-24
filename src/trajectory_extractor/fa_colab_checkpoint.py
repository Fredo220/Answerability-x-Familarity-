"""Content-addressed Colab checkpoints for confirmatory screening shards."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from trajectory_extractor.fa_artifacts import FAArtifactStore, SealedShard


CHECKPOINT_STAGE_RANK = {
    "failure": 0,
    "completion": 1,
    "screened": 2,
    "collection": 3,
}
CHECKPOINT_STAGE_KINDS = {
    "failure": frozenset({"screening_completion"}),
    "completion": frozenset({"screening_completion", "screening_audit"}),
    "screened": frozenset(
        {"screening_completion", "screening_audit", "screened_match"}
    ),
    "collection": frozenset(
        {
            "screening_completion",
            "screening_audit",
            "screened_match",
            "screened_match_collection",
        }
    ),
}


class ColabSplitCheckpointStore:
    """Persist immutable split artifacts without replacing prior checkpoints."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        checkpoint_root: str | Path,
        scratch_root: str | Path,
        run_id: str,
        git_commit: str,
        config_sha256: str,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.checkpoint_root = Path(checkpoint_root).resolve()
        self.scratch_root = Path(scratch_root).resolve()
        self.run_id = run_id
        self.git_commit = git_commit
        self.config_sha256 = config_sha256
        self.store = FAArtifactStore(self.repo_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.scratch_root.mkdir(parents=True, exist_ok=True)

    def verified_split_shards(self, split: str) -> tuple[SealedShard, ...]:
        directory = self.split_artifact_dir(split)
        if not directory.is_dir():
            return ()
        return tuple(
            self.store.verify_shard(path)
            for path in sorted(directory.glob("*.manifest.json"))
        )

    def split_artifact_dir(self, split: str) -> Path:
        return (
            self.repo_root
            / "runs"
            / "familiarity_answerability"
            / self.run_id
            / "shards"
            / split
        )

    def successful_completion_manifest(self, split: str) -> Path | None:
        return self._successful_completion_manifest(
            self.verified_split_shards(split)
        )

    def unique_manifest(self, split: str, record_kind: str) -> Path | None:
        matches = [
            shard.manifest_path
            for shard in self.verified_split_shards(split)
            if shard.record_kind == record_kind
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError(
                f"ambiguous {record_kind} shards in {split}: {matches}"
            )
        return matches[0]

    def next_screening_shard_id(self, split: str) -> str:
        base = f"confirmatory-{split}-screening-v1"
        attempts = [
            shard
            for shard in self.verified_split_shards(split)
            if shard.record_kind == "screening_completion"
        ]
        return base if not attempts else f"{base}-retry-{len(attempts):02d}"

    def checkpoint_split(self, split: str, stage: str) -> Path:
        if stage not in CHECKPOINT_STAGE_RANK:
            raise ValueError(f"unknown checkpoint stage: {stage}")
        all_shards = self.verified_split_shards(split)
        shards = tuple(
            shard
            for shard in all_shards
            if shard.record_kind in CHECKPOINT_STAGE_KINDS[stage]
        )
        if not shards:
            raise ValueError(f"no verified shards exist for {split}")
        observed_stage = self._local_checkpoint_stage(shards)
        if observed_stage < CHECKPOINT_STAGE_RANK[stage]:
            raise ValueError(
                f"{split} has stage {observed_stage}, below requested {stage}"
            )

        local_archive = self.scratch_root / f"screening-{split}-{stage}.zip"
        local_archive.unlink(missing_ok=True)
        with zipfile.ZipFile(
            local_archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as bundle:
            for path in self._checkpoint_files(shards):
                bundle.write(path, path.relative_to(self.repo_root))

        archive_sha256 = _sha256_file(local_archive)
        prefix = f"screening-{split}-{stage}-{archive_sha256[:16]}"
        archive = self.checkpoint_root / f"{prefix}.zip"
        _copy_content_addressed(local_archive, archive, archive_sha256)

        metadata = {
            "git_commit": self.git_commit,
            "config_sha256": self.config_sha256,
            "run_id": self.run_id,
            "split": split,
            "stage": stage,
            "stage_rank": CHECKPOINT_STAGE_RANK[stage],
            "archive_file": archive.name,
            "archive_sha256": archive_sha256,
            "artifact_directory": str(
                self.split_artifact_dir(split).relative_to(self.repo_root)
            ),
            "members": self._member_sha256_map(shards),
            "shards": self._shard_identity_map(shards),
        }
        metadata_bytes = _canonical_bytes(metadata)
        metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
        metadata_path = (
            self.checkpoint_root
            / f"{prefix}-{metadata_sha256[:16]}.checkpoint.json"
        )
        _write_content_addressed(metadata_path, metadata_bytes)

        pointer = self.checkpoint_root / f"screening-{split}-LATEST.json"
        pointer_bytes = _canonical_bytes(
            {
                "metadata_file": metadata_path.name,
                "metadata_sha256": metadata_sha256,
            }
        )
        pointer_partial = self.checkpoint_root / (
            f".{pointer.name}.{metadata_sha256[:16]}.partial"
        )
        pointer_partial.write_bytes(pointer_bytes)
        os.replace(pointer_partial, pointer)
        return metadata_path

    def restore_split_checkpoint(self, split: str) -> bool:
        candidates = self._checkpoint_candidates(split)
        local = self.verified_split_shards(split)
        local_stage = self._local_checkpoint_stage(local)
        if not candidates:
            return local_stage >= CHECKPOINT_STAGE_RANK["completion"]

        checkpoint_stage, _, _, archive, metadata = candidates[-1]
        if local:
            comparable = tuple(
                shard
                for shard in local
                if shard.record_kind in CHECKPOINT_STAGE_KINDS[metadata["stage"]]
            )
            local_identity = self._shard_identity_map(comparable)
            if local_identity == metadata["shards"]:
                return local_stage >= CHECKPOINT_STAGE_RANK["completion"]
            if all(
                local_identity.get(path) == identity
                for path, identity in metadata["shards"].items()
            ):
                return local_stage >= CHECKPOINT_STAGE_RANK["completion"]
            if local_stage <= checkpoint_stage:
                raise ValueError("local split differs from its checkpoint")
            return local_stage >= CHECKPOINT_STAGE_RANK["completion"]

        expected_directory = self.split_artifact_dir(split).relative_to(
            self.repo_root
        )
        if metadata["artifact_directory"] != str(expected_directory):
            raise ValueError("checkpoint artifact directory mismatch")
        with tempfile.TemporaryDirectory(
            prefix=f"fa-restore-{split}-",
            dir=self.scratch_root,
        ) as temporary:
            staging_root = Path(temporary)
            with zipfile.ZipFile(archive) as bundle:
                member_names = tuple(sorted(member.filename for member in bundle.infolist()))
                if member_names != tuple(sorted(metadata["members"])):
                    raise ValueError("checkpoint archive member list mismatch")
                expected_prefix = f"{expected_directory.as_posix()}/"
                if any(not name.startswith(expected_prefix) for name in member_names):
                    raise ValueError("checkpoint archive member is outside the split")
                staged_split = (staging_root / expected_directory).resolve()
                for name in member_names:
                    (staging_root / name).resolve().relative_to(staged_split)
                bundle.extractall(staging_root)
            for relative, expected_sha256 in metadata["members"].items():
                path = staging_root / relative
                if not path.is_file() or _sha256_file(path) != expected_sha256:
                    raise ValueError("checkpoint member hash mismatch")
            staged_store = FAArtifactStore(staging_root)
            staged_shards = tuple(
                staged_store.verify_shard(path)
                for path in sorted(
                    (staging_root / expected_directory).glob("*.manifest.json")
                )
            )
            staged_identity = self._shard_identity_map(
                staged_shards,
                root=staging_root,
            )
            if staged_identity != metadata["shards"]:
                raise ValueError(f"checkpoint shard mismatch for {split}")
            staged_members = self._member_sha256_map(
                staged_shards,
                root=staging_root,
            )
            if staged_members != metadata["members"]:
                raise ValueError(f"checkpoint contains non-shard files for {split}")
            destination = self.split_artifact_dir(split)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_root / expected_directory, destination)

        restored = self.verified_split_shards(split)
        if self._shard_identity_map(restored) != metadata["shards"]:
            raise ValueError(f"restored checkpoint mismatch for {split}")
        return (
            self._local_checkpoint_stage(restored)
            >= CHECKPOINT_STAGE_RANK["completion"]
        )

    def _checkpoint_candidates(
        self,
        split: str,
    ) -> tuple[tuple[int, int, str, Path, dict[str, Any]], ...]:
        candidates = []
        pattern = f"screening-{split}-*.checkpoint.json"
        for metadata_path in sorted(self.checkpoint_root.glob(pattern)):
            metadata_bytes = metadata_path.read_bytes()
            metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
            if f"-{metadata_sha256[:16]}.checkpoint.json" not in metadata_path.name:
                raise ValueError("checkpoint metadata hash mismatch")
            metadata = json.loads(metadata_bytes)
            required = {
                "git_commit",
                "config_sha256",
                "run_id",
                "split",
                "stage",
                "stage_rank",
                "archive_file",
                "archive_sha256",
                "artifact_directory",
                "members",
                "shards",
            }
            if set(metadata) != required:
                raise ValueError("checkpoint metadata schema is invalid")
            if metadata.get("git_commit") != self.git_commit:
                raise ValueError("checkpoint Git commit mismatch")
            if metadata.get("config_sha256") != self.config_sha256:
                raise ValueError("checkpoint config mismatch")
            if metadata.get("run_id") != self.run_id:
                raise ValueError("checkpoint run ID mismatch")
            if metadata.get("split") != split:
                raise ValueError("checkpoint split mismatch")
            stage = metadata.get("stage")
            if stage not in CHECKPOINT_STAGE_RANK:
                raise ValueError("checkpoint stage is invalid")
            if metadata.get("stage_rank") != CHECKPOINT_STAGE_RANK[stage]:
                raise ValueError("checkpoint stage rank is invalid")
            archive_name = metadata.get("archive_file")
            if not isinstance(archive_name, str) or Path(archive_name).name != archive_name:
                raise ValueError("checkpoint archive path is invalid")
            archive = self.checkpoint_root / archive_name
            if not archive.is_file():
                raise ValueError("checkpoint archive is missing")
            if metadata.get("archive_sha256") != _sha256_file(archive):
                raise ValueError("checkpoint archive hash mismatch")
            members = metadata.get("members")
            shards = metadata.get("shards")
            if not isinstance(members, dict) or not members:
                raise ValueError("checkpoint member map is invalid")
            if not isinstance(shards, dict) or not shards:
                raise ValueError("checkpoint shard map is invalid")
            for identity in shards.values():
                if (
                    not isinstance(identity, dict)
                    or set(identity)
                    != {"data_sha256", "manifest_sha256", "record_kind"}
                    or identity["record_kind"] not in CHECKPOINT_STAGE_KINDS[stage]
                ):
                    raise ValueError("checkpoint shard identity is invalid")
            expected_directory = str(
                self.split_artifact_dir(split).relative_to(self.repo_root)
            )
            if metadata.get("artifact_directory") != expected_directory:
                raise ValueError("checkpoint artifact directory mismatch")
            candidates.append(
                (
                    CHECKPOINT_STAGE_RANK[stage],
                    len(shards),
                    metadata_path.name,
                    archive,
                    metadata,
                )
            )
        return tuple(sorted(candidates))

    @staticmethod
    def _successful_completion_manifest(
        shards: Iterable[SealedShard],
    ) -> Path | None:
        successes = []
        for shard in shards:
            if shard.record_kind != "screening_completion":
                continue
            rows = [
                json.loads(line)
                for line in shard.data_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if rows and all(row.get("status") == "completed" for row in rows):
                successes.append(shard.manifest_path)
        if not successes:
            return None
        if len(successes) != 1:
            raise ValueError(
                f"multiple successful screening completions: {successes}"
            )
        return successes[0]

    def _local_checkpoint_stage(self, shards: Iterable[SealedShard]) -> int:
        shards = tuple(shards)
        kinds = {shard.record_kind for shard in shards}
        if "screened_match_collection" in kinds:
            return CHECKPOINT_STAGE_RANK["collection"]
        if "screened_match" in kinds:
            return CHECKPOINT_STAGE_RANK["screened"]
        if self._successful_completion_manifest(shards) is not None:
            return CHECKPOINT_STAGE_RANK["completion"]
        if "screening_completion" in kinds:
            return CHECKPOINT_STAGE_RANK["failure"]
        return -1

    @staticmethod
    def _checkpoint_files(shards: Iterable[SealedShard]) -> tuple[Path, ...]:
        paths = {
            path
            for shard in shards
            for path in (shard.data_path, shard.manifest_path)
        }
        return tuple(sorted(paths))

    def _member_sha256_map(
        self,
        shards: Iterable[SealedShard],
        *,
        root: Path | None = None,
    ) -> dict[str, str]:
        root = self.repo_root if root is None else root
        return {
            str(path.relative_to(root)): _sha256_file(path)
            for path in self._checkpoint_files(shards)
        }

    def _shard_identity_map(
        self,
        shards: Iterable[SealedShard],
        *,
        root: Path | None = None,
    ) -> dict[str, dict[str, str]]:
        root = self.repo_root if root is None else root
        return {
            str(shard.manifest_path.relative_to(root)): {
                "data_sha256": shard.sha256,
                "manifest_sha256": _sha256_file(shard.manifest_path),
                "record_kind": shard.record_kind,
            }
            for shard in shards
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def _copy_content_addressed(
    source: Path,
    destination: Path,
    expected_sha256: str,
) -> None:
    if destination.exists():
        if _sha256_file(destination) != expected_sha256:
            raise ValueError("content-addressed checkpoint archive changed")
        return
    partial = destination.with_suffix(f"{destination.suffix}.partial")
    shutil.copy2(source, partial)
    if _sha256_file(partial) != expected_sha256:
        raise ValueError("checkpoint archive copy hash mismatch")
    os.replace(partial, destination)


def _write_content_addressed(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("content-addressed checkpoint metadata changed")
        return
    partial = path.with_suffix(f"{path.suffix}.partial")
    partial.write_bytes(payload)
    os.replace(partial, path)
