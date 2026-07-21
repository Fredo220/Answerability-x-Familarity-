from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_NAMESPACES = frozenset(
    {
        "pilot",
        "mechanism_train",
        "locked_validation",
        "behavior_test",
        "probe_test",
        "intervention_test",
        "circuit_dev",
    }
)
_ENDPOINTS = frozenset({"behavior_test", "probe_test", "intervention_test"})
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class SealedShard:
    namespace: str
    shard_id: str
    data_path: Path
    manifest_path: Path
    sha256: str
    row_count: int


@dataclass(frozen=True)
class UnlockReceipt:
    endpoint: str
    lease_id: str
    state: str
    preregistration_hash: str
    selection_manifest_hash: str


class FAArtifactStore:
    """Immutable artifacts for the Familiarity-vs-Answerability study only."""

    def __init__(self, root: str | Path):
        self.root = Path(root).absolute()

    def write_completed_shard(
        self,
        run_id: str,
        namespace: str,
        shard_id: str,
        rows: Iterable[Mapping[str, Any]],
        lineage: Mapping[str, Any],
    ) -> SealedShard:
        run = _safe_id(run_id, "run_id")
        namespace = _namespace(namespace)
        shard = _safe_id(shard_id, "shard_id")
        if not isinstance(lineage, Mapping):
            raise ValueError("lineage must be a mapping")
        payload_rows = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("shard rows must be mappings")
            payload_rows.append(_canonical_json(dict(row)) + b"\n")
        payload = b"".join(payload_rows)
        data_path = self._shard_path(run, namespace, shard)
        manifest_path = data_path.with_name(f"{data_path.name}.manifest.json")
        self._reject_existing(data_path)
        self._reject_existing(manifest_path)
        digest = _sha256_bytes(payload)
        manifest = {
            "schema_version": 1,
            "run_id": run,
            "namespace": namespace,
            "shard_id": shard,
            "data_file": data_path.name,
            "sha256": digest,
            "row_count": len(payload_rows),
            "lineage": dict(lineage),
        }
        _canonical_json(manifest)
        data_published = False
        try:
            self._exclusive_write(data_path, payload)
            data_published = True
            self._exclusive_write(manifest_path, _canonical_json(manifest))
        except BaseException:
            if data_published:
                self._remove_regular(data_path)
            raise
        return SealedShard(namespace, shard, data_path, manifest_path, digest, len(payload_rows))

    def verify_shard(self, manifest_path: str | Path) -> SealedShard:
        manifest_path = Path(manifest_path).absolute()
        self._require_under_root(manifest_path)
        self._require_regular_file(manifest_path, "shard manifest")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("shard manifest is unreadable") from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("shard manifest has an invalid schema")
        run = _safe_id(manifest.get("run_id"), "shard manifest run_id")
        namespace = _namespace(manifest.get("namespace"))
        shard = _safe_id(manifest.get("shard_id"), "shard manifest shard_id")
        data_path = self._shard_path(run, namespace, shard)
        expected_manifest = data_path.with_name(f"{data_path.name}.manifest.json")
        if manifest_path != expected_manifest:
            raise ValueError("shard manifest path does not match its identity")
        if manifest.get("data_file") != data_path.name:
            raise ValueError("shard manifest data path is invalid")
        digest = manifest.get("sha256")
        _sha256_value(digest, "shard manifest sha256")
        row_count = manifest.get("row_count")
        if type(row_count) is not int or row_count < 0:
            raise ValueError("shard manifest row_count is invalid")
        if not isinstance(manifest.get("lineage"), dict):
            raise ValueError("shard manifest lineage is invalid")
        self._require_regular_file(data_path, "shard data")
        data = data_path.read_bytes()
        if _sha256_bytes(data) != digest:
            raise ValueError("shard hash mismatch")
        if data.count(b"\n") != row_count:
            raise ValueError("shard row count mismatch")
        return SealedShard(namespace, shard, data_path, manifest_path, digest, row_count)

    def resume_verified_shards(self, run_id: str, namespace: str) -> tuple[SealedShard, ...]:
        """Return only durably completed shards; corrupt sidecars stop resumption."""
        run = _safe_id(run_id, "run_id")
        namespace = _namespace(namespace)
        directory = self._base() / run / "shards" / namespace
        self._assert_existing_ancestors_are_real(directory)
        if not directory.exists():
            return ()
        self._require_directory(directory, "FA shard directory")
        manifests = sorted(directory.glob("*.jsonl.manifest.json"))
        return tuple(self.verify_shard(path) for path in manifests)

    def seal_endpoint(
        self,
        endpoint: str,
        artifacts: Iterable[SealedShard],
        parents: Mapping[str, str],
    ) -> Path:
        endpoint = _endpoint(endpoint)
        validated_parents = _parents(parents)
        sealed = [self._verified_artifact(artifact) for artifact in artifacts]
        if not sealed:
            raise ValueError("endpoint requires at least one verified shard")
        run_ids = {self._run_id_for_shard(artifact) for artifact in sealed}
        if len(run_ids) != 1:
            raise ValueError("endpoint artifacts must belong to one run")
        if any(artifact.namespace != endpoint for artifact in sealed):
            raise ValueError("endpoint artifacts must use their matching namespace")
        if len({artifact.manifest_path for artifact in sealed}) != len(sealed):
            raise ValueError("endpoint artifacts must be unique")
        run_id = run_ids.pop()
        path = self._endpoint_path(run_id, endpoint, "sealed")
        record = {
            "schema_version": 1,
            "state": "sealed",
            "run_id": run_id,
            "endpoint": endpoint,
            "parents": validated_parents,
            "artifacts": [
                {
                    "manifest_path": str(artifact.manifest_path.relative_to(self.root)),
                    "sha256": artifact.sha256,
                }
                for artifact in sorted(sealed, key=lambda item: item.manifest_path)
            ],
        }
        self._exclusive_write(path, _canonical_json(record))
        return path

    def unlock_endpoint(
        self,
        endpoint: str,
        preregistration_hash: str,
        selection_manifest_hash: str,
    ) -> UnlockReceipt:
        endpoint = _endpoint(endpoint)
        _sha256_value(preregistration_hash, "preregistration_hash")
        _sha256_value(selection_manifest_hash, "selection_manifest_hash")
        sealed_path = self._find_endpoint_path(endpoint, "sealed")
        sealed = self._read_endpoint_record(sealed_path, endpoint, "sealed")
        self._verify_endpoint_artifacts(sealed)
        unlocked_path = self._endpoint_path(sealed["run_id"], endpoint, "unlocked_once")
        if unlocked_path.exists() or unlocked_path.is_symlink():
            raise ValueError(f"{endpoint} is already unlocked")
        if sealed["parents"]["preregistration"] != preregistration_hash:
            raise ValueError("preregistration hash does not match sealed endpoint")
        if sealed["parents"]["selection_manifest"] != selection_manifest_hash:
            raise ValueError(
                f"{endpoint.removesuffix('_test')} selection manifest hash does not match sealed endpoint"
            )
        receipt = UnlockReceipt(
            endpoint=endpoint,
            lease_id=uuid.uuid4().hex,
            state="unlocked_once",
            preregistration_hash=preregistration_hash,
            selection_manifest_hash=selection_manifest_hash,
        )
        self._exclusive_write(
            unlocked_path,
            _canonical_json(
                {
                    "schema_version": 1,
                    "state": receipt.state,
                    "run_id": sealed["run_id"],
                    "endpoint": endpoint,
                    "lease_id": receipt.lease_id,
                    "preregistration_hash": preregistration_hash,
                    "selection_manifest_hash": selection_manifest_hash,
                }
            ),
        )
        return receipt

    def mark_evaluated(self, receipt: UnlockReceipt, metrics_path: str | Path) -> Path:
        if not isinstance(receipt, UnlockReceipt) or receipt.state != "unlocked_once":
            raise ValueError("evaluation requires an unlocked_once receipt")
        endpoint = _endpoint(receipt.endpoint)
        _sha256_value(receipt.preregistration_hash, "receipt preregistration_hash")
        _sha256_value(receipt.selection_manifest_hash, "receipt selection_manifest_hash")
        if not isinstance(receipt.lease_id, str) or not re.fullmatch(r"[0-9a-f]{32}", receipt.lease_id):
            raise ValueError("receipt lease_id is invalid")
        sealed_path = self._find_endpoint_path(endpoint, "sealed")
        sealed = self._read_endpoint_record(sealed_path, endpoint, "sealed")
        self._verify_endpoint_artifacts(sealed)
        run_id = sealed["run_id"]
        closed_path = self._endpoint_path(run_id, endpoint, "closed")
        if closed_path.exists() or closed_path.is_symlink():
            raise ValueError(f"{endpoint} is already closed")
        unlocked_path = self._endpoint_path(run_id, endpoint, "unlocked_once")
        unlocked = self._read_endpoint_record(unlocked_path, endpoint, "unlocked_once")
        if (
            unlocked.get("lease_id") != receipt.lease_id
            or unlocked.get("preregistration_hash") != receipt.preregistration_hash
            or unlocked.get("selection_manifest_hash") != receipt.selection_manifest_hash
        ):
            raise ValueError("unlock receipt does not match the endpoint lease")
        metric_data = Path(metrics_path).absolute()
        self._require_under_root(metric_data)
        metric_manifest = metric_data.with_name(f"{metric_data.name}.manifest.json")
        metrics = self.verify_shard(metric_manifest)
        if metrics.data_path != metric_data:
            raise ValueError("metrics path does not match its verified sidecar")
        if metrics.namespace != endpoint:
            raise ValueError("metrics shard must use the matching endpoint namespace")
        if self._run_id_for_shard(metrics) != run_id:
            raise ValueError("metrics shard must belong to the endpoint run")
        sealed_inputs = {
            self._path_from_root_record(item["manifest_path"], "endpoint artifact manifest")
            for item in sealed["artifacts"]
        }
        if metrics.manifest_path in sealed_inputs:
            raise ValueError("metrics shard must not reuse a sealed input")
        evaluated_path = self._endpoint_path(run_id, endpoint, "evaluated")
        record = {
            "schema_version": 1,
            "state": "evaluated",
            "run_id": run_id,
            "endpoint": endpoint,
            "lease_id": receipt.lease_id,
            "metrics_manifest_path": str(metrics.manifest_path.relative_to(self.root)),
            "metrics_sha256": metrics.sha256,
        }
        self._exclusive_write(evaluated_path, _canonical_json(record))
        return evaluated_path

    def close_endpoint(self, endpoint: str) -> Path:
        endpoint = _endpoint(endpoint)
        sealed_path = self._find_endpoint_path(endpoint, "sealed")
        sealed = self._read_endpoint_record(sealed_path, endpoint, "sealed")
        run_id = sealed["run_id"]
        closed_path = self._endpoint_path(run_id, endpoint, "closed")
        if closed_path.exists() or closed_path.is_symlink():
            raise ValueError(f"{endpoint} is already closed")
        evaluated_path = self._endpoint_path(run_id, endpoint, "evaluated")
        evaluated = self._read_endpoint_record(evaluated_path, endpoint, "evaluated")
        if evaluated["run_id"] != run_id:
            raise ValueError("evaluated endpoint state belongs to another run")
        metrics_manifest = self._path_from_root_record(
            evaluated.get("metrics_manifest_path"), "metrics manifest"
        )
        metrics = self.verify_shard(metrics_manifest)
        if metrics.namespace != endpoint or metrics.sha256 != evaluated.get("metrics_sha256"):
            raise ValueError("evaluated endpoint metrics no longer verify")
        self._exclusive_write(
            closed_path,
            _canonical_json(
                {
                    "schema_version": 1,
                    "state": "closed",
                    "run_id": run_id,
                    "endpoint": endpoint,
                    "evaluated_sha256": _sha256_bytes(evaluated_path.read_bytes()),
                }
            ),
        )
        return closed_path

    def _verified_artifact(self, artifact: SealedShard) -> SealedShard:
        if not isinstance(artifact, SealedShard):
            raise ValueError("endpoint artifacts must be SealedShard records")
        verified = self.verify_shard(artifact.manifest_path)
        if verified != artifact:
            raise ValueError("endpoint artifact record does not match its verified manifest")
        return verified

    def _run_id_for_shard(self, shard: SealedShard) -> str:
        relative = shard.data_path.relative_to(self._base())
        if len(relative.parts) != 4 or relative.parts[1:3] != ("shards", shard.namespace):
            raise ValueError("shard path is outside the FA artifact namespace")
        return _safe_id(relative.parts[0], "run_id")

    def _shard_path(self, run_id: str, namespace: str, shard_id: str) -> Path:
        return self._base() / run_id / "shards" / namespace / f"{shard_id}.jsonl"

    def _endpoint_path(self, run_id: str, endpoint: str, state: str) -> Path:
        return self._base() / run_id / "endpoints" / endpoint / f"{state}.json"

    def _find_endpoint_path(self, endpoint: str, state: str) -> Path:
        base = self._base()
        if not base.exists() or base.is_symlink():
            raise ValueError(f"{endpoint} is not sealed")
        matches = []
        for candidate in base.iterdir():
            if candidate.is_symlink() or not candidate.is_dir():
                raise ValueError("FA artifact root contains an unsafe run entry")
            try:
                _safe_id(candidate.name, "run_id")
            except ValueError:
                raise ValueError("FA artifact root contains an unsafe run entry") from None
            path = self._endpoint_path(candidate.name, endpoint, state)
            if path.exists() or path.is_symlink():
                matches.append(path)
        if not matches:
            raise ValueError(f"{endpoint} is not sealed")
        if len(matches) != 1:
            raise ValueError(f"{endpoint} is ambiguous across runs")
        return matches[0]

    def _read_endpoint_record(self, path: Path, endpoint: str, state: str) -> dict[str, Any]:
        self._require_regular_file(path, "endpoint state")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("endpoint state is unreadable") from error
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != 1
            or record.get("state") != state
            or record.get("endpoint") != endpoint
        ):
            raise ValueError("endpoint state has an invalid schema")
        run_id = _safe_id(record.get("run_id"), "endpoint run_id")
        if path != self._endpoint_path(run_id, endpoint, state):
            raise ValueError("endpoint state path does not match its identity")
        if state == "sealed":
            record["parents"] = _parents(record.get("parents"))
        return record

    def _verify_endpoint_artifacts(self, sealed: Mapping[str, Any]) -> None:
        artifacts = sealed.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("sealed endpoint has invalid artifacts")
        manifests = []
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != {"manifest_path", "sha256"}:
                raise ValueError("sealed endpoint has invalid artifacts")
            manifest_path = self._path_from_root_record(
                artifact["manifest_path"], "endpoint artifact manifest"
            )
            shard = self.verify_shard(manifest_path)
            if shard.namespace != sealed["endpoint"] or shard.sha256 != artifact["sha256"]:
                raise ValueError("sealed endpoint artifact no longer verifies")
            if self._run_id_for_shard(shard) != sealed["run_id"]:
                raise ValueError("sealed endpoint artifact belongs to another run")
            manifests.append(manifest_path)
        if len(set(manifests)) != len(manifests):
            raise ValueError("sealed endpoint has duplicate artifacts")

    def _path_from_root_record(self, value: object, label: str) -> Path:
        if not isinstance(value, str):
            raise ValueError(f"{label} path is invalid")
        relative = Path(value)
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"{label} path is invalid")
        path = self.root / relative
        self._require_under_root(path)
        return path

    def _exclusive_write(self, destination: Path, payload: bytes) -> None:
        self._ensure_directory(destination.parent)
        self._reject_existing(destination)
        temporary: Path | None = None
        published = False
        try:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            temporary = Path(name)
            try:
                _write_all(descriptor, payload)
                os.fchmod(descriptor, 0o644)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.link(temporary, destination)
            published = True
            temporary.unlink()
            temporary = None
            _fsync_directory(destination.parent)
        except BaseException:
            if published:
                self._remove_regular(destination)
            raise
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _ensure_directory(self, directory: Path) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._require_directory(self.root, "FA artifact root")
        try:
            relative = directory.relative_to(self.root)
        except ValueError as error:
            raise ValueError("artifact path escapes the FA artifact root") from error
        current = self.root
        for part in relative.parts:
            current = current / part
            try:
                os.mkdir(current)
            except FileExistsError:
                pass
            self._require_directory(current, "FA artifact directory")
            _fsync_directory(current.parent)

    def _base(self) -> Path:
        return self.root / "runs" / "familiarity_answerability"

    def _require_under_root(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("artifact path escapes the FA artifact root") from error

    def _assert_existing_ancestors_are_real(self, path: Path) -> None:
        self._require_under_root(path)
        current = self.root
        if current.exists():
            self._require_directory(current, "FA artifact root")
        try:
            relative = path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("artifact path escapes the FA artifact root") from error
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("FA artifact directory must not be a symlink")
            if not current.exists():
                return
            self._require_directory(current, "FA artifact directory")

    @staticmethod
    def _require_directory(path: Path, label: str) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{label} must be a real directory")

    @staticmethod
    def _require_regular_file(path: Path, label: str) -> None:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular file")

    def _reject_existing(self, path: Path) -> None:
        if path.is_symlink():
            raise ValueError("artifact destination must not be a symlink")
        if path.exists():
            raise FileExistsError(path)

    def _remove_regular(self, path: Path) -> None:
        if path.is_symlink():
            raise ValueError("refusing to remove symlinked artifact")
        if path.exists():
            self._require_regular_file(path, "artifact")
            path.unlink()
            _fsync_directory(path.parent)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} must be a safe identifier")
    return value


def _namespace(value: object) -> str:
    if value not in _NAMESPACES:
        raise ValueError("namespace must be a registered FA namespace")
    return str(value)


def _endpoint(value: object) -> str:
    if value not in _ENDPOINTS:
        raise ValueError("endpoint must be a protected FA test endpoint")
    return str(value)


def _sha256_value(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hash")
    return value


def _parents(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"preregistration", "selection_manifest"}:
        raise ValueError("endpoint parents must bind preregistration and selection manifest")
    return {
        "preregistration": _sha256_value(value["preregistration"], "preregistration parent"),
        "selection_manifest": _sha256_value(
            value["selection_manifest"], "selection_manifest parent"
        ),
    }


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("write made no progress")
        remaining = remaining[written:]


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as error:
        if error.errno in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
    finally:
        os.close(descriptor)
