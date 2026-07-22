from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
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
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _before_final_open(path: Path) -> None:
    """Test seam invoked after secure parent traversal and before a leaf open."""

    del path


@dataclass(frozen=True)
class SealedShard:
    namespace: str
    shard_id: str
    data_path: Path
    manifest_path: Path
    sha256: str
    row_count: int
    record_kind: str


@dataclass(frozen=True)
class UnlockReceipt:
    endpoint: str
    lease_id: str
    state: str
    preregistration_hash: str
    selection_manifest_hash: str


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int


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
        *,
        record_kind: str = "generic",
    ) -> SealedShard:
        run = _safe_id(run_id, "run_id")
        namespace = _namespace(namespace)
        shard = _safe_id(shard_id, "shard_id")
        kind = _safe_id(record_kind, "record_kind")
        if not isinstance(lineage, Mapping):
            raise ValueError("lineage must be a mapping")
        payload_rows = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("shard rows must be mappings")
            value = dict(row)
            if kind != "generic" and value.get("kind") != kind:
                raise ValueError("shard row kind does not match record kind")
            payload_rows.append(_canonical_json(value) + b"\n")
        payload = b"".join(payload_rows)
        data_path = self._shard_path(run, namespace, shard)
        manifest_path = data_path.with_name(f"{data_path.name}.manifest.json")
        digest = _sha256_bytes(payload)
        manifest = {
            "schema_version": 1,
            "run_id": run,
            "namespace": namespace,
            "shard_id": shard,
            "data_file": data_path.name,
            "sha256": digest,
            "row_count": len(payload_rows),
            "record_kind": kind,
            "lineage": dict(lineage),
        }
        _canonical_json(manifest)
        data_identity: _FileIdentity | None = None
        try:
            data_identity = self._exclusive_write(data_path, payload)
            self._exclusive_write(manifest_path, _canonical_json(manifest))
        except BaseException:
            if data_identity is not None:
                self._remove_regular(data_path, data_identity)
            raise
        return SealedShard(
            namespace, shard, data_path, manifest_path, digest, len(payload_rows), kind
        )

    def verify_shard(self, manifest_path: str | Path) -> SealedShard:
        manifest_path = Path(manifest_path).absolute()
        self._require_under_root(manifest_path)
        try:
            manifest = json.loads(self._read_regular_bytes(manifest_path, "shard manifest"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
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
        kind = _safe_id(manifest.get("record_kind"), "shard manifest record kind")
        data = self._read_regular_bytes(data_path, "shard data")
        if _sha256_bytes(data) != digest:
            raise ValueError("shard hash mismatch")
        if data.count(b"\n") != row_count:
            raise ValueError("shard row count mismatch")
        if kind == "metrics":
            if row_count < 1:
                raise ValueError("metrics shard must contain at least one row")
            for line in data.splitlines():
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError("metrics shard contains invalid JSON") from error
                if not isinstance(row, dict) or _canonical_json(row) != line:
                    raise ValueError("metrics rows must be canonical JSON objects")
                if row.get("kind") != "metrics":
                    raise ValueError("shard row kind does not match record kind")
        return SealedShard(namespace, shard, data_path, manifest_path, digest, row_count, kind)

    def resume_verified_shards(self, run_id: str, namespace: str) -> tuple[SealedShard, ...]:
        """Return only durably completed shards; corrupt sidecars stop resumption."""
        run = _safe_id(run_id, "run_id")
        namespace = _namespace(namespace)
        directory = self._base() / run / "shards" / namespace
        self._assert_existing_ancestors_are_real(directory)
        try:
            entries = self._list_directory(directory)
        except FileNotFoundError:
            return ()
        manifests = sorted(
            directory / name for name in entries if name.endswith(".jsonl.manifest.json")
        )
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
        sealed, _ = self._read_endpoint_record(sealed_path, endpoint, "sealed")
        self._verify_endpoint_artifacts(sealed)
        unlocked_path = self._endpoint_path(sealed["run_id"], endpoint, "unlocked_once")
        if self._destination_exists(unlocked_path):
            raise ValueError(f"{endpoint} is already unlocked")
        if sealed["parents"]["preregistration"] != preregistration_hash:
            raise ValueError("preregistration hash does not match sealed endpoint")
        if sealed["parents"]["selection_manifest"] != selection_manifest_hash:
            raise ValueError(
                f"{endpoint.removesuffix('_test')} selection manifest hash does not match sealed endpoint"
            )
        return self._write_unlock_receipt(sealed)

    def verify_endpoint_artifact(
        self, endpoint: str, manifest_path: str | Path
    ) -> SealedShard:
        """Verify that an explicit shard is one of the endpoint's sealed capabilities."""
        endpoint = _endpoint(endpoint)
        sealed_path = self._find_endpoint_path(endpoint, "sealed")
        sealed, _ = self._read_endpoint_record(sealed_path, endpoint, "sealed")
        self._verify_endpoint_artifacts(sealed)
        explicit_path = Path(manifest_path).absolute()
        self._require_under_root(explicit_path)
        registered = {
            self._path_from_root_record(item["manifest_path"], "endpoint artifact manifest"): item
            for item in sealed["artifacts"]
        }
        artifact = registered.get(explicit_path)
        if artifact is None:
            raise ValueError("explicit endpoint artifact is not registered in the sealed endpoint")
        shard = self.verify_shard(explicit_path)
        if (
            shard.namespace != endpoint
            or shard.sha256 != artifact["sha256"]
            or self._run_id_for_shard(shard) != sealed["run_id"]
        ):
            raise ValueError("explicit endpoint artifact no longer matches the sealed endpoint")
        return shard

    def unlock_or_resume_endpoint(
        self, endpoint: str, manifest_path: str | Path
    ) -> UnlockReceipt:
        """Unlock a registered artifact once or resume its still-open receipt."""
        endpoint = _endpoint(endpoint)
        self.verify_endpoint_artifact(endpoint, manifest_path)
        sealed_path = self._find_endpoint_path(endpoint, "sealed")
        sealed, _ = self._read_endpoint_record(sealed_path, endpoint, "sealed")
        run_id = sealed["run_id"]
        if self._destination_exists(self._endpoint_path(run_id, endpoint, "closed")):
            raise ValueError(f"{endpoint} is already closed")
        if self._destination_exists(self._endpoint_path(run_id, endpoint, "evaluated")):
            raise ValueError(f"{endpoint} is already evaluated")
        unlocked_path = self._endpoint_path(run_id, endpoint, "unlocked_once")
        if not self._destination_exists(unlocked_path):
            return self._write_unlock_receipt(sealed)
        unlocked, _ = self._read_endpoint_record(unlocked_path, endpoint, "unlocked_once")
        if (
            unlocked["preregistration_hash"] != sealed["parents"]["preregistration"]
            or unlocked["selection_manifest_hash"]
            != sealed["parents"]["selection_manifest"]
        ):
            raise ValueError("active unlock lease no longer matches the sealed endpoint")
        return UnlockReceipt(
            endpoint=endpoint,
            lease_id=unlocked["lease_id"],
            state="unlocked_once",
            preregistration_hash=unlocked["preregistration_hash"],
            selection_manifest_hash=unlocked["selection_manifest_hash"],
        )

    def endpoint_state(self, endpoint: str, manifest_path: str | Path) -> str:
        """Return the verified durable state for a registered endpoint artifact."""
        endpoint = _endpoint(endpoint)
        self.verify_endpoint_artifact(endpoint, manifest_path)
        sealed_path = self._find_endpoint_path(endpoint, "sealed")
        sealed, _ = self._read_endpoint_record(sealed_path, endpoint, "sealed")
        for state in ("closed", "evaluated", "unlocked_once"):
            path = self._endpoint_path(sealed["run_id"], endpoint, state)
            if self._destination_exists(path):
                self._read_endpoint_record(path, endpoint, state)
                return state
        return "sealed"

    def _write_unlock_receipt(self, sealed: Mapping[str, Any]) -> UnlockReceipt:
        endpoint = _endpoint(sealed["endpoint"])
        parents = _parents(sealed["parents"])
        unlocked_path = self._endpoint_path(sealed["run_id"], endpoint, "unlocked_once")
        receipt = UnlockReceipt(
            endpoint=endpoint,
            lease_id=uuid.uuid4().hex,
            state="unlocked_once",
            preregistration_hash=parents["preregistration"],
            selection_manifest_hash=parents["selection_manifest"],
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
                    "preregistration_hash": receipt.preregistration_hash,
                    "selection_manifest_hash": receipt.selection_manifest_hash,
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
        sealed, _ = self._read_endpoint_record(sealed_path, endpoint, "sealed")
        self._verify_endpoint_artifacts(sealed)
        run_id = sealed["run_id"]
        closed_path = self._endpoint_path(run_id, endpoint, "closed")
        if self._destination_exists(closed_path):
            raise ValueError(f"{endpoint} is already closed")
        unlocked_path = self._endpoint_path(run_id, endpoint, "unlocked_once")
        unlocked, _ = self._read_endpoint_record(unlocked_path, endpoint, "unlocked_once")
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
        if metrics.record_kind != "metrics":
            raise ValueError("endpoint evaluation requires a metrics artifact")
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
        sealed, _ = self._read_endpoint_record(sealed_path, endpoint, "sealed")
        self._verify_endpoint_artifacts(sealed)
        run_id = sealed["run_id"]
        closed_path = self._endpoint_path(run_id, endpoint, "closed")
        if self._destination_exists(closed_path):
            raise ValueError(f"{endpoint} is already closed")
        evaluated_path = self._endpoint_path(run_id, endpoint, "evaluated")
        evaluated, evaluated_bytes = self._read_endpoint_record(
            evaluated_path, endpoint, "evaluated"
        )
        if evaluated["run_id"] != run_id:
            raise ValueError("evaluated endpoint state belongs to another run")
        unlocked_path = self._endpoint_path(run_id, endpoint, "unlocked_once")
        unlocked, _ = self._read_endpoint_record(unlocked_path, endpoint, "unlocked_once")
        if (
            unlocked["preregistration_hash"] != sealed["parents"]["preregistration"]
            or unlocked["selection_manifest_hash"] != sealed["parents"]["selection_manifest"]
        ):
            raise ValueError("active unlock lease no longer matches the sealed endpoint")
        if unlocked["lease_id"] != evaluated["lease_id"]:
            raise ValueError("evaluated endpoint lease does not match the active unlock lease")
        metrics_manifest = self._path_from_root_record(
            evaluated.get("metrics_manifest_path"), "metrics manifest"
        )
        metrics = self.verify_shard(metrics_manifest)
        if metrics.record_kind != "metrics":
            raise ValueError("evaluated endpoint requires a metrics artifact")
        if metrics.namespace != endpoint or metrics.sha256 != evaluated.get("metrics_sha256"):
            raise ValueError("evaluated endpoint metrics no longer verify")
        if self._run_id_for_shard(metrics) != run_id:
            raise ValueError("evaluated endpoint metrics must belong to the endpoint run")
        self._exclusive_write(
            closed_path,
            _canonical_json(
                {
                    "schema_version": 1,
                    "state": "closed",
                    "run_id": run_id,
                    "endpoint": endpoint,
                    "evaluated_sha256": _sha256_bytes(evaluated_bytes),
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
        try:
            entries = self._list_directory(base)
        except (FileNotFoundError, ValueError):
            raise ValueError(f"{endpoint} is not sealed")
        matches = []
        for name in entries:
            candidate = base / name
            try:
                descriptor = self._open_directory_descriptor(candidate)
            except ValueError:
                raise ValueError("FA artifact root contains an unsafe run entry")
            else:
                os.close(descriptor)
            try:
                _safe_id(name, "run_id")
            except ValueError:
                raise ValueError("FA artifact root contains an unsafe run entry") from None
            path = self._endpoint_path(name, endpoint, state)
            if self._destination_exists(path):
                matches.append(path)
        if not matches:
            raise ValueError(f"{endpoint} is not sealed")
        if len(matches) != 1:
            raise ValueError(f"{endpoint} is ambiguous across runs")
        return matches[0]

    def _read_endpoint_record(
        self, path: Path, endpoint: str, state: str
    ) -> tuple[dict[str, Any], bytes]:
        try:
            payload = self._read_regular_bytes(path, "endpoint state")
            record = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("endpoint state is unreadable") from error
        required_keys = {
            "sealed": {"schema_version", "state", "run_id", "endpoint", "parents", "artifacts"},
            "unlocked_once": {
                "schema_version",
                "state",
                "run_id",
                "endpoint",
                "lease_id",
                "preregistration_hash",
                "selection_manifest_hash",
            },
            "evaluated": {
                "schema_version",
                "state",
                "run_id",
                "endpoint",
                "lease_id",
                "metrics_manifest_path",
                "metrics_sha256",
            },
            "closed": {"schema_version", "state", "run_id", "endpoint", "evaluated_sha256"},
        }
        if (
            not isinstance(record, dict)
            or set(record) != required_keys.get(state)
            or record.get("schema_version") != 1
            or record.get("state") != state
            or record.get("endpoint") != endpoint
        ):
            raise ValueError("endpoint state has an invalid schema")
        run_id = _safe_id(record.get("run_id"), "endpoint run_id")
        if path != self._endpoint_path(run_id, endpoint, state):
            if state == "evaluated":
                raise ValueError("evaluated endpoint state belongs to another run")
            raise ValueError("endpoint state path does not match its identity")
        if state == "sealed":
            record["parents"] = _parents(record.get("parents"))
        elif state == "unlocked_once":
            _lease_id(record.get("lease_id"), "unlock lease_id")
            _sha256_value(record.get("preregistration_hash"), "unlock preregistration_hash")
            _sha256_value(record.get("selection_manifest_hash"), "unlock selection_manifest_hash")
        elif state == "evaluated":
            _lease_id(record.get("lease_id"), "evaluated lease_id")
            self._path_from_root_record(record.get("metrics_manifest_path"), "metrics manifest")
            _sha256_value(record.get("metrics_sha256"), "evaluated metrics_sha256")
        elif state == "closed":
            _sha256_value(record.get("evaluated_sha256"), "closed evaluated_sha256")
        return record, payload

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

    def _exclusive_write(self, destination: Path, payload: bytes) -> _FileIdentity:
        parent_descriptor, name = self._open_parent_descriptor(destination, create=True)
        temporary_name: str | None = None
        temporary_descriptor: int | None = None
        temporary_identity: _FileIdentity | None = None
        published_identity: _FileIdentity | None = None
        try:
            self._reject_existing_at(parent_descriptor, name)
            temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
            temporary_descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW,
                0o600,
                dir_fd=parent_descriptor,
            )
            temporary_identity = _file_identity(os.fstat(temporary_descriptor))
            _write_all(temporary_descriptor, payload)
            os.fchmod(temporary_descriptor, 0o644)
            os.fsync(temporary_descriptor)
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            self._verify_published_file(parent_descriptor, name, temporary_identity, payload)
            published_identity = temporary_identity
            temporary_cleaned = self._quarantine_regular_at(
                parent_descriptor, temporary_name, temporary_identity
            )
            temporary_name = None
            if not temporary_cleaned:
                raise ValueError("artifact temporary source was replaced during publication")
            _fsync_descriptor(parent_descriptor)
            self._verify_published_file(parent_descriptor, name, published_identity, payload)
            return published_identity
        except BaseException:
            if published_identity is not None:
                self._quarantine_regular_at(parent_descriptor, name, published_identity)
            raise
        finally:
            if temporary_descriptor is not None:
                os.close(temporary_descriptor)
            if temporary_name is not None:
                if temporary_identity is not None:
                    self._quarantine_regular_at(
                        parent_descriptor, temporary_name, temporary_identity
                    )
            os.close(parent_descriptor)

    def _ensure_directory(self, directory: Path) -> None:
        descriptor = self._open_directory_descriptor(directory, create=True)
        os.close(descriptor)

    def _base(self) -> Path:
        return self.root / "runs" / "familiarity_answerability"

    def _require_under_root(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("artifact path escapes the FA artifact root") from error

    def _assert_existing_ancestors_are_real(self, path: Path) -> None:
        try:
            descriptor = self._open_directory_descriptor(path)
        except FileNotFoundError:
            return
        os.close(descriptor)

    def _remove_regular(self, path: Path, identity: _FileIdentity) -> None:
        parent_descriptor, name = self._open_parent_descriptor(path)
        try:
            self._quarantine_regular_at(parent_descriptor, name, identity)
        finally:
            os.close(parent_descriptor)

    def _open_directory_descriptor(self, directory: Path, create: bool = False) -> int:
        if os.name != "posix" or not _O_DIRECTORY or not _O_NOFOLLOW:
            raise ValueError("descriptor-relative artifact access is unavailable")
        absolute = directory.absolute()
        if not absolute.is_absolute():
            raise ValueError("artifact directory must be absolute")
        descriptor = os.open("/", os.O_RDONLY | _O_DIRECTORY)
        try:
            for part in absolute.parts[1:]:
                descriptor = self._open_child_directory(descriptor, part, create)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _open_parent_descriptor(self, path: Path, create: bool = False) -> tuple[int, str]:
        self._require_under_root(path)
        relative = path.relative_to(self.root)
        if not relative.parts:
            raise ValueError("artifact path must name a file")
        descriptor = self._open_directory_descriptor(self.root, create=create)
        try:
            for part in relative.parts[:-1]:
                descriptor = self._open_child_directory(descriptor, part, create)
            return descriptor, relative.parts[-1]
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _open_child_directory(parent_descriptor: int, name: str, create: bool) -> int:
        flags = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW
        try:
            child_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except FileNotFoundError:
            if not create:
                raise
            try:
                os.mkdir(name, 0o755, dir_fd=parent_descriptor)
            except FileExistsError:
                pass
            try:
                child_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
            except OSError as error:
                raise ValueError("FA artifact directory must be a real directory") from error
            _fsync_descriptor(parent_descriptor)
        except OSError as error:
            raise ValueError("FA artifact directory must be a real directory") from error
        os.close(parent_descriptor)
        return child_descriptor

    def _read_regular_bytes(self, path: Path, label: str) -> bytes:
        parent_descriptor, name = self._open_parent_descriptor(path)
        try:
            _before_final_open(path)
            try:
                descriptor = os.open(name, os.O_RDONLY | _O_NOFOLLOW, dir_fd=parent_descriptor)
            except OSError as error:
                raise ValueError(f"{label} must be a regular file") from error
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError(f"{label} must be a regular file")
                chunks = []
                while True:
                    chunk = os.read(descriptor, 1 << 20)
                    if not chunk:
                        return b"".join(chunks)
                    chunks.append(chunk)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_descriptor)

    def _list_directory(self, directory: Path) -> list[str]:
        descriptor = self._open_directory_descriptor(directory)
        try:
            return os.listdir(descriptor)
        finally:
            os.close(descriptor)

    def _destination_exists(self, path: Path) -> bool:
        try:
            parent_descriptor, name = self._open_parent_descriptor(path)
        except FileNotFoundError:
            return False
        try:
            try:
                os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return False
            return True
        finally:
            os.close(parent_descriptor)

    @staticmethod
    def _reject_existing_at(parent_descriptor: int, name: str) -> None:
        try:
            mode = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise ValueError("artifact destination must not be a symlink")
        raise FileExistsError(name)

    def _verify_published_file(
        self,
        parent_descriptor: int,
        name: str,
        expected_identity: _FileIdentity,
        payload: bytes,
    ) -> None:
        try:
            descriptor = os.open(name, os.O_RDONLY | _O_NOFOLLOW, dir_fd=parent_descriptor)
        except OSError as error:
            raise ValueError("artifact publication does not name the written regular file") from error
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or _file_identity(status) != expected_identity:
                raise ValueError("artifact publication does not name the written regular file")
            contents = bytearray()
            while len(contents) < len(payload):
                chunk = os.read(descriptor, len(payload) - len(contents))
                if not chunk:
                    break
                contents.extend(chunk)
            if bytes(contents) != payload or os.read(descriptor, 1):
                raise ValueError("artifact publication content does not match the written payload")
        finally:
            os.close(descriptor)

    @staticmethod
    def _quarantine_regular_at(
        parent_descriptor: int, name: str, expected_identity: _FileIdentity
    ) -> bool:
        quarantine_name = f".{name}.{uuid.uuid4().hex}.quarantine"
        os.mkdir(quarantine_name, 0o700, dir_fd=parent_descriptor)
        quarantine_descriptor = os.open(
            quarantine_name,
            os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        removed = False
        try:
            try:
                os.rename(
                    name,
                    "entry",
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=quarantine_descriptor,
                )
            except FileNotFoundError:
                return False
            try:
                status = os.stat("entry", dir_fd=quarantine_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return False
            if not stat.S_ISREG(status.st_mode) or _file_identity(status) != expected_identity:
                return False
            os.unlink("entry", dir_fd=quarantine_descriptor)
            _fsync_descriptor(quarantine_descriptor)
            removed = True
            return True
        finally:
            os.close(quarantine_descriptor)
            if removed:
                os.rmdir(quarantine_name, dir_fd=parent_descriptor)
                _fsync_descriptor(parent_descriptor)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_identity(status: os.stat_result) -> _FileIdentity:
    return _FileIdentity(status.st_dev, status.st_ino)


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


def _lease_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError(f"{field} is invalid")
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


def _fsync_descriptor(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
