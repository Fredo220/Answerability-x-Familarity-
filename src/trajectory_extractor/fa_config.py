from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


CONFIRMATORY_SPLIT_COUNTS = {
    "mechanism_train": 64,
    "locked_validation": 32,
    "behavior_test": 48,
    "probe_test": 24,
    "intervention_test": 24,
}
NON_CONFIRMATORY_NAMESPACES = frozenset({"pilot", "circuit_dev"})
REGISTERED_ANCHORS = (
    "target_intro_end",
    "user_prompt_end",
    "assistant_prefix_end",
)

_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class FAConfig:
    schema_version: int
    profile: str
    study_id: str
    run_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    chat_template_sha256: str
    split_seed: int
    split_counts: Mapping[str, int]
    generation: Mapping[str, Any]
    bootstrap_replicates: int
    bootstrap_seed: int
    thresholds: Mapping[str, float]
    anchors: tuple[str, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "FAConfig":
        value = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates
        )
        if not isinstance(value, dict):
            raise ValueError("FA config must be a JSON object")
        try:
            config = cls(
                **{
                    **value,
                    "split_counts": dict(value["split_counts"]),
                    "generation": dict(value["generation"]),
                    "thresholds": dict(value["thresholds"]),
                    "anchors": tuple(value["anchors"]),
                }
            )
        except (KeyError, TypeError) as error:
            raise ValueError("FA config has invalid fields") from error
        config.validate()
        return config

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.profile not in {"smoke", "confirmatory"}:
            raise ValueError("profile must be smoke or confirmatory")
        if not isinstance(self.study_id, str) or not _SAFE_ID.fullmatch(self.study_id):
            raise ValueError("study_id must be a safe identifier")
        if not isinstance(self.run_id, str) or not _SAFE_ID.fullmatch(self.run_id):
            raise ValueError("run_id must be a safe identifier")
        _validate_revision(self.model_revision, "model_revision")
        _validate_revision(self.tokenizer_revision, "tokenizer_revision")
        if self.profile == "confirmatory":
            if self.model_id != "google/gemma-2-2b-it":
                raise ValueError("Qwen is smoke-only; confirmatory profile requires Gemma")
            if not _SHA256.fullmatch(self.chat_template_sha256):
                raise ValueError("chat_template_sha256 must be a SHA-256 hash")
            if dict(self.split_counts) != CONFIRMATORY_SPLIT_COUNTS:
                unknown = set(self.split_counts) - set(CONFIRMATORY_SPLIT_COUNTS)
                if unknown:
                    raise ValueError("split_counts contains an unregistered split")
                raise ValueError("confirmatory split_counts must match the preregistration")
        elif self.model_id != "Qwen/Qwen3-0.6B":
            raise ValueError("smoke profile must use the registered Qwen model")
        elif set(self.split_counts) - NON_CONFIRMATORY_NAMESPACES:
            raise ValueError("smoke split_counts contains an unregistered split")
        elif set(self.split_counts) != NON_CONFIRMATORY_NAMESPACES:
            raise ValueError("smoke split_counts must use pilot and circuit_dev only")

        if any(type(count) is not int or count <= 0 for count in self.split_counts.values()):
            raise ValueError("split_counts must contain positive registered split counts")
        if type(self.split_seed) is not int or self.split_seed <= 0:
            raise ValueError("split_seed must be positive")
        if type(self.bootstrap_replicates) is not int or self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed <= 0:
            raise ValueError("bootstrap_seed must be positive")
        if not isinstance(self.generation, Mapping):
            raise ValueError("generation must be a mapping")
        if not isinstance(self.thresholds, Mapping):
            raise ValueError("thresholds must be a mapping")
        for name, value in self.thresholds.items():
            if not isinstance(name, str) or type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError("thresholds must contain finite numeric values")
        if self.profile == "confirmatory":
            if not self.anchors or len(set(self.anchors)) != len(self.anchors):
                raise ValueError("anchors must be a non-duplicate registered sequence")
            if any(anchor not in REGISTERED_ANCHORS for anchor in self.anchors):
                raise ValueError("anchors contains a nonregistered anchor")
        elif self.thresholds or self.anchors:
            raise ValueError("smoke configurations cannot select confirmatory thresholds or anchors")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate config key: {key}")
        result[key] = value
    return result


def _validate_revision(value: object, field: str) -> None:
    if not isinstance(value, str) or not _IMMUTABLE_REVISION.fullmatch(value):
        raise ValueError(f"{field} must be an immutable revision")
