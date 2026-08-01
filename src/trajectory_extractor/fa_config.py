from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from types import MappingProxyType


CONFIRMATORY_SPLIT_COUNTS = {
    "mechanism_train": 64,
    "locked_validation": 32,
    "behavior_test": 48,
    "probe_test": 24,
    "intervention_test": 24,
}
FEASIBILITY_SAME_STRING_STUDY_ID = (
    "familiarity-answerability-same-string-feasibility-v2"
)
FEASIBILITY_SAME_STRING_RUN_ID = "same-string-feasibility-v2"
SAME_STRING_V1_STUDY_ID = "familiarity-answerability-same-string-gemma2-2b-v1"
SAME_STRING_V1_RUN_ID = "same-string-primary-v1"
FEASIBILITY_SAME_STRING_SPLIT_COUNTS = {
    "behavior_test": 32,
    "mechanism_train": 12,
    "locked_validation": 4,
    "probe_test": 4,
}
NON_CONFIRMATORY_NAMESPACES = frozenset({"pilot", "circuit_dev"})
REGISTERED_ANCHORS = (
    "target_intro_end",
    "user_prompt_end",
    "assistant_prefix_end",
)
CONFIRMATORY_MODEL_ID = "google/gemma-2-2b-it"
CONFIRMATORY_MODEL_REVISION = "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
CONFIRMATORY_CHAT_TEMPLATE_SHA256 = (
    "ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6"
)
CONFIRMATORY_SPLIT_SEED = 20260722
CONFIRMATORY_BOOTSTRAP_REPLICATES = 10000
CONFIRMATORY_BOOTSTRAP_SEED = 20260722
CONFIRMATORY_GENERATION = {
    "do_sample": False,
    "max_new_tokens": 16,
    "temperature": 0.0,
}
LEGACY_SMOKE_MODEL_ID = "Qwen/Qwen3-0.6B"
LEGACY_SMOKE_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SMOKE_MODEL_ID = "Qwen/Qwen3-1.7B"
SMOKE_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SMOKE_CHAT_TEMPLATE_SHA256 = (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
)
SMOKE_MODEL_PINS = MappingProxyType(
    {
        LEGACY_SMOKE_MODEL_ID: MappingProxyType(
            {
                "revision": LEGACY_SMOKE_MODEL_REVISION,
                "chat_template_sha256": SMOKE_CHAT_TEMPLATE_SHA256,
            }
        ),
        SMOKE_MODEL_ID: MappingProxyType(
            {
                "revision": SMOKE_MODEL_REVISION,
                "chat_template_sha256": SMOKE_CHAT_TEMPLATE_SHA256,
            }
        ),
    }
)
CONFIRMATORY_THRESHOLDS = {
    "format_validity_min": 0.95,
    "h1_min_interaction": 0.05,
    "h2_noninferiority_margin": 0.05,
    "h5_relative_log_loss_min": 0.02,
    "h6_relative_log_loss_min": 0.01,
    "h7_average_effect_min": 0.05,
    "h7_control_margin_min": 0.02,
    "intervention_accuracy_drop_max": 0.05,
    "intervention_control_rate_change_max": 0.03,
    "probe_auroc_min": 0.65,
    "probe_balanced_accuracy_min": 0.55,
    "sae_loss_recovery_min": 0.70,
    "sae_finite_fraction_min": 0.95,
    "circuit_proxy_spearman_min": 0.80,
    "circuit_distribution_spearman_min": 0.80,
    "circuit_perturbation_spearman_min": 0.60,
    "circuit_sign_concordance_min": 0.75,
}

_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_counts", _freeze_json_value(self.split_counts))
        object.__setattr__(self, "generation", _freeze_json_value(self.generation))
        object.__setattr__(self, "thresholds", _freeze_json_value(self.thresholds))
        object.__setattr__(self, "anchors", tuple(self.anchors))
        self.validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "FAConfig":
        value = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates
        )
        if not isinstance(value, dict):
            raise ValueError("FA config must be a JSON object")
        try:
            return cls(
                **{
                    **value,
                    "split_counts": value["split_counts"],
                    "generation": value["generation"],
                    "thresholds": value["thresholds"],
                    "anchors": value["anchors"],
                }
            )
        except (KeyError, TypeError) as error:
            raise ValueError("FA config has invalid fields") from error

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "profile": self.profile,
                "study_id": self.study_id,
                "run_id": self.run_id,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "tokenizer_revision": self.tokenizer_revision,
                "chat_template_sha256": self.chat_template_sha256,
                "split_seed": self.split_seed,
                "split_counts": _thaw_json_value(self.split_counts),
                "generation": _thaw_json_value(self.generation),
                "bootstrap_replicates": self.bootstrap_replicates,
                "bootstrap_seed": self.bootstrap_seed,
                "thresholds": _thaw_json_value(self.thresholds),
                "anchors": _thaw_json_value(self.anchors),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
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
            if self.model_id != CONFIRMATORY_MODEL_ID:
                raise ValueError("Qwen is smoke-only; confirmatory profile requires Gemma")
            if self.model_revision != CONFIRMATORY_MODEL_REVISION:
                raise ValueError("confirmatory model_revision must match the official pin")
            if self.tokenizer_revision != CONFIRMATORY_MODEL_REVISION:
                raise ValueError("confirmatory tokenizer_revision must match the official pin")
            if self.chat_template_sha256 != CONFIRMATORY_CHAT_TEMPLATE_SHA256:
                raise ValueError("confirmatory chat_template_sha256 must match the official pin")
            registered_split_counts = (
                FEASIBILITY_SAME_STRING_SPLIT_COUNTS
                if (
                    self.study_id == FEASIBILITY_SAME_STRING_STUDY_ID
                    and self.run_id == FEASIBILITY_SAME_STRING_RUN_ID
                )
                else CONFIRMATORY_SPLIT_COUNTS
            )
            if dict(self.split_counts) != registered_split_counts:
                unknown = set(self.split_counts) - set(CONFIRMATORY_SPLIT_COUNTS)
                if unknown:
                    raise ValueError("split_counts contains an unregistered split")
                raise ValueError("confirmatory split_counts must match the preregistration")
        else:
            smoke_pin = SMOKE_MODEL_PINS.get(self.model_id)
            if smoke_pin is None:
                raise ValueError("smoke profile must use the registered Qwen model")
            if self.model_revision != smoke_pin["revision"]:
                raise ValueError("smoke model_revision must match the registered Qwen pin")
            if self.tokenizer_revision != smoke_pin["revision"]:
                raise ValueError("smoke tokenizer_revision must match the registered Qwen pin")
            expected_template = smoke_pin["chat_template_sha256"]
            if self.model_id == SMOKE_MODEL_ID:
                if self.chat_template_sha256 != expected_template:
                    raise ValueError(
                        "active smoke chat_template_sha256 must match the registered Qwen pin"
                    )
            if set(self.split_counts) - NON_CONFIRMATORY_NAMESPACES:
                raise ValueError("smoke split_counts contains an unregistered split")
            if set(self.split_counts) != NON_CONFIRMATORY_NAMESPACES:
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
            if self.split_seed != CONFIRMATORY_SPLIT_SEED:
                raise ValueError("confirmatory split_seed must match the preregistration")
            if self.bootstrap_replicates != CONFIRMATORY_BOOTSTRAP_REPLICATES:
                raise ValueError("confirmatory bootstrap_replicates must match the preregistration")
            if self.bootstrap_seed != CONFIRMATORY_BOOTSTRAP_SEED:
                raise ValueError("confirmatory bootstrap_seed must match the preregistration")
            if self.anchors != REGISTERED_ANCHORS:
                raise ValueError("confirmatory anchors must match the preregistration order")
            if (
                type(self.generation.get("do_sample")) is not bool
                or type(self.generation.get("max_new_tokens")) is not int
                or type(self.generation.get("temperature")) is not float
            ):
                raise ValueError("confirmatory generation must use exact registered field types")
            if dict(self.generation) != CONFIRMATORY_GENERATION:
                raise ValueError("confirmatory generation must match the registered greedy object")
            if dict(self.thresholds) != CONFIRMATORY_THRESHOLDS:
                raise ValueError("confirmatory thresholds must match the preregistration")
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


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value
