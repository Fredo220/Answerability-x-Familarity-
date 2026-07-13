from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


_SAFE_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONFIDENCE_VALUES = tuple(index / 10 for index in range(11))
_ARMS = ("standard_grpo", "rlmf")
_SPLIT_NAMES = {"pre_sft", "rl_train", "validation", "test"}
_BEHAVIORAL_PROVENANCE_KEYS = frozenset(
    {
        "designated_bundle_hash",
        "auxiliary_bundle_hash",
        "alias_evidence_hash",
        "judge_evidence_hash",
        "config_hash",
    }
)

_COMMON_CONFIG = {
    "schema_version": 1,
    "model_id": "Qwen/Qwen3-0.6B",
    "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "dataset_id": "akariasai/PopQA",
    "dataset_revision": "5cf59972d88d4aaaa7781ac91b83d053563d8268",
    "split_seed": 20260713,
    "arms": ["standard_grpo", "rlmf"],
    "max_prompt_tokens": 192,
    "max_completion_tokens": 96,
    "training_consistency_mode": "leave_one_out_group",
    "metacognition_queries_per_completion": 1,
    "faithfulness_tau": 0.1,
    "sft_learning_rate": 0.00003,
    "sft_weight_decay": 0.01,
    "learning_rate": 0.000005,
    "per_device_train_batch_size": 1,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "lora_targets": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "quantization": "nf4",
    "compute_dtype": "float16",
    "generation": {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.05,
        "enable_thinking": False,
    },
    "reward_weights": {
        "soft_format": 3.0,
        "strict_format": 3.0,
        "factual_calibration": 1.0,
        "correctness": 1.0,
        "faithful_calibration": 12.0,
    },
    "confidence_values": list(_CONFIDENCE_VALUES),
    "bootstrap_seed_mode": "fixed_registered_seeds_prompt_cluster",
    "judge_differential_bias_upper_limit": 0.015,
}

_PROFILE_CONFIG = {
    "smoke": {
        "profile": "smoke",
        "study_id": "rlmf-qwen06b-smoke-v1",
        "split_counts": {"pre_sft": 8, "rl_train": 8, "validation": 4, "test": 4},
        "seeds": [11],
        "rollout_group_size": 2,
        "evaluation_auxiliary_samples": 2,
        "sft_auxiliary_samples": 1,
        "sft_epochs": 1,
        "sft_global_batch_size": 2,
        "rl_steps": 2,
        "save_steps": 1,
        "gradient_accumulation_steps": 2,
        "generation_batch_size": 2,
        "num_generations": 2,
        "behavior_bootstrap_replicates": 10,
        "mechanism_bootstrap_replicates": 10,
    },
    "confirmatory": {
        "profile": "confirmatory",
        "study_id": "rlmf-qwen06b-v1",
        "split_counts": {"pre_sft": 256, "rl_train": 256, "validation": 128, "test": 256},
        "seeds": [11, 22, 33],
        "rollout_group_size": 4,
        "evaluation_auxiliary_samples": 20,
        "sft_auxiliary_samples": 4,
        "sft_epochs": 5,
        "sft_global_batch_size": 8,
        "rl_steps": 200,
        "save_steps": 25,
        "gradient_accumulation_steps": 4,
        "generation_batch_size": 4,
        "num_generations": 4,
        "behavior_bootstrap_replicates": 5000,
        "mechanism_bootstrap_replicates": 5000,
    },
}


def _immutable_mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_sequence(value: Sequence[Any], field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return tuple(_freeze_value(item) for item in value)


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_real(value: Any) -> bool:
    return type(value) in {int, float}


def _validate_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hash")


def _validate_id(value: str, field: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} must be a safe identifier")


def _validate_parent_hashes(value: Mapping[str, str]) -> Mapping[str, str]:
    frozen = _immutable_mapping(value, "parent_hashes")
    for name, digest in frozen.items():
        _validate_id(name, "parent_hashes key")
        _validate_sha256(digest, "parent_hashes")
    return frozen


def _validate_relative_checkpoint_path(value: str) -> None:
    if not isinstance(value, str):
        raise ValueError("checkpoint file paths must be strings")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("checkpoint file paths must be safe relative paths")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class RLMFConfig:
    schema_version: int
    profile: str
    study_id: str
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    split_seed: int
    split_counts: Mapping[str, int]
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    max_prompt_tokens: int
    max_completion_tokens: int
    rollout_group_size: int
    training_consistency_mode: str
    evaluation_auxiliary_samples: int
    metacognition_queries_per_completion: int
    faithfulness_tau: float
    sft_auxiliary_samples: int
    sft_epochs: int
    sft_learning_rate: float
    sft_weight_decay: float
    sft_global_batch_size: int
    rl_steps: int
    save_steps: int
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    generation_batch_size: int
    num_generations: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    quantization: str
    compute_dtype: str
    generation: Mapping[str, Any]
    reward_weights: Mapping[str, float]
    confidence_values: tuple[float, ...]
    behavior_bootstrap_replicates: int
    mechanism_bootstrap_replicates: int
    bootstrap_seed_mode: str
    judge_differential_bias_upper_limit: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_counts", _immutable_mapping(self.split_counts, "split_counts"))
        for field in ("arms", "seeds", "lora_targets", "confidence_values"):
            object.__setattr__(self, field, _freeze_sequence(getattr(self, field), field))
        object.__setattr__(self, "generation", _immutable_mapping(self.generation, "generation"))
        object.__setattr__(self, "reward_weights", _immutable_mapping(self.reward_weights, "reward_weights"))
        self._validate_numeric_fields()
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.profile not in {"smoke", "confirmatory"}:
            raise ValueError("profile must be smoke or confirmatory")
        _validate_id(self.study_id, "study_id")
        if self.model_id != "Qwen/Qwen3-0.6B":
            raise ValueError("model_id must use the registered Qwen model")
        if self.model_revision != "c1899de289a04d12100db370d81485cdf75e47ca":
            raise ValueError("model_revision must use the registered immutable revision")
        if self.dataset_id != "akariasai/PopQA":
            raise ValueError("dataset_id must use the registered PopQA dataset")
        if self.dataset_revision != "5cf59972d88d4aaaa7781ac91b83d053563d8268":
            raise ValueError("dataset_revision must use the registered immutable revision")
        if not _is_int(self.split_seed) or self.split_seed < 1:
            raise ValueError("split_seed must be positive")
        split_counts = self.split_counts
        if set(split_counts) != _SPLIT_NAMES or any(
            not _is_int(count) or count < 1 for count in split_counts.values()
        ):
            raise ValueError("split_counts must contain positive registered split counts")
        if self.arms != _ARMS:
            raise ValueError("arms must be standard_grpo and rlmf in registered order")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(
            not _is_int(seed) or seed < 1 for seed in self.seeds
        ):
            raise ValueError("seeds must be unique positive integers")
        if self.profile == "confirmatory" and self.seeds != (11, 22, 33):
            raise ValueError("confirmatory seeds must be the registered three seeds")
        if self.profile == "smoke" and len(self.seeds) != 1:
            raise ValueError("smoke profile requires exactly one infrastructure seed")
        if self.rollout_group_size < 2:
            raise ValueError("rollout_group_size must support leave-one-out groups")
        if self.training_consistency_mode != "leave_one_out_group":
            raise ValueError("training_consistency_mode must be leave_one_out_group")
        if self.num_generations != self.rollout_group_size:
            raise ValueError("num_generations must equal rollout_group_size")
        if self.generation_batch_size < 1 or self.generation_batch_size % self.num_generations:
            raise ValueError("generation_batch_size must divide exactly by num_generations")
        if self.metacognition_queries_per_completion != 1:
            raise ValueError("metacognition_queries_per_completion must be exactly one")
        if self.profile == "confirmatory":
            if self.evaluation_auxiliary_samples != 20:
                raise ValueError("evaluation_auxiliary_samples must be exactly 20")
            if (self.gradient_accumulation_steps, self.generation_batch_size) != (4, 4):
                raise ValueError("confirmatory batch settings must both be four")
            if dict(split_counts) != {
                "pre_sft": 256,
                "rl_train": 256,
                "validation": 128,
                "test": 256,
            }:
                raise ValueError("confirmatory split_counts must match the preregistration")
        elif (self.gradient_accumulation_steps, self.generation_batch_size) != (2, 2):
            raise ValueError("smoke batch settings must both be two")
        positive = (
            self.max_prompt_tokens,
            self.max_completion_tokens,
            self.sft_auxiliary_samples,
            self.sft_epochs,
            self.sft_global_batch_size,
            self.rl_steps,
            self.save_steps,
            self.per_device_train_batch_size,
            self.gradient_accumulation_steps,
            self.lora_rank,
            self.lora_alpha,
            self.behavior_bootstrap_replicates,
            self.mechanism_bootstrap_replicates,
        )
        if any(not _is_int(value) or value < 1 for value in positive):
            raise ValueError("registered numeric counts must be positive")
        if not 0 <= self.faithfulness_tau <= 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("faithfulness_tau and lora_dropout must be in range")
        if min(self.sft_learning_rate, self.learning_rate) <= 0 or self.sft_weight_decay < 0:
            raise ValueError("learning rates must be positive and weight decay non-negative")
        if tuple(self.lora_targets) != (
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ):
            raise ValueError("lora_targets must match the preregistered modules")
        if self.quantization != "nf4" or self.compute_dtype != "float16":
            raise ValueError("quantization and compute_dtype must match the registered stack")
        generation = self.generation
        if generation != {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.05,
            "enable_thinking": False,
        }:
            raise ValueError("generation must match the frozen non-thinking settings")
        rewards = self.reward_weights
        if rewards != {
            "soft_format": 3.0,
            "strict_format": 3.0,
            "factual_calibration": 1.0,
            "correctness": 1.0,
            "faithful_calibration": 12.0,
        }:
            raise ValueError("reward_weights must match the preregistration")
        if self.confidence_values != _CONFIDENCE_VALUES:
            raise ValueError("confidence_values must contain the eleven registered values")
        if self.bootstrap_seed_mode != "fixed_registered_seeds_prompt_cluster":
            raise ValueError("bootstrap_seed_mode must be registered")
        if not 0 < self.judge_differential_bias_upper_limit < 1:
            raise ValueError("judge_differential_bias_upper_limit must be in range")
        expected = {**_COMMON_CONFIG, **_PROFILE_CONFIG[self.profile]}
        if self.to_dict() != expected:
            raise ValueError(f"frozen {self.profile} config does not match the preregistration")

    def _validate_numeric_fields(self) -> None:
        integers = (
            self.schema_version,
            self.split_seed,
            *self.split_counts.values(),
            self.max_prompt_tokens,
            self.max_completion_tokens,
            self.rollout_group_size,
            self.evaluation_auxiliary_samples,
            self.metacognition_queries_per_completion,
            self.sft_auxiliary_samples,
            self.sft_epochs,
            self.sft_global_batch_size,
            self.rl_steps,
            self.save_steps,
            self.per_device_train_batch_size,
            self.gradient_accumulation_steps,
            self.generation_batch_size,
            self.num_generations,
            self.lora_rank,
            self.lora_alpha,
            self.behavior_bootstrap_replicates,
            self.mechanism_bootstrap_replicates,
            self.generation.get("top_k"),
        )
        reals = (
            self.faithfulness_tau,
            self.sft_learning_rate,
            self.sft_weight_decay,
            self.learning_rate,
            self.lora_dropout,
            self.judge_differential_bias_upper_limit,
            self.generation.get("temperature"),
            self.generation.get("top_p"),
            self.generation.get("min_p"),
            self.generation.get("repetition_penalty"),
            *self.reward_weights.values(),
            *self.confidence_values,
        )
        if not all(_is_int(value) for value in integers) or not all(
            _is_real(value) for value in reals
        ):
            raise ValueError("all scientific numeric fields must be numeric, not booleans")

    @classmethod
    def from_json(cls, path: str | Path) -> "RLMFConfig":
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("RLMF config must be a JSON object")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: _thaw_value(value) for name, value in self.__dict__.items()
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True)
class PopQAExample:
    example_id: str
    subject: str
    question: str
    answers: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "answers", tuple(self.answers))
        _validate_id(self.example_id, "example_id")
        if not isinstance(self.subject, str) or not self.subject:
            raise ValueError("subject must be non-empty")
        if not isinstance(self.question, str) or not self.question:
            raise ValueError("question must be non-empty")
        if not self.answers or any(not isinstance(answer, str) or not answer.strip() for answer in self.answers):
            raise ValueError("answers must be non-empty strings")
        if len(set(self.answers)) != len(self.answers):
            raise ValueError("answers must be unique")


@dataclass(frozen=True)
class ParsedRLMFOutput:
    answer: str
    confidence: float | None = None
    metascore: float | None = None
    valid_format: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str):
            raise ValueError("answer must be a string")
        for name, score in (("confidence", self.confidence), ("metascore", self.metascore)):
            if score is not None and (not _is_real(score) or not 0 <= score <= 1):
                raise ValueError(f"{name} must be in range")
            if score is not None and float(score) not in _CONFIDENCE_VALUES:
                raise ValueError(f"{name} must use a registered confidence value")
        if self.valid_format and (not self.answer.strip() or self.confidence is None):
            raise ValueError("valid_format requires a non-empty answer and confidence")


@dataclass(frozen=True)
class RLMFCompletion:
    study_id: str
    arm: str
    seed: int
    split: str
    example_id: str
    candidate_id: str
    raw_output: str
    parsed: ParsedRLMFOutput
    checkpoint_hash: str
    config_hash: str
    parent_hashes: Mapping[str, str]
    source_question: str = ""

    def __post_init__(self) -> None:
        _validate_id(self.study_id, "study_id")
        if self.arm not in _ARMS:
            raise ValueError("arm must be registered")
        if not _is_int(self.seed) or self.seed < 1:
            raise ValueError("seed must be positive")
        if self.split not in _SPLIT_NAMES:
            raise ValueError("split must be registered")
        _validate_id(self.example_id, "example_id")
        _validate_id(self.candidate_id, "candidate_id")
        if not isinstance(self.raw_output, str):
            raise ValueError("raw_output must be a string")
        if not isinstance(self.source_question, str):
            raise ValueError("source_question must be a string")
        if not isinstance(self.parsed, ParsedRLMFOutput):
            raise ValueError("parsed must be a ParsedRLMFOutput")
        _validate_sha256(self.checkpoint_hash, "checkpoint_hash")
        _validate_sha256(self.config_hash, "config_hash")
        object.__setattr__(self, "parent_hashes", _validate_parent_hashes(self.parent_hashes))

    @classmethod
    def parse_record(cls, value: Mapping[str, Any]) -> "RLMFCompletion":
        if not isinstance(value, Mapping):
            raise ValueError("completion record must be a mapping")
        parsed = value.get("parsed")
        if isinstance(parsed, Mapping):
            parsed = ParsedRLMFOutput(**parsed)
        return cls(
            study_id=value.get("study_id"),
            arm=value.get("arm"),
            seed=value.get("seed"),
            split=value.get("split"),
            example_id=value.get("example_id"),
            candidate_id=value.get("candidate_id", value.get("member_id")),
            raw_output=value.get("raw_output", value.get("response", "")),
            parsed=parsed,
            checkpoint_hash=value.get("checkpoint_hash"),
            config_hash=value.get("config_hash"),
            parent_hashes=value.get("parent_hashes", {}),
            source_question=value.get("source_question", ""),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "arm": self.arm,
            "seed": self.seed,
            "split": self.split,
            "example_id": self.example_id,
            "candidate_id": self.candidate_id,
            "raw_output": self.raw_output,
            "parsed": {
                "answer": self.parsed.answer,
                "confidence": self.parsed.confidence,
                "metascore": self.parsed.metascore,
                "valid_format": self.parsed.valid_format,
            },
            "checkpoint_hash": self.checkpoint_hash,
            "config_hash": self.config_hash,
            "parent_hashes": dict(self.parent_hashes),
            "source_question": self.source_question,
        }


@dataclass(frozen=True)
class CheckpointRecord:
    """Immutable, content-addressed description of one restartable checkpoint."""

    schema_version: int
    study_id: str
    stage: str
    arm: str | None
    seed: int | None
    global_step: int
    micro_step: int
    sampler_cursor: int
    files: Mapping[str, str]
    parent_hashes: Mapping[str, str]
    checkpoint_hash: str
    path: str
    completed: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("checkpoint schema_version must be 1")
        _validate_id(self.study_id, "study_id")
        if self.stage not in {"pre_sft", "rl"}:
            raise ValueError("checkpoint stage must be pre_sft or rl")
        if self.stage == "pre_sft":
            if self.arm is not None or self.seed is not None:
                raise ValueError("pre_sft checkpoint must not bind an arm or seed")
        elif self.arm not in _ARMS or not _is_int(self.seed) or self.seed < 1:
            raise ValueError("rl checkpoint requires a registered arm and positive seed")
        for name in ("global_step", "micro_step", "sampler_cursor"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.completed) is not bool:
            raise ValueError("completed must be boolean")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("checkpoint path must be a non-empty string")
        files = _immutable_mapping(self.files, "files")
        if not files:
            raise ValueError("checkpoint files must not be empty")
        for relative, digest in files.items():
            _validate_relative_checkpoint_path(relative)
            _validate_sha256(digest, "files")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "parent_hashes", _validate_parent_hashes(self.parent_hashes))
        _validate_sha256(self.checkpoint_hash, "checkpoint_hash")
        if self.checkpoint_hash != self.computed_hash:
            raise ValueError("checkpoint_hash does not match checkpoint contents")

    @classmethod
    def create(
        cls,
        *,
        study_id: str,
        stage: str,
        arm: str | None,
        seed: int | None,
        global_step: int,
        micro_step: int,
        sampler_cursor: int,
        files: Mapping[str, str],
        parent_hashes: Mapping[str, str],
        path: str,
        completed: bool,
    ) -> "CheckpointRecord":
        payload = {
            "schema_version": 1,
            "study_id": study_id,
            "stage": stage,
            "arm": arm,
            "seed": seed,
            "global_step": global_step,
            "micro_step": micro_step,
            "sampler_cursor": sampler_cursor,
            "files": dict(files),
            "parent_hashes": dict(parent_hashes),
            "completed": completed,
        }
        checkpoint_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        return cls(checkpoint_hash=checkpoint_hash, path=path, **payload)

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "CheckpointRecord":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("checkpoint record has an invalid schema")
        return cls(**dict(value))

    @property
    def computed_hash(self) -> str:
        payload = self.to_record()
        payload.pop("checkpoint_hash")
        payload.pop("path")
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "stage": self.stage,
            "arm": self.arm,
            "seed": self.seed,
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "sampler_cursor": self.sampler_cursor,
            "files": dict(self.files),
            "parent_hashes": dict(self.parent_hashes),
            "checkpoint_hash": self.checkpoint_hash,
            "path": self.path,
            "completed": self.completed,
        }


@dataclass(frozen=True)
class BehavioralEvaluationRecord:
    """One retained designated response and its 20 auxiliary judgments."""

    arm: str
    seed: int
    example_id: str
    designated_member_id: str
    designated_raw_output: str
    designated: ParsedRLMFOutput
    auxiliary_member_ids: tuple[str, ...]
    auxiliary_proxy_labels: tuple[bool, ...]
    correctness: bool | None
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError("arm must be registered")
        if not _is_int(self.seed) or self.seed < 1:
            raise ValueError("seed must be positive")
        _validate_id(self.example_id, "example_id")
        _validate_id(self.designated_member_id, "designated_member_id")
        if not isinstance(self.designated_raw_output, str):
            raise ValueError("designated_raw_output must be a string")
        if not isinstance(self.designated, ParsedRLMFOutput):
            raise ValueError("designated must be a ParsedRLMFOutput")
        member_ids = tuple(self.auxiliary_member_ids)
        labels = tuple(self.auxiliary_proxy_labels)
        if len(member_ids) != 20 or len(labels) != 20:
            raise ValueError("behavioral evaluation requires exactly 20 auxiliaries")
        for member_id in member_ids:
            _validate_id(member_id, "auxiliary_member_id")
        if len(set(member_ids)) != 20 or self.designated_member_id in member_ids:
            raise ValueError("designated and auxiliary member IDs must be distinct")
        if any(type(label) is not bool for label in labels):
            raise ValueError("auxiliary_proxy_labels must be boolean")
        if self.correctness is not None and type(self.correctness) is not bool:
            raise ValueError("correctness must be boolean or None")
        from trajectory_extractor.rlmf_format import parse_rlmf_output

        if parse_rlmf_output(self.designated_raw_output) != self.designated:
            raise ValueError("designated_raw_output must reparse exactly to designated")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        provenance = dict(self.provenance)
        if set(provenance) != _BEHAVIORAL_PROVENANCE_KEYS:
            raise ValueError("provenance schema is invalid")
        for key, value in provenance.items():
            _validate_sha256(value, f"provenance {key}")
        object.__setattr__(self, "auxiliary_member_ids", member_ids)
        object.__setattr__(self, "auxiliary_proxy_labels", labels)
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    @property
    def confidence(self) -> float | None:
        return self.designated.confidence

    @property
    def intrinsic(self) -> float:
        return sum(self.auxiliary_proxy_labels) / 20.0

    @property
    def valid_format(self) -> bool:
        return self.designated.valid_format

    @property
    def valid_complete_case(self) -> bool:
        return self.designated.valid_format and type(self.correctness) is bool

    @property
    def record_id(self) -> tuple[str, int, str]:
        return (self.arm, self.seed, self.example_id)

    def to_record(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "example_id": self.example_id,
            "designated_member_id": self.designated_member_id,
            "designated_raw_output": self.designated_raw_output,
            "designated": {
                "answer": self.designated.answer,
                "confidence": self.designated.confidence,
                "metascore": self.designated.metascore,
                "valid_format": self.designated.valid_format,
            },
            "auxiliary_member_ids": list(self.auxiliary_member_ids),
            "auxiliary_proxy_labels": list(self.auxiliary_proxy_labels),
            "correctness": self.correctness,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ClaimDecision:
    study_id: str
    endpoint: str
    status: str
    config_hash: str
    parent_hashes: Mapping[str, str]
    reason: str

    def __post_init__(self) -> None:
        _validate_id(self.study_id, "study_id")
        _validate_id(self.endpoint, "endpoint")
        if self.status not in {"supported", "not_supported", "not_evaluable"}:
            raise ValueError("status must be supported, not_supported, or not_evaluable")
        _validate_sha256(self.config_hash, "config_hash")
        object.__setattr__(self, "parent_hashes", _validate_parent_hashes(self.parent_hashes))
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty")
