from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


_SAFE_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONFIDENCE_VALUES = tuple(index / 10 for index in range(11))
_ARMS = ("standard_grpo", "rlmf")
_SPLIT_NAMES = {"pre_sft", "rl_train", "validation", "test"}


def _immutable_mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return MappingProxyType(dict(value))


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
        if not isinstance(self.split_seed, int) or self.split_seed < 1:
            raise ValueError("split_seed must be positive")
        split_counts = _immutable_mapping(self.split_counts, "split_counts")
        if set(split_counts) != _SPLIT_NAMES or any(
            not isinstance(count, int) or count < 1 for count in split_counts.values()
        ):
            raise ValueError("split_counts must contain positive registered split counts")
        object.__setattr__(self, "split_counts", split_counts)
        if tuple(self.arms) != _ARMS:
            raise ValueError("arms must be standard_grpo and rlmf in registered order")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(
            not isinstance(seed, int) or seed < 1 for seed in self.seeds
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
        if any(not isinstance(value, int) or value < 1 for value in positive):
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
        generation = _immutable_mapping(self.generation, "generation")
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
        object.__setattr__(self, "generation", generation)
        rewards = _immutable_mapping(self.reward_weights, "reward_weights")
        if rewards != {
            "soft_format": 3.0,
            "strict_format": 3.0,
            "factual_calibration": 1.0,
            "correctness": 1.0,
            "faithful_calibration": 12.0,
        }:
            raise ValueError("reward_weights must match the preregistration")
        object.__setattr__(self, "reward_weights", rewards)
        if tuple(self.confidence_values) != _CONFIDENCE_VALUES:
            raise ValueError("confidence_values must contain the eleven registered values")
        if self.bootstrap_seed_mode != "fixed_registered_seeds_prompt_cluster":
            raise ValueError("bootstrap_seed_mode must be registered")
        if not 0 < self.judge_differential_bias_upper_limit < 1:
            raise ValueError("judge_differential_bias_upper_limit must be in range")

    @classmethod
    def from_json(cls, path: str | Path) -> "RLMFConfig":
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("RLMF config must be a JSON object")
        for name in ("arms", "seeds", "lora_targets", "confidence_values"):
            if name in value:
                value[name] = tuple(value[name])
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "split_counts": dict(self.split_counts),
            "arms": list(self.arms),
            "seeds": list(self.seeds),
            "lora_targets": list(self.lora_targets),
            "generation": dict(self.generation),
            "reward_weights": dict(self.reward_weights),
            "confidence_values": list(self.confidence_values),
        }

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


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
            if score is not None and (not isinstance(score, (int, float)) or not 0 <= score <= 1):
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

    def __post_init__(self) -> None:
        _validate_id(self.study_id, "study_id")
        if self.arm not in _ARMS:
            raise ValueError("arm must be registered")
        if not isinstance(self.seed, int) or self.seed < 1:
            raise ValueError("seed must be positive")
        if self.split not in _SPLIT_NAMES:
            raise ValueError("split must be registered")
        _validate_id(self.example_id, "example_id")
        _validate_id(self.candidate_id, "candidate_id")
        if not isinstance(self.raw_output, str):
            raise ValueError("raw_output must be a string")
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
