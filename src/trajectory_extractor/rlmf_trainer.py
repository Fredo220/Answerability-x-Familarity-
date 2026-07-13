from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor

from trajectory_extractor.rlmf_advantage import rlmf_advantage, standard_grpo_advantage
from trajectory_extractor.rlmf_artifacts import sha256_file
from trajectory_extractor.rlmf_format import (
    ParsedMetacognitiveOutput,
    parse_metascore_output,
    parse_rlmf_output,
)
from trajectory_extractor.rlmf_types import RLMFCompletion


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TRL_MANIFEST = _REPOSITORY_ROOT / "third_party" / "trl" / "UPSTREAM.json"
_GENERATION = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.05,
    "enable_thinking": False,
}
_RUNTIME_CLASSES: dict[type, type] = {}


def answer_prompt(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")
    return (
        f"Question: {question.strip()}\n"
        "Answer using exactly these two tags and no other text: "
        "<sentence>your answer</sentence><confidence>0.0-1.0 in steps of 0.1</confidence>"
    )


def metacognition_prompt(question: str, candidate_answer: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")
    if not isinstance(candidate_answer, str) or not candidate_answer.strip():
        raise ValueError("candidate_answer must be non-empty")
    return (
        f"Question: {question.strip()}\n"
        f"Your Answer: {candidate_answer.strip()}\n"
        "How well does the linguistic decisiveness of this answer match your true internal "
        "confidence? Respond using exactly one tag and no other text: "
        "<metascore>0.0-1.0 in steps of 0.1</metascore>"
    )


def derive_generation_seed(
    study_id: str,
    seed: int,
    step: int,
    example_id: str,
    group_member: int,
    query_kind: Literal["answer", "metacognition"],
) -> int:
    if query_kind not in {"answer", "metacognition"}:
        raise ValueError("query_kind must be answer or metacognition")
    if any(type(value) is not int or value < 0 for value in (seed, step, group_member)):
        raise ValueError("seed, step, and group_member must be non-negative integers")
    if not all(isinstance(value, str) and value for value in (study_id, example_id)):
        raise ValueError("study_id and example_id must be non-empty strings")
    payload = [study_id, seed, step, example_id, group_member, query_kind]
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def generate_group(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    group_size: int,
    seed: int,
    study_id: str = "generation",
    arm: str = "standard_grpo",
    step: int = 0,
    example_id: str = "example",
    split: str = "rl_train",
    checkpoint_hash: str = "0" * 64,
    config_hash: str = "0" * 64,
    parent_hashes: Mapping[str, str] | None = None,
    generation: Mapping[str, Any] | None = None,
) -> tuple[RLMFCompletion, ...]:
    if type(group_size) is not int or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    generation_settings = dict(_GENERATION if generation is None else generation)
    rendered = answer_prompt(prompt)
    results = []
    for member in range(group_size):
        member_seed = derive_generation_seed(
            study_id, seed, step, example_id, member, "answer"
        )
        raw = _generate_text(
            model, tokenizer, rendered, seed=member_seed, generation=generation_settings
        )
        results.append(
            RLMFCompletion(
                study_id=study_id,
                arm=arm,
                seed=seed,
                split=split,
                example_id=example_id,
                candidate_id=f"{example_id}-member-{member}",
                raw_output=raw,
                parsed=parse_rlmf_output(raw),
                checkpoint_hash=checkpoint_hash,
                config_hash=config_hash,
                parent_hashes={} if parent_hashes is None else parent_hashes,
                source_question=prompt,
            )
        )
    return tuple(results)


def query_metacognitive_score(
    model: Any,
    tokenizer: Any,
    completion: RLMFCompletion,
    *,
    seed: int,
    generation: Mapping[str, Any] | None = None,
    raw_sink: list[str] | None = None,
) -> ParsedMetacognitiveOutput:
    if not isinstance(completion, RLMFCompletion):
        raise ValueError("completion must be an RLMFCompletion")
    if not completion.source_question:
        raise ValueError("completion must retain its source question")
    candidate_answer = (
        completion.parsed.answer.strip()
        or completion.raw_output.strip()
        or "[empty completion]"
    )
    rendered = metacognition_prompt(completion.source_question, candidate_answer)
    raw = _generate_text(
        model,
        tokenizer,
        rendered,
        seed=seed,
        generation=dict(_GENERATION if generation is None else generation),
    )
    if raw_sink is not None:
        raw_sink.append(raw)
    return parse_metascore_output(raw)


def validate_installed_trl() -> type:
    """Load TRL only on demand and fail closed before private integration."""
    manifest = json.loads(_TRL_MANIFEST.read_text())
    try:
        installed_version = importlib.metadata.version("trl")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("pinned trl==0.23.0 runtime is missing") from error
    if installed_version != "0.23.0" or manifest.get("tag") != "v0.23.0":
        raise RuntimeError("installed TRL must be exactly trl==0.23.0")
    module = importlib.import_module("trl.trainer.grpo_trainer")
    source = Path(inspect.getsourcefile(module.GRPOTrainer) or "")
    if not source.is_file() or sha256_file(source) != manifest.get("sha256"):
        raise RuntimeError("installed TRL GRPOTrainer source hash does not match vendored manifest")
    expected = {
        "_calculate_rewards": (
            "self", "inputs", "prompts", "completions", "completion_ids_list"
        ),
        "_generate_and_score_completions": ("self", "inputs"),
    }
    for name, parameters in expected.items():
        actual = tuple(inspect.signature(getattr(module.GRPOTrainer, name)).parameters)
        if actual != parameters:
            raise RuntimeError(f"installed TRL private signature mismatch: {name}")
    return module.GRPOTrainer


class _PairedRLMFMixin:
    def __init__(
        self,
        *args: Any,
        advantage_form: Literal["standard", "mf"],
        metacognition_scorer: Any,
        faith_reward_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        if advantage_form not in {"standard", "mf"}:
            raise ValueError("advantage_form must be standard or mf")
        if not callable(metacognition_scorer):
            raise ValueError("metacognition_scorer must be callable")
        self.advantage_form = advantage_form
        self._metacognition_scorer = metacognition_scorer
        self._faith_reward_index = faith_reward_index
        self.last_reward_matrix: Tensor | None = None
        self.last_metacognitive_rewards: Tensor | None = None
        super().__init__(*args, **kwargs)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards = super()._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        meta = self._metacognition_scorer(inputs, completions)
        meta = torch.as_tensor(meta, dtype=rewards.dtype, device=rewards.device)
        if meta.ndim != 1 or meta.shape[0] != rewards.shape[0] or not torch.isfinite(meta).all():
            raise ValueError("metacognitive rewards must be finite and align with completions")
        self.last_reward_matrix = rewards.detach().clone()
        self.last_metacognitive_rewards = meta.detach().clone()
        return rewards

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if self.last_reward_matrix is None or self.last_metacognitive_rewards is None:
            raise RuntimeError("paired reward path did not record metacognitive rewards")
        rewards = self.last_reward_matrix
        weights = torch.as_tensor(
            self.reward_weights, dtype=rewards.dtype, device=rewards.device
        )
        if weights.ndim != 1 or weights.numel() != rewards.shape[1]:
            raise ValueError("reward weights must align with reward functions")
        faith_index = self._resolve_faith_reward_index(rewards.shape[1])
        weighted = rewards * weights.unsqueeze(0)
        faith = weighted[:, faith_index]
        other = weighted.sum(dim=1) - faith
        group_size = int(self.num_generations)
        if rewards.shape[0] % group_size:
            raise ValueError("reward batch contains an incomplete rollout group")
        advantages = []
        for start in range(0, rewards.shape[0], group_size):
            group = slice(start, start + group_size)
            if self.advantage_form == "standard":
                advantage = standard_grpo_advantage(other[group], faith[group])
            else:
                advantage = rlmf_advantage(
                    other[group], faith[group], self.last_metacognitive_rewards[group]
                )
            advantages.append(advantage)
        all_advantages = torch.cat(advantages)
        process_index = getattr(self.accelerator, "process_index", 0)
        local_size = len(inputs)
        process_slice = slice(process_index * local_size, (process_index + 1) * local_size)
        output["advantages"] = all_advantages[process_slice]
        logs = getattr(self, "_logs", None)
        if isinstance(logs, Mapping) and isinstance(logs.get("advantages"), list):
            del logs["advantages"][-len(all_advantages):]
            logs["advantages"].extend(all_advantages.tolist())
        return output

    def _resolve_faith_reward_index(self, count: int) -> int:
        if self._faith_reward_index is not None:
            index = self._faith_reward_index
        else:
            matches = [
                index
                for index, name in enumerate(self.reward_func_names)
                if name == "faithful_calibration"
            ]
            if len(matches) != 1:
                raise ValueError("exactly one faithful_calibration reward is required")
            index = matches[0]
        if type(index) is not int or not 0 <= index < count:
            raise ValueError("faith reward index is invalid")
        return index


class _LazyTrainerMeta(type):
    def __call__(cls, *args: Any, **kwargs: Any):
        base = kwargs.pop("_base_trainer_cls", None)
        if base is None:
            base = validate_installed_trl()
        runtime = _RUNTIME_CLASSES.get(base)
        if runtime is None:
            runtime = type(
                "PairedRLMFTrainer",
                (_PairedRLMFMixin, base),
                {"__module__": __name__},
            )
            _RUNTIME_CLASSES[base] = runtime
        return runtime(*args, **kwargs)


class PairedRLMFTrainer(metaclass=_LazyTrainerMeta):
    """Lazy constructor for the single checked paired TRL integration path."""


def _generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    seed: int,
    generation: Mapping[str, Any],
) -> str:
    if hasattr(model, "generate_text"):
        result = model.generate_text(prompt, seed=seed, generation=generation)
        if not isinstance(result, str):
            raise ValueError("generate_text must return a string")
        return result
    settings = dict(generation)
    settings.pop("enable_thinking", None)
    encoded = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        encoded = {name: value.to(device) for name, value in encoded.items()}
    generator = torch.Generator(device=device or "cpu").manual_seed(seed)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            **settings,
            max_new_tokens=96,
            generator=generator,
        )
    prompt_length = encoded["input_ids"].shape[-1]
    return tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
