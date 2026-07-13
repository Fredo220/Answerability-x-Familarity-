from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

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
    raw_recorder: Callable[[Mapping[str, Any]], None] | None = None,
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
        candidate_id = f"{example_id}-member-{member}"
        if raw_recorder is not None:
            raw_recorder(
                {
                    "kind": "answer",
                    "step": step,
                    "example_id": example_id,
                    "candidate_id": candidate_id,
                    "group_member": member,
                    "generation_seed": member_seed,
                    "raw_output": raw,
                }
            )
        results.append(
            RLMFCompletion(
                study_id=study_id,
                arm=arm,
                seed=seed,
                split=split,
                example_id=example_id,
                candidate_id=candidate_id,
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
    raw_recorder: Callable[[str], None] | None = None,
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
    if raw_recorder is not None:
        raw_recorder(raw)
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
        study_id: str | None = None,
        generation_seed: int | None = None,
        raw_answer_recorder: Callable[[Mapping[str, Any]], None] | None = None,
        pre_advantage_recorder: Callable[[Mapping[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> None:
        if advantage_form not in {"standard", "mf"}:
            raise ValueError("advantage_form must be standard or mf")
        if not callable(metacognition_scorer):
            raise ValueError("metacognition_scorer must be callable")
        self.advantage_form = advantage_form
        self._metacognition_scorer = metacognition_scorer
        self._faith_reward_index = faith_reward_index
        self._rlmf_study_id = study_id
        self._rlmf_generation_seed = generation_seed
        self._raw_answer_recorder = raw_answer_recorder
        self._pre_advantage_recorder = pre_advantage_recorder
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
        if self._pre_advantage_recorder is not None:
            step = int(getattr(getattr(self, "state", None), "global_step", 0))
            self._pre_advantage_recorder(
                {
                    "step": step,
                    "candidate_ids": _candidate_ids(inputs, step, int(self.num_generations)),
                    "reward_matrix": rewards.detach().cpu().tolist(),
                    "metacognitive_rewards": meta.detach().cpu().tolist(),
                }
            )
        return rewards

    def _generate_and_score_completions(self, inputs):
        with self._seeded_answer_generation(inputs):
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

    @contextmanager
    def _seeded_answer_generation(self, inputs: Sequence[Mapping[str, Any]]) -> Iterator[None]:
        if self._rlmf_study_id is None or self._rlmf_generation_seed is None:
            yield
            return
        if getattr(self, "use_vllm", False) or getattr(self, "use_transformers_paged", False):
            raise RuntimeError("registered per-candidate seeds require regular Transformers generation")
        targets = []
        for candidate in (
            getattr(self, "model_wrapped", None),
            getattr(self, "model", None),
        ):
            if candidate is not None and hasattr(candidate, "generate") and all(
                candidate is not existing for existing in targets
            ):
                targets.append(candidate)
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "unwrap_model"):
            unwrapped = accelerator.unwrap_model(getattr(self, "model_wrapped", None))
            if unwrapped is not None and hasattr(unwrapped, "generate") and all(
                unwrapped is not existing for existing in targets
            ):
                targets.append(unwrapped)
        if not targets:
            raise RuntimeError("paired trainer has no patchable Transformers generation model")
        originals = []
        try:
            for target in targets:
                had_instance_attribute = "generate" in getattr(target, "__dict__", {})
                original = target.generate

                def seeded_generate(*args: Any, __original=original, **kwargs: Any):
                    return _seeded_batch_generate(
                        __original,
                        args,
                        kwargs,
                        inputs=inputs,
                        study_id=self._rlmf_study_id,
                        seed=self._rlmf_generation_seed,
                        step=int(getattr(getattr(self, "state", None), "global_step", 0)),
                        group_size=int(self.num_generations),
                        tokenizer=self.processing_class,
                        pad_token_id=int(getattr(self, "pad_token_id", 0) or 0),
                        raw_recorder=self._raw_answer_recorder,
                    )

                setattr(target, "generate", seeded_generate)
                originals.append((target, original, had_instance_attribute))
            yield
        finally:
            for target, original, had_instance_attribute in reversed(originals):
                if had_instance_attribute:
                    setattr(target, "generate", original)
                else:
                    delattr(target, "generate")

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
    with _isolated_rng(seed, device), torch.inference_mode():
        output = model.generate(**encoded, **settings, max_new_tokens=96)
    prompt_length = encoded["input_ids"].shape[-1]
    return tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)


@contextmanager
def _isolated_rng(seed: int, device: Any = None) -> Iterator[None]:
    torch_device = torch.device("cpu" if device is None else device)
    cuda_devices: list[int] = []
    if torch_device.type == "cuda" and torch.cuda.is_available():
        cuda_devices.append(torch.cuda.current_device() if torch_device.index is None else torch_device.index)
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        yield


def _seeded_batch_generate(
    generate: Callable[..., Tensor],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    inputs: Sequence[Mapping[str, Any]],
    study_id: str,
    seed: int,
    step: int,
    group_size: int,
    tokenizer: Any,
    pad_token_id: int,
    raw_recorder: Callable[[Mapping[str, Any]], None] | None,
) -> Tensor:
    call_kwargs = dict(kwargs)
    if args:
        if len(args) != 1 or "input_ids" in call_kwargs:
            raise ValueError("seeded generation accepts one positional input_ids tensor")
        call_kwargs["input_ids"] = args[0]
    input_ids = call_kwargs.get("input_ids")
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise ValueError("seeded generation requires batched input_ids")
    if input_ids.shape[0] != len(inputs):
        raise ValueError("seeded generation batch must align with trainer inputs")
    rows = []
    prompt_length = input_ids.shape[1]
    for index, row in enumerate(inputs):
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("seeded generation inputs require example_id")
        member = index % group_size
        member_seed = derive_generation_seed(
            study_id, seed, step, example_id, member, "answer"
        )
        row_kwargs = {
            name: _slice_generation_value(value, index, len(inputs))
            for name, value in call_kwargs.items()
        }
        with _isolated_rng(member_seed, input_ids.device):
            generated = generate(**row_kwargs)
        if not isinstance(generated, Tensor) or generated.ndim != 2 or generated.shape[0] != 1:
            raise ValueError("seeded generate must return one token row per candidate")
        rows.append(generated[0])
        if raw_recorder is not None:
            candidate_id = f"{example_id}-step-{step}-member-{member}"
            raw_recorder(
                {
                    "kind": "answer",
                    "step": step,
                    "example_id": example_id,
                    "candidate_id": candidate_id,
                    "group_member": member,
                    "generation_seed": member_seed,
                    "raw_output": tokenizer.decode(
                        generated[0][prompt_length:], skip_special_tokens=True
                    ),
                }
            )
    width = max(row.numel() for row in rows)
    output = input_ids.new_full((len(rows), width), pad_token_id)
    for index, row in enumerate(rows):
        output[index, : row.numel()] = row
    return output


def _slice_generation_value(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[index : index + 1]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return type(value)(value[index : index + 1])
    return value


def _candidate_ids(inputs: Sequence[Mapping[str, Any]], step: int, group_size: int) -> list[str]:
    candidate_ids = []
    for index, row in enumerate(inputs):
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("pre-advantage records require example_id")
        candidate_ids.append(f"{example_id}-step-{step}-member-{index % group_size}")
    return candidate_ids
