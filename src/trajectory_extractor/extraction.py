from __future__ import annotations

import random
from contextlib import nullcontext
from contextlib import AbstractContextManager
from typing import Any

import numpy as np
import torch

from trajectory_extractor.types import ActivationRun, ExperimentConfig, ResponseRun


def generate_and_extract(
    model,
    tokenizer,
    prompt: str,
    *,
    config: ExperimentConfig,
    run_id: str,
    example_id: str,
    track: str,
    split: str,
    label: int,
    provenance: dict[str, Any] | None = None,
    intervention: AbstractContextManager | None = None,
) -> ActivationRun:
    _set_seed(config.seed)
    model.eval()
    inputs, used_chat_template = _tokenize_prompt(tokenizer, prompt)
    device = getattr(model, "device", torch.device(config.device))
    inputs = {name: value.to(device) for name, value in inputs.items()}
    input_token_count = int(inputs["input_ids"].shape[1])

    with intervention or nullcontext():
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=getattr(tokenizer, "pad_token_id", None)
                or getattr(tokenizer, "eos_token_id", None),
            )
        response_ids = generated.sequences[0, input_token_count:]
        if response_ids.numel() == 0:
            raise RuntimeError("Model generated no response tokens")
        if intervention is None:
            hidden = _single_pass_pre_token_states(
                model,
                generated.sequences,
                input_token_count=input_token_count,
                response_token_count=int(response_ids.numel()),
            )
            replay_method = "single_teacher_forced"
        else:
            # Stateful hooks are defined over incremental generation. Replaying
            # each causal prefix preserves their trigger semantics exactly.
            hidden = _prefix_replay_pre_token_states(
                model,
                generated.sequences,
                input_token_count=input_token_count,
                response_token_count=int(response_ids.numel()),
            )
            replay_method = "incremental_prefix"
    token_logprobs, token_entropies = _score_generated_tokens(generated.scores, response_ids)
    metadata = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "seed": config.seed,
        "dtype": config.dtype,
        "device": config.device,
        "response_token_start": input_token_count,
        "response_token_end": input_token_count + int(response_ids.numel()),
        "used_chat_template": used_chat_template,
        "activation_alignment": "pre_response_token",
        "activation_replay": replay_method,
    }
    metadata.update(provenance or {})
    run = ActivationRun(
        run_id=run_id,
        example_id=example_id,
        track=track,
        split=split,
        prompt=prompt,
        response=tokenizer.decode(response_ids, skip_special_tokens=True),
        label=int(label),
        input_token_count=input_token_count,
        response_token_ids=response_ids.detach().cpu().numpy().astype(np.int64),
        hidden_states=hidden.detach().float().cpu().numpy().astype(np.float16),
        token_logprobs=token_logprobs,
        token_entropies=token_entropies,
        provenance=metadata,
    )
    run.validate()
    return run


def _single_pass_pre_token_states(
    model,
    sequences: torch.Tensor,
    *,
    input_token_count: int,
    response_token_count: int,
) -> torch.Tensor:
    """Extract causal pre-token states from one teacher-forced replay.

    In a causal LM, position p cannot attend to positions greater than p. The
    state at p therefore equals the final state from replaying the prefix ending
    at p and is the state used to predict token p + 1.
    """
    with torch.inference_mode():
        replay = _hidden_state_backbone(model)(
            input_ids=sequences,
            attention_mask=torch.ones_like(sequences),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    positions = torch.arange(
        input_token_count - 1,
        input_token_count - 1 + response_token_count,
        device=sequences.device,
    )
    hidden = torch.stack(
        [state[0].index_select(0, positions) for state in replay.hidden_states],
        dim=1,
    )
    del replay
    return hidden


def _prefix_replay_pre_token_states(
    model,
    sequences: torch.Tensor,
    *,
    input_token_count: int,
    response_token_count: int,
) -> torch.Tensor:
    token_states: list[torch.Tensor] = []
    for response_offset in range(response_token_count):
        prefix_end = input_token_count + response_offset
        prefix_ids = sequences[:, :prefix_end]
        with torch.inference_mode():
            replay = _hidden_state_backbone(model)(
                input_ids=prefix_ids,
                attention_mask=torch.ones_like(prefix_ids),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        token_states.append(torch.stack([state[0, -1, :] for state in replay.hidden_states]))
        del replay
    return torch.stack(token_states, dim=0)


def _hidden_state_backbone(model):
    """Prefer the decoder backbone so replay does not materialize vocabulary logits."""
    backbone = getattr(model, "model", None)
    return backbone if callable(backbone) else model


def generate_response(
    model,
    tokenizer,
    prompt: str,
    *,
    config: ExperimentConfig,
    run_id: str,
    example_id: str,
    track: str,
    split: str,
    label: int,
    provenance: dict[str, Any] | None = None,
    intervention: AbstractContextManager | None = None,
) -> ResponseRun:
    """Generate deterministically without the expensive per-token activation replay."""
    _set_seed(config.seed)
    model.eval()
    inputs, used_chat_template = _tokenize_prompt(tokenizer, prompt)
    device = getattr(model, "device", torch.device(config.device))
    inputs = {name: value.to(device) for name, value in inputs.items()}
    input_token_count = int(inputs["input_ids"].shape[1])
    with intervention or nullcontext():
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=getattr(tokenizer, "pad_token_id", None)
                or getattr(tokenizer, "eos_token_id", None),
            )
    response_ids = generated[0, input_token_count:]
    if response_ids.numel() == 0:
        raise RuntimeError("Model generated no response tokens")
    metadata = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "seed": config.seed,
        "dtype": config.dtype,
        "device": config.device,
        "response_token_start": input_token_count,
        "response_token_end": input_token_count + int(response_ids.numel()),
        "used_chat_template": used_chat_template,
        "activation_replay": False,
    }
    if intervention is not None:
        metadata["intervention_triggered"] = getattr(intervention, "triggered", None)
        metadata["intervention_last_score"] = getattr(intervention, "last_score", None)
    metadata.update(provenance or {})
    return ResponseRun(
        run_id=run_id,
        example_id=example_id,
        track=track,
        split=split,
        prompt=prompt,
        response=tokenizer.decode(response_ids, skip_special_tokens=True),
        label=int(label),
        provenance=metadata,
    )


def _score_generated_tokens(scores, response_ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    logprobs: list[float] = []
    entropies: list[float] = []
    for logits, token_id in zip(scores, response_ids, strict=False):
        log_distribution = torch.log_softmax(logits[0].float(), dim=-1)
        distribution = torch.exp(log_distribution)
        logprobs.append(float(log_distribution[int(token_id)].item()))
        entropies.append(float((-(distribution * log_distribution)).sum().item()))
    return np.asarray(logprobs, dtype=np.float32), np.asarray(entropies, dtype=np.float32)


def _tokenize_prompt(tokenizer, prompt: str) -> tuple[dict[str, torch.Tensor], bool]:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            if "attention_mask" not in encoded:
                encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
            return dict(encoded), True
        except (TypeError, ValueError):
            # Tiny/fake tokenizers and non-chat models use the plain path.
            pass
    return tokenizer(prompt, return_tensors="pt"), False


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
