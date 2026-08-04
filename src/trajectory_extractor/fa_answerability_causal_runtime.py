"""Audited Gemma runtime for the Same-String answerability causal pilot."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, is_dataclass, replace
from typing import Any

import numpy as np
import torch

from trajectory_extractor.fa_activations import (
    ANCHOR_NAMES,
    _transformer_layers as _activation_transformer_layers,
)
from trajectory_extractor.fa_answerability_causal import CAUSAL_DIRECTION_ANCHOR


TERMINAL_TOKEN_POLICY = "append_eos"


@dataclass(frozen=True)
class VectorAuditHashes:
    """Canonical source and represented tensor-byte hashes for one vector."""

    source_vector_sha256: str
    applied_vector_sha256: str
    represented_dtype: str


def vector_audit_hashes(
    vector: Any,
    *,
    represented_dtype: str | torch.dtype,
) -> VectorAuditHashes:
    """Hash a source vector and its runtime-dtype representation identically to hooks."""
    if isinstance(vector, torch.Tensor):
        source = vector.detach().clone()
    else:
        source = torch.from_numpy(np.array(vector, copy=True))
    if source.ndim != 1 or source.numel() == 0 or not torch.is_floating_point(source):
        raise ValueError("vector hashes require a nonempty floating-point vector")
    if not bool(torch.isfinite(source).all()):
        raise ValueError("vector hashes require finite values")
    dtype = _runtime_dtype(represented_dtype)
    represented = source.to(dtype=dtype)
    return VectorAuditHashes(
        source_vector_sha256=_tensor_sha256(source),
        applied_vector_sha256=_tensor_sha256(represented),
        represented_dtype=str(dtype),
    )


@dataclass(frozen=True)
class ResolvedCausalAnchor:
    """Exact prompt replay and one registered intervention position."""

    example_id: str
    anchor_name: str
    position: int
    rendered_prompt_sha256: str
    input_ids: tuple[int, ...]
    prompt_token_ids_sha256: str


@dataclass(frozen=True)
class InterventionAudit:
    """Prompt identity and residual-write evidence for one model forward pass."""

    example_id: str
    rendered_prompt_sha256: str
    prompt_token_ids_sha256: str
    model_prefix_token_ids_sha256: str
    model_prefix_length: int
    layer_id: int
    position: int
    modified_site_count: int
    source_vector_sha256: str
    applied_vector_sha256: str
    represented_dtype: str
    represented_device: str
    hook_call_count: int
    hook_cleanup_verified: bool

    def __post_init__(self) -> None:
        if self.prompt_token_ids_sha256 != self.model_prefix_token_ids_sha256:
            raise ValueError("model input prefix does not match the registered prompt")
        if self.hook_call_count != 1 or self.modified_site_count != 1:
            raise ValueError("intervention must modify exactly one residual site")
        if not self.hook_cleanup_verified:
            raise ValueError("intervention hook cleanup was not verified")


@dataclass(frozen=True)
class SequenceLogProbability:
    candidate_text: str
    candidate_token_ids: tuple[int, ...]
    raw_log_probability: float
    length_normalized_log_probability: float
    terminal_token_policy: str
    audit: InterventionAudit | None


@dataclass(frozen=True)
class AnswerabilityCandidateScores:
    correct: SequenceLogProbability
    unknown: SequenceLogProbability
    raw_margin: float
    length_normalized_margin: float


@dataclass(frozen=True)
class CausalGeneration:
    generated_text: str
    generated_token_ids: tuple[int, ...]
    audit: InterventionAudit | None


class _TemporaryResidualAddition(AbstractContextManager):
    """Own one decoder hook and retain only the evidence needed for its audit."""

    def __init__(
        self,
        model: Any,
        *,
        layer_id: int,
        position: int,
        vector: Any,
    ):
        if type(layer_id) is not int or layer_id < 0:
            raise ValueError("layer_id must be a nonnegative integer")
        if type(position) is not int or position < 0:
            raise ValueError("position must be a nonnegative integer")
        source = torch.as_tensor(vector).detach().clone()
        if source.ndim != 1 or source.numel() == 0:
            raise ValueError("intervention vector must be one-dimensional and nonempty")
        if not torch.is_floating_point(source) or not bool(torch.isfinite(source).all()):
            raise ValueError("intervention vector must contain finite floating-point values")
        self.model = model
        self.layer_id = layer_id
        self.position = position
        self._source_vector = source
        self.source_vector_sha256 = vector_audit_hashes(
            source, represented_dtype=source.dtype
        ).source_vector_sha256
        self.applied_vector_sha256: str | None = None
        self.represented_dtype: str | None = None
        self.represented_device: str | None = None
        self.hook_call_count = 0
        self.modified_site_count = 0
        self.removed = False
        self._handle: Any | None = None

    def __enter__(self) -> "_TemporaryResidualAddition":
        layers = _activation_transformer_layers(self.model)
        if self.layer_id >= len(layers):
            raise ValueError("layer_id exceeds the Gemma decoder layer count")
        self._handle = layers[self.layer_id].register_forward_hook(self._hook)
        return self

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        self.hook_call_count += 1
        hidden = _decoder_hidden(output)
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("decoder hidden state must have shape [1, token, hidden]")
        if self.position >= hidden.shape[1]:
            raise ValueError("intervention position exceeds decoder sequence length")
        if self._source_vector.shape[0] != hidden.shape[2]:
            raise ValueError("intervention vector does not match the hidden dimension")

        represented = self._source_vector.to(
            dtype=hidden.dtype,
            device=hidden.device,
        )
        self.represented_dtype = str(represented.dtype)
        self.represented_device = str(represented.device)
        self.applied_vector_sha256 = vector_audit_hashes(
            self._source_vector, represented_dtype=represented.dtype
        ).applied_vector_sha256
        changed = hidden.clone()
        changed[0, self.position, :] = changed[0, self.position, :] + represented
        self.modified_site_count += 1
        return _replace_decoder_hidden(output, hidden, changed)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.removed = True


def temporary_residual_addition(
    model: Any,
    *,
    layer_id: int,
    position: int,
    vector: Any,
) -> _TemporaryResidualAddition:
    """Return a context manager for one additive Gemma residual-site hook."""

    return _TemporaryResidualAddition(
        model,
        layer_id=layer_id,
        position=position,
        vector=vector,
    )


def resolve_causal_anchor(
    prompt: Any,
    rendered_prompt: str,
    tokenizer: Any,
    *,
    anchor_name: str = CAUSAL_DIRECTION_ANCHOR,
) -> ResolvedCausalAnchor:
    """Replay an already rendered prompt and fail on any byte or token drift."""

    if anchor_name not in ANCHOR_NAMES:
        raise ValueError("causal anchor name is not registered")
    if not isinstance(rendered_prompt, str) or not rendered_prompt:
        raise ValueError("rendered prompt must be nonempty text")
    observed_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
    expected_hash = getattr(prompt, "rendered_prompt_sha256", None)
    if observed_hash != expected_hash:
        raise ValueError("rendered prompt hash does not match the causal prompt")

    expected_ids = tuple(int(value) for value in getattr(prompt, "rendered_token_ids", ()))
    if not expected_ids:
        raise ValueError("causal prompt must record rendered token IDs")
    replayed_ids = _tokenize_ids(tokenizer, rendered_prompt)
    if replayed_ids != expected_ids:
        raise ValueError("rendered prompt token IDs do not match the causal prompt")

    anchor_positions = _resolve_anchor_positions(
        prompt,
        rendered_prompt,
        tokenizer,
        expected_ids,
    )
    return ResolvedCausalAnchor(
        example_id=str(getattr(prompt, "example_id")),
        anchor_name=anchor_name,
        position=anchor_positions[anchor_name],
        rendered_prompt_sha256=observed_hash,
        input_ids=expected_ids,
        prompt_token_ids_sha256=_token_ids_sha256(expected_ids),
    )


def score_answerability_candidates(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    rendered_prompt: str,
    *,
    layer_id: int | None = None,
    vector: Any | None = None,
    anchor_name: str = CAUSAL_DIRECTION_ANCHOR,
) -> AnswerabilityCandidateScores:
    """Score the correct archive code and UNKNOWN with identical EOS handling."""

    _validate_optional_intervention(layer_id, vector)
    resolved = resolve_causal_anchor(
        prompt,
        rendered_prompt,
        tokenizer,
        anchor_name=anchor_name,
    )
    correct_text = getattr(prompt, "registry_code", None)
    if not isinstance(correct_text, str) or not correct_text:
        raise ValueError("causal prompt must expose a nonempty registry_code")
    terminal_token_id = _eos_token_id(tokenizer)
    _set_eval(model)
    correct = _score_candidate(
        model,
        tokenizer,
        resolved,
        rendered_prompt,
        correct_text,
        terminal_token_id=terminal_token_id,
        layer_id=layer_id,
        vector=vector,
    )
    unknown = _score_candidate(
        model,
        tokenizer,
        resolved,
        rendered_prompt,
        "UNKNOWN",
        terminal_token_id=terminal_token_id,
        layer_id=layer_id,
        vector=vector,
    )
    return AnswerabilityCandidateScores(
        correct=correct,
        unknown=unknown,
        raw_margin=correct.raw_log_probability - unknown.raw_log_probability,
        length_normalized_margin=(
            correct.length_normalized_log_probability
            - unknown.length_normalized_log_probability
        ),
    )


def generate_causal_completion(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    rendered_prompt: str,
    *,
    layer_id: int | None = None,
    vector: Any | None = None,
    anchor_name: str = CAUSAL_DIRECTION_ANCHOR,
    max_new_tokens: int = 16,
) -> CausalGeneration:
    """Run one hooked prefill and an unhooked deterministic greedy cache loop."""

    _validate_optional_intervention(layer_id, vector)
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    resolved = resolve_causal_anchor(
        prompt,
        rendered_prompt,
        tokenizer,
        anchor_name=anchor_name,
    )
    _set_eval(model)
    device = _model_device(model)
    prefill_ids = torch.tensor([resolved.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(prefill_ids)
    outputs, audit = _forward_once(
        model,
        input_ids=prefill_ids,
        attention_mask=attention_mask,
        use_cache=True,
        resolved=resolved,
        layer_id=layer_id,
        vector=vector,
    )
    generated: list[int] = []
    eos_token_id = _eos_token_id(tokenizer)
    past_key_values = _model_past_key_values(outputs)

    while len(generated) < max_new_tokens:
        next_token = int(torch.argmax(_model_logits(outputs)[0, -1, :]).item())
        generated.append(next_token)
        if next_token == eos_token_id or len(generated) == max_new_tokens:
            break
        if past_key_values is None:
            raise ValueError("deterministic generation requires model cache output")
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
            ),
            dim=1,
        )
        next_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
        with torch.inference_mode():
            outputs = model(
                input_ids=next_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = _model_past_key_values(outputs)

    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise ValueError("tokenizer must expose decode for deterministic generation")
    generated_text = decode(generated, skip_special_tokens=True)
    if not isinstance(generated_text, str):
        raise ValueError("tokenizer.decode must return text")
    return CausalGeneration(
        generated_text=generated_text,
        generated_token_ids=tuple(generated),
        audit=audit,
    )


def _score_candidate(
    model: Any,
    tokenizer: Any,
    resolved: ResolvedCausalAnchor,
    rendered_prompt: str,
    candidate_text: str,
    *,
    terminal_token_id: int,
    layer_id: int | None,
    vector: Any | None,
) -> SequenceLogProbability:
    full_ids = _tokenize_ids(tokenizer, rendered_prompt + candidate_text)
    prompt_length = len(resolved.input_ids)
    if full_ids[:prompt_length] != resolved.input_ids:
        raise ValueError("candidate sequence token prefix does not match the causal prompt")
    continuation = (*full_ids[prompt_length:], terminal_token_id)
    if len(continuation) < 2:
        raise ValueError("candidate sequence must contain text before the terminal token")
    sequence = (*resolved.input_ids, *continuation)
    device = _model_device(model)
    input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    outputs, audit = _forward_once(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        resolved=resolved,
        layer_id=layer_id,
        vector=vector,
    )
    logits = _model_logits(outputs)
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("causal language model logits must have shape [1, token, vocab]")
    start = prompt_length - 1
    selected_logits = logits[0, start : start + len(continuation), :]
    if selected_logits.shape[0] != len(continuation):
        raise ValueError("model logits do not cover the complete candidate sequence")
    labels = torch.tensor(continuation, dtype=torch.long, device=logits.device)
    token_log_probabilities = torch.log_softmax(selected_logits.float(), dim=-1).gather(
        1, labels.unsqueeze(1)
    )
    raw = float(token_log_probabilities.sum().item())
    return SequenceLogProbability(
        candidate_text=candidate_text,
        candidate_token_ids=tuple(continuation),
        raw_log_probability=raw,
        length_normalized_log_probability=raw / len(continuation),
        terminal_token_policy=TERMINAL_TOKEN_POLICY,
        audit=audit,
    )


def _forward_once(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    use_cache: bool,
    resolved: ResolvedCausalAnchor,
    layer_id: int | None,
    vector: Any | None,
) -> tuple[Any, InterventionAudit | None]:
    prefix = tuple(int(value) for value in input_ids[0, : len(resolved.input_ids)].tolist())
    if prefix != resolved.input_ids:
        raise ValueError("model input prefix does not match the registered prompt")
    if layer_id is None:
        with torch.inference_mode():
            return (
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                ),
                None,
            )

    with temporary_residual_addition(
        model,
        layer_id=layer_id,
        position=resolved.position,
        vector=vector,
    ) as intervention:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
    audit = _intervention_audit(
        resolved,
        prefix,
        layer_id=layer_id,
        intervention=intervention,
    )
    return outputs, audit


def _intervention_audit(
    resolved: ResolvedCausalAnchor,
    prefix: Sequence[int],
    *,
    layer_id: int,
    intervention: _TemporaryResidualAddition,
) -> InterventionAudit:
    if (
        intervention.applied_vector_sha256 is None
        or intervention.represented_dtype is None
        or intervention.represented_device is None
    ):
        raise ValueError("intervention hook did not record an applied vector")
    return InterventionAudit(
        example_id=resolved.example_id,
        rendered_prompt_sha256=resolved.rendered_prompt_sha256,
        prompt_token_ids_sha256=resolved.prompt_token_ids_sha256,
        model_prefix_token_ids_sha256=_token_ids_sha256(prefix),
        model_prefix_length=len(prefix),
        layer_id=layer_id,
        position=resolved.position,
        modified_site_count=intervention.modified_site_count,
        source_vector_sha256=intervention.source_vector_sha256,
        applied_vector_sha256=intervention.applied_vector_sha256,
        represented_dtype=intervention.represented_dtype,
        represented_device=intervention.represented_device,
        hook_call_count=intervention.hook_call_count,
        hook_cleanup_verified=intervention.removed,
    )


def _decoder_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        hidden = output
    elif isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    elif hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state
    elif isinstance(output, Mapping) and "last_hidden_state" in output:
        hidden = output["last_hidden_state"]
    else:
        raise ValueError("decoder output does not expose a hidden-state tensor")
    if not isinstance(hidden, torch.Tensor):
        raise ValueError("decoder output hidden state must be a tensor")
    return hidden


def _replace_decoder_hidden(output: Any, original: torch.Tensor, changed: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return changed
    if isinstance(output, list):
        copied = list(output)
        copied[0] = changed
        return copied
    if isinstance(output, tuple):
        values = (changed, *output[1:])
        if hasattr(output, "_fields"):
            return type(output)(*values)
        return values
    if is_dataclass(output) and not isinstance(output, type):
        return replace(output, last_hidden_state=changed)
    if isinstance(output, Mapping):
        values = dict(output)
        values["last_hidden_state"] = changed
        try:
            return type(output)(**values)
        except TypeError:
            copied = copy.copy(output)
            copied["last_hidden_state"] = changed
            return copied
    if hasattr(output, "last_hidden_state"):
        copied = copy.copy(output)
        setattr(copied, "last_hidden_state", changed)
        return copied
    raise ValueError("decoder output cannot preserve its container after intervention")


def _model_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get("logits")
    if logits is None and isinstance(output, (tuple, list)) and output:
        logits = output[0]
    if not isinstance(logits, torch.Tensor):
        raise ValueError("causal language model output must expose tensor logits")
    return logits


def _model_past_key_values(output: Any) -> Any:
    if hasattr(output, "past_key_values"):
        return output.past_key_values
    if isinstance(output, Mapping):
        return output.get("past_key_values")
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[1]
    return None


def _tokenize_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("tokenizer must return an input_ids mapping")
    raw = encoded["input_ids"]
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().tolist()
    if raw and isinstance(raw[0], Sequence) and not isinstance(raw[0], (str, bytes)):
        if len(raw) != 1:
            raise ValueError("tokenizer must return exactly one token sequence")
        raw = raw[0]
    try:
        result = tuple(int(value) for value in raw)
    except (TypeError, ValueError) as error:
        raise ValueError("tokenizer returned invalid token IDs") from error
    if not result:
        raise ValueError("tokenizer returned an empty token sequence")
    return result


def _resolve_anchor_positions(
    prompt: Any,
    rendered_prompt: str,
    tokenizer: Any,
    expected_ids: tuple[int, ...],
) -> dict[str, int]:
    user_text = getattr(prompt, "user_text", None)
    target_text = getattr(prompt, "target_text", None)
    if not isinstance(user_text, str) or not user_text:
        raise ValueError("causal prompt must expose nonempty user_text")
    if not isinstance(target_text, str) or not target_text:
        raise ValueError("causal prompt must expose nonempty target_text")
    render = getattr(tokenizer, "apply_chat_template", None)
    if not callable(render):
        raise ValueError("causal anchor resolution requires apply_chat_template")
    messages = [{"role": "user", "content": user_text}]
    without_generation = render(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    replayed = render(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if replayed != rendered_prompt:
        raise ValueError("chat-template replay does not match the rendered prompt")
    if (
        not isinstance(without_generation, str)
        or not rendered_prompt.startswith(without_generation)
        or len(rendered_prompt) == len(without_generation)
    ):
        raise ValueError("chat template must add a nonempty assistant prefix")
    template_ids = render(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if _normalize_raw_ids(template_ids) != expected_ids:
        raise ValueError("chat-template token IDs do not match the causal prompt")

    encoded = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if not isinstance(encoded, Mapping):
        raise ValueError("tokenizer must return exact offset mapping")
    offsets = encoded.get("offset_mapping")
    if isinstance(offsets, torch.Tensor):
        offsets = offsets.detach().cpu().tolist()
    if offsets and isinstance(offsets[0], Sequence) and len(offsets) == 1 and offsets[0]:
        first = offsets[0][0]
        if isinstance(first, Sequence):
            offsets = offsets[0]
    try:
        normalized_offsets = tuple((int(start), int(end)) for start, end in offsets)
    except (TypeError, ValueError) as error:
        raise ValueError("tokenizer must return exact offset mapping") from error
    if len(normalized_offsets) != len(expected_ids):
        raise ValueError("offset mapping does not align with causal prompt token IDs")

    occurrences = []
    start = 0
    while True:
        found = rendered_prompt.find(user_text, start)
        if found < 0:
            break
        occurrences.append(found)
        start = found + 1
    if len(occurrences) != 1:
        raise ValueError("rendered prompt contains an ambiguous user-text occurrence")
    user_start = occurrences[0]
    intro_span = _validated_target_span(prompt, "target_intro_span", user_text, target_text)
    query_span = _validated_target_span(prompt, "target_query_span", user_text, target_text)
    if intro_span[1] > query_span[0]:
        raise ValueError("causal target spans must preserve introduction/query order")
    target_rendered_span = tuple(user_start + value for value in intro_span)
    user_rendered_span = (user_start, user_start + len(user_text))
    positions = {
        "target_intro_end": _last_overlapping_position(
            normalized_offsets,
            target_rendered_span,
            "target introduction",
        ),
        "user_prompt_end": _last_overlapping_position(
            normalized_offsets,
            user_rendered_span,
            "user prompt",
        ),
        "assistant_prefix_end": len(expected_ids) - 1,
    }
    if not (
        positions["target_intro_end"]
        < positions["user_prompt_end"]
        < positions["assistant_prefix_end"]
    ):
        raise ValueError("causal anchor positions must be strictly ordered")
    return positions


def _normalize_raw_ids(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, Mapping):
        raw = raw.get("input_ids")
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().tolist()
    if raw and isinstance(raw[0], Sequence) and not isinstance(raw[0], (str, bytes)):
        if len(raw) != 1:
            raise ValueError("tokenizer must return exactly one token sequence")
        raw = raw[0]
    try:
        return tuple(int(value) for value in raw)
    except (TypeError, ValueError) as error:
        raise ValueError("tokenizer returned invalid token IDs") from error


def _validated_target_span(
    prompt: Any,
    name: str,
    user_text: str,
    target_text: str,
) -> tuple[int, int]:
    value = getattr(prompt, name, None)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise ValueError("causal target spans must be integer pairs")
    start, end = value
    if start < 0 or end <= start or user_text[start:end] != target_text:
        raise ValueError("causal target span does not identify target_text")
    return start, end


def _last_overlapping_position(
    offsets: Sequence[tuple[int, int]],
    span: tuple[int, int],
    name: str,
) -> int:
    start, end = span
    matches = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_start < end and token_end > start
    ]
    if not matches:
        raise ValueError(f"exact offset mapping does not cover {name}")
    return matches[-1]


def _token_ids_sha256(values: Sequence[int]) -> str:
    payload = json.dumps(
        [int(value) for value in values],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().contiguous().cpu()
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    raw = contiguous.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _runtime_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        dtype = value
    elif isinstance(value, str) and value.startswith("torch."):
        dtype = getattr(torch, value.removeprefix("torch."), None)
    else:
        dtype = None
    if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("represented dtype must be a floating-point torch dtype")
    return dtype


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _eos_token_id(tokenizer: Any) -> int:
    value = getattr(tokenizer, "eos_token_id", None)
    if type(value) is not int or value < 0:
        raise ValueError("tokenizer must expose one nonnegative eos_token_id")
    return value


def _set_eval(model: Any) -> None:
    evaluate = getattr(model, "eval", None)
    if not callable(evaluate):
        raise ValueError("causal runtime model must expose eval")
    evaluate()


def _validate_optional_intervention(layer_id: int | None, vector: Any | None) -> None:
    if (layer_id is None) != (vector is None):
        raise ValueError("layer_id and intervention vector must be supplied together")
