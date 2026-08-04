from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trajectory_extractor.fa_answerability_causal_runtime import (
    generate_causal_completion,
    resolve_causal_anchor,
    score_answerability_candidates,
    temporary_residual_addition,
)


TOKENIZER_REVISION = "b" * 40


class FakeTokenizer:
    chat_template = "fake-gemma-template-v1"
    name_or_path = "fake/gemma-tokenizer"
    init_kwargs = {"revision": TOKENIZER_REVISION, "use_fast": True}
    all_special_ids = (1, 2, 3, 4)
    eos_token_id = 4

    _specials = {"<bos>": 1, "<user>": 2, "<assistant>": 3}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        rendered = f"<bos><user>{messages[0]['content']}"
        if add_generation_prompt:
            rendered += "<assistant>"
        if not tokenize:
            return rendered
        return self(rendered, add_special_tokens=False)["input_ids"]

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
    ):
        assert add_special_tokens is False
        input_ids = []
        offsets = []
        index = 0
        while index < len(text):
            special = next(
                (value for value in self._specials if text.startswith(value, index)),
                None,
            )
            if special is None:
                input_ids.append(10 + ord(text[index]) % 70)
                offsets.append((index, index + 1))
                index += 1
            else:
                input_ids.append(self._specials[special])
                offsets.append((index, index + len(special)))
                index += len(special)
        result = {"input_ids": input_ids}
        if return_special_tokens_mask:
            result["special_tokens_mask"] = [
                int(token in self.all_special_ids) for token in input_ids
            ]
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, token_ids, *, skip_special_tokens):
        decoded = []
        for token_id in token_ids:
            if token_id == self.eos_token_id and skip_special_tokens:
                continue
            if token_id == 90:
                decoded.append("Z")
            elif token_id not in self.all_special_ids:
                decoded.append(f"<{token_id}>")
        return "".join(decoded)


def causal_prompt(tokenizer: FakeTokenizer):
    user_text = (
        "Task: Causa0000 has archive code Z0001. "
        "Question: What is the archive code for Causa0000?"
    )
    target_text = "Causa0000"
    intro_start = user_text.index(target_text)
    query_start = user_text.rindex(target_text)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return SimpleNamespace(
        example_id="causal-example-1",
        user_text=user_text,
        target_text=target_text,
        target_intro_span=(intro_start, intro_start + len(target_text)),
        target_query_span=(query_start, query_start + len(target_text)),
        registry_code="Z0001",
        rendered_token_ids=tuple(input_ids),
        rendered_prompt_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    ), rendered


@dataclass
class FakeLayerOutput:
    last_hidden_state: torch.Tensor
    cache_marker: str


class FakeDecoderLayer(nn.Module):
    def __init__(self, output_kind="tensor", *, raises=False):
        super().__init__()
        self.output_kind = output_kind
        self.raises = raises
        self.forward_calls = 0

    def forward(self, hidden_states):
        self.forward_calls += 1
        if self.raises:
            raise RuntimeError("decoder failed")
        if self.output_kind == "tensor":
            return hidden_states
        if self.output_kind == "tuple":
            return hidden_states, "cache"
        if self.output_kind == "list":
            return [hidden_states, "cache"]
        if self.output_kind == "model_output":
            return FakeLayerOutput(hidden_states, "cache")
        raise AssertionError("unknown fake output kind")


class GemmaShapedModel(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([layer]))
        self._device_marker = nn.Parameter(torch.zeros(()), requires_grad=False)


def hidden_from(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    return output.last_hidden_state


@pytest.mark.parametrize("output_kind", ["tensor", "tuple", "list", "model_output"])
def test_temporary_hook_adds_once_and_preserves_decoder_output_container(output_kind):
    layer = FakeDecoderLayer(output_kind)
    model = GemmaShapedModel(layer)
    source = torch.arange(12, dtype=torch.float16).reshape(1, 3, 4)
    vector = torch.tensor([0.5, 1.0, -1.5, 2.0], dtype=torch.float64)

    with temporary_residual_addition(
        model, layer_id=0, position=1, vector=vector
    ) as intervention:
        output = layer(source)

    changed = hidden_from(output)
    assert type(output) is {
        "tensor": torch.Tensor,
        "tuple": tuple,
        "list": list,
        "model_output": FakeLayerOutput,
    }[output_kind]
    assert torch.equal(changed[0, 0], source[0, 0])
    assert torch.equal(changed[0, 2], source[0, 2])
    assert torch.equal(changed[0, 1], source[0, 1] + vector.to(torch.float16))
    assert torch.equal(source, torch.arange(12, dtype=torch.float16).reshape(1, 3, 4))
    assert intervention.hook_call_count == 1
    assert intervention.modified_site_count == 1
    assert intervention.represented_dtype == "torch.float16"
    assert intervention.represented_device == "cpu"
    assert intervention.applied_vector_sha256
    assert intervention.removed is True
    assert not layer._forward_hooks


@pytest.mark.parametrize(
    ("position", "vector", "shape", "message"),
    [
        (3, torch.ones(4), (1, 3, 4), "position"),
        (1, torch.ones(5), (1, 3, 4), "hidden dimension"),
        (1, torch.ones(4), (3, 4), "shape"),
    ],
)
def test_temporary_hook_rejects_invalid_site_or_hidden_dimension_and_cleans_up(
    position, vector, shape, message
):
    layer = FakeDecoderLayer()
    model = GemmaShapedModel(layer)

    with pytest.raises(ValueError, match=message):
        with temporary_residual_addition(
            model, layer_id=0, position=position, vector=vector
        ):
            layer(torch.zeros(shape))

    assert not layer._forward_hooks


def test_temporary_hook_cleans_up_when_decoder_forward_raises():
    layer = FakeDecoderLayer(raises=True)
    model = GemmaShapedModel(layer)

    with pytest.raises(RuntimeError, match="decoder failed"):
        with temporary_residual_addition(
            model, layer_id=0, position=0, vector=torch.ones(4)
        ):
            layer(torch.zeros(1, 2, 4))

    assert not layer._forward_hooks


def test_anchor_resolution_rejects_rendered_prompt_hash_and_token_mismatches():
    tokenizer = FakeTokenizer()
    prompt, rendered = causal_prompt(tokenizer)

    resolved = resolve_causal_anchor(prompt, rendered, tokenizer)
    assert resolved.position < len(prompt.rendered_token_ids)
    assert resolved.rendered_prompt_sha256 == prompt.rendered_prompt_sha256
    assert resolved.input_ids == prompt.rendered_token_ids

    with pytest.raises(ValueError, match="rendered prompt hash"):
        resolve_causal_anchor(prompt, rendered + "x", tokenizer)

    prompt.rendered_token_ids = (*prompt.rendered_token_ids[:-1], 91)
    with pytest.raises(ValueError, match="token IDs"):
        resolve_causal_anchor(prompt, rendered, tokenizer)


class UniformScoringModel(GemmaShapedModel):
    def __init__(self, *, vocab_size=96, hidden_size=4):
        super().__init__(FakeDecoderLayer())
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def forward(self, *, input_ids, use_cache, attention_mask=None, **_kwargs):
        assert use_cache is False
        hidden = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], self.hidden_size, device=input_ids.device
        )
        hidden = self.model.layers[0](hidden)
        logits = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], self.vocab_size, device=input_ids.device
        )
        logits = logits + hidden[..., :1]
        return SimpleNamespace(logits=logits)


def test_teacher_forced_scores_are_deterministic_and_zero_vector_matches_baseline():
    tokenizer = FakeTokenizer()
    prompt, rendered = causal_prompt(tokenizer)
    model = UniformScoringModel()

    baseline = score_answerability_candidates(model, tokenizer, prompt, rendered)
    repeated = score_answerability_candidates(model, tokenizer, prompt, rendered)
    steered = score_answerability_candidates(
        model,
        tokenizer,
        prompt,
        rendered,
        layer_id=0,
        vector=torch.zeros(4, dtype=torch.float64),
    )

    expected_correct = -6 * math.log(model.vocab_size)
    expected_unknown = -8 * math.log(model.vocab_size)
    assert baseline == repeated
    assert baseline.correct.raw_log_probability == pytest.approx(expected_correct)
    assert baseline.unknown.raw_log_probability == pytest.approx(expected_unknown)
    assert baseline.raw_margin == pytest.approx(2 * math.log(model.vocab_size))
    assert baseline.length_normalized_margin == pytest.approx(0.0)
    assert steered.correct.raw_log_probability == pytest.approx(
        baseline.correct.raw_log_probability
    )
    assert steered.unknown.raw_log_probability == pytest.approx(
        baseline.unknown.raw_log_probability
    )
    assert steered.correct.audit.hook_call_count == 1
    assert steered.unknown.audit.hook_call_count == 1
    assert steered.correct.audit.modified_site_count == 1
    assert steered.correct.audit.prompt_token_ids_sha256 == (
        steered.correct.audit.model_prefix_token_ids_sha256
    )
    assert steered.correct.terminal_token_policy == "append_eos"
    assert steered.unknown.terminal_token_policy == "append_eos"


class GreedyModel(GemmaShapedModel):
    def __init__(self):
        super().__init__(FakeDecoderLayer())
        self.hidden_size = 4
        self.vocab_size = 96

    def forward(
        self,
        *,
        input_ids,
        use_cache,
        attention_mask=None,
        past_key_values=None,
        **_kwargs,
    ):
        assert use_cache is True
        hidden = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], self.hidden_size, device=input_ids.device
        )
        hidden = self.model.layers[0](hidden)
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        next_token = 90 if past_key_values is None else 4
        logits[:, -1, next_token] = 100.0 + hidden[:, -1, 0]
        return SimpleNamespace(
            logits=logits,
            past_key_values=1 if past_key_values is None else past_key_values + 1,
        )


def test_generation_is_deterministic_and_intervenes_only_during_prefill():
    tokenizer = FakeTokenizer()
    prompt, rendered = causal_prompt(tokenizer)
    model = GreedyModel()

    first = generate_causal_completion(
        model,
        tokenizer,
        prompt,
        rendered,
        layer_id=0,
        vector=torch.zeros(4, dtype=torch.float64),
        max_new_tokens=4,
    )
    second = generate_causal_completion(
        model,
        tokenizer,
        prompt,
        rendered,
        layer_id=0,
        vector=torch.zeros(4, dtype=torch.float64),
        max_new_tokens=4,
    )

    assert first.generated_text == "Z"
    assert first.generated_token_ids == (90, tokenizer.eos_token_id)
    assert first == second
    assert first.audit.hook_call_count == 1
    assert first.audit.modified_site_count == 1
    assert first.audit.hook_cleanup_verified is True
    assert model.model.layers[0].forward_calls == 4
    assert not model.model.layers[0]._forward_hooks
