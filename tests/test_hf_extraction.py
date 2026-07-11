from types import SimpleNamespace

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from trajectory_extractor.extraction import (
    _prefix_replay_pre_token_states,
    _single_pass_pre_token_states,
    generate_and_extract,
)
from trajectory_extractor.types import ExperimentConfig


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, text, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "answer"


class FakeModel:
    device = torch.device("cpu")
    config = SimpleNamespace(hidden_size=4, num_hidden_layers=2, _name_or_path="fake/model")

    def eval(self):
        return self

    def generate(self, **kwargs):
        sequences = torch.tensor([[1, 2, 3, 4, 5]])
        score_a = torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
        score_b = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 3.0]])
        return SimpleNamespace(sequences=sequences, scores=(score_a, score_b))

    def __call__(self, input_ids, attention_mask, **kwargs):
        self.forward_calls = getattr(self, "forward_calls", 0) + 1
        seq_len = input_ids.shape[1]
        states = []
        for layer in range(3):
            value = torch.arange(seq_len * 4, dtype=torch.float32).reshape(1, seq_len, 4)
            states.append(value + layer * 100)
        return SimpleNamespace(hidden_states=tuple(states))


def test_generate_and_extract_replays_full_answer_trajectory():
    config = ExperimentConfig(model_id="fake/model", max_new_tokens=2, seed=11)

    run = generate_and_extract(
        FakeModel(),
        FakeTokenizer(),
        "prompt",
        config=config,
        run_id="run",
        example_id="example",
        track="concept_mixing",
        split="test",
        label=1,
    )

    assert run.response == "answer"
    assert run.input_token_count == 3
    assert run.response_token_ids.tolist() == [4, 5]
    assert run.hidden_states.shape == (2, 3, 4)
    assert run.hidden_states.dtype == np.float16
    assert run.token_logprobs.shape == (2,)
    assert run.token_entropies.shape == (2,)
    assert run.provenance["used_chat_template"] is False
    assert run.provenance["activation_alignment"] == "pre_response_token"
    assert run.provenance["activation_replay"] == "single_teacher_forced"
    assert run.provenance["response_token_end"] - run.provenance["response_token_start"] == 2
    np.testing.assert_array_equal(run.hidden_states[0, 0], [8, 9, 10, 11])


def test_single_pass_replay_matches_incremental_prefix_replay():
    sequences = torch.tensor([[1, 2, 3, 4, 5]])
    single = _single_pass_pre_token_states(
        FakeModel(), sequences, input_token_count=3, response_token_count=2
    )
    incremental = _prefix_replay_pre_token_states(
        FakeModel(), sequences, input_token_count=3, response_token_count=2
    )
    torch.testing.assert_close(single, incremental)


def test_single_pass_replay_uses_one_forward_call():
    model = FakeModel()
    _single_pass_pre_token_states(
        model,
        torch.tensor([[1, 2, 3, 4, 5]]),
        input_token_count=3,
        response_token_count=2,
    )
    assert model.forward_calls == 1


class CausalFakeModel(FakeModel):
    def __call__(self, input_ids, attention_mask, **kwargs):
        cumulative = input_ids.float().cumsum(dim=1)[..., None]
        offsets = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
        states = tuple(cumulative + offsets + layer * 100 for layer in range(3))
        return SimpleNamespace(hidden_states=states)


def test_single_pass_pre_token_states_do_not_depend_on_future_tokens():
    original = torch.tensor([[1, 2, 3, 4, 5]])
    future_changed = torch.tensor([[1, 2, 3, 4, 99]])
    original_states = _single_pass_pre_token_states(
        CausalFakeModel(), original, input_token_count=3, response_token_count=2
    )
    changed_states = _single_pass_pre_token_states(
        CausalFakeModel(), future_changed, input_token_count=3, response_token_count=2
    )
    torch.testing.assert_close(original_states, changed_states)


def test_single_pass_replay_matches_prefixes_on_local_tiny_llama():
    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    ).eval()
    sequences = torch.tensor([[1, 2, 3, 4, 5]])
    single = _single_pass_pre_token_states(
        model, sequences, input_token_count=3, response_token_count=2
    )
    incremental = _prefix_replay_pre_token_states(
        model, sequences, input_token_count=3, response_token_count=2
    )
    assert single.shape == (2, 3, 8)
    torch.testing.assert_close(single, incremental, rtol=1e-5, atol=1e-6)


def test_generate_and_extract_is_deterministic_for_fake_model():
    config = ExperimentConfig(model_id="fake/model", max_new_tokens=2, seed=11)
    kwargs = dict(
        config=config,
        run_id="run",
        example_id="example",
        track="concept_mixing",
        split="test",
        label=1,
    )
    first = generate_and_extract(FakeModel(), FakeTokenizer(), "prompt", **kwargs)
    second = generate_and_extract(FakeModel(), FakeTokenizer(), "prompt", **kwargs)
    np.testing.assert_array_equal(first.hidden_states, second.hidden_states)
    np.testing.assert_array_equal(first.response_token_ids, second.response_token_ids)
