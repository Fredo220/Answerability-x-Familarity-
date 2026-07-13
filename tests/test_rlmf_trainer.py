import hashlib
import importlib.util
import json
import sys

import pytest
import torch

from trajectory_extractor.rlmf_types import ParsedRLMFOutput, RLMFCompletion
from trajectory_extractor.rlmf_trainer import (
    PairedRLMFTrainer,
    answer_prompt,
    derive_generation_seed,
    generate_group,
    metacognition_prompt,
    query_metacognitive_score,
    validate_installed_trl,
)


def test_import_does_not_load_colab_only_packages():
    assert "trl" not in sys.modules
    assert "peft" not in sys.modules
    assert "bitsandbytes" not in sys.modules


def test_registered_seed_derivation_is_arm_independent_and_query_specific():
    payload = ["study", 11, 25, "example-7", 3, "answer"]
    expected = int.from_bytes(
        hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).digest()[:8],
        "big",
    )

    assert derive_generation_seed("study", 11, 25, "example-7", 3, "answer") == expected
    assert derive_generation_seed("study", 11, 25, "example-7", 3, "answer") != (
        derive_generation_seed("study", 11, 25, "example-7", 3, "metacognition")
    )
    with pytest.raises(TypeError, match="arm"):
        derive_generation_seed(
            "study", 11, 25, "example-7", 3, "answer", arm="rlmf"
        )


class FakeTextModel:
    def __init__(self):
        self.calls = []

    def generate_text(self, prompt, *, seed, generation):
        self.calls.append((prompt, seed, dict(generation)))
        if "<metascore>" in prompt:
            return "<metascore>0.8</metascore>"
        return f"<sentence>answer-{seed}</sentence><confidence>0.7</confidence>"


def test_answer_and_metacognition_generation_use_disjoint_exact_schemas():
    model = FakeTextModel()
    question = "Who wrote the source novel?"
    completions = generate_group(
        model,
        object(),
        question,
        group_size=2,
        seed=11,
        study_id="study",
        arm="standard_grpo",
        step=4,
        example_id="example-1",
    )

    assert len(completions) == 2
    assert all(item.source_question == question for item in completions)
    assert all(item.parsed.valid_format for item in completions)
    assert all(item.parsed.metascore is None for item in completions)
    assert all("<metascore>" not in item.raw_output for item in completions)
    assert "<sentence>" in answer_prompt(question)
    assert "<confidence>" in answer_prompt(question)
    assert "<metascore>" not in answer_prompt(question)

    parsed = query_metacognitive_score(model, object(), completions[0], seed=99)
    query = model.calls[-1][0]
    assert parsed.valid_format
    assert parsed.metascore == 0.8
    assert question in query
    assert completions[0].parsed.answer in query
    assert "<metascore>" in metacognition_prompt(
        question, completions[0].parsed.answer
    )
    assert "<sentence>" not in query
    assert "<confidence>" not in query


def test_malformed_answer_still_receives_one_metacognition_query():
    model = FakeTextModel()
    completion = RLMFCompletion(
        study_id="study", arm="rlmf", seed=11, split="rl_train",
        example_id="example-1", candidate_id="candidate-1",
        raw_output="I am not sure", parsed=ParsedRLMFOutput(answer=""),
        checkpoint_hash="0" * 64, config_hash="0" * 64, parent_hashes={},
        source_question="Who wrote the source novel?",
    )

    parsed = query_metacognitive_score(model, object(), completion, seed=101)

    assert parsed.valid_format
    assert "I am not sure" in model.calls[-1][0]


class FakeGRPOTrainer:
    def __init__(self, *, reward_matrix, **kwargs):
        self.reward_matrix = reward_matrix
        self.reward_weights = torch.tensor([1.0, 1.0])
        self.reward_func_names = ["other", "faithful_calibration"]
        self.num_generations = 2
        self.accelerator = type("Accelerator", (), {"process_index": 0})()
        self.kwargs = kwargs

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        return self.reward_matrix.clone()

    def _generate_and_score_completions(self, inputs):
        completions = [item["completion"] for item in inputs]
        rewards = self._calculate_rewards(inputs, [], completions, [[], []])
        return {"advantages": torch.zeros(len(inputs)), "base_rewards": rewards}


def test_both_arms_share_generation_reward_and_metacognition_path():
    rewards = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    inputs = [
        {"completion": "a", "metacognitive_reward": 0.2},
        {"completion": "b", "metacognitive_reward": 0.8},
    ]
    observed = []

    def score_metacognition(rows, completions):
        observed.append((tuple(row["completion"] for row in rows), tuple(completions)))
        return torch.tensor([row["metacognitive_reward"] for row in rows])

    standard = PairedRLMFTrainer(
        advantage_form="standard",
        reward_matrix=rewards,
        metacognition_scorer=score_metacognition,
        _base_trainer_cls=FakeGRPOTrainer,
    )
    rlmf = PairedRLMFTrainer(
        advantage_form="mf",
        reward_matrix=rewards,
        metacognition_scorer=score_metacognition,
        _base_trainer_cls=FakeGRPOTrainer,
    )

    standard_output = standard._generate_and_score_completions(inputs)
    rlmf_output = rlmf._generate_and_score_completions(inputs)

    assert type(standard) is type(rlmf)
    assert observed == [(('a', 'b'), ('a', 'b')), (('a', 'b'), ('a', 'b'))]
    assert torch.equal(standard.last_reward_matrix, rlmf.last_reward_matrix)
    assert torch.equal(standard.last_metacognitive_rewards, rlmf.last_metacognitive_rewards)
    assert torch.equal(standard_output["base_rewards"], rlmf_output["base_rewards"])
    assert not torch.equal(standard_output["advantages"], rlmf_output["advantages"])


@pytest.mark.skipif(
    importlib.util.find_spec("trl") is None,
    reason="local CPU environment intentionally lacks pinned trl==0.23.0 runtime",
)
def test_installed_trl_private_api_matches_vendored_manifest():
    validate_installed_trl()
