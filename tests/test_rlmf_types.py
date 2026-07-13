import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from trajectory_extractor.rlmf_types import (
    ClaimDecision,
    ParsedRLMFOutput,
    PopQAExample,
    RLMFCompletion,
    RLMFConfig,
)


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load_config_payload(name):
    return json.loads((CONFIGS / name).read_text())


def write_config(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def test_frozen_smoke_and_confirmatory_configs_parse():
    smoke = RLMFConfig.from_json(CONFIGS / "rlmf_qwen06b_smoke.json")
    confirmatory = RLMFConfig.from_json(CONFIGS / "rlmf_qwen06b_confirmatory.json")

    assert smoke.profile == "smoke"
    assert smoke.seeds == (11,)
    assert smoke.gradient_accumulation_steps == 2
    assert smoke.generation_batch_size == 2
    assert confirmatory.profile == "confirmatory"
    assert confirmatory.seeds == (11, 22, 33)
    assert confirmatory.gradient_accumulation_steps == 4
    assert confirmatory.generation_batch_size == 4
    assert confirmatory.model_revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert confirmatory.dataset_revision == "5cf59972d88d4aaaa7781ac91b83d053563d8268"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value.update(arms=["standard_grpo", "experimental"]), "arms"),
        (lambda value: value.update(seeds=[11, 11]), "seeds"),
        (lambda value: value.update(model_revision="main"), "model_revision"),
        (lambda value: value.update(dataset_revision="main"), "dataset_revision"),
        (lambda value: value.update(rollout_group_size=1), "rollout_group_size"),
        (
            lambda value: value.update(evaluation_auxiliary_samples=19),
            "evaluation_auxiliary_samples",
        ),
        (
            lambda value: value.update(metacognition_queries_per_completion=2),
            "metacognition_queries_per_completion",
        ),
        (lambda value: value.update(generation_batch_size=3), "generation_batch_size"),
    ],
)
def test_config_rejects_unregistered_or_incompatible_values(tmp_path, change, message):
    payload = load_config_payload("rlmf_qwen06b_confirmatory.json")
    change(payload)

    with pytest.raises(ValueError, match=message):
        RLMFConfig.from_json(write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("profile", "accumulation", "generation"),
    [("smoke", 4, 2), ("confirmatory", 2, 4)],
)
def test_profiles_reject_wrong_registered_batch_values(
    tmp_path, profile, accumulation, generation
):
    payload = load_config_payload(f"rlmf_qwen06b_{profile}.json")
    payload["gradient_accumulation_steps"] = accumulation
    payload["generation_batch_size"] = generation

    with pytest.raises(ValueError, match="batch"):
        RLMFConfig.from_json(write_config(tmp_path, payload))


def test_immutable_records_validate_and_parse_completion():
    example = PopQAExample(
        example_id="popqa-001",
        subject="Q42",
        question="Who wrote Hamlet?",
        answers=("William Shakespeare", "Shakespeare"),
    )
    parsed = ParsedRLMFOutput(
        answer="William Shakespeare",
        confidence=0.8,
        metascore=0.9,
        valid_format=True,
    )
    completion = RLMFCompletion.parse_record(
        {
            "study_id": "rlmf-qwen06b-v1",
            "arm": "rlmf",
            "seed": 11,
            "split": "validation",
            "example_id": example.example_id,
            "candidate_id": "candidate-0",
            "raw_output": "<sentence>William Shakespeare</sentence>",
            "parsed": {
                "answer": parsed.answer,
                "confidence": parsed.confidence,
                "metascore": parsed.metascore,
                "valid_format": parsed.valid_format,
            },
            "checkpoint_hash": "a" * 64,
            "config_hash": "b" * 64,
            "parent_hashes": {"snapshot": "c" * 64},
        }
    )
    decision = ClaimDecision(
        study_id=completion.study_id,
        endpoint="behavior",
        status="not_evaluable",
        config_hash=completion.config_hash,
        parent_hashes={"completion": "d" * 64},
        reason="validation audit incomplete",
    )

    assert completion.parsed == parsed
    assert decision.status == "not_evaluable"
    with pytest.raises(FrozenInstanceError):
        example.question = "changed"
    with pytest.raises(ValueError, match="parent_hashes"):
        RLMFCompletion.parse_record({**completion.to_record(), "parent_hashes": {"bad": "main"}})


def test_records_reject_out_of_range_values():
    with pytest.raises(ValueError, match="confidence"):
        ParsedRLMFOutput(answer="answer", confidence=1.1, valid_format=True)
    with pytest.raises(ValueError, match="answers"):
        PopQAExample(example_id="x", subject="s", question="q", answers=())
    with pytest.raises(ValueError, match="status"):
        ClaimDecision(
            study_id="rlmf-qwen06b-v1",
            endpoint="behavior",
            status="pending",
            config_hash="a" * 64,
            parent_hashes={},
            reason="invalid",
        )


def test_popqa_aliases_are_immutable_after_direct_construction():
    example = PopQAExample(
        example_id="popqa-002",
        subject="Q43",
        question="Who painted Guernica?",
        answers=["Pablo Picasso"],
    )

    assert example.answers == ("Pablo Picasso",)
