import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from trajectory_extractor.rlmf_types import (
    BehavioralEvaluationRecord,
    CheckpointRecord,
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


@pytest.mark.parametrize(
    "field",
    [
        "split_seed",
        "faithfulness_tau",
        "sft_learning_rate",
        "rl_steps",
        "behavior_bootstrap_replicates",
        "judge_differential_bias_upper_limit",
    ],
)
def test_confirmatory_config_rejects_changes_to_every_frozen_contract_area(tmp_path, field):
    payload = load_config_payload("rlmf_qwen06b_confirmatory.json")
    replacements = {
        "split_seed": 1,
        "faithfulness_tau": 0.9,
        "sft_learning_rate": 0.001,
        "rl_steps": 1,
        "behavior_bootstrap_replicates": 1,
        "judge_differential_bias_upper_limit": 0.02,
    }
    payload[field] = replacements[field]

    with pytest.raises(ValueError, match="frozen confirmatory config"):
        RLMFConfig.from_json(write_config(tmp_path, payload))


def test_smoke_config_allows_only_its_explicitly_registered_values(tmp_path):
    payload = load_config_payload("rlmf_qwen06b_smoke.json")
    payload["sft_epochs"] = 2

    with pytest.raises(ValueError, match="frozen smoke config"):
        RLMFConfig.from_json(write_config(tmp_path, payload))


def test_config_direct_constructor_deep_freezes_sequences_and_mappings():
    payload = load_config_payload("rlmf_qwen06b_smoke.json")
    config = RLMFConfig(**payload)
    original_hash = config.config_hash

    payload["split_counts"]["pre_sft"] = 999
    payload["arms"][1] = "tampered"
    payload["seeds"][0] = 999
    payload["lora_targets"].append("tampered")
    payload["generation"]["temperature"] = 0.1
    payload["reward_weights"]["correctness"] = 999.0
    payload["confidence_values"][0] = 0.5

    assert config.split_counts["pre_sft"] == 8
    assert config.arms == ("standard_grpo", "rlmf")
    assert config.seeds == (11,)
    assert config.lora_targets[-1] == "down_proj"
    assert config.generation["temperature"] == 0.7
    assert config.reward_weights["correctness"] == 1.0
    assert config.confidence_values[0] == 0.0
    assert config.config_hash == original_hash


def test_record_constructor_deep_freezes_mapping_inputs():
    completion_parents = {"snapshot": "c" * 64}
    decision_parents = {"completion": "d" * 64}
    completion = RLMFCompletion(
        study_id="rlmf-qwen06b-v1",
        arm="rlmf",
        seed=11,
        split="validation",
        example_id="popqa-001",
        candidate_id="candidate-0",
        raw_output="answer",
        parsed=ParsedRLMFOutput(answer="answer", confidence=0.8, valid_format=True),
        checkpoint_hash="a" * 64,
        config_hash="b" * 64,
        parent_hashes=completion_parents,
    )
    decision = ClaimDecision(
        study_id="rlmf-qwen06b-v1",
        endpoint="behavior",
        status="not_evaluable",
        config_hash="b" * 64,
        parent_hashes=decision_parents,
        reason="validation audit incomplete",
    )

    completion_parents["snapshot"] = "e" * 64
    decision_parents["completion"] = "f" * 64

    assert completion.parent_hashes["snapshot"] == "c" * 64
    assert decision.parent_hashes["completion"] == "d" * 64


def test_behavioral_evaluation_record_is_immutable_provenanced_and_allows_malformed_designated():
    provenance = _behavioral_provenance()
    record = BehavioralEvaluationRecord(
        arm="rlmf",
        seed=11,
        example_id="popqa-001",
        designated_member_id="designated-0",
        designated_raw_output="malformed output",
        designated=ParsedRLMFOutput(answer=""),
        auxiliary_member_ids=tuple(f"aux-{index}" for index in range(20)),
        auxiliary_proxy_labels=(False,) * 20,
        correctness=None,
        provenance=provenance,
    )

    provenance["designated_bundle_hash"] = "d" * 64

    assert record.designated.valid_format is False
    assert record.valid_complete_case is False
    assert record.provenance["designated_bundle_hash"] == "a" * 64
    assert len(record.auxiliary_member_ids) == 20
    with pytest.raises(FrozenInstanceError):
        record.correctness = True


def test_behavioral_evaluation_record_requires_designated_plus_twenty_distinct_auxiliaries():
    fields = {
        "arm": "rlmf",
        "seed": 11,
        "example_id": "popqa-001",
        "designated_member_id": "designated-0",
        "designated_raw_output": "<sentence>A</sentence><confidence>0.8</confidence>",
        "designated": ParsedRLMFOutput(answer="A", confidence=0.8, valid_format=True),
        "auxiliary_member_ids": tuple(f"aux-{index}" for index in range(20)),
        "auxiliary_proxy_labels": (True,) * 20,
        "correctness": True,
        "provenance": _behavioral_provenance(),
    }

    with pytest.raises(ValueError, match="exactly 20"):
        BehavioralEvaluationRecord(
            **{
                **fields,
                "auxiliary_member_ids": fields["auxiliary_member_ids"][:-1],
            }
        )


def test_behavioral_evaluation_record_requires_exact_hash_provenance_and_raw_parse_binding():
    fields = {
        "arm": "rlmf",
        "seed": 11,
        "example_id": "popqa-001",
        "designated_member_id": "designated-0",
        "designated_raw_output": "<sentence>A</sentence><confidence>0.8</confidence>",
        "designated": ParsedRLMFOutput(answer="A", confidence=0.8, valid_format=True),
        "auxiliary_member_ids": tuple(f"aux-{index}" for index in range(20)),
        "auxiliary_proxy_labels": (True,) * 20,
        "correctness": True,
        "provenance": _behavioral_provenance(),
    }

    with pytest.raises(ValueError, match="provenance schema"):
        BehavioralEvaluationRecord(
            **{**fields, "provenance": {"bundle_hash": "a" * 64}}
        )
    with pytest.raises(ValueError, match="SHA-256"):
        BehavioralEvaluationRecord(
            **{
                **fields,
                "provenance": {**fields["provenance"], "config_hash": "not-a-hash"},
            }
        )
    with pytest.raises(ValueError, match="reparse"):
        BehavioralEvaluationRecord(
            **{**fields, "designated_raw_output": "malformed output"}
        )


def _behavioral_provenance():
    return {
        "designated_bundle_hash": "a" * 64,
        "auxiliary_bundle_hash": "b" * 64,
        "alias_evidence_hash": "c" * 64,
        "judge_evidence_hash": "d" * 64,
        "config_hash": "e" * 64,
    }
    with pytest.raises(ValueError, match="distinct"):
        BehavioralEvaluationRecord(
            **{
                **fields,
                "auxiliary_member_ids": ("aux-0",) * 20,
            }
        )


def _set_nested(payload, path, value):
    target = payload
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value


@pytest.mark.parametrize(
    "path",
    [
        ("schema_version",),
        ("split_seed",),
        ("split_counts", "pre_sft"),
        ("max_prompt_tokens",),
        ("max_completion_tokens",),
        ("rollout_group_size",),
        ("evaluation_auxiliary_samples",),
        ("metacognition_queries_per_completion",),
        ("faithfulness_tau",),
        ("sft_auxiliary_samples",),
        ("sft_epochs",),
        ("sft_learning_rate",),
        ("sft_weight_decay",),
        ("sft_global_batch_size",),
        ("rl_steps",),
        ("save_steps",),
        ("learning_rate",),
        ("per_device_train_batch_size",),
        ("gradient_accumulation_steps",),
        ("generation_batch_size",),
        ("num_generations",),
        ("lora_rank",),
        ("lora_alpha",),
        ("lora_dropout",),
        ("generation", "temperature"),
        ("generation", "top_p"),
        ("generation", "top_k"),
        ("generation", "min_p"),
        ("generation", "repetition_penalty"),
        ("reward_weights", "soft_format"),
        ("confidence_values", 0),
        ("behavior_bootstrap_replicates",),
        ("mechanism_bootstrap_replicates",),
        ("judge_differential_bias_upper_limit",),
    ],
)
def test_config_rejects_booleans_for_all_numeric_fields(tmp_path, path):
    payload = load_config_payload("rlmf_qwen06b_confirmatory.json")
    _set_nested(payload, path, True)

    with pytest.raises(ValueError, match="numeric"):
        RLMFConfig.from_json(write_config(tmp_path, payload))


def test_records_reject_boolean_numeric_values():
    with pytest.raises(ValueError, match="confidence"):
        ParsedRLMFOutput(answer="answer", confidence=True, valid_format=True)
    with pytest.raises(ValueError, match="seed"):
        RLMFCompletion(
            study_id="rlmf-qwen06b-v1",
            arm="rlmf",
            seed=True,
            split="validation",
            example_id="popqa-001",
            candidate_id="candidate-0",
            raw_output="answer",
            parsed=ParsedRLMFOutput(answer="answer", confidence=0.8, valid_format=True),
            checkpoint_hash="a" * 64,
            config_hash="b" * 64,
            parent_hashes={},
        )


def test_checkpoint_record_is_immutable_canonical_and_self_hashing():
    record = CheckpointRecord.create(
        study_id="study", stage="rl", arm="rlmf", seed=11,
        global_step=25, micro_step=100, sampler_cursor=100,
        files={"optimizer.pt": "a" * 64, "trainer_state.json": "b" * 64},
        parent_hashes={"pre_sft": "c" * 64}, path="/tmp/checkpoint-25",
        completed=False,
    )

    assert CheckpointRecord.from_record(record.to_record()) == record
    assert len(record.checkpoint_hash) == 64
    with pytest.raises(TypeError):
        record.files["new"] = "d" * 64
    with pytest.raises(FrozenInstanceError):
        record.global_step = 26


def test_checkpoint_record_rejects_invalid_arm_stage_and_self_hash():
    record = CheckpointRecord.create(
        study_id="study", stage="pre_sft", arm=None, seed=None,
        global_step=5, micro_step=40, sampler_cursor=40,
        files={"trainer_state.json": "a" * 64}, parent_hashes={},
        path="/tmp/checkpoint-5", completed=True,
    )
    payload = record.to_record()
    payload["checkpoint_hash"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint_hash"):
        CheckpointRecord.from_record(payload)
    with pytest.raises(ValueError, match="pre_sft"):
        CheckpointRecord.create(
            study_id="study", stage="pre_sft", arm="rlmf", seed=11,
            global_step=1, micro_step=1, sampler_cursor=1,
            files={"trainer_state.json": "a" * 64}, parent_hashes={},
            path="/tmp/checkpoint", completed=False,
        )
