import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import pytest
import torch
from packaging.requirements import Requirement

from trajectory_extractor.rlmf_types import ParsedRLMFOutput, RLMFCompletion
from trajectory_extractor.rlmf_artifacts import RLMFArtifactStore
import trajectory_extractor.rlmf_trainer as rlmf_trainer
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
        token_rows = torch.arange(len(inputs), dtype=torch.long).unsqueeze(1)
        return {
            "prompt_ids": token_rows,
            "prompt_mask": torch.ones_like(token_rows),
            "completion_ids": token_rows + 10,
            "completion_mask": torch.ones_like(token_rows),
            "advantages": torch.zeros(len(inputs)),
            "num_items_in_batch": torch.tensor(len(inputs)),
            "base_rewards": rewards,
        }


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
    assert set(standard_output) == {
        "prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
        "advantages", "num_items_in_batch", "base_rewards",
    }


class Transformers457Tokenizer:
    def __call__(self, text, *, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return str(int(token_ids[0]))


class Transformers457GenerateModel:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        assert "generator" not in kwargs
        self.calls.append((torch.initial_seed(), tuple(kwargs["input_ids"].shape)))
        sampled = torch.randint(3, 1000, (1, 1))
        return torch.cat((kwargs["input_ids"], sampled), dim=1)


def test_transformers_457_generation_is_rng_isolated_without_generator_keyword():
    model = Transformers457GenerateModel()
    before = torch.random.get_rng_state().clone()

    first = generate_group(
        model, Transformers457Tokenizer(), "Question?", group_size=2, seed=11,
        study_id="study", step=3, example_id="example-1",
    )

    assert torch.equal(torch.random.get_rng_state(), before)
    assert [seed for seed, _ in model.calls] == [
        derive_generation_seed("study", 11, 3, "example-1", 0, "answer"),
        derive_generation_seed("study", 11, 3, "example-1", 1, "answer"),
    ]
    model.calls.clear()
    second = generate_group(
        model, Transformers457Tokenizer(), "Question?", group_size=2, seed=11,
        study_id="study", step=3, example_id="example-1",
    )
    assert [item.raw_output for item in first] == [item.raw_output for item in second]


class SeedControlledBatchModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append((torch.initial_seed(), kwargs["input_ids"].shape[0]))
        sampled = torch.randint(3, 1000, (kwargs["input_ids"].shape[0], 1))
        return torch.cat((kwargs["input_ids"], sampled), dim=1)


class FakeSeededGRPOTrainer(FakeGRPOTrainer):
    def __init__(self, *, model, reward_matrix, **kwargs):
        super().__init__(reward_matrix=reward_matrix, **kwargs)
        self.model_wrapped = model
        self.processing_class = Transformers457Tokenizer()
        self.state = type("State", (), {"global_step": 4})()

    def _generate_and_score_completions(self, inputs):
        encoded = torch.tensor([[1, 2], [1, 2]])
        generated = self.model_wrapped.generate(
            input_ids=encoded, attention_mask=torch.ones_like(encoded)
        )
        completions = [str(int(row[-1])) for row in generated]
        rewards = self._calculate_rewards(inputs, [], completions, [[int(row[-1])] for row in generated])
        token_rows = torch.arange(len(inputs), dtype=torch.long).unsqueeze(1)
        return {
            "prompt_ids": token_rows,
            "prompt_mask": torch.ones_like(token_rows),
            "completion_ids": generated[:, -1:],
            "completion_mask": torch.ones_like(token_rows),
            "advantages": torch.zeros(len(inputs)),
            "num_items_in_batch": torch.tensor(len(inputs)),
            "base_rewards": rewards,
        }


def test_trainer_answer_generations_are_controlled_by_registered_candidate_seeds():
    rewards = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    inputs = [
        {"example_id": "example-1", "completion": "a", "metacognitive_reward": 0.2},
        {"example_id": "example-1", "completion": "b", "metacognitive_reward": 0.8},
    ]
    expected_seeds = [
        derive_generation_seed("study", 11, 4, "example-1", member, "answer")
        for member in range(2)
    ]
    outputs = []
    for advantage_form in ("standard", "mf"):
        model = SeedControlledBatchModel()
        trainer = PairedRLMFTrainer(
            advantage_form=advantage_form,
            reward_matrix=rewards,
            metacognition_scorer=lambda rows, _: [row["metacognitive_reward"] for row in rows],
            study_id="study",
            generation_seed=11,
            model=model,
            _base_trainer_cls=FakeSeededGRPOTrainer,
        )
        before = torch.random.get_rng_state().clone()
        outputs.append(trainer._generate_and_score_completions(inputs))
        assert model.calls == [(expected_seeds[0], 1), (expected_seeds[1], 1)]
        assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(outputs[0]["completion_ids"], outputs[1]["completion_ids"])


def test_raw_metacognition_is_durable_before_parser_failure(tmp_path, monkeypatch):
    store = RLMFArtifactStore(tmp_path)
    completion = RLMFCompletion(
        study_id="study", arm="rlmf", seed=11, split="rl_train",
        example_id="example-1", candidate_id="candidate-1",
        raw_output="<sentence>A</sentence><confidence>0.7</confidence>",
        parsed=ParsedRLMFOutput(answer="A", confidence=0.7, valid_format=True),
        checkpoint_hash="0" * 64, config_hash="0" * 64, parent_hashes={},
        source_question="Question?",
    )

    def persist(raw):
        store.append_jsonl("study", "training_audit", "raw", {"raw": raw})

    monkeypatch.setattr(
        rlmf_trainer, "parse_metascore_output",
        lambda _: (_ for _ in ()).throw(RuntimeError("parser crashed")),
    )
    with pytest.raises(RuntimeError, match="parser crashed"):
        query_metacognitive_score(
            FakeTextModel(), object(), completion, seed=99, raw_recorder=persist
        )

    path = Path(tmp_path, "runs", "rlmf", "study", "training_audit", "raw.jsonl")
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"raw": "<metascore>0.8</metascore>"}
    ]


def test_raw_answer_is_durable_before_parser_failure(tmp_path, monkeypatch):
    store = RLMFArtifactStore(tmp_path)

    def persist(record):
        store.append_jsonl("study", "training_audit", "raw", record)

    monkeypatch.setattr(
        rlmf_trainer, "parse_rlmf_output",
        lambda _: (_ for _ in ()).throw(RuntimeError("parser crashed")),
    )
    with pytest.raises(RuntimeError, match="parser crashed"):
        generate_group(
            FakeTextModel(), object(), "Question?", group_size=1, seed=11,
            study_id="study", step=4, example_id="example-1",
            raw_recorder=persist,
        )

    generation_seed = derive_generation_seed(
        "study", 11, 4, "example-1", 0, "answer"
    )
    path = Path(tmp_path, "runs", "rlmf", "study", "training_audit", "raw.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [{
        "candidate_id": "example-1-member-0",
        "example_id": "example-1",
        "generation_seed": generation_seed,
        "group_member": 0,
        "kind": "answer",
        "raw_output": (
            f"<sentence>answer-{generation_seed}</sentence>"
            "<confidence>0.7</confidence>"
        ),
        "step": 4,
    }]


def test_pre_advantage_snapshot_persists_candidate_bound_common_arrays():
    rewards = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    inputs = [
        {"example_id": "example-1", "completion": "a", "metacognitive_reward": 0.2},
        {"example_id": "example-1", "completion": "b", "metacognitive_reward": 0.8},
    ]
    snapshots = []
    for advantage_form in ("standard", "mf"):
        trainer = PairedRLMFTrainer(
            advantage_form=advantage_form,
            reward_matrix=rewards,
            metacognition_scorer=lambda rows, _: [row["metacognitive_reward"] for row in rows],
            pre_advantage_recorder=snapshots.append,
            _base_trainer_cls=FakeGRPOTrainer,
        )
        trainer.state = type("State", (), {"global_step": 0})()
        trainer._generate_and_score_completions(inputs)

    assert [snapshot["candidate_ids"] for snapshot in snapshots] == [
        ["example-1-step-0-member-0", "example-1-step-0-member-1"],
        ["example-1-step-0-member-0", "example-1-step-0-member-1"],
    ]
    assert snapshots[0]["reward_matrix"] == snapshots[1]["reward_matrix"]
    assert snapshots[0]["metacognitive_rewards"] == snapshots[1]["metacognitive_rewards"]


def _pinned_runtime_unavailable_reason():
    requirements = Path(__file__).resolve().parents[1] / "requirements-rlmf-colab.txt"
    mismatches = []
    for raw in requirements.read_text().splitlines():
        requirement = Requirement(raw)
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"missing {requirement.name}")
            continue
        if installed not in requirement.specifier:
            mismatches.append(
                f"{requirement.name}=={installed} does not satisfy {requirement.specifier}"
            )
    if mismatches:
        return "pinned Colab runtime unavailable: " + "; ".join(mismatches)
    return None


_PINNED_RUNTIME_UNAVAILABLE = _pinned_runtime_unavailable_reason()


@pytest.mark.skipif(
    _PINNED_RUNTIME_UNAVAILABLE is not None,
    reason=_PINNED_RUNTIME_UNAVAILABLE or "pinned runtime available",
)
def test_installed_trl_private_api_matches_vendored_manifest():
    validate_installed_trl()


@pytest.mark.skipif(
    _PINNED_RUNTIME_UNAVAILABLE is not None,
    reason=_PINNED_RUNTIME_UNAVAILABLE or "pinned runtime available",
)
def test_real_pinned_trl_tiny_model_both_arms_seal_resume_and_preserve_contract(
    tmp_path,
):
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, TaskType
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig

    from trajectory_extractor.rlmf_training import (
        _ensure_custom_restart_state,
        _restore_custom_restart_state,
        seal_checkpoint,
        validate_runtime_versions,
    )
    from trajectory_extractor.rlmf_types import RLMFConfig

    validate_runtime_versions()
    validate_installed_trl()
    config = RLMFConfig.from_json(
        Path(__file__).resolve().parents[1] / "configs" / "rlmf_qwen06b_smoke.json"
    )
    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "Question": 3,
        "tiny": 4,
        "answer": 5,
        "certain": 6,
        "uncertain": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.padding_side = "left"
    dataset = Dataset.from_list([
        {"prompt": "Question tiny", "example_id": "tiny-example"}
    ])

    def new_base_model():
        torch.manual_seed(314159)
        model_config = GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=16,
            n_ctx=16,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model_config._name_or_path = "tiny-local-grpo"
        return GPT2LMHeadModel(model_config)

    def other_reward(completions, **_):
        return [1.0 if index == 0 else 0.0 for index in range(len(completions))]

    def faithful_calibration(completions, **_):
        return [0.0 if index == 0 else 2.0 for index in range(len(completions))]

    def make_trainer(arm, output_dir, max_steps, adapter_path=None):
        model = new_base_model()
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["c_attn"],
            fan_in_fan_out=True,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(
                model, str(adapter_path), is_trainable=True
            )
            peft_config = None
        args = GRPOConfig(
            output_dir=str(output_dir),
            use_cpu=True,
            max_steps=max_steps,
            save_strategy="steps",
            save_steps=1,
            logging_strategy="no",
            report_to="none",
            disable_tqdm=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            generation_batch_size=2,
            num_generations=2,
            max_prompt_length=8,
            max_completion_length=2,
            beta=0.0,
            reward_weights=[1.0, 1.0],
            scale_rewards="none",
            fp16=False,
            bf16=False,
            use_vllm=False,
            seed=11,
            data_seed=11,
        )
        raw_records = []
        trainer = PairedRLMFTrainer(
            model=model,
            reward_funcs=[other_reward, faithful_calibration],
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            advantage_form="standard" if arm == "standard_grpo" else "mf",
            metacognition_scorer=lambda rows, _: [
                0.2 if index == 0 else 0.8 for index in range(len(rows))
            ],
            study_id=config.study_id,
            generation_seed=11,
            raw_answer_recorder=raw_records.append,
        )
        return trainer, raw_records

    direct_inputs = [dataset[0], dataset[0]]
    contract_keys = {
        "prompt_ids",
        "prompt_mask",
        "completion_ids",
        "completion_mask",
        "advantages",
        "num_items_in_batch",
    }
    store = RLMFArtifactStore(tmp_path / "artifacts")
    initial_outputs = {}
    for arm in ("standard_grpo", "rlmf"):
        trainer, raw_records = make_trainer(
            arm, tmp_path / f"train-{arm}", max_steps=1
        )
        output = trainer._generate_and_score_completions(direct_inputs)
        assert set(output) == contract_keys
        assert output["completion_ids"].shape[0] == 2
        assert len(raw_records) == 2
        initial_outputs[arm] = {
            key: value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in output.items()
        }

        trainer.train()
        assert trainer.state.global_step == 1
        source = Path(trainer.args.output_dir) / "checkpoint-1"
        _ensure_custom_restart_state(
            source,
            trainer,
            global_step=1,
            micro_step=int(trainer._step),
            sampler_cursor=1,
        )
        record = seal_checkpoint(
            store,
            config,
            source,
            stage="rl",
            arm=arm,
            seed=11,
            global_step=1,
            micro_step=int(trainer._step),
            sampler_cursor=1,
            parent_hashes={"pre_sft": "a" * 64},
            completed=False,
        )
        assert record.completed is False

        resumed, _ = make_trainer(
            arm,
            tmp_path / f"resume-{arm}",
            max_steps=2,
            adapter_path=Path(record.path),
        )
        _restore_custom_restart_state(record, resumed)
        resumed.train(resume_from_checkpoint=record.path)
        assert resumed.state.global_step == 2

    for key in contract_keys - {"advantages"}:
        assert torch.equal(
            initial_outputs["standard_grpo"][key], initial_outputs["rlmf"][key]
        )
    assert not torch.equal(
        initial_outputs["standard_grpo"]["advantages"],
        initial_outputs["rlmf"]["advantages"],
    )
