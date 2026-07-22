from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_config import FAConfig
import trajectory_extractor.fa_runtime as runtime_module
from trajectory_extractor.fa_runtime import (
    HFModelRunner,
    load_pinned_tokenizer,
    resume_generation,
    run_generation_shard,
    validate_runner_binding,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "familiarity_answerability_qwen06b_smoke.json"
)


@dataclass(frozen=True)
class Example:
    example_id: str
    canonical_payload_sha256: str
    user_text: str
    split: str = "pilot"
    entity_unit_id: str = "unit-1"
    template_family: str = "train_registry_direct"
    target_familiarity: str = "screened_real"
    distractor_familiarity: str = "matched_synthetic"
    answerability: str = "target_bound"
    registry_code: str = "K7M2Q"
    block: str = "factorial"
    exposure: str | None = None


@dataclass(frozen=True)
class Manifest:
    config_hash: str
    manifest_sha256: str
    examples: tuple[Example, ...]
    chat_template_sha256: str | None = None
    tokenizer_pin_sha256: str | None = None


class FakeRunner:
    calls = 0
    model_id = "Qwen/Qwen3-0.6B"
    model_revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    tokenizer_revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    chat_template_sha256 = "f" * 64

    def generate(self, prompts, generation):
        type(self).calls += 1
        return ["UNKNOWN" if "not stated" in prompt else "K7M2Q" for prompt in prompts]


class FailingRunner(FakeRunner):
    def generate(self, prompts, generation):
        type(self).calls += 1
        raise RuntimeError("generation backend unavailable")


def config() -> FAConfig:
    return FAConfig.from_json(CONFIG_PATH)


def pinned_config() -> FAConfig:
    return replace(config(), chat_template_sha256="f" * 64)


def pilot_manifest(active_config: FAConfig | None = None) -> Manifest:
    active_config = active_config or pinned_config()
    payload = "a" * 64
    return Manifest(
        config_hash=active_config.config_hash,
        manifest_sha256="b" * 64,
        examples=(Example(example_id=payload, canonical_payload_sha256=payload, user_text="What is stated?"),),
        chat_template_sha256=active_config.chat_template_sha256 or None,
        tokenizer_pin_sha256="e" * 64,
    )


def store(tmp_path) -> FAArtifactStore:
    return FAArtifactStore(tmp_path)


def test_hf_runner_uses_auto_placement_and_memory_bounded_microbatches(monkeypatch):
    active_config = config()
    load_kwargs = {}

    class FakeTokenizer:
        pad_token_id = None
        eos_token = "<eos>"
        pad_token = None

        def __init__(self):
            self.calls = []
            self.decode_calls = 0

        def __call__(self, prompts, *, return_tensors, padding):
            self.calls.append((tuple(prompts), return_tensors, padding))
            return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

        def batch_decode(self, values, *, skip_special_tokens):
            assert tuple(values.shape) == (1, 1)
            assert skip_special_tokens is True
            self.decode_calls += 1
            return [f"completion-{self.decode_calls}"]

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.generate_calls = 0

        def generate(self, input_ids, **generation):
            self.generate_calls += 1
            suffix = torch.tensor([[99]], device=input_ids.device)
            return torch.cat((input_ids, suffix), dim=1)

    tokenizer = FakeTokenizer()
    model = FakeModel()

    monkeypatch.setattr(
        runtime_module,
        "load_pinned_tokenizer",
        lambda supplied: SimpleNamespace(
            tokenizer=tokenizer,
            chat_template_sha256="f" * 64,
        ),
    )
    import transformers

    def load_model(model_id, **kwargs):
        load_kwargs.update({"model_id": model_id, **kwargs})
        return model

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        load_model,
    )

    runner = HFModelRunner(active_config)
    completions = runner.generate(("one", "two", "three"), {"max_new_tokens": 1})

    assert load_kwargs["model_id"] == active_config.model_id
    assert load_kwargs["revision"] == active_config.model_revision
    assert load_kwargs["torch_dtype"] == "auto"
    assert load_kwargs["device_map"] == "auto"
    assert all(call[0] in {("one",), ("two",), ("three",)} for call in tokenizer.calls)
    assert all(call[2] is False for call in tokenizer.calls)
    assert model.generate_calls == 3
    assert completions == ["completion-1", "completion-2", "completion-3"]


def replace_shard_payload(shard, payload: bytes) -> None:
    shard.data_path.write_bytes(payload)
    sidecar = json.loads(shard.manifest_path.read_text(encoding="utf-8"))
    sidecar["sha256"] = hashlib.sha256(payload).hexdigest()
    sidecar["row_count"] = payload.count(b"\n")
    shard.manifest_path.write_text(json.dumps(sidecar), encoding="utf-8")


def test_resume_skips_only_verified_completed_shards(tmp_path):
    FakeRunner.calls = 0
    active_config = pinned_config()
    first = run_generation_shard(FakeRunner(), pilot_manifest(active_config), store(tmp_path), "0001", config=active_config)
    second = run_generation_shard(FakeRunner(), pilot_manifest(active_config), store(tmp_path), "0001", config=active_config)

    assert first == second
    assert FakeRunner.calls == 1
    assert resume_generation(store(tmp_path), active_config.run_id, "pilot") == (first,)


def test_generation_records_bind_exact_provenance_and_infrastructure_failures_are_retryable(tmp_path):
    FakeRunner.calls = 0
    active_config = pinned_config()
    manifest = pilot_manifest(active_config)
    failed = run_generation_shard(FailingRunner(), manifest, store(tmp_path), "0001", config=active_config)
    retry = run_generation_shard(FakeRunner(), manifest, store(tmp_path), "0001", config=active_config)
    resumed_retry = run_generation_shard(FakeRunner(), manifest, store(tmp_path), "0001", config=active_config)
    failed_row = json.loads(failed.data_path.read_text(encoding="utf-8"))
    retry_row = json.loads(retry.data_path.read_text(encoding="utf-8"))
    retry_manifest = json.loads(retry.manifest_path.read_text(encoding="utf-8"))

    assert failed != retry
    assert resumed_retry == retry
    assert FakeRunner.calls == 1
    assert failed_row["status"] == "infrastructure_failure"
    assert failed_row["exception_class"] == "RuntimeError"
    assert retry_row["status"] == "completed"
    assert retry_row["example_sha256"] == "a" * 64
    assert retry_row["config_sha256"] == active_config.config_hash
    assert retry_row["model_sha256"]
    assert retry_row["tokenizer_sha256"]
    assert retry_row["tokenizer_pin_sha256"] == "e" * 64
    assert retry_row["chat_template_sha256"] == "f" * 64
    assert retry_row["rendered_prompt_sha256"]
    assert retry_row["generation"] == dict(active_config.generation)
    assert retry_row["raw_output"] == "K7M2Q"
    assert retry_row["wall_time_seconds"] >= 0
    assert type(retry_row["peak_memory_bytes"]) is int
    assert retry_manifest["lineage"] == {
        "chat_template_sha256": "f" * 64,
        "config_sha256": active_config.config_hash,
        "model_sha256": retry_row["model_sha256"],
        "source_manifest_sha256": manifest.manifest_sha256,
        "tokenizer_sha256": retry_row["tokenizer_sha256"],
        "tokenizer_pin_sha256": "e" * 64,
    }
    assert retry_manifest["record_kind"] == "generation"


def test_resume_generation_never_accepts_non_generation_shards(tmp_path):
    active_config = pinned_config()
    artifact_store = store(tmp_path)
    artifact_store.write_completed_shard(
        active_config.run_id,
        "pilot",
        "metrics",
        [{"kind": "metrics", "status": "completed"}],
        {"config_sha256": active_config.config_hash},
        record_kind="metrics",
    )

    assert resume_generation(artifact_store, active_config.run_id, "pilot") == ()


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json\n",
        b"[]\n",
        b'{"kind":"metrics","status":"completed"}\n',
    ],
    ids=["non-json", "non-object", "non-generation"],
)
def test_malformed_verified_candidate_is_ineligible_and_retry_remains_auditable(
    tmp_path, payload
):
    active_config = pinned_config()
    manifest = pilot_manifest(active_config)
    artifact_store = store(tmp_path)
    candidate = run_generation_shard(
        FailingRunner(), manifest, artifact_store, "0001", config=active_config
    )
    replace_shard_payload(candidate, payload)

    verified_candidate = artifact_store.verify_shard(candidate.manifest_path)
    FakeRunner.calls = 0
    retry = run_generation_shard(
        FakeRunner(), manifest, artifact_store, "0001", config=active_config
    )

    assert verified_candidate.shard_id == "0001"
    assert retry.shard_id == "0001.retry-1"
    assert FakeRunner.calls == 1
    assert candidate.data_path.read_bytes() == payload


def test_resume_generation_keeps_store_integrity_errors_fail_closed(tmp_path):
    active_config = pinned_config()
    manifest = pilot_manifest(active_config)
    artifact_store = store(tmp_path)
    candidate = run_generation_shard(
        FailingRunner(), manifest, artifact_store, "0001", config=active_config
    )
    candidate.data_path.write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        run_generation_shard(
            FakeRunner(), manifest, artifact_store, "0001", config=active_config
        )


def test_successful_generation_is_not_reused_for_different_source_lineage(tmp_path):
    FakeRunner.calls = 0
    active_config = pinned_config()
    first_manifest = pilot_manifest(active_config)
    second_manifest = replace(first_manifest, manifest_sha256="c" * 64)

    first = run_generation_shard(
        FakeRunner(), first_manifest, store(tmp_path), "0001", config=active_config
    )
    second = run_generation_shard(
        FakeRunner(), second_manifest, store(tmp_path), "0001", config=active_config
    )

    assert second != first
    assert second.shard_id == "0001.retry-1"
    assert FakeRunner.calls == 2


def test_resume_skips_hash_valid_completed_shard_with_mismatched_request_provenance(
    tmp_path,
):
    active_config = pinned_config()
    manifest = pilot_manifest(active_config)
    artifact_store = store(tmp_path)
    seed = run_generation_shard(
        FakeRunner(), manifest, artifact_store, "seed", config=active_config
    )
    forged_row = json.loads(seed.data_path.read_text(encoding="utf-8"))
    forged_lineage = json.loads(seed.manifest_path.read_text(encoding="utf-8"))[
        "lineage"
    ]
    forged_row["example"]["user_text"] = "A different request"
    forged_row["generation"] = {"max_new_tokens": 999}
    forged_row["rendered_prompt_sha256"] = "0" * 64
    forged_row["config_sha256"] = "1" * 64
    forged_row["model_sha256"] = "2" * 64
    forged_row["tokenizer_sha256"] = "3" * 64
    forged_row["chat_template_sha256"] = "4" * 64
    forged = artifact_store.write_completed_shard(
        active_config.run_id,
        "pilot",
        "0001",
        [forged_row],
        forged_lineage,
        record_kind="generation",
    )

    FakeRunner.calls = 0
    fresh = run_generation_shard(
        FakeRunner(), manifest, artifact_store, "0001", config=active_config
    )

    assert fresh != forged
    assert fresh.shard_id == "0001.retry-1"
    assert FakeRunner.calls == 1
    fresh_row = json.loads(fresh.data_path.read_text(encoding="utf-8"))
    assert fresh_row["example"]["user_text"] == "What is stated?"
    assert fresh_row["generation"] == dict(active_config.generation)


class FakeTokenizer:
    chat_template = "verified template bytes"


def test_tokenizer_preparation_loads_only_the_pin_and_verifies_actual_template_bytes():
    expected_hash = hashlib.sha256(FakeTokenizer.chat_template.encode("utf-8")).hexdigest()
    active_config = replace(config(), chat_template_sha256=expected_hash)
    calls = []

    def loader(model_id, *, revision):
        calls.append((model_id, revision))
        return FakeTokenizer()

    prepared = load_pinned_tokenizer(active_config, tokenizer_loader=loader)

    assert calls == [(active_config.model_id, active_config.tokenizer_revision)]
    assert prepared.tokenizer.__class__ is FakeTokenizer
    assert prepared.chat_template_bytes == b"verified template bytes"
    assert prepared.chat_template_sha256 == expected_hash

    with pytest.raises(ValueError, match="chat template hash"):
        load_pinned_tokenizer(replace(active_config, chat_template_sha256="0" * 64), tokenizer_loader=loader)


def test_runner_binding_checks_model_tokenizer_and_template_before_generation():
    active_config = pinned_config()
    validate_runner_binding(FakeRunner(), active_config, expected_chat_template_sha256="f" * 64)

    runner = FakeRunner()
    runner.model_revision = "0" * 40
    with pytest.raises(ValueError, match="model revision"):
        validate_runner_binding(runner, active_config, expected_chat_template_sha256="f" * 64)


def test_smoke_generation_fails_closed_without_a_prepared_template_pin(tmp_path):
    unprepared = config()
    manifest = pilot_manifest(unprepared)

    with pytest.raises(ValueError, match="prepared chat template pin"):
        run_generation_shard(FakeRunner(), manifest, store(tmp_path), "0001", config=unprepared)
