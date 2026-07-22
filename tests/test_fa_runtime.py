from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_runtime import resume_generation, run_generation_shard


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


class FakeRunner:
    calls = 0
    model_id = "fake/model"
    model_revision = "fake-revision"
    tokenizer_revision = "fake-tokenizer"
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


def pilot_manifest() -> Manifest:
    payload = "a" * 64
    return Manifest(
        config_hash=config().config_hash,
        manifest_sha256="b" * 64,
        examples=(Example(example_id=payload, canonical_payload_sha256=payload, user_text="What is stated?"),),
    )


def store(tmp_path) -> FAArtifactStore:
    return FAArtifactStore(tmp_path)


def test_resume_skips_only_verified_completed_shards(tmp_path):
    FakeRunner.calls = 0
    first = run_generation_shard(FakeRunner(), pilot_manifest(), store(tmp_path), "0001", config=config())
    second = run_generation_shard(FakeRunner(), pilot_manifest(), store(tmp_path), "0001", config=config())

    assert first == second
    assert FakeRunner.calls == 1
    assert resume_generation(store(tmp_path), config().run_id, "pilot") == (first,)


def test_generation_records_bind_exact_provenance_and_infrastructure_failures_are_retryable(tmp_path):
    FakeRunner.calls = 0
    failed = run_generation_shard(FailingRunner(), pilot_manifest(), store(tmp_path), "0001", config=config())
    retry = run_generation_shard(FakeRunner(), pilot_manifest(), store(tmp_path), "0001", config=config())
    failed_row = json.loads(failed.data_path.read_text(encoding="utf-8"))
    retry_row = json.loads(retry.data_path.read_text(encoding="utf-8"))

    assert failed != retry
    assert failed_row["status"] == "infrastructure_failure"
    assert failed_row["exception_class"] == "RuntimeError"
    assert retry_row["status"] == "completed"
    assert retry_row["example_sha256"] == "a" * 64
    assert retry_row["config_sha256"] == config().config_hash
    assert retry_row["model_sha256"]
    assert retry_row["tokenizer_sha256"]
    assert retry_row["chat_template_sha256"] == "f" * 64
    assert retry_row["rendered_prompt_sha256"]
    assert retry_row["generation"] == dict(config().generation)
    assert retry_row["raw_output"] == "K7M2Q"
    assert retry_row["wall_time_seconds"] >= 0
    assert type(retry_row["peak_memory_bytes"]) is int
