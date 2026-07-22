"""Resumable, provenance-bound generation for Familiarity-vs-Answerability."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from trajectory_extractor.fa_artifacts import FAArtifactStore, SealedShard
from trajectory_extractor.fa_config import FAConfig


class ModelRunner(Protocol):
    """Small adapter surface that keeps unit tests independent of model downloads."""

    def generate(self, prompts: Sequence[str], generation: Mapping[str, Any]) -> Sequence[str]: ...


class HFModelRunner:
    """Hugging Face adapter loaded only by the CLI execution path."""

    def __init__(self, config: FAConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - dependency error is environment-specific
            raise ImportError("Install the Hugging Face generation dependencies.") from error
        self._torch = torch
        self.model_id = config.model_id
        self.model_revision = config.model_revision
        self.tokenizer_revision = config.tokenizer_revision
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.tokenizer_revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        template = getattr(self.tokenizer, "chat_template", "") or ""
        self.chat_template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id, revision=config.model_revision, low_cpu_mem_usage=True
        )
        self.model.eval()

    def render_prompt(self, user_text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompts: Sequence[str], generation: Mapping[str, Any]) -> Sequence[str]:
        encoded = self.tokenizer(list(prompts), return_tensors="pt", padding=True)
        with self._torch.no_grad():
            generated = self.model.generate(**encoded, **dict(generation))
        prompt_length = encoded["input_ids"].shape[1]
        return self.tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)


def resume_generation(
    store: FAArtifactStore, run_id: str, namespace: str
) -> tuple[SealedShard, ...]:
    """Return only verified shards whose every row is a completed generation."""
    completed: list[SealedShard] = []
    for shard in store.resume_verified_shards(run_id, namespace):
        rows = _read_jsonl(shard.data_path)
        if rows and all(row.get("status") == "completed" for row in rows):
            completed.append(shard)
    return tuple(completed)


def run_generation_shard(
    runner: ModelRunner,
    manifest: Any,
    store: FAArtifactStore,
    shard_id: str,
    *,
    config: FAConfig | None = None,
    namespace: str | None = None,
) -> SealedShard:
    """Generate one immutable shard, resuming only a verified completed sidecar.

    A failed backend call is persisted as an immutable infrastructure-failure shard.
    Re-invocation uses a new retry sidecar rather than replacing the original output.
    """
    examples = tuple(getattr(manifest, "examples", ()))
    if not examples:
        raise ValueError("generation manifest must contain examples")
    inferred_namespace = namespace or _single_namespace(examples)
    if inferred_namespace not in {"pilot", "mechanism_train", "locked_validation", "circuit_dev", "behavior_test", "probe_test", "intervention_test"}:
        raise ValueError("generation namespace is not registered")
    if any(getattr(example, "split", None) != inferred_namespace for example in examples):
        raise ValueError("generation examples must all belong to the requested namespace")
    run_id, config_hash, generation = _generation_identity(manifest, config)
    _validate_runner_binding(runner, config)
    existing = resume_generation(store, run_id, inferred_namespace)
    for shard in existing:
        if shard.shard_id == shard_id:
            return shard
    destination_id = _retry_shard_id(store, run_id, inferred_namespace, shard_id)
    prompts = [_render_prompt(runner, str(getattr(example, "user_text", ""))) for example in examples]
    started = time.perf_counter()
    try:
        completions = tuple(runner.generate(prompts, generation))
        if len(completions) != len(examples) or any(not isinstance(item, str) for item in completions):
            raise RuntimeError("model runner returned an invalid completion batch")
        exception_class = None
        status = "completed"
    except Exception as error:  # failures are data, but never count as completed work
        completions = (None,) * len(examples)
        exception_class = type(error).__name__
        status = "infrastructure_failure"
    wall_time = time.perf_counter() - started
    peak_memory = _peak_memory_bytes()
    rows = [
        _generation_record(
            example,
            prompt,
            completion,
            runner=runner,
            config_hash=config_hash,
            generation=generation,
            status=status,
            exception_class=exception_class,
            wall_time_seconds=wall_time,
            peak_memory_bytes=peak_memory,
        )
        for example, prompt, completion in zip(examples, prompts, completions, strict=True)
    ]
    lineage = {
        "config_sha256": config_hash,
        "source_manifest_sha256": _required_sha256(getattr(manifest, "manifest_sha256", None), "manifest_sha256"),
    }
    return store.write_completed_shard(run_id, inferred_namespace, destination_id, rows, lineage)


def _generation_identity(manifest: Any, config: FAConfig | None) -> tuple[str, str, dict[str, Any]]:
    config_hash = _required_sha256(getattr(manifest, "config_hash", None), "manifest config_hash")
    if config is not None:
        if config.config_hash != config_hash:
            raise ValueError("generation manifest config hash does not match config")
        return config.run_id, config_hash, dict(config.generation)
    run_id = getattr(manifest, "run_id", None)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("generation requires config or a manifest run_id")
    generation = getattr(manifest, "generation", None)
    if not isinstance(generation, Mapping):
        raise ValueError("generation requires config or a manifest generation object")
    return run_id, config_hash, dict(generation)


def _generation_record(
    example: Any,
    prompt: str,
    completion: str | None,
    *,
    runner: ModelRunner,
    config_hash: str,
    generation: Mapping[str, Any],
    status: str,
    exception_class: str | None,
    wall_time_seconds: float,
    peak_memory_bytes: int,
) -> dict[str, Any]:
    model_id = str(getattr(runner, "model_id", "unbound-model"))
    model_revision = str(getattr(runner, "model_revision", "unbound-revision"))
    tokenizer_revision = str(getattr(runner, "tokenizer_revision", "unbound-tokenizer"))
    template = getattr(runner, "chat_template_sha256", None)
    template_hash = template if isinstance(template, str) and len(template) == 64 else _sha256_text("")
    return {
        "schema_version": 1,
        "example": _json_value(example),
        "example_id": str(getattr(example, "example_id")),
        "example_sha256": _example_hash(example),
        "config_sha256": config_hash,
        "model_sha256": _sha256_json({"model_id": model_id, "revision": model_revision}),
        "tokenizer_sha256": _sha256_json({"model_id": model_id, "revision": tokenizer_revision}),
        "chat_template_sha256": template_hash,
        "rendered_prompt_sha256": _sha256_text(prompt),
        "generation": dict(generation),
        "raw_output": completion,
        "status": status,
        "exception_class": exception_class,
        "wall_time_seconds": wall_time_seconds,
        "peak_memory_bytes": peak_memory_bytes,
    }


def _validate_runner_binding(runner: ModelRunner, config: FAConfig | None) -> None:
    if config is None or not config.chat_template_sha256:
        return
    actual = getattr(runner, "chat_template_sha256", None)
    if actual != config.chat_template_sha256:
        raise ValueError("runner chat template hash does not match the registered config")


def _single_namespace(examples: Sequence[Any]) -> str:
    values = {getattr(example, "split", None) for example in examples}
    if len(values) != 1 or not isinstance(next(iter(values)), str):
        raise ValueError("generation examples must have one registered namespace")
    return next(iter(values))


def _retry_shard_id(store: FAArtifactStore, run_id: str, namespace: str, shard_id: str) -> str:
    all_shards = store.resume_verified_shards(run_id, namespace)
    if not any(shard.shard_id == shard_id for shard in all_shards):
        return shard_id
    index = 1
    while any(shard.shard_id == f"{shard_id}.retry-{index}" for shard in all_shards):
        index += 1
    return f"{shard_id}.retry-{index}"


def _render_prompt(runner: ModelRunner, user_text: str) -> str:
    render = getattr(runner, "render_prompt", None)
    if callable(render):
        value = render(user_text)
        if not isinstance(value, str):
            raise ValueError("model runner rendered prompt must be text")
        return value
    return user_text


def _example_hash(example: Any) -> str:
    value = getattr(example, "canonical_payload_sha256", None)
    return _required_sha256(value, "example canonical_payload_sha256")


def _json_value(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "__dict__"):
        value = dict(value.__dict__)
    if not isinstance(value, Mapping):
        raise ValueError("generation example must be serializable")
    return json.loads(json.dumps(dict(value), sort_keys=True, default=list))


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("generation sidecar row must be an object")
        rows.append(value)
    return tuple(rows)


def _required_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return _sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _peak_memory_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024
