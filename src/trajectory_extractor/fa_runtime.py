"""Resumable, provenance-bound generation for Familiarity-vs-Answerability."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from trajectory_extractor.fa_artifacts import FAArtifactStore, SealedShard
from trajectory_extractor.fa_config import FAConfig


class ModelRunner(Protocol):
    """Small adapter surface that keeps unit tests independent of model downloads."""

    def generate(self, prompts: Sequence[str], generation: Mapping[str, Any]) -> Sequence[str]: ...


@dataclass(frozen=True)
class PreparedTokenizer:
    tokenizer: Any
    chat_template_bytes: bytes
    chat_template_sha256: str


class _CandidateDataError(ValueError):
    """A checksum-valid resume candidate whose payload is not usable generation data."""


def load_pinned_tokenizer(
    config: FAConfig, *, tokenizer_loader: Any | None = None
) -> PreparedTokenizer:
    """Load tokenizer files only and bind the exact configured chat-template bytes."""
    if tokenizer_loader is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - environment-specific
            raise ImportError("Install the Hugging Face tokenizer dependency.") from error
        tokenizer_loader = AutoTokenizer.from_pretrained
    tokenizer = tokenizer_loader(config.model_id, revision=config.tokenizer_revision)
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("pinned tokenizer must expose nonempty chat template bytes")
    template_bytes = template.encode("utf-8")
    template_hash = hashlib.sha256(template_bytes).hexdigest()
    if config.chat_template_sha256 and template_hash != config.chat_template_sha256:
        raise ValueError("tokenizer chat template hash does not match the registered config")
    return PreparedTokenizer(tokenizer, template_bytes, template_hash)


class HFModelRunner:
    """Hugging Face adapter loaded only by the CLI execution path."""

    def __init__(self, config: FAConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as error:  # pragma: no cover - dependency error is environment-specific
            raise ImportError("Install the Hugging Face generation dependencies.") from error
        self._torch = torch
        self.model_id = config.model_id
        self.model_revision = config.model_revision
        self.tokenizer_revision = config.tokenizer_revision
        prepared = load_pinned_tokenizer(config)
        self.tokenizer = prepared.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.chat_template_sha256 = prepared.chat_template_sha256
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
        if shard.record_kind != "generation":
            continue
        try:
            rows = _read_jsonl(shard.data_path)
        except _CandidateDataError:
            continue
        if rows and all(
            row.get("kind") == "generation" and row.get("status") == "completed"
            for row in rows
        ):
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
    expected_template_hash = _prepared_template_hash(manifest, config)
    tokenizer_pin_sha256 = _prepared_tokenizer_pin_sha256(manifest)
    validate_runner_binding(
        runner, config, expected_chat_template_sha256=expected_template_hash
    )
    lineage = {
        "config_sha256": config_hash,
        "source_manifest_sha256": _required_sha256(
            getattr(manifest, "manifest_sha256", None), "manifest_sha256"
        ),
        "model_sha256": _runner_model_hash(runner),
        "tokenizer_sha256": _runner_tokenizer_hash(runner),
        "tokenizer_pin_sha256": tokenizer_pin_sha256,
        "chat_template_sha256": expected_template_hash,
    }
    prompts = tuple(
        _render_prompt(runner, str(getattr(example, "user_text", "")))
        for example in examples
    )
    resumed = _latest_exact_success(
        store,
        run_id,
        inferred_namespace,
        shard_id,
        lineage,
        examples,
        prompts,
        generation,
    )
    if resumed is not None:
        return resumed
    destination_id = _retry_shard_id(store, run_id, inferred_namespace, shard_id)
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
            tokenizer_pin_sha256=tokenizer_pin_sha256,
            generation=generation,
            status=status,
            exception_class=exception_class,
            wall_time_seconds=wall_time,
            peak_memory_bytes=peak_memory,
        )
        for example, prompt, completion in zip(examples, prompts, completions, strict=True)
    ]
    return store.write_completed_shard(
        run_id,
        inferred_namespace,
        destination_id,
        rows,
        lineage,
        record_kind="generation",
    )


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
    tokenizer_pin_sha256: str,
    generation: Mapping[str, Any],
    status: str,
    exception_class: str | None,
    wall_time_seconds: float,
    peak_memory_bytes: int,
) -> dict[str, Any]:
    template_hash = _required_sha256(
        getattr(runner, "chat_template_sha256", None), "runner chat_template_sha256"
    )
    return {
        "kind": "generation",
        "schema_version": 1,
        "example": _json_value(example),
        "example_id": str(getattr(example, "example_id")),
        "example_sha256": _example_hash(example),
        "config_sha256": config_hash,
        "model_sha256": _runner_model_hash(runner),
        "tokenizer_sha256": _runner_tokenizer_hash(runner),
        "tokenizer_pin_sha256": _required_sha256(
            tokenizer_pin_sha256, "tokenizer_pin_sha256"
        ),
        "chat_template_sha256": template_hash,
        "rendered_prompt_sha256": _sha256_text(prompt),
        "generation": dict(generation),
        "raw_output": completion,
        "status": status,
        "exception_class": exception_class,
        "wall_time_seconds": wall_time_seconds,
        "peak_memory_bytes": peak_memory_bytes,
    }


def validate_runner_binding(
    runner: ModelRunner,
    config: FAConfig | None,
    *,
    expected_chat_template_sha256: str,
) -> None:
    """Validate every immutable runner pin before generation."""
    expected_template = _required_sha256(
        expected_chat_template_sha256, "expected chat_template_sha256"
    )
    if config is None:
        raise ValueError("runner binding requires an immutable FA config")
    if getattr(runner, "model_id", None) != config.model_id:
        raise ValueError("runner model id does not match the registered config")
    if getattr(runner, "model_revision", None) != config.model_revision:
        raise ValueError("runner model revision does not match the registered config")
    if getattr(runner, "tokenizer_revision", None) != config.tokenizer_revision:
        raise ValueError("runner tokenizer revision does not match the registered config")
    if getattr(runner, "chat_template_sha256", None) != expected_template:
        raise ValueError("runner chat template hash does not match the prepared pin")
    if not callable(getattr(runner, "generate", None)):
        raise ValueError("runner must expose a generate method")


def _prepared_template_hash(manifest: Any, config: FAConfig | None) -> str:
    manifest_hash = getattr(manifest, "chat_template_sha256", None)
    config_hash = config.chat_template_sha256 if config is not None else None
    if config_hash and manifest_hash and config_hash != manifest_hash:
        raise ValueError("manifest chat template pin does not match the registered config")
    candidate = config_hash or manifest_hash
    if not candidate:
        raise ValueError("generation requires a prepared chat template pin")
    return _required_sha256(candidate, "prepared chat_template_sha256")


def _prepared_tokenizer_pin_sha256(manifest: Any) -> str:
    return _required_sha256(
        getattr(manifest, "tokenizer_pin_sha256", None),
        "prepared tokenizer_pin_sha256",
    )


def _runner_model_hash(runner: ModelRunner) -> str:
    return _sha256_json(
        {
            "model_id": str(getattr(runner, "model_id", "")),
            "revision": str(getattr(runner, "model_revision", "")),
        }
    )


def _runner_tokenizer_hash(runner: ModelRunner) -> str:
    return _sha256_json(
        {
            "model_id": str(getattr(runner, "model_id", "")),
            "revision": str(getattr(runner, "tokenizer_revision", "")),
        }
    )


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


def _latest_exact_success(
    store: FAArtifactStore,
    run_id: str,
    namespace: str,
    shard_id: str,
    lineage: Mapping[str, Any],
    examples: Sequence[Any],
    prompts: Sequence[str],
    generation: Mapping[str, Any],
) -> SealedShard | None:
    expected = {
        str(getattr(example, "example_id")): _expected_completed_record(
            example,
            prompt,
            lineage=lineage,
            generation=generation,
        )
        for example, prompt in zip(examples, prompts, strict=True)
    }
    if len(expected) != len(examples):
        raise ValueError("generation request contains duplicate example IDs")
    matches = []
    for shard in resume_generation(store, run_id, namespace):
        retry_index = _retry_index(shard.shard_id, shard_id)
        if retry_index is None:
            continue
        sidecar = json.loads(shard.manifest_path.read_text(encoding="utf-8"))
        if sidecar.get("lineage") != dict(lineage):
            continue
        rows = _read_jsonl(shard.data_path)
        observed_ids = {str(row.get("example_id")) for row in rows}
        if (
            len(rows) == len(expected)
            and observed_ids == set(expected)
            and all(_completed_record_matches_request(row, expected) for row in rows)
        ):
            matches.append((retry_index, shard))
    return max(matches, default=(None, None), key=lambda item: item[0])[1]


def _expected_completed_record(
    example: Any,
    prompt: str,
    *,
    lineage: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "example": _json_value(example),
        "example_sha256": _example_hash(example),
        "config_sha256": lineage["config_sha256"],
        "model_sha256": lineage["model_sha256"],
        "tokenizer_sha256": lineage["tokenizer_sha256"],
        "tokenizer_pin_sha256": lineage["tokenizer_pin_sha256"],
        "chat_template_sha256": lineage["chat_template_sha256"],
        "rendered_prompt_sha256": _sha256_text(prompt),
        "generation": dict(generation),
    }


def _completed_record_matches_request(
    row: Mapping[str, Any], expected_by_id: Mapping[str, Mapping[str, Any]]
) -> bool:
    if (
        row.get("kind") != "generation"
        or row.get("schema_version") != 1
        or row.get("status") != "completed"
        or row.get("exception_class") is not None
        or not isinstance(row.get("raw_output"), str)
    ):
        return False
    expected = expected_by_id.get(str(row.get("example_id")))
    if expected is None:
        return False
    return all(row.get(name) == value for name, value in expected.items())


def _retry_index(candidate: str, shard_id: str) -> int | None:
    if candidate == shard_id:
        return 0
    prefix = f"{shard_id}.retry-"
    suffix = candidate.removeprefix(prefix)
    if not candidate.startswith(prefix) or not suffix.isdigit() or int(suffix) < 1:
        return None
    return int(suffix)


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
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise _CandidateDataError("generation sidecar is not UTF-8 JSONL") from error
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise _CandidateDataError("generation sidecar row is not JSON") from error
        if not isinstance(value, dict):
            raise _CandidateDataError("generation sidecar row must be an object")
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
