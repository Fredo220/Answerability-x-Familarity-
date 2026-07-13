from __future__ import annotations

import importlib
import importlib.metadata
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from packaging.requirements import Requirement

from trajectory_extractor.rlmf_artifacts import RLMFArtifactStore, sha256_file
from trajectory_extractor.rlmf_trainer import (
    PairedRLMFTrainer,
    answer_prompt,
    generate_group,
    metacognition_prompt,
    derive_generation_seed,
    query_metacognitive_score,
    validate_installed_trl,
)
from trajectory_extractor.rlmf_format import (
    alias_exact_match,
    completion_equivalent,
    parse_rlmf_output,
)
from trajectory_extractor.rlmf_metrics import (
    factual_calibration_reward,
    faithful_calibration_reward,
    gold_faithfulness_level,
    metacognitive_reward,
    soft_format_reward,
    strict_format_reward,
    training_leave_one_out_confidence,
)
from trajectory_extractor.rlmf_types import (
    CheckpointRecord,
    RLMFCompletion,
    RLMFConfig,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_REQUIREMENTS = _REPOSITORY_ROOT / "requirements-rlmf-colab.txt"
REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
    "rlmf_state.json",
    "generation_buffer.pt",
)
_CHECKPOINT_MANIFEST = "checkpoint.json"
MAX_CHECKPOINT_ARCHIVE_BYTES = 2 * 1024**3
MAX_CHECKPOINT_ARCHIVE_MEMBERS = 256
MAX_CHECKPOINT_MANIFEST_BYTES = 1024**2
MAX_CHECKPOINT_MEMBER_BYTES = 1024**3
MAX_CHECKPOINT_UNCOMPRESSED_BYTES = 4 * 1024**3


class TrainingAuditTrail:
    """Durable append-only training evidence with prefix-bound resume state."""

    def __init__(
        self,
        store: RLMFArtifactStore,
        config: RLMFConfig,
        arm: str,
        seed: int,
    ) -> None:
        advantage_form_for_arm(arm)
        if seed not in config.seeds:
            raise ValueError("audit trail seed must be registered")
        self.store = store
        self.config = config
        self.arm = arm
        self.seed = seed
        self._paths = {
            "raw": store.directory_path(
                config.study_id,
                "training_audit",
                f"raw-{arm}-seed-{seed}",
                create_parent=True,
            ).with_suffix(".jsonl"),
            "pre_advantage": store.directory_path(
                config.study_id,
                "training_audit",
                f"pre-advantage-{arm}-seed-{seed}",
                create_parent=True,
            ).with_suffix(".jsonl"),
        }
        self._counts = {
            name: self._existing_count(path) for name, path in self._paths.items()
        }

    def record_raw(self, record: Mapping[str, Any]) -> None:
        self._append("raw", record)

    def record_pre_advantage(self, record: Mapping[str, Any]) -> None:
        self._append("pre_advantage", record)

    def state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "study_id": self.config.study_id,
            "config_hash": self.config.config_hash,
            "arm": self.arm,
            "seed": self.seed,
            **{
                name: self._file_state(name, path)
                for name, path in self._paths.items()
            },
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != {
            "schema_version", "study_id", "config_hash", "arm", "seed", "raw",
            "pre_advantage",
        }:
            raise ValueError("raw-record resume state schema is invalid")
        if (
            state["schema_version"] != 1
            or state["study_id"] != self.config.study_id
            or state["config_hash"] != self.config.config_hash
            or state["arm"] != self.arm
            or state["seed"] != self.seed
        ):
            raise ValueError("raw-record resume state binding is inconsistent")
        for name, path in self._paths.items():
            file_state = state[name]
            if not isinstance(file_state, Mapping) or set(file_state) != {
                "record_count", "byte_size", "sha256_prefix"
            }:
                raise ValueError("raw-record file state schema is invalid")
            record_count = file_state["record_count"]
            byte_size = file_state["byte_size"]
            digest = file_state["sha256_prefix"]
            if (
                type(record_count) is not int
                or record_count < 0
                or type(byte_size) is not int
                or byte_size < 0
                or not isinstance(digest, str)
            ):
                raise ValueError("raw-record file state values are invalid")
            payload = path.read_bytes() if path.exists() else b""
            if len(payload) < byte_size:
                raise ValueError("append-only raw-record artifact was truncated")
            prefix = payload[:byte_size]
            if prefix.count(b"\n") != record_count or hashlib.sha256(prefix).hexdigest() != digest:
                raise ValueError("append-only raw-record prefix does not match the checkpoint")
            self._counts[name] = self._count_complete_records(payload)

    def _append(self, name: str, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            raise ValueError("training audit records must be mappings")
        reserved = {"schema_version", "study_id", "config_hash", "arm", "seed", "sequence"}
        if reserved.intersection(record):
            raise ValueError("training audit record attempts to replace bound fields")
        payload = {
            "schema_version": 1,
            "study_id": self.config.study_id,
            "config_hash": self.config.config_hash,
            "arm": self.arm,
            "seed": self.seed,
            "sequence": self._counts[name],
            **dict(record),
        }
        artifact_name = self._paths[name].stem
        self.store.append_jsonl(
            self.config.study_id, "training_audit", artifact_name, payload
        )
        self._counts[name] += 1

    def _file_state(self, name: str, path: Path) -> dict[str, Any]:
        payload = path.read_bytes() if path.exists() else b""
        if self._count_complete_records(payload) != self._counts[name]:
            raise ValueError("append-only raw-record count changed unexpectedly")
        return {
            "record_count": self._counts[name],
            "byte_size": len(payload),
            "sha256_prefix": hashlib.sha256(payload).hexdigest(),
        }

    @classmethod
    def _existing_count(cls, path: Path) -> int:
        if path.is_symlink():
            raise ValueError("training audit artifact must not be a symlink")
        return cls._count_complete_records(path.read_bytes() if path.exists() else b"")

    @staticmethod
    def _count_complete_records(payload: bytes) -> int:
        if payload and not payload.endswith(b"\n"):
            raise ValueError("append-only training audit has a partial record")
        return payload.count(b"\n")


def validate_runtime_versions(
    *, installed: Mapping[str, str] | None = None
) -> dict[str, str]:
    requirements = tuple(
        Requirement(line)
        for line in _RUNTIME_REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    versions = dict(installed) if installed is not None else {}
    missing = []
    mismatches = []
    for requirement in requirements:
        name = requirement.name
        if installed is None:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                missing.append(name)
                continue
        elif name not in versions:
            missing.append(name)
            continue
        if versions[name] not in requirement.specifier:
            mismatches.append(f"{requirement} (installed {versions[name]})")
    if missing:
        raise RuntimeError(f"RLMF runtime packages are missing: {', '.join(sorted(missing))}")
    if mismatches:
        raise RuntimeError("RLMF runtime version mismatch: " + "; ".join(mismatches))
    return {requirement.name: versions[requirement.name] for requirement in requirements}


def validate_lora_targets(module_names: Iterable[str], targets: Sequence[str]) -> None:
    names = tuple(module_names)
    missing = [
        target
        for target in targets
        if not any(name == target or name.endswith(f".{target}") for name in names)
    ]
    if missing:
        raise ValueError("LoRA targets matched no modules: " + ", ".join(missing))


def rl_training_parameters(
    config: RLMFConfig, *, seed: int, stop_after_step: int | None = None
) -> dict[str, Any]:
    del stop_after_step
    generation = dict(config.generation)
    generation.pop("enable_thinking")
    return {
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "generation_batch_size": config.generation_batch_size,
        "num_generations": config.num_generations,
        "max_prompt_length": config.max_prompt_tokens,
        "max_completion_length": config.max_completion_tokens,
        "max_steps": config.rl_steps,
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "save_only_model": False,
        "gradient_checkpointing": True,
        "fp16": True,
        "scale_rewards": "none",
        "seed": seed,
        "data_seed": seed,
        "generation_kwargs": generation,
        "reward_weights": [
            config.reward_weights[name]
            for name in (
                "soft_format",
                "strict_format",
                "factual_calibration",
                "correctness",
                "faithful_calibration",
            )
        ],
        "report_to": "none",
    }


def sft_training_parameters(config: RLMFConfig) -> dict[str, Any]:
    return {
        "num_train_epochs": config.sft_epochs,
        "learning_rate": config.sft_learning_rate,
        "weight_decay": config.sft_weight_decay,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.sft_global_batch_size,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "gradient_checkpointing": True,
        "fp16": True,
        "report_to": "none",
    }


def build_sft_records(
    config: RLMFConfig,
    examples: Sequence[Any],
    model: Any,
    tokenizer: Any,
    store: RLMFArtifactStore,
) -> tuple[tuple[dict[str, str], ...], tuple[dict[str, str], ...]]:
    if not isinstance(config, RLMFConfig) or not isinstance(store, RLMFArtifactStore):
        raise ValueError("SFT construction requires the active config and artifact store")
    path = store.directory_path(
        config.study_id, "pre_sft", "base_generated_bundle", create_parent=True
    ).with_suffix(".json")
    frozen_examples = sorted(
        (value for value in examples if _example_value(value, "split") == "pre_sft"),
        key=lambda value: _example_value(value, "example_id"),
    )
    if len(frozen_examples) != config.split_counts["pre_sft"]:
        raise ValueError("SFT construction requires every registered pre_sft subject")
    if path.exists():
        return _load_sft_bundle(config, frozen_examples, path)

    seen_examples: set[str] = set()
    seen_subjects: set[str] = set()
    rows = []
    train_subject_count = 230 if config.profile == "confirmatory" else 7

    def record_raw_generation(record: Mapping[str, Any]) -> None:
        store.append_jsonl(
            config.study_id,
            "training_audit",
            "raw-pre-sft",
            {
                "schema_version": 1,
                "study_id": config.study_id,
                "config_hash": config.config_hash,
                "stage": "pre_sft",
                **dict(record),
            },
        )

    for index, value in enumerate(frozen_examples):
        example_id = _example_value(value, "example_id")
        subject = _example_value(value, "subject")
        question = _example_value(value, "question")
        aliases = tuple(_example_value(value, "answers", _example_value(value, "aliases", ())))
        if not all(isinstance(item, str) and item for item in (example_id, subject, question)):
            raise ValueError("SFT examples require example_id, subject, and question")
        if not aliases or any(not isinstance(alias, str) or not alias for alias in aliases):
            raise ValueError("SFT examples require frozen aliases")
        if example_id in seen_examples or subject in seen_subjects:
            raise ValueError("SFT examples must have unique subject and example IDs")
        seen_examples.add(example_id)
        seen_subjects.add(subject)
        generated = generate_group(
            model,
            tokenizer,
            question,
            group_size=1 + config.sft_auxiliary_samples,
            seed=config.split_seed,
            study_id=config.study_id,
            arm="standard_grpo",
            step=0,
            example_id=example_id,
            split="pre_sft",
            config_hash=config.config_hash,
            generation=config.generation,
            raw_recorder=record_raw_generation,
        )
        if any(not completion.parsed.valid_format for completion in generated):
            raise ValueError("base-generated SFT answers must satisfy the frozen answer schema")
        official, *auxiliaries = generated
        equivalent = [
            completion_equivalent(
                official.parsed.answer, auxiliary.parsed.answer, aliases
            )
            for auxiliary in auxiliaries
        ]
        fraction = sum(equivalent) / config.sft_auxiliary_samples
        confidence = min(
            config.confidence_values,
            key=lambda bucket: (abs(bucket - fraction), bucket),
        )
        correctness = float(alias_exact_match(official.parsed.answer, aliases))
        f_gold = int(abs(confidence - correctness) <= config.faithfulness_tau)
        answer_completion = (
            f"<sentence>{official.parsed.answer}</sentence>"
            f"<confidence>{confidence:.1f}</confidence>"
        )
        dataset_split = "train" if index < train_subject_count else "validation"
        rows.append(
            {
                "example_id": example_id,
                "subject": subject,
                "question": question,
                "aliases": list(aliases),
                "dataset_split": dataset_split,
                "official_raw_output": official.raw_output,
                "auxiliary_raw_outputs": [item.raw_output for item in auxiliaries],
                "confidence": confidence,
                "g": correctness,
                "f_gold": f_gold,
                "answer_record": {
                    "example_id": example_id,
                    "prompt": answer_prompt(question),
                    "completion": answer_completion,
                },
                "metacognition_record": {
                    "example_id": example_id,
                    "prompt": metacognition_prompt(question, official.parsed.answer),
                    "completion": f"<metascore>{float(f_gold):.1f}</metascore>",
                },
            }
        )
    payload = {
        "schema_version": 1,
        "study_id": config.study_id,
        "config_hash": config.config_hash,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "sft_auxiliary_samples": config.sft_auxiliary_samples,
        "train_subject_ids": [row["subject"] for row in rows if row["dataset_split"] == "train"],
        "validation_subject_ids": [
            row["subject"] for row in rows if row["dataset_split"] == "validation"
        ],
        "rows": rows,
    }
    payload["bundle_hash"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    store.write_json(config.study_id, "pre_sft", "base_generated_bundle", payload)
    return _load_sft_bundle(config, frozen_examples, path)


def _load_sft_bundle(
    config: RLMFConfig, frozen_examples: Sequence[Any], path: Path
) -> tuple[tuple[dict[str, str], ...], tuple[dict[str, str], ...]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("sealed base-generated SFT bundle is missing")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("sealed base-generated SFT bundle is unreadable") from error
    required = {
        "schema_version", "study_id", "config_hash", "model_id", "model_revision",
        "sft_auxiliary_samples", "train_subject_ids", "validation_subject_ids", "rows",
        "bundle_hash",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("sealed base-generated SFT bundle schema is invalid")
    expected_hash = payload.pop("bundle_hash")
    actual_hash = hashlib.sha256(_canonical_json(payload)).hexdigest()
    payload["bundle_hash"] = expected_hash
    if expected_hash != actual_hash:
        raise ValueError("sealed base-generated SFT bundle hash mismatch")
    if (
        payload["schema_version"] != 1
        or payload["study_id"] != config.study_id
        or payload["config_hash"] != config.config_hash
        or payload["model_id"] != config.model_id
        or payload["model_revision"] != config.model_revision
        or payload["sft_auxiliary_samples"] != config.sft_auxiliary_samples
    ):
        raise ValueError("sealed base-generated SFT bundle provenance mismatch")
    expected_ids = [_example_value(value, "example_id") for value in frozen_examples]
    rows = payload["rows"]
    if not isinstance(rows, list) or [row.get("example_id") for row in rows] != expected_ids:
        raise ValueError("sealed base-generated SFT bundle subjects do not match the snapshot")
    expected_train = 230 if config.profile == "confirmatory" else 7
    if len(payload["train_subject_ids"]) != expected_train or len(
        payload["validation_subject_ids"]
    ) != len(rows) - expected_train:
        raise ValueError("sealed base-generated SFT bundle split is invalid")
    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    for row in rows:
        destination = train if row.get("dataset_split") == "train" else validation
        for record_name in ("answer_record", "metacognition_record"):
            record = row.get(record_name)
            if not isinstance(record, dict) or set(record) != {"example_id", "prompt", "completion"}:
                raise ValueError("sealed base-generated SFT bundle record is invalid")
            destination.append(dict(record))
    return tuple(train), tuple(validation)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_lora_config(config: RLMFConfig):
    validate_runtime_versions()
    peft = importlib.import_module("peft")
    return peft.LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_new_quantized_policy(config: RLMFConfig, peft_config=None):
    """Build a fresh quantized base; TRL receives the one PEFT config separately."""
    validate_runtime_versions()
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    if peft_config is not None:
        configured = tuple(getattr(peft_config, "target_modules", ()))
        if set(configured) != set(config.lora_targets):
            raise ValueError("PEFT config must use every registered LoRA target")
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    validate_lora_targets((name for name, _ in model.named_modules()), config.lora_targets)
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def load_trainable_adapter(config: RLMFConfig, adapter_path: Path):
    """Load an adapter for initialization or exact resume without a second PEFT config."""
    validate_runtime_versions()
    peft = importlib.import_module("peft")
    base = build_new_quantized_policy(config, peft_config=None)
    base = peft.prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True
    )
    return peft.PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)


def seal_checkpoint(
    store: RLMFArtifactStore,
    config: RLMFConfig,
    source: str | Path,
    *,
    stage: str,
    arm: str | None,
    seed: int | None,
    global_step: int,
    micro_step: int,
    sampler_cursor: int,
    parent_hashes: Mapping[str, str],
    completed: bool,
    canonical: bool = False,
) -> CheckpointRecord:
    source_path = Path(source)
    files = _validate_checkpoint_source(source_path)
    supplied_parents = dict(parent_hashes)
    if stage == "pre_sft" and supplied_parents:
        raise ValueError("pre_sft checkpoints must bind only the active config")
    if stage == "rl" and set(supplied_parents) != {"pre_sft"}:
        raise ValueError("RL checkpoints must bind exactly one pre-SFT parent")
    if canonical and stage != "pre_sft":
        raise ValueError("only pre_sft checkpoints can be canonical parents")
    checkpoint_name = _checkpoint_name(stage, arm, seed, global_step, canonical=canonical)
    destination = store.directory_path(
        config.study_id, "checkpoints", checkpoint_name, create_parent=True
    )
    record = CheckpointRecord.create(
        study_id=config.study_id,
        stage=stage,
        arm=arm,
        seed=seed,
        global_step=global_step,
        micro_step=micro_step,
        sampler_cursor=sampler_cursor,
        files=files,
        parent_hashes={"config": config.config_hash, **supplied_parents},
        path=str(destination),
        completed=completed,
    )
    with tempfile.TemporaryDirectory(prefix="rlmf-checkpoint-seal-") as temporary:
        staging = Path(temporary)
        for relative in REQUIRED_CHECKPOINT_FILES:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path / relative, target)
        (staging / _CHECKPOINT_MANIFEST).write_text(
            json.dumps(record.to_record(), indent=2, sort_keys=True)
        )
        published = store.publish_directory(
            config.study_id, "checkpoints", checkpoint_name, staging
        )
    verified = _verify_checkpoint_directory(published)
    if verified.checkpoint_hash != record.checkpoint_hash:
        raise RuntimeError("published checkpoint hash changed")
    return verified


def latest_verified_checkpoint(path: Path) -> Path | None:
    root = Path(path)
    if not root.exists():
        return None
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint path must be a real directory")
    candidates = [root] if (root / _CHECKPOINT_MANIFEST).exists() else sorted(root.iterdir())
    verified: list[tuple[int, int, Path]] = []
    for candidate in candidates:
        if candidate.name.startswith("."):
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError("partial checkpoint entry is not a real directory")
        if not (candidate / _CHECKPOINT_MANIFEST).is_file():
            if any(candidate.iterdir()):
                raise ValueError(f"partial checkpoint: {candidate}")
            continue
        record = _verify_checkpoint_directory(candidate)
        verified.append((record.global_step, record.micro_step, candidate))
    if not verified:
        return None
    return max(verified, key=lambda item: (item[0], item[1], item[2].name))[2]


def export_checkpoint(
    store: RLMFArtifactStore,
    checkpoint: CheckpointRecord | str | Path,
    destination: str | Path,
) -> Path:
    record = (
        checkpoint
        if isinstance(checkpoint, CheckpointRecord)
        else _verify_checkpoint_directory(Path(checkpoint))
    )
    source = Path(record.path)
    verified = _verify_checkpoint_directory(source)
    if verified.checkpoint_hash != record.checkpoint_hash:
        raise ValueError("checkpoint record does not match checkpoint directory")
    if not store.owns_path(record.study_id, source):
        raise ValueError("checkpoint is outside the supplied artifact store namespace")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temporary = Path(name)
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for relative in (_CHECKPOINT_MANIFEST, *sorted(record.files)):
                path = source / relative
                info = archive.gettarinfo(str(path), arcname=relative)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
        temporary.unlink()
        temporary = None
        _fsync_directory(output.parent)
        return output
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def import_checkpoint(
    store: RLMFArtifactStore,
    archive: str | Path,
    *,
    config: RLMFConfig,
    expected_stage: str,
    expected_arm: str | None,
    expected_seed: int | None,
    expected_pre_sft_hash: str | None,
) -> CheckpointRecord:
    source = Path(archive)
    if source.is_symlink() or not source.is_file():
        raise ValueError("checkpoint archive must be a regular file")
    if source.stat().st_size > MAX_CHECKPOINT_ARCHIVE_BYTES:
        raise ValueError("checkpoint archive compressed size exceeds the limit")
    with tarfile.open(source, mode="r:*") as bundle:
        members: dict[str, tarfile.TarInfo] = {}
        names: set[str] = set()
        uncompressed_size = 0
        for member_count, member in enumerate(bundle, start=1):
            if member_count > MAX_CHECKPOINT_ARCHIVE_MEMBERS:
                raise ValueError("checkpoint archive member count exceeds the limit")
            _validate_archive_member(member, names)
            if member.name == _CHECKPOINT_MANIFEST and member.size > MAX_CHECKPOINT_MANIFEST_BYTES:
                raise ValueError("checkpoint archive manifest size exceeds the limit")
            if member.size > MAX_CHECKPOINT_MEMBER_BYTES:
                raise ValueError("checkpoint archive member size exceeds the limit")
            uncompressed_size += member.size
            if uncompressed_size > MAX_CHECKPOINT_UNCOMPRESSED_BYTES:
                raise ValueError("checkpoint archive uncompressed size exceeds the limit")
            members[member.name] = member
        if _CHECKPOINT_MANIFEST not in names:
            raise ValueError("checkpoint archive has no manifest")
        manifest_member = members[_CHECKPOINT_MANIFEST]
        manifest_file = bundle.extractfile(manifest_member)
        if manifest_file is None:
            raise ValueError("checkpoint manifest is unreadable")
        try:
            manifest_payload = manifest_file.read(MAX_CHECKPOINT_MANIFEST_BYTES + 1)
            if len(manifest_payload) > MAX_CHECKPOINT_MANIFEST_BYTES:
                raise ValueError("checkpoint archive manifest size exceeds the limit")
            record = CheckpointRecord.from_record(json.loads(manifest_payload))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            if isinstance(error, ValueError) and "manifest size" in str(error):
                raise
            raise ValueError("checkpoint manifest is invalid") from error
        _validate_checkpoint_binding(
            record,
            config,
            stage=expected_stage,
            arm=expected_arm,
            seed=expected_seed,
            expected_pre_sft_hash=expected_pre_sft_hash,
        )
        expected = {_CHECKPOINT_MANIFEST, *record.files}
        if names != expected or set(record.files) != set(REQUIRED_CHECKPOINT_FILES):
            raise ValueError("checkpoint archive contains unregistered files")
        checkpoint_name = _checkpoint_name(
            record.stage, record.arm, record.seed, record.global_step
        )
        destination = store.directory_path(
            record.study_id, "checkpoints", checkpoint_name, create_parent=True
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(prefix="rlmf-checkpoint-import-") as temporary:
            staging = Path(temporary)
            for relative in sorted(record.files):
                member = members[relative]
                handle = bundle.extractfile(member)
                if handle is None:
                    raise ValueError(f"checkpoint member is unreadable: {relative}")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                if sha256_file(target) != record.files[relative]:
                    raise ValueError(f"checkpoint file hash mismatch: {relative}")
            imported = replace(record, path=str(destination))
            (staging / _CHECKPOINT_MANIFEST).write_text(
                json.dumps(imported.to_record(), indent=2, sort_keys=True)
            )
            store.publish_directory(
                record.study_id, "checkpoints", checkpoint_name, staging
            )
    return _verify_checkpoint_directory(destination)


def load_training_examples(
    config: RLMFConfig, store: RLMFArtifactStore
) -> tuple[dict[str, Any], ...]:
    path = store.directory_path(config.study_id, "data", "popqa_snapshot").with_suffix(
        ".jsonl"
    )
    if path.is_symlink() or not path.is_file():
        raise ValueError("verified PopQA training snapshot is missing")
    rows = tuple(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows:
        raise ValueError("PopQA training snapshot is empty")
    return rows


def advantage_form_for_arm(arm: str) -> str:
    if arm == "standard_grpo":
        return "standard"
    if arm == "rlmf":
        return "mf"
    raise ValueError("arm must be standard_grpo or rlmf")


def run_pre_sft(
    config: RLMFConfig,
    examples: Sequence[Any],
    store: RLMFArtifactStore,
    *,
    resume: bool,
) -> CheckpointRecord:
    return _run_stage(
        config,
        examples,
        store,
        stage="pre_sft",
        arm=None,
        seed=None,
        resume=resume,
        stop_after_step=None,
        infrastructure_pilot=False,
    )


def run_rl_arm(
    config: RLMFConfig,
    arm: str,
    seed: int,
    examples: Sequence[Any],
    store: RLMFArtifactStore,
    *,
    resume: bool,
    stop_after_step: int | None = None,
    infrastructure_pilot: bool = False,
) -> CheckpointRecord:
    advantage_form_for_arm(arm)
    if seed not in config.seeds:
        raise ValueError("seed must be registered in the selected config")
    if type(infrastructure_pilot) is not bool:
        raise ValueError("infrastructure_pilot must be boolean")
    if infrastructure_pilot:
        if config.profile != "confirmatory":
            raise ValueError("infrastructure pilot requires the confirmatory config")
        if seed != config.seeds[0]:
            raise ValueError("infrastructure pilot requires the first registered seed")
        if stop_after_step != config.save_steps or stop_after_step != 25:
            raise ValueError("confirmatory infrastructure pilot must stop at step 25")
    elif stop_after_step is not None and config.profile != "smoke":
        raise ValueError("confirmatory truncation requires the explicit infrastructure pilot flag")
    if stop_after_step is not None and (
        type(stop_after_step) is not int or not 0 < stop_after_step <= config.rl_steps
    ):
        raise ValueError("stop_after_step must be within the registered RL budget")
    return _run_stage(
        config,
        examples,
        store,
        stage="rl",
        arm=arm,
        seed=seed,
        resume=resume,
        stop_after_step=stop_after_step,
        infrastructure_pilot=infrastructure_pilot,
    )


def _run_stage(
    config: RLMFConfig,
    examples: Sequence[Any],
    store: RLMFArtifactStore,
    *,
    stage: str,
    arm: str | None,
    seed: int | None,
    resume: bool,
    stop_after_step: int | None,
    infrastructure_pilot: bool,
) -> CheckpointRecord:
    versions = validate_runtime_versions()
    validate_installed_trl()
    existing = _checkpoint_records(store, config.study_id)
    matching = [
        record
        for record in existing
        if record.stage == stage and record.arm == arm and record.seed == seed
    ]
    if any(record.completed for record in matching):
        raise FileExistsError("completed training arm must not be overwritten")
    parent_record = None
    if stage == "rl":
        parent_record = _select_canonical_pre_sft_parent(existing, config)
    resume_record = (
        _select_resume_checkpoint(
            existing,
            config,
            stage=stage,
            arm=arm,
            seed=seed,
            expected_pre_sft_hash=(
                None if parent_record is None else parent_record.checkpoint_hash
            ),
        )
        if resume
        else None
    )
    if resume and resume_record is None:
        raise ValueError("resume requested but no complete restartable checkpoint exists")

    started = time.monotonic()
    run_name = "pre-sft" if stage == "pre_sft" else f"{arm}-seed-{seed}"
    output_dir = store.directory_path(
        config.study_id, "working", run_name, create_parent=True
    )
    if output_dir.is_symlink():
        raise ValueError("training working directory must not be a symlink")
    output_dir.mkdir(exist_ok=True)
    trainer = _build_trainer(
        config,
        examples,
        stage=stage,
        arm=arm,
        advantage_form=None if arm is None else advantage_form_for_arm(arm),
        seed=seed,
        adapter_path=(
            Path(resume_record.path)
            if resume_record is not None
            else (Path(parent_record.path) if parent_record is not None else None)
        ),
        exact_resume=resume_record is not None,
        stop_after_step=stop_after_step,
        output_dir=output_dir,
        store=store,
    )
    parents = {}
    if parent_record is not None:
        parents["pre_sft"] = parent_record.checkpoint_hash
    checkpoint_sealer = _attach_checkpoint_sealer(
        trainer,
        store,
        config,
        stage=stage,
        arm=arm,
        seed=seed,
        parent_hashes=parents,
        stop_after_step=stop_after_step,
    )
    if stop_after_step is not None:
        _attach_stop_after_step(trainer, stop_after_step)
    if resume_record is not None:
        _restore_custom_restart_state(resume_record, trainer)
    trainer.train(
        resume_from_checkpoint=(str(resume_record.path) if resume_record is not None else None)
    )
    if stage == "pre_sft":
        source = _canonical_pre_sft_source(trainer, Path(trainer.args.output_dir))
        state = json.loads((source / "trainer_state.json").read_text())
        global_step = int(state["global_step"])
        micro_step = int(
            getattr(trainer, "_step", global_step * config.gradient_accumulation_steps)
        )
        sampler_cursor = int(getattr(trainer, "sampler_cursor", micro_step))
        _ensure_custom_restart_state(source, trainer, global_step, micro_step, sampler_cursor)
        record = seal_checkpoint(
            store,
            config,
            source,
            stage=stage,
            arm=arm,
            seed=seed,
            global_step=global_step,
            micro_step=micro_step,
            sampler_cursor=sampler_cursor,
            parent_hashes=parents,
            completed=True,
            canonical=True,
        )
        _write_operational_log(
            store, config, record, trainer, versions, time.monotonic() - started, len(examples)
        )
        return record
    if checkpoint_sealer.last_record is not None:
        record = checkpoint_sealer.last_record
        _write_operational_log(
            store,
            config,
            record,
            trainer,
            versions,
            time.monotonic() - started,
            len(examples),
        )
        return record
    source = _latest_trainer_checkpoint(Path(trainer.args.output_dir))
    state = json.loads((source / "trainer_state.json").read_text())
    global_step = int(state["global_step"])
    micro_step = int(getattr(trainer, "_step", global_step * config.gradient_accumulation_steps))
    sampler_cursor = int(getattr(trainer, "sampler_cursor", micro_step))
    _ensure_custom_restart_state(source, trainer, global_step, micro_step, sampler_cursor)
    target_steps = config.sft_epochs if stage == "pre_sft" else (stop_after_step or config.rl_steps)
    completed = global_step >= target_steps and stop_after_step is None
    record = seal_checkpoint(
        store,
        config,
        source,
        stage=stage,
        arm=arm,
        seed=seed,
        global_step=global_step,
        micro_step=micro_step,
        sampler_cursor=sampler_cursor,
        parent_hashes=parents,
        completed=completed,
    )
    _write_operational_log(
        store, config, record, trainer, versions, time.monotonic() - started, len(examples)
    )
    return record


def _attach_checkpoint_sealer(
    trainer: Any,
    store: RLMFArtifactStore,
    config: RLMFConfig,
    *,
    stage: str,
    arm: str | None,
    seed: int | None,
    parent_hashes: Mapping[str, str],
    stop_after_step: int | None,
):
    transformers = importlib.import_module("transformers")

    class CheckpointSealer(transformers.TrainerCallback):
        def __init__(self):
            self.last_record = None

        def on_save(self, args, state, control, **kwargs):
            del kwargs
            source = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            micro_step = int(
                getattr(trainer, "_step", state.global_step * config.gradient_accumulation_steps)
            )
            sampler_cursor = int(getattr(trainer, "sampler_cursor", micro_step))
            _ensure_custom_restart_state(
                source, trainer, int(state.global_step), micro_step, sampler_cursor
            )
            completed = (
                stage == "rl"
                and stop_after_step is None
                and state.global_step >= config.rl_steps
            )
            self.last_record = seal_checkpoint(
                store,
                config,
                source,
                stage=stage,
                arm=arm,
                seed=seed,
                global_step=int(state.global_step),
                micro_step=micro_step,
                sampler_cursor=sampler_cursor,
                parent_hashes=parent_hashes,
                completed=completed,
            )
            return control

    callback = CheckpointSealer()
    trainer.add_callback(callback)
    return callback


def _attach_stop_after_step(trainer: Any, stop_after_step: int):
    if type(stop_after_step) is not int or stop_after_step < 1:
        raise ValueError("stop_after_step must be a positive integer")
    transformers = importlib.import_module("transformers")

    class StopAfterStep(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            if state.global_step >= stop_after_step:
                control.should_save = True
                control.should_training_stop = True
            return control

    callback = StopAfterStep()
    trainer.add_callback(callback)
    return callback


def _build_trainer(
    config: RLMFConfig,
    examples: Sequence[Any],
    *,
    stage: str,
    arm: str | None,
    advantage_form: str | None,
    seed: int | None,
    adapter_path: Path | None,
    exact_resume: bool,
    stop_after_step: int | None,
    output_dir: Path,
    store: RLMFArtifactStore,
):
    del exact_resume
    trl = importlib.import_module("trl")
    datasets = importlib.import_module("datasets")
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if stage == "pre_sft":
        peft_config = None
        if adapter_path is None:
            peft_config = build_lora_config(config)
            model = build_new_quantized_policy(config, peft_config)
        else:
            model = load_trainable_adapter(config, adapter_path)
        train_rows, validation = build_sft_records(
            config, examples, model, tokenizer, store
        )
        args = trl.SFTConfig(
            output_dir=str(output_dir),
            max_length=config.max_prompt_tokens + config.max_completion_tokens,
            **sft_training_parameters(config),
        )
        return trl.SFTTrainer(
            model=model,
            args=args,
            train_dataset=datasets.Dataset.from_list(list(train_rows)),
            eval_dataset=datasets.Dataset.from_list(list(validation)),
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    if adapter_path is None or seed is None or advantage_form is None:
        raise ValueError("RL trainer requires a pre-SFT or exact-resume adapter")
    model = load_trainable_adapter(config, adapter_path)
    train_rows = _rl_training_records(config, examples, tokenizer)
    args = trl.GRPOConfig(
        output_dir=str(output_dir),
        **rl_training_parameters(config, seed=seed, stop_after_step=stop_after_step),
    )
    if arm is None:
        raise ValueError("RL trainer requires a stored arm")
    audit_trail = TrainingAuditTrail(store, config, arm, seed)
    scorer = _MetacognitionScorer(model, tokenizer, config, arm, seed, audit_trail)
    trainer = PairedRLMFTrainer(
        model=model,
        reward_funcs=_reward_functions(config),
        args=args,
        train_dataset=datasets.Dataset.from_list(list(train_rows)),
        processing_class=tokenizer,
        peft_config=None,
        advantage_form=advantage_form,
        metacognition_scorer=scorer,
        study_id=config.study_id,
        generation_seed=seed,
        raw_answer_recorder=audit_trail.record_raw,
        pre_advantage_recorder=audit_trail.record_pre_advantage,
        _base_trainer_cls=validate_installed_trl(),
    )
    scorer.trainer = trainer
    trainer._rlmf_metacognition_scorer = scorer
    return trainer


def _rl_training_records(
    config: RLMFConfig, examples: Sequence[Any], tokenizer: Any
) -> tuple[dict[str, Any], ...]:
    rows = []
    for value in examples:
        if _example_value(value, "split") != "rl_train":
            continue
        question = _example_value(value, "question")
        example_id = _example_value(value, "example_id")
        answers = tuple(_example_value(value, "answers", ()))
        if not question or not example_id or not answers:
            raise ValueError("RL examples require question, example_id, and answers")
        prompt = answer_prompt(question)
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        rows.append(
            {
                "prompt": prompt,
                "question": question,
                "example_id": example_id,
                "answers": list(answers),
                "study_id": config.study_id,
            }
        )
    if not rows:
        raise ValueError("RL training examples are missing")
    return tuple(rows)


def _reward_functions(config: RLMFConfig) -> list[Any]:
    group_size = config.num_generations

    def soft_format(prompts, completions, **kwargs):
        del prompts, kwargs
        return soft_format_reward([_completion_text(value) for value in completions]).tolist()

    def strict_format(prompts, completions, **kwargs):
        del prompts, kwargs
        parsed = [parse_rlmf_output(_completion_text(value)) for value in completions]
        return strict_format_reward(parsed).tolist()

    def correctness(prompts, completions, answers, **kwargs):
        del prompts, kwargs
        return [
            float(alias_exact_match(parse_rlmf_output(_completion_text(value)).answer, aliases))
            for value, aliases in zip(completions, answers, strict=True)
        ]

    def factual_calibration(prompts, completions, answers, **kwargs):
        del prompts, kwargs
        parsed = [parse_rlmf_output(_completion_text(value)) for value in completions]
        confidence = np.asarray(
            [item.confidence if item.confidence is not None else 0.0 for item in parsed]
        )
        correct = np.asarray(
            [
                float(alias_exact_match(item.answer, aliases))
                for item, aliases in zip(parsed, answers, strict=True)
            ]
        )
        return factual_calibration_reward(confidence, correct).tolist()

    def faithful_calibration(prompts, completions, answers, **kwargs):
        del prompts, kwargs
        if len(completions) % group_size:
            raise ValueError("faithful reward received an incomplete rollout group")
        result = []
        for start in range(0, len(completions), group_size):
            group_completions = completions[start : start + group_size]
            group_aliases = answers[start : start + group_size]
            parsed = [parse_rlmf_output(_completion_text(value)) for value in group_completions]
            raw_answers = [item.answer for item in parsed]
            alias_map = {
                answer: tuple(aliases)
                for answer, aliases in zip(raw_answers, group_aliases, strict=True)
            }
            intrinsic = training_leave_one_out_confidence(raw_answers, alias_map)
            confidence = np.asarray(
                [item.confidence if item.confidence is not None else 0.0 for item in parsed]
            )
            result.extend(faithful_calibration_reward(confidence, intrinsic).tolist())
        return result

    soft_format.__name__ = "soft_format"
    strict_format.__name__ = "strict_format"
    factual_calibration.__name__ = "factual_calibration"
    correctness.__name__ = "correctness"
    faithful_calibration.__name__ = "faithful_calibration"
    return [soft_format, strict_format, factual_calibration, correctness, faithful_calibration]


class _MetacognitionScorer:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: RLMFConfig,
        arm: str,
        seed: int,
        audit_trail: TrainingAuditTrail,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.arm = arm
        self.seed = seed
        self.audit_trail = audit_trail
        self.trainer = None
        self.raw_records: list[dict[str, Any]] = []

    def __call__(self, inputs: Sequence[Mapping[str, Any]], completions: Sequence[Any]):
        group_size = self.config.num_generations
        if len(completions) != len(inputs) or len(completions) % group_size:
            raise ValueError("metacognition queries must align with complete rollout groups")
        scores = []
        step = int(getattr(getattr(self.trainer, "state", None), "global_step", 0))
        for start in range(0, len(completions), group_size):
            group_inputs = inputs[start : start + group_size]
            raw_answers = [_completion_text(value) for value in completions[start : start + group_size]]
            parsed_answers = [parse_rlmf_output(raw) for raw in raw_answers]
            alias_map = {
                item.answer: tuple(_example_value(row, "answers", ()))
                for item, row in zip(parsed_answers, group_inputs, strict=True)
            }
            intrinsic = training_leave_one_out_confidence(
                [item.answer for item in parsed_answers], alias_map
            )
            confidence = np.asarray(
                [item.confidence if item.confidence is not None else 0.0 for item in parsed_answers]
            )
            gold = gold_faithfulness_level(
                confidence, intrinsic, tau=self.config.faithfulness_tau
            )
            for member, (row, raw, parsed, gold_level) in enumerate(
                zip(group_inputs, raw_answers, parsed_answers, gold, strict=True)
            ):
                question = _example_value(row, "question")
                example_id = _example_value(row, "example_id")
                completion = RLMFCompletion(
                    study_id=self.config.study_id,
                    arm=self.arm,
                    seed=self.seed,
                    split="rl_train",
                    example_id=example_id,
                    candidate_id=f"{example_id}-step-{step}-member-{member}",
                    raw_output=raw,
                    parsed=parsed,
                    checkpoint_hash="0" * 64,
                    config_hash=self.config.config_hash,
                    parent_hashes={},
                    source_question=question,
                )
                query_seed = derive_generation_seed(
                    self.config.study_id,
                    self.seed,
                    step,
                    example_id,
                    member,
                    "metacognition",
                )
                raw_meta: list[str] = []
                parsed_meta = query_metacognitive_score(
                    self.model,
                    self.tokenizer,
                    completion,
                    seed=query_seed,
                    generation=self.config.generation,
                    raw_sink=raw_meta,
                    raw_recorder=lambda raw, *, _step=step, _example_id=example_id,
                    _member=member, _candidate_id=completion.candidate_id,
                    _query_seed=query_seed: self.audit_trail.record_raw(
                        {
                            "kind": "metacognition",
                            "step": _step,
                            "example_id": _example_id,
                            "candidate_id": _candidate_id,
                            "group_member": _member,
                            "generation_seed": _query_seed,
                            "raw_output": raw,
                        }
                    ),
                )
                self.raw_records.append(
                    {
                        "step": step,
                        "example_id": example_id,
                        "group_member": member,
                        "answer_seed": derive_generation_seed(
                            self.config.study_id,
                            self.seed,
                            step,
                            example_id,
                            member,
                            "answer",
                        ),
                        "metacognition_seed": query_seed,
                        "answer_raw": raw,
                        "metacognition_raw": raw_meta[0],
                    }
                )
                metascore = parsed_meta.metascore if parsed_meta.metascore is not None else 0.0
                scores.append(float(metacognitive_reward([metascore], [gold_level])[0]))
        return scores


def _completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and value and isinstance(value[-1], Mapping):
        content = value[-1].get("content")
        if isinstance(content, str):
            return content
    raise ValueError("completion must be text or one assistant message")


def _validate_checkpoint_source(source: Path) -> dict[str, str]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("checkpoint source must be a real directory")
    discovered = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    symlinks = [path for path in source.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError("checkpoint source must not contain symlinks")
    missing = set(REQUIRED_CHECKPOINT_FILES) - discovered
    if missing:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        raise ValueError("partial checkpoint: " + "; ".join(details))
    return {relative: sha256_file(source / relative) for relative in REQUIRED_CHECKPOINT_FILES}


def _verify_checkpoint_directory(path: Path) -> CheckpointRecord:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("checkpoint directory must be real")
    manifest = path / _CHECKPOINT_MANIFEST
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("partial checkpoint: manifest missing")
    try:
        record = CheckpointRecord.from_record(json.loads(manifest.read_text()))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("checkpoint manifest is invalid") from error
    expected = {_CHECKPOINT_MANIFEST, *record.files}
    discovered = {
        entry.relative_to(path).as_posix()
        for entry in path.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    }
    if any(entry.is_symlink() for entry in path.rglob("*")) or discovered != expected:
        raise ValueError("partial checkpoint contains missing or unregistered files")
    if set(record.files) != set(REQUIRED_CHECKPOINT_FILES):
        raise ValueError("partial checkpoint does not bind all restart state")
    for relative, digest in record.files.items():
        if sha256_file(path / relative) != digest:
            raise ValueError(f"checkpoint file hash mismatch: {relative}")
    if Path(record.path) != path:
        raise ValueError("checkpoint manifest path does not match published directory")
    return record


def _validate_archive_member(member: tarfile.TarInfo, names: set[str]) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError("checkpoint archive contains an unsafe path")
    if member.name in names:
        raise ValueError(f"checkpoint archive contains duplicate member: {member.name}")
    names.add(member.name)
    if not member.isfile() or member.issym() or member.islnk():
        raise ValueError("checkpoint archive contains a link or special member")


def _checkpoint_name(
    stage: str,
    arm: str | None,
    seed: int | None,
    step: int,
    *,
    canonical: bool = False,
) -> str:
    if stage == "pre_sft":
        if canonical:
            return "checkpoint-pre-sft-best"
        return f"checkpoint-pre-sft-step-{step}"
    return f"checkpoint-{arm}-seed-{seed}-step-{step}"


def _checkpoint_records(store: RLMFArtifactStore, study_id: str) -> tuple[CheckpointRecord, ...]:
    probe = store.directory_path(study_id, "checkpoints", "probe", create_parent=True)
    records = []
    for path in probe.parent.iterdir():
        if path.name.startswith("."):
            continue
        records.append(_verify_checkpoint_directory(path))
    return tuple(records)


def _validate_checkpoint_binding(
    record: CheckpointRecord,
    config: RLMFConfig,
    *,
    stage: str,
    arm: str | None,
    seed: int | None,
    expected_pre_sft_hash: str | None,
) -> None:
    if record.study_id != config.study_id:
        raise ValueError("checkpoint study ID does not match the active study")
    if record.parent_hashes.get("config") != config.config_hash:
        raise ValueError("checkpoint config hash does not match the active config")
    if (record.stage, record.arm, record.seed) != (stage, arm, seed):
        raise ValueError("checkpoint stage, arm, or seed does not match the active run")
    expected_parent_keys = {"config"} if stage == "pre_sft" else {"config", "pre_sft"}
    if set(record.parent_hashes) != expected_parent_keys:
        raise ValueError("checkpoint parent binding schema is invalid")
    if stage == "rl" and record.parent_hashes.get("pre_sft") != expected_pre_sft_hash:
        raise ValueError("checkpoint pre-SFT parent does not match the active parent")


def _select_resume_checkpoint(
    records: Sequence[CheckpointRecord],
    config: RLMFConfig,
    *,
    stage: str,
    arm: str | None,
    seed: int | None,
    expected_pre_sft_hash: str | None,
) -> CheckpointRecord | None:
    scoped = [
        record
        for record in records
        if (record.stage, record.arm, record.seed) == (stage, arm, seed)
    ]
    for record in scoped:
        _validate_checkpoint_binding(
            record,
            config,
            stage=stage,
            arm=arm,
            seed=seed,
            expected_pre_sft_hash=expected_pre_sft_hash,
        )
    incomplete = [record for record in scoped if not record.completed]
    if not incomplete:
        return None
    return max(incomplete, key=lambda item: (item.global_step, item.micro_step))


def _select_canonical_pre_sft_parent(
    records: Sequence[CheckpointRecord], config: RLMFConfig
) -> CheckpointRecord:
    completed = [
        record for record in records if record.stage == "pre_sft" and record.completed
    ]
    if len(completed) != 1:
        raise ValueError("RL training requires exactly one canonical completed pre-SFT parent")
    parent = completed[0]
    _validate_checkpoint_binding(
        parent,
        config,
        stage="pre_sft",
        arm=None,
        seed=None,
        expected_pre_sft_hash=None,
    )
    return parent


def _canonical_pre_sft_source(trainer: Any, output_dir: Path) -> Path:
    selected = getattr(getattr(trainer, "state", None), "best_model_checkpoint", None)
    if not isinstance(selected, str) or not selected:
        raise ValueError("pre-SFT trainer did not select a best validation checkpoint")
    source = Path(selected)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("best pre-SFT checkpoint must be a real directory")
    try:
        source.resolve().relative_to(output_dir.resolve())
    except ValueError as error:
        raise ValueError("best pre-SFT checkpoint is outside the active working directory") from error
    return source


def _latest_trainer_checkpoint(output_dir: Path) -> Path:
    candidates = []
    if output_dir.is_dir():
        for path in output_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-"):
                try:
                    step = int(path.name.removeprefix("checkpoint-"))
                except ValueError:
                    continue
                candidates.append((step, path))
    if not candidates:
        raise ValueError("trainer produced no complete restartable checkpoint")
    return max(candidates, key=lambda item: item[0])[1]


def _ensure_custom_restart_state(
    source: Path, trainer: Any, global_step: int, micro_step: int, sampler_cursor: int
) -> None:
    torch = importlib.import_module("torch")
    state = {
        "global_step": global_step,
        "micro_step": micro_step,
        "sampler_cursor": sampler_cursor,
    }
    (source / "rlmf_state.json").write_text(json.dumps(state, sort_keys=True))
    scorer = getattr(trainer, "_rlmf_metacognition_scorer", None)
    audit_trail = getattr(scorer, "audit_trail", None)
    raw_record_state = (
        audit_trail.state()
        if audit_trail is not None
        else {"record_count": len(getattr(scorer, "raw_records", []))}
    )
    torch.save(
        {
            "buffered_inputs": getattr(trainer, "_buffered_inputs", None),
            "sampler_state": getattr(trainer, "_rlmf_sampler_state", None),
            "raw_record_state": raw_record_state,
        },
        source / "generation_buffer.pt",
    )


def _restore_custom_restart_state(record: CheckpointRecord, trainer: Any) -> None:
    verified = _verify_checkpoint_directory(Path(record.path))
    if verified.checkpoint_hash != record.checkpoint_hash:
        raise ValueError("resume checkpoint record does not match its directory")
    source = Path(record.path)
    try:
        custom_state = json.loads((source / "rlmf_state.json").read_text())
        trainer_state = json.loads((source / "trainer_state.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("resume checkpoint state is unreadable") from error
    expected_state = {
        "global_step": record.global_step,
        "micro_step": record.micro_step,
        "sampler_cursor": record.sampler_cursor,
    }
    if custom_state != expected_state:
        mismatched = next(
            (name for name, value in expected_state.items() if custom_state.get(name) != value),
            "schema",
        )
        raise ValueError(f"resume checkpoint {mismatched} is inconsistent")
    if trainer_state.get("global_step") != record.global_step:
        raise ValueError("resume trainer_state global_step is inconsistent")
    torch = importlib.import_module("torch")
    try:
        buffer = torch.load(
            source / "generation_buffer.pt", map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("resume generation buffer is unreadable") from error
    if not isinstance(buffer, Mapping) or set(buffer) != {
        "buffered_inputs", "sampler_state", "raw_record_state"
    }:
        raise ValueError("resume generation buffer schema is inconsistent")
    trainer._step = record.micro_step
    trainer.sampler_cursor = record.sampler_cursor
    trainer._buffered_inputs = buffer["buffered_inputs"]
    trainer._rlmf_sampler_state = buffer["sampler_state"]
    scorer = getattr(trainer, "_rlmf_metacognition_scorer", None)
    audit_trail = getattr(scorer, "audit_trail", None)
    if audit_trail is None:
        if buffer["raw_record_state"] not in ({}, {"record_count": 0}):
            raise ValueError("resume raw-record state has no active audit trail")
    else:
        audit_trail.restore_state(buffer["raw_record_state"])


def _write_operational_log(
    store: RLMFArtifactStore,
    config: RLMFConfig,
    record: CheckpointRecord,
    trainer: Any,
    versions: Mapping[str, str],
    wall_time: float,
    examples_seen: int,
) -> None:
    torch = importlib.import_module("torch")
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    store.write_json(
        config.study_id,
        "operations",
        f"training-{record.checkpoint_hash[:16]}",
        {
            "peak_vram_bytes": peak_vram,
            "wall_time_seconds": wall_time,
            "examples_seen": examples_seen,
            "optimizer_steps": record.global_step,
            "checkpoint_hash": record.checkpoint_hash,
            "package_versions": dict(versions),
        },
    )


def _example_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
