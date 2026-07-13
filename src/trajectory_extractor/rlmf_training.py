from __future__ import annotations

import importlib
import importlib.metadata
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
    metacognition_prompt,
    derive_generation_seed,
    query_metacognitive_score,
    validate_installed_trl,
)
from trajectory_extractor.rlmf_format import (
    alias_exact_match,
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
        "max_steps": stop_after_step or config.rl_steps,
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
    examples: Sequence[Any], *, validation_rows: int = 26
) -> tuple[tuple[dict[str, str], ...], tuple[dict[str, str], ...]]:
    if type(validation_rows) is not int or validation_rows < 1:
        raise ValueError("validation_rows must be a positive integer")
    grouped: dict[str, list[dict[str, str]]] = {"pre_sft": [], "validation": []}
    for value in examples:
        split = _example_value(value, "split")
        if split not in grouped:
            continue
        question = _example_value(value, "question")
        aliases = _example_value(value, "answers", _example_value(value, "aliases", ()))
        if isinstance(aliases, str) or not aliases:
            raise ValueError("SFT examples require at least one frozen answer")
        answer = str(tuple(aliases)[0])
        grouped[split].extend(
            (
                {
                    "prompt": answer_prompt(question),
                    "completion": (
                        f"<sentence>{answer}</sentence><confidence>1.0</confidence>"
                    ),
                },
                {
                    "prompt": metacognition_prompt(question, answer),
                    "completion": "<metascore>1.0</metascore>",
                },
            )
        )
    if not grouped["pre_sft"]:
        raise ValueError("SFT training examples are missing")
    if len(grouped["validation"]) < validation_rows:
        raise ValueError("fixed SFT validation subset is incomplete")
    return tuple(grouped["pre_sft"]), tuple(grouped["validation"][:validation_rows])


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
) -> CheckpointRecord:
    source_path = Path(source)
    files = _validate_checkpoint_source(source_path)
    checkpoint_name = _checkpoint_name(stage, arm, seed, global_step)
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
        parent_hashes={"config": config.config_hash, **dict(parent_hashes)},
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
    del store
    record = (
        checkpoint
        if isinstance(checkpoint, CheckpointRecord)
        else _verify_checkpoint_directory(Path(checkpoint))
    )
    source = Path(record.path)
    verified = _verify_checkpoint_directory(source)
    if verified.checkpoint_hash != record.checkpoint_hash:
        raise ValueError("checkpoint record does not match checkpoint directory")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with tarfile.open(output, mode="x", format=tarfile.PAX_FORMAT) as archive:
        for relative in (_CHECKPOINT_MANIFEST, *sorted(record.files)):
            path = source / relative
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return output


def import_checkpoint(store: RLMFArtifactStore, archive: str | Path) -> CheckpointRecord:
    source = Path(archive)
    if source.is_symlink() or not source.is_file():
        raise ValueError("checkpoint archive must be a regular file")
    with tarfile.open(source, mode="r:*") as bundle:
        members = bundle.getmembers()
        names: set[str] = set()
        for member in members:
            _validate_archive_member(member, names)
        if _CHECKPOINT_MANIFEST not in names:
            raise ValueError("checkpoint archive has no manifest")
        manifest_member = bundle.getmember(_CHECKPOINT_MANIFEST)
        manifest_file = bundle.extractfile(manifest_member)
        if manifest_file is None:
            raise ValueError("checkpoint manifest is unreadable")
        try:
            record = CheckpointRecord.from_record(json.loads(manifest_file.read()))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("checkpoint manifest is invalid") from error
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
                member = bundle.getmember(relative)
                handle = bundle.extractfile(member)
                if handle is None:
                    raise ValueError(f"checkpoint member is unreadable: {relative}")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        output.write(block)
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
) -> CheckpointRecord:
    advantage_form_for_arm(arm)
    if seed not in config.seeds:
        raise ValueError("seed must be registered in the selected config")
    if stop_after_step is not None:
        if config.profile != "smoke":
            raise ValueError("--stop-after-step is limited to the smoke profile")
        if type(stop_after_step) is not int or not 0 < stop_after_step <= config.rl_steps:
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
    resume_record = max(matching, key=lambda item: item.global_step) if resume and matching else None
    if resume and resume_record is None:
        raise ValueError("resume requested but no complete restartable checkpoint exists")
    parent_record = None
    if stage == "rl":
        pre_sft = [record for record in existing if record.stage == "pre_sft" and record.completed]
        if not pre_sft:
            raise ValueError("RL training requires a completed pre_sft checkpoint")
        parent_record = max(pre_sft, key=lambda item: item.global_step)

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
    trainer.train(
        resume_from_checkpoint=(str(resume_record.path) if resume_record is not None else None)
    )
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
                stage == "pre_sft"
                and state.epoch is not None
                and state.epoch >= config.sft_epochs
            ) or (
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


def _build_trainer(
    config: RLMFConfig,
    examples: Sequence[Any],
    *,
    stage: str,
    advantage_form: str | None,
    seed: int | None,
    adapter_path: Path | None,
    exact_resume: bool,
    stop_after_step: int | None,
    output_dir: Path,
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
        validation_rows = 26 if config.profile == "confirmatory" else min(
            26,
            2 * sum(_example_value(row, "split") == "validation" for row in examples),
        )
        train_rows, validation = build_sft_records(
            examples, validation_rows=validation_rows
        )
        peft_config = None
        if adapter_path is None:
            peft_config = build_lora_config(config)
            model = build_new_quantized_policy(config, peft_config)
        else:
            model = load_trainable_adapter(config, adapter_path)
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
    scorer = _MetacognitionScorer(model, tokenizer, config, seed)
    trainer = PairedRLMFTrainer(
        model=model,
        reward_funcs=_reward_functions(config),
        args=args,
        train_dataset=datasets.Dataset.from_list(list(train_rows)),
        processing_class=tokenizer,
        peft_config=None,
        advantage_form=advantage_form,
        metacognition_scorer=scorer,
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
    def __init__(self, model: Any, tokenizer: Any, config: RLMFConfig, seed: int):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.seed = seed
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
                    arm="standard_grpo",
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


def _checkpoint_name(stage: str, arm: str | None, seed: int | None, step: int) -> str:
    if stage == "pre_sft":
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
    torch.save(
        {
            "buffered_inputs": getattr(trainer, "_buffered_inputs", None),
            "raw_generation_records": getattr(scorer, "raw_records", []),
        },
        source / "generation_buffer.pt",
    )


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
