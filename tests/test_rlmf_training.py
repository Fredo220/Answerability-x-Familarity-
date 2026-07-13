import io
import json
import tarfile
from pathlib import Path

import pytest

from trajectory_extractor.rlmf_artifacts import RLMFArtifactStore
from trajectory_extractor.rlmf_training import (
    REQUIRED_CHECKPOINT_FILES,
    build_sft_records,
    export_checkpoint,
    import_checkpoint,
    latest_verified_checkpoint,
    rl_training_parameters,
    seal_checkpoint,
    sft_training_parameters,
    validate_lora_targets,
    validate_runtime_versions,
)
from trajectory_extractor.rlmf_types import RLMFConfig


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def smoke_config():
    return RLMFConfig.from_json(CONFIGS / "rlmf_qwen06b_smoke.json")


def _checkpoint_source(path: Path):
    path.mkdir(parents=True)
    for relative in REQUIRED_CHECKPOINT_FILES:
        destination = path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"state:{relative}".encode())


def _sealed_checkpoint(tmp_path):
    config = smoke_config()
    source = tmp_path / "source-checkpoint"
    _checkpoint_source(source)
    store = RLMFArtifactStore(tmp_path / "store")
    record = seal_checkpoint(
        store,
        config,
        source,
        stage="rl",
        arm="standard_grpo",
        seed=11,
        global_step=1,
        micro_step=2,
        sampler_cursor=2,
        parent_hashes={"pre_sft": "a" * 64},
        completed=False,
    )
    return store, record


def test_runtime_versions_fail_closed_on_any_pin_mismatch():
    exact = {
        "torch": "2.7.1", "transformers": "4.57.1", "trl": "0.23.0",
        "peft": "0.18.0", "accelerate": "1.12.0", "bitsandbytes": "0.48.2",
        "datasets": "4.3.0", "scipy": "1.12.0", "numpy": "1.26.4",
        "pandas": "2.2.3", "scikit-learn": "1.4.2",
    }
    assert validate_runtime_versions(installed=exact)["trl"] == "0.23.0"
    with pytest.raises(RuntimeError, match="trl==0.23.0"):
        validate_runtime_versions(installed={**exact, "trl": "0.24.0"})
    with pytest.raises(RuntimeError, match="missing"):
        validate_runtime_versions(
            installed={key: value for key, value in exact.items() if key != "peft"}
        )


def test_every_registered_lora_target_must_match_a_model_module():
    names = [f"model.layers.0.self_attn.{name}" for name in smoke_config().lora_targets]
    validate_lora_targets(names, smoke_config().lora_targets)
    with pytest.raises(ValueError, match="down_proj"):
        validate_lora_targets(names[:-1], smoke_config().lora_targets)


def test_registered_training_parameters_leave_steps_per_generation_unset():
    config = RLMFConfig.from_json(CONFIGS / "rlmf_qwen06b_confirmatory.json")

    rl = rl_training_parameters(config, seed=11)
    sft = sft_training_parameters(config)

    assert rl["per_device_train_batch_size"] == 1
    assert rl["num_generations"] == 4
    assert rl["generation_batch_size"] == 4
    assert rl["gradient_accumulation_steps"] == 4
    assert rl["save_steps"] == 25
    assert rl["max_steps"] == 200
    assert rl["fp16"] is True
    assert rl["generation_kwargs"] == {
        "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        "min_p": 0.0, "repetition_penalty": 1.05,
    }
    assert "steps_per_generation" not in rl
    assert sft["num_train_epochs"] == 5
    assert sft["learning_rate"] == 3e-5
    assert sft["weight_decay"] == 0.01
    assert sft["lr_scheduler_type"] == "cosine"
    assert sft["load_best_model_at_end"] is True


def test_pre_sft_records_keep_answer_and_metacognition_schemas_separate():
    examples = [
        {
            "example_id": f"pre-{index}", "split": "pre_sft",
            "question": f"Question {index}?", "answers": [f"Answer {index}"],
        }
        for index in range(4)
    ] + [
        {
            "example_id": f"val-{index}", "split": "validation",
            "question": f"Validation {index}?", "answers": [f"Gold {index}"],
        }
        for index in range(13)
    ]

    train, validation = build_sft_records(examples, validation_rows=26)

    assert len(train) == 8
    assert len(validation) == 26
    for row in (*train, *validation):
        completion = row["completion"]
        is_answer = "<sentence>" in completion and "<confidence>" in completion
        is_meta = completion.startswith("<metascore>") and "<sentence>" not in completion
        assert is_answer ^ is_meta


def test_checkpoint_is_complete_restartable_and_latest_is_verified(tmp_path):
    _, record = _sealed_checkpoint(tmp_path)
    checkpoint = Path(record.path)
    assert set(record.files) == set(REQUIRED_CHECKPOINT_FILES)
    assert latest_verified_checkpoint(checkpoint.parent) == checkpoint
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="partial checkpoint"):
        latest_verified_checkpoint(checkpoint.parent)


def test_adapter_only_checkpoint_is_never_resume_eligible(tmp_path):
    checkpoint = tmp_path / "checkpoint-25"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="partial checkpoint"):
        latest_verified_checkpoint(tmp_path)


def test_checkpoint_export_import_round_trip_is_hash_bound_and_no_overwrite(tmp_path):
    store, record = _sealed_checkpoint(tmp_path)
    archive = export_checkpoint(store, record, tmp_path / "checkpoint.tar")
    imported_store = RLMFArtifactStore(tmp_path / "imported")
    imported = import_checkpoint(imported_store, archive)
    assert imported.checkpoint_hash == record.checkpoint_hash
    assert imported.files == record.files
    with pytest.raises(FileExistsError):
        import_checkpoint(imported_store, archive)


@pytest.mark.parametrize(
    ("name", "kind"),
    [("../escape", "file"), ("/absolute", "file"), ("extra.bin", "file"),
     ("linked", "symlink"), ("hard", "hardlink")],
)
def test_checkpoint_import_rejects_unsafe_or_unregistered_members(tmp_path, name, kind):
    store, record = _sealed_checkpoint(tmp_path)
    archive = export_checkpoint(store, record, tmp_path / "checkpoint.tar")
    rewritten = tmp_path / f"unsafe-{kind}.tar"
    with tarfile.open(archive, "r") as source, tarfile.open(rewritten, "w") as target:
        for member in source.getmembers():
            payload = source.extractfile(member) if member.isfile() else None
            target.addfile(member, payload)
        member = tarfile.TarInfo(name)
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "checkpoint.json"
            target.addfile(member)
        elif kind == "hardlink":
            member.type = tarfile.LNKTYPE
            member.linkname = "checkpoint.json"
            target.addfile(member)
        else:
            payload = b"unregistered"
            member.size = len(payload)
            target.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError):
        import_checkpoint(RLMFArtifactStore(tmp_path / "unsafe-store"), rewritten)


def test_checkpoint_import_rejects_duplicate_member_and_digest_mismatch(tmp_path):
    store, record = _sealed_checkpoint(tmp_path)
    archive = export_checkpoint(store, record, tmp_path / "checkpoint.tar")
    duplicate = tmp_path / "duplicate.tar"
    with tarfile.open(archive, "r") as source, tarfile.open(duplicate, "w") as target:
        members = source.getmembers()
        for member in members:
            payload = source.extractfile(member) if member.isfile() else None
            target.addfile(member, payload)
        manifest = next(member for member in members if member.name == "checkpoint.json")
        payload = json.dumps({"tampered": True}).encode()
        manifest.size = len(payload)
        target.addfile(manifest, io.BytesIO(payload))
    with pytest.raises(ValueError, match="duplicate"):
        import_checkpoint(RLMFArtifactStore(tmp_path / "duplicate-store"), duplicate)

    tampered = tmp_path / "tampered.tar"
    with tarfile.open(archive, "r") as source, tarfile.open(tampered, "w") as target:
        for member in source.getmembers():
            if member.name == "optimizer.pt":
                payload = b"tampered"
                member.size = len(payload)
                target.addfile(member, io.BytesIO(payload))
            else:
                payload = source.extractfile(member) if member.isfile() else None
                target.addfile(member, payload)
    with pytest.raises(ValueError, match="hash mismatch"):
        import_checkpoint(RLMFArtifactStore(tmp_path / "tampered-store"), tampered)
