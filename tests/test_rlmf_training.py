import io
import json
import tarfile
from pathlib import Path

import pytest
import torch

from trajectory_extractor.rlmf_artifacts import RLMFArtifactStore
import trajectory_extractor.rlmf_training as rlmf_training
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
    pilot = rl_training_parameters(config, seed=11, stop_after_step=25)
    assert pilot["max_steps"] == config.rl_steps
    assert pilot == rl
    assert sft["num_train_epochs"] == 5
    assert sft["learning_rate"] == 3e-5
    assert sft["weight_decay"] == 0.01
    assert sft["lr_scheduler_type"] == "cosine"
    assert sft["load_best_model_at_end"] is True


class FakeSFTBaseModel:
    def __init__(self):
        self.calls = []
        self.counts = {}

    def generate_text(self, prompt, *, seed, generation):
        question_index = int(prompt.split("Question ", 1)[1].split("?", 1)[0])
        member = self.counts.get(question_index, 0)
        self.counts[question_index] = member + 1
        self.calls.append((question_index, member, seed, dict(generation)))
        answer = f"Gold {question_index}"
        if member == 1 and question_index % 2:
            answer = f"Other {question_index}"
        return f"<sentence>{answer}</sentence><confidence>0.5</confidence>"


def test_pre_sft_bundle_is_base_generated_sealed_and_split_by_subject(tmp_path):
    config = smoke_config()
    examples = [
        {
            "example_id": f"pre-{index}", "subject": f"subject-{index}",
            "split": "pre_sft", "question": f"Question {index}?",
            "answers": [f"Gold {index}"],
        }
        for index in range(8)
    ] + [
        {
            "example_id": "validation-poison", "subject": "validation-subject",
            "split": "validation", "question": "Question 99?", "answers": ["Poison"],
        }
    ]
    model = FakeSFTBaseModel()
    store = RLMFArtifactStore(tmp_path / "store")

    train, validation = build_sft_records(config, examples, model, object(), store)

    assert len(train) == 14
    assert len(validation) == 2
    assert {row["example_id"] for row in train} == {f"pre-{index}" for index in range(7)}
    assert {row["example_id"] for row in validation} == {"pre-7"}
    for row in (*train, *validation):
        completion = row["completion"]
        is_answer = "<sentence>" in completion and "<confidence>" in completion
        is_meta = completion.startswith("<metascore>") and "<sentence>" not in completion
        assert is_answer ^ is_meta
    answer_records = [row for row in (*train, *validation) if "<sentence>" in row["completion"]]
    meta_records = [row for row in (*train, *validation) if "<metascore>" in row["completion"]]
    assert [row["completion"] for row in answer_records[:2]] == [
        "<sentence>Gold 0</sentence><confidence>1.0</confidence>",
        "<sentence>Gold 1</sentence><confidence>0.0</confidence>",
    ]
    assert [row["completion"] for row in meta_records[:2]] == [
        "<metascore>1.0</metascore>", "<metascore>0.0</metascore>"
    ]
    assert "Your Answer: Gold 0" in meta_records[0]["prompt"]
    assert "<sentence>" not in meta_records[0]["prompt"]
    assert "<confidence>" not in meta_records[0]["prompt"]
    assert len(model.calls) == 8 * (1 + config.sft_auxiliary_samples)
    assert len({seed for _, _, seed, _ in model.calls}) == len(model.calls)

    raw_path = store.directory_path(
        config.study_id, "training_audit", "raw-pre-sft"
    ).with_suffix(".jsonl")
    raw_records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert len(raw_records) == 8 * (1 + config.sft_auxiliary_samples)
    assert all(record["stage"] == "pre_sft" for record in raw_records)
    assert all(record["config_hash"] == config.config_hash for record in raw_records)

    bundle_path = store.directory_path(
        config.study_id, "pre_sft", "base_generated_bundle"
    ).with_suffix(".json")
    bundle = json.loads(bundle_path.read_text())
    assert bundle["config_hash"] == config.config_hash
    assert bundle["train_subject_ids"] == [f"subject-{index}" for index in range(7)]
    assert bundle["validation_subject_ids"] == ["subject-7"]
    assert len(bundle["bundle_hash"]) == 64
    assert bundle["rows"][1]["confidence"] == 0.0
    assert bundle["rows"][1]["f_gold"] == 0

    unused_model = FakeSFTBaseModel()
    resumed = build_sft_records(config, examples, unused_model, object(), store)
    assert resumed == (train, validation)
    assert unused_model.calls == []


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
    imported = import_checkpoint(
        imported_store,
        archive,
        config=smoke_config(),
        expected_stage="rl",
        expected_arm="standard_grpo",
        expected_seed=11,
        expected_pre_sft_hash="a" * 64,
    )
    assert imported.checkpoint_hash == record.checkpoint_hash
    assert imported.files == record.files
    with pytest.raises(FileExistsError):
        import_checkpoint(
            imported_store,
            archive,
            config=smoke_config(),
            expected_stage="rl",
            expected_arm="standard_grpo",
            expected_seed=11,
            expected_pre_sft_hash="a" * 64,
        )


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
        import_checkpoint(
            RLMFArtifactStore(tmp_path / "unsafe-store"),
            rewritten,
            config=smoke_config(),
            expected_stage="rl",
            expected_arm="standard_grpo",
            expected_seed=11,
            expected_pre_sft_hash="a" * 64,
        )


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
        import_checkpoint(
            RLMFArtifactStore(tmp_path / "duplicate-store"),
            duplicate,
            config=smoke_config(),
            expected_stage="rl",
            expected_arm="standard_grpo",
            expected_seed=11,
            expected_pre_sft_hash="a" * 64,
        )

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
        import_checkpoint(
            RLMFArtifactStore(tmp_path / "tampered-store"),
            tampered,
            config=smoke_config(),
            expected_stage="rl",
            expected_arm="standard_grpo",
            expected_seed=11,
            expected_pre_sft_hash="a" * 64,
        )


@pytest.mark.parametrize(
    ("limit_name", "limit", "message"),
    [
        ("MAX_CHECKPOINT_ARCHIVE_BYTES", 1, "compressed size"),
        ("MAX_CHECKPOINT_ARCHIVE_MEMBERS", 1, "member count"),
        ("MAX_CHECKPOINT_MANIFEST_BYTES", 1, "manifest size"),
        ("MAX_CHECKPOINT_MEMBER_BYTES", 1, "member size"),
        ("MAX_CHECKPOINT_UNCOMPRESSED_BYTES", 1, "uncompressed size"),
    ],
)
def test_checkpoint_import_enforces_resource_limits_before_extraction(
    tmp_path, monkeypatch, limit_name, limit, message
):
    store, record = _sealed_checkpoint(tmp_path)
    archive = export_checkpoint(store, record, tmp_path / "checkpoint.tar")
    monkeypatch.setattr(rlmf_training, limit_name, limit, raising=False)

    with pytest.raises(ValueError, match=message):
        import_checkpoint(
            RLMFArtifactStore(tmp_path / f"limited-{limit_name}"),
            archive,
            config=smoke_config(),
            expected_stage="rl",
            expected_arm="standard_grpo",
            expected_seed=11,
            expected_pre_sft_hash="a" * 64,
        )


def test_checkpoint_export_rejects_paths_outside_supplied_store(tmp_path):
    store, record = _sealed_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="artifact store namespace"):
        export_checkpoint(
            RLMFArtifactStore(tmp_path / "other-store"),
            record,
            tmp_path / "outside.tar",
        )


def test_checkpoint_export_failure_never_publishes_partial_destination(tmp_path, monkeypatch):
    store, record = _sealed_checkpoint(tmp_path)
    destination = tmp_path / "atomic.tar"
    original = tarfile.TarFile.addfile
    calls = 0

    def fail_after_first(self, tarinfo, fileobj=None):
        nonlocal calls
        calls += 1
        original(self, tarinfo, fileobj)
        if calls == 1:
            raise OSError("simulated archive failure")

    monkeypatch.setattr(tarfile.TarFile, "addfile", fail_after_first)
    with pytest.raises(OSError, match="simulated"):
        export_checkpoint(store, record, destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".atomic.tar.*")) == []


class FakeAuditTrail:
    def __init__(self, state=None):
        self._state = {"record_count": 0} if state is None else state
        self.restored = None

    def state(self):
        return dict(self._state)

    def restore_state(self, state):
        self.restored = dict(state)


def _restartable_checkpoint(tmp_path, *, inconsistent=False):
    config = smoke_config()
    source = tmp_path / ("bad-resume" if inconsistent else "exact-resume")
    _checkpoint_source(source)
    (source / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
    custom_step = 2 if inconsistent else 1
    (source / "rlmf_state.json").write_text(
        json.dumps({"global_step": custom_step, "micro_step": 2, "sampler_cursor": 7})
    )
    torch.save(
        {
            "buffered_inputs": [{"prompt_ids": torch.tensor([[1, 2]])}],
            "sampler_state": {"epoch": 3, "cursor": 7},
            "raw_record_state": {"record_count": 4, "last_sequence": 3},
        },
        source / "generation_buffer.pt",
    )
    store = RLMFArtifactStore(tmp_path / ("bad-store" if inconsistent else "resume-store"))
    record = seal_checkpoint(
        store,
        config,
        source,
        stage="rl",
        arm="standard_grpo",
        seed=11,
        global_step=1,
        micro_step=2,
        sampler_cursor=7,
        parent_hashes={"pre_sft": "a" * 64},
        completed=False,
    )
    return config, record


def test_exact_resume_restores_and_validates_all_custom_state(tmp_path):
    _, record = _restartable_checkpoint(tmp_path)
    trail = FakeAuditTrail()
    scorer = type("Scorer", (), {"audit_trail": trail})()
    trainer = type(
        "Trainer", (),
        {
            "_step": 0,
            "sampler_cursor": 0,
            "_buffered_inputs": None,
            "_rlmf_sampler_state": None,
            "_rlmf_metacognition_scorer": scorer,
        },
    )()

    rlmf_training._restore_custom_restart_state(record, trainer)

    assert trainer._step == 2
    assert trainer.sampler_cursor == 7
    assert trainer._rlmf_sampler_state == {"epoch": 3, "cursor": 7}
    assert torch.equal(trainer._buffered_inputs[0]["prompt_ids"], torch.tensor([[1, 2]]))
    assert trail.restored == {"record_count": 4, "last_sequence": 3}


def test_exact_resume_rejects_inconsistent_custom_state(tmp_path):
    _, record = _restartable_checkpoint(tmp_path, inconsistent=True)
    trainer = type("Trainer", (), {})()

    with pytest.raises(ValueError, match="global_step"):
        rlmf_training._restore_custom_restart_state(record, trainer)


def _record(config, *, stage, arm, seed, step, completed, pre_sft=None):
    parents = {"config": config.config_hash}
    if pre_sft is not None:
        parents["pre_sft"] = pre_sft
    return rlmf_training.CheckpointRecord.create(
        study_id=config.study_id,
        stage=stage,
        arm=arm,
        seed=seed,
        global_step=step,
        micro_step=step * 2,
        sampler_cursor=step * 2,
        files={"trainer_state.json": "f" * 64},
        parent_hashes=parents,
        path=f"/tmp/{stage}-{arm}-{seed}-{step}",
        completed=completed,
    )


def test_resume_selection_rejects_self_consistent_unrelated_config_and_parent():
    config = smoke_config()
    unrelated_config = RLMFConfig.from_json(CONFIGS / "rlmf_qwen06b_confirmatory.json")
    wrong_config = _record(
        unrelated_config,
        stage="rl", arm="standard_grpo", seed=11, step=1, completed=False,
        pre_sft="a" * 64,
    )
    wrong_parent = _record(
        config,
        stage="rl", arm="standard_grpo", seed=11, step=1, completed=False,
        pre_sft="b" * 64,
    )

    with pytest.raises(ValueError, match="study|config"):
        rlmf_training._select_resume_checkpoint(
            [wrong_config], config, stage="rl", arm="standard_grpo", seed=11,
            expected_pre_sft_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="pre-SFT parent"):
        rlmf_training._select_resume_checkpoint(
            [wrong_parent], config, stage="rl", arm="standard_grpo", seed=11,
            expected_pre_sft_hash="a" * 64,
        )


def test_canonical_pre_sft_parent_is_unique_and_uses_best_model_checkpoint(tmp_path):
    config = smoke_config()
    output = tmp_path / "working"
    best = output / "checkpoint-1"
    latest = output / "checkpoint-2"
    best.mkdir(parents=True)
    latest.mkdir()
    trainer = type(
        "Trainer", (),
        {"state": type("State", (), {"best_model_checkpoint": str(best)})()},
    )()

    assert rlmf_training._canonical_pre_sft_source(trainer, output) == best

    canonical = _record(
        config, stage="pre_sft", arm=None, seed=None, step=1, completed=True
    )
    assert rlmf_training._select_canonical_pre_sft_parent([canonical], config) == canonical
    duplicate = _record(
        config, stage="pre_sft", arm=None, seed=None, step=2, completed=True
    )
    with pytest.raises(ValueError, match="exactly one canonical"):
        rlmf_training._select_canonical_pre_sft_parent([canonical, duplicate], config)


def test_confirmatory_infrastructure_pilot_preserves_hyperparameters_and_stops_incomplete(
    tmp_path, monkeypatch
):
    config = RLMFConfig.from_json(CONFIGS / "rlmf_qwen06b_confirmatory.json")
    captured = {}

    def run_stage(*args, **kwargs):
        captured.update(kwargs)
        return "pilot-record"

    monkeypatch.setattr(rlmf_training, "_run_stage", run_stage)
    record = rlmf_training.run_rl_arm(
        config,
        "standard_grpo",
        11,
        ("example",),
        RLMFArtifactStore(tmp_path),
        resume=False,
        stop_after_step=25,
        infrastructure_pilot=True,
    )

    assert record == "pilot-record"
    assert captured["stop_after_step"] == 25
    assert captured["infrastructure_pilot"] is True
    assert rl_training_parameters(config, seed=11, stop_after_step=25) == (
        rl_training_parameters(config, seed=11)
    )
    with pytest.raises(ValueError, match="explicit infrastructure pilot"):
        rlmf_training.run_rl_arm(
            config,
            "standard_grpo",
            11,
            (),
            RLMFArtifactStore(tmp_path),
            resume=False,
            stop_after_step=25,
        )
    with pytest.raises(ValueError, match="step 25"):
        rlmf_training.run_rl_arm(
            config,
            "standard_grpo",
            11,
            (),
            RLMFArtifactStore(tmp_path),
            resume=False,
            stop_after_step=24,
            infrastructure_pilot=True,
        )


def test_pilot_stop_callback_requests_stop_and_seals_incomplete(
    tmp_path, monkeypatch
):
    config = RLMFConfig.from_json(CONFIGS / "rlmf_qwen06b_confirmatory.json")
    callbacks = []
    trainer = type(
        "Trainer", (),
        {
            "add_callback": callbacks.append,
            "_step": 100,
            "sampler_cursor": 100,
        },
    )()
    rlmf_training._attach_stop_after_step(trainer, 25)
    stop_callback = callbacks[0]
    control = type(
        "Control", (), {"should_training_stop": False, "should_save": False}
    )()
    state = type("State", (), {"global_step": 25})()

    returned = stop_callback.on_step_end(object(), state, control)

    assert returned is control
    assert control.should_training_stop is True
    assert control.should_save is True

    captured = {}
    monkeypatch.setattr(
        rlmf_training,
        "seal_checkpoint",
        lambda *args, **kwargs: captured.update(kwargs) or "sealed-record",
    )
    output_dir = tmp_path / "working"
    (output_dir / "checkpoint-25").mkdir(parents=True)
    sealer = rlmf_training._attach_checkpoint_sealer(
        trainer,
        RLMFArtifactStore(tmp_path / "store"),
        config,
        stage="rl",
        arm="standard_grpo",
        seed=11,
        parent_hashes={"pre_sft": "a" * 64},
        stop_after_step=25,
    )
    sealer.on_save(
        type("Args", (), {"output_dir": str(output_dir)})(),
        state,
        control,
    )
    assert captured["completed"] is False
    assert sealer.last_record == "sealed-record"


def test_training_audit_trail_survives_crash_and_restores_append_state(tmp_path):
    config = smoke_config()
    store = RLMFArtifactStore(tmp_path / "store")
    trail = rlmf_training.TrainingAuditTrail(
        store, config, "standard_grpo", 11
    )
    trail.record_raw(
        {
            "kind": "answer", "step": 0, "example_id": "pre-0",
            "candidate_id": "pre-0-step-0-member-0", "raw_output": "answer",
        }
    )
    checkpoint_state = trail.state()

    restarted = rlmf_training.TrainingAuditTrail(
        RLMFArtifactStore(tmp_path / "store"), config, "standard_grpo", 11
    )
    restarted.restore_state(checkpoint_state)
    restarted.record_raw(
        {
            "kind": "metacognition", "step": 0, "example_id": "pre-0",
            "candidate_id": "pre-0-step-0-member-0", "raw_output": "meta",
        }
    )

    path = store.directory_path(
        config.study_id, "training_audit", "raw-standard_grpo-seed-11"
    ).with_suffix(".jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == [0, 1]
    assert [row["kind"] for row in rows] == ["answer", "metacognition"]
    assert checkpoint_state["raw"]["record_count"] == 1
    assert checkpoint_state["raw"]["byte_size"] < path.stat().st_size
