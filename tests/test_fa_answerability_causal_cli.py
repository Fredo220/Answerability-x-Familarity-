import argparse
import json
from pathlib import Path

import pytest
from trajectory_extractor.fa_answerability_causal_runtime import vector_audit_hashes
from trajectory_extractor.fa_answerability_causal_analysis import (
    BOOTSTRAP_SEED,
    PERMUTATION_SEED,
)

from trajectory_extractor.fa_answerability_causal_cli import (
    AtomicJSONReceiptStore,
    CausalDependencies,
    CausalTokenizerBinding,
    InterventionRequest,
    RunObservation,
    RuntimeReceipt,
    _load_evaluation_seal,
    _load_prepared,
    _verify_shard_runtime_evidence,
    evaluate_causal,
    expected_causal_shards,
    load_causal_config,
    prepare_causal,
    run_causal_shard,
    run_causal_validation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "familiarity_answerability_causal_pilot_v1.json"
V3_CORPUS = (
    ROOT
    / "release"
    / "familiarity_answerability"
    / "representation_replication_v3"
    / "same_string_replication_v3_manifest.json"
)
V3_TRAIN_ACTIVATIONS = (
    ROOT
    / "release"
    / "familiarity_answerability"
    / "representation_replication_v3"
    / "activations"
    / "activations-representation_train.manifest.json"
)


class FakeCausalTokenizer:
    name_or_path = "google/gemma-2-2b-it"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<bos>{messages[0]['content']}<assistant>"

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": list(text.encode("utf-8"))}


def tokenizer_binding(config):
    return CausalTokenizerBinding(
        tokenizer=FakeCausalTokenizer(),
        model_id=config.model_id,
        model_revision=config.model_revision,
        tokenizer_id=config.model_id,
        tokenizer_revision=config.tokenizer_revision,
        chat_template_sha256=config.chat_template_sha256,
    )


def test_load_causal_config_rejects_changed_registered_grid(tmp_path):
    config = load_causal_config(CONFIG)
    assert config.study_id == "same-string-answerability-causal-pilot-v1"
    assert len(config.config_sha256) == 64
    assert config.statistics["bootstrap_seed"] == BOOTSTRAP_SEED == 20260804
    assert config.statistics["sign_flip_seed"] == PERMUTATION_SEED == 20260804

    changed = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed["validation_selection"]["multipliers"] = [0.25, 0.5]
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="registered causal config"):
        load_causal_config(changed_path)


def test_prepare_binds_all_identities_without_constructing_a_model(tmp_path):
    def no_model(_config):
        raise AssertionError("prepare must not construct a model runner")

    args = argparse.Namespace(
        config=str(CONFIG),
        root=str(tmp_path),
        v3_corpus_manifest=str(V3_CORPUS),
        v3_training_activation_manifest=str(V3_TRAIN_ACTIVATIONS),
        output_dir="prepared",
    )
    result = prepare_causal(
        args,
        dependencies=CausalDependencies(
            tokenizer_loader=tokenizer_binding,
            runner_factory=no_model,
        ),
    )

    manifest_path = Path(result["prepare_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["status"] == "prepared"
    assert payload["kind"] == "causal_pre_outcome_identity_seal"
    assert payload["audit"]["passed"] is True
    assert payload["selection_sha256"] == "0" * 64
    for field in (
        "config_sha256",
        "implementation_sha256",
        "model_sha256",
        "tokenizer_sha256",
        "corpus_sha256",
        "direction_bundle_sha256",
        "selection_sha256",
        "runtime_sha256",
        "split_sha256",
        "control_sha256",
        "request_sha256",
        "manifest_sha256",
    ):
        assert len(payload[field]) == 64
    assert Path(payload["causal_corpus_manifest"]).is_file()
    assert Path(payload["direction_bundle"]).is_file()


def test_atomic_receipt_resumes_only_the_same_request_hash(tmp_path):
    store = AtomicJSONReceiptStore(tmp_path)
    first = store.write_or_resume(
        "split/control/unit.json",
        request_sha256="1" * 64,
        payload={"value": 7},
        resume=True,
    )
    resumed = store.write_or_resume(
        "split/control/unit.json",
        request_sha256="1" * 64,
        payload={"value": 999},
        resume=True,
    )

    assert first.resumed is False
    assert resumed.resumed is True
    assert resumed.payload["value"] == 7
    assert not tuple(tmp_path.rglob("*.tmp"))

    with pytest.raises(ValueError, match="request hash does not match"):
        store.write_or_resume(
            "split/control/unit.json",
            request_sha256="2" * 64,
            payload={"value": 7},
            resume=True,
        )


def test_runtime_identity_hash_excludes_observed_peak_memory():
    base = RuntimeReceipt(
        loader_mode="cuda_4bit",
        device="cuda:0",
        dtype="torch.float16",
        peak_memory_bytes=2_000_000,
        memory_limit_bytes=16_000_000_000,
        quantization="bitsandbytes_nf4",
        batch_size=1,
        gradients_enabled=False,
        smoke_request_sha256="1" * 64,
    )
    repeated = RuntimeReceipt(
        **{**base.__dict__, "peak_memory_bytes": 2_500_000}
    )

    assert base.runtime_sha256 == repeated.runtime_sha256


class FakeCausalRunner:
    def __init__(self):
        self.observe_calls = 0
        self.smoke_calls = 0

    def smoke(self, prompt, request):
        self.smoke_calls += 1
        assert prompt.split == "causal_validation"
        assert isinstance(request, InterventionRequest)
        return RuntimeReceipt(
            loader_mode="registered_full_precision",
            device="cpu",
            dtype="torch.float32",
            peak_memory_bytes=2_000_000,
            memory_limit_bytes=8_000_000_000,
            quantization="none",
            batch_size=1,
            gradients_enabled=False,
            smoke_request_sha256=request.request_sha256,
        )

    def runtime_identity(self, request_sha256):
        return RuntimeReceipt(
            loader_mode="registered_full_precision",
            device="cpu",
            dtype="torch.float32",
            peak_memory_bytes=2_000_000,
            memory_limit_bytes=8_000_000_000,
            quantization="none",
            batch_size=1,
            gradients_enabled=False,
            smoke_request_sha256=request_sha256,
        )

    def observe(self, prompt, request):
        self.observe_calls += 1
        baseline = 1.0 if prompt.answerability == "target_bound" else -1.0
        effect = 0.0
        if request.sign:
            effect = request.sign * request.multiplier * (1.0 + request.layer_id / 100.0)
        generated = prompt.registry_code if prompt.answerability == "target_bound" else "UNKNOWN"
        hashes = (
            vector_audit_hashes(
                request.vector,
                represented_dtype="torch.float32",
            )
            if request.sign
            else None
        )
        margin_audit = (
            {
                "source_vector_sha256": hashes.source_vector_sha256,
                "applied_vector_sha256": hashes.applied_vector_sha256,
                "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
                "example_id": prompt.example_id,
                "hook_call_count": 1,
                "modified_site_count": 1,
                "hook_cleanup_verified": True,
            }
            if hashes is not None
            else None
        )
        return RunObservation(
            raw_margin=baseline + effect,
            length_normalized_margin=(baseline + effect) / 2.0,
            generated_text=generated,
            primary_projection_delta=effect,
            audit={
                "request_sha256": request.request_sha256,
                "example_id": prompt.example_id,
                "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
                "hook_call_count": 1 if request.sign else 0,
                "modified_site_count": 1 if request.sign else 0,
                "hook_cleanup_verified": True,
                "represented_device": "cpu",
                "represented_dtype": "torch.float32",
                "source_vector_sha256": (
                    hashes.source_vector_sha256 if hashes is not None else "0" * 64
                ),
                "applied_vector_sha256": (
                    hashes.applied_vector_sha256 if hashes is not None else "0" * 64
                ),
                "margin_forward_audits": (
                    [dict(margin_audit), dict(margin_audit)]
                    if margin_audit is not None
                    else []
                ),
            },
        )

    def unrelated_preservation(self, request):
        assert isinstance(request, InterventionRequest)
        return {
            "passed": True,
            "rows": [
                {
                    "prompt_id": f"unrelated-code-lookup-{index}",
                    "expected": f"U{index:04d}",
                    "generated": f"U{index:04d}",
                }
                for index in range(4)
            ],
        }


def test_validation_runs_the_complete_grid_and_seals_one_candidate(tmp_path):
    runner = FakeCausalRunner()
    dependencies = CausalDependencies(
        tokenizer_loader=tokenizer_binding,
        runner_factory=lambda _config: runner,
    )
    prepared = prepare_causal(
        argparse.Namespace(
            config=str(CONFIG),
            root=str(tmp_path),
            v3_corpus_manifest=str(V3_CORPUS),
            v3_training_activation_manifest=str(V3_TRAIN_ACTIVATIONS),
            output_dir="prepared",
        ),
        dependencies=dependencies,
    )
    result = run_causal_validation(
        argparse.Namespace(
            config=str(CONFIG),
            root=str(tmp_path),
            prepare_manifest=prepared["prepare_manifest"],
            output_dir="validation",
            resume=True,
        ),
        dependencies=dependencies,
    )

    selection = json.loads(Path(result["selection_manifest"]).read_text(encoding="utf-8"))
    seal = json.loads(Path(result["seal_manifest"]).read_text(encoding="utf-8"))
    raw = tuple((tmp_path / "validation" / "raw").glob("candidate-*.json"))
    assert runner.smoke_calls == 1
    assert len(raw) == 15
    assert (selection["layer_id"], selection["multiplier"]) == (25, 1.0)
    assert seal["selection_sha256"] == selection["selection_sha256"]
    assert seal["status"] == "selected_and_sealed"
    assert len(seal["seal_sha256"]) == 64
    assert set(seal["expected_unit_ids_by_split"]) == {
        "causal_entity_test",
        "causal_template_test",
    }


def test_shard_runs_one_registered_unit_and_resumes_without_model_work(tmp_path):
    runner = FakeCausalRunner()
    dependencies = CausalDependencies(
        tokenizer_loader=tokenizer_binding,
        runner_factory=lambda _config: runner,
    )
    prepared = prepare_causal(
        argparse.Namespace(
            config=str(CONFIG),
            root=str(tmp_path),
            v3_corpus_manifest=str(V3_CORPUS),
            v3_training_activation_manifest=str(V3_TRAIN_ACTIVATIONS),
            output_dir="prepared",
        ),
        dependencies=dependencies,
    )
    validation = run_causal_validation(
        argparse.Namespace(
            config=str(CONFIG),
            root=str(tmp_path),
            prepare_manifest=prepared["prepare_manifest"],
            output_dir="validation",
            resume=True,
        ),
        dependencies=dependencies,
    )
    seal = json.loads(Path(validation["seal_manifest"]).read_text(encoding="utf-8"))
    unit_id = seal["expected_unit_ids_by_split"]["causal_entity_test"][0]
    args = argparse.Namespace(
        config=str(CONFIG),
        root=str(tmp_path),
        prepare_manifest=prepared["prepare_manifest"],
        seal_manifest=validation["seal_manifest"],
        split="causal_entity_test",
        control="primary",
        unit_id=unit_id,
        member=None,
        output_dir="evidence",
        resume=True,
    )
    before = runner.observe_calls
    first = run_causal_shard(args, dependencies=dependencies)
    after_first = runner.observe_calls
    resumed = run_causal_shard(args, dependencies=dependencies)

    payload = json.loads(Path(first["shard_receipt"]).read_text(encoding="utf-8"))
    assert after_first - before == 6
    assert runner.observe_calls == after_first
    assert first["status"] == "completed"
    assert resumed["status"] == "resumed"
    assert len(payload["rows"]) == 4
    assert payload["split"] == "causal_entity_test"
    assert payload["control"] == "primary"
    assert payload["seal_sha256"] == seal["seal_sha256"]

    with pytest.raises(ValueError, match="only norm_matched_random"):
        run_causal_shard(
            argparse.Namespace(**{**vars(args), "member": 2}),
            dependencies=dependencies,
        )


@pytest.mark.parametrize("tamper", ["margin_audit", "prompt_id", "request_hash"])
def test_evaluator_rejects_tampered_runtime_or_prompt_evidence(tmp_path, tamper):
    runner = FakeCausalRunner()
    dependencies = CausalDependencies(
        tokenizer_loader=tokenizer_binding,
        runner_factory=lambda _config: runner,
    )
    prepared = prepare_causal(
        argparse.Namespace(
            config=str(CONFIG),
            root=str(tmp_path),
            v3_corpus_manifest=str(V3_CORPUS),
            v3_training_activation_manifest=str(V3_TRAIN_ACTIVATIONS),
            output_dir="prepared",
        ),
        dependencies=dependencies,
    )
    validation = run_causal_validation(
        argparse.Namespace(
            config=str(CONFIG),
            root=str(tmp_path),
            prepare_manifest=prepared["prepare_manifest"],
            output_dir="validation",
            resume=True,
        ),
        dependencies=dependencies,
    )
    seal_json = json.loads(Path(validation["seal_manifest"]).read_text())
    unit_id = seal_json["expected_unit_ids_by_split"]["causal_entity_test"][0]
    shard = run_causal_shard(
        argparse.Namespace(
            config=str(CONFIG),
            root=str(tmp_path),
            prepare_manifest=prepared["prepare_manifest"],
            seal_manifest=validation["seal_manifest"],
            split="causal_entity_test",
            control="primary",
            unit_id=unit_id,
            member=None,
            output_dir="evidence",
            resume=True,
        ),
        dependencies=dependencies,
    )
    payload = json.loads(Path(shard["shard_receipt"]).read_text())
    if tamper == "margin_audit":
        payload["rows"][0]["observation"]["audit"]["margin_forward_audits"][0][
            "applied_vector_sha256"
        ] = "9" * 64
    elif tamper == "prompt_id":
        payload["rows"][0]["example_id"] = "wrong-example"
    else:
        payload["request_sha256"] = "9" * 64

    load_args = argparse.Namespace(
        config=str(CONFIG),
        root=str(tmp_path),
        prepare_manifest=prepared["prepare_manifest"],
    )
    _config, prepare, corpus, bundle, _binding = _load_prepared(
        load_args, dependencies
    )
    seal, _record = _load_evaluation_seal(
        validation["seal_manifest"], prepare=prepare
    )
    with pytest.raises(ValueError):
        _verify_shard_runtime_evidence(
            payload,
            prepare=prepare,
            seal=seal,
            corpus=corpus,
            bundle=bundle,
        )


def test_evaluate_requires_every_registered_shard_before_endpoint_state(tmp_path):
    runner = FakeCausalRunner()
    dependencies = CausalDependencies(
        tokenizer_loader=tokenizer_binding,
        runner_factory=lambda _config: runner,
    )
    prepared = prepare_causal(
        argparse.Namespace(
            config=str(CONFIG),
            root=str(tmp_path),
            v3_corpus_manifest=str(V3_CORPUS),
            v3_training_activation_manifest=str(V3_TRAIN_ACTIVATIONS),
            output_dir="prepared",
        ),
        dependencies=dependencies,
    )
    validation = run_causal_validation(
        argparse.Namespace(
            config=str(CONFIG),
            root=str(tmp_path),
            prepare_manifest=prepared["prepare_manifest"],
            output_dir="validation",
            resume=True,
        ),
        dependencies=dependencies,
    )
    seal = json.loads(Path(validation["seal_manifest"]).read_text(encoding="utf-8"))
    assert len(expected_causal_shards(seal)) == 432

    with pytest.raises(ValueError, match="all registered shards"):
        evaluate_causal(
            argparse.Namespace(
                config=str(CONFIG),
                root=str(tmp_path),
                prepare_manifest=prepared["prepare_manifest"],
                seal_manifest=validation["seal_manifest"],
                evidence_dir="evidence",
                output_dir="results",
            ),
            dependencies=dependencies,
        )

    assert not (tmp_path / "results" / "endpoint_state.json").exists()
    assert not (tmp_path / "results" / "result.json").exists()
