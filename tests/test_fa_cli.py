import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trajectory_extractor import cli
from trajectory_extractor.fa_artifacts import FAArtifactStore, UnlockReceipt
import trajectory_extractor.fa_cli as fa_cli
import trajectory_extractor.fa_probes as fa_probes
from trajectory_extractor.fa_cli import dispatch_fa, register_fa_subcommands
from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_data import (
    CONFIRMATORY_POWER_SIMULATIONS,
    REGISTERED_POWER_GRID,
    PowerAudit,
    PowerCell,
    build_factorial_examples,
)
from trajectory_extractor.fa_entities import EntityMatch, NaturalnessRating
from trajectory_extractor.fa_runtime import run_generation_shard
from trajectory_extractor.fa_probes import (
    OUTPUT_CONTROL_SCHEMA_SHA256,
    ProbeRow,
    ProbeTestAuthorization,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "familiarity_answerability_qwen06b_smoke.json"
)

MATCH = {
    "pair_id": "Q1--syn-1",
    "real_entity_id": "Q1",
    "real_qid": "Q1",
    "synthetic_candidate_id": "syn-1",
    "real_name": "Old Vale",
    "synthetic_name": "New Vale",
    "coarse_type": "place",
    "split": "pilot",
    "generator_revision": "names-v1",
    "tokenizer_revision": "tokenizer-v1",
    "real_token_count": 2,
    "synthetic_token_count": 2,
    "real_word_count": 2,
    "synthetic_word_count": 2,
    "real_character_count": 8,
    "synthetic_character_count": 8,
    "character_length_delta": 0,
    "character_tolerance": 2,
    "capitalization_pattern_equal": True,
}


class FakeTokenizer:
    chat_template = "fake qwen template"
    all_special_ids = ()

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        rendered = messages[0]["content"] + " <assistant>"
        return self.encode(rendered) if tokenize else rendered


CHAT_TEMPLATE_BYTES = FakeTokenizer.chat_template.encode("utf-8")
CHAT_TEMPLATE_SHA256 = hashlib.sha256(CHAT_TEMPLATE_BYTES).hexdigest()


@pytest.fixture(autouse=True)
def register_fake_smoke_template(monkeypatch):
    monkeypatch.setattr(fa_cli, "_SMOKE_CHAT_TEMPLATE_SHA256", CHAT_TEMPLATE_SHA256)


def install_fake_tokenizer(monkeypatch):
    monkeypatch.setattr(
        fa_cli,
        "_TOKENIZER_LOADER",
        lambda model_id, *, revision: FakeTokenizer(),
    )


def sha256_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def probe_rows_for_examples(examples):
    rows = []
    domains = ("person", "place", "organization", "creative_work")
    for index, example in enumerate(sorted(examples, key=lambda row: row.example_id)):
        residual = np.zeros((3, 26, 4), dtype=np.float64)
        residual[:, :, 0] = float(index % 2)
        for task in ("familiarity", "answerability", "unsupported_answer"):
            if task == "unsupported_answer" and example.answerability == "target_bound":
                continue
            if task == "familiarity":
                label = int(example.target_familiarity == "screened_real")
            elif task == "answerability":
                label = example.answerability
            else:
                label = index % 2
            rows.append(
                ProbeRow(
                    example_id=example.example_id,
                    split=example.split,
                    task=task,
                    label=label,
                    entity_id=f"{example.split}-entity-{index}",
                    template_id=f"{example.split}-template-{index}",
                    relation_id=f"{example.split}-relation-{index}",
                    domain=domains[index % len(domains)],
                    condition=f"condition-{index}",
                    answerability_condition=example.answerability,
                    target_familiarity_condition=example.target_familiarity,
                    distractor_familiarity_condition=example.distractor_familiarity,
                    surface_features=(float(index),),
                    output_margin_features=tuple(float(index) for _ in range(11)),
                    residual_features=residual,
                    sae_features=None,
                    outcome_status="valid",
                    source_sha256=example.canonical_payload_sha256,
                    activation_sha256=sha256_json(
                        {"activation": example.example_id}
                    ),
                    metadata_manifest_sha256=sha256_json(
                        {"metadata": example.split}
                    ),
                    metadata_row_sha256=sha256_json(
                        {"metadata-row": example.example_id}
                    ),
                    output_control_schema_sha256=OUTPUT_CONTROL_SCHEMA_SHA256,
                    output_evidence_sha256=sha256_json(
                        {"output": example.example_id}
                    ),
                )
            )
    return tuple(rows)


def write_probe_rows_artifact(root, config, split, rows, *, lineage=None):
    return FAArtifactStore(root).write_completed_shard(
        config.run_id,
        split,
        f"probe-rows-{split}",
        [{"kind": "probe_rows", "row": row.to_record()} for row in rows],
        {"config_sha256": config.config_hash, **(lineage or {})},
        record_kind="probe_rows",
    )


def naturalness_ratings_manifest(root, config, *, rating_schema_version=1):
    store = FAArtifactStore(root)
    ratings = [
        asdict(
            NaturalnessRating(
                pair_id=MATCH["pair_id"],
                rater_id=rater_id,
                real_naturalness=4,
                synthetic_naturalness=4,
                real_type_fit=4,
                synthetic_type_fit=4,
                synthetic_malformed=False,
            )
        )
        for rater_id in ("rater-a", "rater-b")
    ]
    ratings[0]["schema_version"] = rating_schema_version
    assignments = [
        {
            "pair_id": MATCH["pair_id"],
            "rater_id": rater_id,
            "blind_slot": blind_slot,
            "submission_sha256": sha256_json(
                {"pair_id": MATCH["pair_id"], "rater_id": rater_id}
            ),
        }
        for rater_id, blind_slot in (
            ("rater-a", "slot-a"),
            ("rater-b", "slot-b"),
        )
    ]
    preregistration = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "familiarity_answerability_preregistration.md"
    )
    protocol_sha256 = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    blinding_sha256 = sha256_json(assignments)
    row = {
        "kind": "naturalness_ratings",
        "schema_version": 1,
        "config_sha256": config.config_hash,
        "protocol_sha256": protocol_sha256,
        "blinding_manifest_sha256": blinding_sha256,
        "assignments": assignments,
        "ratings": ratings,
    }
    shard = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        f"ratings-{rating_schema_version}",
        [row],
        {
            "config_sha256": config.config_hash,
            "protocol_sha256": protocol_sha256,
            "blinding_manifest_sha256": blinding_sha256,
        },
        record_kind="naturalness_ratings",
    )
    return shard.manifest_path


def valid_examples(config, split):
    match = dict(MATCH)
    match["split"] = split
    return build_factorial_examples(
        config, (EntityMatch(**match),), tokenizer=FakeTokenizer()
    )


def prompt_capability(
    root, config, split, *, full_hash="b" * 64, template_bytes=CHAT_TEMPLATE_BYTES
):
    examples = valid_examples(config, split)
    store = FAArtifactStore(root)
    template_hash = hashlib.sha256(template_bytes).hexdigest()
    prepared = SimpleNamespace(
        chat_template_bytes=template_bytes,
        chat_template_sha256=template_hash,
    )
    try:
        tokenizer_pin = fa_cli._write_tokenizer_pin(
            store, config, prepared, full_hash
        )
    except FileExistsError:
        tokenizer_pin = store.verify_shard(
            store.root
            / "runs"
            / "familiarity_answerability"
            / config.run_id
            / "shards"
            / ("mechanism_train" if config.profile == "confirmatory" else "pilot")
            / f"tokenizer-pin-{template_hash[:16]}.jsonl.manifest.json"
        )
    shard = fa_cli._write_prompt_capability(
        store,
        config,
        full_hash,
        split,
        examples,
        template_hash,
        tokenizer_pin,
    )
    return examples, shard


def test_fa_commands_are_registered_with_explicit_config_and_root(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    matches = tmp_path / "matches.json"
    matches.write_text(json.dumps([MATCH]), encoding="utf-8")
    args = parser.parse_args(
        [
            "fa-build-pilot",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--matches-manifest",
            str(matches),
        ]
    )

    assert args.command == "fa-build-pilot"
    assert args.config == str(CONFIG_PATH)
    assert args.root == str(tmp_path)


def test_generation_parser_accepts_explicit_resume_flag(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    args = parser.parse_args(
        [
            "fa-run-generation",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "prompts.jsonl.manifest.json"),
            "--shard-id",
            "0001",
            "--namespace",
            "pilot",
            "--resume",
        ]
    )

    assert args.resume is True


def test_activation_parser_requires_explicit_manifest_namespace_and_shard(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    args = parser.parse_args(
        [
            "fa-extract-activations",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "prompts.jsonl.manifest.json"),
            "--namespace",
            "pilot",
            "--shard-id",
            "0001",
            "--layers",
            "0,4,8",
            "--resume",
        ]
    )

    assert args.namespace == "pilot"
    assert args.shard_id == "0001"
    assert args.layers == "0,4,8"
    assert args.resume is True


def test_activation_cli_wires_verified_prompt_to_resumable_writer(
    tmp_path, capsys, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, "pilot")
    calls = {}

    class FakeModelRunner:
        def __init__(self, supplied):
            assert supplied == config
            self.model = object()
            self.tokenizer = object()
            self.model_id = supplied.model_id
            self.model_revision = supplied.model_revision
            self.tokenizer_revision = supplied.tokenizer_revision
            self.chat_template_sha256 = CHAT_TEMPLATE_SHA256

        def generate(self, prompts, generation):
            raise AssertionError("activation extraction must not generate completions")

    class FakeSelectedRunner:
        def __init__(self, model, tokenizer, **pins):
            calls["runner"] = (model, tokenizer, pins)

    def fake_write(runner, examples, registered_layers, *, destination):
        calls["write"] = (runner, tuple(examples), tuple(registered_layers), destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        for path in (
            destination,
            destination.with_suffix(".jsonl"),
            destination.with_suffix(".manifest.json"),
        ):
            path.write_bytes(b"sealed")
        return SimpleNamespace(
            manifest_path=destination.with_suffix(".manifest.json"),
            request_sha256="a" * 64,
            row_count=len(examples),
        )

    monkeypatch.setattr(fa_cli, "HFModelRunner", FakeModelRunner)
    monkeypatch.setattr(fa_cli, "_ACTIVATION_RUNNER_FACTORY", FakeSelectedRunner)
    monkeypatch.setattr(fa_cli, "_ACTIVATION_SHARD_WRITER", fake_write)

    exit_code = cli.main(
        [
            "fa-extract-activations",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--namespace",
            "pilot",
            "--shard-id",
            "0001",
            "--layers",
            "0,4,8",
            "--resume",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "extracted"
    assert payload["request_sha256"] == "a" * 64
    assert calls["write"][2] == (0, 4, 8)
    assert calls["write"][3] == (
        tmp_path
        / "runs"
        / "familiarity_answerability"
        / config.run_id
        / "activations"
        / "pilot"
        / "0001.npz"
    )


def test_generic_activation_cli_rejects_protected_namespace_before_model_load(
    tmp_path, capsys, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, "probe_test")

    class MustNotConstruct:
        def __init__(self, _config):
            raise AssertionError("protected extraction must use its endpoint transaction")

    monkeypatch.setattr(fa_cli, "HFModelRunner", MustNotConstruct)
    exit_code = cli.main(
        [
            "fa-extract-activations",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--namespace",
            "probe_test",
            "--shard-id",
            "0001",
            "--layers",
            "0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "error"
    assert "protected test namespaces" in payload["error"]["message"]


def test_behavior_test_command_closes_one_use_endpoint_with_canonical_metrics(
    tmp_path, capsys, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, "behavior_test")
    preregistration_hash = hashlib.sha256(
        (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "familiarity_answerability_preregistration.md"
        ).read_bytes()
    ).hexdigest()
    selection_hash = "d" * 64
    store = FAArtifactStore(tmp_path)
    store.seal_endpoint(
        "behavior_test",
        (prompts,),
        {
            "preregistration": preregistration_hash,
            "selection_manifest": selection_hash,
        },
    )

    class CanonicalRecord:
        def __init__(self, value):
            self.value = value

        def to_record(self):
            return dict(self.value)

    monkeypatch.setattr(fa_cli, "HFModelRunner", FakeRunner)
    monkeypatch.setattr(
        fa_cli,
        "_BEHAVIOR_BOOTSTRAP",
        lambda rows, replicates, seed: CanonicalRecord(
            {"replicates": replicates, "rows": len(rows), "seed": seed}
        ),
    )
    monkeypatch.setattr(
        fa_cli,
        "_BEHAVIOR_GATE",
        lambda metrics, bootstrap, **kwargs: CanonicalRecord(
            {"status": "not_evaluable", "config_hash": kwargs["config_hash"]}
        ),
    )

    exit_code = cli.main(
        [
            "fa-evaluate-behavior-test",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--shard-id",
            "confirmatory-0001",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "evaluated"
    assert payload["endpoint_state"] == "closed"
    assert store.endpoint_state("behavior_test", prompts.manifest_path) == "closed"
    metrics = store.verify_shard(payload["metrics_manifest"])
    assert metrics.record_kind == "metrics"
    assert metrics.namespace == "behavior_test"


def test_fa_dispatch_is_isolated_and_cli_routes_fa_commands(tmp_path, capsys, monkeypatch):
    install_fake_tokenizer(monkeypatch)
    args = argparse.Namespace(command="rlmf-prepare-data")
    assert dispatch_fa(args) is None

    matches = tmp_path / "matches.json"
    matches.write_text(json.dumps([MATCH]), encoding="utf-8")
    exit_code = cli.main(
        [
            "fa-build-pilot",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--matches-manifest",
            str(matches),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["command"] == "fa-build-pilot"
    assert payload["status"] == "built"


def test_pilot_prompt_capability_references_verified_tokenizer_pin(tmp_path, monkeypatch):
    install_fake_tokenizer(monkeypatch)
    config = FAConfig.from_json(CONFIG_PATH)
    matches = tmp_path / "matches.json"
    matches.write_text(json.dumps([MATCH]), encoding="utf-8")

    payload = fa_cli._build_manifest(
        config,
        tmp_path,
        SimpleNamespace(matches_manifest=matches),
        confirmatory=False,
    )
    store = FAArtifactStore(tmp_path)
    prompt = store.verify_shard(payload["manifest"])
    pin = store.verify_shard(payload["tokenizer_pin_manifest"])
    prompt_row = fa_cli._read_json_rows(prompt.data_path)[0]
    prompt_lineage = json.loads(prompt.manifest_path.read_text(encoding="utf-8"))[
        "lineage"
    ]
    pin_row = fa_cli._read_json_rows(pin.data_path)[0]

    assert prompt_row["tokenizer_pin_manifest"] == str(
        pin.manifest_path.relative_to(store.root)
    )
    assert prompt_row["tokenizer_pin_sha256"] == pin.sha256
    assert prompt_lineage["tokenizer_pin_sha256"] == pin.sha256
    assert hashlib.sha256(bytes.fromhex(pin_row["chat_template_utf8_hex"])).hexdigest() == (
        pin_row["chat_template_sha256"]
    )


def test_probe_selection_reads_only_hash_bound_prompt_identities(
    tmp_path, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    examples, prompt = prompt_capability(tmp_path, config, "probe_test")

    def forbidden_prompt_payload_read(*args, **kwargs):
        raise AssertionError("selection must not parse protected prompt payloads")

    monkeypatch.setattr(fa_cli, "_read_json_rows", forbidden_prompt_payload_read)
    identities, verified = fa_cli._load_prompt_source_identities(
        FAArtifactStore(tmp_path),
        prompt.manifest_path,
        config,
        expected_namespace="probe_test",
    )

    assert verified.sha256 == prompt.sha256
    assert [identity.example_id for identity in identities] == sorted(
        example.example_id for example in examples
    )
    assert all(len(identity.canonical_payload_sha256) == 64 for identity in identities)


def test_protected_probe_rows_open_only_after_authorization_and_match_task_identities(
    tmp_path,
):
    config = FAConfig.from_json(CONFIG_PATH)
    examples, prompt = prompt_capability(tmp_path, config, "probe_test")
    store = FAArtifactStore(tmp_path)
    task_identities, _ = fa_cli._load_prompt_task_source_identities(
        store,
        prompt.manifest_path,
        config,
        expected_namespace="probe_test",
    )
    rows = probe_rows_for_examples(examples)
    probe_rows = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "probe-rows",
        [{"kind": "probe_rows", "row": row.to_record()} for row in rows],
        {
            "config_sha256": config.config_hash,
            "prompt_manifest_sha256": prompt.sha256,
            "task_source_identities_sha256": fa_cli._task_source_identities_sha256(
                task_identities
            ),
        },
        record_kind="probe_rows",
    )

    with pytest.raises(ValueError, match="require a probe_test authorization"):
        fa_cli._load_probe_rows_manifest(
            store,
            probe_rows.manifest_path,
            config,
            expected_namespace="probe_test",
        )

    authorization = ProbeTestAuthorization.from_unlock_receipt(
        UnlockReceipt(
            endpoint="probe_test",
            lease_id="a" * 32,
            state="unlocked_once",
            preregistration_hash="b" * 64,
            selection_manifest_hash="c" * 64,
        )
    )
    loaded, verified = fa_cli._load_probe_rows_manifest(
        store,
        probe_rows.manifest_path,
        config,
        expected_namespace="probe_test",
        authorization=authorization,
        expected_prompt_sha256=prompt.sha256,
        expected_task_identities=task_identities,
    )

    assert verified.sha256 == probe_rows.sha256
    assert [row.to_record() for row in loaded] == [row.to_record() for row in rows]


def test_f2a_cli_selects_seals_evaluates_and_recovers_atomically(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fa_probes, "_TRAIN_ONLY_CV_FAST_PATH_FOR_TESTS", True)
    monkeypatch.setattr(fa_probes, "_BOOTSTRAP_DRAW_OVERRIDE_FOR_TESTS", 80)
    config = FAConfig.from_json(CONFIG_PATH)
    train = write_probe_rows_artifact(
        tmp_path,
        config,
        "mechanism_train",
        probe_rows_for_examples(valid_examples(config, "mechanism_train")),
    )
    validation = write_probe_rows_artifact(
        tmp_path,
        config,
        "locked_validation",
        probe_rows_for_examples(valid_examples(config, "locked_validation")),
    )
    test_examples, prompt = prompt_capability(tmp_path, config, "probe_test")

    def smoke_nulls(
        train_rows,
        validation_rows,
        *,
        seeds,
        protected_test_ids,
        probe_test_source_identities,
        _allow_test_seed_override,
    ):
        del protected_test_ids
        base = fa_probes.fit_selection(train_rows, validation_rows)
        results = []
        for kind in (
            "label_permutation",
            "layer_order",
            "random_map",
            "output_aligned_11d",
        ):
            seed = seeds[0]
            provenance = {"kind": kind, "seed": seed, "config": {"smoke": True}}
            selection = replace(base, null_provenance=provenance)
            frozen = {"kind": kind, "seed": seed, "transform": {"smoke": True}}
            results.append(
                fa_probes.NullSelectionResult(
                    kind=kind,
                    seed=seed,
                    config=frozen,
                    config_sha256=sha256_json(frozen),
                    selection=selection,
                    max_norm_error=0.0,
                    test_source_identities=tuple(probe_test_source_identities),
                    test_transform={
                        "seed": seed,
                        "row_count": len(probe_test_source_identities),
                    },
                )
            )
        return tuple(results)

    monkeypatch.setattr(fa_cli, "_PROBE_NULL_SELECTOR", smoke_nulls)
    payload = fa_cli._fit_probes(
        config,
        tmp_path,
        SimpleNamespace(
            train_rows_manifest=train.manifest_path,
            validation_rows_manifest=validation.manifest_path,
            probe_test_manifest=prompt.manifest_path,
            shard_id="selection-smoke",
        ),
    )
    bundle = fa_cli._load_f2a_selection_bundle(
        FAArtifactStore(tmp_path), payload["selection_manifest"], config
    )

    assert payload["status"] == "selected"
    assert bundle.selection_bundle_hash == payload["selection_bundle_hash"]
    assert all(len(bundle.null_selections[task]) == 4 for task in fa_probes.TASKS)

    task_identities, _ = fa_cli._load_prompt_task_source_identities(
        FAArtifactStore(tmp_path),
        prompt.manifest_path,
        config,
        expected_namespace="probe_test",
    )
    test_rows = write_probe_rows_artifact(
        tmp_path,
        config,
        "probe_test",
        probe_rows_for_examples(test_examples),
        lineage={
            "prompt_manifest_sha256": prompt.sha256,
            "task_source_identities_sha256": fa_cli._task_source_identities_sha256(
                task_identities
            ),
        },
    )
    sealed = fa_cli._seal_probe_selection(
        config,
        tmp_path,
        SimpleNamespace(
            selection_manifest=payload["selection_manifest"],
            probe_test_manifest=prompt.manifest_path,
            probe_rows_manifest=test_rows.manifest_path,
        ),
    )
    evaluated = fa_cli._evaluate_probe_test(
        config,
        tmp_path,
        SimpleNamespace(
            selection_manifest=payload["selection_manifest"],
            probe_test_manifest=prompt.manifest_path,
            probe_rows_manifest=test_rows.manifest_path,
        ),
    )
    recovered = fa_cli._evaluate_probe_test(
        config,
        tmp_path,
        SimpleNamespace(
            selection_manifest=payload["selection_manifest"],
            probe_test_manifest=prompt.manifest_path,
            probe_rows_manifest=test_rows.manifest_path,
        ),
    )

    assert sealed["endpoint_state"] == "sealed"
    assert evaluated["status"] == "evaluated"
    assert evaluated["endpoint_state"] == "closed"
    assert recovered["status"] == "recovered"
    assert recovered["metrics_manifest"] == evaluated["metrics_manifest"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "attacker/model", "model identity"),
        ("model_revision", "0" * 40, "model identity"),
        ("tokenizer_revision", "1" * 40, "model identity"),
        ("chat_template_utf8_hex", b"alternate template".hex(), "claimed hash"),
    ],
)
def test_prompt_verifier_rejects_forged_tokenizer_pin_identity_or_template_bytes(
    tmp_path, field, value, message
):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompt = prompt_capability(tmp_path, config, "pilot")
    store = FAArtifactStore(tmp_path)
    prompt_row = fa_cli._read_json_rows(prompt.data_path)[0]
    prompt_lineage = json.loads(prompt.manifest_path.read_text(encoding="utf-8"))[
        "lineage"
    ]
    pin_path = store.root / prompt_row["tokenizer_pin_manifest"]
    pin = store.verify_shard(pin_path)
    pin_row = fa_cli._read_json_rows(pin.data_path)[0]
    pin_lineage = json.loads(pin.manifest_path.read_text(encoding="utf-8"))["lineage"]
    pin_row[field] = value
    forged_pin = store.write_completed_shard(
        config.run_id,
        "pilot",
        f"forged-pin-{field}",
        [pin_row],
        pin_lineage,
        record_kind="tokenizer_pin",
    )
    prompt_row["tokenizer_pin_manifest"] = str(
        forged_pin.manifest_path.relative_to(store.root)
    )
    prompt_row["tokenizer_pin_sha256"] = forged_pin.sha256
    prompt_row["subset_manifest_sha256"] = fa_cli._prompt_subset_sha256(
        prompt_row["config_hash"],
        prompt_row["full_manifest_sha256"],
        prompt_row["namespace"],
        prompt_row["chat_template_sha256"],
        forged_pin.sha256,
        None,
        tuple(prompt_row["examples"]),
    )
    prompt_lineage["subset_manifest_sha256"] = prompt_row[
        "subset_manifest_sha256"
    ]
    prompt_lineage["tokenizer_pin_sha256"] = forged_pin.sha256
    forged_prompt = store.write_completed_shard(
        config.run_id,
        "pilot",
        f"forged-prompt-{field}",
        [prompt_row],
        prompt_lineage,
        record_kind="prompt_manifest",
    )

    with pytest.raises(ValueError, match=message):
        fa_cli._load_manifest(store, forged_prompt.manifest_path, config)


def test_fa_commands_require_explicit_input_manifests_and_restrict_generation_namespaces(tmp_path, capsys):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(["fa-run-generation", "--config", str(CONFIG_PATH)])
    argument_error = json.loads(capsys.readouterr().out)
    assert argument_error["status"] == "error"
    assert argument_error["error"]["type"] == "ArgumentError"
    args = parser.parse_args(
        [
            "fa-run-generation",
            "--config",
            str(CONFIG_PATH),
            "--manifest",
                str(tmp_path / "examples.json"),
            "--shard-id",
            "0001",
            "--namespace",
            "behavior_test",
        ]
    )
    (tmp_path / "examples.json").write_text(
        json.dumps(
            {
                "config_hash": FAConfig.from_json(CONFIG_PATH).config_hash,
                "manifest_sha256": "a" * 64,
                "examples": [],
            }
        ),
        encoding="utf-8",
    )
    exit_code = dispatch_fa(args)
    payload = json.loads(capsys.readouterr().out)
    assert exit_code != 0
    assert payload == {
        "command": "fa-run-generation",
        "error": {
            "message": (
                "fa-run-generation is generic-only and cannot evaluate protected "
                "test namespaces"
            ),
            "type": "ValueError",
        },
        "status": "error",
    }


def test_pilot_gate_json_contract_stops_confirmatory_construction(tmp_path, capsys):
    config_payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_payload["split_counts"] = {"pilot": 1, "circuit_dev": 1}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    template_hash = CHAT_TEMPLATE_SHA256
    active_config = FAConfig.from_json(config_path)
    _, prompts = prompt_capability(tmp_path, active_config, "pilot")

    class BlockingRunner:
        model_id = active_config.model_id
        model_revision = active_config.model_revision
        tokenizer_revision = active_config.tokenizer_revision
        chat_template_sha256 = template_hash

        def generate(self, prompts, generation):
            return ["UNKNOWN"] * len(prompts)

    manifest = fa_cli._load_manifest(FAArtifactStore(tmp_path), prompts.manifest_path, active_config)
    generation = run_generation_shard(
        BlockingRunner(), manifest, FAArtifactStore(tmp_path), "responses", config=active_config
    )

    exit_code = cli.main(
        [
            "fa-score-behavior",
            "--config",
            str(config_path),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--generation-manifest",
            str(generation.manifest_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["command"] == "fa-score-behavior"
    assert payload["pilot_gate"]["status"] == "blocked"
    assert "target_bound_accuracy_below_70_percent" in payload["pilot_gate"]["reasons"]
    assert Path(payload["pilot_gate_manifest"]).exists()
    with pytest.raises(ValueError, match="exact registered smoke config"):
        fa_cli._load_verified_pilot_gate(
            FAArtifactStore(tmp_path),
            payload["pilot_gate_manifest"],
            FAConfig.from_json(CONFIG_PATH),
        )


def test_pilot_gate_rejects_self_consistent_template_without_tokenizer_pin_chain(
    tmp_path,
):
    config = FAConfig.from_json(CONFIG_PATH)
    alternate_template = b"arbitrary alternate self-consistent template"
    alternate_hash = hashlib.sha256(alternate_template).hexdigest()
    _, prompts = prompt_capability(
        tmp_path, config, "pilot", template_bytes=alternate_template
    )
    store = FAArtifactStore(tmp_path)
    assert alternate_hash != CHAT_TEMPLATE_SHA256
    with pytest.raises(ValueError, match="registered smoke tokenizer revision"):
        fa_cli._load_manifest(store, prompts.manifest_path, config)


class FakeRunner:
    def __init__(self, config):
        self.model_id = config.model_id
        self.model_revision = config.model_revision
        self.tokenizer_revision = config.tokenizer_revision
        self.chat_template_sha256 = CHAT_TEMPLATE_SHA256

    def generate(self, prompts, generation):
        return ["K7M2Q" for _ in prompts]


def test_run_generation_uses_fake_runner_for_generic_namespace(tmp_path, capsys, monkeypatch):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, "pilot")
    monkeypatch.setattr(fa_cli, "HFModelRunner", FakeRunner)

    exit_code = cli.main(
        [
            "fa-run-generation",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--shard-id",
            "0001",
            "--namespace",
            "pilot",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "generated"
    assert Path(payload["shard_manifest"]).exists()


def test_behavior_scoring_rejects_mutable_raw_generation_rows(tmp_path, capsys):
    config = FAConfig.from_json(CONFIG_PATH)
    _, manifest = prompt_capability(tmp_path, config, "pilot")
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")

    exit_code = cli.main(
        [
            "fa-score-behavior",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(manifest.manifest_path),
            "--generation-manifest",
            str(raw),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code != 0
    assert payload["status"] == "error"
    assert "verified generation sidecar manifest" in payload["error"]["message"]


@pytest.mark.parametrize(
    "namespace", ["behavior_test", "probe_test", "intervention_test"]
)
def test_generic_generation_rejects_protected_namespaces_before_runner_construction(
    tmp_path, capsys, monkeypatch, namespace
):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, namespace)

    class MustNotConstruct:
        def __init__(self, config):
            raise AssertionError("protected generation must use a dedicated later command")

    monkeypatch.setattr(fa_cli, "HFModelRunner", MustNotConstruct)
    exit_code = cli.main(
        [
            "fa-run-generation",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--shard-id",
            "0001",
            "--namespace",
            namespace,
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["error"]["message"] == (
        "fa-run-generation is generic-only and cannot evaluate protected test namespaces"
    )


@pytest.mark.parametrize(
    ("command", "required_option"),
    [
        ("fa-fit-probes", "--train-rows-manifest"),
        ("fa-seal-selection", "--selection-manifest"),
        ("fa-evaluate-probe-test", "--selection-manifest"),
    ],
)
def test_f2a_commands_require_explicit_artifacts_before_dispatch(
    capsys, command, required_option
):
    with pytest.raises(SystemExit) as raised:
        cli.main([command, "--config", str(CONFIG_PATH)])
    payload = json.loads(capsys.readouterr().out)

    assert raised.value.code == 2
    assert payload["command"] == command
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "ArgumentError"
    assert required_option in payload["error"]["message"]


def test_intervention_test_command_remains_not_implemented(capsys):
    exit_code = cli.main(
        ["fa-evaluate-intervention-test", "--config", str(CONFIG_PATH)]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "not_implemented"


def test_standalone_unlock_command_is_fail_closed_and_cannot_create_a_lease(
    tmp_path, capsys
):
    config = FAConfig.from_json(CONFIG_PATH)

    exit_code = cli.main(
        [
            "fa-unlock-endpoint",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["error"]["message"] == (
        "standalone endpoint unlock is disabled; a dedicated protected evaluation "
        "command must acquire and close its lease atomically"
    )
    endpoint_root = (
        tmp_path
        / "runs"
        / "familiarity_answerability"
        / config.run_id
        / "endpoints"
    )
    assert not endpoint_root.exists()


def exhaustive_power_audit(rows, *, power=0.8):
    cells = tuple(
        PowerCell(
            absent_attempt_rate=absent,
            entity_icc=entity,
            template_icc=template,
            invalid_format_rate=invalid,
            interaction=interaction,
            estimated_power=power,
            monte_carlo_standard_error=math.sqrt(
                power * (1 - power) / CONFIRMATORY_POWER_SIMULATIONS
            ),
            simulations=CONFIRMATORY_POWER_SIMULATIONS,
        )
        for absent, entity, template, invalid, interaction in product(
            REGISTERED_POWER_GRID.absent_attempt_rates,
            REGISTERED_POWER_GRID.entity_iccs,
            REGISTERED_POWER_GRID.template_iccs,
            REGISTERED_POWER_GRID.invalid_format_rates,
            REGISTERED_POWER_GRID.interactions,
        )
    )
    return PowerAudit(
        design_sha256=fa_cli._design_sha256(rows),
        seed=20260722,
        simulations=CONFIRMATORY_POWER_SIMULATIONS,
        cells=cells,
        registered_grid=True,
    )


def test_registered_power_preparation_uses_exact_fa_data_signature_and_typed_artifact(
    tmp_path, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    rows = valid_examples(config, "behavior_test")
    audit = exhaustive_power_audit(rows)
    calls = []

    def execute(design, effects, correlations, seed, *, simulations):
        calls.append((tuple(design), effects, correlations, seed, simulations))
        return audit

    monkeypatch.setattr(fa_cli, "_POWER_EXECUTOR", execute)
    loaded, shard = fa_cli._prepare_power_audit(
        FAArtifactStore(tmp_path),
        config,
        rows,
        None,
        run_registered=True,
    )

    assert loaded == audit
    assert shard.record_kind == "power_audit"
    assert calls == [
        (
            tuple(rows),
            REGISTERED_POWER_GRID.interactions,
            {
                "entity_icc": REGISTERED_POWER_GRID.entity_iccs,
                "template_icc": REGISTERED_POWER_GRID.template_iccs,
                "invalid_format_rate": REGISTERED_POWER_GRID.invalid_format_rates,
            },
            20260722,
            2000,
        )
    ]


def test_confirmatory_power_modes_are_explicit_and_mutually_exclusive(tmp_path, capsys):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)
    common = [
        "fa-build-confirmatory",
        "--config",
        str(CONFIG_PATH),
        "--matches-manifest",
        str(tmp_path / "matches.json"),
        "--pilot-gate-manifest",
        str(tmp_path / "gate.json"),
        "--naturalness-ratings-manifest",
        str(tmp_path / "ratings.json"),
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(common)
    capsys.readouterr()
    with pytest.raises(SystemExit):
        parser.parse_args(
            common
            + [
                "--power-audit-manifest",
                str(tmp_path / "power.json"),
                "--run-registered-power-audit",
            ]
        )


def test_prompt_loader_rejects_raw_or_self_attested_manifests(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    raw = tmp_path / "manifest.json"
    raw.write_text(json.dumps({"manifest_sha256": "a" * 64}), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable shard manifest"):
        fa_cli._load_manifest(FAArtifactStore(tmp_path), raw, config)

    fake = FAArtifactStore(tmp_path).write_completed_shard(
        config.run_id,
        "pilot",
        "self-attested",
        [
            {
                "kind": "prompt_manifest",
                "config_hash": config.config_hash,
                "full_manifest_sha256": "a" * 64,
                "subset_manifest_sha256": "b" * 64,
                "chat_template_sha256": "d" * 64,
                "namespace": "pilot",
                "model_sha256": "e" * 64,
                "tokenizer_sha256": "f" * 64,
                "generation": dict(config.generation),
                "examples": [{"example_id": "self-attested"}],
            }
        ],
        {
            "config_sha256": config.config_hash,
            "source_manifest_sha256": "a" * 64,
            "subset_manifest_sha256": "b" * 64,
            "chat_template_sha256": "d" * 64,
        },
        record_kind="prompt_manifest",
    )
    with pytest.raises(ValueError, match="invalid schema"):
        fa_cli._load_manifest(FAArtifactStore(tmp_path), fake.manifest_path, config)


def test_confirmatory_index_contains_only_ids_hashes_and_capability_paths(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    pilot_rows, pilot = prompt_capability(tmp_path, config, "pilot")
    protected_rows, protected = prompt_capability(tmp_path, config, "behavior_test")
    store = FAArtifactStore(tmp_path)
    power = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        "power-index-parent",
        [{"kind": "power_audit", "audit": {}}],
        {"config_sha256": config.config_hash},
        record_kind="power_audit",
    )
    pin = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        "pin-index-parent",
        [{"kind": "tokenizer_pin"}],
        {"config_sha256": config.config_hash},
        record_kind="tokenizer_pin",
    )

    index = fa_cli._confirmatory_index_record(
        store,
        config,
        "f" * 64,
        pilot_rows + protected_rows,
        {"pilot": pilot, "behavior_test": protected},
        power,
        pin,
    )
    encoded = json.dumps(index, sort_keys=True)

    assert "user_text" not in encoded
    assert "target_text" not in encoded
    assert "expected_output" not in encoded
    assert all(row.user_text not in encoded for row in pilot_rows + protected_rows)
    assert set(index["capabilities"]) == {"pilot", "behavior_test"}


def test_confirmatory_build_prepares_capabilities_without_sealing_endpoints(
    tmp_path, monkeypatch
):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    config = FAConfig.from_json(config_path)
    store = FAArtifactStore(tmp_path)
    matches = tmp_path / "matches.json"
    matches.write_text(json.dumps([MATCH]), encoding="utf-8")
    ratings = naturalness_ratings_manifest(tmp_path, config)
    rows = tuple(
        SimpleNamespace(
            example_id=sha256_json({"namespace": namespace}),
                canonical_payload_sha256=sha256_json({"namespace": namespace}),
                split=namespace,
                answerability="code_absent",
            )
        for namespace in config.split_counts
    )
    power = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        "prepared-power",
        [{"kind": "power_audit", "audit": {}}],
        {"config_sha256": config.config_hash},
        record_kind="power_audit",
    )
    observed_smoke_configs = []

    def load_gate(artifact_store, path, expected_config):
        observed_smoke_configs.append(expected_config)
        return {"status": "passed", "evidence_sha256": "a" * 64}

    monkeypatch.setattr(fa_cli, "_load_verified_pilot_gate", load_gate)
    monkeypatch.setattr(
        fa_cli,
        "load_pinned_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(
            tokenizer=FakeTokenizer(),
            chat_template_bytes=b"registered confirmatory template",
            chat_template_sha256=config.chat_template_sha256,
        ),
    )
    monkeypatch.setattr(fa_cli, "build_factorial_examples", lambda *args, **kwargs: rows)
    monkeypatch.setattr(fa_cli, "build_same_string_examples", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        fa_cli,
        "_prepare_power_audit",
        lambda *args, **kwargs: (SimpleNamespace(), power),
    )
    monkeypatch.setattr(
        fa_cli,
        "build_manifest",
        lambda *args, **kwargs: SimpleNamespace(manifest_sha256="f" * 64),
    )

    def must_not_seal(*args, **kwargs):
        raise AssertionError("confirmatory construction must not seal an endpoint")

    monkeypatch.setattr(FAArtifactStore, "seal_endpoint", must_not_seal)
    payload = fa_cli._build_manifest(
        config,
        tmp_path,
        SimpleNamespace(
            matches_manifest=matches,
            pilot_gate_manifest=tmp_path / "gate.json",
            power_audit_manifest=None,
            run_registered_power_audit=True,
            naturalness_ratings_manifest=ratings,
        ),
        confirmatory=True,
    )

    assert observed_smoke_configs == [FAConfig.from_json(CONFIG_PATH)]
    assert set(payload["namespace_manifests"]) == set(config.split_counts)
    assert "protected_endpoint_manifests" not in payload
    audit = store.verify_shard(payload["naturalness_audit_manifest"])
    assert audit.record_kind == "naturalness_audit"
    assert audit.sha256 == payload["naturalness_audit_sha256"]
    for manifest_path in payload["namespace_manifests"].values():
        prompt = store.verify_shard(manifest_path)
        row = fa_cli._read_json_rows(prompt.data_path)[0]
        assert row["naturalness_audit_sha256"] == audit.sha256
    assert not (
        tmp_path
        / "runs"
        / "familiarity_answerability"
        / config.run_id
        / "endpoints"
    ).exists()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_generation_sidecar_requires_exact_example_multiset(tmp_path, mutation):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, "pilot")
    store = FAArtifactStore(tmp_path)
    manifest = fa_cli._load_manifest(store, prompts.manifest_path, config)
    generated = run_generation_shard(
        FakeRunner(config), manifest, store, "valid", config=config, namespace="pilot"
    )
    rows = list(fa_cli._read_json_rows(generated.data_path))
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        extra = dict(rows[0])
        extra["example_id"] = "0" * 64
        rows.append(extra)
    lineage = json.loads(generated.manifest_path.read_text(encoding="utf-8"))["lineage"]
    forged = store.write_completed_shard(
        config.run_id,
        "pilot",
        f"forged-{mutation}",
        rows,
        lineage,
        record_kind="generation",
    )

    with pytest.raises(ValueError, match="every expected example exactly once"):
        fa_cli._load_verified_generation_sidecar(
            store, forged.manifest_path, manifest, config
        )


def test_power_audit_rejects_duplicate_registered_cells(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    rows = valid_examples(config, "behavior_test")
    audit = exhaustive_power_audit(rows)
    forged = PowerAudit(
        design_sha256=audit.design_sha256,
        seed=audit.seed,
        simulations=audit.simulations,
        cells=audit.cells[:-1] + (audit.cells[0],),
        registered_grid=True,
    )
    shard = FAArtifactStore(tmp_path).write_completed_shard(
        config.run_id,
        "mechanism_train",
        "forged-power",
        [{"kind": "power_audit", "audit": asdict(forged)}],
        {"config_sha256": config.config_hash, "design_sha256": audit.design_sha256},
        record_kind="power_audit",
    )

    with pytest.raises(ValueError, match="180 unique"):
        fa_cli._prepare_power_audit(
            FAArtifactStore(tmp_path),
            config,
            rows,
            shard.manifest_path,
            run_registered=False,
        )


def test_pilot_gate_recomputes_and_rejects_forged_stored_pass(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, "pilot")
    store = FAArtifactStore(tmp_path)
    manifest = fa_cli._load_manifest(store, prompts.manifest_path, config)

    class BlockingRunner(FakeRunner):
        def generate(self, prompts, generation):
            return ["INVALID"] * len(prompts)

    generation = run_generation_shard(
        BlockingRunner(config), manifest, store, "blocked", config=config
    )
    forged_gate = {"status": "passed", "reasons": []}
    metrics = {"forged": True}
    evidence_hash = sha256_json({"metrics": metrics, "pilot_gate": forged_gate})
    gate = store.write_completed_shard(
        config.run_id,
        "pilot",
        "forged-gate",
        [
            {
                "kind": "pilot_gate",
                "config_sha256": config.config_hash,
                "source_manifest_sha256": manifest.manifest_sha256,
                "prompt_manifest": str(prompts.manifest_path.relative_to(store.root)),
                "prompt_manifest_sha256": prompts.sha256,
                "tokenizer_pin_manifest": str(
                    manifest.tokenizer_pin_manifest_path.relative_to(store.root)
                ),
                "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
                "chat_template_sha256": manifest.chat_template_sha256,
                "generation_sidecar_manifest": str(
                    generation.manifest_path.relative_to(store.root)
                ),
                "generation_sidecar_sha256": generation.sha256,
                "pilot_gate": forged_gate,
                "metrics": metrics,
                "evidence_sha256": evidence_hash,
            }
        ],
        {
            "config_sha256": config.config_hash,
            "source_manifest_sha256": manifest.manifest_sha256,
            "generation_sidecar_sha256": generation.sha256,
            "prompt_manifest_sha256": prompts.sha256,
            "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
            "chat_template_sha256": manifest.chat_template_sha256,
        },
        record_kind="pilot_gate",
    )
    gate_row = fa_cli._read_json_rows(gate.data_path)[0]
    gate_lineage = json.loads(gate.manifest_path.read_text(encoding="utf-8"))[
        "lineage"
    ]
    unrelated_run_gate = store.write_completed_shard(
        "unrelated-run",
        "pilot",
        "forged-gate",
        [gate_row],
        gate_lineage,
        record_kind="pilot_gate",
    )

    with pytest.raises(ValueError, match="registered smoke run"):
        fa_cli._load_verified_pilot_gate(
            store, unrelated_run_gate.manifest_path, config
        )

    with pytest.raises(ValueError, match="deterministic recomputation"):
        fa_cli._load_verified_pilot_gate(store, gate.manifest_path, config)


def test_build_confirmatory_rejects_raw_passed_gate_before_tokenizer_loading(
    tmp_path, capsys, monkeypatch
):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    matches = tmp_path / "matches.json"
    matches.write_text(json.dumps([MATCH]), encoding="utf-8")
    gate = tmp_path / "raw-gate.json"
    gate.write_text(json.dumps({"status": "passed"}), encoding="utf-8")

    def must_not_load(*args, **kwargs):
        raise AssertionError("tokenizer must not load before gate verification")

    monkeypatch.setattr(fa_cli, "load_pinned_tokenizer", must_not_load)
    exit_code = cli.main(
        [
            "fa-build-confirmatory",
            "--config",
            str(config_path),
            "--root",
            str(tmp_path),
            "--matches-manifest",
            str(matches),
            "--pilot-gate-manifest",
            str(gate),
            "--naturalness-ratings-manifest",
            str(tmp_path / "ratings.json"),
            "--run-registered-power-audit",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code != 0
    assert payload["status"] == "error"
    assert "verified pilot gate sidecar manifest" in payload["error"]["message"]


def test_confirmatory_build_requires_human_naturalness_ratings_at_parse_time(
    tmp_path, capsys
):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "fa-build-confirmatory",
                "--config",
                str(
                    Path(__file__).resolve().parents[1]
                    / "configs"
                    / "familiarity_answerability_gemma2_2b.json"
                ),
                "--matches-manifest",
                str(tmp_path / "matches.json"),
                "--pilot-gate-manifest",
                str(tmp_path / "gate.json"),
                "--run-registered-power-audit",
            ]
        )

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["type"] == "ArgumentError"
    assert "naturalness-ratings-manifest" in payload["error"]["message"]


def test_naturalness_ratings_require_verified_blinded_input_and_current_schema(
    tmp_path,
):
    config = FAConfig.from_json(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    raw = tmp_path / "ratings.json"
    raw.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable shard manifest"):
        fa_cli._load_verified_naturalness_ratings(
            FAArtifactStore(tmp_path), raw, config
        )

    unsupported = naturalness_ratings_manifest(
        tmp_path, config, rating_schema_version=2
    )
    with pytest.raises(ValueError, match="schema_version"):
        fa_cli._load_verified_naturalness_ratings(
            FAArtifactStore(tmp_path), unsupported, config
        )


def test_prompt_capability_writer_enforces_profile_specific_human_audit(tmp_path):
    confirmatory = FAConfig.from_json(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    store = FAArtifactStore(tmp_path)
    pin = store.write_completed_shard(
        confirmatory.run_id,
        "mechanism_train",
        "pin-for-profile-check",
        [{"kind": "tokenizer_pin"}],
        {"config_sha256": confirmatory.config_hash},
        record_kind="tokenizer_pin",
    )
    example = SimpleNamespace(
        example_id="a" * 64,
        canonical_payload_sha256="b" * 64,
        split="mechanism_train",
    )
    with pytest.raises(ValueError, match="confirmatory prompt capability requires"):
        fa_cli._write_prompt_capability(
            store,
            confirmatory,
            "c" * 64,
            "mechanism_train",
            (example,),
            confirmatory.chat_template_sha256,
            pin,
        )
