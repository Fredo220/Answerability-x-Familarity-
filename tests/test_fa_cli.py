import argparse
import hashlib
import json
from pathlib import Path

import pytest

from trajectory_extractor import cli
from trajectory_extractor.fa_artifacts import FAArtifactStore
import trajectory_extractor.fa_cli as fa_cli
from trajectory_extractor.fa_cli import dispatch_fa, register_fa_subcommands
from trajectory_extractor.fa_config import FAConfig


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


def test_fa_dispatch_is_isolated_and_cli_routes_fa_commands(tmp_path, capsys):
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


def test_fa_commands_require_explicit_input_manifests_and_restrict_generation_namespaces(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(["fa-run-generation", "--config", str(CONFIG_PATH)])
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
    with pytest.raises(ValueError, match="endpoint manifest"):
        dispatch_fa(args)


def test_pilot_gate_json_contract_stops_confirmatory_construction(tmp_path, capsys):
    config_payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_payload["split_counts"] = {"pilot": 1, "circuit_dev": 1}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    manifest = {
        "config_hash": hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "manifest_sha256": "a" * 64,
        "examples": [],
    }
    manifest_path = tmp_path / "pilot-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    responses_path = tmp_path / "responses.jsonl"
    responses_path.write_text(
        json.dumps(
            {
                "example": {
                    "example_id": "e1",
                    "entity_unit_id": "u1",
                    "template_family": "train_registry_direct",
                    "target_familiarity": "screened_real",
                    "distractor_familiarity": "matched_synthetic",
                    "answerability": "target_bound",
                    "registry_code": "K7M2Q",
                    "block": "factorial",
                    "exposure": None,
                },
                "raw_output": "UNKNOWN",
                "status": "completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "fa-score-behavior",
            "--config",
            str(config_path),
            "--root",
            str(tmp_path),
            "--manifest",
            str(manifest_path),
            "--generation-manifest",
            str(responses_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["command"] == "fa-score-behavior"
    assert payload["pilot_gate"]["status"] == "blocked"
    assert "target_bound_accuracy_below_70_percent" in payload["pilot_gate"]["reasons"]


class FakeRunner:
    def __init__(self, config):
        self.model_id = config.model_id
        self.model_revision = config.model_revision
        self.tokenizer_revision = config.tokenizer_revision
        self.chat_template_sha256 = "d" * 64

    def generate(self, prompts, generation):
        return ["K7M2Q" for _ in prompts]


def test_run_generation_uses_fake_runner_and_explicit_protected_endpoint_manifest(tmp_path, capsys, monkeypatch):
    config = FAConfig.from_json(CONFIG_PATH)
    example = {
        "example_id": "a" * 64,
        "canonical_payload_sha256": "a" * 64,
        "user_text": "What is stated?",
        "split": "behavior_test",
        "entity_unit_id": "unit-1",
        "template_family": "behavior_catalog_direct",
        "target_familiarity": "screened_real",
        "distractor_familiarity": "matched_synthetic",
        "answerability": "target_bound",
        "registry_code": "K7M2Q",
        "block": "factorial",
        "exposure": None,
    }
    manifest = tmp_path / "examples.json"
    manifest.write_text(
        json.dumps({"config_hash": config.config_hash, "manifest_sha256": "b" * 64, "examples": [example]}),
        encoding="utf-8",
    )
    store = FAArtifactStore(tmp_path)
    selection = store.write_completed_shard(
        config.run_id,
        "behavior_test",
        "selection",
        [{"example_id": "selection"}],
        {"config_sha256": config.config_hash, "source_manifest_sha256": "c" * 64},
    )
    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text(
        json.dumps(
            {
                "endpoint": "behavior_test",
                "preregistration_hash": "e" * 64,
                "selection_manifest_hash": "f" * 64,
                "selection_shard_manifests": [str(selection.manifest_path)],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fa_cli, "HFModelRunner", FakeRunner)

    exit_code = cli.main(
        [
            "fa-run-generation",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--shard-id",
            "0001",
            "--namespace",
            "behavior_test",
            "--endpoint-manifest",
            str(endpoint),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "generated"
    assert Path(payload["closed_endpoint"]).exists()
