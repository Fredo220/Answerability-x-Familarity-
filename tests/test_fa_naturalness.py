from __future__ import annotations

import csv
import json
import stat
from dataclasses import asdict
from pathlib import Path

import pytest

from trajectory_extractor import fa_cli, main as cli
from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_entities import EntityMatch, audit_naturalness_manifest
from trajectory_extractor.fa_naturalness import (
    compile_adjudication_response_from_issuance,
    packet_issuance_record,
    prepare_adjudication_packet,
    verify_submission_record,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "familiarity_answerability_qwen17b_smoke.json"
)


def _match(index: int, real_name: str, synthetic_name: str) -> EntityMatch:
    return EntityMatch(
        pair_id=f"real-{index}--synthetic-{index}",
        real_entity_id=f"real-{index}",
        real_qid=f"Q{index + 1}",
        synthetic_candidate_id=f"synthetic-{index}",
        real_name=real_name,
        synthetic_name=synthetic_name,
        coarse_type="person",
        split="pilot",
        generator_revision="test-v1",
        tokenizer_revision="test-tokenizer",
        real_token_count=2,
        synthetic_token_count=2,
        real_word_count=2,
        synthetic_word_count=2,
        real_character_count=len(real_name),
        synthetic_character_count=len(synthetic_name),
        character_length_delta=len(synthetic_name) - len(real_name),
        character_tolerance=2,
        capitalization_pattern_equal=True,
    )


def _matches() -> tuple[EntityMatch, ...]:
    return (
        _match(1, "Nova Hall", "Lira Hall"),
        _match(2, "Mira Vale", "Lora Vale"),
    )


def _write_matches(path: Path) -> None:
    path.write_text(
        json.dumps([asdict(value) for value in _matches()]),
        encoding="utf-8",
    )


def _fill_response(
    path: Path,
    *,
    real_rating: int = 4,
    synthetic_rating: int = 4,
    key_path: Path,
) -> None:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    mappings = {
        row["item_id"]: row
        for row in key["items"]
        if row["rater_id"] == path.name.removesuffix("-response.csv")
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = tuple(rows[0])
    for row in rows:
        mapping = mappings[row["item_id"]]
        for label in ("a", "b"):
            role = mapping[f"candidate_{label}_role"]
            row[f"candidate_{label}_naturalness"] = str(
                real_rating if role == "real" else synthetic_rating
            )
            row[f"candidate_{label}_type_fit"] = "4"
            row[f"candidate_{label}_malformed"] = "false"
        row["independence_attested"] = "true"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_adjudication_issuance_accepts_registered_subset_of_matches(tmp_path):
    matches = (*_matches(), _match(3, "Tara Lane", "Lena Lane"))
    disagreement = matches[0].pair_id
    packet = prepare_adjudication_packet(
        matches,
        pair_ids=(disagreement,),
        config_sha256="a" * 64,
        protocol_sha256="b" * 64,
        rating_protocol_sha256="c" * 64,
        output_dir=tmp_path / "adjudication",
        adjudicator_id="rater-c",
    )
    issuance, _ = packet_issuance_record(packet["private_key"])
    response = (
        tmp_path / "adjudication" / "public" / "rater-c-response.csv"
    )
    _fill_response(
        response,
        key_path=tmp_path
        / "adjudication"
        / "private"
        / "unblinding-key.json",
    )

    ratings, assignments, responses = compile_adjudication_response_from_issuance(
        matches,
        issuance=issuance,
        response_path=response,
        expected_pair_ids=(disagreement,),
        config_sha256="a" * 64,
        protocol_sha256="b" * 64,
        rating_protocol_sha256="c" * 64,
    )

    assert len(ratings) == len(assignments) == len(responses) == 1
    assert ratings[0].pair_id == disagreement


def test_rating_packets_are_deterministic_blinded_and_counterbalanced(
    tmp_path, capsys
):
    matches_path = tmp_path / "matches.json"
    _write_matches(matches_path)
    outputs = []
    payloads = []
    for suffix in ("a", "b"):
        output = tmp_path / f"packets-{suffix}"
        exit_code = cli.main(
            [
                "fa-prepare-naturalness-ratings",
                "--config",
                str(CONFIG_PATH),
                "--root",
                str(tmp_path),
                "--matches-manifest",
                str(matches_path),
                "--output-dir",
                str(output),
                "--rater-id",
                "rater-a",
                "--rater-id",
                "rater-b",
                "--shard-id",
                f"packet-issuance-{suffix}",
            ]
        )
        assert exit_code == 0
        outputs.append(output)
        payloads.append(json.loads(capsys.readouterr().out))

    for payload in payloads:
        manifest_path = Path(payload["issuance_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_path = manifest_path.parent / manifest["data_file"]
        assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(data_path.stat().st_mode) == 0o600

    for rater in ("rater-a", "rater-b"):
        first = json.loads(
            (outputs[0] / "public" / f"{rater}-packet.json").read_text(
                encoding="utf-8"
            )
        )
        second = json.loads(
            (outputs[1] / "public" / f"{rater}-packet.json").read_text(
                encoding="utf-8"
            )
        )
        assert first == second
        serialized = json.dumps(first)
        assert "pair_id" not in serialized
        assert "real_name" not in serialized
        assert "synthetic_name" not in serialized
        with (
            outputs[0] / "public" / f"{rater}-response.csv"
        ).open(newline="", encoding="utf-8") as handle:
            worksheet = list(csv.DictReader(handle))
        first_by_id = {row["item_id"]: row for row in first["items"]}
        assert {
            "coarse_type",
            "candidate_a",
            "candidate_b",
            "sentence_a",
            "sentence_b",
        }.issubset(worksheet[0])
        for row in worksheet:
            item = first_by_id[row["item_id"]]
            assert row["coarse_type"] == item["coarse_type"]
            assert row["candidate_a"] == item["candidate_a"]
            assert row["candidate_b"] == item["candidate_b"]

    key = json.loads(
        (outputs[0] / "private" / "unblinding-key.json").read_text(
            encoding="utf-8"
        )
    )
    by_pair = {}
    for row in key["items"]:
        by_pair.setdefault(row["pair_id"], []).append(row["candidate_a_role"])
    assert all(sorted(roles) == ["real", "synthetic"] for roles in by_pair.values())


def test_compiler_writes_verifiable_ratings_artifact(tmp_path, capsys):
    matches_path = tmp_path / "matches.json"
    _write_matches(matches_path)
    packet_dir = tmp_path / "packets"
    assert (
        cli.main(
            [
                "fa-prepare-naturalness-ratings",
                "--config",
                str(CONFIG_PATH),
                "--root",
                str(tmp_path),
                "--matches-manifest",
                str(matches_path),
                "--output-dir",
                str(packet_dir),
                "--rater-id",
                "rater-a",
                "--rater-id",
                "rater-b",
                "--shard-id",
                "packet-issuance",
            ]
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)
    key_path = packet_dir / "private" / "unblinding-key.json"
    responses = [
        packet_dir / "public" / "rater-a-response.csv",
        packet_dir / "public" / "rater-b-response.csv",
    ]
    for response in responses:
        _fill_response(response, key_path=key_path)

    assert (
        cli.main(
            [
                "fa-compile-naturalness-ratings",
                "--config",
                str(CONFIG_PATH),
                "--root",
                str(tmp_path),
                "--matches-manifest",
                str(matches_path),
                "--issuance-manifest",
                prepared["issuance_manifest"],
                "--response",
                str(responses[0]),
                "--response",
                str(responses[1]),
                "--shard-id",
                "human-ratings-v1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    config = FAConfig.from_json(CONFIG_PATH)
    ratings, shard = fa_cli._load_verified_naturalness_ratings(
        FAArtifactStore(tmp_path), payload["ratings_manifest"], config
    )
    audit = audit_naturalness_manifest(_matches(), ratings)

    assert shard.sha256 == payload["sha256"]
    assert len(ratings) == 4
    assert audit.accepted_pair_ids == tuple(
        sorted(value.pair_id for value in _matches())
    )
    ratings_manifest = json.loads(
        Path(payload["ratings_manifest"]).read_text(encoding="utf-8")
    )
    ratings_row = json.loads(
        (
            Path(payload["ratings_manifest"]).parent
            / ratings_manifest["data_file"]
        ).read_text(encoding="utf-8")
    )
    spliced_lineage = dict(ratings_manifest["lineage"])
    spliced_lineage["matches_sha256"] = "0" * 64
    spliced = FAArtifactStore(tmp_path).write_completed_shard(
        config.run_id,
        "mechanism_train",
        "spliced-matches-ratings",
        (ratings_row,),
        spliced_lineage,
        record_kind="naturalness_ratings",
    )
    with pytest.raises(ValueError, match="entity pairs do not match"):
        fa_cli._load_verified_naturalness_ratings(
            FAArtifactStore(tmp_path), spliced.manifest_path, config
        )

    submission_manifest = json.loads(
        Path(payload["initial_submission_manifest"]).read_text(encoding="utf-8")
    )
    submission_path = (
        Path(payload["initial_submission_manifest"]).parent
        / submission_manifest["data_file"]
    )
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    issuance_manifest_path = (
        tmp_path / submission["issuance_manifest"]
    )
    issuance_manifest = json.loads(
        issuance_manifest_path.read_text(encoding="utf-8")
    )
    issuance = json.loads(
        (issuance_manifest_path.parent / issuance_manifest["data_file"]).read_text(
            encoding="utf-8"
        )
    )
    extra_response = dict(submission["responses"][0])
    extra_response["rater_id"] = "unregistered-rater"
    extra_response["item_id"] = "unregistered-item"
    submission["responses"].append(extra_response)
    with pytest.raises(ValueError, match="raters do not match"):
        verify_submission_record(issuance, submission)


def test_compiler_rejects_edited_human_facing_stimulus(tmp_path, capsys):
    matches_path = tmp_path / "matches.json"
    _write_matches(matches_path)
    packet_dir = tmp_path / "packets"
    assert cli.main(
        [
            "fa-prepare-naturalness-ratings",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--matches-manifest",
            str(matches_path),
            "--output-dir",
            str(packet_dir),
            "--rater-id",
            "rater-a",
            "--rater-id",
            "rater-b",
            "--shard-id",
            "packet-issuance",
        ]
    ) == 0
    prepared = json.loads(capsys.readouterr().out)
    key_path = packet_dir / "private" / "unblinding-key.json"
    responses = [
        packet_dir / "public" / "rater-a-response.csv",
        packet_dir / "public" / "rater-b-response.csv",
    ]
    for response in responses:
        _fill_response(response, key_path=key_path)
    with responses[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = tuple(rows[0])
    rows[0]["candidate_a"] = "Edited Name"
    with responses[0].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    exit_code = cli.main(
        [
            "fa-compile-naturalness-ratings",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--matches-manifest",
            str(matches_path),
            "--issuance-manifest",
            prepared["issuance_manifest"],
            "--response",
            str(responses[0]),
            "--response",
            str(responses[1]),
            "--shard-id",
            "human-ratings-v1",
        ]
    )

    assert exit_code == 2
    assert "stimulus was edited" in json.loads(capsys.readouterr().out)["error"][
        "message"
    ]


def test_compiler_uses_sealed_issuance_after_packet_file_is_modified(
    tmp_path, capsys
):
    matches_path = tmp_path / "matches.json"
    _write_matches(matches_path)
    packet_dir = tmp_path / "packets"
    assert (
        cli.main(
            [
                "fa-prepare-naturalness-ratings",
                "--config",
                str(CONFIG_PATH),
                "--root",
                str(tmp_path),
                "--matches-manifest",
                str(matches_path),
                "--output-dir",
                str(packet_dir),
                "--rater-id",
                "rater-a",
                "--rater-id",
                "rater-b",
                "--shard-id",
                "packet-issuance",
            ]
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)
    key_path = packet_dir / "private" / "unblinding-key.json"
    responses = [
        packet_dir / "public" / "rater-a-response.csv",
        packet_dir / "public" / "rater-b-response.csv",
    ]
    for response in responses:
        _fill_response(response, key_path=key_path)
    packet_path = packet_dir / "public" / "rater-a-packet.json"
    packet_path.write_bytes(packet_path.read_bytes() + b"\n")

    exit_code = cli.main(
        [
            "fa-compile-naturalness-ratings",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--matches-manifest",
            str(matches_path),
            "--issuance-manifest",
            prepared["issuance_manifest"],
            "--response",
            str(responses[0]),
            "--response",
            str(responses[1]),
            "--shard-id",
            "tampered-ratings",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "compiled"


def test_disagreement_requires_and_accepts_independent_adjudication(tmp_path, capsys):
    matches_path = tmp_path / "matches.json"
    _write_matches(matches_path)
    packet_dir = tmp_path / "packets"
    assert (
        cli.main(
            [
                "fa-prepare-naturalness-ratings",
                "--config",
                str(CONFIG_PATH),
                "--root",
                str(tmp_path),
                "--matches-manifest",
                str(matches_path),
                "--output-dir",
                str(packet_dir),
                "--rater-id",
                "rater-a",
                "--rater-id",
                "rater-b",
                "--shard-id",
                "packet-issuance",
            ]
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)
    key_path = packet_dir / "private" / "unblinding-key.json"
    response_a = packet_dir / "public" / "rater-a-response.csv"
    response_b = packet_dir / "public" / "rater-b-response.csv"
    _fill_response(response_a, key_path=key_path)
    _fill_response(
        response_b,
        real_rating=5,
        synthetic_rating=1,
        key_path=key_path,
    )
    adjudication_dir = tmp_path / "adjudication"
    base_args = [
        "fa-compile-naturalness-ratings",
        "--config",
        str(CONFIG_PATH),
        "--root",
        str(tmp_path),
        "--matches-manifest",
        str(matches_path),
        "--issuance-manifest",
        prepared["issuance_manifest"],
        "--response",
        str(response_a),
        "--response",
        str(response_b),
        "--shard-id",
        "human-ratings-v2",
    ]
    assert cli.main(base_args) == 2
    missing_adjudicator = json.loads(capsys.readouterr().out)
    assert "requires --adjudicator-id" in missing_adjudicator["error"]["message"]
    shard_dir = (
        tmp_path
        / "runs"
        / "familiarity_answerability"
        / "smoke-qwen17b-v1"
        / "shards"
        / "mechanism_train"
    )
    assert not list(shard_dir.glob("human-ratings-v2-initial*"))

    assert (
        cli.main(
            [
                *base_args,
                "--adjudicator-id",
                "rater-a",
                "--adjudication-output-dir",
                str(tmp_path / "invalid-adjudication"),
            ]
        )
        == 2
    )
    reused_rater = json.loads(capsys.readouterr().out)
    assert reused_rater["error"]["message"] == "third rater must be independent"
    assert not list(shard_dir.glob("human-ratings-v2-initial*"))

    assert (
        cli.main(
            [
                *base_args,
                "--adjudicator-id",
                "rater-c",
                "--adjudication-output-dir",
                str(adjudication_dir),
            ]
        )
        == 0
    )
    pending = json.loads(capsys.readouterr().out)
    assert pending["status"] == "needs_adjudication"
    assert pending["disagreement_pair_count"] == 2

    adjudication_key = adjudication_dir / "private" / "unblinding-key.json"
    adjudication_response = (
        adjudication_dir / "public" / "rater-c-response.csv"
    )
    _fill_response(adjudication_response, key_path=adjudication_key)
    assert (
        cli.main(
            [
                "fa-finalize-naturalness-adjudication",
                "--config",
                str(CONFIG_PATH),
                "--root",
                str(tmp_path),
                "--matches-manifest",
                str(matches_path),
                "--initial-submission-manifest",
                pending["initial_submission_manifest"],
                "--adjudication-issuance-manifest",
                pending["adjudication_issuance_manifest"],
                "--adjudication-response",
                str(adjudication_response),
                "--shard-id",
                "human-ratings-final",
            ]
        )
        == 0
    )
    complete = json.loads(capsys.readouterr().out)
    config = FAConfig.from_json(CONFIG_PATH)
    ratings, _ = fa_cli._load_verified_naturalness_ratings(
        FAArtifactStore(tmp_path), complete["ratings_manifest"], config
    )

    assert complete["status"] == "compiled"
    assert len(ratings) == 6
    assert {
        rating.rater_id for rating in ratings if rating.round == 2
    } == {"rater-c"}

    store = FAArtifactStore(tmp_path)
    adjudication_manifest_path = Path(
        complete["adjudication_submission_manifest"]
    )
    adjudication_manifest = json.loads(
        adjudication_manifest_path.read_text(encoding="utf-8")
    )
    adjudication_row = json.loads(
        (
            adjudication_manifest_path.parent
            / adjudication_manifest["data_file"]
        ).read_text(encoding="utf-8")
    )
    spliced_adjudication_lineage = dict(adjudication_manifest["lineage"])
    spliced_adjudication_lineage["initial_submission_sha256"] = "0" * 64
    spliced_adjudication = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        "spliced-adjudication",
        (adjudication_row,),
        spliced_adjudication_lineage,
        record_kind="naturalness_submission",
    )
    final_manifest_path = Path(complete["ratings_manifest"])
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    final_row = json.loads(
        (final_manifest_path.parent / final_manifest["data_file"]).read_text(
            encoding="utf-8"
        )
    )
    spliced_final_lineage = dict(final_manifest["lineage"])
    spliced_final_lineage.update(
        {
            "adjudication_submission_manifest": str(
                spliced_adjudication.manifest_path.relative_to(store.root)
            ),
            "adjudication_submission_sha256": spliced_adjudication.sha256,
        }
    )
    spliced_final = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        "spliced-final-ratings",
        (final_row,),
        spliced_final_lineage,
        record_kind="naturalness_ratings",
    )
    with pytest.raises(ValueError, match="does not bind the initial submission"):
        fa_cli._load_verified_naturalness_ratings(
            store, spliced_final.manifest_path, config
        )
