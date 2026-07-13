import json
from pathlib import Path

import pytest

from trajectory_extractor import cli
from trajectory_extractor.rlmf_artifacts import RLMFArtifactStore, sha256_file
from trajectory_extractor.rlmf_types import RLMFConfig


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "rlmf_qwen06b_confirmatory.json"
)


def test_rlmf_prepare_data_writes_snapshot_and_reports_registered_provenance(
    tmp_path, monkeypatch, capsys
):
    expected = {
        "source_rows": tmp_path / "source.jsonl",
        "normalized_rows": tmp_path / "snapshot.jsonl",
        "aliases": tmp_path / "aliases.jsonl",
        "discarded_rows": tmp_path / "discarded.jsonl",
        "split_manifest": tmp_path / "manifest.json",
        "completion": tmp_path / "completion.json",
    }
    captured = {}

    def write_snapshot(config, store):
        captured["config"] = config
        captured["store"] = store
        return expected

    monkeypatch.setattr(cli, "write_popqa_snapshot", write_snapshot)

    exit_code = cli.main(
        ["rlmf-prepare-data", "--config", str(CONFIG_PATH), "--root", str(tmp_path)]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert captured["config"].study_id == "rlmf-qwen06b-v1"
    assert captured["store"].root == tmp_path
    assert payload == {
        "study_id": "rlmf-qwen06b-v1",
        "count": 896,
        "split_counts": {
            "pre_sft": 256,
            "rl_train": 256,
            "validation": 128,
            "test": 256,
        },
        "dataset_revision": "5cf59972d88d4aaaa7781ac91b83d053563d8268",
        "artifacts": {name: str(path) for name, path in expected.items()},
    }


def test_build_requires_verified_candidate_and_locked_prerequisite_endpoints(
    tmp_path,
):
    candidates = _audit_candidates(per_stratum=125, phase="test")
    _seal_aliases(tmp_path, candidates)
    raw = (
        tmp_path
        / "runs"
        / "rlmf"
        / _config().study_id
        / "evaluation"
        / "audit_candidates_test.jsonl"
    )
    raw.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(raw, candidates)

    with pytest.raises(ValueError, match="candidate endpoint"):
        cli.main(
            [
                "rlmf-build-judge-audit",
                "--config",
                str(CONFIG_PATH),
                "--phase",
                "test",
                "--root",
                str(tmp_path),
            ]
        )

    raw.unlink()
    _seal_candidates(tmp_path, candidates, phase="test", locked_hash="a" * 64)
    with pytest.raises(ValueError, match="locked_judge_audit"):
        cli.main(
            [
                "rlmf-build-judge-audit",
                "--config",
                str(CONFIG_PATH),
                "--phase",
                "test",
                "--root",
                str(tmp_path),
            ]
        )


def test_cli_seals_independent_rating_sources_and_appends_test_extensions(
    tmp_path, capsys
):
    locked_candidates = _audit_candidates(per_stratum=50, phase="locked")
    test_candidates = _audit_candidates(per_stratum=250, phase="test")
    _seal_aliases(tmp_path, [*locked_candidates, *test_candidates])
    _seal_candidates(tmp_path, locked_candidates, phase="locked")

    locked_build = _run_build(tmp_path, capsys, phase="locked", size=400)
    locked_metadata = json.loads(Path(locked_build["metadata"]).read_text())
    assert locked_metadata["sampling_seed"] == 20260713
    assert locked_metadata["phase"] == "locked"
    assert locked_metadata["size"] == 400
    assert locked_metadata["parser_source_hash"] == sha256_file(
        Path(cli.__file__).with_name("rlmf_format.py")
    )
    assert locked_metadata["normalization_version"]
    assert locked_metadata["alias_artifact_hash"]
    assert locked_metadata["candidate_endpoint"] == "audit_candidates_locked"
    assert locked_metadata["candidate_marker_hash"]
    assert locked_metadata["parent_sample_hash"] is None

    locked_manifest = _rating_manifest(tmp_path, Path(locked_build["ledger"]), "locked-400")
    assert (
        cli.main(
            [
                "rlmf-record-judge-audit",
                str(locked_manifest),
                "--config",
                str(CONFIG_PATH),
                "--phase",
                "locked",
                "--root",
                str(tmp_path),
            ]
        )
        == 0
    )
    locked_record = json.loads(capsys.readouterr().out)
    locked_marker = Path(locked_record["marker"])
    assert RLMFArtifactStore(tmp_path).verify_endpoint(
        _config().study_id, "locked_judge_audit"
    )

    _seal_candidates(
        tmp_path,
        test_candidates,
        phase="test",
        locked_hash=sha256_file(locked_marker),
    )
    first_build = _run_build(tmp_path, capsys, phase="test", size=1000)
    assert Path(first_build["sample"]).name == "test_1000_sample.jsonl"
    assert Path(first_build["ledger"]).name == "test_1000_ledger.jsonl"
    assert first_build["rows"] == 1000
    assert first_build["cumulative_size"] == 1000

    first_manifest = _rating_manifest(
        tmp_path, Path(first_build["ledger"]), "test-1000"
    )
    first_exit = cli.main(
        [
            "rlmf-record-judge-audit",
            str(first_manifest),
            "--config",
            str(CONFIG_PATH),
            "--phase",
            "test",
            "--root",
            str(tmp_path),
            "--size",
            "1000",
        ]
    )
    first_record = json.loads(capsys.readouterr().out)
    assert first_exit != 0
    assert first_record["status"] == "endpoint_propagation_required"
    assert "passed" not in first_record
    assert "bounded" not in first_record
    assert RLMFArtifactStore(tmp_path).verify_endpoint(
        _config().study_id, "test_judge_audit_evidence_1000"
    )
    endpoints = locked_marker.parent
    assert not (endpoints / "test_judge_audit.complete.json").exists()
    assert not list(locked_marker.parent.parent.glob("audits/test_differential_bias.json"))

    completed_1000 = _read_jsonl(Path(first_record["completed"]))
    with pytest.raises(ValueError, match="extension request"):
        _run_build(tmp_path, capsys, phase="test", size=1250)
    _seal_extension_request(tmp_path, from_size=1000, requested_size=1250)
    second_build = _run_build(tmp_path, capsys, phase="test", size=1250)
    ledger_1250 = _read_jsonl(Path(second_build["ledger"]))
    appended_payload = _read_jsonl(Path(second_build["sample"]))

    assert second_build["rows"] == 250
    assert second_build["cumulative_size"] == 1250
    assert ledger_1250[:1000] == completed_1000
    assert len(appended_payload) == 250
    assert json.loads(Path(second_build["metadata"]).read_text())[
        "parent_sample_hash"
    ] == sha256_file(Path(first_record["completed"]))

    second_manifest = _rating_manifest(
        tmp_path, Path(second_build["ledger"]), "test-1250"
    )
    second_args = [
        "rlmf-record-judge-audit",
        str(second_manifest),
        "--config",
        str(CONFIG_PATH),
        "--phase",
        "test",
        "--root",
        str(tmp_path),
        "--size",
        "1250",
    ]
    assert cli.main(second_args) != 0
    second_record = json.loads(capsys.readouterr().out)
    assert second_record["status"] == "endpoint_propagation_required"
    assert RLMFArtifactStore(tmp_path).verify_endpoint(
        _config().study_id, "test_judge_audit_evidence_1250"
    )

    # A retry verifies and reuses byte-identical partial artifacts instead of overwriting them.
    assert cli.main(second_args) != 0
    assert json.loads(capsys.readouterr().out) == second_record


def test_rating_manifest_requires_distinct_identities_ordered_timestamps_and_no_hidden_fields(
    tmp_path, capsys
):
    candidates = _audit_candidates(per_stratum=50, phase="locked")
    _seal_aliases(tmp_path, candidates)
    _seal_candidates(tmp_path, candidates, phase="locked")
    build = _run_build(tmp_path, capsys, phase="locked", size=400)
    manifest_path = _rating_manifest(tmp_path, Path(build["ledger"]), "invalid")
    manifest = json.loads(manifest_path.read_text())
    manifest["rater_b"]["identity"] = manifest["rater_a"]["identity"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="distinct"):
        cli.main(
            [
                "rlmf-record-judge-audit",
                str(manifest_path),
                "--config",
                str(CONFIG_PATH),
                "--phase",
                "locked",
                "--root",
                str(tmp_path),
            ]
        )

    manifest["rater_b"]["identity"] = "rater-b"
    manifest["adjudication"]["timestamp"] = "2029-12-31T23:59:59+00:00"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="after both"):
        cli.main(
            [
                "rlmf-record-judge-audit",
                str(manifest_path),
                "--config",
                str(CONFIG_PATH),
                "--phase",
                "locked",
                "--root",
                str(tmp_path),
            ]
        )

    rater_a = manifest_path.parent / manifest["rater_a"]["path"]
    rows = _read_jsonl(rater_a)
    rows[0]["proxy_label"] = True
    _write_jsonl(rater_a, rows)
    manifest["rater_a"]["sha256"] = sha256_file(rater_a)
    manifest["adjudication"]["timestamp"] = "2030-01-01T00:00:03+00:00"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hidden metadata"):
        cli.main(
            [
                "rlmf-record-judge-audit",
                str(manifest_path),
                "--config",
                str(CONFIG_PATH),
                "--phase",
                "locked",
                "--root",
                str(tmp_path),
            ]
        )


def _config():
    return RLMFConfig.from_json(CONFIG_PATH)


def _run_build(tmp_path, capsys, *, phase, size):
    args = [
        "rlmf-build-judge-audit",
        "--config",
        str(CONFIG_PATH),
        "--phase",
        phase,
        "--root",
        str(tmp_path),
        "--size",
        str(size),
    ]
    assert cli.main(args) == 0
    return json.loads(capsys.readouterr().out)


def _seal_aliases(tmp_path, candidates):
    store = RLMFArtifactStore(tmp_path)
    aliases_by_example = {
        row["example_id"]: row["gold_aliases"] for row in candidates
    }
    path = store.write_jsonl(
        _config().study_id,
        "data",
        "aliases",
        (
            {"example_id": example_id, "aliases": aliases}
            for example_id, aliases in sorted(aliases_by_example.items())
        ),
    )
    store.complete_endpoint(_config().study_id, "prepare-data", _config(), [path])


def _seal_candidates(tmp_path, candidates, *, phase, locked_hash=None):
    store = RLMFArtifactStore(tmp_path)
    path = store.write_jsonl(
        _config().study_id,
        "evaluation",
        f"audit_candidates_{phase}",
        candidates,
    )
    parents = {"locked_judge_audit": locked_hash} if locked_hash is not None else None
    store.complete_endpoint(
        _config().study_id,
        f"audit_candidates_{phase}",
        _config(),
        [path],
        parent_hashes=parents,
    )


def _seal_extension_request(tmp_path, *, from_size, requested_size):
    store = RLMFArtifactStore(tmp_path)
    evidence_marker = (
        tmp_path
        / "runs"
        / "rlmf"
        / _config().study_id
        / "endpoints"
        / f"test_judge_audit_evidence_{from_size}.complete.json"
    )
    request = store.write_json(
        _config().study_id,
        "audits",
        f"test_{requested_size}_extension_request",
        {
            "status": "extension_required",
            "estimand": "delta_cMFG_star",
            "from_size": from_size,
            "requested_size": requested_size,
        },
    )
    store.complete_endpoint(
        _config().study_id,
        f"test_judge_audit_extension_request_{requested_size}",
        _config(),
        [request],
        parent_hashes={"test_judge_audit_evidence": sha256_file(evidence_marker)},
    )


def _rating_manifest(tmp_path, ledger_path, stem):
    rows = _read_jsonl(ledger_path)
    pending = [row for row in rows if row["rater_a"] is None and row["rater_b"] is None]
    ratings = [
        {
            "audit_id": row["audit_id"],
            "label": "correct" if row["proxy_label"] else "incorrect",
        }
        for row in pending
    ]
    directory = tmp_path / "manual"
    directory.mkdir(exist_ok=True)
    paths = {
        "rater_a": directory / f"{stem}-rater-a.jsonl",
        "rater_b": directory / f"{stem}-rater-b.jsonl",
        "adjudication": directory / f"{stem}-adjudication.jsonl",
    }
    _write_jsonl(paths["rater_a"], ratings)
    _write_jsonl(paths["rater_b"], ratings)
    _write_jsonl(paths["adjudication"], [])
    manifest = {
        "schema_version": 1,
        "rater_a": {
            "identity": "rater-a",
            "timestamp": "2030-01-01T00:00:01+00:00",
            "path": paths["rater_a"].name,
            "sha256": sha256_file(paths["rater_a"]),
        },
        "rater_b": {
            "identity": "rater-b",
            "timestamp": "2030-01-01T00:00:02+00:00",
            "path": paths["rater_b"].name,
            "sha256": sha256_file(paths["rater_b"]),
        },
        "adjudication": {
            "identity": "adjudicator",
            "timestamp": "2030-01-01T00:00:03+00:00",
            "path": paths["adjudication"].name,
            "sha256": sha256_file(paths["adjudication"]),
        },
    }
    manifest_path = directory / f"{stem}-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _audit_candidates(*, per_stratum, phase):
    split = "validation" if phase == "locked" else "test"
    candidates = []
    for arm in ("standard_grpo", "rlmf"):
        for judgment_type in ("correctness", "equivalence"):
            for proxy_label in (False, True):
                for index in range(per_stratum):
                    candidate_id = f"{arm}-{judgment_type}-{proxy_label}-{index}"
                    candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "example_id": f"example-{candidate_id}",
                            "split": split,
                            "arm": arm,
                            "judgment_type": judgment_type,
                            "proxy_label": proxy_label,
                            "question": "Who wrote Hamlet?",
                            "answer": "William Shakespeare" if proxy_label else "Marlowe",
                            "comparison_answer": (
                                "William Shakespeare" if proxy_label else "Jonson"
                            ),
                            "reference_answer": "William Shakespeare",
                            "gold_aliases": ["william shakespeare"],
                        }
                    )
    return candidates


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
