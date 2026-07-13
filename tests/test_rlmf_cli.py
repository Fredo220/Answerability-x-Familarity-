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
    development_candidates = _audit_candidates(per_stratum=25, phase="development")
    locked_candidates = _audit_candidates(per_stratum=50, phase="locked")
    test_candidates = _audit_candidates(per_stratum=250, phase="test")
    _seal_aliases(
        tmp_path, [*development_candidates, *locked_candidates, *test_candidates]
    )
    development_marker = _complete_development_audit(
        tmp_path, capsys, development_candidates
    )
    _seal_candidates(
        tmp_path,
        locked_candidates,
        phase="locked",
        development_hash=sha256_file(development_marker),
    )

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
    assert locked_metadata["development_judge_audit_marker_hash"] == sha256_file(
        development_marker
    )
    assert len(locked_metadata["sampling_design"]["strata"]) == 8
    for stratum in locked_metadata["sampling_design"]["strata"].values():
        assert stratum["population_count"] == 50
        assert stratum["sample_count"] == 50
        assert stratum["inclusion_probability"] == {"numerator": 1, "denominator": 1}

    locked_manifest = _sealed_rating_manifest(
        tmp_path, capsys, Path(locked_build["ledger"]), "locked-400", phase="locked", size=400
    )
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
    locked_endpoint = RLMFArtifactStore(tmp_path).verify_endpoint(
        _config().study_id, "locked_judge_audit"
    )
    assert locked_endpoint["parent_hashes"]["development_judge_audit"] == sha256_file(
        development_marker
    )
    for role in ("rater_a", "rater_b", "adjudication"):
        assert locked_endpoint["parent_hashes"][f"{role}_endpoint"]
    persisted_sources = json.loads(
        (
            locked_marker.parent.parent
            / "audits"
            / "locked_400_rating_sources.json"
        ).read_text()
    )
    assert persisted_sources["schema_version"] == 2
    assert all(
        "endpoint" in persisted_sources["sources"][role]
        for role in ("rater_a", "rater_b", "adjudication")
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

    first_manifest = _sealed_rating_manifest(
        tmp_path,
        capsys,
        Path(first_build["ledger"]),
        "test-1000",
        phase="test",
        size=1000,
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

    second_manifest = _sealed_rating_manifest(
        tmp_path,
        capsys,
        Path(second_build["ledger"]),
        "test-1250",
        phase="test",
        size=1250,
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


def test_locked_build_requires_postdevelopment_candidate_and_exact_proxy_freeze(
    tmp_path, capsys, monkeypatch
):
    development = _audit_candidates(per_stratum=25, phase="development")
    candidates = _audit_candidates(per_stratum=50, phase="locked")
    _seal_aliases(tmp_path, [*development, *candidates])
    _seal_candidates(tmp_path, candidates, phase="locked")

    with pytest.raises(ValueError, match="development_judge_audit"):
        _run_build(tmp_path, capsys, phase="locked", size=400)

    wrong_order_root = tmp_path / "wrong-order"
    _seal_aliases(wrong_order_root, [*development, *candidates])
    _seal_candidates(wrong_order_root, candidates, phase="locked")
    development_marker = _complete_development_audit(
        wrong_order_root, capsys, development
    )
    with pytest.raises(ValueError, match="postdate|bind"):
        _run_build(wrong_order_root, capsys, phase="locked", size=400)

    changed_parser_root = tmp_path / "changed-parser"
    _seal_aliases(changed_parser_root, [*development, *candidates])
    development_marker = _complete_development_audit(
        changed_parser_root, capsys, development
    )
    _seal_candidates(
        changed_parser_root,
        candidates,
        phase="locked",
        development_hash=sha256_file(development_marker),
    )
    monkeypatch.setattr(cli, "PARSER_VERSION", "rlmf-output-parser-changed")
    with pytest.raises(ValueError, match="development proxy freeze"):
        _run_build(changed_parser_root, capsys, phase="locked", size=400)

    changed_alias_root = tmp_path / "changed-alias"
    _seal_aliases(changed_alias_root, [*development, *candidates])
    development_marker = _complete_development_audit(
        changed_alias_root, capsys, development
    )
    _seal_candidates(
        changed_alias_root,
        candidates,
        phase="locked",
        development_hash=sha256_file(development_marker),
    )
    aliases = (
        changed_alias_root / "runs" / "rlmf" / _config().study_id / "data" / "aliases.jsonl"
    )
    aliases.write_text(aliases.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        _run_build(changed_alias_root, capsys, phase="locked", size=400)


@pytest.mark.parametrize("same_path", [True, False])
def test_rater_endpoints_reject_same_source_path_or_bytes(
    tmp_path, capsys, same_path
):
    development = _audit_candidates(per_stratum=25, phase="development")
    _seal_aliases(tmp_path, development)
    _seal_candidates(tmp_path, development, phase="development")
    build = _run_build(tmp_path, capsys, phase="development", size=200)
    rows = _pending_ratings(Path(build["ledger"]))
    manual = tmp_path / "manual"
    manual.mkdir()
    rater_a = manual / "rater-a.jsonl"
    rater_b = rater_a if same_path else manual / "rater-b.jsonl"
    _write_jsonl(rater_a, rows)
    if not same_path:
        _write_jsonl(rater_b, rows)
    _seal_rating(tmp_path, capsys, rater_a, role="rater_a", phase="development", size=200)

    with pytest.raises(ValueError, match="distinct.*path|distinct.*hash"):
        _seal_rating(
            tmp_path, capsys, rater_b, role="rater_b", phase="development", size=200
        )


def test_adjudication_requires_both_rater_endpoints_and_final_record_uses_marker_order(
    tmp_path, capsys
):
    development = _audit_candidates(per_stratum=25, phase="development")
    _seal_aliases(tmp_path, development)
    _seal_candidates(tmp_path, development, phase="development")
    build = _run_build(tmp_path, capsys, phase="development", size=200)
    ratings = _pending_ratings(Path(build["ledger"]))
    manual = tmp_path / "manual"
    manual.mkdir()
    rater_a = manual / "rater-a.jsonl"
    rater_b = manual / "rater-b.jsonl"
    adjudication = manual / "adjudication.jsonl"
    _write_jsonl(rater_a, ratings)
    _write_jsonl(rater_b, reversed(ratings))
    _write_jsonl(adjudication, [])
    _seal_rating(tmp_path, capsys, rater_a, role="rater_a", phase="development", size=200)

    with pytest.raises(ValueError, match="rater_b"):
        _seal_adjudication(
            tmp_path, capsys, adjudication, phase="development", size=200
        )

    _seal_rating(tmp_path, capsys, rater_b, role="rater_b", phase="development", size=200)
    adjudication_result = _seal_adjudication(
        tmp_path, capsys, adjudication, phase="development", size=200
    )
    adjudication_marker = Path(adjudication_result["marker"])
    marker_record = json.loads(adjudication_marker.read_text())
    marker_record["created_at"] = "2000-01-01T00:00:00+00:00"
    adjudication_marker.write_text(json.dumps(marker_record, indent=2, sort_keys=True))
    manifest = _endpoint_manifest(
        tmp_path, phase="development", size=200, stem="too-early"
    )

    with pytest.raises(ValueError, match="adjudication.*after both"):
        cli.main(
            [
                "rlmf-record-judge-audit",
                str(manifest),
                "--config",
                str(CONFIG_PATH),
                "--phase", "development",
                "--root",
                str(tmp_path),
            ]
        )


def test_final_record_rejects_missing_development_endpoint_and_fabricated_manifest_time(
    tmp_path, capsys
):
    development = _audit_candidates(per_stratum=25, phase="development")
    locked = _audit_candidates(per_stratum=50, phase="locked")
    _seal_aliases(tmp_path, [*development, *locked])
    development_marker = _complete_development_audit(tmp_path, capsys, development)
    _seal_candidates(
        tmp_path,
        locked,
        phase="locked",
        development_hash=sha256_file(development_marker),
    )
    build = _run_build(tmp_path, capsys, phase="locked", size=400)
    manifest_path = _sealed_rating_manifest(
        tmp_path,
        capsys,
        Path(build["ledger"]),
        "locked-record-adversarial",
        phase="locked",
        size=400,
    )
    args = [
        "rlmf-record-judge-audit", str(manifest_path),
        "--config", str(CONFIG_PATH),
        "--phase", "locked",
        "--root", str(tmp_path),
    ]
    hidden_marker = development_marker.with_suffix(".removed")
    development_marker.rename(hidden_marker)
    with pytest.raises(ValueError, match="development_judge_audit"):
        cli.main(args)
    hidden_marker.rename(development_marker)

    manifest = json.loads(manifest_path.read_text())
    manifest["rater_a"]["timestamp"] = "1999-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="endpoint schema"):
        cli.main(args)


def test_cli_help_documents_staged_human_input_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "rlmf-seal-judge-rating" in output
    assert "rlmf-seal-judge-adjudication" in output



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


def _seal_candidates(
    tmp_path, candidates, *, phase, locked_hash=None, development_hash=None
):
    store = RLMFArtifactStore(tmp_path)
    path = store.write_jsonl(
        _config().study_id,
        "evaluation",
        f"audit_candidates_{phase}",
        candidates,
    )
    parents = {}
    if locked_hash is not None:
        parents["locked_judge_audit"] = locked_hash
    if development_hash is not None:
        parents["development_judge_audit"] = development_hash
    store.complete_endpoint(
        _config().study_id,
        f"audit_candidates_{phase}",
        _config(),
        [path],
        parent_hashes=parents or None,
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


def _pending_ratings(ledger_path):
    rows = _read_jsonl(ledger_path)
    pending = [row for row in rows if row["rater_a"] is None and row["rater_b"] is None]
    return [
        {
            "audit_id": row["audit_id"],
            "label": "correct" if row["proxy_label"] else "incorrect",
        }
        for row in pending
    ]


def _seal_rating(tmp_path, capsys, path, *, role, phase, size):
    assert cli.main(
        [
            "rlmf-seal-judge-rating",
            str(path),
            "--identity", f"{role}-identity",
            "--role", role,
            "--config", str(CONFIG_PATH),
            "--phase", phase,
            "--size", str(size),
            "--root", str(tmp_path),
        ]
    ) == 0
    return json.loads(capsys.readouterr().out)


def _seal_adjudication(tmp_path, capsys, path, *, phase, size):
    result = cli.main(
        [
            "rlmf-seal-judge-adjudication",
            str(path),
            "--identity", "adjudicator-identity",
            "--config", str(CONFIG_PATH),
            "--phase", phase,
            "--size", str(size),
            "--root", str(tmp_path),
        ]
    )
    if result != 0:
        return result
    return json.loads(capsys.readouterr().out)


def _endpoint_manifest(tmp_path, *, phase, size, stem):
    endpoints = tmp_path / "runs" / "rlmf" / _config().study_id / "endpoints"
    manifest = {"schema_version": 2}
    for role in ("rater_a", "rater_b", "adjudication"):
        endpoint = f"{phase}_judge_audit_{role}_{size}"
        marker = endpoints / f"{endpoint}.complete.json"
        manifest[role] = {
            "endpoint": endpoint,
            "marker_sha256": sha256_file(marker),
        }
    directory = tmp_path / "manual"
    directory.mkdir(exist_ok=True)
    path = directory / f"{stem}-endpoint-manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _sealed_rating_manifest(tmp_path, capsys, ledger_path, stem, *, phase, size):
    ratings = _pending_ratings(ledger_path)
    directory = tmp_path / "manual"
    directory.mkdir(exist_ok=True)
    paths = {
        "rater_a": directory / f"{stem}-rater-a.jsonl",
        "rater_b": directory / f"{stem}-rater-b.jsonl",
        "adjudication": directory / f"{stem}-adjudication.jsonl",
    }
    _write_jsonl(paths["rater_a"], ratings)
    _write_jsonl(paths["rater_b"], reversed(ratings))
    _write_jsonl(paths["adjudication"], [])
    _seal_rating(
        tmp_path, capsys, paths["rater_a"], role="rater_a", phase=phase, size=size
    )
    _seal_rating(
        tmp_path, capsys, paths["rater_b"], role="rater_b", phase=phase, size=size
    )
    _seal_adjudication(
        tmp_path, capsys, paths["adjudication"], phase=phase, size=size
    )
    return _endpoint_manifest(tmp_path, phase=phase, size=size, stem=stem)


def _complete_development_audit(tmp_path, capsys, candidates):
    _seal_candidates(tmp_path, candidates, phase="development")
    build = _run_build(tmp_path, capsys, phase="development", size=200)
    manifest = _sealed_rating_manifest(
        tmp_path,
        capsys,
        Path(build["ledger"]),
        "development-200",
        phase="development",
        size=200,
    )
    assert cli.main(
        [
            "rlmf-record-judge-audit", str(manifest),
            "--config", str(CONFIG_PATH),
            "--phase", "development",
            "--root", str(tmp_path),
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    return Path(result["marker"])


def _audit_candidates(*, per_stratum, phase):
    if phase == "development":
        groups = ((split, None) for split in ("pre_sft", "rl_train"))
    else:
        split = "validation" if phase == "locked" else "test"
        groups = ((split, arm) for arm in ("standard_grpo", "rlmf"))
    candidates = []
    for split, arm in groups:
        for judgment_type in ("correctness", "equivalence"):
            for proxy_label in (False, True):
                for index in range(per_stratum):
                    candidate_id = f"{arm or split}-{judgment_type}-{proxy_label}-{index}"
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
