import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import trajectory_extractor.fa_colab_screening as colab_screening
from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_config import FAConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/familiarity_answerability_gemma2_2b.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _materialize_source_v5(root: Path) -> None:
    source_root = root / "data/fa/confirmatory_source_v5"
    materialized = {}
    synthetic_files = {}
    for spec in colab_screening.SOURCE_V5_SPLITS:
        candidates = [
            {"entity_id": f"{spec.split}-{index}", "split": spec.split}
            for index in range(spec.candidate_count)
        ]
        questions = [
            {"question_id": f"{spec.split}-question-{index}"}
            for index in range(spec.completion_count)
        ]
        synthetic = [
            {"candidate_id": f"{spec.split}-synthetic-{index}", "split": spec.split}
            for index in range(spec.completion_count)
        ]
        candidate_path = root / spec.candidate_path
        question_path = root / spec.question_path
        synthetic_path = root / spec.synthetic_path
        _write_json(candidate_path, candidates)
        _write_json(question_path, questions)
        _write_json(synthetic_path, synthetic)
        materialized[spec.split] = {
            "candidate_manifest": str(spec.candidate_path),
            "candidate_sha256": _sha256(candidate_path),
            "question_manifest": str(spec.question_path),
            "question_sha256": _sha256(question_path),
        }
        synthetic_files[spec.split] = {
            "path": str(spec.synthetic_path),
            "sha256": _sha256(synthetic_path),
        }
    source_snapshot = source_root / "source_snapshot_v1.json"
    synthetic_snapshot = source_root / "synthetic_source_snapshot_v1.json"
    _write_json(source_snapshot, {"source_revision": "fa-confirmatory-wikidata-v5"})
    _write_json(
        synthetic_snapshot,
        {"generator_revision": "fa-confirmatory-pseudonyms-v3"},
    )
    _write_json(
        source_root / "source_integrity_v1.json",
        {
            "schema_version": 1,
            "source_revision": "fa-confirmatory-wikidata-v5",
            "materialized_files": materialized,
            "synthetic_files": synthetic_files,
            "source_snapshot": str(
                Path("data/fa/confirmatory_source_v5/source_snapshot_v1.json")
            ),
            "source_snapshot_sha256": _sha256(source_snapshot),
            "synthetic_snapshot": str(
                Path(
                    "data/fa/confirmatory_source_v5/"
                    "synthetic_source_snapshot_v1.json"
                )
            ),
            "synthetic_snapshot_sha256": _sha256(synthetic_snapshot),
        },
    )


def _execution_args(root: Path) -> argparse.Namespace:
    bundle = root / "drive/fa-study.bundle"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"frozen git bundle")
    commit = "a" * 40
    launch = root / "drive/fa-study-launch.json"
    _write_json(
        launch,
        {
            "schema_version": 1,
            "git_commit": commit,
            "bundle_file": bundle.name,
            "bundle_sha256": _sha256(bundle),
        },
    )
    return argparse.Namespace(
        checkpoint_root=str(root / "drive/checkpoints"),
        scratch_root=str(root / "scratch"),
        git_commit=commit,
        bundle_path=str(bundle),
        bundle_sha256=_sha256(bundle),
        launch_manifest=str(launch),
    )


def _install_runtime_fixture(root: Path) -> None:
    lock = root / "requirements/fa-core.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("torch==2.7.1\n", encoding="utf-8")


def _transaction_fakes(root: Path, config: FAConfig):
    store = FAArtifactStore(root)
    calls = {"generation": [], "screening": [], "assembly": 0}
    by_split = {spec.split: spec for spec in colab_screening.SOURCE_V5_SPLITS}

    def run_screening(observed_config, observed_root, args):
        assert observed_config == config
        assert observed_root.resolve() == root.resolve()
        spec = by_split[args.namespace]
        calls["generation"].append(args.namespace)
        shard = store.write_completed_shard(
            config.run_id,
            args.namespace,
            args.shard_id,
            [
                {
                    "kind": "screening_completion",
                    "status": "completed",
                    "index": index,
                }
                for index in range(spec.completion_count)
            ],
            {"config_sha256": config.config_hash},
            record_kind="screening_completion",
        )
        return {
            "status": "generated",
            "shard_manifest": str(shard.manifest_path),
            "count": spec.completion_count,
        }

    def screen_entities(observed_config, observed_root, args):
        assert observed_config == config
        completion = store.verify_shard(args.screening_manifest)
        spec = by_split[completion.namespace]
        calls["screening"].append(spec.split)
        audit = store.write_completed_shard(
            config.run_id,
            spec.split,
            f"audit-{spec.split}",
            [{"kind": "screening_audit", "decision": "passed"}],
            {"config_sha256": config.config_hash},
            record_kind="screening_audit",
        )
        matches = store.write_completed_shard(
            config.run_id,
            spec.split,
            f"matches-{spec.split}",
            [
                {"kind": "screened_match", "index": index}
                for index in range(spec.match_count)
            ],
            {"config_sha256": config.config_hash},
            record_kind="screened_match",
        )
        return {
            "status": "screened",
            "manifest": str(matches.manifest_path),
            "audit_manifest": str(audit.manifest_path),
            "count": spec.match_count,
        }

    def assemble(observed_config, observed_root, args):
        assert observed_config == config
        assert len(args.screened_matches_manifest) == 5
        calls["assembly"] += 1
        collection = store.write_completed_shard(
            config.run_id,
            "mechanism_train",
            args.shard_id,
            [
                {"kind": "screened_match_collection", "index": index}
                for index in range(colab_screening.ASSEMBLED_MATCH_COUNT)
            ],
            {"config_sha256": config.config_hash},
            record_kind="screened_match_collection",
        )
        return {
            "status": "assembled",
            "manifest": str(collection.manifest_path),
            "count": colab_screening.ASSEMBLED_MATCH_COUNT,
            "matches_sha256": collection.sha256,
        }

    return calls, run_screening, screen_entities, assemble


def test_colab_screening_runs_exact_source_v5_counts_and_stops_at_collection(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    _materialize_source_v5(root)
    _install_runtime_fixture(root)
    args = _execution_args(root)
    config = FAConfig.from_json(CONFIG_PATH)
    calls, run_screening, screen_entities, assemble = _transaction_fakes(
        root, config
    )
    monkeypatch.setattr(
        colab_screening,
        "_verify_frozen_checkout",
        lambda *arguments: None,
    )
    lock_sha256 = _sha256(root / "requirements/fa-core.lock")
    monkeypatch.setattr(
        colab_screening,
        "run_colab_preflight",
        lambda observed_root, lock: {
            "status": "ready",
            "lock_sha256": lock_sha256,
            "torch_version": "2.7.1",
            "transformers_version": "4.57.1",
            "accelerate_version": "1.12.0",
            "cuda_available": True,
            "gpu_name": "NVIDIA Test GPU",
        },
    )

    payload = colab_screening.run_colab_screening(
        config,
        root,
        args,
        run_screening=run_screening,
        screen_entities=screen_entities,
        assemble_screened_matches=assemble,
    )

    expected_splits = [spec.split for spec in colab_screening.SOURCE_V5_SPLITS]
    assert calls == {
        "generation": expected_splits,
        "screening": expected_splits,
        "assembly": 1,
    }
    assert payload["status"] == "assembled"
    assert payload["count"] == 244
    assert payload["screening_completion_count"] == 1152
    assert payload["source_candidate_count"] == 384
    assert payload["protected_endpoints_accessed"] is False
    assert payload["stopped_before"] == "naturalness_and_f1_f2a"
    assert Path(payload["execution_identity_path"]).is_file()
    assert Path(payload["runtime_observation_path"]).is_file()
    assert FAArtifactStore(root).verify_shard(payload["manifest"]).row_count == 244


def test_colab_screening_restores_all_split_checkpoints_without_rerunning(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    _materialize_source_v5(root)
    _install_runtime_fixture(root)
    args = _execution_args(root)
    config = FAConfig.from_json(CONFIG_PATH)
    _, run_screening, screen_entities, assemble = _transaction_fakes(root, config)
    monkeypatch.setattr(
        colab_screening,
        "_verify_frozen_checkout",
        lambda *arguments: None,
    )
    lock_sha256 = _sha256(root / "requirements/fa-core.lock")
    monkeypatch.setattr(
        colab_screening,
        "run_colab_preflight",
        lambda observed_root, lock: {
            "status": "ready",
            "lock_sha256": lock_sha256,
            "cuda_available": True,
        },
    )
    first = colab_screening.run_colab_screening(
        config,
        root,
        args,
        run_screening=run_screening,
        screen_entities=screen_entities,
        assemble_screened_matches=assemble,
    )
    shutil.rmtree(root / "runs")

    def unexpected(*arguments, **keywords):
        raise AssertionError("completed screening transaction reran after restore")

    resumed = colab_screening.run_colab_screening(
        config,
        root,
        args,
        run_screening=unexpected,
        screen_entities=unexpected,
        assemble_screened_matches=unexpected,
    )

    assert resumed["status"] == "assembled"
    assert resumed["manifest"] == first["manifest"]
    assert resumed["restored_splits"] == [
        spec.split for spec in colab_screening.SOURCE_V5_SPLITS
    ]


def test_colab_screening_rejects_source_count_drift_before_generation(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    _materialize_source_v5(root)
    _install_runtime_fixture(root)
    args = _execution_args(root)
    config = FAConfig.from_json(CONFIG_PATH)
    candidate_path = root / colab_screening.SOURCE_V5_SPLITS[0].candidate_path
    rows = json.loads(candidate_path.read_text(encoding="utf-8"))
    _write_json(candidate_path, rows[:-1])
    integrity_path = (
        root / "data/fa/confirmatory_source_v5/source_integrity_v1.json"
    )
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["materialized_files"]["mechanism_train"][
        "candidate_sha256"
    ] = _sha256(candidate_path)
    _write_json(integrity_path, integrity)
    monkeypatch.setattr(
        colab_screening,
        "_verify_frozen_checkout",
        lambda *arguments: None,
    )

    with pytest.raises(ValueError, match="candidate count"):
        colab_screening.run_colab_screening(
            config,
            root,
            args,
            run_screening=lambda *arguments: pytest.fail(
                "generation accessed after source drift"
            ),
            screen_entities=lambda *arguments: pytest.fail(
                "screening accessed after source drift"
            ),
            assemble_screened_matches=lambda *arguments: pytest.fail(
                "assembly accessed after source drift"
            ),
        )


def test_frozen_checkout_requires_declared_commit_in_bundle_heads(
    tmp_path, monkeypatch
):
    requested_commit = "a" * 40

    def fake_git_output(repo_root, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return requested_commit
        if arguments == ("ls-files", "--others", "--exclude-standard"):
            return ""
        if arguments[:2] == ("bundle", "list-heads"):
            return f"{'b' * 40} refs/heads/other"
        return ""

    monkeypatch.setattr(colab_screening, "_git_output", fake_git_output)

    with pytest.raises(RuntimeError, match="not advertised by the Git bundle"):
        colab_screening._verify_frozen_checkout(
            tmp_path,
            requested_commit,
            tmp_path / "fa-study.bundle",
        )
