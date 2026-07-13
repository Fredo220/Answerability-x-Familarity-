import json
from pathlib import Path

import pytest

from trajectory_extractor import cli


def test_rlmf_prepare_data_writes_snapshot_and_reports_registered_provenance(tmp_path, monkeypatch, capsys):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "rlmf_qwen06b_confirmatory.json"
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

    exit_code = cli.main(["rlmf-prepare-data", "--config", str(config_path), "--root", str(tmp_path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert captured["config"].study_id == "rlmf-qwen06b-v1"
    assert captured["store"].root == tmp_path
    assert payload == {
        "study_id": "rlmf-qwen06b-v1",
        "count": 896,
        "split_counts": {"pre_sft": 256, "rl_train": 256, "validation": 128, "test": 256},
        "dataset_revision": "5cf59972d88d4aaaa7781ac91b83d053563d8268",
        "artifacts": {name: str(path) for name, path in expected.items()},
    }


def test_rlmf_audit_cli_builds_blinded_sample_and_test_recording_never_revises_proxy(
    tmp_path, monkeypatch, capsys
):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "rlmf_qwen06b_confirmatory.json"
    candidates = _audit_candidates(per_stratum=125)
    source = tmp_path / "runs" / "rlmf" / "rlmf-qwen06b-v1" / "evaluation"
    source.mkdir(parents=True)
    candidate_path = source / "audit_candidates_test.jsonl"
    candidate_path.write_text("".join(json.dumps(candidate) + "\n" for candidate in candidates))

    assert cli.main([
        "rlmf-build-judge-audit", "--config", str(config_path), "--phase", "test", "--root", str(tmp_path)
    ]) == 0
    build_payload = json.loads(capsys.readouterr().out)
    sample_path = Path(build_payload["sample"])
    sample_rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
    assert len(sample_rows) == 1000
    assert not {"arm", "seed", "confidence", "reward", "model_id", "proxy_label"} & set(sample_rows[0])

    manual_path = tmp_path / "manual_test.jsonl"
    manual_path.write_text(
        "".join(
            json.dumps(
                {
                    "audit_id": row["audit_id"],
                    "rater_a": "correct" if index % 2 else "incorrect",
                    "rater_b": "correct" if index % 2 else "incorrect",
                    "proxy_label": True,
                }
            )
            + "\n"
            for index, row in enumerate(sample_rows)
        )
    )
    with pytest.raises(ValueError, match="proxy"):
        cli.main([
            "rlmf-record-judge-audit", str(manual_path), "--config", str(config_path), "--phase", "test", "--root", str(tmp_path)
        ])


def _audit_candidates(*, per_stratum):
    candidates = []
    for arm in ("standard_grpo", "rlmf"):
        for judgment_type in ("correctness", "equivalence"):
            for proxy_label in (False, True):
                for index in range(per_stratum):
                    candidates.append(
                        {
                            "candidate_id": f"{arm}-{judgment_type}-{proxy_label}-{index}",
                            "arm": arm,
                            "judgment_type": judgment_type,
                            "proxy_label": proxy_label,
                            "question": "Who wrote Hamlet?",
                            "answer": "William Shakespeare",
                            "comparison_answer": "William Shakespeare",
                            "reference_answer": "William Shakespeare",
                        }
                    )
    return candidates
