import json
from pathlib import Path

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
