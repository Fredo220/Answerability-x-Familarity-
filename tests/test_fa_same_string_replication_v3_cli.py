from __future__ import annotations

import argparse
import json
from pathlib import Path

import trajectory_extractor.fa_cli as fa_cli
from trajectory_extractor.fa_config import FAConfig

from test_fa_same_string_replication_v3 import _CharacterTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "familiarity_answerability_same_string_replication_v3.json"


def test_prepare_replication_v3_freezes_corpus_and_sensitivity(monkeypatch, tmp_path):
    config = FAConfig.from_json(CONFIG)
    tokenizer = _CharacterTokenizer()
    monkeypatch.setattr(
        fa_cli,
        "load_pinned_tokenizer",
        lambda _config: argparse.Namespace(
            tokenizer=tokenizer,
            chat_template_sha256=config.chat_template_sha256,
        ),
    )
    args = argparse.Namespace(output_dir=str(tmp_path / "release"))

    result = fa_cli._prepare_replication_v3(config, tmp_path, args)

    assert result["status"] == "frozen"
    assert result["row_count"] == 320
    assert Path(result["corpus_manifest"]).is_file()
    sensitivity = json.loads(Path(result["sensitivity_manifest"]).read_text())
    assert sensitivity["outcomes_opened"] is False
    assert sensitivity["corpus_manifest_sha256"] == result["corpus_manifest_sha256"]
