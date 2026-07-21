import json
from pathlib import Path

import pytest

from trajectory_extractor.fa_config import (
    CONFIRMATORY_SPLIT_COUNTS,
    NON_CONFIRMATORY_NAMESPACES,
    FAConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_CONFIG = REPO_ROOT / "configs" / "familiarity_answerability_gemma2_2b.json"
SMOKE_CONFIG = REPO_ROOT / "configs" / "familiarity_answerability_qwen06b_smoke.json"


def test_confirmatory_config_is_canonical_and_pinned():
    config = FAConfig.from_json(CONFIRMATORY_CONFIG)

    assert config.profile == "confirmatory"
    assert config.model_id == "google/gemma-2-2b-it"
    assert len(config.model_revision) == 40
    assert dict(config.split_counts) == CONFIRMATORY_SPLIT_COUNTS
    assert set(config.split_counts).isdisjoint(NON_CONFIRMATORY_NAMESPACES)
    assert len(config.config_hash) == 64
    assert config.canonical_bytes == json.dumps(
        json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8")),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_config_rejects_mutable_revision_and_unknown_split(tmp_path):
    payload = json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8"))
    payload["model_revision"] = "main"
    path = tmp_path / "mutable.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable revision"):
        FAConfig.from_json(path)

    payload = json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8"))
    payload["split_counts"]["unregistered"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="registered split"):
        FAConfig.from_json(path)


def test_config_rejects_nonfinite_threshold_and_confirmatory_qwen(tmp_path):
    payload = json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8"))
    payload["thresholds"]["h1_min_interaction"] = float("nan")
    path = tmp_path / "nonfinite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        FAConfig.from_json(path)

    payload = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    payload["profile"] = "confirmatory"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Qwen"):
        FAConfig.from_json(path)


def test_smoke_config_uses_only_nonconfirmatory_namespaces():
    config = FAConfig.from_json(SMOKE_CONFIG)

    assert config.profile == "smoke"
    assert set(config.split_counts) == NON_CONFIRMATORY_NAMESPACES
    assert config.thresholds == {}
    assert config.anchors == ()
