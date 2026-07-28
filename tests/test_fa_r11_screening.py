import hashlib
import json
from pathlib import Path

import pytest

from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_r11_screening import run_r11_screening


class FakeRunner:
    def render_prompt(self, prompt: str) -> str:
        return f"rendered::{prompt}"

    def generate(self, prompts, generation):
        del generation
        return ["Paris" if "capital" in prompt else "France" for prompt in prompts]


class FailingRunner(FakeRunner):
    def generate(self, prompts, generation):
        raise AssertionError("verified checkpoints must resume without generation")


def _write_source(root: Path) -> Path:
    rows = []
    for split in ("instrument_development", "construction_validation"):
        rows.extend(
            [
                {
                    "schema_version": 1,
                    "kind": "fa_r11_screening_prompt",
                    "split": split,
                    "domain": "place",
                    "entity_id": f"r11-{split}-place-q1",
                    "qid": "Q1",
                    "entity_name": "Example",
                    "sitelinks": 10,
                    "relation_id": "P17",
                    "prompt": "Which country is Example in?",
                    "accepted_aliases": ["France", "France."],
                    "source_provenance": "test",
                },
                {
                    "schema_version": 1,
                    "kind": "fa_r11_screening_prompt",
                    "split": split,
                    "domain": "place",
                    "entity_id": f"r11-{split}-place-q1",
                    "qid": "Q1",
                    "entity_name": "Example",
                    "sitelinks": 10,
                    "relation_id": "P36",
                    "prompt": "What is the capital of Example?",
                    "accepted_aliases": ["Paris", "Paris."],
                    "source_provenance": "test",
                },
            ]
        )
    rows_path = root / "screening_prompts_v1.jsonl"
    encoded = "".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
        for row in rows
    )
    rows_path.write_text(encoded, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "kind": "fa_r11_source_manifest",
        "source_revision": "fa-development-source-v6-r11",
        "rows_file": rows_path.name,
        "rows_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }
    (root / "source_manifest_v1.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    audit_items = [
        {
            "entity_id": row["entity_id"],
            "relation_id": row["relation_id"],
            "structural_pass": True,
            "semantic_pass": True,
            "structural_auditor_id": "auditor-structural",
            "semantic_auditor_id": "auditor-semantic",
        }
        for row in rows
    ]
    audit_items_path = root / "pre_model_semantic_audit_items_v1.jsonl"
    audit_items_encoded = "".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
        for row in audit_items
    )
    audit_items_path.write_text(audit_items_encoded, encoding="utf-8")
    audit = {
        "schema_version": 1,
        "kind": "fa_r11_pre_model_semantic_audit",
        "status": "passed",
        "blocker_count": 0,
        "rows_sha256": manifest["rows_sha256"],
        "items_file": audit_items_path.name,
        "items_sha256": hashlib.sha256(audit_items_encoded.encode()).hexdigest(),
        "item_count": len(audit_items),
    }
    audit_path = root / "pre_model_semantic_audit_v1.json"
    audit_path.write_text(json.dumps(audit, sort_keys=True), encoding="utf-8")
    return audit_path


def test_run_r11_screening_preserves_relation_identity(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    audit_path = _write_source(source_root)
    config = FAConfig.from_json(
        "configs/familiarity_answerability_qwen06b_smoke.json"
    )

    result = run_r11_screening(
        config,
        source_root=source_root,
        split="instrument_development",
        output_root=tmp_path / "output",
        pre_model_semantic_audit=audit_path,
        batch_size=2,
        runner=FakeRunner(),
        git_commit="a" * 40,
    )

    items = [
        json.loads(line)
        for line in Path(result["items_path"]).read_text().splitlines()
    ]
    assert {row["relation_id"] for row in items} == {"P17", "P36"}
    assert all(row["is_correct"] for row in items)
    assert result["resumed_batch_count"] == 0

    resumed = run_r11_screening(
        config,
        source_root=source_root,
        split="instrument_development",
        output_root=tmp_path / "output",
        pre_model_semantic_audit=audit_path,
        batch_size=2,
        runner=FailingRunner(),
        git_commit="a" * 40,
    )
    assert resumed["resumed_batch_count"] == 1

    batch_path = (
        Path(result["items_path"]).parent / "batch-000000.jsonl"
    )
    tampered = [
        json.loads(line) for line in batch_path.read_text().splitlines()
    ]
    tampered[0]["is_correct"] = False
    batch_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in tampered)
    )
    with pytest.raises(ValueError, match="checkpoint"):
        run_r11_screening(
            config,
            source_root=source_root,
            split="instrument_development",
            output_root=tmp_path / "output",
            pre_model_semantic_audit=audit_path,
            batch_size=2,
            runner=FailingRunner(),
            git_commit="a" * 40,
        )
