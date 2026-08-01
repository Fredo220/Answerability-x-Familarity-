from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "fa_same_string_primary_colab.ipynb"
RUNBOOK = ROOT / "docs" / "fa_same_string_primary_runbook.md"
README = ROOT / "README.md"


def notebook_text() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ())) for cell in payload.get("cells", ())
    )


def test_same_string_colab_is_thin_pinned_and_secret_safe():
    text = notebook_text()

    assert re.search(r'PINNED_REPO_COMMIT\s*=\s*"[0-9a-f]{40}"', text)
    assert "git checkout" in text
    assert "pip install --no-deps -e" in text
    assert 'os.environ.get("HF_TOKEN")' in text
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert "/content/drive/MyDrive" in text
    assert "resume" in text.casefold()


def test_same_string_colab_calls_registered_cli_in_gate_order():
    text = notebook_text()
    commands = (
        "fa-prepare-same-string-matches",
        "fa-prepare-naturalness-ratings",
        "fa-compile-naturalness-ratings",
        "fa-build-same-string-confirmatory",
        "fa-audit-manifest",
        "fa-seal-behavior-test",
        "fa-evaluate-behavior-test",
    )

    positions = [text.index(command) for command in commands]
    assert positions == sorted(positions)
    assert "fa-finalize-naturalness-adjudication" in text
    assert "configs/familiarity_answerability_qwen17b_smoke.json" in text
    assert "behavior_test" in text
    assert "same_string_confirmatory_index" in text


def test_same_string_colab_uses_durable_root_and_self_contained_smoke():
    text = notebook_text()
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    run_cell = next(
        "".join(cell.get("source", ()))
        for cell in payload["cells"]
        if "def run_fa" in "".join(cell.get("source", ()))
    )

    assert 'ARTIFACT_ROOT = Path("/content/fa-same-string-artifacts")' in text
    assert 'DRIVE_CHECKPOINT_ROOT = Path("/content/drive/MyDrive' in text
    assert "LOCAL_ARTIFACT_ROOT" not in text
    assert "checkpoint()" in text
    assert "VerifiedColabSnapshotStore" in text
    assert "SNAPSHOTS.restore_latest()" in text
    assert "shutil.make_archive" not in text
    assert "durable_interval_seconds=60" in text
    assert "copy_verified" in text
    assert "CHECKED_SOURCE_ROOT" in text
    assert "SOURCE_ROOT = ARTIFACT_ROOT" in text
    assert "load_state" in text and "save_state" in text
    assert "fa-run-screening" in text
    assert "fa-screen-entities" in text
    assert 'smoke_generation["shard_manifest"]' in text
    assert "check=True" not in run_cell
    assert "allow_infrastructure_failure=True" in text


def test_same_string_colab_separates_compilation_from_resumable_adjudication():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(cell.get("source", ())) for cell in payload["cells"]]
    compile_cell = next(
        index for index, cell in enumerate(cells)
        if "fa-compile-naturalness-ratings" in cell
    )
    adjudication_cell = next(
        index for index, cell in enumerate(cells)
        if "fa-finalize-naturalness-adjudication" in cell
    )

    assert compile_cell < adjudication_cell
    assert 'load_state("naturalness_ratings")' in cells[adjudication_cell]


def test_same_string_runbook_states_counts_gates_and_claim_boundary():
    text = RUNBOOK.read_text(encoding="utf-8")

    for count in ("64", "32", "48", "24"):
        assert count in text
    for command in (
        "fa-prepare-same-string-matches",
        "fa-prepare-naturalness-ratings",
        "fa-build-same-string-confirmatory",
        "fa-audit-manifest",
        "fa-seal-behavior-test",
        "fa-evaluate-behavior-test",
    ):
        assert command in text
    assert "two independent raters" in text
    assert "protected endpoint" in text
    assert "not empirical evidence" in text


def test_readme_points_to_active_same_string_study_without_rewriting_r11():
    text = README.read_text(encoding="utf-8")

    assert "Same-String Primary Study" in text
    assert "docs/fa_same_string_primary_runbook.md" in text
    assert "docs/amendments/2026-08-01-fa-same-string-primary.md" in text
    assert "docs/superpowers/specs/2026-08-01-same-string-primary-hybrid-design.md" in text
    assert "R11" in text
    assert "not yet an empirical result" in text
