from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "fa_same_string_primary_colab.ipynb"
V2_NOTEBOOK = ROOT / "notebooks" / "fa_same_string_feasibility_v2_colab.ipynb"
RUNBOOK = ROOT / "docs" / "fa_same_string_primary_runbook.md"
README = ROOT / "README.md"
V2_RESULT = ROOT / "docs" / "results" / "same_string_feasibility_v2_behavior_result.json"
V2_SNAPSHOT = (
    ROOT
    / "release"
    / "familiarity_answerability"
    / "fa-58f1f069cb6a1906ff17a0282805f859675ae80b0f707fc0f768fc7a956178e3.zip"
)


def notebook_text() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ())) for cell in payload.get("cells", ())
    )


def v2_notebook_text() -> str:
    payload = json.loads(V2_NOTEBOOK.read_text(encoding="utf-8"))
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


def test_same_string_colab_uses_published_anonymous_ratings():
    text = notebook_text()
    ratings_root = ROOT / "data/fa/human_ratings/same_string_primary_v1"
    expected_hashes = {
        "rater-a-response.csv": "aed43152d66555d3546bf24ced1ea0f075ed2accfc4378050d6d1ff1ed773614",
        "rater-b-response.csv": "bee9d6ad5f6fef140264450dccaf5869634dd077c04670aa6dfda60d2220efa8",
        "rater-c-response.csv": "d1b26fb6681a3d08872545d1c45b978880ad663c1c34d5a0e1a06ae0296b8784",
    }

    assert "data/fa/human_ratings/same_string_primary_v1" in text
    for name, expected_hash in expected_hashes.items():
        path = ratings_root / name
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        assert f'"{name}"' in text
        assert expected_hash in text
    assert "FA_ADJUDICATION_RESPONSE" not in text
    assert text.index('ratings["naturalness_gate"]["status"]') < text.index(
        "fa-run-screening"
    )


def test_feasibility_v2_colab_replays_v1_then_allocates_before_model_compute():
    text = v2_notebook_text()

    assert re.search(r'PINNED_REPO_COMMIT\s*=\s*"[0-9a-f]{40}"', text)
    assert "USE_DRIVE_CHECKPOINTS = False" in text
    assert "if USE_DRIVE_CHECKPOINTS:" in text
    assert 'drive.mount("/content/drive", timeout_ms=60_000)' in text
    assert 'Path("/content/fa-same-string-feasibility-v2-checkpoints")' in text
    assert "Drive unavailable; using local checkpoints" in text
    assert 'sys.path.insert(0, str(CHECKOUT / "src"))' in text
    assert 'python -m pip uninstall -y torchvision' in text
    assert "familiarity_answerability_same_string_gemma2_2b.json" in text
    assert "familiarity_answerability_same_string_feasibility_v2.json" in text
    assert "/content/fa-same-string-feasibility-v2-artifacts" in text
    commands = (
        "fa-prepare-same-string-matches",
        "fa-prepare-naturalness-ratings",
        "fa-compile-naturalness-ratings",
        "fa-finalize-naturalness-adjudication",
        "fa-prepare-same-string-v2-matches",
        "fa-run-screening",
        "fa-build-same-string-confirmatory",
        "fa-seal-behavior-test",
        "fa-evaluate-behavior-test",
    )
    positions = [text.index(command) for command in commands]
    assert positions == sorted(positions)
    assert 'ratings["naturalness_gate"]["status"] == "failed"' in text
    assert '"behavior_test": 32' in text
    assert "intervention_test\": 24" not in text[text.index("v2_matches =") :]


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


def test_readme_reports_completed_v2_study_without_rewriting_r11():
    text = README.read_text(encoding="utf-8")

    assert "Same-String Balanced Pilot v2" in text
    assert "evaluable, and `not_supported`" in text
    assert "same_string_feasibility_v2_behavior_result.md" in text
    assert "same_string_feasibility_v2_behavior_result.json" in text
    assert "docs/fa_same_string_primary_runbook.md" in text
    assert "docs/amendments/2026-08-01-fa-same-string-primary.md" in text
    assert "docs/superpowers/specs/2026-08-01-same-string-primary-hybrid-design.md" in text
    assert "R11" in text


def test_v2_public_result_records_closed_endpoint_and_failed_gate():
    result = json.loads(V2_RESULT.read_text(encoding="utf-8"))

    assert result["decision"]["endpoint_status"] == "evaluable"
    assert result["decision"]["endpoint_state"] == "closed"
    assert result["decision"]["registered_gate"] == "not_supported"
    assert result["decision"]["mechanistic_followup"] == "not_run_behavior_gate_failed"
    assert result["sample"] == {
        "complete_units": 32,
        "prompt_rows": 128,
        "units_per_domain": 8,
    }
    assert (
        result["registered_effects"]["exposure_by_answerability_interaction"]["estimate"]
        == -0.09375
    )
    assert result["bootstrap"]["valid_draws"] == 10_000
    assert result["bootstrap"]["discarded_draws"] == 0
    assert result["registered_effects"]["exposure_by_answerability_interaction"] == {
        "estimate": -0.09375,
        "confidence_interval_95": {
            "lower": -0.39999999999999997,
            "upper": 0.18181818181818177,
        },
        "minimum_registered_effect": 0.05,
    }
    assert result["registered_effects"]["target_bound_capability_difference"] == {
        "estimate": 0.15625,
        "confidence_interval_95": {
            "lower": -0.07692307692307698,
            "upper": 0.43333333333333335,
        },
        "noninferiority_margin": -0.05,
    }
    assert {
        cell: metrics["attempt_rate"]
        for cell, metrics in result["cell_metrics"].items()
    } == {
        "high_exposure_code_absent": 0.0625,
        "low_exposure_code_absent": 0.0,
        "high_exposure_target_bound": 0.90625,
        "low_exposure_target_bound": 0.75,
    }
    assert result["decision"]["gate_reasons"] == [
        "interaction_point_estimate_below_minimum",
        "interaction_interval_not_positive",
        "capability_noninferiority_lower_bound",
    ]
    assert result["model"]["revision"] == (
        "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
    )
    assert result["provenance"]["snapshot_sha256"] == (
        "58f1f069cb6a1906ff17a0282805f859675ae80b0f707fc0f768fc7a956178e3"
    )
    assert result["protocol_deviations"] == [
        {
            "id": "missing_preoutcome_power_or_mde_audit",
            "description": (
                "The amendment required a pre-outcome power or minimum-detectable-effect "
                "audit, but no such artifact is present in the verified snapshot."
            ),
            "impact": (
                "The machine-evaluated endpoint and registered gate remain not_supported "
                "because the amendment made power descriptive rather than decision-changing "
                "and did not list it as an endpoint-opening hard stop. The omission materially "
                "limits the strength of inference from this small pilot and is not repaired "
                "post hoc."
            ),
        }
    ]


def test_v2_released_snapshot_matches_declared_hash_and_index():
    result = json.loads(V2_RESULT.read_text(encoding="utf-8"))

    assert hashlib.sha256(V2_SNAPSHOT.read_bytes()).hexdigest() == (
        result["provenance"]["snapshot_sha256"]
    )
    with zipfile.ZipFile(V2_SNAPSHOT) as archive:
        assert archive.testzip() is None
        index = json.loads(archive.read("_snapshot_index.json"))
    assert index["schema_version"] == 1
    assert len(index["members"]) == 729
