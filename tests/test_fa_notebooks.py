from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "fa_same_string_primary_colab.ipynb"
V2_NOTEBOOK = ROOT / "notebooks" / "fa_same_string_feasibility_v2_colab.ipynb"
CAUSAL_NOTEBOOK = ROOT / "notebooks" / "fa_answerability_causal_pilot_colab.ipynb"
CAUSAL_RUNBOOK = ROOT / "docs" / "fa_answerability_causal_pilot_runbook.md"
CAUSAL_REPLICATION_NOTEBOOK = (
    ROOT / "notebooks" / "fa_answerability_causal_replication_v2_colab.ipynb"
)
CAUSAL_REPLICATION_RUNBOOK = (
    ROOT / "docs" / "fa_answerability_causal_replication_v2_runbook.md"
)
CAUSAL_COLAB_REQUIREMENTS = ROOT / "requirements" / "fa-causal-colab.lock"
RUNBOOK = ROOT / "docs" / "fa_same_string_primary_runbook.md"
README = ROOT / "README.md"
V2_RESULT = ROOT / "docs" / "results" / "same_string_feasibility_v2_behavior_result.json"
REPRESENTATION_RESULT = (
    ROOT / "docs" / "results" / "same_string_representation_pilot_v2.json"
)
REPRESENTATION_RELEASE = (
    ROOT / "release" / "familiarity_answerability" / "representation_pilot_v2"
)
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


def causal_notebook_text() -> str:
    payload = json.loads(CAUSAL_NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ())) for cell in payload.get("cells", ())
    )


def causal_replication_notebook_text() -> str:
    payload = json.loads(CAUSAL_REPLICATION_NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ())) for cell in payload.get("cells", ())
    )


def test_causal_replication_colab_uses_the_frozen_pin_and_keeps_secrets_out():
    text = causal_replication_notebook_text()

    assert (
        'PINNED_REPO_COMMIT = "26188c9b9105d96446c0ea276fc84be5e444bd0e"'
        in text
    )
    assert "__PINNED_COMMIT__" not in text
    assert 'os.environ.get("HF_TOKEN")' in text
    assert 'userdata.get("HF_TOKEN")' in text
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert "requirements/fa-causal-colab.lock" in text
    assert text.count("HFCausalRunner.from_pretrained") == 1


def test_causal_replication_colab_binds_v2_root_config_and_locked_validation():
    text = causal_replication_notebook_text()

    assert "configs/familiarity_answerability_causal_replication_v2.json" in text
    assert 'Path("/content/fa-causal-replication-v2")' in text
    assert 'Path("/content/drive/MyDrive/fa-causal-replication-v2")' in text
    assert "fa-causal-pilot-v1" not in text
    assert 'config.validation_selection["mode"] == "locked_from_v1"' in text
    assert 'config.validation_selection["locked_layer"] == 18' in text
    assert 'config.validation_selection["locked_multiplier"] == 1.0' in text
    assert 'selection["layer_id"] == 18' in text
    assert 'selection["multiplier"] == 1.0' in text


def test_causal_replication_colab_runs_complete_resumable_schedule_and_downloads_zip():
    text = causal_replication_notebook_text()
    positions = [
        text.index(name)
        for name in (
            "prepare_causal",
            "run_causal_validation",
            "expected_causal_shards",
            "run_causal_shard",
            "evaluate_causal",
            "make_archive",
            "files.download",
        )
    ]

    assert positions == sorted(positions)
    assert "assert len(schedule) == 432" in text
    assert "resume=True" in text
    assert "Each receipt is atomic" in text
    assert "One-use protected evaluation" in text
    assert 'result["study_id"]' not in text
    assert "same-string-answerability-causal-replication-v2" in text


def test_causal_replication_runbook_states_pin_resume_and_claim_boundaries():
    text = CAUSAL_REPLICATION_RUNBOOK.read_text(encoding="utf-8")

    assert "26188c9b9105d96446c0ea276fc84be5e444bd0e" in text
    assert re.search(r"`[0-9a-f]{40}`", text)
    assert "familiarity_answerability_causal_replication_v2.json" in text
    assert "fa-causal-replication-v2" in text
    assert "locked" in text.casefold()
    assert "layer 18" in text.casefold()
    assert "multiplier 1.0" in text.casefold()
    assert "432" in text
    assert "resume" in text.casefold()
    assert "not_evaluable" in text
    assert "not_supported" in text
    assert "causally_supported" in text
    assert "does not establish general metacognition" in text


def test_causal_colab_is_pinned_secret_safe_and_single_load():
    text = causal_notebook_text()

    assert re.search(r'PINNED_REPO_COMMIT\s*=\s*"[0-9a-f]{40}"', text)
    assert 'PINNED_REPO_COMMIT = "' + "0" * 40 + '"' not in text
    assert 'os.environ.get("HF_TOKEN")' in text
    assert 'userdata.get("HF_TOKEN")' in text
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert "requirements/fa-causal-colab.lock" in text
    assert "requirements/fa-core.lock" not in text
    assert text.count("HFCausalRunner.from_pretrained") == 1
    assert "/content/drive/MyDrive/fa-causal-pilot-v1" in text
    assert "resume=True" in text


def test_causal_colab_keeps_colab_numerical_stack_and_falls_back_from_drive():
    notebook = causal_notebook_text()
    requirements = CAUSAL_COLAB_REQUIREMENTS.read_text(encoding="utf-8")
    installed = "\n".join(
        line for line in requirements.splitlines() if line and not line.startswith("#")
    )

    assert "numpy" not in installed.casefold()
    assert "pandas" not in installed.casefold()
    assert "scikit-learn" not in installed.casefold()
    assert "torch==" not in installed.casefold()
    assert "bitsandbytes==0.49.2" in installed
    assert 'drive.mount("/content/drive", timeout_ms=60_000)' in notebook
    assert "Drive unavailable; using local checkpoints" in notebook


def test_causal_colab_preserves_gate_order_and_complete_schedule():
    text = causal_notebook_text()
    positions = [
        text.index(name)
        for name in (
            "prepare_causal",
            "run_causal_validation",
            "expected_causal_shards",
            "run_causal_shard",
            "evaluate_causal",
        )
    ]

    assert positions == sorted(positions)
    assert "assert torch.cuda.is_available()" in text
    assert "Each receipt is atomic" in text
    assert "One-use protected evaluation" in text


def test_causal_runbook_states_completed_audit_and_resume_contract():
    text = CAUSAL_RUNBOOK.read_text(encoding="utf-8")

    assert "free-Colab T4" in text
    assert "not_evaluable_as_confirmatory_causal_test" in text
    assert "label-shuffled direction was bit-for-bit identical" in text
    assert "fresh test units" in text
    assert "432 atomic unit receipts" in text
    assert re.search(r"same request\s+hash", text)
    assert re.search(r"Do\s+not replace the loop", text)
    assert "general metacognition" in text


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


def test_readme_separates_supported_representation_from_invalidated_causal_followup():
    text = README.read_text(encoding="utf-8")

    assert "Behavioral pilot" in text
    assert "**Not supported**" in text
    assert "same_string_feasibility_v2_behavior_result.md" in text
    assert "same_string_feasibility_v2_behavior_result.json" in text
    assert "Representation replication" in text
    assert "Supported on this controlled task" in text
    assert "Causal follow-up: completed, but not confirmatory" in text
    assert "confirmatory causal test not evaluable after control audit" in text
    assert "bit-for-bit identical to the primary vector" in text
    assert "The machine result remains unchanged" in text
    assert "changed none of the 144" in text
    assert "432 atomic receipts" in text
    assert "fa_answerability_causal_pilot_colab.ipynb" in text
    assert "POST_RUN_AUDIT.md" in text


def test_representation_result_is_exploratory_and_matches_released_artifacts():
    result = json.loads(REPRESENTATION_RESULT.read_text(encoding="utf-8"))

    assert result["status"] == "complete_exploratory"
    assert result["claim_scope"] == "exploratory_representation_only"
    assert result["sample"] == {
        "prompt_count": 80,
        "training_group_count": 16,
        "test_group_count": 4,
        "test_prompt_count": 16,
    }
    assert result["fixed_layers"] == [0, 6, 12, 18, 25]
    assert result["results"]["exposure"]["residual_static_auroc_by_layer"] == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert result["results"]["answerability"]["early_anchor_auroc_by_layer"] == [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    ]
    assert result["results"]["answerability"]["surface_baseline_auroc"] == 1.0

    for artifact, provenance_key in (
        ("same-string-v2-representation-pilot-metrics.jsonl", "metrics_sha256"),
        (
            "same-string-v2-representation-pilot-predictions.jsonl",
            "predictions_sha256",
        ),
    ):
        data_path = REPRESENTATION_RELEASE / artifact
        manifest_path = REPRESENTATION_RELEASE / f"{artifact}.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()

        assert manifest["data_file"] == artifact
        assert manifest["sha256"] == digest
        assert result["provenance"][provenance_key] == digest


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
