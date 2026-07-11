import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trajectory_extractor.artifacts import RunStore
from trajectory_extractor.cli import _concept_pilot, _dataset_provenance, _jailbreak_pilot
from trajectory_extractor.datasets.concept_mixing import generate_concept_mixing_examples
from trajectory_extractor.datasets.jailbreak import (
    JailbreakExample,
    build_jailbreak_study_file,
    grouped_folds,
    load_official_jailbreakbench,
    validate_jailbreak_study_set,
)
from trajectory_extractor.evaluation import threshold_metrics
from trajectory_extractor.judge import stratified_audit_indices
from trajectory_extractor.reporting import classify_detection, classify_intervention
from trajectory_extractor.steering import OperatorResidualSteering
from trajectory_extractor.intervention_study import grouped_category_partition
from trajectory_extractor.types import ResponseRun


class IdentityProjection:
    def transform(self, values):
        return np.asarray(values)


class PassLayer(torch.nn.Module):
    def forward(self, hidden_states):
        return (hidden_states + 1.0,)


class FakeLlama(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([PassLayer(), PassLayer()])

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


def test_run_store_creates_required_run_layout(tmp_path):
    store = RunStore(tmp_path)
    store.write_manifest("study", {"seed": 42})
    store.write_json("study", "metrics", "concept", {"auroc": 0.7})
    assert (tmp_path / "study" / "manifest.json").exists()
    assert (tmp_path / "study" / "metrics" / "concept.json").exists()
    assert (tmp_path / "study" / "bootstrap").is_dir()
    assert (tmp_path / "study" / "figures").is_dir()


def test_run_store_persists_response_only_intervention_records(tmp_path):
    store = RunStore(tmp_path)
    run = ResponseRun(
        run_id="intervention",
        example_id="one",
        track="concept_intervention",
        split="test",
        prompt="prompt",
        response="answer",
        label=0,
        provenance={"activation_replay": False},
    )
    store.write_response(run)
    loaded = store.read_response("intervention", "one")
    assert loaded.response == "answer"
    assert store.judgable_ids("intervention") == ["one"]
    assert store.has_response("intervention", "one")


def test_jailbreak_loader_and_grouped_folds(tmp_path):
    path = tmp_path / "behaviors.jsonl"
    rows = [
        {"id": "a", "behavior": "frozen-a", "category": "cat-a", "benign": False},
        {"id": "b", "behavior": "frozen-b", "category": "cat-b", "benign": True},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    examples = load_official_jailbreakbench(path)
    folds = grouped_folds(examples)
    assert len(examples) == 2
    assert len(folds) == 2
    assert examples[1].benign is True


def test_threshold_metrics_is_prefix_causal():
    scores = np.array([[[0.1, 0.2], [0.8, 0.1]], [[0.7, 0.1], [0.2, 0.1]]])
    result = threshold_metrics(scores, np.array([1, 0]), threshold=0.5)
    assert result.earliest_crossing.tolist() == [2, 0]
    assert result.false_positive_rate == 1.0


def test_frozen_acceptance_rules():
    assert classify_detection(0.03, 0.001, 0.06).outcome == "supported"
    assert classify_detection(0.02, 0.001, 0.04).outcome == "partially_supported"
    assert classify_intervention(0.20, 0.05, 0.001, 0.3).outcome == "supported"
    assert classify_detection(-0.01, -0.05, 0.02).outcome == "not_supported"


def test_strict_jailbreak_validation_requires_balanced_official_artifacts(tmp_path):
    path = tmp_path / "study.jsonl"
    rows = []
    for index in range(3):
        rows.append(
            {
                "id": f"h-{index}",
                "behavior": "frozen harmful behavior",
                "category": f"cat-{index}",
                "benign": False,
                "source": "JailbreakBench",
                "artifact": "published frozen artifact",
                "pair_id": f"pair-{index}",
            }
        )
        rows.append(
            {
                "id": f"b-{index}",
                "behavior": "matched benign behavior",
                "category": f"cat-{index}",
                "benign": True,
                "source": "JailbreakBench",
                "pair_id": f"pair-{index}",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows))
    summary = validate_jailbreak_study_set(
        load_official_jailbreakbench(path), expected_harmful=3, expected_benign=3
    )
    assert summary == {"total": 6, "harmful": 3, "benign": 3, "pairs": 3}


def test_jailbreak_builder_uses_goals_not_behavior_labels(tmp_path):
    fieldnames = ["Index", "Goal", "Target", "Behavior", "Category", "Source"]
    harmful = tmp_path / "harmful.csv"
    benign = tmp_path / "benign.csv"
    import csv

    for path, prefix in ((harmful, "harmful"), (benign, "benign")):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(100):
                writer.writerow(
                    {
                        "Index": index,
                        "Goal": f"{prefix} goal {index}",
                        "Target": "target",
                        "Behavior": f"behavior {index}",
                        "Category": f"category {index % 10}",
                        "Source": "Original",
                    }
                )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            [
                {
                    "index": index,
                    "goal": f"harmful goal {index}",
                    "behavior": f"behavior {index}",
                    "category": f"category {index % 10}",
                    "prompt": f"frozen prompt {index}",
                }
                for index in range(100)
            ]
        )
    )
    output = tmp_path / "study.jsonl"
    manifest = build_jailbreak_study_file(
        harmful, benign, artifact, output, artifact_commit="abc123"
    )
    examples = load_official_jailbreakbench(output)
    assert manifest["counts"]["pairs"] == 100
    assert examples[0].behavior == "frozen prompt 0"
    assert examples[0].behavior_name == "behavior 0"
    assert examples[1].behavior == "benign goal 0"
    assert examples[1].artifact is None
    assert output.with_suffix(".jsonl.manifest.json").exists()


def test_stratified_audit_includes_each_label():
    indices = stratified_audit_indices([0] * 10 + [1] * 10, fraction=0.2, seed=1)
    assert len(indices) == 4
    assert any(index < 10 for index in indices)
    assert any(index >= 10 for index in indices)


def test_stratified_audit_has_exact_fraction_for_composite_strata():
    strata = [(f"cat-{index % 5}", bool(index % 2), bool(index % 3)) for index in range(100)]
    indices = stratified_audit_indices(strata, fraction=0.2, seed=2)
    assert len(indices) == 20
    assert len(set(indices)) == 20


def test_operator_residual_steering_triggers_and_restores():
    model = FakeLlama()
    hidden = torch.zeros((1, 1, 2))
    baseline = model(hidden)
    with OperatorResidualSteering(
        model,
        from_layer_idx=0,
        pca_from=IdentityProjection(),
        pca_to=IdentityProjection(),
        operator=np.zeros((2, 2)),
        threshold=0.1,
        direction=torch.tensor([1.0, 0.0]),
        strength=2.0,
    ) as intervention:
        steered = model(hidden)
    assert intervention.triggered
    torch.testing.assert_close(steered[0, -1], baseline[0, -1] + torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(model(hidden), baseline)


def test_preregistration_and_configs_exist():
    study = json.loads(Path("configs/study.json").read_text())
    assert study["detection_min_auc_gain"] == 0.03
    assert "not_supported" in Path("docs/preregistration.md").read_text()


def test_concept_pilot_preserves_every_split():
    selected = _concept_pilot(generate_concept_mixing_examples(total=30), 3)
    assert len(selected) == 9
    assert {split: sum(row.split == split for row in selected) for split in ("train", "val", "test")} == {
        "train": 3,
        "val": 3,
        "test": 3,
    }


def test_jailbreak_pilot_preserves_matched_pairs_per_category():
    examples = [
        JailbreakExample(
            example_id=f"{category}-{pair}-{benign}",
            behavior="prompt",
            category=category,
            benign=benign,
            source="JailbreakBench",
            artifact=None if benign else "frozen",
            pair_id=f"{category}-{pair}",
        )
        for category in ("a", "b")
        for pair in range(3)
        for benign in (False, True)
    ]
    selected = _jailbreak_pilot(examples, 1)
    assert len(selected) == 4
    assert len({row.pair_id for row in selected}) == 2
    assert all({row.benign for row in selected if row.category == category} == {False, True} for category in ("a", "b"))


def test_jailbreak_intervention_category_split_is_seeded_and_label_independent():
    categories = [f"category-{index}" for index in range(10)]
    first = grouped_category_partition(categories, seed=42)
    second = grouped_category_partition(list(reversed(categories)), seed=42)
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 6,
        "val": 2,
        "test": 2,
    }
    assert set(first["train"]).isdisjoint(first["val"] + first["test"])


def test_dataset_provenance_hashes_data_and_sidecar(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"id": 1}\n')
    sidecar = data.with_suffix(".jsonl.manifest.json")
    sidecar.write_text('{"version": 1}\n')
    provenance = _dataset_provenance(data)
    assert provenance["path"] == str(data)
    assert len(provenance["sha256"]) == 64
    assert provenance["manifest_path"] == str(sidecar)
    assert len(provenance["manifest_sha256"]) == 64
