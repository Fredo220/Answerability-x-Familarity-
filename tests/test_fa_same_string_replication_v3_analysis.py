from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from trajectory_extractor.fa_activations import ANCHOR_NAMES
from trajectory_extractor.fa_same_string_replication_v3 import build_replication_corpus
from trajectory_extractor.fa_same_string_replication_v3_analysis import (
    REP_V3_LAYERS,
    ReplicationAnalysisRowV3,
    analyze_replication_v3,
    build_replication_analysis_rows,
    simulate_replication_sensitivity,
)

from test_fa_same_string_replication_v3 import _CharacterTokenizer


def _rows() -> tuple[ReplicationAnalysisRowV3, ...]:
    corpus = build_replication_corpus(_CharacterTokenizer())
    rng = np.random.default_rng(20260803)
    rows = []
    for prompt in corpus.prompts:
        answer = int(prompt.answerability == "target_bound")
        exposure = int(prompt.exposure == "high_exposure")
        values = rng.normal(0.0, 0.15, size=(len(ANCHOR_NAMES), len(REP_V3_LAYERS), 24))
        values[1, :, 0] += 2.5 * answer
        values[0, :, 1] += 2.5 * exposure
        rows.append(ReplicationAnalysisRowV3.from_prompt(prompt, values))
    return tuple(rows)


def test_v3_analysis_keeps_test_splits_separate_and_reports_primary_increment():
    result = analyze_replication_v3(
        _rows(), bootstrap_draws=200, permutation_count=49
    )

    assert result.example_count == 320
    assert result.training_unit_count == 32
    assert result.test_unit_counts == {"entity_test": 20, "template_test": 20}
    assert {row["test_split"] for row in result.primary_records} == {
        "entity_test",
        "template_test",
    }
    assert all(row["permutation_count"] == 49 for row in result.primary_records)
    assert all(row["mean_paired_log_loss_improvement"] > 0 for row in result.primary_records)
    assert all(row["mean_auroc_improvement"] >= 0 for row in result.primary_records)
    assert {row["test_split"] for row in result.prediction_records} == {
        "entity_test",
        "template_test",
    }


def test_v3_analysis_is_deterministic_and_has_no_best_layer_selection():
    first = analyze_replication_v3(_rows(), bootstrap_draws=100, permutation_count=19)
    second = analyze_replication_v3(_rows(), bootstrap_draws=100, permutation_count=19)

    assert first.analysis_sha256 == second.analysis_sha256
    assert first.primary_records == second.primary_records
    assert not any("best" in key for row in first.metric_records for key in row)


def test_v3_analysis_reports_temporal_negative_control():
    result = analyze_replication_v3(_rows(), bootstrap_draws=100, permutation_count=19)

    controls = [
        row
        for row in result.metric_records
        if row["task"] == "answerability"
        and row["anchor"] == "target_intro_end"
        and row["model_family"] == "activation"
    ]
    assert controls
    assert all(row["claim_role"] == "temporal_negative_control" for row in controls)


def test_v3_sensitivity_audit_is_fixed_seed_and_preoutcome():
    first = simulate_replication_sensitivity(simulations=200)
    second = simulate_replication_sensitivity(simulations=200)

    assert first == second
    assert first["kind"] == "same_string_replication_v3_sensitivity"
    assert first["unit_count_per_test_split"] == 20
    assert first["outcomes_opened"] is False
    assert len(first["scenarios"]) >= 4


def test_v3_analysis_rows_bind_activation_prompt_provenance():
    corpus = build_replication_corpus(_CharacterTokenizer())
    prompt = corpus.prompts[0]
    values = np.ones((len(ANCHOR_NAMES), len(REP_V3_LAYERS), 16))
    record = SimpleNamespace(
        example_id=prompt.example_id,
        anchors=SimpleNamespace(input_ids=prompt.rendered_token_ids),
        layer_ids=REP_V3_LAYERS,
        anchor_names=ANCHOR_NAMES,
        activations=values,
    )

    rows = build_replication_analysis_rows((prompt,), (record,))

    assert len(rows) == 1
    assert rows[0].example_id == prompt.example_id
