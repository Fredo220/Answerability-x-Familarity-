from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from trajectory_extractor.fa_activations import ANCHOR_NAMES
from trajectory_extractor.fa_pilot_analysis import (
    PILOT_LAYER_IDS,
    PilotAnalysisRow,
    _permute_within_strata,
    analyze_pilot_rows,
)


def _pilot_rows() -> tuple[PilotAnalysisRow, ...]:
    rng = np.random.default_rng(20260723)
    rows = []
    index = 0
    for group_index in range(8):
        for familiarity_index, familiarity in enumerate(
            ("matched_synthetic", "screened_real")
        ):
            for answerability_index, answerability in enumerate(
                ("target_bound", "distractor_bound", "code_absent")
            ):
                activations = rng.normal(0.0, 0.05, size=(3, 4, 16))
                activations[0, :, 0] += 3.0 * familiarity_index
                activations[1, :, 1 + answerability_index] += 3.0
                activations[2, :, 0] += 2.0 * familiarity_index
                activations[2, :, 1 + answerability_index] += 2.0
                rows.append(
                    PilotAnalysisRow(
                        example_id=f"example-{index:03d}",
                        entity_unit_id=f"entity-{group_index}",
                        template_family=(
                            "train_registry_direct",
                            "train_registry_possessive",
                            "train_registry_query",
                        )[index % 3],
                        target_familiarity=familiarity,
                        answerability=answerability,
                        surface_features=(
                            float(group_index % 2),
                            float(index % 3),
                            float(index % 5),
                        ),
                        layer_ids=PILOT_LAYER_IDS,
                        anchor_names=ANCHOR_NAMES,
                        activations=activations,
                    )
                )
                index += 1
    return tuple(rows)


def test_pilot_analysis_reports_every_frozen_layer_and_holds_out_entities():
    rows = _pilot_rows()

    result = analyze_pilot_rows(rows, permutation_seeds=(101, 102))

    assert result.example_count == 48
    assert result.group_count == 8
    assert len(result.metric_records) == 38
    assert len(result.prediction_records) == 38 * 48
    for task in ("familiarity", "answerability"):
        for anchor in (
            "target_intro_end" if task == "familiarity" else "user_prompt_end",
            "assistant_prefix_end",
        ):
            for family in ("residual_static", "morphology_plus_residual"):
                metrics = [
                    row
                    for row in result.metric_records
                    if row["task"] == task
                    and row["anchor"] == anchor
                    and row["model_family"] == family
                ]
                assert {row["layer_id"] for row in metrics} == set(PILOT_LAYER_IDS)
                assert all(row["permutation_count"] == 2 for row in metrics)
                assert all(row["permutation_p_max_layer"] is not None for row in metrics)
                assert all(row["mean_layer_omnibus_p"] is not None for row in metrics)
    assert all(
        row["entity_unit_id"] == row["held_out_entity_unit_id"]
        for row in result.prediction_records
    )


def test_pilot_analysis_is_deterministic_and_does_not_select_a_best_layer():
    rows = _pilot_rows()

    first = analyze_pilot_rows(rows, permutation_seeds=(201,))
    second = analyze_pilot_rows(rows, permutation_seeds=(201,))

    assert first.analysis_sha256 == second.analysis_sha256
    assert first.metric_records == second.metric_records
    assert not any("best" in key for row in first.metric_records for key in row)


def test_within_group_permutation_preserves_each_entity_class_count():
    rows = _pilot_rows()
    labels = np.asarray(
        [
            ("target_bound", "distractor_bound", "code_absent").index(
                row.answerability
            )
            for row in rows
        ]
    )

    permuted = _permute_within_strata(
        labels, rows, task="answerability", seed=2026072300
    )

    assert not np.array_equal(permuted, labels)
    for group in sorted({row.entity_unit_id for row in rows}):
        indices = np.asarray([row.entity_unit_id == group for row in rows])
        assert Counter(permuted[indices].tolist()) == Counter(labels[indices].tolist())


def test_pilot_analysis_rejects_unregistered_layers():
    row = _pilot_rows()[0]

    with pytest.raises(ValueError, match="frozen layer IDs"):
        PilotAnalysisRow(
            example_id=row.example_id,
            entity_unit_id=row.entity_unit_id,
            template_family=row.template_family,
            target_familiarity=row.target_familiarity,
            answerability=row.answerability,
            surface_features=row.surface_features,
            layer_ids=(0, 8, 18, 27),
            anchor_names=row.anchor_names,
            activations=row.activations,
        )
