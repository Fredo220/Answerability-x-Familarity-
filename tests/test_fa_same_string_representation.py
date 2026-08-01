from __future__ import annotations

import numpy as np
import pytest

from trajectory_extractor.fa_activations import ANCHOR_NAMES
from trajectory_extractor.fa_same_string_representation import (
    REPRESENTATION_LAYER_IDS,
    SameStringRepresentationRow,
    analyze_same_string_representations,
)


def _rows() -> tuple[SameStringRepresentationRow, ...]:
    rng = np.random.default_rng(20260802)
    rows = []
    index = 0
    splits = (
        ("mechanism_train", 12),
        ("locked_validation", 4),
        ("probe_test", 4),
    )
    for split, group_count in splits:
        for group_index in range(group_count):
            unit = f"{split}-unit-{group_index:02d}"
            for exposure_index, exposure in enumerate(
                ("low_exposure", "high_exposure")
            ):
                for answerability_index, answerability in enumerate(
                    ("code_absent", "target_bound")
                ):
                    activations = rng.normal(
                        0.0, 0.05, size=(3, len(REPRESENTATION_LAYER_IDS), 16)
                    )
                    activations[0, :, 0] += 3.0 * exposure_index
                    activations[1, :, 1] += 3.0 * answerability_index
                    rows.append(
                        SameStringRepresentationRow(
                            example_id=f"example-{index:03d}",
                            entity_unit_id=unit,
                            split=split,
                            template_family="train_registry_direct",
                            exposure=exposure,
                            answerability=answerability,
                            surface_features=(float(index % 3), float(index % 5)),
                            layer_ids=REPRESENTATION_LAYER_IDS,
                            anchor_names=ANCHOR_NAMES,
                            activations=activations,
                        )
                    )
                    index += 1
    return tuple(rows)


def test_representation_pilot_uses_fixed_test_units_and_reports_nulls():
    result = analyze_same_string_representations(
        _rows(), permutation_seeds=(101, 102), bootstrap_seed=103, bootstrap_draws=50
    )

    assert result.example_count == 80
    assert result.training_group_count == 16
    assert result.test_group_count == 4
    assert len(result.metric_records) == 42
    assert len(result.prediction_records) == 42 * 16
    assert all(row["split"] == "probe_test" for row in result.prediction_records)
    assert all(
        row["claim_scope"] == "exploratory_representation_only"
        for row in result.metric_records
    )
    internal = [
        row for row in result.metric_records if row["layer_id"] is not None
    ]
    assert all(row["permutation_count"] == 2 for row in internal)
    assert all(row["auroc_ci95"] is not None for row in result.metric_records)


def test_representation_pilot_is_deterministic_and_does_not_select_a_best_layer():
    first = analyze_same_string_representations(
        _rows(), permutation_seeds=(201,), bootstrap_seed=202, bootstrap_draws=50
    )
    second = analyze_same_string_representations(
        _rows(), permutation_seeds=(201,), bootstrap_seed=202, bootstrap_draws=50
    )

    assert first.analysis_sha256 == second.analysis_sha256
    assert first.metric_records == second.metric_records
    assert not any("best" in key for row in first.metric_records for key in row)


def test_representation_pilot_rejects_incomplete_same_string_units():
    with pytest.raises(ValueError, match="complete 2x2"):
        analyze_same_string_representations(
            _rows()[:-1], permutation_seeds=(301,), bootstrap_seed=302, bootstrap_draws=50
        )


def test_representation_row_rejects_unregistered_layers():
    row = _rows()[0]
    with pytest.raises(ValueError, match="fixed layer IDs"):
        SameStringRepresentationRow(
            example_id=row.example_id,
            entity_unit_id=row.entity_unit_id,
            split=row.split,
            template_family=row.template_family,
            exposure=row.exposure,
            answerability=row.answerability,
            surface_features=row.surface_features,
            layer_ids=(0, 5, 12, 18, 25),
            anchor_names=row.anchor_names,
            activations=row.activations,
        )
