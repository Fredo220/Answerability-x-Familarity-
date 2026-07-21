from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, replace

import pytest

from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_data import (
    CONFIRMATORY_POWER_SIMULATIONS,
    REGISTERED_POWER_GRID,
    TEST_TEMPLATE_FAMILIES,
    TRAIN_TEMPLATE_FAMILIES,
    VALIDATION_TEMPLATE_FAMILIES,
    FAManifest,
    audit_dataset,
    build_factorial_examples,
    build_same_string_examples,
    build_manifest,
    simulate_interaction_power,
)
from trajectory_extractor.fa_entities import EntityMatch


class FakeTokenizer:
    all_special_ids = (0,)

    def encode(self, text, add_special_tokens=False):
        tokens = text.replace(".", " .").replace("?", " ?").split()
        return ([0] if add_special_tokens else []) + tokens + ([0] if add_special_tokens else [])


def config() -> FAConfig:
    return FAConfig(
        schema_version=1,
        profile="confirmatory",
        study_id="familiarity-answerability-gemma2-2b-v1",
        run_id="confirmatory-v1",
        model_id="google/gemma-2-2b-it",
        model_revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        tokenizer_revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        chat_template_sha256="ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6",
        split_seed=20260722,
        split_counts={
            "mechanism_train": 64,
            "locked_validation": 32,
            "behavior_test": 48,
            "probe_test": 24,
            "intervention_test": 24,
        },
        generation={"do_sample": False, "max_new_tokens": 16, "temperature": 0.0},
        bootstrap_replicates=10000,
        bootstrap_seed=20260722,
        thresholds={
            "format_validity_min": 0.95,
            "h1_min_interaction": 0.05,
            "h2_noninferiority_margin": 0.05,
            "h5_relative_log_loss_min": 0.02,
            "h6_relative_log_loss_min": 0.01,
            "h7_average_effect_min": 0.05,
            "h7_control_margin_min": 0.02,
            "intervention_accuracy_drop_max": 0.05,
            "intervention_control_rate_change_max": 0.03,
            "probe_auroc_min": 0.65,
            "probe_balanced_accuracy_min": 0.55,
            "sae_loss_recovery_min": 0.70,
            "sae_finite_fraction_min": 0.95,
            "circuit_proxy_spearman_min": 0.80,
            "circuit_distribution_spearman_min": 0.80,
            "circuit_perturbation_spearman_min": 0.60,
            "circuit_sign_concordance_min": 0.75,
        },
        anchors=("target_intro_end", "user_prompt_end", "assistant_prefix_end"),
    )


def entity_unit(
    index: int = 0,
    *,
    split: str = "mechanism_train",
    coarse_type: str = "place",
) -> EntityMatch:
    suffix = str(index + 10)
    real_name = f"Old Vale{suffix}"
    synthetic_name = f"New Hill{suffix}"
    return EntityMatch(
        pair_id=f"Q{suffix}--syn-{suffix}",
        real_entity_id=f"Q{suffix}",
        real_qid=f"Q{suffix}",
        synthetic_candidate_id=f"syn-{suffix}",
        real_name=real_name,
        synthetic_name=synthetic_name,
        coarse_type=coarse_type,
        split=split,
        generator_revision="names-v1",
        tokenizer_revision="test-tokenizer-v1",
        real_token_count=2,
        synthetic_token_count=2,
        real_word_count=2,
        synthetic_word_count=2,
        real_character_count=len(real_name),
        synthetic_character_count=len(synthetic_name),
        character_length_delta=0,
        character_tolerance=2,
        capitalization_pattern_equal=True,
    )


def registered_examples(tokenizer=FakeTokenizer()):
    matches = (
        entity_unit(0, split="mechanism_train", coarse_type="place"),
        entity_unit(1, split="mechanism_train", coarse_type="person"),
        entity_unit(2, split="locked_validation", coarse_type="place"),
        entity_unit(3, split="behavior_test", coarse_type="organization"),
        entity_unit(4, split="probe_test", coarse_type="creative_work"),
        entity_unit(5, split="intervention_test", coarse_type="person"),
    )
    return build_factorial_examples(config(), matches, tokenizer=tokenizer), matches


def test_each_entity_unit_expands_over_its_registered_template_families():
    tokenizer = FakeTokenizer()
    rows, _ = registered_examples(tokenizer)

    expected_families = {
        "mechanism_train": TRAIN_TEMPLATE_FAMILIES,
        "locked_validation": VALIDATION_TEMPLATE_FAMILIES,
        "behavior_test": TEST_TEMPLATE_FAMILIES,
        "probe_test": TEST_TEMPLATE_FAMILIES,
        "intervention_test": TEST_TEMPLATE_FAMILIES,
    }
    for split, families in expected_families.items():
        split_rows = [row for row in rows if row.split == split]
        unit_count = len({row.entity_unit_id for row in split_rows})
        assert len(split_rows) == 12 * len(families) * unit_count
        assert {row.template_family for row in split_rows} == set(families)
        assert Counter(
            (row.target_familiarity, row.distractor_familiarity, row.answerability)
            for row in split_rows
        ) == Counter(
            {
                (target, distractor, answerability): len(families) * unit_count
                for target in ("screened_real", "matched_synthetic")
                for distractor in ("screened_real", "matched_synthetic")
                for answerability in ("target_bound", "distractor_bound", "code_absent")
            }
        )


def test_target_and_distractor_bound_pairs_preserve_lexical_multiset_and_code():
    rows, _ = registered_examples()
    target = next(row for row in rows if row.answerability == "target_bound")
    distractor = next(
        row
        for row in rows
        if row.entity_unit_id == target.entity_unit_id
        and row.template_family == target.template_family
        and row.target_familiarity == target.target_familiarity
        and row.distractor_familiarity == target.distractor_familiarity
        and row.answerability == "distractor_bound"
    )

    tokenizer = FakeTokenizer()
    assert Counter(tokenizer.encode(target.user_text)) == Counter(tokenizer.encode(distractor.user_text))
    assert target.registry_code == distractor.registry_code
    assert target.rendered_token_count == distractor.rendered_token_count
    assert target.special_token_sequence == distractor.special_token_sequence


def test_same_string_rows_fix_target_and_token_budget_and_use_one_balanced_family():
    factorial, matches = registered_examples()
    rows = build_same_string_examples(config(), matches, tokenizer=FakeTokenizer())

    assert len(rows) == 4 * len(matches)
    for match in matches:
        unit_rows = [row for row in rows if row.entity_unit_id == match.pair_id]
        assert {row.target_text for row in unit_rows} == {match.synthetic_name}
        assert {(row.exposure, row.answerability) for row in unit_rows} == {
            ("high_exposure", "target_bound"),
            ("high_exposure", "code_absent"),
            ("low_exposure", "target_bound"),
            ("low_exposure", "code_absent"),
        }
        assert len({row.template_family for row in unit_rows}) == 1
        by_answerability = {
            answerability: [row for row in unit_rows if row.answerability == answerability]
            for answerability in ("target_bound", "code_absent")
        }
        for pair in by_answerability.values():
            assert pair[0].rendered_token_count == pair[1].rendered_token_count
            assert pair[0].special_token_sequence == pair[1].special_token_sequence

    assert not {row.example_id for row in factorial} & {row.example_id for row in rows}


def test_examples_and_manifests_are_immutable_and_content_addressed():
    rows, _ = registered_examples()
    manifest = build_manifest(config(), rows)

    assert isinstance(manifest, FAManifest)
    assert manifest.manifest_sha256 == build_manifest(config(), tuple(reversed(rows))).manifest_sha256
    assert len(manifest.manifest_sha256) == 64
    assert rows[0].canonical_payload_sha256 in rows[0].example_id
    with pytest.raises(FrozenInstanceError):
        rows[0].user_text = "changed"
    with pytest.raises(ValueError, match="canonical_payload_sha256"):
        replace(rows[0], canonical_payload_sha256="0" * 64)


def test_audit_covers_registered_controls_and_detects_lexical_tampering():
    rows, _ = registered_examples()
    same_string = build_same_string_examples(config(), _, tokenizer=FakeTokenizer())
    audit = audit_dataset(rows, same_string, tokenizer=FakeTokenizer())

    assert audit.passed
    assert set(audit.checks) == {
        "independent_target_distractor_variation",
        "entity_order",
        "query_role",
        "relation_order",
        "code_position",
        "code_vocabulary",
        "template_overlap",
        "entity_overlap",
        "rendered_token_length",
        "special_token_sequence",
        "lexical_multiset",
        "same_string_token_budget",
    }
    tampered = next(row for row in rows if row.answerability == "distractor_bound")
    object.__setattr__(tampered, "user_text", "tampered")
    object.__setattr__(tampered, "rendered_token_count", 1)
    object.__setattr__(tampered, "rendered_token_ids", ("tampered",))
    broken = audit_dataset(rows, same_string, tokenizer=FakeTokenizer())
    assert not broken.passed
    assert not broken.checks["lexical_multiset"]


def test_power_simulation_uses_registered_grid_is_deterministic_and_has_fast_override():
    rows, _ = registered_examples()
    first = simulate_interaction_power(
        rows,
        effect_grid=(0.0, 0.05),
        within_entity_correlations={"entity_icc": (0.30,), "template_icc": (0.10,), "invalid_format_rate": (0.05,)},
        seed=20260722,
        simulations=25,
    )
    second = simulate_interaction_power(
        rows,
        effect_grid=(0.0, 0.05),
        within_entity_correlations={"entity_icc": (0.30,), "template_icc": (0.10,), "invalid_format_rate": (0.05,)},
        seed=20260722,
        simulations=25,
    )

    assert first == second
    assert len(first.cells) == 3 * 1 * 1 * 1 * 2
    assert all(0.0 <= cell.estimated_power <= 1.0 for cell in first.cells)
    assert all(cell.monte_carlo_standard_error >= 0.0 for cell in first.cells)
    assert REGISTERED_POWER_GRID.interactions[2] == 0.05
    assert CONFIRMATORY_POWER_SIMULATIONS == 2000


def test_confirmatory_manifest_fails_closed_without_a_registered_power_audit():
    matches = tuple(
        entity_unit(index, split=split, coarse_type=("place", "person", "organization", "creative_work")[index % 4])
        for index, split in enumerate(
            ["mechanism_train"] * 64
            + ["locked_validation"] * 32
            + ["behavior_test"] * 48
            + ["probe_test"] * 24
            + ["intervention_test"] * 24
        )
    )
    rows = build_factorial_examples(config(), matches, tokenizer=FakeTokenizer())

    with pytest.raises(ValueError, match="power audit"):
        build_manifest(config(), rows)
