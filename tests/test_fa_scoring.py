from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
import hashlib
from pathlib import Path

import pytest

from trajectory_extractor.fa_config import CONFIRMATORY_THRESHOLDS, FAConfig
from trajectory_extractor.fa_data import FAExample, build_factorial_examples
from trajectory_extractor.fa_entities import EntityMatch
from trajectory_extractor.fa_scoring import (
    OutcomeClass,
    SameStringSealEvidence,
    behavioral_gate,
    crossed_bootstrap,
    estimate_behavior,
    score_response,
)


class RegisteredTokenizer:
    all_special_ids = ()
    chat_template = "registered-scoring-test-template-v1"

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        return self.encode(messages[0]["content"])


def production_example_and_vocabulary() -> tuple[FAExample, frozenset[str], str]:
    """Build real registered rows so vocabulary scoring cannot rely on test-only fields."""
    config = FAConfig.from_json(
        Path(__file__).parents[1] / "configs" / "familiarity_answerability_gemma2_2b.json"
    )
    object.__setattr__(
        config,
        "chat_template_sha256",
        hashlib.sha256(RegisteredTokenizer.chat_template.encode("utf-8")).hexdigest(),
    )
    matches = tuple(
        EntityMatch(
            pair_id=f"Q{index}--syn-{index}",
            real_entity_id=f"Q{index}",
            real_qid=f"Q{index}",
            synthetic_candidate_id=f"syn-{index}",
            real_name=f"Old Vale{index}",
            synthetic_name=f"New Hill{index}",
            coarse_type="place",
            split="mechanism_train",
            generator_revision="names-v1",
            tokenizer_revision="test-tokenizer-v1",
            real_token_count=2,
            synthetic_token_count=2,
            real_word_count=2,
            synthetic_word_count=2,
            real_character_count=len(f"Old Vale{index}"),
            synthetic_character_count=len(f"New Hill{index}"),
            character_length_delta=0,
            character_tolerance=2,
            capitalization_pattern_equal=True,
        )
        for index in (10, 11)
    )
    rows = build_factorial_examples(config, matches, tokenizer=RegisteredTokenizer())
    example = next(row for row in rows if row.answerability == "code_absent")
    vocabulary = frozenset(row.registry_code for row in rows)
    another_registered_code = next(code for code in vocabulary if code != example.registry_code)
    return example, vocabulary, another_registered_code


@dataclass(frozen=True)
class Example:
    example_id: str
    entity_unit_id: str
    template_family: str
    target_familiarity: str
    distractor_familiarity: str
    answerability: str
    registry_code: str = "K7M2Q"
    block: str = "factorial"
    exposure: str | None = None


_TEST_REGISTERED_CODES = frozenset(("K7M2Q", "K8N3R", "K9P4S"))


def score(example: Example, text: str | None, **kwargs):
    return score_response(example, text, registered_codes=_TEST_REGISTERED_CODES, **kwargs)


def example(
    index: int = 0,
    *,
    target_familiarity: str = "screened_real",
    distractor_familiarity: str = "screened_real",
    answerability: str = "code_absent",
    template_family: str = "test_catalog_direct",
    block: str = "factorial",
    exposure: str | None = None,
) -> Example:
    return Example(
        example_id=f"example-{index}-{target_familiarity}-{distractor_familiarity}-{answerability}",
        entity_unit_id=f"unit-{index}",
        template_family=template_family,
        target_familiarity=target_familiarity,
        distractor_familiarity=distractor_familiarity,
        answerability=answerability,
        block=block,
        exposure=exposure,
    )


def absent_example() -> Example:
    return example()


@pytest.mark.parametrize("text", ["UNKNOWN", " UNKNOWN\n", "U\u004eKNOWN"])
def test_exact_normalized_unknown_is_only_non_attempt(text):
    scored = score(absent_example(), text)

    assert scored.answer_attempt == 0
    assert scored.outcome is OutcomeClass.ABSTENTION
    assert scored.completed is True


@pytest.mark.parametrize("text", ["I do not know", "unknown.", "", "K7M2 because..."])
def test_every_other_completion_is_intention_to_treat_attempt(text):
    assert score(absent_example(), text).answer_attempt == 1


def test_parser_precedence_uses_contextual_target_distractor_and_registered_novel_codes():
    target = score(example(answerability="target_bound"), "K7M2Q")
    distractor = score(example(answerability="distractor_bound"), "K7M2Q")
    novel = score(example(answerability="code_absent"), "K8N3R")

    assert target.outcome is OutcomeClass.EXACT_TARGET_CODE
    assert distractor.outcome is OutcomeClass.DISTRACTOR_CODE_COPY
    assert novel.outcome is OutcomeClass.NOVEL_CODE_ASSERTION


def test_real_fa_example_requires_explicit_registered_vocabulary_for_another_registered_code():
    row, vocabulary, another_registered_code = production_example_and_vocabulary()

    with pytest.raises(TypeError):
        score_response(row, another_registered_code)

    try:
        scored = score_response(row, another_registered_code, registered_codes=vocabulary)
    except TypeError as error:
        pytest.fail(f"score_response must accept explicit registered_codes: {error}")

    assert scored.outcome is OutcomeClass.NOVEL_CODE_ASSERTION


@pytest.mark.parametrize(
    ("text", "kwargs"),
    [
        ("", {}),
        ("K7M2Q\nwith explanation", {}),
        ("K7M\t2Q", {}),
        ("K7M2Q", {"truncated": True}),
        ("generation backend unavailable", {"infrastructure_marked": True}),
    ],
)
def test_invalid_format_is_an_attempt_and_is_never_silently_removed(text, kwargs):
    scored = score(absent_example(), text, **kwargs)

    assert scored.outcome is OutcomeClass.INVALID_FORMAT
    assert scored.answer_attempt == 1
    assert scored.valid_format is False


def test_missing_infrastructure_response_is_incomplete_but_received_invalid_output_is_complete():
    missing = score(absent_example(), None, infrastructure_marked=True)
    empty = score(absent_example(), "")

    assert missing.completed is False
    assert empty.completed is True
    assert missing.outcome is empty.outcome is OutcomeClass.INVALID_FORMAT


def complete_factorial_rows(*, invalid_synthetic_target_bound: bool = False):
    rows = []
    for unit in range(2):
        for template_family in ("test_catalog_direct", "test_ledger_direct"):
            for target in ("screened_real", "matched_synthetic"):
                for distractor in ("screened_real", "matched_synthetic"):
                    for answerability in ("target_bound", "distractor_bound", "code_absent"):
                        row = example(
                            unit,
                            target_familiarity=target,
                            distractor_familiarity=distractor,
                            answerability=answerability,
                            template_family=template_family,
                        )
                        if answerability == "target_bound":
                            output = "K7M2Q"
                        elif target == "screened_real":
                            output = "K8N3R"
                        else:
                            output = "UNKNOWN"
                        if invalid_synthetic_target_bound and target == "matched_synthetic" and answerability == "target_bound":
                            output = ""
                        rows.append(score(row, output))
    return tuple(rows)


def test_registered_h1_interaction_equal_weights_absent_states_and_distractor_familiarity():
    metrics = estimate_behavior(complete_factorial_rows())

    assert metrics.status == "evaluable"
    for distractor in ("screened_real", "matched_synthetic"):
        assert metrics.cell_rates[("screened_real", distractor, "distractor_bound")] == pytest.approx(1.0)
        assert metrics.cell_rates[("matched_synthetic", distractor, "distractor_bound")] == pytest.approx(0.0)
    assert metrics.interaction == pytest.approx(1.0)
    assert metrics.h2_accuracy_difference == pytest.approx(0.0)


def test_h1_and_h2_average_distractor_familiarity_equally_despite_unequal_row_counts():
    rows = list(complete_factorial_rows())
    source = next(
        row
        for row in rows
        if (
            row.target_familiarity == "matched_synthetic"
            and row.distractor_familiarity == "screened_real"
            and row.answerability == "target_bound"
        )
    )
    for _ in range(36):
        rows.append(
            replace(
                source,
                raw_output="UNKNOWN",
                normalized_output="UNKNOWN",
                outcome=OutcomeClass.ABSTENTION,
                answer_attempt=0,
            )
        )

    metrics = estimate_behavior(rows)

    assert set(metrics.denominators) == {
        (target, distractor, answerability)
        for target in ("screened_real", "matched_synthetic")
        for distractor in ("screened_real", "matched_synthetic")
        for answerability in ("target_bound", "distractor_bound", "code_absent")
    }
    assert metrics.interaction == pytest.approx(0.55)
    assert metrics.h2_accuracy_difference == pytest.approx(-0.45)


def test_crossed_bootstrap_is_seeded_and_uses_entity_template_multiplicity_weights():
    rows = complete_factorial_rows()

    first = crossed_bootstrap(rows, replicates=31, seed=20260722)
    second = crossed_bootstrap(rows, replicates=31, seed=20260722)

    assert first == second
    assert len(first.interaction_samples) == 31
    assert first.interaction_interval.lower <= first.interaction_interval.upper
    assert all(denominator > 0 for denominator in first.weighted_denominators)


def test_crossed_bootstrap_records_exact_draw_and_resampling_provenance():
    distribution = crossed_bootstrap(
        complete_factorial_rows(), replicates=31, seed=20260722
    )

    assert distribution.requested_draws == 31
    assert distribution.valid_draws == 31
    assert distribution.discarded_draws == 0
    assert distribution.resampling_unit == (
        "entity_unit_id",
        "template_family",
    )
    assert distribution.seed == 20260722
    assert distribution.alpha == pytest.approx(0.05)
    assert distribution.to_record()["resampling_unit"] == [
        "entity_unit_id",
        "template_family",
    ]


def test_bootstrap_interval_keeps_the_observed_statistic_separate_from_resample_distribution():
    rows = list(complete_factorial_rows())
    for position, row in enumerate(rows):
        if (
            row.entity_unit_id == "unit-1"
            and row.target_familiarity == "screened_real"
            and row.answerability in {"distractor_bound", "code_absent"}
        ):
            rows[position] = score(
                example(
                    1,
                    target_familiarity=row.target_familiarity,
                    distractor_familiarity=row.distractor_familiarity,
                    answerability=row.answerability,
                    template_family=row.template_family,
                ),
                "UNKNOWN",
            )

    metrics = estimate_behavior(rows)
    distribution = crossed_bootstrap(rows, replicates=17, seed=123)

    assert metrics.interaction == pytest.approx(0.5)
    assert distribution.interaction_interval.estimate == pytest.approx(metrics.interaction)


def test_completion_below_95_percent_makes_endpoint_not_evaluable_without_dropping_invalid_rows():
    rows = list(complete_factorial_rows())
    source = next(
        row
        for row in rows
        if row.target_familiarity == "matched_synthetic" and row.answerability == "code_absent"
    )
    rows[rows.index(source)] = score(
        example(
            0,
            target_familiarity="matched_synthetic",
            distractor_familiarity=source.distractor_familiarity,
            answerability="code_absent",
            template_family=source.template_family,
        ),
        None,
        infrastructure_marked=True,
    )

    metrics = estimate_behavior(rows)

    assert metrics.status == "not_evaluable"
    assert any(reason.startswith("completion<0.95") for reason in metrics.reasons)
    cell = ("matched_synthetic", source.distractor_familiarity, "code_absent")
    assert metrics.denominators[cell] == 4
    assert metrics.invalid_format_counts[cell] == 1


def same_string_rows():
    rows = []
    for unit in range(2):
        for exposure in ("high_exposure", "low_exposure"):
            for answerability in ("target_bound", "code_absent"):
                row = example(
                    unit,
                    answerability=answerability,
                    template_family=("test_catalog_direct", "test_ledger_direct")[unit],
                    block="same_string",
                    exposure=exposure,
                )
                output = "K7M2Q" if answerability == "target_bound" else (
                    "K8N3R" if exposure == "high_exposure" else "UNKNOWN"
                )
                rows.append(score(row, output))
    return tuple(rows)


def test_gate_reports_h1_h2_and_h2b_separately_and_h2b_cannot_rescue_h1():
    factorial = complete_factorial_rows(invalid_synthetic_target_bound=True)
    metrics = estimate_behavior((*factorial, *same_string_rows()))
    distribution = crossed_bootstrap((*factorial, *same_string_rows()), replicates=101, seed=42)

    gate = behavioral_gate(
        metrics,
        distribution,
        thresholds=CONFIRMATORY_THRESHOLDS,
        same_string_sealed=True,
        config_hash="a" * 64,
        manifest_hash="b" * 64,
    )

    assert gate.h1.status == "not_supported"
    assert gate.h2.status == "not_supported"
    assert gate.h2b.status == "supported"
    assert gate.status == "not_supported"
    assert gate.h2b_cannot_rescue_h1 is True


def test_behavioral_gate_requires_and_records_exact_registered_provenance():
    rows = (*complete_factorial_rows(), *same_string_rows())
    metrics = estimate_behavior(rows)
    distribution = crossed_bootstrap(rows, replicates=17, seed=42)

    with pytest.raises(TypeError):
        behavioral_gate(metrics, distribution)

    gate = behavioral_gate(
        metrics,
        distribution,
        thresholds=CONFIRMATORY_THRESHOLDS,
        same_string_sealed=True,
        config_hash="a" * 64,
        manifest_hash="b" * 64,
    )

    assert gate.same_string_sealed is True
    assert gate.config_hash == "a" * 64
    assert gate.manifest_hash == "b" * 64
    assert dict(gate.thresholds) == CONFIRMATORY_THRESHOLDS
    assert gate.to_record()["thresholds"] == CONFIRMATORY_THRESHOLDS

    altered_thresholds = {**CONFIRMATORY_THRESHOLDS, "h5_relative_log_loss_min": 0.03}
    with pytest.raises(ValueError, match="registered thresholds"):
        behavioral_gate(
            metrics,
            distribution,
            thresholds=altered_thresholds,
            same_string_sealed=True,
            config_hash="a" * 64,
            manifest_hash="b" * 64,
        )
    with pytest.raises(ValueError, match="config_hash"):
        behavioral_gate(
            metrics,
            distribution,
            thresholds=CONFIRMATORY_THRESHOLDS,
            same_string_sealed=True,
            config_hash="A" * 64,
            manifest_hash="b" * 64,
        )


def test_same_string_seal_is_typed_immutable_and_manifest_bound():
    rows = (*complete_factorial_rows(), *same_string_rows())
    metrics = estimate_behavior(rows)
    distribution = crossed_bootstrap(rows, replicates=17, seed=42)
    seal = SameStringSealEvidence.from_registered_block(
        source_manifest_sha256="b" * 64,
        example_ids=("same-string-2", "same-string-1"),
    )

    gate = behavioral_gate(
        metrics,
        distribution,
        thresholds=CONFIRMATORY_THRESHOLDS,
        same_string_sealed=True,
        config_hash="a" * 64,
        manifest_hash="b" * 64,
        same_string_seal=seal,
    )

    assert gate.same_string_seal == seal
    assert gate.to_record()["same_string_seal_sha256"] == seal.sha256
    assert gate.to_record()["same_string_seal"]["example_ids"] == [
        "same-string-1",
        "same-string-2",
    ]
    with pytest.raises(FrozenInstanceError):
        seal.block = "factorial"
    with pytest.raises(ValueError, match="source manifest"):
        behavioral_gate(
            metrics,
            distribution,
            thresholds=CONFIRMATORY_THRESHOLDS,
            same_string_sealed=True,
            config_hash="a" * 64,
            manifest_hash="c" * 64,
            same_string_seal=seal,
        )


def test_scored_responses_and_provenance_outputs_are_immutable_and_serializable():
    scored = score(absent_example(), "UNKNOWN")

    with pytest.raises(FrozenInstanceError):
        scored.answer_attempt = 1
    assert scored.to_record()["outcome"] == "abstention"
    assert scored.to_record()["example_id"] == scored.example_id
