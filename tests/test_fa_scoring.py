from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from trajectory_extractor.fa_scoring import (
    OutcomeClass,
    behavioral_gate,
    crossed_bootstrap,
    estimate_behavior,
    score_response,
)


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
    registered_codes: tuple[str, ...] = ("K7M2Q", "K8N3R", "K9P4S")


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
    scored = score_response(absent_example(), text)

    assert scored.answer_attempt == 0
    assert scored.outcome is OutcomeClass.ABSTENTION
    assert scored.completed is True


@pytest.mark.parametrize("text", ["I do not know", "unknown.", "", "K7M2 because..."])
def test_every_other_completion_is_intention_to_treat_attempt(text):
    assert score_response(absent_example(), text).answer_attempt == 1


def test_parser_precedence_uses_contextual_target_distractor_and_registered_novel_codes():
    target = score_response(example(answerability="target_bound"), "K7M2Q")
    distractor = score_response(example(answerability="distractor_bound"), "K7M2Q")
    novel = score_response(example(answerability="code_absent"), "K8N3R")

    assert target.outcome is OutcomeClass.EXACT_TARGET_CODE
    assert distractor.outcome is OutcomeClass.DISTRACTOR_CODE_COPY
    assert novel.outcome is OutcomeClass.NOVEL_CODE_ASSERTION


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
    scored = score_response(absent_example(), text, **kwargs)

    assert scored.outcome is OutcomeClass.INVALID_FORMAT
    assert scored.answer_attempt == 1
    assert scored.valid_format is False


def test_missing_infrastructure_response_is_incomplete_but_received_invalid_output_is_complete():
    missing = score_response(absent_example(), None, infrastructure_marked=True)
    empty = score_response(absent_example(), "")

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
                        rows.append(score_response(row, output))
    return tuple(rows)


def test_registered_h1_interaction_equal_weights_absent_states_and_distractor_familiarity():
    metrics = estimate_behavior(complete_factorial_rows())

    assert metrics.status == "evaluable"
    assert metrics.cell_rates[("screened_real", "distractor_bound")] == pytest.approx(1.0)
    assert metrics.cell_rates[("matched_synthetic", "distractor_bound")] == pytest.approx(0.0)
    assert metrics.interaction == pytest.approx(1.0)
    assert metrics.h2_accuracy_difference == pytest.approx(0.0)


def test_crossed_bootstrap_is_seeded_and_uses_entity_template_multiplicity_weights():
    rows = complete_factorial_rows()

    first = crossed_bootstrap(rows, replicates=31, seed=20260722)
    second = crossed_bootstrap(rows, replicates=31, seed=20260722)

    assert first == second
    assert len(first.interaction_samples) == 31
    assert first.interaction_interval.lower <= first.interaction_interval.upper
    assert all(denominator > 0 for denominator in first.weighted_denominators)


def test_bootstrap_interval_keeps_the_observed_statistic_separate_from_resample_distribution():
    rows = list(complete_factorial_rows())
    for position, row in enumerate(rows):
        if (
            row.entity_unit_id == "unit-1"
            and row.target_familiarity == "screened_real"
            and row.answerability in {"distractor_bound", "code_absent"}
        ):
            rows[position] = score_response(
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
    rows[rows.index(source)] = score_response(
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
    assert metrics.denominators[("matched_synthetic", "code_absent")] == 8
    assert metrics.invalid_format_counts[("matched_synthetic", "code_absent")] == 1


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
                rows.append(score_response(row, output))
    return tuple(rows)


def test_gate_reports_h1_h2_and_h2b_separately_and_h2b_cannot_rescue_h1():
    factorial = complete_factorial_rows(invalid_synthetic_target_bound=True)
    metrics = estimate_behavior((*factorial, *same_string_rows()))
    distribution = crossed_bootstrap((*factorial, *same_string_rows()), replicates=101, seed=42)

    gate = behavioral_gate(metrics, distribution, same_string_sealed=True)

    assert gate.h1.status == "not_supported"
    assert gate.h2.status == "not_supported"
    assert gate.h2b.status == "supported"
    assert gate.status == "not_supported"
    assert gate.h2b_cannot_rescue_h1 is True


def test_scored_responses_and_provenance_outputs_are_immutable_and_serializable():
    scored = score_response(absent_example(), "UNKNOWN")

    with pytest.raises(FrozenInstanceError):
        scored.answer_attempt = 1
    assert scored.to_record()["outcome"] == "abstention"
    assert scored.to_record()["example_id"] == scored.example_id
