from dataclasses import replace

import pytest

from trajectory_extractor.rlmf_format import (
    AuditRow,
    alias_exact_match,
    bound_differential_judge_bias,
    build_judge_audit_sample,
    completion_equivalent,
    normalized_answer,
    parse_metascore_output,
    parse_rlmf_output,
    score_blinded_judge_audit,
)


def test_strict_answer_and_metascore_schemas_accept_only_the_registered_form():
    answer = parse_rlmf_output(
        "<sentence>William Shakespeare</sentence>\n<confidence>0.8</confidence>"
    )
    metascore = parse_metascore_output("<metascore>0.3</metascore>")

    assert answer.answer == "William Shakespeare"
    assert answer.confidence == 0.8
    assert answer.valid_format is True
    assert metascore.metascore == 0.3
    assert metascore.valid_format is True


@pytest.mark.parametrize(
    "text",
    [
        "<sentence>answer</sentence><confidence>0.8</confidence> trailing",
        "<sentence>answer</sentence><confidence>0.8</confidence><confidence>0.8</confidence>",
        "<sentence>answer</sentence>",
        "<sentence>answer</sentence><confidence>NaN</confidence>",
        "<sentence>answer</sentence><confidence>inf</confidence>",
        "<sentence>answer</sentence><confidence>0.81</confidence>",
        "<sentence>answer</sentence><confidence>1.1</confidence>",
        "<sentence></sentence><confidence>0.8</confidence>",
    ],
)
def test_malformed_answer_outputs_are_invalid_without_guessing(text):
    parsed = parse_rlmf_output(text)

    assert parsed.valid_format is False
    assert parsed.answer == ""
    assert parsed.confidence is None


@pytest.mark.parametrize(
    "text",
    [
        "<metascore>0.4</metascore> trailing",
        "<metascore>0.4</metascore><metascore>0.4</metascore>",
        "<metascore>nan</metascore>",
        "<metascore>1.01</metascore>",
        "<metascore>0.40</metascore>",
    ],
)
def test_malformed_metascores_are_invalid_without_guessing(text):
    parsed = parse_metascore_output(text)

    assert parsed.valid_format is False
    assert parsed.metascore is None


def test_alias_judge_uses_unicode_normalization_and_never_substrings():
    assert normalized_answer(" The\u00a0\uff37\uff49\uff4c\uff4c\uff49\uff41\uff4d, Shakespeare! ") == "william shakespeare"
    assert alias_exact_match("The U.S.A.", ("usa",))
    assert not alias_exact_match("Shakespeare", ("William Shakespeare",))
    assert completion_equivalent("William Shakespeare", "Shakespeare", ("William Shakespeare", "Shakespeare"))
    assert completion_equivalent("A Mercury", "Mercury", ("Mercury",))
    assert not completion_equivalent("Mercury", "Freddie Mercury", ("Mercury",))


def test_audit_sampling_is_deterministic_balanced_and_blinded():
    candidates = _candidates(per_stratum=50)

    first = build_judge_audit_sample(candidates, phase="locked", size=400, seed=17)
    second = build_judge_audit_sample(list(reversed(candidates)), phase="locked", size=400, seed=17)

    assert first == second
    assert len(first) == 400
    assert _stratum_counts(first) == {
        (arm, judgment_type, label): 50
        for arm in ("standard_grpo", "rlmf")
        for judgment_type in ("correctness", "equivalence")
        for label in (False, True)
    }
    payload = first[0].rater_payload()
    assert set(payload) == {"audit_id", "judgment_type", "question", "answer", "comparison_answer", "reference_answer"}
    assert not {"arm", "seed", "confidence", "reward", "model_id", "proxy_label"} & set(payload)


def test_audit_sampling_enforces_registered_phase_sizes_and_available_strata():
    candidates = _candidates(per_stratum=125)

    test_rows = build_judge_audit_sample(candidates, phase="test", size=1000, seed=17)
    assert set(_stratum_counts(test_rows).values()) == {125}
    with pytest.raises(ValueError, match="registered"):
        build_judge_audit_sample(candidates, phase="locked", size=200, seed=17)
    with pytest.raises(ValueError, match="insufficient"):
        build_judge_audit_sample(candidates[:-1], phase="test", size=1000, seed=17)


def test_locked_audit_requires_two_raters_adjudication_and_registered_gates():
    rows = _rated(build_judge_audit_sample(_candidates(per_stratum=50), phase="locked", size=400, seed=1))
    decision = score_blinded_judge_audit(rows)

    assert decision.phase == "locked"
    assert decision.passed is True
    assert decision.status == "passed"
    assert decision.kappa == 1.0
    assert all(value == 1.0 for value in decision.sensitivity.values())
    assert all(value == 1.0 for value in decision.specificity.values())

    disagreement = replace(rows[0], rater_b="incorrect", adjudicated_label=None)
    with pytest.raises(ValueError, match="adjudicated"):
        score_blinded_judge_audit((disagreement, *rows[1:]))

    ambiguous = tuple(
        replace(row, rater_a="ambiguous", rater_b="ambiguous") if index < 21 else row
        for index, row in enumerate(rows)
    )
    ambiguous_decision = score_blinded_judge_audit(ambiguous)
    assert ambiguous_decision.passed is False
    assert ambiguous_decision.ambiguous_fraction > 0.05


def test_test_audit_only_measures_bias_and_bounds_differential_error_deterministically():
    rows = _rated(build_judge_audit_sample(_candidates(per_stratum=125), phase="test", size=1000, seed=2))
    decision = score_blinded_judge_audit(rows)
    bound = bound_differential_judge_bias(rows, replicates=200)

    assert decision.status == "measurement_bias_only"
    assert decision.passed is None
    assert bound.upper == 0.0

    biased = tuple(
        replace(row, rater_b="incorrect", adjudicated_label="incorrect")
        if (
            row.arm == "rlmf"
            and row.proxy_label
            and row.judgment_type == "correctness"
            and int(row.source_id.rsplit("-", 1)[1]) < 20
        )
        else row
        for row in rows
    )
    biased_bound = bound_differential_judge_bias(biased, replicates=200)
    assert biased_bound.estimate > 0.015
    assert biased_bound.upper >= biased_bound.estimate


def _candidates(*, per_stratum):
    candidates = []
    for arm in ("standard_grpo", "rlmf"):
        for judgment_type in ("correctness", "equivalence"):
            for proxy_label in (False, True):
                for index in range(per_stratum):
                    candidates.append(
                        {
                            "candidate_id": f"{arm}-{judgment_type}-{proxy_label}-{index}",
                            "arm": arm,
                            "seed": 11,
                            "confidence": 0.8,
                            "reward": 1.0,
                            "model_id": "hidden-model",
                            "judgment_type": judgment_type,
                            "proxy_label": proxy_label,
                            "question": "Who wrote Hamlet?",
                            "answer": "William Shakespeare" if proxy_label else "Marlowe",
                            "comparison_answer": "William Shakespeare" if proxy_label else "Marlowe",
                            "reference_answer": "William Shakespeare",
                        }
                    )
    return candidates


def _rated(rows):
    return tuple(
        replace(
            row,
            rater_a="correct" if row.proxy_label else "incorrect",
            rater_b="correct" if row.proxy_label else "incorrect",
        )
        for row in rows
    )


def _stratum_counts(rows):
    return {
        key: sum(
            (row.arm, row.judgment_type, row.proxy_label) == key
            for row in rows
        )
        for key in {
            (row.arm, row.judgment_type, row.proxy_label)
            for row in rows
        }
    }
