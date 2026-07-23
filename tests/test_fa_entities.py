import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, validators

from trajectory_extractor.fa_entities import (
    CandidateEntity,
    EntityMatch,
    NaturalnessRating,
    ScreeningQuestion,
    SyntheticCandidate,
    audit_naturalness_manifest,
    match_synthetic_entities,
    order_screening_questions,
    score_screening,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "data" / "fa" / "schemas"
STRICT_DRAFT202012_VALIDATOR = validators.extend(
    Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
        "integer", lambda _checker, value: type(value) is int
    ),
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.replace(" ", "|").split("|")


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


def candidate(**changes):
    payload = {
        "entity_id": "Q90",
        "qid": "Q90",
        "name": "Paris",
        "coarse_type": "place",
        "split": "mechanism_train",
        "source_query": "wikidata-query-v1",
        "source_provenance": "CC0-1.0",
        "screening_aliases": (("Paris",), ("France",), ("French Republic", "France")),
    }
    payload.update(changes)
    return CandidateEntity(**payload)


def entity_match(**changes):
    payload = {
        "pair_id": "Q90--syn-2",
        "real_entity_id": "Q90",
        "real_qid": "Q90",
        "synthetic_candidate_id": "syn-2",
        "real_name": "Old Vale",
        "synthetic_name": "New Vale",
        "coarse_type": "place",
        "split": "mechanism_train",
        "generator_revision": "names-v1",
        "tokenizer_revision": "test-tokenizer-v1",
        "real_token_count": 9,
        "synthetic_token_count": 9,
        "real_word_count": 2,
        "synthetic_word_count": 2,
        "real_character_count": 8,
        "synthetic_character_count": 8,
        "character_length_delta": 0,
        "character_tolerance": 2,
        "capitalization_pattern_equal": True,
    }
    payload.update(changes)
    return EntityMatch(**payload)


def synthetic_pool():
    return (
        SyntheticCandidate(
            candidate_id="syn-2",
            name="New Vale",
            coarse_type="place",
            split="mechanism_train",
            generator_revision="names-v1",
        ),
        SyntheticCandidate(
            candidate_id="syn-1",
            name="Far Hill",
            coarse_type="person",
            split="mechanism_train",
            generator_revision="names-v1",
        ),
    )


def test_records_are_immutable_and_require_qid_source_and_three_alias_sets():
    entity = candidate()
    assert entity.screening_aliases[2] == ("French Republic", "France")
    with pytest.raises(FrozenInstanceError):
        entity.name = "Lutetia"
    with pytest.raises(ValueError, match="QID"):
        candidate(qid="Paris")
    with pytest.raises(ValueError, match="source provenance"):
        candidate(source_provenance="")
    with pytest.raises(ValueError, match="three"):
        candidate(screening_aliases=(("Paris",),))


def test_screening_requires_two_of_three_alias_correct_answers():
    result = score_screening(candidate(), [" Paris ", "FRANCE", "wrong"])

    assert result.qualifies is True
    assert result.recall_score == pytest.approx(2 / 3)
    assert result.correct_answers == (True, True, False)


def test_screening_rejects_substring_and_requires_three_completions():
    result = score_screening(candidate(), ["Paris, France", "France", "wrong"])
    assert result.correct_answers == (False, True, False)
    with pytest.raises(ValueError, match="three completions"):
        score_screening(candidate(), ["Paris"])


def test_screening_questions_join_exactly_three_alias_consistent_rows():
    entity = candidate()
    questions = tuple(
        ScreeningQuestion(
            question_id=f"Q90-{index}",
            qid="Q90",
            prompt=f"Question {index}",
            accepted_aliases=aliases,
            source_provenance="https://www.wikidata.org/wiki/Q90",
        )
        for index, aliases in enumerate(entity.screening_aliases, start=1)
    )

    joined = order_screening_questions([entity], reversed(questions))

    assert joined[0][0] == entity
    assert tuple(question.question_id for question in joined[0][1]) == (
        "Q90-1",
        "Q90-2",
        "Q90-3",
    )
    with pytest.raises(ValueError, match="exactly three"):
        order_screening_questions([entity], questions[:2])
    with pytest.raises(ValueError, match="aliases"):
        order_screening_questions(
            [entity],
            (
                ScreeningQuestion(
                    question_id="Q90-1",
                    qid="Q90",
                    prompt="Question 1",
                    accepted_aliases=("wrong",),
                    source_provenance="source",
                ),
                *questions[1:],
            ),
        )


def test_v3_pilot_pool_is_complete_balanced_and_alias_consistent():
    input_dir = REPO_ROOT / "data" / "fa" / "pilot_inputs"
    candidate_rows = json.loads(
        (input_dir / "candidates_v3.json").read_text(encoding="utf-8")
    )
    question_rows = json.loads(
        (input_dir / "screening_questions_v3.json").read_text(encoding="utf-8")
    )
    synthetic_rows = json.loads(
        (input_dir / "synthetic_candidates_v3.json").read_text(encoding="utf-8")
    )
    candidates = tuple(
        CandidateEntity(**{key: value for key, value in row.items() if key != "schema_version"})
        for row in candidate_rows
    )
    questions = tuple(
        ScreeningQuestion(**{key: value for key, value in row.items() if key != "schema_version"})
        for row in question_rows
    )
    synthetics = tuple(
        SyntheticCandidate(**{key: value for key, value in row.items() if key != "schema_version"})
        for row in synthetic_rows
    )

    assert len(candidates) == 20
    assert len(questions) == 60
    assert len(synthetics) == 20
    assert {
        domain: sum(candidate.coarse_type == domain for candidate in candidates)
        for domain in ("person", "place", "organization", "creative_work")
    } == {
        "person": 5,
        "place": 5,
        "organization": 5,
        "creative_work": 5,
    }
    assert {
        domain: sum(candidate.coarse_type == domain for candidate in synthetics)
        for domain in ("person", "place", "organization", "creative_work")
    } == {
        "person": 5,
        "place": 5,
        "organization": 5,
        "creative_work": 5,
    }
    assert len(order_screening_questions(candidates, questions)) == len(candidates)
    assert {candidate.qid for candidate in candidates} == {
        question.qid for question in questions
    }


def test_v4_pilot_pool_is_append_only_and_alias_consistent():
    input_dir = REPO_ROOT / "data" / "fa" / "pilot_inputs"
    candidate_v3 = json.loads(
        (input_dir / "candidates_v3.json").read_text(encoding="utf-8")
    )
    question_v3 = json.loads(
        (input_dir / "screening_questions_v3.json").read_text(encoding="utf-8")
    )
    synthetic_v3 = json.loads(
        (input_dir / "synthetic_candidates_v3.json").read_text(encoding="utf-8")
    )
    candidate_rows = json.loads(
        (input_dir / "candidates_v4.json").read_text(encoding="utf-8")
    )
    question_rows = json.loads(
        (input_dir / "screening_questions_v4.json").read_text(encoding="utf-8")
    )
    synthetic_rows = json.loads(
        (input_dir / "synthetic_candidates_v4.json").read_text(encoding="utf-8")
    )
    candidates = tuple(
        CandidateEntity(**{key: value for key, value in row.items() if key != "schema_version"})
        for row in candidate_rows
    )
    questions = tuple(
        ScreeningQuestion(**{key: value for key, value in row.items() if key != "schema_version"})
        for row in question_rows
    )

    assert candidate_rows[: len(candidate_v3)] == candidate_v3
    assert question_rows[: len(question_v3)] == question_v3
    assert synthetic_rows[: len(synthetic_v3)] == synthetic_v3
    assert [row["entity_id"] for row in candidate_rows[len(candidate_v3) :]] == [
        "pilot-work-jaws",
        "pilot-work-mulan",
        "pilot-work-moana",
        "pilot-work-skyfall",
    ]
    assert len(candidates) == 24
    assert len(questions) == 72
    assert len(synthetic_rows) == 21
    assert len(order_screening_questions(candidates, questions)) == len(candidates)
    assert {candidate.qid for candidate in candidates} == {
        question.qid for question in questions
    }


def test_matching_enforces_token_and_surface_constraints(fake_tokenizer):
    match = match_synthetic_entities([candidate(name="Old Vale")], synthetic_pool(), fake_tokenizer)[0]
    assert match.real_token_count == match.synthetic_token_count
    assert match.real_word_count == match.synthetic_word_count
    assert match.capitalization_pattern_equal
    assert match.character_length_delta == 0
    assert match.synthetic_name == "New Vale"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"pair_id": "bad-pair"}, "pair_id"),
        ({"pair_id": "Q91--syn-2"}, "pair_id"),
        ({"real_entity_id": "bad id"}, "real_entity_id"),
        ({"real_qid": "Q0"}, "real_qid"),
        ({"synthetic_candidate_id": "bad id"}, "synthetic_candidate_id"),
        ({"real_name": ""}, "real_name"),
        ({"synthetic_name": "Old Vale"}, "synthetic_name"),
        ({"coarse_type": ""}, "coarse_type"),
        ({"split": "unregistered"}, "split"),
        ({"generator_revision": ""}, "generator_revision"),
        ({"tokenizer_revision": ""}, "tokenizer_revision"),
        ({"real_token_count": 0}, "real_token_count"),
        ({"synthetic_token_count": 8}, "token counts"),
        ({"real_word_count": 1}, "real_word_count"),
        ({"synthetic_word_count": 1}, "synthetic_word_count"),
        ({"real_character_count": 7}, "real_character_count"),
        ({"synthetic_character_count": 7}, "synthetic_character_count"),
        ({"character_length_delta": 1}, "character_length_delta"),
        ({"character_tolerance": 1}, "character_tolerance"),
        ({"character_tolerance": True}, "character_tolerance"),
        ({"real_token_count": True}, "real_token_count"),
        ({"real_word_count": 2.0}, "real_word_count"),
        ({"real_character_count": 8.0}, "real_character_count"),
        ({"synthetic_name": "new Vale"}, "capitalization_pattern_equal"),
        ({"capitalization_pattern_equal": False}, "capitalization_pattern_equal"),
        ({"capitalization_pattern_equal": 1}, "capitalization_pattern_equal"),
    ],
)
def test_entity_match_rejects_malformed_or_inconsistent_audit_inputs(changes, message):
    with pytest.raises(ValueError, match=message):
        entity_match(**changes)


def test_matching_rejects_duplicate_names_reuse_and_split_crossing_reserves(fake_tokenizer):
    real = candidate(name="Old Vale")
    with pytest.raises(ValueError, match="duplicate"):
        match_synthetic_entities([real, candidate(entity_id="Q91", qid="Q91", name="Old Vale")], synthetic_pool(), fake_tokenizer)
    with pytest.raises(ValueError, match="duplicate"):
        match_synthetic_entities(
            [real, candidate(entity_id="entity-91", qid="Q90", name="Old Dale")],
            synthetic_pool(),
            fake_tokenizer,
        )
    only_candidate = SyntheticCandidate("syn-1", "New Vale", "place", "mechanism_train", "names-v1")
    with pytest.raises(ValueError, match="reuse"):
        match_synthetic_entities(
            [real, candidate(entity_id="Q91", qid="Q91", name="Old Dale")],
            [only_candidate],
            fake_tokenizer,
        )
    crossing_reserve = SyntheticCandidate(
        candidate_id="syn-cross",
        name="New Vale",
        coarse_type="place",
        split="behavior_test",
        generator_revision="names-v1",
    )
    with pytest.raises(ValueError, match="split"):
        match_synthetic_entities([real], [crossing_reserve], fake_tokenizer)


def test_matching_rejects_registered_character_tolerance_and_surface_mismatches(fake_tokenizer):
    real = candidate(name="Old Vale")
    too_long = SyntheticCandidate("syn-1", "Very Long Vale", "place", "mechanism_train", "names-v1")
    wrong_case = SyntheticCandidate("syn-2", "new Vale", "place", "mechanism_train", "names-v1")
    with pytest.raises(ValueError, match="no eligible"):
        match_synthetic_entities([real], [too_long], fake_tokenizer)
    with pytest.raises(ValueError, match="no eligible"):
        match_synthetic_entities([real], [wrong_case], fake_tokenizer)


def rating(pair_id, rater_id, real=4, synthetic=4, malformed=False, round=1, disagreement_registered=False, schema_version=1):
    return NaturalnessRating(
        pair_id=pair_id,
        rater_id=rater_id,
        real_naturalness=real,
        synthetic_naturalness=synthetic,
        real_type_fit=5,
        synthetic_type_fit=5,
        synthetic_malformed=malformed,
        round=round,
        disagreement_registered=disagreement_registered,
        schema_version=schema_version,
    )


def test_naturalness_audit_requires_two_independent_raters_and_excludes_bad_pairs(fake_tokenizer):
    match = match_synthetic_entities([candidate(name="Old Vale")], synthetic_pool(), fake_tokenizer)[0]
    audit = audit_naturalness_manifest(
        [match], [rating(match.pair_id, "rater-a", real=5, synthetic=3), rating(match.pair_id, "rater-b", real=5, synthetic=3)]
    )
    assert audit.accepted_pair_ids == ()
    assert audit.excluded_pair_ids == (match.pair_id,)
    with pytest.raises(ValueError, match="independent"):
        audit_naturalness_manifest([match], [rating(match.pair_id, "rater-a"), rating(match.pair_id, "rater-a")])


def test_naturalness_audit_allows_registered_third_rater_only_for_disagreement(fake_tokenizer):
    match = match_synthetic_entities([candidate(name="Old Vale")], synthetic_pool(), fake_tokenizer)[0]
    audit = audit_naturalness_manifest(
        [match],
        [
            rating(match.pair_id, "rater-a", real=5, synthetic=3),
            rating(match.pair_id, "rater-b", real=4, synthetic=4),
            rating(match.pair_id, "rater-c", real=4, synthetic=4, round=2, disagreement_registered=True),
        ],
    )
    assert audit.accepted_pair_ids == (match.pair_id,)
    with pytest.raises(ValueError, match="third rater"):
        audit_naturalness_manifest(
            [match],
            [rating(match.pair_id, "rater-a"), rating(match.pair_id, "rater-b"), rating(match.pair_id, "rater-c", round=2, disagreement_registered=True)],
        )


@pytest.mark.parametrize(
    ("factory", "changes"),
    [
        (candidate, {"schema_version": True}),
        (candidate, {"schema_version": 1.0}),
        (
            lambda **changes: ScreeningQuestion(
                question_id="Q90-1",
                qid="Q90",
                prompt="What country is Paris in?",
                accepted_aliases=("France",),
                source_provenance="CC0-1.0",
                **changes,
            ),
            {"schema_version": True},
        ),
        (
            lambda **changes: ScreeningQuestion(
                question_id="Q90-1",
                qid="Q90",
                prompt="What country is Paris in?",
                accepted_aliases=("France",),
                source_provenance="CC0-1.0",
                **changes,
            ),
            {"schema_version": 1.0},
        ),
        (
            lambda **changes: SyntheticCandidate(
                "syn-2",
                "New Vale",
                "place",
                "mechanism_train",
                "names-v1",
                **changes,
            ),
            {"schema_version": True},
        ),
        (
            lambda **changes: SyntheticCandidate(
                "syn-2",
                "New Vale",
                "place",
                "mechanism_train",
                "names-v1",
                **changes,
            ),
            {"schema_version": 1.0},
        ),
        (entity_match, {"schema_version": True}),
        (entity_match, {"schema_version": 1.0}),
        (lambda **changes: rating("Q90--syn-2", "rater-a", **changes), {"schema_version": True}),
        (lambda **changes: rating("Q90--syn-2", "rater-a", **changes), {"schema_version": 1.0}),
    ],
)
def test_runtime_audit_records_reject_schema_version_aliases(factory, changes):
    with pytest.raises(ValueError, match="schema_version"):
        factory(**changes)


def test_naturalness_rating_requires_a_canonical_pair_id():
    with pytest.raises(ValueError, match="pair_id"):
        rating("not-a-pair", "rater-a")


@pytest.mark.parametrize(
    ("factory", "schema_name"),
    [
        (candidate, "candidate_entity.schema.json"),
        (
            lambda: ScreeningQuestion(
                question_id="Q90-1",
                qid="Q90",
                prompt="What country is Paris in?",
                accepted_aliases=("France",),
                source_provenance="CC0-1.0",
            ),
            "screening_question.schema.json",
        ),
        (
            lambda: SyntheticCandidate(
                "syn-2",
                "New Vale",
                "place",
                "mechanism_train",
                "names-v1",
            ),
            "synthetic_candidate.schema.json",
        ),
        (entity_match, "synthetic_match.schema.json"),
        (lambda: rating("Q90--syn-2", "rater-a"), "naturalness_rating.schema.json"),
    ],
)
def test_runtime_records_serialize_to_their_complete_schemas(factory, schema_name):
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    payload = json.loads(json.dumps(asdict(factory())))

    STRICT_DRAFT202012_VALIDATOR(schema).validate(payload)


@pytest.mark.parametrize(
    "schema_name",
    [
        "candidate_entity.schema.json",
        "screening_question.schema.json",
        "synthetic_candidate.schema.json",
        "synthetic_match.schema.json",
        "naturalness_rating.schema.json",
    ],
)
@pytest.mark.parametrize("schema_version", [True, 1.0, 2])
def test_contract_schemas_require_exact_integer_schema_version(schema_name, schema_version):
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    valid_payloads = {
        "candidate_entity.schema.json": asdict(candidate()),
        "screening_question.schema.json": {
            "schema_version": 1,
            "question_id": "Q90-1",
            "qid": "Q90",
            "prompt": "What country is Paris in?",
            "accepted_aliases": ["France"],
            "source_provenance": "CC0-1.0",
        },
        "synthetic_candidate.schema.json": asdict(
            SyntheticCandidate(
                "syn-2",
                "New Vale",
                "place",
                "mechanism_train",
                "names-v1",
            )
        ),
        "synthetic_match.schema.json": asdict(entity_match()),
        "naturalness_rating.schema.json": asdict(rating("Q90--syn-2", "rater-a")),
    }
    payload = json.loads(json.dumps(valid_payloads[schema_name]))
    payload["schema_version"] = schema_version

    with pytest.raises(Exception):
        STRICT_DRAFT202012_VALIDATOR(schema).validate(payload)


@pytest.mark.parametrize(
    "schema_name, required",
    [
        ("candidate_entity.schema.json", {"schema_version", "qid", "source_provenance", "split"}),
        ("screening_question.schema.json", {"schema_version", "question_id", "accepted_aliases"}),
        (
            "screening_completion.schema.json",
            {
                "schema_version",
                "question_id",
                "raw_output",
                "answer_text",
                "config_sha256",
            },
        ),
        (
            "synthetic_candidate.schema.json",
            {"schema_version", "candidate_id", "generator_revision", "split"},
        ),
        ("synthetic_match.schema.json", {"schema_version", "pair_id", "tokenizer_revision", "character_tolerance"}),
        ("naturalness_rating.schema.json", {"schema_version", "pair_id", "rater_id", "synthetic_malformed"}),
    ],
)
def test_contract_schemas_are_strict_and_cover_provenance(schema_name, required):
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert required <= set(schema["required"])
