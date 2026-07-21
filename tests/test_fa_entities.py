import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from trajectory_extractor.fa_entities import (
    CandidateEntity,
    NaturalnessRating,
    SyntheticCandidate,
    audit_naturalness_manifest,
    match_synthetic_entities,
    score_screening,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "data" / "fa" / "schemas"


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


def test_matching_enforces_token_and_surface_constraints(fake_tokenizer):
    match = match_synthetic_entities([candidate(name="Old Vale")], synthetic_pool(), fake_tokenizer)[0]
    assert match.real_token_count == match.synthetic_token_count
    assert match.real_word_count == match.synthetic_word_count
    assert match.capitalization_pattern_equal
    assert match.character_length_delta == 0
    assert match.synthetic_name == "New Vale"


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


def rating(pair_id, rater_id, real=4, synthetic=4, malformed=False, round=1, disagreement_registered=False):
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
    "schema_name, required",
    [
        ("candidate_entity.schema.json", {"schema_version", "qid", "source_provenance", "split"}),
        ("screening_question.schema.json", {"schema_version", "question_id", "accepted_aliases"}),
        ("synthetic_match.schema.json", {"schema_version", "pair_id", "tokenizer_revision", "character_tolerance"}),
        ("naturalness_rating.schema.json", {"schema_version", "pair_id", "rater_id", "synthetic_malformed"}),
    ],
)
def test_contract_schemas_are_strict_and_cover_provenance(schema_name, required):
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert required <= set(schema["required"])
