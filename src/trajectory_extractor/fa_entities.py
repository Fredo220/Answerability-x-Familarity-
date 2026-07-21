"""Immutable entity-screening, matching, and naturalness-audit contracts."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from types import MappingProxyType
from typing import Any

from trajectory_extractor.fa_config import CONFIRMATORY_SPLIT_COUNTS, NON_CONFIRMATORY_NAMESPACES


CHARACTER_TOLERANCE = 2
REGISTERED_SPLITS = frozenset(CONFIRMATORY_SPLIT_COUNTS) | NON_CONFIRMATORY_NAMESPACES
TOKENIZER_SENTENCE_FRAME = "In the Alder Registry, {name} has archive color amber."
_QID = re.compile(r"Q[1-9][0-9]*\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class CandidateEntity:
    """A checked-in real entity with its three exact screening answer sets."""

    entity_id: str
    qid: str
    name: str
    coarse_type: str
    split: str
    source_query: str
    source_provenance: str
    screening_aliases: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]

    def __post_init__(self) -> None:
        _safe_id(self.entity_id, "entity_id")
        if not isinstance(self.qid, str) or not _QID.fullmatch(self.qid):
            raise ValueError("QID must be a canonical Wikidata QID")
        _nonempty_text(self.name, "name")
        _nonempty_text(self.coarse_type, "coarse_type")
        _split(self.split)
        _nonempty_text(self.source_query, "source_query")
        _nonempty_text(self.source_provenance, "source provenance")
        alias_sets = tuple(tuple(aliases) for aliases in self.screening_aliases)
        if len(alias_sets) != 3:
            raise ValueError("screening_aliases must contain exactly three alias sets")
        for aliases in alias_sets:
            if not aliases or any(not isinstance(alias, str) or not _normal_form(alias) for alias in aliases):
                raise ValueError("screening aliases must be nonempty strings")
            if len({_normal_form(alias) for alias in aliases}) != len(aliases):
                raise ValueError("screening aliases must not contain duplicates")
        object.__setattr__(self, "screening_aliases", alias_sets)


@dataclass(frozen=True)
class ScreeningResult:
    entity_id: str
    qid: str
    completions: tuple[str, str, str]
    correct_answers: tuple[bool, bool, bool]
    recall_score: float
    qualifies: bool


@dataclass(frozen=True)
class SyntheticCandidate:
    """A deterministic pseudonym candidate reserved for exactly one split."""

    candidate_id: str
    name: str
    coarse_type: str
    split: str
    generator_revision: str

    def __post_init__(self) -> None:
        _safe_id(self.candidate_id, "candidate_id")
        _nonempty_text(self.name, "name")
        _nonempty_text(self.coarse_type, "coarse_type")
        _split(self.split)
        _nonempty_text(self.generator_revision, "generator_revision")


@dataclass(frozen=True)
class EntityMatch:
    pair_id: str
    real_entity_id: str
    real_qid: str
    synthetic_candidate_id: str
    real_name: str
    synthetic_name: str
    coarse_type: str
    split: str
    generator_revision: str
    tokenizer_revision: str
    real_token_count: int
    synthetic_token_count: int
    real_word_count: int
    synthetic_word_count: int
    real_character_count: int
    synthetic_character_count: int
    character_length_delta: int
    character_tolerance: int
    capitalization_pattern_equal: bool


@dataclass(frozen=True)
class NaturalnessRating:
    pair_id: str
    rater_id: str
    real_naturalness: int
    synthetic_naturalness: int
    real_type_fit: int
    synthetic_type_fit: int
    synthetic_malformed: bool
    round: int = 1
    disagreement_registered: bool = False

    def __post_init__(self) -> None:
        _safe_id(self.pair_id, "pair_id")
        _safe_id(self.rater_id, "rater_id")
        for name in ("real_naturalness", "synthetic_naturalness", "real_type_fit", "synthetic_type_fit"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 5:
                raise ValueError(f"{name} must be an integer from 1 to 5")
        if type(self.synthetic_malformed) is not bool:
            raise ValueError("synthetic_malformed must be boolean")
        if type(self.round) is not int or self.round not in {1, 2}:
            raise ValueError("round must be 1 or 2")
        if type(self.disagreement_registered) is not bool:
            raise ValueError("disagreement_registered must be boolean")


@dataclass(frozen=True)
class NaturalnessAudit:
    accepted_pair_ids: tuple[str, ...]
    excluded_pair_ids: tuple[str, ...]
    third_rater_pair_ids: tuple[str, ...]
    decisions: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_pair_ids", tuple(self.accepted_pair_ids))
        object.__setattr__(self, "excluded_pair_ids", tuple(self.excluded_pair_ids))
        object.__setattr__(self, "third_rater_pair_ids", tuple(self.third_rater_pair_ids))
        object.__setattr__(self, "decisions", MappingProxyType(dict(self.decisions)))


def score_screening(candidate: CandidateEntity, completions: Sequence[str]) -> ScreeningResult:
    """Score three forced-answer completions by exact normalized alias membership."""
    if not isinstance(candidate, CandidateEntity):
        raise TypeError("candidate must be a CandidateEntity")
    values = tuple(completions)
    if len(values) != 3 or any(not isinstance(value, str) for value in values):
        raise ValueError("screening requires exactly three completions")
    correct = tuple(
        _normal_form(completion) in {_normal_form(alias) for alias in aliases}
        for completion, aliases in zip(values, candidate.screening_aliases, strict=True)
    )
    recall_score = sum(correct) / 3
    return ScreeningResult(
        entity_id=candidate.entity_id,
        qid=candidate.qid,
        completions=values,
        correct_answers=correct,
        recall_score=recall_score,
        qualifies=sum(correct) >= 2,
    )


def match_synthetic_entities(
    real_entities: Sequence[CandidateEntity],
    synthetic_candidates: Sequence[SyntheticCandidate],
    tokenizer: Any,
) -> tuple[EntityMatch, ...]:
    """Build a deterministic, split-isolated, one-to-one eligible matching."""
    reals = tuple(real_entities)
    synthetics = tuple(synthetic_candidates)
    if not reals:
        return ()
    if any(not isinstance(entity, CandidateEntity) for entity in reals):
        raise TypeError("real_entities must contain CandidateEntity records")
    if any(not isinstance(candidate, SyntheticCandidate) for candidate in synthetics):
        raise TypeError("synthetic_candidates must contain SyntheticCandidate records")
    _reject_duplicates(reals, lambda entity: entity.entity_id, "real entity IDs")
    _reject_duplicates(reals, lambda entity: entity.qid, "real entity QIDs")
    _reject_duplicates(reals, lambda entity: _normal_form(entity.name), "real entity names")
    _reject_duplicates(synthetics, lambda candidate: candidate.candidate_id, "synthetic candidate IDs")
    _reject_duplicates(synthetics, lambda candidate: _normal_form(candidate.name), "synthetic candidate names")

    real_names = {_normal_form(entity.name) for entity in reals}
    if any(_normal_form(candidate.name) in real_names for candidate in synthetics):
        raise ValueError("synthetic candidate name collides with a real entity name")

    ordered_reals = tuple(sorted(reals, key=lambda entity: (entity.entity_id, entity.qid)))
    ordered_synthetics = tuple(sorted(synthetics, key=lambda candidate: candidate.candidate_id))
    eligible: dict[str, tuple[SyntheticCandidate, ...]] = {}
    for entity in ordered_reals:
        same_split = tuple(candidate for candidate in ordered_synthetics if candidate.split == entity.split)
        compatible_elsewhere = any(
            candidate.split != entity.split and _surface_compatible(entity, candidate, tokenizer)
            for candidate in ordered_synthetics
        )
        choices = tuple(candidate for candidate in same_split if _surface_compatible(entity, candidate, tokenizer))
        if not choices:
            if compatible_elsewhere:
                raise ValueError("reserve candidate crosses split boundary")
            raise ValueError(f"no eligible synthetic candidate for {entity.entity_id}")
        eligible[entity.entity_id] = choices

    if len(ordered_synthetics) < len(ordered_reals):
        raise ValueError("synthetic candidate reuse would be required")
    assignments = _deterministic_assignment(ordered_reals, eligible)
    if len(assignments) != len(ordered_reals):
        raise ValueError("synthetic candidate reuse would be required")

    return tuple(
        _make_match(entity, assignments[entity.entity_id], tokenizer)
        for entity in ordered_reals
    )


def audit_naturalness_manifest(
    matches: Sequence[EntityMatch], ratings: Sequence[NaturalnessRating]
) -> NaturalnessAudit:
    """Require independent blinded ratings and registered third-rater adjudication."""
    match_rows = tuple(matches)
    if any(not isinstance(match, EntityMatch) for match in match_rows):
        raise TypeError("matches must contain EntityMatch records")
    _reject_duplicates(match_rows, lambda match: match.pair_id, "match pair IDs")
    match_ids = {match.pair_id for match in match_rows}
    grouped: dict[str, list[NaturalnessRating]] = defaultdict(list)
    for rating in ratings:
        if not isinstance(rating, NaturalnessRating):
            raise TypeError("ratings must contain NaturalnessRating records")
        if rating.pair_id not in match_ids:
            raise ValueError("naturalness rating references an unknown pair")
        grouped[rating.pair_id].append(rating)

    accepted: list[str] = []
    excluded: list[str] = []
    adjudicated: list[str] = []
    decisions: dict[str, str] = {}
    for match in sorted(match_rows, key=lambda value: value.pair_id):
        pair_ratings = grouped.get(match.pair_id, [])
        initial = sorted((rating for rating in pair_ratings if rating.round == 1), key=lambda rating: rating.rater_id)
        third = [rating for rating in pair_ratings if rating.round == 2]
        if len(initial) != 2 or len({rating.rater_id for rating in initial}) != 2:
            raise ValueError("naturalness audit requires two independent initial raters")
        if len(third) > 1 or len(pair_ratings) != len(initial) + len(third):
            raise ValueError("naturalness audit has an invalid rater count")
        initial_verdicts = [_rating_passes(rating) for rating in initial]
        disagreement = initial_verdicts[0] != initial_verdicts[1]
        if third:
            adjudicator = third[0]
            if not disagreement:
                raise ValueError("third rater is allowed only for a registered disagreement")
            if not adjudicator.disagreement_registered:
                raise ValueError("third rater disagreement must be registered")
            if adjudicator.rater_id in {rating.rater_id for rating in initial}:
                raise ValueError("third rater must be independent")
            used_ratings = (*initial, adjudicator)
            adjudicated.append(match.pair_id)
        else:
            if disagreement:
                raise ValueError("registered disagreement requires a third rater")
            used_ratings = tuple(initial)

        malformed = any(rating.synthetic_malformed for rating in used_ratings)
        gap = abs(median(rating.real_naturalness for rating in used_ratings) - median(rating.synthetic_naturalness for rating in used_ratings))
        if malformed:
            excluded.append(match.pair_id)
            decisions[match.pair_id] = "excluded_malformed"
        elif gap > 1:
            excluded.append(match.pair_id)
            decisions[match.pair_id] = "excluded_naturalness_gap"
        else:
            accepted.append(match.pair_id)
            decisions[match.pair_id] = "accepted"
    return NaturalnessAudit(tuple(accepted), tuple(excluded), tuple(adjudicated), decisions)


def _deterministic_assignment(
    reals: Sequence[CandidateEntity], eligible: Mapping[str, Sequence[SyntheticCandidate]]
) -> dict[str, SyntheticCandidate]:
    assigned: dict[str, str] = {}
    candidate_lookup = {
        candidate.candidate_id: candidate for values in eligible.values() for candidate in values
    }

    def assign(entity: CandidateEntity, seen: set[str]) -> bool:
        for candidate in eligible[entity.entity_id]:
            candidate_id = candidate.candidate_id
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            incumbent_id = assigned.get(candidate_id)
            if incumbent_id is None or assign(next(item for item in reals if item.entity_id == incumbent_id), seen):
                assigned[candidate_id] = entity.entity_id
                return True
        return False

    for entity in reals:
        if not assign(entity, set()):
            break
    return {entity_id: candidate_lookup[candidate_id] for candidate_id, entity_id in assigned.items()}


def _make_match(entity: CandidateEntity, synthetic: SyntheticCandidate, tokenizer: Any) -> EntityMatch:
    real_characters = len(entity.name)
    synthetic_characters = len(synthetic.name)
    return EntityMatch(
        pair_id=f"{entity.entity_id}--{synthetic.candidate_id}",
        real_entity_id=entity.entity_id,
        real_qid=entity.qid,
        synthetic_candidate_id=synthetic.candidate_id,
        real_name=entity.name,
        synthetic_name=synthetic.name,
        coarse_type=entity.coarse_type,
        split=entity.split,
        generator_revision=synthetic.generator_revision,
        tokenizer_revision=_tokenizer_revision(tokenizer),
        real_token_count=_token_count(tokenizer, entity.name),
        synthetic_token_count=_token_count(tokenizer, synthetic.name),
        real_word_count=len(entity.name.split()),
        synthetic_word_count=len(synthetic.name.split()),
        real_character_count=real_characters,
        synthetic_character_count=synthetic_characters,
        character_length_delta=synthetic_characters - real_characters,
        character_tolerance=CHARACTER_TOLERANCE,
        capitalization_pattern_equal=_capitalization_pattern(entity.name) == _capitalization_pattern(synthetic.name),
    )


def _surface_compatible(entity: CandidateEntity, synthetic: SyntheticCandidate, tokenizer: Any) -> bool:
    return (
        entity.coarse_type == synthetic.coarse_type
        and _token_count(tokenizer, entity.name) == _token_count(tokenizer, synthetic.name)
        and len(entity.name.split()) == len(synthetic.name.split())
        and _capitalization_pattern(entity.name) == _capitalization_pattern(synthetic.name)
        and abs(len(entity.name) - len(synthetic.name)) <= CHARACTER_TOLERANCE
        and _allowed_character_inventory(entity.name, synthetic.name)
    )


def _token_count(tokenizer: Any, name: str) -> int:
    frame = TOKENIZER_SENTENCE_FRAME.format(name=name)
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(frame, add_special_tokens=False)
    elif callable(tokenizer):
        result = tokenizer(frame, add_special_tokens=False)
        tokens = result["input_ids"] if isinstance(result, Mapping) else result
    else:
        raise TypeError("tokenizer must provide encode() or be callable")
    return len(tokens)


def _capitalization_pattern(name: str) -> tuple[str, ...]:
    pattern = []
    for word in name.split():
        letters = "".join(character for character in word if character.isalpha())
        if not letters:
            pattern.append("none")
        elif letters.isupper():
            pattern.append("upper")
        elif letters.islower():
            pattern.append("lower")
        elif letters[0].isupper() and letters[1:].islower():
            pattern.append("title")
        else:
            pattern.append("mixed")
    return tuple(pattern)


def _allowed_character_inventory(real_name: str, synthetic_name: str) -> bool:
    real_non_alphanumeric = {character for character in real_name if not character.isalnum()}
    synthetic_non_alphanumeric = {character for character in synthetic_name if not character.isalnum()}
    real_non_ascii = {character for character in real_name if not character.isascii()}
    synthetic_non_ascii = {character for character in synthetic_name if not character.isascii()}
    return synthetic_non_alphanumeric <= real_non_alphanumeric and synthetic_non_ascii <= real_non_ascii


def _tokenizer_revision(tokenizer: Any) -> str:
    for attribute in ("revision", "name_or_path"):
        value = getattr(tokenizer, attribute, None)
        if isinstance(value, str) and value:
            return value
    return "unregistered-test-tokenizer"


def _rating_passes(rating: NaturalnessRating) -> bool:
    return not rating.synthetic_malformed and abs(rating.real_naturalness - rating.synthetic_naturalness) <= 1


def _normal_form(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _nonempty_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _safe_id(value: object, field: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} must be a safe identifier")


def _split(value: object) -> None:
    if not isinstance(value, str) or value not in REGISTERED_SPLITS:
        raise ValueError("split must be a registered split")


def _reject_duplicates(values: Iterable[Any], key, label: str) -> None:
    seen = set()
    for value in values:
        item = key(value)
        if item in seen:
            raise ValueError(f"duplicate {label} are not allowed")
        seen.add(item)
