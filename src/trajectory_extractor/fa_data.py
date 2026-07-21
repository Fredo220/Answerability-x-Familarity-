"""Deterministic construction and audit contracts for Familiarity-vs-Answerability.

This module deliberately constructs prompts from registered templates rather than a
generic repetition count.  It contains no model or outcome code: its only job is
to make the preregistered design content-addressed and auditable before generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_entities import EntityMatch


TRAIN_TEMPLATE_FAMILIES = ("train_registry_direct", "train_registry_possessive", "train_registry_query")
VALIDATION_TEMPLATE_FAMILIES = (
    "validation_archive_direct",
    "validation_archive_possessive",
    "validation_archive_query",
)
TEST_TEMPLATE_FAMILIES = (
    "test_catalog_direct",
    "test_catalog_inverse",
    "test_ledger_direct",
    "test_ledger_query",
)
CONFIRMATORY_POWER_SIMULATIONS = 2000
CONFIRMATORY_POWER_SEED = 20260722
_CODE = re.compile(r"K[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{4}\Z")
_FAMILIES_BY_SPLIT = {
    "mechanism_train": TRAIN_TEMPLATE_FAMILIES,
    "locked_validation": VALIDATION_TEMPLATE_FAMILIES,
    "behavior_test": TEST_TEMPLATE_FAMILIES,
    "probe_test": TEST_TEMPLATE_FAMILIES,
    "intervention_test": TEST_TEMPLATE_FAMILIES,
    "pilot": TRAIN_TEMPLATE_FAMILIES,
    "circuit_dev": TEST_TEMPLATE_FAMILIES,
}
_TEMPLATE_TEXT = {
    "train_registry_direct": (
        "In the Alder Registry, {first} has archive {first_relation} {first_value}. "
        "{second} has archive {second_relation} {second_value}. "
        "What is {query}'s archive code? Answer with the code, or UNKNOWN if it is not stated."
    ),
    "train_registry_possessive": (
        "The Alder Registry records {first}'s archive {first_relation} as {first_value} and "
        "{second}'s archive {second_relation} as {second_value}. "
        "Give {query}'s archive code, or UNKNOWN if it is not stated."
    ),
    "train_registry_query": (
        "Alder Registry entry: {first}, archive {first_relation}, {first_value}; "
        "{second}, archive {second_relation}, {second_value}. "
        "Return {query}'s archive code or UNKNOWN if it is not stated."
    ),
    "validation_archive_direct": (
        "The Briar Archive says {first} holds {first_relation} {first_value}. "
        "It says {second} holds {second_relation} {second_value}. "
        "What is {query}'s archive code? Reply with the code or UNKNOWN if it is not stated."
    ),
    "validation_archive_possessive": (
        "In the Briar Archive, {first}'s {first_relation} is {first_value}; "
        "{second}'s {second_relation} is {second_value}. "
        "Reply with {query}'s archive code, or UNKNOWN if it is not stated."
    ),
    "validation_archive_query": (
        "Briar Archive record: {first} / {first_relation} / {first_value}; "
        "{second} / {second_relation} / {second_value}. "
        "State {query}'s archive code or UNKNOWN if it is not stated."
    ),
    "test_catalog_direct": (
        "The Cinder Catalog lists {first}: {first_relation} {first_value}; "
        "{second}: {second_relation} {second_value}. "
        "What is the archive code for {query}? Answer with the code or UNKNOWN if it is not stated."
    ),
    "test_catalog_inverse": (
        "Catalog note for {first} gives {first_value} as its archive {first_relation}; "
        "the note for {second} gives {second_value} as its archive {second_relation}. "
        "Give the archive code for {query}, or UNKNOWN if it is not stated."
    ),
    "test_ledger_direct": (
        "Cinder Ledger: {first} has {first_relation} {first_value}; "
        "{second} has {second_relation} {second_value}. "
        "Return the archive code for {query}, or UNKNOWN if it is not stated."
    ),
    "test_ledger_query": (
        "From the Cinder Ledger, {first}'s archive {first_relation} is {first_value}, while "
        "{second}'s archive {second_relation} is {second_value}. "
        "Which archive code belongs to {query}? Use UNKNOWN if it is not stated."
    ),
}


@dataclass(frozen=True)
class RegisteredPowerGrid:
    absent_attempt_rates: tuple[float, ...]
    entity_iccs: tuple[float, ...]
    template_iccs: tuple[float, ...]
    invalid_format_rates: tuple[float, ...]
    interactions: tuple[float, ...]


REGISTERED_POWER_GRID = RegisteredPowerGrid(
    absent_attempt_rates=(0.10, 0.25, 0.50),
    entity_iccs=(0.05, 0.15, 0.30),
    template_iccs=(0.02, 0.10),
    invalid_format_rates=(0.00, 0.05),
    interactions=(0.00, 0.025, 0.05, 0.075, 0.10),
)


@dataclass(frozen=True)
class FAExample:
    example_id: str
    entity_unit_id: str
    split: str
    template_family: str
    target_familiarity: str
    distractor_familiarity: str
    answerability: str
    target_text: str
    distractor_text: str
    registry_code: str
    expected_output: str
    user_text: str
    canonical_payload_sha256: str
    target_entity_id: str
    distractor_entity_id: str
    entity_order: str
    query_role: str
    relation_order: str
    code_position: str
    rendered_token_count: int
    rendered_token_ids: tuple[Any, ...]
    special_token_sequence: tuple[Any, ...]
    block: str = "factorial"
    exposure: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rendered_token_ids", tuple(self.rendered_token_ids))
        object.__setattr__(self, "special_token_sequence", tuple(self.special_token_sequence))
        _require_text(self.example_id, "example_id")
        _require_text(self.entity_unit_id, "entity_unit_id")
        _require_text(self.target_entity_id, "target_entity_id")
        _require_text(self.distractor_entity_id, "distractor_entity_id")
        if self.split not in _FAMILIES_BY_SPLIT:
            raise ValueError("split must have registered template families")
        if self.template_family not in _FAMILIES_BY_SPLIT[self.split]:
            raise ValueError("template_family is not registered for split")
        if self.target_familiarity not in {"screened_real", "matched_synthetic"}:
            raise ValueError("target_familiarity is invalid")
        if self.distractor_familiarity not in {"screened_real", "matched_synthetic"}:
            raise ValueError("distractor_familiarity is invalid")
        if self.answerability not in {"target_bound", "distractor_bound", "code_absent"}:
            raise ValueError("answerability is invalid")
        if self.block not in {"factorial", "same_string"}:
            raise ValueError("block is invalid")
        if self.block == "same_string" and self.exposure not in {"high_exposure", "low_exposure"}:
            raise ValueError("same_string rows require registered exposure")
        if self.block == "factorial" and self.exposure is not None:
            raise ValueError("factorial rows cannot have exposure")
        for field in ("target_text", "distractor_text", "expected_output", "user_text"):
            _require_text(getattr(self, field), field)
        if not _CODE.fullmatch(self.registry_code):
            raise ValueError("registry_code must use the registered code vocabulary")
        if self.expected_output != (self.registry_code if self.answerability == "target_bound" else "UNKNOWN"):
            raise ValueError("expected_output must follow answerability")
        if self.entity_order not in {"target_first", "target_second"}:
            raise ValueError("entity_order is invalid")
        if self.query_role not in {"first", "second"}:
            raise ValueError("query_role is invalid")
        if self.query_role != ("first" if self.entity_order == "target_first" else "second"):
            raise ValueError("query_role must agree with entity_order")
        if self.relation_order not in {"code_first", "code_second", "code_absent"}:
            raise ValueError("relation_order is invalid")
        if self.code_position not in {"first", "second", "absent"}:
            raise ValueError("code_position is invalid")
        if type(self.rendered_token_count) is not int or self.rendered_token_count < 1:
            raise ValueError("rendered_token_count must be positive")
        if self.rendered_token_count != len(self.rendered_token_ids):
            raise ValueError("rendered_token_count must match rendered_token_ids")
        canonical_sha256 = _example_sha256(self)
        if self.canonical_payload_sha256 != canonical_sha256:
            raise ValueError("canonical_payload_sha256 must derive from canonical content")
        if self.example_id != canonical_sha256:
            raise ValueError("example_id must derive from canonical content")


@dataclass(frozen=True)
class PowerCell:
    absent_attempt_rate: float
    entity_icc: float
    template_icc: float
    invalid_format_rate: float
    interaction: float
    estimated_power: float
    monte_carlo_standard_error: float
    simulations: int


@dataclass(frozen=True)
class PowerAudit:
    design_sha256: str
    seed: int
    simulations: int
    cells: tuple[PowerCell, ...]
    registered_grid: bool

    @property
    def five_point_power_passes(self) -> bool:
        relevant = [cell for cell in self.cells if math.isclose(cell.interaction, 0.05)]
        return bool(relevant) and all(cell.estimated_power >= 0.80 for cell in relevant)


@dataclass(frozen=True)
class DatasetAudit:
    checks: Mapping[str, bool]
    violations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))
        object.__setattr__(self, "violations", tuple(self.violations))

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


@dataclass(frozen=True)
class FAManifest:
    config_hash: str
    examples: tuple[FAExample, ...]
    manifest_sha256: str
    power_audit: PowerAudit | None = None

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.examples, key=lambda row: row.example_id))
        object.__setattr__(self, "examples", ordered)
        if len({row.example_id for row in ordered}) != len(ordered):
            raise ValueError("manifest cannot contain duplicate example IDs")
        expected = _manifest_sha256(self.config_hash, ordered)
        if self.manifest_sha256 != expected:
            raise ValueError("manifest_sha256 must derive from canonical content")


def build_factorial_examples(
    config: FAConfig,
    matches: Sequence[EntityMatch],
    *,
    tokenizer: Any | None = None,
) -> tuple[FAExample, ...]:
    """Build the 2 x 2 x 3 core rows over every split-specific family."""
    prepared = _validate_matches(config, matches)
    code_by_unit = _allocate_codes(prepared)
    distractor_by_unit = _assign_distractor_units(prepared, config.split_seed)
    rows: list[FAExample] = []
    for match in prepared:
        distractor = distractor_by_unit[match.pair_id]
        for family in _FAMILIES_BY_SPLIT[match.split]:
            latin_offset = _hash_int(config.split_seed, "latin-square", match.pair_id, family) % 2
            for target_familiarity in ("screened_real", "matched_synthetic"):
                for distractor_familiarity in ("screened_real", "matched_synthetic"):
                    target_first = (
                        latin_offset
                        + (target_familiarity == "matched_synthetic")
                        + (distractor_familiarity == "matched_synthetic")
                    ) % 2 == 0
                    for answerability in ("target_bound", "distractor_bound", "code_absent"):
                        rows.append(
                            _build_core_row(
                                match,
                                distractor,
                                family,
                                target_familiarity,
                                distractor_familiarity,
                                answerability,
                                target_first,
                                code_by_unit[match.pair_id],
                                tokenizer,
                            )
                        )
    return tuple(sorted(rows, key=lambda row: row.example_id))


def build_same_string_examples(
    config: FAConfig,
    matches: Sequence[EntityMatch],
    *,
    tokenizer: Any | None = None,
) -> tuple[FAExample, ...]:
    """Build the sealed four-row contextual-familiarization block per unit."""
    prepared = _validate_matches(config, matches)
    code_by_unit = _allocate_codes(prepared)
    family_by_unit = _balanced_same_string_families(prepared, config.split_seed)
    rows: list[FAExample] = []
    for match in prepared:
        family = family_by_unit[match.pair_id]
        for exposure in ("high_exposure", "low_exposure"):
            for answerability in ("target_bound", "code_absent"):
                rows.append(
                    _build_same_string_row(
                        match, family, exposure, answerability, code_by_unit[match.pair_id], tokenizer
                    )
                )
    return tuple(sorted(rows, key=lambda row: row.example_id))


def build_manifest(
    config: FAConfig,
    examples: Sequence[FAExample],
    *,
    power_audit: PowerAudit | None = None,
) -> FAManifest:
    """Seal deterministic rows and fail closed for a full confirmatory design."""
    rows = tuple(sorted(examples, key=lambda row: row.example_id))
    if _is_full_confirmatory_design(config, rows):
        if power_audit is None:
            raise ValueError("confirmatory manifest requires a registered power audit")
        if (
            not power_audit.registered_grid
            or power_audit.seed != CONFIRMATORY_POWER_SEED
            or power_audit.simulations != CONFIRMATORY_POWER_SIMULATIONS
            or power_audit.design_sha256 != _design_sha256(rows)
            or not power_audit.five_point_power_passes
        ):
            raise ValueError("confirmatory manifest requires a passing registered power audit")
    return FAManifest(
        config_hash=config.config_hash,
        examples=rows,
        manifest_sha256=_manifest_sha256(config.config_hash, rows),
        power_audit=power_audit,
    )


def audit_dataset(
    rows: Sequence[FAExample],
    same_string_rows: Sequence[FAExample] = (),
    *,
    tokenizer: Any | None = None,
) -> DatasetAudit:
    """Audit every registered construction control without inspecting outcomes."""
    core = tuple(rows)
    replication = tuple(same_string_rows)
    checks = {
        "independent_target_distractor_variation": _check_factorial_balance(core),
        "entity_order": _check_counterbalance(core, "entity_order", {"target_first", "target_second"}),
        "query_role": _check_counterbalance(core, "query_role", {"first", "second"}),
        "relation_order": _check_relation_order(core),
        "code_position": _check_code_position(core),
        "code_vocabulary": _check_code_vocabulary(core + replication),
        "template_overlap": _check_template_isolation(core + replication),
        "entity_overlap": _check_entity_isolation(core + replication),
        "rendered_token_length": _check_rendered_tokens(core + replication, tokenizer),
        "special_token_sequence": _check_special_tokens(core + replication, tokenizer),
        "lexical_multiset": _check_lexical_multisets(core, tokenizer),
        "same_string_token_budget": _check_same_string_budget(replication, tokenizer),
    }
    violations = tuple(name for name, passed in checks.items() if not passed)
    return DatasetAudit(checks=checks, violations=violations)


def simulate_interaction_power(
    design: Sequence[FAExample],
    effect_grid: Sequence[float] | None = None,
    within_entity_correlations: Mapping[str, Sequence[float]] | None = None,
    seed: int = CONFIRMATORY_POWER_SEED,
    *,
    simulations: int = CONFIRMATORY_POWER_SIMULATIONS,
) -> PowerAudit:
    """Run the preregistered conservative crossed-cluster interaction simulation.

    The simulation samples interaction estimates from a normal approximation whose
    variance is inflated by both entity and template design effects.  It is
    conservative for the balanced binary design and keeps the 360 registered grid
    cells fast enough for local, pre-outcome sealing.
    """
    rows = tuple(design)
    if not rows or any(row.block != "factorial" for row in rows):
        raise ValueError("power design must contain factorial rows")
    if type(simulations) is not int or simulations < 1:
        raise ValueError("simulations must be positive")
    interactions = tuple(REGISTERED_POWER_GRID.interactions if effect_grid is None else effect_grid)
    correlations = within_entity_correlations or {
        "entity_icc": REGISTERED_POWER_GRID.entity_iccs,
        "template_icc": REGISTERED_POWER_GRID.template_iccs,
        "invalid_format_rate": REGISTERED_POWER_GRID.invalid_format_rates,
    }
    entity_iccs = tuple(correlations.get("entity_icc", ()))
    template_iccs = tuple(correlations.get("template_icc", ()))
    invalid_rates = tuple(correlations.get("invalid_format_rate", ()))
    values = (
        *REGISTERED_POWER_GRID.absent_attempt_rates,
        *entity_iccs,
        *template_iccs,
        *invalid_rates,
        *interactions,
    )
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("power grid values must be finite")
    if any(not 0.0 <= value <= 1.0 for value in (*REGISTERED_POWER_GRID.absent_attempt_rates, *entity_iccs, *template_iccs, *invalid_rates)):
        raise ValueError("power rates and ICCs must be in [0, 1]")
    if any(value < 0.0 or value > 1.0 for value in interactions):
        raise ValueError("interaction effects must be in [0, 1]")

    absent_rows = sum(row.answerability != "target_bound" for row in rows)
    entity_count = len({row.entity_unit_id for row in rows})
    template_count = len({row.template_family for row in rows})
    if not absent_rows or not entity_count or not template_count:
        raise ValueError("power design must span entities, templates, and absent rows")
    cells: list[PowerCell] = []
    for absent_rate in REGISTERED_POWER_GRID.absent_attempt_rates:
        for entity_icc in entity_iccs:
            for template_icc in template_iccs:
                for invalid_rate in invalid_rates:
                    for interaction in interactions:
                        standard_error = _conservative_interaction_se(
                            absent_rows,
                            entity_count,
                            template_count,
                            absent_rate,
                            entity_icc,
                            template_icc,
                            invalid_rate,
                        )
                        cell_seed = _hash_int(
                            seed, absent_rate, entity_icc, template_icc, invalid_rate, interaction
                        )
                        rng = random.Random(cell_seed)
                        successes = sum(
                            rng.gauss(interaction, standard_error) > 1.959963984540054 * standard_error
                            for _ in range(simulations)
                        )
                        estimated_power = successes / simulations
                        cells.append(
                            PowerCell(
                                absent_attempt_rate=float(absent_rate),
                                entity_icc=float(entity_icc),
                                template_icc=float(template_icc),
                                invalid_format_rate=float(invalid_rate),
                                interaction=float(interaction),
                                estimated_power=estimated_power,
                                monte_carlo_standard_error=math.sqrt(
                                    estimated_power * (1.0 - estimated_power) / simulations
                                ),
                                simulations=simulations,
                            )
                        )
    registered = (
        tuple(interactions) == REGISTERED_POWER_GRID.interactions
        and tuple(entity_iccs) == REGISTERED_POWER_GRID.entity_iccs
        and tuple(template_iccs) == REGISTERED_POWER_GRID.template_iccs
        and tuple(invalid_rates) == REGISTERED_POWER_GRID.invalid_format_rates
        and seed == CONFIRMATORY_POWER_SEED
        and simulations == CONFIRMATORY_POWER_SIMULATIONS
    )
    return PowerAudit(
        design_sha256=_design_sha256(rows),
        seed=seed,
        simulations=simulations,
        cells=tuple(cells),
        registered_grid=registered,
    )


def _build_core_row(
    target_match: EntityMatch,
    distractor_match: EntityMatch,
    family: str,
    target_familiarity: str,
    distractor_familiarity: str,
    answerability: str,
    target_first: bool,
    code: str,
    tokenizer: Any | None,
) -> FAExample:
    target = _entity_text(target_match, target_familiarity)
    distractor = _entity_text(distractor_match, distractor_familiarity)
    target_id = _entity_id(target_match, target_familiarity)
    distractor_id = _entity_id(distractor_match, distractor_familiarity)
    bindings = {
        "target_bound": (("code", code), ("color", "amber")),
        "distractor_bound": (("color", "amber"), ("code", code)),
        "code_absent": (("color", "amber"), ("shape", "oval")),
    }
    (target_relation, target_value), (distractor_relation, distractor_value) = bindings[answerability]
    if target_first:
        first, second = target, distractor
        first_relation, first_value = target_relation, target_value
        second_relation, second_value = distractor_relation, distractor_value
        entity_order, query_role = "target_first", "first"
    else:
        first, second = distractor, target
        first_relation, first_value = distractor_relation, distractor_value
        second_relation, second_value = target_relation, target_value
        entity_order, query_role = "target_second", "second"
    user_text = _normalize_prompt_punctuation(_TEMPLATE_TEXT[family].format(
        first=first,
        second=second,
        first_relation=first_relation,
        first_value=first_value,
        second_relation=second_relation,
        second_value=second_value,
        query=target,
    ))
    if answerability == "code_absent":
        relation_order, code_position = "code_absent", "absent"
    else:
        code_position = "first" if first_value == code else "second"
        relation_order = f"code_{code_position}"
    return _example(
        entity_unit_id=target_match.pair_id,
        split=target_match.split,
        template_family=family,
        target_familiarity=target_familiarity,
        distractor_familiarity=distractor_familiarity,
        answerability=answerability,
        target_text=target,
        distractor_text=distractor,
        registry_code=code,
        expected_output=code if answerability == "target_bound" else "UNKNOWN",
        user_text=user_text,
        target_entity_id=target_id,
        distractor_entity_id=distractor_id,
        entity_order=entity_order,
        query_role=query_role,
        relation_order=relation_order,
        code_position=code_position,
        tokenizer=tokenizer,
    )


def _build_same_string_row(
    match: EntityMatch,
    family: str,
    exposure: str,
    answerability: str,
    code: str,
    tokenizer: Any | None,
) -> FAExample:
    subject = match.synthetic_name if exposure == "high_exposure" else match.real_name
    facts = (
        ("archive color", "amber"),
        ("catalog shape", "oval"),
        ("ledger season", "spring"),
        ("registry material", "linen"),
    )
    prefix = " ".join(f"{subject} has {relation} {value}." for relation, value in facts)
    target_binding = f"{match.synthetic_name} has archive code {code}." if answerability == "target_bound" else f"{match.synthetic_name} has archive color amber."
    user_text = (
        f"{prefix} In the Alder Registry, {target_binding} "
        f"What is {match.synthetic_name}'s archive code? Answer with the code, or UNKNOWN if it is not stated."
    )
    return _example(
        entity_unit_id=match.pair_id,
        split=match.split,
        template_family=family,
        target_familiarity="matched_synthetic",
        distractor_familiarity="matched_synthetic",
        answerability=answerability,
        target_text=match.synthetic_name,
        distractor_text=match.real_name,
        registry_code=code,
        expected_output=code if answerability == "target_bound" else "UNKNOWN",
        user_text=user_text,
        target_entity_id=match.synthetic_candidate_id,
        distractor_entity_id=match.real_entity_id,
        entity_order="target_second",
        query_role="second",
        relation_order="code_second" if answerability == "target_bound" else "code_absent",
        code_position="second" if answerability == "target_bound" else "absent",
        tokenizer=tokenizer,
        block="same_string",
        exposure=exposure,
    )


def _example(*, tokenizer: Any | None, block: str = "factorial", exposure: str | None = None, **kwargs: Any) -> FAExample:
    token_ids, special_tokens = _token_metadata(kwargs["user_text"], tokenizer)
    payload = {
        **kwargs,
        "rendered_token_count": len(token_ids),
        "rendered_token_ids": token_ids,
        "special_token_sequence": special_tokens,
        "block": block,
        "exposure": exposure,
    }
    digest = _payload_sha256(payload)
    return FAExample(
        example_id=digest,
        canonical_payload_sha256=digest,
        **payload,
    )


def _validate_matches(config: FAConfig, matches: Sequence[EntityMatch]) -> tuple[EntityMatch, ...]:
    prepared = tuple(matches)
    if not prepared or any(not isinstance(match, EntityMatch) for match in prepared):
        raise ValueError("matches must contain audited EntityMatch records")
    if len({match.pair_id for match in prepared}) != len(prepared):
        raise ValueError("matches must have unique pair IDs")
    if any(match.split not in _FAMILIES_BY_SPLIT for match in prepared):
        raise ValueError("matches use an unregistered data split")
    if config.profile == "confirmatory" and any(match.split in {"pilot", "circuit_dev"} for match in prepared):
        raise ValueError("confirmatory construction cannot use non-confirmatory namespaces")
    return tuple(sorted(prepared, key=lambda match: (match.split, match.pair_id)))


def _allocate_codes(matches: Sequence[EntityMatch]) -> dict[str, str]:
    assigned: dict[str, str] = {}
    used: set[str] = set()
    for match in sorted(matches, key=lambda item: _hash_int(item.split, item.pair_id)):
        attempt = 0
        while True:
            code = _code_for(match.split, match.pair_id, attempt)
            names = f"{match.real_name} {match.synthetic_name}".casefold()
            if code not in used and code.casefold() not in names:
                assigned[match.pair_id] = code
                used.add(code)
                break
            attempt += 1
    return assigned


def _assign_distractor_units(matches: Sequence[EntityMatch], seed: int) -> dict[str, EntityMatch]:
    by_split: dict[str, list[EntityMatch]] = defaultdict(list)
    for match in matches:
        by_split[match.split].append(match)
    assigned: dict[str, EntityMatch] = {}
    for group in by_split.values():
        ordered = sorted(group, key=lambda item: _hash_int(seed, "distractor", item.pair_id))
        for index, match in enumerate(ordered):
            assigned[match.pair_id] = ordered[(index + 1) % len(ordered)]
    return assigned


def _balanced_same_string_families(matches: Sequence[EntityMatch], seed: int) -> dict[str, str]:
    grouped: dict[tuple[str, str], list[EntityMatch]] = defaultdict(list)
    for match in matches:
        grouped[(match.split, match.coarse_type)].append(match)
    assigned: dict[str, str] = {}
    for (split, _domain), group in grouped.items():
        families = _FAMILIES_BY_SPLIT[split]
        ordered = sorted(group, key=lambda item: _hash_int(seed, "same-string", item.pair_id))
        for index, match in enumerate(ordered):
            assigned[match.pair_id] = families[index % len(families)]
    return assigned


def _entity_text(match: EntityMatch, familiarity: str) -> str:
    return match.real_name if familiarity == "screened_real" else match.synthetic_name


def _entity_id(match: EntityMatch, familiarity: str) -> str:
    return match.real_entity_id if familiarity == "screened_real" else match.synthetic_candidate_id


def _token_metadata(text: str, tokenizer: Any | None) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if tokenizer is None:
        token_ids = tuple(re.findall(r"\S+", text))
        return token_ids, ()
    token_ids = tuple(_encode(tokenizer, text, add_special_tokens=True))
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    return token_ids, tuple(token for token in token_ids if token in special_ids)


def _normalize_prompt_punctuation(text: str) -> str:
    """Keep relation-binding swaps token-stable even for simple word tokenizers."""
    return re.sub(r"\s+", " ", re.sub(r"([;,:])", r" \1 ", text)).strip()


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> Sequence[Any]:
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if callable(tokenizer):
        value = tokenizer(text, add_special_tokens=add_special_tokens)
        return value["input_ids"] if isinstance(value, Mapping) else value
    raise TypeError("tokenizer must provide encode() or be callable")


def _check_factorial_balance(rows: Sequence[FAExample]) -> bool:
    groups: dict[tuple[str, str], list[FAExample]] = defaultdict(list)
    for row in rows:
        if row.block != "factorial":
            return False
        groups[(row.entity_unit_id, row.template_family)].append(row)
    expected = {
        (target, distractor, answerability)
        for target in ("screened_real", "matched_synthetic")
        for distractor in ("screened_real", "matched_synthetic")
        for answerability in ("target_bound", "distractor_bound", "code_absent")
    }
    return bool(groups) and all(
        Counter((row.target_familiarity, row.distractor_familiarity, row.answerability) for row in group)
        == Counter({cell: 1 for cell in expected})
        for group in groups.values()
    )


def _check_counterbalance(rows: Sequence[FAExample], field: str, expected: set[str]) -> bool:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        groups[(row.entity_unit_id, row.template_family)].append(getattr(row, field))
    return bool(groups) and all(set(values) == expected for values in groups.values())


def _check_relation_order(rows: Sequence[FAExample]) -> bool:
    groups: dict[tuple[str, str, str, str], list[FAExample]] = defaultdict(list)
    for row in rows:
        groups[(row.entity_unit_id, row.template_family, row.target_familiarity, row.distractor_familiarity)].append(row)
    return all(
        {row.relation_order for row in group if row.answerability != "code_absent"} == {"code_first", "code_second"}
        and {row.relation_order for row in group if row.answerability == "code_absent"} == {"code_absent"}
        for group in groups.values()
    )


def _check_code_position(rows: Sequence[FAExample]) -> bool:
    groups: dict[tuple[str, str, str, str], list[FAExample]] = defaultdict(list)
    for row in rows:
        groups[(row.entity_unit_id, row.template_family, row.target_familiarity, row.distractor_familiarity)].append(row)
    return all(
        {row.code_position for row in group} == {"first", "second", "absent"} for group in groups.values()
    )


def _check_code_vocabulary(rows: Sequence[FAExample]) -> bool:
    by_unit: dict[str, set[str]] = defaultdict(set)
    code_owner: dict[str, str] = {}
    for row in rows:
        if not _CODE.fullmatch(row.registry_code):
            return False
        by_unit[row.entity_unit_id].add(row.registry_code)
        owner = code_owner.setdefault(row.registry_code, row.entity_unit_id)
        if owner != row.entity_unit_id:
            return False
    return bool(rows) and all(len(codes) == 1 for codes in by_unit.values())


def _check_template_isolation(rows: Sequence[FAExample]) -> bool:
    return bool(rows) and all(row.template_family in _FAMILIES_BY_SPLIT.get(row.split, ()) for row in rows)


def _check_entity_isolation(rows: Sequence[FAExample]) -> bool:
    owner: dict[str, str] = {}
    for row in rows:
        entities = (
            (f"id:{row.target_entity_id}", row.target_text),
            (f"id:{row.distractor_entity_id}", row.distractor_text),
        )
        for entity_id, entity_text in entities:
            split = owner.setdefault(entity_id, row.split)
            if split != row.split:
                return False
            name_key = f"name:{entity_text.casefold().strip()}"
            split = owner.setdefault(name_key, row.split)
            if split != row.split:
                return False
    return bool(rows)


def _check_rendered_tokens(rows: Sequence[FAExample], tokenizer: Any | None) -> bool:
    for row in rows:
        token_ids, _special = _token_metadata(row.user_text, tokenizer)
        if len(token_ids) != row.rendered_token_count or tuple(token_ids) != row.rendered_token_ids:
            return False
    return bool(rows)


def _check_special_tokens(rows: Sequence[FAExample], tokenizer: Any | None) -> bool:
    for row in rows:
        _token_ids, special_tokens = _token_metadata(row.user_text, tokenizer)
        if tuple(special_tokens) != row.special_token_sequence:
            return False
    groups: dict[tuple[str, str, str, str], dict[str, FAExample]] = defaultdict(dict)
    for row in rows:
        if row.block == "factorial":
            groups[(row.entity_unit_id, row.template_family, row.target_familiarity, row.distractor_familiarity)][row.answerability] = row
    return bool(rows) and all(
        group.get("target_bound") is not None
        and group.get("distractor_bound") is not None
        and group["target_bound"].special_token_sequence == group["distractor_bound"].special_token_sequence
        for group in groups.values()
    )


def _check_lexical_multisets(rows: Sequence[FAExample], tokenizer: Any | None) -> bool:
    groups: dict[tuple[str, str, str, str], dict[str, FAExample]] = defaultdict(dict)
    for row in rows:
        groups[(row.entity_unit_id, row.template_family, row.target_familiarity, row.distractor_familiarity)][row.answerability] = row
    for group in groups.values():
        target, distractor = group.get("target_bound"), group.get("distractor_bound")
        if target is None or distractor is None:
            return False
        if target.registry_code != distractor.registry_code:
            return False
        left = Counter(_encode(tokenizer, target.user_text, add_special_tokens=False)) if tokenizer else Counter(re.findall(r"\S+", target.user_text))
        right = Counter(_encode(tokenizer, distractor.user_text, add_special_tokens=False)) if tokenizer else Counter(re.findall(r"\S+", distractor.user_text))
        if left != right or target.rendered_token_count != distractor.rendered_token_count:
            return False
        if _example_sha256(target) != target.canonical_payload_sha256 or _example_sha256(distractor) != distractor.canonical_payload_sha256:
            return False
    return bool(groups)


def _check_same_string_budget(rows: Sequence[FAExample], tokenizer: Any | None) -> bool:
    groups: dict[str, list[FAExample]] = defaultdict(list)
    for row in rows:
        if row.block != "same_string":
            return False
        groups[row.entity_unit_id].append(row)
    for group in groups.values():
        if len(group) != 4 or len({row.target_text for row in group}) != 1 or len({row.template_family for row in group}) != 1:
            return False
        combinations = {(row.exposure, row.answerability) for row in group}
        if combinations != {
            ("high_exposure", "target_bound"),
            ("low_exposure", "target_bound"),
            ("high_exposure", "code_absent"),
            ("low_exposure", "code_absent"),
        }:
            return False
        for answerability in ("target_bound", "code_absent"):
            pair = [row for row in group if row.answerability == answerability]
            if len({row.rendered_token_count for row in pair}) != 1:
                return False
            if len({row.special_token_sequence for row in pair}) != 1:
                return False
            if tokenizer is not None and any(len(_encode(tokenizer, row.user_text, add_special_tokens=True)) != row.rendered_token_count for row in pair):
                return False
    return bool(groups)


def _conservative_interaction_se(
    absent_rows: int,
    entity_count: int,
    template_count: int,
    absent_rate: float,
    entity_icc: float,
    template_icc: float,
    invalid_rate: float,
) -> float:
    effective_rows = max(1.0, absent_rows * (1.0 - invalid_rate))
    binomial_variance = 16.0 * absent_rate * (1.0 - absent_rate) / effective_rows
    entity_design_effect = 1.0 + (effective_rows / entity_count - 1.0) * entity_icc
    template_design_effect = 1.0 + (effective_rows / template_count - 1.0) * template_icc
    return math.sqrt(binomial_variance * entity_design_effect * template_design_effect)


def _is_full_confirmatory_design(config: FAConfig, rows: Sequence[FAExample]) -> bool:
    if config.profile != "confirmatory" or any(row.block != "factorial" for row in rows):
        return False
    observed = Counter(row.entity_unit_id for row in rows)
    split_by_unit = {row.entity_unit_id: row.split for row in rows}
    expected_units = Counter(config.split_counts)
    observed_units = Counter(split_by_unit.values())
    if observed_units != expected_units:
        return False
    return all(
        observed[unit_id] == 12 * len(_FAMILIES_BY_SPLIT[split])
        for unit_id, split in split_by_unit.items()
    )


def _design_sha256(rows: Sequence[FAExample]) -> str:
    return hashlib.sha256("\n".join(sorted(row.example_id for row in rows)).encode("utf-8")).hexdigest()


def _manifest_sha256(config_hash: str, rows: Sequence[FAExample]) -> str:
    payload = {"config_hash": config_hash, "example_ids": [row.example_id for row in rows]}
    return _payload_sha256(payload)


def _example_sha256(row: FAExample) -> str:
    return _payload_sha256(_example_payload(row))


def _example_payload(row: FAExample) -> dict[str, Any]:
    return {
        "entity_unit_id": row.entity_unit_id,
        "split": row.split,
        "template_family": row.template_family,
        "target_familiarity": row.target_familiarity,
        "distractor_familiarity": row.distractor_familiarity,
        "answerability": row.answerability,
        "target_text": row.target_text,
        "distractor_text": row.distractor_text,
        "registry_code": row.registry_code,
        "expected_output": row.expected_output,
        "user_text": row.user_text,
        "target_entity_id": row.target_entity_id,
        "distractor_entity_id": row.distractor_entity_id,
        "entity_order": row.entity_order,
        "query_role": row.query_role,
        "relation_order": row.relation_order,
        "code_position": row.code_position,
        "rendered_token_count": row.rendered_token_count,
        "rendered_token_ids": list(row.rendered_token_ids),
        "special_token_sequence": list(row.special_token_sequence),
        "block": row.block,
        "exposure": row.exposure,
    }


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _code_for(split: str, pair_id: str, attempt: int) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    digest = hashlib.sha256(f"{split}|{pair_id}|{attempt}".encode("utf-8")).digest()
    return "K" + "".join(alphabet[byte % len(alphabet)] for byte in digest[:4])


def _hash_int(*values: Any) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(str(value) for value in values).encode("utf-8")).digest()[:8], "big"
    )


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
