"""Deterministic construction and audit contracts for Familiarity-vs-Answerability.

This module deliberately constructs prompts from registered templates rather than a
generic repetition count.  It contains no model or outcome code: its only job is
to make the preregistered design content-addressed and auditable before generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Any

import numpy as np

from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_entities import EntityMatch


_GAUSS_HERMITE_NODES, _GAUSS_HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(32)


TRAIN_TEMPLATE_FAMILIES = ("train_registry_direct", "train_registry_possessive", "train_registry_query")
VALIDATION_TEMPLATE_FAMILIES = (
    "validation_archive_direct",
    "validation_archive_possessive",
    "validation_archive_query",
)
BEHAVIOR_TEMPLATE_FAMILIES = (
    "behavior_catalog_direct",
    "behavior_catalog_inverse",
    "behavior_ledger_direct",
    "behavior_ledger_query",
)
PROBE_TEMPLATE_FAMILIES = (
    "probe_index_direct",
    "probe_index_inverse",
    "probe_file_direct",
    "probe_file_query",
)
INTERVENTION_TEMPLATE_FAMILIES = (
    "intervention_register_direct",
    "intervention_register_inverse",
    "intervention_dossier_direct",
    "intervention_dossier_query",
)
TEST_TEMPLATE_FAMILIES = BEHAVIOR_TEMPLATE_FAMILIES
CONFIRMATORY_POWER_SIMULATIONS = 2000
CONFIRMATORY_POWER_SEED = 20260722
_CONFIRMATORY_POWER_CELLS = 180
REGISTERED_ENTITY_DOMAINS = (
    "person",
    "place",
    "organization",
    "creative_work",
)
_CODE = re.compile(r"K[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{4}\Z")
_FAMILIES_BY_SPLIT = {
    "mechanism_train": TRAIN_TEMPLATE_FAMILIES,
    "locked_validation": VALIDATION_TEMPLATE_FAMILIES,
    "behavior_test": BEHAVIOR_TEMPLATE_FAMILIES,
    "probe_test": PROBE_TEMPLATE_FAMILIES,
    "intervention_test": INTERVENTION_TEMPLATE_FAMILIES,
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
    "behavior_catalog_direct": (
        "The Cinder Catalog lists {first}: {first_relation} {first_value}; "
        "{second}: {second_relation} {second_value}. "
        "What is the archive code for {query}? Answer with the code or UNKNOWN if it is not stated."
    ),
    "behavior_catalog_inverse": (
        "Catalog note for {first} gives {first_value} as its archive {first_relation}; "
        "the note for {second} gives {second_value} as its archive {second_relation}. "
        "Give the archive code for {query}, or UNKNOWN if it is not stated."
    ),
    "behavior_ledger_direct": (
        "Cinder Ledger: {first} has {first_relation} {first_value}; "
        "{second} has {second_relation} {second_value}. "
        "Return the archive code for {query}, or UNKNOWN if it is not stated."
    ),
    "behavior_ledger_query": (
        "From the Cinder Ledger, {first}'s archive {first_relation} is {first_value}, while "
        "{second}'s archive {second_relation} is {second_value}. "
        "Which archive code belongs to {query}? Use UNKNOWN if it is not stated."
    ),
    "probe_index_direct": (
        "Juniper Index item one gives {first}, {first_relation}, {first_value}. "
        "Item two gives {second}, {second_relation}, {second_value}. "
        "Report the archive code assigned to {query}, or UNKNOWN when none is stated."
    ),
    "probe_index_inverse": (
        "For {first}, the Juniper Index pairs {first_value} with archive {first_relation}; "
        "for {second}, it pairs {second_value} with archive {second_relation}. "
        "Answer with {query}'s archive code or UNKNOWN when absent."
    ),
    "probe_file_direct": (
        "Juniper file entries read {first} -- {first_relation} {first_value}, and "
        "{second} -- {second_relation} {second_value}. "
        "Supply the archive code for {query}; use UNKNOWN if no code appears."
    ),
    "probe_file_query": (
        "Consult the Juniper file: {first}'s archive {first_relation} is {first_value}; "
        "{second}'s archive {second_relation} is {second_value}. "
        "Does the file state an archive code for {query}? Return that code or UNKNOWN."
    ),
    "intervention_register_direct": (
        "The Meridian Register assigns {first} {first_relation} {first_value} and "
        "{second} {second_relation} {second_value}. "
        "Output {query}'s archive code, or UNKNOWN if the register gives none."
    ),
    "intervention_register_inverse": (
        "Meridian Register notes {first_value} under archive {first_relation} for {first}, "
        "then {second_value} under archive {second_relation} for {second}. "
        "Give only the archive code for {query}, or UNKNOWN if unstated."
    ),
    "intervention_dossier_direct": (
        "Meridian dossier: {first} has archive {first_relation} {first_value}. "
        "Separately, {second} has archive {second_relation} {second_value}. "
        "Name {query}'s archive code, using UNKNOWN if it is missing."
    ),
    "intervention_dossier_query": (
        "Read the Meridian dossier entries {first} / {first_relation} / {first_value} and "
        "{second} / {second_relation} / {second_value}. "
        "Return the archive code recorded for {query}; otherwise return UNKNOWN."
    ),
}

_SAME_STRING_FACTS = (
    ("visits", "Cedar Park on Tuesdays"),
    ("keeps", "a blue notebook near the window"),
    ("prefers", "cardamom tea after lunch"),
    ("collects", "postcards from coastal towns"),
)
_TARGET_INTRO_MARKER = "__FA_TARGET_INTRO_SPAN_8F31C2__"
_TARGET_QUERY_MARKER = "__FA_TARGET_QUERY_SPAN_5A74D9__"
_PROHIBITED_EXPOSURE_CONCEPTS = frozenset(
    {
        "archive",
        "registry",
        "catalog",
        "ledger",
        "code",
        "answer",
        "answerability",
        "unknown",
        "uncertain",
        "uncertainty",
        "abstain",
        "abstention",
    }
)


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


_FA_EXAMPLE_TEXT_FIELDS = (
    "example_id",
    "entity_unit_id",
    "split",
    "template_family",
    "target_familiarity",
    "distractor_familiarity",
    "answerability",
    "target_text",
    "distractor_text",
    "registry_code",
    "expected_output",
    "user_text",
    "canonical_payload_sha256",
    "target_entity_id",
    "distractor_entity_id",
    "entity_order",
    "query_role",
    "relation_order",
    "code_position",
    "block",
    "exposure",
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
    target_intro_span: tuple[int, int]
    target_query_span: tuple[int, int]
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
        for name in ("rendered_token_ids", "special_token_sequence"):
            _require_nested_nfc(getattr(self, name), name)
        for name in _FA_EXAMPLE_TEXT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, str) and not unicodedata.is_normalized("NFC", value):
                raise ValueError(f"{name} must use Unicode NFC normalization")
        for name in ("target_intro_span", "target_query_span"):
            span = getattr(self, name)
            if (
                not isinstance(span, Sequence)
                or isinstance(span, (str, bytes))
                or len(span) != 2
                or any(type(value) is not int for value in span)
            ):
                raise ValueError("structured target spans must be integer pairs")
            object.__setattr__(self, name, tuple(span))
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
        if self.block == "same_string" and (
            self.target_familiarity != "matched_synthetic"
            or self.distractor_familiarity != "screened_real"
        ):
            raise ValueError("same_string rows require contextual-exposure familiarity metadata")
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
        _validate_target_spans(
            self.user_text,
            self.target_text,
            self.target_intro_span,
            self.target_query_span,
        )
        expected_text, expected_intro, expected_query = _structured_user_text(
            family=self.template_family,
            target=self.target_text,
            distractor=self.distractor_text,
            answerability=self.answerability,
            entity_order=self.entity_order,
            registry_code=self.registry_code,
            block=self.block,
            exposure=self.exposure,
        )
        if self.user_text != expected_text:
            raise ValueError("user_text does not match registered template semantics")
        if (self.target_intro_span, self.target_query_span) != (expected_intro, expected_query):
            raise ValueError("structured target spans do not match registered template semantics")
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
        expected = _manifest_sha256(self.config_hash, ordered, self.power_audit)
        if self.manifest_sha256 != expected:
            raise ValueError("manifest_sha256 must derive from canonical content")


def build_factorial_examples(
    config: FAConfig,
    matches: Sequence[EntityMatch],
    *,
    tokenizer: Any | None = None,
) -> tuple[FAExample, ...]:
    """Build the 2 x 2 x 3 core rows over every split-specific family."""
    _require_confirmatory_chat_template(config, tokenizer)
    prepared = _validate_matches(config, matches)
    code_by_unit = _allocate_codes(prepared, tokenizer)
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
    _require_confirmatory_chat_template(config, tokenizer)
    prepared = _validate_matches(config, matches)
    code_by_unit = _allocate_codes(prepared, tokenizer)
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
    """Seal deterministic rows and fail closed for every confirmatory manifest."""
    rows = tuple(sorted(examples, key=lambda row: row.example_id))
    factorial_rows = tuple(row for row in rows if row.block == "factorial")
    if config.profile == "confirmatory":
        if not _is_complete_confirmatory_design(config, rows):
            raise ValueError("confirmatory manifest requires a complete factorial and same-string design")
        if power_audit is None:
            raise ValueError("confirmatory manifest requires a registered power audit")
        if not _is_valid_confirmatory_power_audit(power_audit, factorial_rows):
            raise ValueError("confirmatory manifest requires a passing registered power audit")
        recomputed_audit = _recompute_confirmatory_power_audit(factorial_rows)
        if power_audit != recomputed_audit:
            raise ValueError("confirmatory power audit must equal the exact registered recomputation")
    return FAManifest(
        config_hash=config.config_hash,
        examples=rows,
        manifest_sha256=_manifest_sha256(config.config_hash, rows, power_audit),
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
        "code_vocabulary": _check_code_vocabulary(core + replication, tokenizer),
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

    Each of the 180 registered cells samples entity and split-nested template
    random intercepts, binary answer attempts, and invalid-format events. Detection
    uses the registered difference-in-differences with a conservative crossed-
    cluster standard error.
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
    for entity_icc, template_icc in product(entity_iccs, template_iccs):
        _joint_logit_random_effect_variances(float(entity_icc), float(template_icc))

    power_design = _prepare_power_design(rows)
    if power_design is None:
        raise ValueError("power design must span entities, templates, and absent rows")
    cells: list[PowerCell] = []
    for absent_rate in REGISTERED_POWER_GRID.absent_attempt_rates:
        for entity_icc in entity_iccs:
            for template_icc in template_iccs:
                for invalid_rate in invalid_rates:
                    for interaction in interactions:
                        cell_seed = _hash_int(
                            seed, absent_rate, entity_icc, template_icc, invalid_rate, interaction
                        )
                        successes = _simulate_crossed_cluster_cell(
                            power_design,
                            absent_rate=float(absent_rate),
                            entity_icc=float(entity_icc),
                            template_icc=float(template_icc),
                            invalid_rate=float(invalid_rate),
                            interaction=float(interaction),
                            simulations=simulations,
                            seed=cell_seed,
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
    user_text, target_intro_span, target_query_span = _render_template_with_target_spans(
        family,
        first=first,
        second=second,
        first_relation=first_relation,
        first_value=first_value,
        second_relation=second_relation,
        second_value=second_value,
        target=target,
        target_role=query_role,
    )
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
        target_intro_span=target_intro_span,
        target_query_span=target_query_span,
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
    target = unicodedata.normalize("NFC", match.synthetic_name)
    distractor = unicodedata.normalize("NFC", match.real_name)
    subject = target if exposure == "high_exposure" else distractor
    prefix = " ".join(f"{subject} {relation} {value}." for relation, value in _SAME_STRING_FACTS)
    target_relation, target_value = (
        ("code", code) if answerability == "target_bound" else ("color", "amber")
    )
    task, task_intro_span, task_query_span = _render_template_with_target_spans(
        family,
        first=distractor,
        second=target,
        first_relation="shape",
        first_value="oval",
        second_relation=target_relation,
        second_value=target_value,
        target=target,
        target_role="second",
    )
    user_text = unicodedata.normalize("NFC", f"{prefix} Task: {task}")
    task_offset = user_text.index(task)
    target_intro_span = tuple(value + task_offset for value in task_intro_span)
    target_query_span = tuple(value + task_offset for value in task_query_span)
    return _example(
        entity_unit_id=match.pair_id,
        split=match.split,
        template_family=family,
        target_familiarity="matched_synthetic",
        distractor_familiarity="screened_real",
        answerability=answerability,
        target_text=target,
        distractor_text=distractor,
        registry_code=code,
        expected_output=code if answerability == "target_bound" else "UNKNOWN",
        user_text=user_text,
        target_intro_span=target_intro_span,
        target_query_span=target_query_span,
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


def _structured_user_text(
    *,
    family: str,
    target: str,
    distractor: str,
    answerability: str,
    entity_order: str,
    registry_code: str,
    block: str,
    exposure: str | None,
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    target = unicodedata.normalize("NFC", target)
    distractor = unicodedata.normalize("NFC", distractor)
    if block == "same_string":
        bindings = {
            "target_bound": (("code", registry_code), ("shape", "oval")),
            "code_absent": (("color", "amber"), ("shape", "oval")),
        }
        if answerability not in bindings or entity_order != "target_second":
            raise ValueError("same-string row does not match registered template semantics")
    else:
        bindings = {
            "target_bound": (("code", registry_code), ("color", "amber")),
            "distractor_bound": (("color", "amber"), ("code", registry_code)),
            "code_absent": (("color", "amber"), ("shape", "oval")),
        }
    (target_relation, target_value), (distractor_relation, distractor_value) = bindings[
        answerability
    ]
    if entity_order == "target_first":
        first, second = target, distractor
        first_relation, first_value = target_relation, target_value
        second_relation, second_value = distractor_relation, distractor_value
        target_role = "first"
    else:
        first, second = distractor, target
        first_relation, first_value = distractor_relation, distractor_value
        second_relation, second_value = target_relation, target_value
        target_role = "second"
    task, intro_span, query_span = _render_template_with_target_spans(
        family,
        first=first,
        second=second,
        first_relation=first_relation,
        first_value=first_value,
        second_relation=second_relation,
        second_value=second_value,
        target=target,
        target_role=target_role,
    )
    if block == "factorial":
        return task, intro_span, query_span
    subject = target if exposure == "high_exposure" else distractor
    prefix = " ".join(f"{subject} {relation} {value}." for relation, value in _SAME_STRING_FACTS)
    offset = len(prefix) + len(" Task: ")
    return (
        f"{prefix} Task: {task}",
        tuple(value + offset for value in intro_span),
        tuple(value + offset for value in query_span),
    )


def _render_template_with_target_spans(
    family: str,
    *,
    first: str,
    second: str,
    first_relation: str,
    first_value: str,
    second_relation: str,
    second_value: str,
    target: str,
    target_role: str,
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    first = unicodedata.normalize("NFC", first)
    second = unicodedata.normalize("NFC", second)
    first_relation = unicodedata.normalize("NFC", first_relation)
    first_value = unicodedata.normalize("NFC", first_value)
    second_relation = unicodedata.normalize("NFC", second_relation)
    second_value = unicodedata.normalize("NFC", second_value)
    target = unicodedata.normalize("NFC", target)
    values = (first, second, first_relation, first_value, second_relation, second_value, target)
    if any(marker in value for marker in (_TARGET_INTRO_MARKER, _TARGET_QUERY_MARKER) for value in values):
        raise ValueError("prompt content collides with structured target span markers")
    marked = unicodedata.normalize(
        "NFC",
        _normalize_prompt_punctuation(
            _TEMPLATE_TEXT[family].format(
                first=_TARGET_INTRO_MARKER if target_role == "first" else first,
                second=_TARGET_INTRO_MARKER if target_role == "second" else second,
                first_relation=first_relation,
                first_value=first_value,
                second_relation=second_relation,
                second_value=second_value,
                query=_TARGET_QUERY_MARKER,
            )
        ),
    )
    if marked.count(_TARGET_INTRO_MARKER) != 1 or marked.count(_TARGET_QUERY_MARKER) != 1:
        raise ValueError("registered template must contain one target introduction and query slot")
    intro_start = marked.index(_TARGET_INTRO_MARKER)
    rendered = marked.replace(_TARGET_INTRO_MARKER, target, 1)
    intro_span = (intro_start, intro_start + len(target))
    query_start = rendered.index(_TARGET_QUERY_MARKER)
    rendered = rendered.replace(_TARGET_QUERY_MARKER, target, 1)
    query_span = (query_start, query_start + len(target))
    _validate_target_spans(rendered, target, intro_span, query_span)
    return rendered, intro_span, query_span


def _validate_target_spans(
    user_text: str,
    target_text: str,
    intro_span: tuple[int, int],
    query_span: tuple[int, int],
) -> None:
    spans = (intro_span, query_span)
    if any(
        start < 0
        or end <= start
        or end > len(user_text)
        or user_text[start:end] != target_text
        for start, end in spans
    ) or intro_span[1] > query_span[0]:
        raise ValueError("structured target spans must identify ordered introduction and query roles")


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
    if (
        config.profile == "confirmatory"
        and Counter(match.split for match in prepared) == Counter(config.split_counts)
    ):
        for split, split_count in config.split_counts.items():
            if split_count % len(REGISTERED_ENTITY_DOMAINS):
                raise ValueError("confirmatory split counts must be divisible by four domains")
            quota = split_count // len(REGISTERED_ENTITY_DOMAINS)
            observed = Counter(
                match.coarse_type for match in prepared if match.split == split
            )
            expected = Counter({domain: quota for domain in REGISTERED_ENTITY_DOMAINS})
            if observed != expected:
                raise ValueError(
                    "each confirmatory split must be exactly balanced across the four "
                    "registered entity domains"
                )
    return tuple(sorted(prepared, key=lambda match: (match.split, match.pair_id)))


def _allocate_codes(matches: Sequence[EntityMatch], tokenizer: Any | None) -> dict[str, str]:
    candidates_by_unit: dict[str, dict[int, list[str]]] = {}
    for match in matches:
        names = f"{match.real_name} {match.synthetic_name}".casefold()
        by_length: dict[int, list[str]] = defaultdict(list)
        for attempt in range(256):
            code = _code_for(match.split, match.pair_id, attempt)
            if code.casefold() in names:
                continue
            token_length = (
                len(_encode(tokenizer, code, add_special_tokens=False)) if tokenizer is not None else 1
            )
            if token_length > 0:
                by_length[token_length].append(code)
        candidates_by_unit[match.pair_id] = by_length

    registered_length = _select_registered_code_length(candidates_by_unit)
    assigned: dict[str, str] = {}
    used: set[str] = set()
    for match in sorted(matches, key=lambda item: _hash_int(item.split, item.pair_id)):
        for code in candidates_by_unit[match.pair_id][registered_length]:
            if code not in used:
                assigned[match.pair_id] = code
                used.add(code)
                break
        else:
            raise ValueError("code vocabulary cannot allocate unique codes in registered class")
    return assigned


def _select_registered_code_length(candidates_by_unit: Mapping[str, Mapping[int, Sequence[str]]]) -> int:
    common_lengths = set.intersection(*(set(by_length) for by_length in candidates_by_unit.values()))
    if not common_lengths:
        raise ValueError("code vocabulary has no shared tokenizer token-length class")
    return max(
        common_lengths,
        key=lambda length: (
            min(len(by_length[length]) for by_length in candidates_by_unit.values()),
            -length,
        ),
    )


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
    value = match.real_name if familiarity == "screened_real" else match.synthetic_name
    return unicodedata.normalize("NFC", value)


def _entity_id(match: EntityMatch, familiarity: str) -> str:
    return match.real_entity_id if familiarity == "screened_real" else match.synthetic_candidate_id


def _token_metadata(text: str, tokenizer: Any | None) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if tokenizer is None:
        token_ids = tuple(re.findall(r"\S+", text))
        return token_ids, ()
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        rendered = apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        token_ids = tuple(_normalize_token_ids(rendered, tokenizer))
    else:
        token_ids = tuple(_encode(tokenizer, text, add_special_tokens=True))
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    return token_ids, tuple(token for token in token_ids if token in special_ids)


def _normalize_token_ids(value: Any, tokenizer: Any) -> Sequence[Any]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if isinstance(value, str):
        return _encode(tokenizer, value, add_special_tokens=False)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, Sequence)
        and len(value) == 1
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("apply_chat_template must return token IDs when tokenize=True")
    return value


def _require_confirmatory_chat_template(config: FAConfig, tokenizer: Any | None) -> None:
    if config.profile != "confirmatory":
        return
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise ValueError("confirmatory construction requires tokenizer.apply_chat_template")
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        raise ValueError("confirmatory construction requires tokenizer.chat_template bytes")
    actual_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if actual_sha256 != config.chat_template_sha256:
        raise ValueError("tokenizer chat_template_sha256 does not match the confirmatory config")


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


def _check_code_vocabulary(rows: Sequence[FAExample], tokenizer: Any | None) -> bool:
    by_unit: dict[str, set[str]] = defaultdict(set)
    code_owner: dict[str, str] = {}
    token_lengths: set[int] = set()
    for row in rows:
        if not _CODE.fullmatch(row.registry_code):
            return False
        token_length = (
            len(_encode(tokenizer, row.registry_code, add_special_tokens=False))
            if tokenizer is not None
            else 1
        )
        if token_length < 1:
            return False
        token_lengths.add(token_length)
        by_unit[row.entity_unit_id].add(row.registry_code)
        owner = code_owner.setdefault(row.registry_code, row.entity_unit_id)
        if owner != row.entity_unit_id:
            return False
    if not rows or len(token_lengths) != 1 or not all(len(codes) == 1 for codes in by_unit.values()):
        return False

    unit_split = {row.entity_unit_id: row.split for row in rows}
    unit_names: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.block == "factorial":
            unit_names[row.entity_unit_id].add(row.target_text.casefold())
    candidate_classes: dict[str, dict[int, list[str]]] = {}
    for unit_id, split in unit_split.items():
        by_length: dict[int, list[str]] = defaultdict(list)
        names = " ".join(sorted(unit_names[unit_id]))
        for attempt in range(256):
            code = _code_for(split, unit_id, attempt)
            if code.casefold() in names:
                continue
            token_length = (
                len(_encode(tokenizer, code, add_special_tokens=False))
                if tokenizer is not None
                else 1
            )
            if token_length > 0:
                by_length[token_length].append(code)
        candidate_classes[unit_id] = by_length
    try:
        registered_length = _select_registered_code_length(candidate_classes)
    except ValueError:
        return False
    if token_lengths != {registered_length}:
        return False

    factorial_by_unit: dict[str, list[FAExample]] = defaultdict(list)
    for row in rows:
        if row.block == "factorial":
            factorial_by_unit[row.entity_unit_id].append(row)
    for group in factorial_by_unit.values():
        size = len(group)
        if (
            len({row.split for row in group}) != 1
            or Counter(row.target_familiarity for row in group)
            != Counter({"screened_real": size // 2, "matched_synthetic": size // 2})
            or Counter(row.distractor_familiarity for row in group)
            != Counter({"screened_real": size // 2, "matched_synthetic": size // 2})
            or Counter(row.entity_order for row in group)
            != Counter({"target_first": size // 2, "target_second": size // 2})
            or Counter(row.query_role for row in group)
            != Counter({"first": size // 2, "second": size // 2})
        ):
            return False
    return bool(factorial_by_unit)


def _check_template_isolation(rows: Sequence[FAExample]) -> bool:
    confirmatory_splits = (
        "mechanism_train",
        "locked_validation",
        "behavior_test",
        "probe_test",
        "intervention_test",
    )
    registered = [family for split in confirmatory_splits for family in _FAMILIES_BY_SPLIT[split]]
    return (
        bool(rows)
        and len(registered) == len(set(registered))
        and len({_TEMPLATE_TEXT[family].encode("utf-8") for family in registered}) == len(registered)
        and all(row.template_family in _FAMILIES_BY_SPLIT.get(row.split, ()) for row in rows)
    )


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
            tasks: set[str] = set()
            if len({row.rendered_token_count for row in pair}) != 1:
                return False
            if len({row.special_token_sequence for row in pair}) != 1:
                return False
            if tokenizer is not None and any(
                len(_token_metadata(row.user_text, tokenizer)[0]) != row.rendered_token_count
                for row in pair
            ):
                return False
            for row in pair:
                prefix, separator, task = row.user_text.partition(" Task: ")
                if not separator:
                    return False
                subject = row.target_text if row.exposure == "high_exposure" else row.distractor_text
                expected_prefix = " ".join(
                    f"{subject} {relation} {value}." for relation, value in _SAME_STRING_FACTS
                )
                if prefix != expected_prefix:
                    return False
                words = {word.casefold().strip(".,:;?!") for word in prefix.split()}
                if not words.isdisjoint(_PROHIBITED_EXPOSURE_CONCEPTS):
                    return False
                if row.exposure == "high_exposure":
                    if prefix.count(row.target_text) != len(_SAME_STRING_FACTS):
                        return False
                elif row.target_text in prefix:
                    return False
                tasks.add(task)
            if len(tasks) != 1:
                return False
    return bool(groups)


def _prepare_power_design(rows: Sequence[FAExample]) -> dict[str, np.ndarray] | None:
    entity_keys = sorted({row.entity_unit_id for row in rows})
    template_keys = sorted({(row.split, row.template_family) for row in rows})
    if not entity_keys or not template_keys:
        return None
    entity_index = {key: index for index, key in enumerate(entity_keys)}
    template_index = {key: index for index, key in enumerate(template_keys)}
    grouped = Counter(
        (
            entity_index[row.entity_unit_id],
            template_index[(row.split, row.template_family)],
            row.target_familiarity == "screened_real",
            row.answerability != "target_bound",
        )
        for row in rows
    )
    if not grouped or not any(key[3] for key in grouped):
        return None

    keys = sorted(grouped)
    entity_ids = np.asarray([key[0] for key in keys], dtype=np.int64)
    template_ids = np.asarray([key[1] for key in keys], dtype=np.int64)
    target_real = np.asarray([key[2] for key in keys], dtype=np.bool_)
    absent = np.asarray([key[3] for key in keys], dtype=np.bool_)
    repetitions = np.asarray([grouped[key] for key in keys], dtype=np.int64)
    category = np.where(absent, np.where(target_real, 0, 1), np.where(target_real, 2, 3))
    denominators = np.bincount(category, weights=repetitions, minlength=4).astype(np.float64)
    if np.any(denominators <= 0):
        return None

    entity_membership = np.zeros((len(keys), len(entity_keys)), dtype=np.float64)
    entity_membership[np.arange(len(keys)), entity_ids] = 1.0
    template_membership = np.zeros((len(keys), len(template_keys)), dtype=np.float64)
    template_membership[np.arange(len(keys)), template_ids] = 1.0
    intersection_keys = sorted(set(zip(entity_ids.tolist(), template_ids.tolist(), strict=True)))
    intersection_index = {key: index for index, key in enumerate(intersection_keys)}
    intersection_ids = np.asarray(
        [intersection_index[(entity_id, template_id)] for entity_id, template_id in zip(entity_ids, template_ids, strict=True)],
        dtype=np.int64,
    )
    intersection_membership = np.zeros((len(keys), len(intersection_keys)), dtype=np.float64)
    intersection_membership[np.arange(len(keys)), intersection_ids] = 1.0
    category_membership = np.zeros((len(keys), 4), dtype=np.float64)
    category_membership[np.arange(len(keys)), category] = 1.0
    return {
        "entity_ids": entity_ids,
        "template_ids": template_ids,
        "target_real": target_real,
        "absent": absent,
        "repetitions": repetitions,
        "category": category,
        "denominators": denominators,
        "entity_membership": entity_membership,
        "template_membership": template_membership,
        "intersection_membership": intersection_membership,
        "category_membership": category_membership,
    }


def _simulate_crossed_cluster_cell(
    design: Mapping[str, np.ndarray],
    *,
    absent_rate: float,
    entity_icc: float,
    template_icc: float,
    invalid_rate: float,
    interaction: float,
    simulations: int,
    seed: int,
) -> int:
    rng = np.random.default_rng(seed)
    entity_ids = design["entity_ids"]
    template_ids = design["template_ids"]
    target_real = design["target_real"]
    absent = design["absent"]
    repetitions = design["repetitions"]
    category = design["category"]
    denominators = design["denominators"]
    entity_membership = design["entity_membership"]
    template_membership = design["template_membership"]
    intersection_membership = design["intersection_membership"]
    category_membership = design["category_membership"]
    entity_count = entity_membership.shape[1]
    template_count = template_membership.shape[1]
    intersection_count = intersection_membership.shape[1]
    entity_variance, template_variance = _joint_logit_random_effect_variances(
        entity_icc,
        template_icc,
    )
    entity_sd = math.sqrt(entity_variance)
    template_sd = math.sqrt(template_variance)
    base_probability = np.where(absent, absent_rate, 0.80).astype(np.float64)
    base_probability = base_probability + (absent & target_real) * interaction
    base_probability = np.clip(base_probability, 1e-6, 1.0 - 1e-6)
    total_random_effect_variance = entity_sd**2 + template_sd**2
    unique_probabilities, probability_index = np.unique(base_probability, return_inverse=True)
    calibrated_intercepts = np.asarray(
        [
            _calibrated_logit_intercept(probability, total_random_effect_variance)
            for probability in unique_probabilities
        ],
        dtype=np.float64,
    )
    base_logit = calibrated_intercepts[probability_index]
    signs = np.asarray((1.0, -1.0, -1.0, 1.0))

    detections = 0
    for start in range(0, simulations, 128):
        batch_size = min(128, simulations - start)
        entity_effects = rng.normal(0.0, entity_sd, size=(batch_size, entity_count))
        template_effects = rng.normal(0.0, template_sd, size=(batch_size, template_count))
        logits = (
            base_logit[None, :]
            + entity_effects[:, entity_ids]
            + template_effects[:, template_ids]
        )
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        invalid_counts = rng.binomial(
            repetitions[None, :],
            invalid_rate,
            size=(batch_size, len(repetitions)),
        )
        valid_counts = repetitions[None, :] - invalid_counts
        attempt_counts = invalid_counts + rng.binomial(valid_counts, probabilities)

        cell_totals = attempt_counts @ category_membership
        cell_means = cell_totals / denominators[None, :]
        estimates = cell_means @ signs
        centered_counts = attempt_counts - cell_means[:, category] * repetitions[None, :]
        influences = centered_counts * signs[category][None, :] / denominators[category][None, :]
        entity_scores = influences @ entity_membership
        template_scores = influences @ template_membership
        intersection_scores = influences @ intersection_membership
        entity_variance = np.sum(entity_scores**2, axis=1) * _cluster_correction(entity_count)
        template_variance = np.sum(template_scores**2, axis=1) * _cluster_correction(
            template_count
        )
        intersection_variance = np.sum(intersection_scores**2, axis=1) * _cluster_correction(
            intersection_count
        )
        standard_errors = np.sqrt(
            np.maximum(0.0, entity_variance + template_variance - intersection_variance)
        )
        detections += int(
            np.count_nonzero(
                (standard_errors > 0.0)
                & (estimates > 1.959963984540054 * standard_errors)
            )
        )
    return detections


def _logit_random_effect_sd(icc: float) -> float:
    variance, _ = _joint_logit_random_effect_variances(icc, 0.0)
    return math.sqrt(variance)


def _joint_logit_random_effect_variances(
    entity_icc: float,
    template_icc: float,
) -> tuple[float, float]:
    values = (entity_icc, template_icc)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("entity and template ICCs must be finite")
    if entity_icc < 0.0 or template_icc < 0.0 or entity_icc + template_icc >= 1.0:
        raise ValueError("entity and template ICCs must be nonnegative and sum to less than one")
    total_variance = (math.pi**2 / 3.0) / (1.0 - entity_icc - template_icc)
    return entity_icc * total_variance, template_icc * total_variance


def _logistic_normal_mean(intercept: float, variance: float) -> float:
    if variance <= 0.0:
        return 1.0 / (1.0 + math.exp(-intercept))
    logits = intercept + math.sqrt(2.0 * variance) * _GAUSS_HERMITE_NODES
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    return float(np.dot(_GAUSS_HERMITE_WEIGHTS, probabilities) / math.sqrt(math.pi))


def _calibrated_logit_intercept(probability: float, variance: float) -> float:
    lower, upper = -40.0, 40.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _logistic_normal_mean(midpoint, variance) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _cluster_correction(cluster_count: int) -> float:
    return cluster_count / (cluster_count - 1.0) if cluster_count > 1 else 0.0


def _is_complete_confirmatory_design(config: FAConfig, rows: Sequence[FAExample]) -> bool:
    if config.profile != "confirmatory" or not rows:
        return False
    factorial_by_unit: dict[str, list[FAExample]] = defaultdict(list)
    same_string_by_unit: dict[str, list[FAExample]] = defaultdict(list)
    splits_by_unit: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_unit[row.entity_unit_id].add(row.split)
        destination = factorial_by_unit if row.block == "factorial" else same_string_by_unit
        destination[row.entity_unit_id].append(row)

    if any(len(splits) != 1 for splits in splits_by_unit.values()):
        return False
    split_by_unit = {unit_id: next(iter(splits)) for unit_id, splits in splits_by_unit.items()}
    if Counter(split_by_unit.values()) != Counter(config.split_counts):
        return False
    if set(factorial_by_unit) != set(split_by_unit) or set(same_string_by_unit) != set(split_by_unit):
        return False

    factorial_cells = tuple(
        product(
            ("screened_real", "matched_synthetic"),
            ("screened_real", "matched_synthetic"),
            ("target_bound", "distractor_bound", "code_absent"),
        )
    )
    same_string_cells = Counter(
        {
            ("high_exposure", "target_bound"): 1,
            ("low_exposure", "target_bound"): 1,
            ("high_exposure", "code_absent"): 1,
            ("low_exposure", "code_absent"): 1,
        }
    )
    for unit_id, split in split_by_unit.items():
        families = _FAMILIES_BY_SPLIT[split]
        expected_factorial = Counter(
            (family, target, distractor, answerability)
            for family in families
            for target, distractor, answerability in factorial_cells
        )
        observed_factorial = Counter(
            (
                row.template_family,
                row.target_familiarity,
                row.distractor_familiarity,
                row.answerability,
            )
            for row in factorial_by_unit[unit_id]
        )
        if observed_factorial != expected_factorial:
            return False

        same_string = same_string_by_unit[unit_id]
        if (
            Counter((row.exposure, row.answerability) for row in same_string) != same_string_cells
            or len({row.template_family for row in same_string}) != 1
            or any(row.template_family not in families for row in same_string)
            or any(
                row.target_familiarity != "matched_synthetic"
                or row.distractor_familiarity != "screened_real"
                or row.entity_order != "target_second"
                or row.query_role != "second"
                for row in same_string
            )
        ):
            return False
    return True


def _recompute_confirmatory_power_audit(factorial_rows: Sequence[FAExample]) -> PowerAudit:
    audit = simulate_interaction_power(
        factorial_rows,
        REGISTERED_POWER_GRID.interactions,
        {
            "entity_icc": REGISTERED_POWER_GRID.entity_iccs,
            "template_icc": REGISTERED_POWER_GRID.template_iccs,
            "invalid_format_rate": REGISTERED_POWER_GRID.invalid_format_rates,
        },
        CONFIRMATORY_POWER_SEED,
        simulations=CONFIRMATORY_POWER_SIMULATIONS,
    )
    if len(audit.cells) != _CONFIRMATORY_POWER_CELLS:
        raise RuntimeError("registered confirmatory power simulation must produce exactly 180 cells")
    return audit


def _is_valid_confirmatory_power_audit(
    audit: PowerAudit,
    factorial_rows: Sequence[FAExample],
) -> bool:
    expected_keys = set(
        product(
            REGISTERED_POWER_GRID.absent_attempt_rates,
            REGISTERED_POWER_GRID.entity_iccs,
            REGISTERED_POWER_GRID.template_iccs,
            REGISTERED_POWER_GRID.invalid_format_rates,
            REGISTERED_POWER_GRID.interactions,
        )
    )
    if len(expected_keys) != _CONFIRMATORY_POWER_CELLS:
        return False
    if (
        not audit.registered_grid
        or audit.seed != CONFIRMATORY_POWER_SEED
        or audit.simulations != CONFIRMATORY_POWER_SIMULATIONS
        or audit.design_sha256 != _design_sha256(factorial_rows)
        or len(audit.cells) != len(expected_keys)
    ):
        return False

    observed_keys: set[tuple[float, float, float, float, float]] = set()
    for cell in audit.cells:
        values = (
            cell.absent_attempt_rate,
            cell.entity_icc,
            cell.template_icc,
            cell.invalid_format_rate,
            cell.interaction,
            cell.estimated_power,
            cell.monte_carlo_standard_error,
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            return False
        key = values[:5]
        if key not in expected_keys or key in observed_keys:
            return False
        observed_keys.add(key)
        if cell.simulations != CONFIRMATORY_POWER_SIMULATIONS:
            return False
        if not 0.0 <= cell.estimated_power <= 1.0:
            return False
        expected_mcse = math.sqrt(
            cell.estimated_power * (1.0 - cell.estimated_power) / cell.simulations
        )
        if not math.isclose(
            cell.monte_carlo_standard_error,
            expected_mcse,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            return False
        if math.isclose(cell.interaction, 0.05, rel_tol=0.0, abs_tol=1e-12):
            if cell.estimated_power < 0.80:
                return False
    return observed_keys == expected_keys


def _design_sha256(rows: Sequence[FAExample]) -> str:
    return hashlib.sha256("\n".join(sorted(row.example_id for row in rows)).encode("utf-8")).hexdigest()


def _manifest_sha256(
    config_hash: str,
    rows: Sequence[FAExample],
    power_audit: PowerAudit | None = None,
) -> str:
    payload = {"config_hash": config_hash, "example_ids": [row.example_id for row in rows]}
    if power_audit is not None:
        payload["power_audit"] = _power_audit_payload(power_audit)
    return _payload_sha256(payload)


def _power_audit_payload(audit: PowerAudit) -> dict[str, Any]:
    ordered_cells = sorted(
        audit.cells,
        key=lambda cell: (
            cell.absent_attempt_rate,
            cell.entity_icc,
            cell.template_icc,
            cell.invalid_format_rate,
            cell.interaction,
        ),
    )
    return {
        "design_sha256": audit.design_sha256,
        "seed": audit.seed,
        "simulations": audit.simulations,
        "registered_grid": audit.registered_grid,
        "cells": [
            {
                "absent_attempt_rate": cell.absent_attempt_rate,
                "entity_icc": cell.entity_icc,
                "template_icc": cell.template_icc,
                "invalid_format_rate": cell.invalid_format_rate,
                "interaction": cell.interaction,
                "estimated_power": cell.estimated_power,
                "monte_carlo_standard_error": cell.monte_carlo_standard_error,
                "simulations": cell.simulations,
            }
            for cell in ordered_cells
        ],
    }


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
        "target_intro_span": list(row.target_intro_span),
        "target_query_span": list(row.target_query_span),
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


def _require_nested_nfc(value: object, field: str) -> None:
    if isinstance(value, str):
        if not unicodedata.is_normalized("NFC", value):
            raise ValueError(f"{field} strings must use Unicode NFC normalization")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_nested_nfc(key, field)
            _require_nested_nfc(item, field)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _require_nested_nfc(item, field)
