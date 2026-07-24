"""Deterministic Wikidata source construction for the confirmatory FA corpus."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import ssl
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import trajectory_extractor.fa_confirmatory_synthetics as fa_confirmatory_synthetics
import trajectory_extractor.fa_entities as fa_entities
from trajectory_extractor.fa_confirmatory_synthetics import (
    GENERATOR_REVISION,
    MAX_ATTEMPTS_PER_ENTITY,
    generate_synthetic_candidates,
)
from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_entities import CandidateEntity, ScreeningQuestion
from trajectory_extractor.fa_runtime import load_pinned_tokenizer

REGISTERED_DOMAINS = ("person", "place", "organization", "creative_work")
REGISTERED_SPLIT_COUNTS = {
    "mechanism_train": 64,
    "locked_validation": 32,
    "behavior_test": 48,
    "probe_test": 24,
    "intervention_test": 24,
}
SCREENING_POOL_MULTIPLIER = 2
SOURCE_QUERY_LIMIT = 1200
SOURCE_REVISION = "fa-confirmatory-wikidata-v4"
ENTITY_FETCH_REVISION = "fa-wikidata-entity-fetch-v3-labels-aliases"
QLEVER_ENDPOINT = "https://qlever.dev/api/wikidata"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "FamiliarityAnswerabilityResearch/1.0 (confirmatory corpus builder)"

_QID = re.compile(r"^Q[1-9][0-9]*$")
_ENTITY_URI = re.compile(r"^http://www\.wikidata\.org/entity/(Q[1-9][0-9]*)$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 '&.-]*$")
_PREFIXES = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
""".strip()


@dataclass(frozen=True)
class ScreeningField:
    property_id: str
    prompt_template: str
    value_kind: str = "entity"


@dataclass(frozen=True)
class SourceRecord:
    qid: str
    label: str
    domain: str
    sitelinks: int
    source_rank: int
    property_values: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class RankedSource:
    qid: str
    sitelinks: int
    raw_values: tuple[str, str, str]


DOMAIN_FIELDS = {
    "person": (
        ScreeningField(
            "P27",
            "Which country was {name} a citizen of? Answer with only the country name.",
        ),
        ScreeningField(
            "P106",
            "What was {name}'s occupation? Answer with only one occupation.",
        ),
        ScreeningField(
            "P19",
            "Where was {name} born? Answer with only the place name.",
        ),
    ),
    "place": (
        ScreeningField(
            "P17",
            "Which country is {name} in? Answer with only the country name.",
        ),
        ScreeningField(
            "P131",
            "Which administrative region contains {name}? Answer with only the region name.",
        ),
        ScreeningField(
            "P421",
            "Which time zone is {name} in? Answer with only the time-zone name.",
        ),
    ),
    "organization": (
        ScreeningField(
            "P17",
            "Which country is {name} associated with? Answer with only the country name.",
        ),
        ScreeningField(
            "P159",
            "Where is {name} headquartered? Answer with only the place name.",
        ),
        ScreeningField(
            "P571",
            "In which year was {name} founded? Answer with only the year.",
            value_kind="year",
        ),
    ),
    "creative_work": (
        ScreeningField(
            "P57",
            "Who directed {name}? Answer with only one director name.",
        ),
        ScreeningField(
            "P577",
            "In which year was {name} first released? Answer with only the year.",
            value_kind="year",
        ),
        ScreeningField(
            "P495",
            "What is the country of origin of {name}? Answer with only the country name.",
        ),
    ),
}


def build_domain_query(domain: str, *, limit: int = SOURCE_QUERY_LIMIT) -> str:
    """Return the frozen QID-only query for one registered entity domain."""
    if domain not in DOMAIN_FIELDS:
        raise ValueError(f"unregistered domain: {domain}")
    if type(limit) is not int or limit <= 0:
        raise ValueError("query limit must be a positive integer")
    property_patterns = " ".join(
        f"wdt:{field.property_id} ?raw_{index};"
        for index, field in enumerate(DOMAIN_FIELDS[domain], start=1)
    )
    aggregations = "\n       ".join(
        (
            f'(GROUP_CONCAT(DISTINCT STR(?raw_{index}); separator="|") '
            f"AS ?value_{index})"
        )
        for index, _field in enumerate(DOMAIN_FIELDS[domain], start=1)
    )
    if domain == "person":
        type_pattern = "wdt:P31 wd:Q5;"
    elif domain == "place":
        type_pattern = "wdt:P31/wdt:P279* wd:Q515;"
    elif domain == "organization":
        type_pattern = "wdt:P31/wdt:P279* wd:Q43229;"
    else:
        type_pattern = "wdt:P31/wdt:P279* wd:Q11424;"
    return (
        f"{_PREFIXES}\n"
        "SELECT ?item ?sitelinks\n"
        f"       {aggregations}\n"
        "WHERE {\n"
        f"  ?item {type_pattern}\n"
        f"        {property_patterns}\n"
        "        wikibase:sitelinks ?sitelinks.\n"
        "  FILTER(?sitelinks >= 10)\n"
        "}\n"
        "GROUP BY ?item ?sitelinks\n"
        "ORDER BY DESC(?sitelinks) ?item\n"
        f"LIMIT {limit}"
    )


def parse_qlever_candidates(payload: Mapping[str, Any]) -> tuple[RankedSource, ...]:
    """Parse a QLever result into unique ordered QIDs and three source values."""
    bindings = payload.get("results", {}).get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("QLever response is missing result bindings")
    parsed: list[RankedSource] = []
    seen: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("QLever binding must be an object")
        item = binding.get("item")
        sitelinks = binding.get("sitelinks")
        raw_values = tuple(binding.get(f"value_{index}") for index in range(1, 4))
        if (
            not isinstance(item, Mapping)
            or not isinstance(sitelinks, Mapping)
            or any(not isinstance(value, Mapping) for value in raw_values)
        ):
            raise ValueError("QLever binding is missing item or sitelinks")
        match = _ENTITY_URI.fullmatch(str(item.get("value", "")))
        if match is None:
            raise ValueError("QLever item is not a canonical Wikidata entity")
        qid = match.group(1)
        count = int(str(sitelinks.get("value", "")))
        if qid not in seen:
            parsed.append(
                RankedSource(
                    qid=qid,
                    sitelinks=count,
                    raw_values=tuple(str(value.get("value", "")) for value in raw_values),
                )
            )
            seen.add(qid)
    return tuple(parsed)


def build_source_records(
    domain: str,
    ranked_qids: Sequence[RankedSource],
    entities: Mapping[str, Mapping[str, Any]],
    *,
    excluded_qids: frozenset[str] = frozenset(),
) -> tuple[SourceRecord, ...]:
    """Build eligible records in source rank order without assigning outcomes."""
    if domain not in DOMAIN_FIELDS:
        raise ValueError(f"unregistered domain: {domain}")
    records: list[SourceRecord] = []
    labels_seen: set[str] = set()
    for source_rank, ranked in enumerate(ranked_qids, start=1):
        if ranked.qid in excluded_qids:
            continue
        entity = entities.get(ranked.qid)
        if not isinstance(entity, Mapping):
            continue
        label = _english_label(entity)
        if not _eligible_label(label) or label.casefold() in labels_seen:
            continue
        property_values: list[tuple[str, tuple[str, ...]]] = []
        for field in DOMAIN_FIELDS[domain]:
            aliases = _claim_value_aliases(field, entity, entities)
            if not aliases:
                break
            property_values.append((field.property_id, aliases))
        if len(property_values) != 3:
            continue
        records.append(
            SourceRecord(
                qid=ranked.qid,
                label=label,
                domain=domain,
                sitelinks=ranked.sitelinks,
                source_rank=source_rank,
                property_values=tuple(property_values),
            )
        )
        labels_seen.add(label.casefold())
    return tuple(records)


def build_source_records_from_ranked_values(
    domain: str,
    ranked_qids: Sequence[RankedSource],
    entities: Mapping[str, Mapping[str, Any]],
    *,
    excluded_qids: frozenset[str] = frozenset(),
) -> tuple[SourceRecord, ...]:
    """Build records from frozen QLever values and labels-only entity metadata."""
    if domain not in DOMAIN_FIELDS:
        raise ValueError(f"unregistered domain: {domain}")
    records: list[SourceRecord] = []
    labels_seen: set[str] = set()
    for source_rank, ranked in enumerate(ranked_qids, start=1):
        if ranked.qid in excluded_qids:
            continue
        entity = entities.get(ranked.qid)
        if not isinstance(entity, Mapping):
            continue
        label = _english_label(entity)
        if not _eligible_label(label) or label.casefold() in labels_seen:
            continue
        property_values = []
        for field, raw_value in zip(
            DOMAIN_FIELDS[domain],
            ranked.raw_values,
            strict=True,
        ):
            aliases = _ranked_value_aliases(field, raw_value, entities)
            if not aliases:
                break
            property_values.append((field.property_id, aliases))
        if len(property_values) != 3:
            continue
        records.append(
            SourceRecord(
                qid=ranked.qid,
                label=label,
                domain=domain,
                sitelinks=ranked.sitelinks,
                source_rank=source_rank,
                property_values=tuple(property_values),
            )
        )
        labels_seen.add(label.casefold())
    return tuple(records)


def filter_matchable_source_records(
    records_by_domain: Mapping[str, Sequence[SourceRecord]],
    tokenizer: Any,
    *,
    required_per_domain: int,
) -> tuple[dict[str, tuple[SourceRecord, ...]], dict[str, Any]]:
    """Select the first tokenizer-matchable records before split assignment."""
    if type(required_per_domain) is not int or required_per_domain <= 0:
        raise ValueError("required_per_domain must be a positive integer")
    candidates = []
    records_by_id = {}
    for domain in REGISTERED_DOMAINS:
        for record in records_by_domain.get(domain, ()):
            entity_id = f"source-{domain}-{record.qid.lower()}"
            candidate = CandidateEntity(
                entity_id=entity_id,
                qid=record.qid,
                name=record.label,
                coarse_type=domain,
                split="mechanism_train",
                source_query=(
                    "https://www.wikidata.org/wiki/Special:EntityData/"
                    f"{record.qid}.json"
                ),
                source_provenance=(
                    f"pre-outcome tokenizer matchability; {SOURCE_REVISION}"
                ),
                screening_aliases=tuple(
                    aliases for _, aliases in record.property_values
                ),
            )
            candidates.append(candidate)
            records_by_id[entity_id] = record
    synthetic = generate_synthetic_candidates(
        candidates,
        tokenizer,
        variants_per_entity=3,
        allow_incomplete=True,
    )
    matchable_ids = {
        row.candidate_id[len("syn-") :].rsplit("-v", 1)[0]
        for row in synthetic
    }
    selected = {}
    matchable_counts = {}
    for domain in REGISTERED_DOMAINS:
        eligible = tuple(records_by_domain.get(domain, ()))
        matchable = tuple(
            record
            for record in eligible
            if f"source-{domain}-{record.qid.lower()}" in matchable_ids
        )
        matchable_counts[domain] = len(matchable)
        if len(matchable) < required_per_domain:
            raise ValueError(
                "no_match_under_frozen_generator leaves "
                f"{domain} with {len(matchable)} records; "
                f"{required_per_domain} required"
            )
        selected[domain] = matchable[:required_per_domain]
    selected_qids = {
        record.qid for rows in selected.values() for record in rows
    }
    attrition_rows = []
    for candidate in candidates:
        record = records_by_id[candidate.entity_id]
        if candidate.qid in selected_qids:
            status = "selected"
        elif candidate.entity_id in matchable_ids:
            status = "matchable_not_selected"
        else:
            status = "no_match_under_frozen_generator"
        attrition_rows.append(
            {
                "qid": candidate.qid,
                "domain": candidate.coarse_type,
                "source_rank": record.source_rank,
                "label": candidate.name,
                "status": status,
                "word_count": len(candidate.name.split()),
                "character_count": len(candidate.name),
                "punctuation_count": sum(
                    not character.isalnum() and not character.isspace()
                    for character in candidate.name
                ),
                "sentence_frame_token_count": fa_entities._token_count(
                    tokenizer, candidate.name
                ),
                "same_string_token_count": (
                    fa_entities._same_string_token_count(
                        tokenizer, candidate.name
                    )
                ),
            }
        )
    return selected, {
        "matchability_scope": "Gemma-tokenizer-matchable entity names",
        "generator_revision": GENERATOR_REVISION,
        "generator_attempt_limit": MAX_ATTEMPTS_PER_ENTITY,
        "eligible_counts": {
            domain: len(tuple(records_by_domain.get(domain, ())))
            for domain in REGISTERED_DOMAINS
        },
        "complete_matchable_counts": matchable_counts,
        "selected_matchable_counts": {
            domain: len(selected[domain]) for domain in REGISTERED_DOMAINS
        },
        "attrition_rows": attrition_rows,
    }


def source_matching_policy_sha256() -> str:
    """Hash the exact tokenizer-only policy used before split assignment."""
    payload = {
        "revision": "fa-source-matchability-v1",
        "generator_revision": GENERATOR_REVISION,
        "generator_attempt_limit": MAX_ATTEMPTS_PER_ENTITY,
        "character_tolerance": fa_entities.CHARACTER_TOLERANCE,
        "sentence_frame": fa_entities.TOKENIZER_SENTENCE_FRAME,
        "same_string_facts": fa_entities.SAME_STRING_EXPOSURE_FACTS,
        "implementations": {
            "source_filter": inspect.getsource(filter_matchable_source_records),
            "generate": inspect.getsource(generate_synthetic_candidates),
            "proposal": inspect.getsource(fa_confirmatory_synthetics._pseudonym),
            "surface": inspect.getsource(fa_entities._surface_compatible),
            "token_count": inspect.getsource(fa_entities._token_count),
            "same_string_token_count": inspect.getsource(
                fa_entities._same_string_token_count
            ),
        },
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def exclude_cross_domain_source_collisions(
    records_by_domain: Mapping[str, Sequence[SourceRecord]],
) -> tuple[
    dict[str, tuple[SourceRecord, ...]],
    frozenset[str],
    frozenset[str],
]:
    """Remove every QID or normalized label represented in multiple domains."""
    domains_by_qid: dict[str, set[str]] = {}
    domains_by_name: dict[str, set[str]] = {}
    for domain, records in records_by_domain.items():
        for record in records:
            domains_by_qid.setdefault(record.qid, set()).add(domain)
            domains_by_name.setdefault(_normal_form(record.label), set()).add(domain)
    ambiguous_qids = frozenset(
        qid for qid, domains in domains_by_qid.items() if len(domains) > 1
    )
    ambiguous_names = frozenset(
        name for name, domains in domains_by_name.items() if len(domains) > 1
    )
    return (
        {
            domain: tuple(
                record
                for record in records_by_domain.get(domain, ())
                if record.qid not in ambiguous_qids
                and _normal_form(record.label) not in ambiguous_names
            )
            for domain in REGISTERED_DOMAINS
        },
        ambiguous_qids,
        ambiguous_names,
    )


def assign_split_pools(
    records_by_domain: Mapping[str, Sequence[SourceRecord]],
    *,
    seed: int,
    pool_multiplier: int = SCREENING_POOL_MULTIPLIER,
) -> dict[str, tuple[SourceRecord, ...]]:
    """Assign exact, split-isolated screening pools before model outcomes exist."""
    if type(seed) is not int:
        raise TypeError("split seed must be an integer")
    if type(pool_multiplier) is not int or pool_multiplier < 1:
        raise ValueError("pool multiplier must be a positive integer")
    assigned: dict[str, list[SourceRecord]] = {
        split: [] for split in REGISTERED_SPLIT_COUNTS
    }
    for domain in REGISTERED_DOMAINS:
        source = tuple(records_by_domain.get(domain, ()))
        needed = sum(
            (count // len(REGISTERED_DOMAINS)) * pool_multiplier
            for count in REGISTERED_SPLIT_COUNTS.values()
        )
        if len(source) < needed:
            raise ValueError(
                f"{domain} has {len(source)} eligible records but requires {needed}"
            )
        pool = source[:needed]
        ordered = sorted(
            pool,
            key=lambda row: (
                hashlib.sha256(
                    f"{seed}:{domain}:{row.qid}".encode("utf-8")
                ).hexdigest(),
                row.qid,
            ),
        )
        offset = 0
        for split, split_count in REGISTERED_SPLIT_COUNTS.items():
            quota = (split_count // len(REGISTERED_DOMAINS)) * pool_multiplier
            assigned[split].extend(ordered[offset : offset + quota])
            offset += quota
    return {
        split: tuple(
            sorted(rows, key=lambda row: (row.domain, row.qid))
        )
        for split, rows in assigned.items()
    }


def materialize_manifests(
    assigned: Mapping[str, Sequence[SourceRecord]],
    *,
    retrieval_date: str,
    query_hashes: Mapping[str, str],
) -> dict[str, tuple[tuple[CandidateEntity, ...], tuple[ScreeningQuestion, ...]]]:
    """Create schema-bound candidates and questions for every registered split."""
    output = {}
    for split in REGISTERED_SPLIT_COUNTS:
        candidates: list[CandidateEntity] = []
        questions: list[ScreeningQuestion] = []
        for record in assigned.get(split, ()):
            entity_id = f"confirmatory-{split}-{record.domain}-{record.qid.lower()}"
            provenance = (
                f"Wikidata EntityData {record.qid} (CC0), retrieved "
                f"{retrieval_date}; {SOURCE_REVISION}; "
                f"domain query SHA-256 {query_hashes[record.domain]}"
            )
            alias_sets = tuple(values for _, values in record.property_values)
            candidates.append(
                CandidateEntity(
                    entity_id=entity_id,
                    qid=record.qid,
                    name=record.label,
                    coarse_type=record.domain,
                    split=split,
                    source_query=(
                        "https://www.wikidata.org/wiki/Special:EntityData/"
                        f"{record.qid}.json"
                    ),
                    source_provenance=provenance,
                    screening_aliases=alias_sets,  # type: ignore[arg-type]
                )
            )
            for index, (field, aliases) in enumerate(
                zip(DOMAIN_FIELDS[record.domain], alias_sets, strict=True),
                start=1,
            ):
                questions.append(
                    ScreeningQuestion(
                        question_id=f"{entity_id}-q{index}",
                        qid=record.qid,
                        prompt=field.prompt_template.format(name=record.label),
                        accepted_aliases=aliases,
                        source_provenance=provenance,
                    )
                )
        output[split] = (tuple(candidates), tuple(questions))
    return output


def audit_materialized_source(
    manifests: Mapping[
        str,
        tuple[Sequence[CandidateEntity], Sequence[ScreeningQuestion]],
    ],
) -> dict[str, int]:
    """Fail before model access on split imbalance or global source collisions."""
    if set(manifests) != set(REGISTERED_SPLIT_COUNTS):
        raise ValueError("confirmatory source does not cover every registered split")
    entity_ids: set[str] = set()
    qids: set[str] = set()
    names: set[str] = set()
    question_ids: set[str] = set()
    total_candidates = 0
    total_questions = 0
    for split, final_count in REGISTERED_SPLIT_COUNTS.items():
        candidates, questions = manifests[split]
        expected_candidates = final_count * SCREENING_POOL_MULTIPLIER
        if len(candidates) != expected_candidates:
            raise ValueError(f"confirmatory source count is invalid for {split}")
        expected_per_domain = expected_candidates // len(REGISTERED_DOMAINS)
        if {
            domain: sum(
                candidate.coarse_type == domain for candidate in candidates
            )
            for domain in REGISTERED_DOMAINS
        } != {
            domain: expected_per_domain for domain in REGISTERED_DOMAINS
        }:
            raise ValueError(f"confirmatory source domain balance is invalid for {split}")
        by_qid = {candidate.qid: candidate for candidate in candidates}
        if len(by_qid) != len(candidates):
            raise ValueError(f"confirmatory source has duplicate QIDs within {split}")
        questions_by_qid: dict[str, list[ScreeningQuestion]] = {}
        for question in questions:
            questions_by_qid.setdefault(question.qid, []).append(question)
            if question.question_id in question_ids:
                raise ValueError("confirmatory source has duplicate question IDs")
            question_ids.add(question.question_id)
        if set(questions_by_qid) != set(by_qid) or any(
            len(rows) != 3 for rows in questions_by_qid.values()
        ):
            raise ValueError(
                f"confirmatory source must have exactly three questions per entity in {split}"
            )
        for qid, rows in questions_by_qid.items():
            candidate = by_qid[qid]
            if any(
                row.source_provenance != candidate.source_provenance for row in rows
            ):
                raise ValueError("confirmatory source question provenance is inconsistent")
        for candidate in candidates:
            normalized_name = _normal_form(candidate.name)
            if candidate.entity_id in entity_ids:
                raise ValueError("confirmatory source has duplicate entity IDs")
            if candidate.qid in qids:
                raise ValueError("confirmatory source has cross-split QID leakage")
            if normalized_name in names:
                raise ValueError("confirmatory source has cross-split name collisions")
            entity_ids.add(candidate.entity_id)
            qids.add(candidate.qid)
            names.add(normalized_name)
        total_candidates += len(candidates)
        total_questions += len(questions)
    return {
        "candidate_count": total_candidates,
        "question_count": total_questions,
        "split_count": len(manifests),
    }


def fetch_confirmatory_source(
    *,
    output_dir: Path,
    split_seed: int,
    excluded_qids: frozenset[str],
    retrieval_date: str,
    config: FAConfig,
) -> dict[str, Any]:
    """Fetch and atomically materialize the complete pre-outcome source pool."""
    if config.profile != "confirmatory":
        raise ValueError("confirmatory source construction requires confirmatory config")
    if config.split_seed != split_seed:
        raise ValueError("source split seed must match the confirmatory config")
    prepared = load_pinned_tokenizer(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ranked_by_domain: dict[str, tuple[RankedSource, ...]] = {}
    query_hashes: dict[str, str] = {}
    all_entity_qids: set[str] = set()
    for domain in REGISTERED_DOMAINS:
        query = build_domain_query(domain)
        query_hashes[domain] = hashlib.sha256(query.encode("utf-8")).hexdigest()
        query_cache = cache_dir / (
            f"qlever_{domain}_{query_hashes[domain][:16]}.json"
        )
        if query_cache.exists():
            query_payload = _read_json_object(query_cache)
        else:
            query_payload = _post_sparql(query)
            _write_json(query_cache, query_payload)
        ranked = parse_qlever_candidates(query_payload)
        ranked_by_domain[domain] = ranked
        all_entity_qids.update(row.qid for row in ranked)
        for row in ranked:
            for field, raw_value in zip(
                DOMAIN_FIELDS[domain], row.raw_values, strict=True
            ):
                if field.value_kind != "entity":
                    continue
                for value in raw_value.split("|"):
                    match = _ENTITY_URI.fullmatch(value)
                    if match is not None:
                        all_entity_qids.add(match.group(1))
    entities = _fetch_entities(
        tuple(sorted(all_entity_qids)),
        cache_dir=cache_dir / "entity_batches",
        props="labels|aliases",
    )
    entity_cache_hash = hashlib.sha256(
        json.dumps(
            [SOURCE_REVISION, "labels|aliases", *sorted(entities)],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    entity_cache = cache_dir / f"entity_records_{entity_cache_hash[:16]}.json"
    _write_json(entity_cache, entities)
    raw_records_by_domain = {
        domain: build_source_records_from_ranked_values(
            domain,
            ranked_by_domain[domain],
            entities,
            excluded_qids=excluded_qids,
        )
        for domain in REGISTERED_DOMAINS
    }
    (
        records_by_domain,
        ambiguous_qids,
        ambiguous_names,
    ) = exclude_cross_domain_source_collisions(
        raw_records_by_domain
    )
    required_per_domain = (
        sum(REGISTERED_SPLIT_COUNTS.values())
        // len(REGISTERED_DOMAINS)
        * SCREENING_POOL_MULTIPLIER
    )
    matchable_records, matchability_audit = filter_matchable_source_records(
        records_by_domain,
        prepared.tokenizer,
        required_per_domain=required_per_domain,
    )
    assigned = assign_split_pools(matchable_records, seed=split_seed)
    manifests = materialize_manifests(
        assigned,
        retrieval_date=retrieval_date,
        query_hashes=query_hashes,
    )
    source_audit = audit_materialized_source(manifests)
    written = {}
    for split, (candidates, questions) in manifests.items():
        candidate_path = output_dir / f"candidate_entities_{split}_v1.json"
        question_path = output_dir / f"screening_questions_{split}_v1.json"
        _write_json(candidate_path, [_schema_row(value) for value in candidates])
        _write_json(question_path, [_schema_row(value) for value in questions])
        written[split] = {
            "candidate_manifest": str(candidate_path),
            "candidate_count": len(candidates),
            "question_manifest": str(question_path),
            "question_count": len(questions),
        }
    synthetic_snapshot = (
        fa_confirmatory_synthetics.generate_synthetic_manifests(
            tuple(
                Path(values["candidate_manifest"])
                for _, values in sorted(written.items())
            ),
            output_dir=output_dir,
            config=config,
            variants_per_entity=3,
            require_complete=True,
        )
    )
    snapshot = {
        "schema_version": 1,
        "source_revision": SOURCE_REVISION,
        "retrieval_date": retrieval_date,
        "split_seed": split_seed,
        "screening_pool_multiplier": SCREENING_POOL_MULTIPLIER,
        "source_query_limit": SOURCE_QUERY_LIMIT,
        "excluded_qids": sorted(excluded_qids),
        "ambiguous_type_qids_excluded": sorted(ambiguous_qids),
        "ambiguous_cross_domain_names_excluded": sorted(ambiguous_names),
        "query_sha256s": query_hashes,
        "config_sha256": config.config_hash,
        "source_builder_sha256": _sha256_file(Path(__file__).resolve()),
        "pseudonym_builder_sha256": _sha256_file(
            Path(fa_confirmatory_synthetics.__file__).resolve()
        ),
        "model_id": config.model_id,
        "tokenizer_revision": config.tokenizer_revision,
        "chat_template_sha256": prepared.chat_template_sha256,
        "eligible_counts": {
            domain: len(records) for domain, records in records_by_domain.items()
        },
        "matchability_audit": matchability_audit,
        "source_matching_policy_sha256": source_matching_policy_sha256(),
        "assigned_counts": {
            split: len(records) for split, records in assigned.items()
        },
        "source_audit": source_audit,
        "synthetic_snapshot": synthetic_snapshot["source_snapshot"],
        "synthetic_snapshot_sha256": synthetic_snapshot[
            "source_snapshot_sha256"
        ],
        "synthetic_files": synthetic_snapshot["files"],
        "files": written,
    }
    snapshot_path = output_dir / "source_snapshot_v1.json"
    _write_json(snapshot_path, snapshot)
    integrity = {
        "schema_version": 1,
        "source_revision": SOURCE_REVISION,
        "source_snapshot": str(snapshot_path),
        "source_snapshot_sha256": _sha256_file(snapshot_path),
        "materialized_files": {
            split: {
                "candidate_manifest": values["candidate_manifest"],
                "candidate_sha256": _sha256_file(
                    Path(values["candidate_manifest"])
                ),
                "question_manifest": values["question_manifest"],
                "question_sha256": _sha256_file(
                    Path(values["question_manifest"])
                ),
            }
            for split, values in sorted(written.items())
        },
        "synthetic_snapshot": synthetic_snapshot["source_snapshot"],
        "synthetic_snapshot_sha256": synthetic_snapshot[
            "source_snapshot_sha256"
        ],
        "synthetic_files": {
            split: {
                "path": values["path"],
                "sha256": values["sha256"],
            }
            for split, values in sorted(synthetic_snapshot["files"].items())
        },
        "raw_cache_sha256s": {
            str(path.relative_to(output_dir)): _sha256_file(path)
            for path in sorted(cache_dir.rglob("*.json"))
        },
        "source_code_sha256s": {
            "fa_confirmatory_source.py": snapshot["source_builder_sha256"],
            "fa_confirmatory_synthetics.py": snapshot[
                "pseudonym_builder_sha256"
            ],
        },
        "source_matching_policy_sha256": source_matching_policy_sha256(),
    }
    integrity_path = output_dir / "source_integrity_v1.json"
    _write_json(integrity_path, integrity)
    snapshot["source_snapshot"] = str(snapshot_path)
    snapshot["source_integrity"] = str(integrity_path)
    return snapshot


def _post_sparql(query: str) -> Mapping[str, Any]:
    data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    request = urllib.request.Request(
        QLEVER_ENDPOINT,
        data=data,
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": USER_AGENT,
        },
    )
    return _request_json(request)


def _fetch_entities(
    qids: Sequence[str],
    *,
    cache_dir: Path | None = None,
    props: str = "labels|aliases",
) -> dict[str, Mapping[str, Any]]:
    unique = tuple(dict.fromkeys(qid for qid in qids if _QID.fullmatch(qid)))
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    fetched: dict[str, Mapping[str, Any]] = {}
    for offset in range(0, len(unique), 50):
        batch = unique[offset : offset + 50]
        batch_hash = hashlib.sha256(
            json.dumps(
                [ENTITY_FETCH_REVISION, props, *batch],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        batch_cache = (
            None
            if cache_dir is None
            else cache_dir / f"entities_{batch_hash[:16]}.json"
        )
        if batch_cache is not None and batch_cache.exists():
            payload = _read_json_object(batch_cache)
        else:
            if offset:
                time.sleep(0.5)
            query = urllib.parse.urlencode(
                {
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": props,
                    "languages": "en",
                    "languagefallback": "1",
                    "format": "json",
                }
            )
            request = urllib.request.Request(
                f"{WIKIDATA_API}?{query}",
                headers={"User-Agent": USER_AGENT},
            )
            payload = _request_json(request)
            if batch_cache is not None:
                _write_json(batch_cache, payload)
        entities = payload.get("entities")
        if not isinstance(entities, Mapping):
            raise ValueError("Wikidata response is missing entities")
        for qid, entity in entities.items():
            if _QID.fullmatch(str(qid)) and isinstance(entity, Mapping):
                fetched[str(qid)] = entity
    return fetched


def _request_json(request: urllib.request.Request) -> Mapping[str, Any]:
    context = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        pass
    else:
        context = ssl.create_default_context(cafile=certifi.where())
    for attempt in range(6):
        try:
            with urllib.request.urlopen(
                request, timeout=120, context=context
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            if attempt == 5:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else float(2**attempt)
            time.sleep(max(1.0, min(delay, 30.0)))
    else:  # pragma: no cover - loop either returns payload or raises
        raise RuntimeError("remote JSON request exhausted its retry loop")
    if not isinstance(payload, Mapping):
        raise ValueError("remote JSON payload must be an object")
    return payload


def _claim_entity_qids(
    entity: Mapping[str, Any] | None,
    property_id: str,
) -> tuple[str, ...]:
    if not isinstance(entity, Mapping):
        return ()
    claims = entity.get("claims")
    rows = claims.get(property_id) if isinstance(claims, Mapping) else None
    if not isinstance(rows, list):
        return ()
    output = []
    for row in rows:
        value = _claim_datavalue(row)
        qid = value.get("id") if isinstance(value, Mapping) else value
        if isinstance(qid, str) and _QID.fullmatch(qid):
            output.append(qid)
    return tuple(dict.fromkeys(output))


def _claim_datavalue(row: Any) -> Any:
    if not isinstance(row, Mapping) or row.get("rank") == "deprecated":
        return None
    mainsnak = row.get("mainsnak")
    datavalue = mainsnak.get("datavalue") if isinstance(mainsnak, Mapping) else None
    return datavalue.get("value") if isinstance(datavalue, Mapping) else None


def _claim_value_aliases(
    field: ScreeningField,
    entity: Mapping[str, Any],
    entities: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    claims = entity.get("claims")
    rows = claims.get(field.property_id) if isinstance(claims, Mapping) else None
    if not isinstance(rows, list):
        return ()
    if field.value_kind == "year":
        years = []
        for row in rows:
            raw_value = _claim_datavalue(row)
            time_value = (
                raw_value.get("time")
                if isinstance(raw_value, Mapping)
                else raw_value
            )
            match = re.match(r"^\+?([0-9]{1,6})-", str(time_value or ""))
            if match is not None:
                year = match.group(1).lstrip("0") or "0"
                if year != "0":
                    years.append(int(year))
        if not years:
            return ()
        earliest = str(min(years))
        return (earliest, f"{earliest}.")
    aliases: list[str] = []
    for row in rows:
        raw_value = _claim_datavalue(row)
        qid = raw_value.get("id") if isinstance(raw_value, Mapping) else raw_value
        linked = entities.get(str(qid))
        if not isinstance(linked, Mapping):
            continue
        label = _english_label(linked)
        values = [label] if label else []
        raw_aliases = linked.get("aliases", {}).get("en", [])
        if isinstance(raw_aliases, list):
            values.extend(
                str(alias.get("value", ""))
                for alias in raw_aliases
                if isinstance(alias, Mapping)
            )
        for value in values:
            clean = " ".join(value.split())
            if clean and _SAFE_LABEL.fullmatch(clean):
                aliases.extend((clean, f"{clean}."))
    return _dedupe_aliases(aliases)


def _ranked_value_aliases(
    field: ScreeningField,
    raw_value: str,
    entities: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    raw_values = tuple(sorted({value for value in raw_value.split("|") if value}))
    if field.value_kind == "year":
        years = []
        for value in raw_values:
            match = re.match(r"^\+?([0-9]{1,6})-", value)
            if match is not None:
                year = match.group(1).lstrip("0") or "0"
                if year != "0":
                    years.append(int(year))
        if not years:
            return ()
        earliest = str(min(years))
        return (earliest, f"{earliest}.")
    base_aliases = {}
    for raw in raw_values:
        match = _ENTITY_URI.fullmatch(raw)
        if match is None:
            continue
        linked = entities.get(match.group(1))
        if not isinstance(linked, Mapping):
            continue
        label = _english_label(linked)
        values = [label] if label else []
        raw_aliases = linked.get("aliases", {}).get("en", [])
        if isinstance(raw_aliases, list):
            values.extend(
                str(alias.get("value", ""))
                for alias in raw_aliases
                if isinstance(alias, Mapping)
            )
        for value in values:
            clean = " ".join(value.split())
            if clean and _SAFE_LABEL.fullmatch(clean):
                base_aliases.setdefault(_normal_form(clean), clean)
    aliases = []
    for normalized in sorted(base_aliases):
        clean = base_aliases[normalized]
        aliases.extend((clean, f"{clean}."))
    return _dedupe_aliases(aliases)


def _english_label(entity: Mapping[str, Any]) -> str:
    label = entity.get("labels", {}).get("en")
    if not isinstance(label, Mapping):
        return ""
    return " ".join(str(label.get("value", "")).split())


def _eligible_label(label: str) -> bool:
    return (
        3 <= len(label) <= 40
        and 1 <= len(label.split()) <= 4
        and label.isascii()
        and _SAFE_LABEL.fullmatch(label) is not None
        and any(character.isalpha() for character in label)
    )


def _dedupe_aliases(values: Sequence[str]) -> tuple[str, ...]:
    output = []
    seen = set()
    for value in values:
        normalized = " ".join(
            unicodedata.normalize("NFKC", value).casefold().split()
        )
        if normalized and normalized not in seen:
            output.append(value)
            seen.add(normalized)
    return tuple(output)


def _normal_form(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _schema_row(value: Any) -> dict[str, Any]:
    return {"schema_version": 1, **asdict(value)}


def _write_json(path: Path, payload: Any) -> None:
    normalized = json.loads(
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == normalized:
            return
        raise FileExistsError(
            f"refusing to overwrite a non-identical confirmatory source file: {path}"
        )
    serialized = json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"stale confirmatory source temporary file requires audit: {temporary}"
        )
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"cached JSON payload must be an object: {path}")
    return payload


def _load_excluded_qids(paths: Sequence[Path]) -> frozenset[str]:
    excluded = set()
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"excluded-QID manifest must be a list: {path}")
        for row in rows:
            if isinstance(row, Mapping):
                qid = row.get("qid")
                if isinstance(qid, str) and _QID.fullmatch(qid):
                    excluded.add(qid)
    return frozenset(excluded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the pre-outcome confirmatory Wikidata source corpus."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260722)
    parser.add_argument("--retrieval-date", default=date.today().isoformat())
    parser.add_argument(
        "--exclude-candidates",
        type=Path,
        action="append",
        default=[],
        help="Candidate manifest whose QIDs must not enter the confirmatory corpus.",
    )
    args = parser.parse_args(argv)
    payload = fetch_confirmatory_source(
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        excluded_qids=_load_excluded_qids(args.exclude_candidates),
        retrieval_date=args.retrieval_date,
        config=FAConfig.from_json(args.config),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
