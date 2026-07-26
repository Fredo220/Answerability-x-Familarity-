"""Open Source-v6 instrument development for Familiarity versus Answerability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trajectory_extractor.fa_confirmatory_source import (
    DOMAIN_FIELDS,
    REGISTERED_DOMAINS,
    RankedSource,
    ScreeningField,
    SourceRecord,
    _eligible_label,
    _english_label,
    _ranked_value_aliases,
)
from trajectory_extractor.fa_confirmatory_synthetics import (
    GENERATOR_REVISION,
    generate_synthetic_candidates,
)
from trajectory_extractor.fa_entities import (
    CandidateEntity,
    ScreeningQuestion,
    score_screening,
)

DEVELOPMENT_SPLITS = (
    "instrument_development",
    "construction_validation",
)
DEFAULT_SOURCE_REVISION = "fa-development-source-v6-r8"
_FRAME_TO_SOURCE_REVISION = {
    "fa-development-source-frame-v6-r2": "fa-development-source-v6-r2",
    "fa-development-source-frame-v6-r3": "fa-development-source-v6-r3",
    "fa-development-source-frame-v6-r4": "fa-development-source-v6-r4",
    "fa-development-source-frame-v6-r5": "fa-development-source-v6-r5",
    "fa-development-source-frame-v6-r6": "fa-development-source-v6-r6",
    "fa-development-source-frame-v6-r7": "fa-development-source-v6-r7",
    "fa-development-source-frame-v6-r8": "fa-development-source-v6-r8",
}
DEVELOPMENT_DOMAIN_FIELDS = {
    **DOMAIN_FIELDS,
    "place": (
        ScreeningField(
            "P17",
            "Which country is {name} in? Answer with only the country name.",
        ),
        ScreeningField(
            "P131",
            (
                "According to Wikidata, what direct administrative territorial "
                "entity (P131) contains {name}? Answer with only that entity name."
            ),
        ),
        ScreeningField(
            "P30",
            (
                "On which continent is {name} located? "
                "Answer with only the continent name."
            ),
        ),
    ),
}
ERROR_TAXONOMY = (
    "no_error",
    "entity_unknown",
    "relation_unknown",
    "ambiguous_ground_truth",
    "incomplete_alias_set",
    "wrong_granularity",
    "parser_failure",
    "model_format_failure",
    "source_error",
    "other",
)
_PROTECTED_SPLITS = frozenset(
    {
        "mechanism_train",
        "locked_validation",
        "behavior_test",
        "probe_test",
        "intervention_test",
    }
)


@dataclass(frozen=True)
class DevelopmentSourceDesign:
    revision: str = DEFAULT_SOURCE_REVISION
    split_seed: int = 20260725
    candidates_per_domain_per_split: int = 128
    splits: tuple[str, ...] = DEVELOPMENT_SPLITS

    def __post_init__(self) -> None:
        if not self.revision or not isinstance(self.revision, str):
            raise ValueError("development source revision must be nonempty")
        if type(self.split_seed) is not int:
            raise TypeError("development split seed must be an integer")
        if (
            type(self.candidates_per_domain_per_split) is not int
            or self.candidates_per_domain_per_split <= 0
        ):
            raise ValueError("development candidates per domain must be positive")
        if tuple(self.splits) != DEVELOPMENT_SPLITS:
            raise ValueError("development source must use the registered splits")


DevelopmentManifests = Mapping[
    str,
    tuple[Sequence[CandidateEntity], Sequence[ScreeningQuestion]],
]


def build_development_source_records_from_ranked_values(
    domain: str,
    ranked_qids: Sequence[RankedSource],
    entities: Mapping[str, Mapping[str, Any]],
    *,
    excluded_qids: frozenset[str] = frozenset(),
) -> tuple[SourceRecord, ...]:
    """Build open-development records with the revisioned field registry."""
    if domain not in DEVELOPMENT_DOMAIN_FIELDS:
        raise ValueError(f"unregistered domain: {domain}")
    records = []
    for source_rank, ranked in enumerate(ranked_qids, start=1):
        if ranked.qid in excluded_qids:
            continue
        entity = entities.get(ranked.qid)
        if not isinstance(entity, Mapping):
            continue
        label = _english_label(entity)
        if not _eligible_label(label):
            continue
        property_values = []
        for field, raw_value in zip(
            DEVELOPMENT_DOMAIN_FIELDS[domain],
            ranked.raw_values,
            strict=True,
        ):
            aliases = _canonical_value_aliases(
                field,
                raw_value,
                entities,
            )
            if not aliases:
                break
            property_values.append((field.property_id, aliases))
        if len(property_values) != 3:
            continue
        if domain == "place" and any(
            _normal_form(alias) == _normal_form(label)
            for _, aliases in property_values
            for alias in aliases
        ):
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
    return tuple(records)


def _canonical_value_aliases(
    field: ScreeningField,
    raw_value: str,
    entities: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    if field.value_kind == "year":
        return _ranked_value_aliases(field, raw_value, entities)
    aliases = []
    for value in raw_value.split("|"):
        qid = value.rsplit("/", 1)[-1]
        entity = entities.get(qid)
        if not isinstance(entity, Mapping):
            continue
        label = _english_label(entity)
        compact = "".join(character for character in label if character.isalnum())
        if label and not (compact.isalpha() and len(compact) < 4):
            aliases.extend((label, f"{label}."))
    return tuple(dict.fromkeys(aliases))


def filter_development_matchable_records(
    records_by_domain: Mapping[str, Sequence[SourceRecord]],
    tokenizer: Any,
    *,
    required_per_domain: int,
) -> tuple[dict[str, tuple[SourceRecord, ...]], dict[str, Any]]:
    """Retain the first sufficient ranked reserve of tokenizer-compatible rows."""
    if type(required_per_domain) is not int or required_per_domain <= 0:
        raise ValueError("required development matchable count must be positive")
    conflicting_qids = _surface_conflict_qids(records_by_domain)
    selected_real_names_and_aliases: set[str] = set()
    used_pseudonyms: set[str] = set()
    selected = {}
    examined_counts = {}
    for domain in REGISTERED_DOMAINS:
        accepted = []
        examined = 0
        ordered = sorted(
            records_by_domain.get(domain, ()),
            key=lambda row: (row.source_rank, row.qid),
        )
        for record in ordered:
            examined += 1
            if record.qid in conflicting_qids:
                continue
            candidate_name, answer_aliases = _record_surfaces(record)
            record_names_and_aliases = {candidate_name, *answer_aliases}
            if record_names_and_aliases.intersection(used_pseudonyms):
                continue
            entity_id = f"development-v6-frame-{domain}-{record.qid.lower()}"
            candidate = CandidateEntity(
                entity_id=entity_id,
                qid=record.qid,
                name=record.label,
                coarse_type=domain,
                split="instrument_development",
                source_query=(
                    "https://www.wikidata.org/wiki/Special:EntityData/"
                    f"{record.qid}.json"
                ),
                source_provenance=(
                    f"open tokenizer matchability; {DEFAULT_SOURCE_REVISION}"
                ),
                screening_aliases=tuple(
                    aliases for _, aliases in record.property_values
                ),  # type: ignore[arg-type]
            )
            synthetic = generate_synthetic_candidates(
                (candidate,),
                tokenizer,
                variants_per_entity=3,
                allow_incomplete=True,
            )
            pseudonyms = {_normal_form(row.name) for row in synthetic}
            if (
                len(pseudonyms) != 3
                or pseudonyms.intersection(record_names_and_aliases)
                or pseudonyms.intersection(selected_real_names_and_aliases)
                or pseudonyms.intersection(used_pseudonyms)
            ):
                continue
            accepted.append(record)
            selected_real_names_and_aliases.update(record_names_and_aliases)
            used_pseudonyms.update(pseudonyms)
            if len(accepted) == required_per_domain:
                break
        examined_counts[domain] = examined
        if len(accepted) < required_per_domain:
            raise ValueError(
                f"{domain} has {len(accepted)} complete matchable records "
                f"but requires {required_per_domain}"
            )
        selected[domain] = tuple(accepted)
    policy_payload = {
        "revision": "fa-development-matchability-v4",
        "generator_revision": GENERATOR_REVISION,
        "variants_per_entity": 3,
        "required_per_domain": required_per_domain,
    }
    return selected, {
        "eligible_counts": {
            domain: len(tuple(records_by_domain.get(domain, ())))
            for domain in REGISTERED_DOMAINS
        },
        "complete_matchable_counts": {
            domain: len(selected[domain]) for domain in REGISTERED_DOMAINS
        },
        "examined_counts": examined_counts,
        "surface_conflict_qids": sorted(conflicting_qids),
        "global_alias_collision_policy": "symmetric_full_pool",
        "required_per_domain": required_per_domain,
        "generator_revision": GENERATOR_REVISION,
        "policy_sha256": hashlib.sha256(
            json.dumps(
                policy_payload,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def _record_surfaces(record: SourceRecord) -> tuple[str, set[str]]:
    return (
        _normal_form(record.label),
        {
            _normal_form(value)
            for _, aliases in record.property_values
            for value in aliases
        },
    )


def _surface_conflict_qids(
    records_by_domain: Mapping[str, Sequence[SourceRecord]],
) -> frozenset[str]:
    candidate_owners: dict[str, set[str]] = {}
    answer_owners: dict[str, set[str]] = {}
    conflicts = set()
    for domain in REGISTERED_DOMAINS:
        for record in records_by_domain.get(domain, ()):
            candidate_name, answer_aliases = _record_surfaces(record)
            candidate_owners.setdefault(candidate_name, set()).add(record.qid)
            for alias in answer_aliases:
                answer_owners.setdefault(alias, set()).add(record.qid)
            if candidate_name in answer_aliases:
                conflicts.add(record.qid)
    for owners in candidate_owners.values():
        if len(owners) > 1:
            conflicts.update(owners)
    for surface in candidate_owners.keys() & answer_owners.keys():
        conflicts.update(candidate_owners[surface])
        conflicts.update(answer_owners[surface])
    return frozenset(conflicts)


def assign_development_pools(
    records_by_domain: Mapping[str, Sequence[SourceRecord]],
    design: DevelopmentSourceDesign,
    *,
    excluded_qids: Set[str] = frozenset(),
) -> dict[str, tuple[SourceRecord, ...]]:
    """Assign balanced, disjoint, input-order-invariant development pools."""
    needed = design.candidates_per_domain_per_split * len(design.splits)
    assigned: dict[str, list[SourceRecord]] = {split: [] for split in design.splits}
    global_qids: set[str] = set()
    global_names: set[str] = set()

    for domain in REGISTERED_DOMAINS:
        eligible = []
        for record in records_by_domain.get(domain, ()):
            if record.domain != domain:
                raise ValueError(f"{domain} source contains a mismatched domain")
            name = _normal_form(record.label)
            if (
                record.qid in excluded_qids
                or record.qid in global_qids
                or name in global_names
            ):
                continue
            eligible.append(record)
            global_qids.add(record.qid)
            global_names.add(name)
        if len(eligible) < needed:
            raise ValueError(
                f"{domain} has {len(eligible)} eligible records but requires {needed}"
            )
        ordered = sorted(
            eligible,
            key=lambda row: (
                hashlib.sha256(
                    f"{design.split_seed}:{domain}:{row.qid}".encode()
                ).hexdigest(),
                row.qid,
            ),
        )
        for index, split in enumerate(design.splits):
            start = index * design.candidates_per_domain_per_split
            stop = start + design.candidates_per_domain_per_split
            assigned[split].extend(ordered[start:stop])

    return {
        split: tuple(sorted(rows, key=lambda row: (row.domain, row.qid)))
        for split, rows in assigned.items()
    }


def materialize_development_manifests(
    assigned: Mapping[str, Sequence[SourceRecord]],
    *,
    design: DevelopmentSourceDesign,
    retrieval_date: str,
    query_hashes: Mapping[str, str],
) -> dict[
    str,
    tuple[tuple[CandidateEntity, ...], tuple[ScreeningQuestion, ...]],
]:
    """Create development-only candidate and screening-question manifests."""
    if set(assigned) != set(design.splits):
        raise ValueError("development assignment does not cover registered splits")
    if set(query_hashes) != set(REGISTERED_DOMAINS):
        raise ValueError("development query hashes do not cover registered domains")

    manifests = {}
    for split in design.splits:
        candidates = []
        questions = []
        for record in assigned[split]:
            entity_id = f"development-v6-{split}-{record.domain}-{record.qid.lower()}"
            provenance = (
                f"Wikidata EntityData {record.qid} (CC0), retrieved "
                f"{retrieval_date}; {design.revision}; domain query SHA-256 "
                f"{query_hashes[record.domain]}"
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
                zip(DEVELOPMENT_DOMAIN_FIELDS[record.domain], alias_sets, strict=True),
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
        manifests[split] = (tuple(candidates), tuple(questions))
    return manifests


def audit_development_source(
    manifests: DevelopmentManifests,
    *,
    design: DevelopmentSourceDesign,
    excluded_qids: Set[str] = frozenset(),
) -> dict[str, Any]:
    """Fail closed on imbalance, protected names, or source leakage."""
    if set(manifests) != set(design.splits):
        raise ValueError("development source does not cover registered splits")
    if set(manifests).intersection(_PROTECTED_SPLITS):
        raise ValueError("development source uses a protected split name")

    entity_ids: set[str] = set()
    qids: set[str] = set()
    names: set[str] = set()
    question_ids: set[str] = set()
    semantic_rows = []

    for split in design.splits:
        candidates, questions = manifests[split]
        expected = design.candidates_per_domain_per_split * len(REGISTERED_DOMAINS)
        if len(candidates) != expected:
            raise ValueError(f"development source count is invalid for {split}")
        counts = Counter(candidate.coarse_type for candidate in candidates)
        if counts != Counter(
            {
                domain: design.candidates_per_domain_per_split
                for domain in REGISTERED_DOMAINS
            }
        ):
            raise ValueError(
                f"development source domain balance is invalid for {split}"
            )

        by_qid = {}
        for candidate in candidates:
            normalized_name = _normal_form(candidate.name)
            if candidate.qid in qids:
                raise ValueError("development source has cross-split QID leakage")
            if candidate.entity_id in entity_ids:
                raise ValueError("development source has duplicate entity IDs")
            if normalized_name in names:
                raise ValueError("development source has cross-split name collisions")
            if candidate.qid in excluded_qids:
                raise ValueError("development source overlaps an excluded QID")
            if candidate.split != split:
                raise ValueError("development candidate split is inconsistent")
            if not candidate.entity_id.startswith(f"development-v6-{split}-"):
                raise ValueError("development candidate has a non-development ID")
            entity_ids.add(candidate.entity_id)
            qids.add(candidate.qid)
            names.add(normalized_name)
            by_qid[candidate.qid] = candidate
            semantic_rows.append(asdict(candidate))

        questions_by_qid: dict[str, list[ScreeningQuestion]] = {}
        for question in questions:
            if question.question_id in question_ids:
                raise ValueError("development source has duplicate question IDs")
            question_ids.add(question.question_id)
            questions_by_qid.setdefault(question.qid, []).append(question)
            semantic_rows.append(asdict(question))
        if set(questions_by_qid) != set(by_qid) or any(
            len(rows) != 3 for rows in questions_by_qid.values()
        ):
            raise ValueError(
                f"development source requires three questions per entity in {split}"
            )
        for qid, rows in questions_by_qid.items():
            candidate = by_qid[qid]
            if any(
                row.source_provenance != candidate.source_provenance for row in rows
            ):
                raise ValueError("development question provenance is inconsistent")

    semantic_sha256 = hashlib.sha256(
        json.dumps(
            semantic_rows,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "candidate_count": len(entity_ids),
        "question_count": len(question_ids),
        "split_count": len(manifests),
        "semantic_sha256": semantic_sha256,
    }


def summarize_screening_yield(
    candidates: Sequence[CandidateEntity],
    completions_by_entity: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Summarize development screening without making a confirmatory decision."""
    rows = tuple(candidates)
    expected_ids = {candidate.entity_id for candidate in rows}
    if set(completions_by_entity) != expected_ids:
        raise ValueError("screening completion identities do not match candidates")

    scored = [
        (
            candidate,
            score_screening(
                candidate,
                completions_by_entity[candidate.entity_id],
            ),
        )
        for candidate in rows
    ]
    distribution = Counter(
        f"{sum(result.correct_answers)}_of_3" for _, result in scored
    )
    ordered_distribution = {
        f"{count}_of_3": distribution.get(f"{count}_of_3", 0) for count in range(4)
    }
    qualified = sum(result.qualifies for _, result in scored)
    by_domain = {}
    for domain in REGISTERED_DOMAINS:
        domain_results = [
            result for candidate, result in scored if candidate.coarse_type == domain
        ]
        domain_qualified = sum(result.qualifies for result in domain_results)
        by_domain[domain] = {
            "entity_count": len(domain_results),
            "qualified_count": domain_qualified,
            "qualification_rate": domain_qualified / len(domain_results),
            "qualification_interval": _wilson_interval(
                domain_qualified, len(domain_results)
            ),
        }
    relation_observations: dict[str, list[bool]] = {}
    for candidate, result in scored:
        for field, correct in zip(
            DEVELOPMENT_DOMAIN_FIELDS[candidate.coarse_type],
            result.correct_answers,
            strict=True,
        ):
            relation_observations.setdefault(field.property_id, []).append(correct)
    by_relation = {}
    for property_id, observations in sorted(relation_observations.items()):
        successes = sum(observations)
        by_relation[property_id] = {
            "item_count": len(observations),
            "success_count": successes,
            "success_rate": successes / len(observations),
            "success_interval": _wilson_interval(successes, len(observations)),
        }
    return {
        "entity_count": len(scored),
        "qualified_count": qualified,
        "qualification_rate": qualified / len(scored),
        "qualification_interval": _wilson_interval(qualified, len(scored)),
        "score_distribution": ordered_distribution,
        "by_domain": by_domain,
        "by_relation": by_relation,
        "question_position_success": [
            sum(result.correct_answers[index] for _, result in scored) / len(scored)
            for index in range(3)
        ],
    }


def build_manual_error_audit_packet(
    items: Sequence[Mapping[str, Any]],
    *,
    sample_per_domain: int,
    success_sample_per_domain: int = 0,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Create a deterministic, stratified packet without model outcome labels."""
    if type(sample_per_domain) is not int or sample_per_domain <= 0:
        raise ValueError("manual audit sample per domain must be positive")
    if (
        type(success_sample_per_domain) is not int
        or success_sample_per_domain < 0
    ):
        raise ValueError("manual audit success sample per domain must be nonnegative")
    if type(seed) is not int:
        raise TypeError("manual audit seed must be an integer")
    errors = []
    successes = []
    seen_questions: set[str] = set()
    for item in items:
        required = {
            "question_id",
            "entity_id",
            "qid",
            "domain",
            "prompt",
            "completion",
            "accepted_aliases",
            "is_correct",
        }
        if not required.issubset(item):
            raise ValueError("manual audit item is missing required fields")
        question_id = str(item["question_id"])
        if question_id in seen_questions:
            raise ValueError("manual audit items contain duplicate question IDs")
        seen_questions.add(question_id)
        if item["domain"] not in REGISTERED_DOMAINS:
            raise ValueError("manual audit item has an unregistered domain")
        if type(item["is_correct"]) is not bool:
            raise ValueError("manual audit is_correct must be boolean")
        (successes if item["is_correct"] else errors).append(item)

    selected = []
    for domain in REGISTERED_DOMAINS:
        strata = (
            ("error", errors, sample_per_domain),
            ("success", successes, success_sample_per_domain),
        )
        for stratum, rows, limit in strata:
            domain_rows = [row for row in rows if row["domain"] == domain]
            ordered = sorted(
                domain_rows,
                key=lambda row: (
                    hashlib.sha256(
                        f"{seed}:{domain}:{stratum}:{row['question_id']}".encode()
                    ).hexdigest(),
                    str(row["question_id"]),
                ),
            )
            for row in ordered[:limit]:
                audit_id = hashlib.sha256(
                    f"{seed}:{row['question_id']}".encode()
                ).hexdigest()[:20]
                selected.append(
                    {
                        "audit_id": f"fa-v6-audit-{audit_id}",
                        "question_id": row["question_id"],
                        "entity_id": row["entity_id"],
                        "qid": row["qid"],
                        "domain": row["domain"],
                        "prompt": row["prompt"],
                        "completion": row["completion"],
                        "accepted_aliases": list(row["accepted_aliases"]),
                        "allowed_error_labels": list(ERROR_TAXONOMY),
                    }
                )
    return tuple(sorted(selected, key=lambda row: row["audit_id"]))


def compile_manual_error_audit(
    packet: Sequence[Mapping[str, Any]],
    ratings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require two initial raters and a distinct adjudicator for disagreements."""
    packet_ids = {str(row.get("audit_id", "")) for row in packet}
    if len(packet_ids) != len(packet) or "" in packet_ids:
        raise ValueError("manual audit packet has invalid audit IDs")
    grouped: dict[str, list[Mapping[str, Any]]] = {
        audit_id: [] for audit_id in packet_ids
    }
    for rating in ratings:
        audit_id = str(rating.get("audit_id", ""))
        if audit_id not in grouped:
            raise ValueError("manual audit rating references an unknown audit ID")
        label = rating.get("error_label")
        if label not in ERROR_TAXONOMY:
            raise ValueError("manual audit rating has an invalid error label")
        grouped[audit_id].append(rating)

    decisions = {}
    adjudicated = 0
    for audit_id, audit_ratings in grouped.items():
        initial = [row for row in audit_ratings if row.get("round") == 1]
        final = [row for row in audit_ratings if row.get("round") == 2]
        initial_raters = {str(row.get("rater_id", "")) for row in initial}
        if len(initial) != 2 or len(initial_raters) != 2 or "" in initial_raters:
            raise ValueError("manual audit requires two independent initial raters")
        labels = {str(row["error_label"]) for row in initial}
        if len(labels) == 1:
            if final:
                raise ValueError(
                    "manual audit adjudicator is allowed only on disagreement"
                )
            decisions[audit_id] = next(iter(labels))
            continue
        if len(final) != 1:
            raise ValueError("manual audit disagreement requires one adjudicator")
        adjudicator = final[0]
        adjudicator_id = str(adjudicator.get("rater_id", ""))
        if not adjudicator_id or adjudicator_id in initial_raters:
            raise ValueError("manual audit adjudicator must be independent")
        decisions[audit_id] = str(adjudicator["error_label"])
        adjudicated += 1

    return {
        "item_count": len(packet_ids),
        "adjudicated_count": adjudicated,
        "decision_counts": dict(sorted(Counter(decisions.values()).items())),
        "decisions": dict(sorted(decisions.items())),
    }


def write_development_source(
    output_dir: Path,
    manifests: DevelopmentManifests,
    *,
    design: DevelopmentSourceDesign,
    excluded_qids: Set[str] = frozenset(),
    future_excluded_qids: Set[str] | None = None,
    source_frame_sha256: str | None = None,
) -> dict[str, Any]:
    """Persist immutable Source-v6 manifests and their integrity lineage."""
    audit = audit_development_source(
        manifests,
        design=design,
        excluded_qids=excluded_qids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_qids = {
        candidate.qid for split in design.splits for candidate in manifests[split][0]
    }
    source_v6_qids = sorted(
        selected_qids
        if future_excluded_qids is None
        else set(future_excluded_qids) | selected_qids
    )
    exclusions_path = output_dir / "source_v7_exclusions_v1.json"
    _write_json_immutable(
        exclusions_path,
        {
            "schema_version": 1,
            "kind": "source_v7_exclusions",
            "source_revision": design.revision,
            "excluded_qids": source_v6_qids,
        },
    )
    materialized = {}
    for split in design.splits:
        candidates, questions = manifests[split]
        candidate_path = output_dir / f"candidate_entities_{split}_v1.json"
        question_path = output_dir / f"screening_questions_{split}_v1.json"
        _write_json_immutable(candidate_path, [asdict(row) for row in candidates])
        _write_json_immutable(question_path, [asdict(row) for row in questions])
        materialized[split] = {
            "candidate_manifest": str(candidate_path.relative_to(output_dir)),
            "candidate_sha256": _sha256_file(candidate_path),
            "question_manifest": str(question_path.relative_to(output_dir)),
            "question_sha256": _sha256_file(question_path),
        }

    snapshot = {
        "schema_version": 1,
        "source_revision": design.revision,
        "split_seed": design.split_seed,
        "candidates_per_domain_per_split": (design.candidates_per_domain_per_split),
        "splits": list(design.splits),
        "excluded_qids": sorted(excluded_qids),
        "source_v7_exclusions": exclusions_path.name,
        "source_v7_exclusions_sha256": _sha256_file(exclusions_path),
        **(
            {"source_frame_sha256": source_frame_sha256}
            if source_frame_sha256 is not None
            else {}
        ),
        "audit": audit,
        "materialized_files": materialized,
        "claim_scope": "instrument_development_only",
    }
    snapshot_path = output_dir / "source_snapshot_v1.json"
    _write_json_immutable(snapshot_path, snapshot)
    snapshot_sha256 = _sha256_file(snapshot_path)
    integrity = {
        "schema_version": 1,
        "source_revision": design.revision,
        "source_snapshot": snapshot_path.name,
        "source_snapshot_sha256": snapshot_sha256,
        "materialized_files": materialized,
        "source_v7_exclusions": exclusions_path.name,
        "source_v7_exclusions_sha256": _sha256_file(exclusions_path),
    }
    integrity_path = output_dir / "source_integrity_v1.json"
    _write_json_immutable(integrity_path, integrity)
    return {
        "source_snapshot": snapshot_path,
        "source_snapshot_sha256": snapshot_sha256,
        "source_integrity": integrity_path,
        "source_integrity_sha256": _sha256_file(integrity_path),
        "source_v7_exclusions": exclusions_path,
        "audit": audit,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the open Source-v6 development corpus."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--source-frame", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument(
        "--exclude-candidates",
        type=Path,
        action="append",
        default=[],
    )
    materialize.add_argument("--split-seed", type=int, default=20260725)
    materialize.add_argument(
        "--candidates-per-domain-per-split",
        type=int,
        default=128,
    )
    args = parser.parse_args(argv)

    if args.command == "materialize":
        frame, records = _load_source_frame(args.source_frame)
        design = DevelopmentSourceDesign(
            revision=_FRAME_TO_SOURCE_REVISION[frame["source_revision"]],
            split_seed=args.split_seed,
            candidates_per_domain_per_split=(args.candidates_per_domain_per_split),
        )
        excluded_qids = _load_candidate_qids(args.exclude_candidates)
        manifests = materialize_development_manifests(
            assign_development_pools(
                records,
                design,
                excluded_qids=excluded_qids,
            ),
            design=design,
            retrieval_date=frame["retrieval_date"],
            query_hashes=frame["query_sha256s"],
        )
        result = write_development_source(
            args.output_dir,
            manifests,
            design=design,
            excluded_qids=excluded_qids,
            source_frame_sha256=_sha256_file(args.source_frame),
        )
        print(
            json.dumps(
                {
                    "status": "materialized",
                    **result["audit"],
                    "source_snapshot": str(result["source_snapshot"]),
                    "source_snapshot_sha256": result["source_snapshot_sha256"],
                    "source_integrity": str(result["source_integrity"]),
                    "source_integrity_sha256": result["source_integrity_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable development source command")


def _load_source_frame(
    path: Path,
) -> tuple[dict[str, Any], dict[str, tuple[SourceRecord, ...]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("development source frame has an invalid schema")
    if payload.get("source_revision") not in _FRAME_TO_SOURCE_REVISION:
        raise ValueError("development source frame revision is unsupported")
    payload_without_hash = dict(payload)
    stored_payload_sha256 = payload_without_hash.pop("frame_payload_sha256", None)
    matchability = payload.get("matchability_audit")
    if (
        not isinstance(payload.get("retrieval_date"), str)
        or set(payload.get("query_sha256s", {})) != set(REGISTERED_DOMAINS)
        or set(payload.get("records_by_domain", {})) != set(REGISTERED_DOMAINS)
        or not isinstance(matchability, Mapping)
        or not isinstance(matchability.get("policy_sha256"), str)
        or len(matchability["policy_sha256"]) != 64
        or stored_payload_sha256
        != hashlib.sha256(
            json.dumps(
                payload_without_hash,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    ):
        raise ValueError("development source frame is incomplete")
    records_by_domain = {}
    for domain in REGISTERED_DOMAINS:
        rows = payload["records_by_domain"][domain]
        if not isinstance(rows, list):
            raise ValueError("development source frame records must be arrays")
        if len(rows) != payload.get("required_per_domain"):
            raise ValueError("development source frame domain count is incomplete")
        records = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("development source frame record must be an object")
            property_values = tuple(
                (str(property_id), tuple(str(alias) for alias in aliases))
                for property_id, aliases in row["property_values"]
            )
            records.append(
                SourceRecord(
                    qid=str(row["qid"]),
                    label=str(row["label"]),
                    domain=str(row["domain"]),
                    sitelinks=int(row["sitelinks"]),
                    source_rank=int(row["source_rank"]),
                    property_values=property_values,
                )
            )
        records_by_domain[domain] = tuple(records)
    return payload, records_by_domain


def _load_candidate_qids(paths: Sequence[Path]) -> frozenset[str]:
    qids = set()
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("excluded candidate manifest must be an array")
        for row in rows:
            if isinstance(row, Mapping) and isinstance(row.get("qid"), str):
                qids.add(row["qid"])
    return frozenset(qids)


def _wilson_interval(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires observations")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return {"lower": center - margin, "upper": center + margin}


def _normal_form(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _write_json_immutable(path: Path, value: Any) -> None:
    normalized = json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) == normalized:
            return
        raise FileExistsError(
            f"refusing to overwrite a non-identical development source file: {path}"
        )
    serialized = (
        json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"stale development source temporary file requires audit: {temporary}"
        )
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
