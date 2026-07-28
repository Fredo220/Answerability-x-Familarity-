"""R11 broad, relation-bank source construction for open instrument work."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajectory_extractor.fa_confirmatory_source import (
    _PREFIXES,
    REGISTERED_DOMAINS,
    ScreeningField,
    _claim_entity_qids,
    _claim_value_aliases,
    _eligible_label,
    _english_label,
    _fetch_entities,
    _post_sparql,
)

_ENTITY_URI = re.compile(r"^http://www\.wikidata\.org/entity/(Q[1-9][0-9]*)$")


@dataclass(frozen=True)
class RankedEntity:
    qid: str
    sitelinks: int


R11_RELATION_FIELDS = {
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
        ScreeningField(
            "P569",
            "In which year was {name} born? Answer with only the year.",
            value_kind="year",
        ),
        ScreeningField(
            "P570",
            "In which year did {name} die? Answer with only the year.",
            value_kind="year",
        ),
    ),
    "place": (
        ScreeningField(
            "P36",
            "What is the capital of {name}? Answer with only the place name.",
        ),
        ScreeningField(
            "P30",
            "On which continent is {name} located? "
            "Answer with only the continent name.",
        ),
        ScreeningField(
            "P37",
            "What is an official language of {name}? "
            "Answer with only one language.",
        ),
        ScreeningField(
            "P38",
            "What currency is used in {name}? Answer with only one currency.",
        ),
        ScreeningField(
            "P47",
            "Which country shares a land border with {name}? "
            "Answer with only one country name.",
        ),
    ),
    "organization": (
        ScreeningField(
            "P17",
            "Which country is {name} associated with? "
            "Answer with only the country name.",
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
        ScreeningField(
            "P452",
            "What industry is {name} part of? Answer with only one industry.",
        ),
        ScreeningField(
            "P112",
            "Who founded {name}? Answer with only one founder name.",
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
            "What is the country of origin of {name}? "
            "Answer with only the country name.",
        ),
        ScreeningField(
            "P136",
            "What is the genre of {name}? Answer with only one genre.",
        ),
        ScreeningField(
            "P364",
            "What was the original language of {name}? Answer with only one language.",
        ),
    ),
}


def build_r11_domain_query(domain: str, *, limit: int) -> str:
    """Return a relation-blind ranked entity query for R11 development."""
    if domain not in R11_RELATION_FIELDS:
        raise ValueError(f"unregistered domain: {domain}")
    if type(limit) is not int or limit <= 0:
        raise ValueError("query limit must be a positive integer")
    type_pattern = {
        "person": "wdt:P31 wd:Q5",
        "place": "wdt:P31 wd:Q6256",
        "organization": "wdt:P31/wdt:P279* wd:Q43229",
        "creative_work": "wdt:P31/wdt:P279* wd:Q11424",
    }[domain]
    return (
        f"{_PREFIXES}\n"
        "SELECT ?item ?sitelinks\n"
        "WHERE {\n"
        f"  ?item {type_pattern};\n"
        "        wikibase:sitelinks ?sitelinks.\n"
        "  FILTER(?sitelinks >= 10)\n"
        "}\n"
        "ORDER BY DESC(?sitelinks) ?item\n"
        f"LIMIT {limit}"
    )


def parse_r11_ranked_entities(payload: Mapping[str, Any]) -> tuple[RankedEntity, ...]:
    """Parse unique ranked QIDs from a QLever response."""
    bindings = payload.get("results", {}).get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("QLever response is missing result bindings")
    output = []
    seen = set()
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("QLever binding must be an object")
        item = binding.get("item")
        sitelinks = binding.get("sitelinks")
        if not isinstance(item, Mapping) or not isinstance(sitelinks, Mapping):
            raise ValueError("QLever binding is incomplete")
        match = _ENTITY_URI.fullmatch(str(item.get("value", "")))
        if match is None:
            raise ValueError("QLever item is not a canonical entity URI")
        qid = match.group(1)
        if qid not in seen:
            output.append(
                RankedEntity(
                    qid=qid,
                    sitelinks=int(str(sitelinks.get("value", ""))),
                )
            )
            seen.add(qid)
    return tuple(output)


def materialize_r11_rows(
    ranked_by_domain: Mapping[str, Sequence[RankedEntity]],
    entities: Mapping[str, Mapping[str, Any]],
    *,
    development_candidates_per_domain: int,
    validation_candidates_per_domain: int,
    split_seed: int,
    excluded_qids: Set[str],
    exclusion_manifest_sha256: str,
    exclusion_parent_sha256s: Sequence[str],
    retrieval_date: str,
) -> tuple[dict[str, Any], ...]:
    """Create balanced, entity-disjoint screening rows from audited facts."""
    if set(ranked_by_domain) != set(REGISTERED_DOMAINS):
        raise ValueError("ranked source must cover exactly the registered domains")
    if (
        type(development_candidates_per_domain) is not int
        or development_candidates_per_domain <= 0
        or type(validation_candidates_per_domain) is not int
        or validation_candidates_per_domain <= 0
        or type(split_seed) is not int
    ):
        raise ValueError("R11 split candidate counts must be positive")
    if not retrieval_date:
        raise ValueError("retrieval date must be nonempty")
    if (
        re.fullmatch(r"[0-9a-f]{64}", exclusion_manifest_sha256) is None
        or not exclusion_parent_sha256s
        or any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in exclusion_parent_sha256s
        )
    ):
        raise ValueError("R11 requires a frozen predecessor exclusion manifest")

    rows = []
    selected_qids = set()
    required = development_candidates_per_domain + validation_candidates_per_domain
    for domain in REGISTERED_DOMAINS:
        eligible = []
        ranked = sorted(
            ranked_by_domain[domain],
            key=lambda row: (-row.sitelinks, row.qid),
        )
        for ranked_entity in ranked:
            if (
                ranked_entity.qid in excluded_qids
                or ranked_entity.qid in selected_qids
            ):
                continue
            entity = entities.get(ranked_entity.qid)
            if not isinstance(entity, Mapping):
                continue
            label = _english_label(entity)
            if not _eligible_label(label):
                continue
            facts = []
            for field in R11_RELATION_FIELDS[domain]:
                aliases = _claim_value_aliases(field, entity, entities)
                if aliases:
                    facts.append((field, aliases))
            if len(facts) == len(R11_RELATION_FIELDS[domain]):
                eligible.append((ranked_entity, label, facts))
            if len(eligible) == required:
                break
        if len(eligible) != required:
            raise ValueError(
                f"R11 {domain} has {len(eligible)} eligible candidates; "
                f"requires {required}"
            )

        assigned = sorted(
            eligible,
            key=lambda row: (
                hashlib.sha256(
                    f"{split_seed}:{domain}:{row[0].qid}".encode()
                ).hexdigest(),
                row[0].qid,
            ),
        )
        for index, (ranked_entity, label, facts) in enumerate(assigned):
            split = (
                "instrument_development"
                if index < development_candidates_per_domain
                else "construction_validation"
            )
            selected_qids.add(ranked_entity.qid)
            entity_id = f"r11-{split}-{domain}-{ranked_entity.qid.lower()}"
            for field, aliases in facts:
                rows.append(
                    {
                        "schema_version": 1,
                        "kind": "fa_r11_screening_prompt",
                        "split": split,
                        "domain": domain,
                        "entity_id": entity_id,
                        "qid": ranked_entity.qid,
                        "entity_name": label,
                        "sitelinks": ranked_entity.sitelinks,
                        "relation_id": field.property_id,
                        "prompt": field.prompt_template.format(name=label),
                        "accepted_aliases": list(aliases),
                        "source_provenance": (
                            f"Wikidata EntityData {ranked_entity.qid} (CC0), "
                            f"retrieved {retrieval_date}; fa-r11"
                        ),
                    }
                )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                str(row["split"]),
                str(row["domain"]),
                -int(row["sitelinks"]),
                str(row["qid"]),
                str(row["relation_id"]),
            ),
        )
    )


def build_r11_source(
    *,
    output_dir: Path,
    query_limit: int,
    development_candidates_per_domain: int,
    validation_candidates_per_domain: int,
    split_seed: int,
    excluded_qids: Set[str],
    exclusion_manifest_sha256: str,
    exclusion_parent_sha256s: Sequence[str],
    retrieval_date: str,
) -> dict[str, Any]:
    """Fetch, materialize, and hash-bind the R11 relation-bank source."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ranked_by_domain = {}
    query_sha256s = {}
    candidate_qids = set()
    for domain in REGISTERED_DOMAINS:
        query = build_r11_domain_query(domain, limit=query_limit)
        query_sha256 = _sha256_bytes(query.encode("utf-8"))
        query_sha256s[domain] = query_sha256
        cache_path = cache_dir / f"qlever_{domain}_{query_sha256[:16]}.json"
        if cache_path.exists():
            payload = _read_json(cache_path)
        else:
            payload = _post_sparql(query)
            _write_json_immutable(cache_path, payload)
        ranked = parse_r11_ranked_entities(payload)
        ranked_by_domain[domain] = ranked
        candidate_qids.update(row.qid for row in ranked)

    candidate_entities = _fetch_entities(
        tuple(sorted(candidate_qids)),
        cache_dir=cache_dir / "candidate_batches",
        props="labels|aliases|claims",
    )
    linked_qids = set()
    for domain in REGISTERED_DOMAINS:
        for ranked_entity in ranked_by_domain[domain]:
            entity = candidate_entities.get(ranked_entity.qid)
            for field in R11_RELATION_FIELDS[domain]:
                linked_qids.update(
                    _claim_entity_qids(entity, field.property_id)
                )
    linked_entities = _fetch_entities(
        tuple(sorted(linked_qids.difference(candidate_entities))),
        cache_dir=cache_dir / "linked_batches",
        props="labels|aliases",
    )
    entities = {**candidate_entities, **linked_entities}
    rows = materialize_r11_rows(
        ranked_by_domain,
        entities,
        development_candidates_per_domain=development_candidates_per_domain,
        validation_candidates_per_domain=validation_candidates_per_domain,
        split_seed=split_seed,
        excluded_qids=excluded_qids,
        exclusion_manifest_sha256=exclusion_manifest_sha256,
        exclusion_parent_sha256s=exclusion_parent_sha256s,
        retrieval_date=retrieval_date,
    )
    rows_path = output_dir / "screening_prompts_v1.jsonl"
    rows_bytes = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
        for row in rows
    ).encode("utf-8")
    _write_bytes_immutable(rows_path, rows_bytes)
    manifest = {
        "schema_version": 1,
        "kind": "fa_r11_source_manifest",
        "source_revision": "fa-development-source-v6-r11",
        "claim_scope": "open_instrument_development_only",
        "retrieval_date": retrieval_date,
        "query_limit": query_limit,
        "query_sha256s": query_sha256s,
        "relation_bank": {
            domain: [
                field.property_id for field in R11_RELATION_FIELDS[domain]
            ]
            for domain in REGISTERED_DOMAINS
        },
        "development_candidates_per_domain": development_candidates_per_domain,
        "validation_candidates_per_domain": validation_candidates_per_domain,
        "split_seed": split_seed,
        "excluded_qid_count": len(excluded_qids),
        "excluded_qids_sha256": _canonical_sha256(sorted(excluded_qids)),
        "exclusion_manifest_sha256": exclusion_manifest_sha256,
        "exclusion_parent_sha256s": sorted(exclusion_parent_sha256s),
        "row_count": len(rows),
        "rows_file": rows_path.name,
        "rows_sha256": _sha256_bytes(rows_bytes),
    }
    manifest_path = output_dir / "source_manifest_v1.json"
    _write_json_immutable(manifest_path, manifest)
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "rows_path": str(rows_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query-limit", type=int, default=1000)
    parser.add_argument("--retrieval-date", required=True)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    config = _read_json(args.config)
    configured_bank = config.get("relation_bank")
    expected_bank = {
        domain: [field.property_id for field in R11_RELATION_FIELDS[domain]]
        for domain in REGISTERED_DOMAINS
    }
    if configured_bank != expected_bank:
        raise ValueError("R11 config relation bank does not match implementation")
    exclusion_manifest = _read_json(args.exclusion_manifest)
    excluded_rows = exclusion_manifest.get("qids")
    parents = exclusion_manifest.get("parents")
    if (
        exclusion_manifest.get("kind") != "fa_r11_predecessor_exclusions"
        or not isinstance(excluded_rows, list)
        or not isinstance(parents, list)
        or not parents
        or exclusion_manifest.get("qids_sha256")
        != _canonical_sha256(sorted(excluded_rows))
    ):
        raise ValueError("R11 predecessor exclusion manifest is invalid")
    parent_sha256s = tuple(
        str(row.get("sha256", "")) for row in parents if isinstance(row, Mapping)
    )
    if len(parent_sha256s) != len(parents):
        raise ValueError("R11 predecessor exclusion parents are invalid")
    excluded = {str(qid) for qid in excluded_rows}
    result = build_r11_source(
        output_dir=args.output_dir,
        query_limit=args.query_limit,
        development_candidates_per_domain=int(
            config["development_candidates_per_domain"]
        ),
        validation_candidates_per_domain=int(
            config["validation_candidates_per_domain"]
        ),
        split_seed=int(config["split_seed"]),
        excluded_qids=excluded,
        exclusion_manifest_sha256=_sha256_file(args.exclusion_manifest),
        exclusion_parent_sha256s=parent_sha256s,
        retrieval_date=args.retrieval_date,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _canonical_sha256(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact must be an object")
    return payload


def _write_json_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_bytes_immutable(path, encoded)


def _write_bytes_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
