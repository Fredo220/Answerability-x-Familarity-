"""Bounded open source-frame construction for FA Source-v6 development."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_confirmatory_source import (
    DOMAIN_FIELDS,
    REGISTERED_DOMAINS,
    SourceRecord,
    _fetch_entities,
    _post_sparql,
    build_domain_query,
    build_source_records_from_ranked_values,
    exclude_cross_domain_source_collisions,
    parse_qlever_candidates,
)
from trajectory_extractor.fa_development_source import (
    filter_development_matchable_records,
)
from trajectory_extractor.fa_runtime import load_pinned_tokenizer

SOURCE_FRAME_REVISION = "fa-development-source-frame-v6"
_ENTITY_URI = re.compile(r"^http://www\.wikidata\.org/entity/(Q[1-9][0-9]*)$")


def build_development_frame(
    *,
    output_dir: Path,
    tokenizer: Any,
    tokenizer_revision: str,
    query_limit: int,
    required_per_domain: int,
    excluded_qids: frozenset[str],
    retrieval_date: str,
    model_id: str = "google/gemma-2-2b-it",
    chat_template_sha256: str | None = None,
) -> dict[str, Any]:
    """Build or replay one immutable, bounded, development-only source frame."""
    _validate_design(
        tokenizer_revision=tokenizer_revision,
        query_limit=query_limit,
        required_per_domain=required_per_domain,
        excluded_qids=excluded_qids,
        retrieval_date=retrieval_date,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "development_source_frame_v1.json"
    requested_design = _design_payload(
        tokenizer_revision=tokenizer_revision,
        query_limit=query_limit,
        required_per_domain=required_per_domain,
        excluded_qids=excluded_qids,
        retrieval_date=retrieval_date,
        model_id=model_id,
        chat_template_sha256=chat_template_sha256,
    )
    if frame_path.exists():
        frame = _read_json_object(frame_path)
        if frame.get("design") != requested_design:
            raise ValueError(
                "existing development frame does not match the requested design"
            )
        provenance = frame.get("provenance")
        if not isinstance(provenance, dict) or frame.get(
            "provenance_sha256"
        ) != _canonical_sha256(provenance):
            raise ValueError("existing development frame provenance hash is invalid")
        stored_payload_sha256 = frame.pop("frame_payload_sha256", None)
        if stored_payload_sha256 != _canonical_sha256(frame):
            raise ValueError("existing development frame payload hash is invalid")
        frame["frame_payload_sha256"] = stored_payload_sha256
        return _result(frame_path, frame)

    cache_dir = output_dir / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ranked_by_domain = {}
    query_hashes = {}
    cache_hashes = {}
    entity_qids = set()
    for domain in REGISTERED_DOMAINS:
        query = build_domain_query(domain, limit=query_limit)
        query_sha256 = _sha256_bytes(query.encode("utf-8"))
        query_hashes[domain] = query_sha256
        cache_path = cache_dir / f"qlever_{domain}_{query_sha256[:16]}.json"
        if cache_path.exists():
            payload = _read_json_object(cache_path)
        else:
            payload = _post_sparql(query)
            _write_json_immutable(cache_path, payload)
        cache_hashes[str(cache_path.relative_to(output_dir))] = _sha256_file(cache_path)
        ranked = parse_qlever_candidates(payload)
        ranked_by_domain[domain] = ranked
        entity_qids.update(row.qid for row in ranked)
        for row in ranked:
            for field, raw_value in zip(
                DOMAIN_FIELDS[domain],
                row.raw_values,
                strict=True,
            ):
                if field.value_kind != "entity":
                    continue
                for value in raw_value.split("|"):
                    match = _ENTITY_URI.fullmatch(value)
                    if match is not None:
                        entity_qids.add(match.group(1))

    entities = _fetch_entities(
        tuple(sorted(entity_qids)),
        cache_dir=cache_dir / "entity_batches",
        props="labels|aliases",
    )
    entity_cache = cache_dir / "entity_records_v1.json"
    _write_json_immutable(entity_cache, entities)
    cache_hashes[str(entity_cache.relative_to(output_dir))] = _sha256_file(entity_cache)
    for path in sorted((cache_dir / "entity_batches").glob("*.json")):
        cache_hashes[str(path.relative_to(output_dir))] = _sha256_file(path)

    raw_records = {
        domain: build_source_records_from_ranked_values(
            domain,
            ranked_by_domain[domain],
            entities,
            excluded_qids=excluded_qids,
        )
        for domain in REGISTERED_DOMAINS
    }
    collision_free, ambiguous_qids, ambiguous_names = (
        exclude_cross_domain_source_collisions(raw_records)
    )
    matchable, matchability_audit = filter_development_matchable_records(
        collision_free,
        tokenizer,
        required_per_domain=required_per_domain,
    )
    selected: dict[str, tuple[SourceRecord, ...]] = {
        domain: tuple(
            sorted(matchable[domain], key=lambda row: (row.source_rank, row.qid))[
                :required_per_domain
            ]
        )
        for domain in REGISTERED_DOMAINS
    }
    selected_counts = {domain: len(selected[domain]) for domain in REGISTERED_DOMAINS}
    expected_counts = {domain: required_per_domain for domain in REGISTERED_DOMAINS}
    if selected_counts != expected_counts:
        raise ValueError(
            f"development frame exact domain gate failed: {selected_counts}"
        )

    provenance = {
        "source_revision": SOURCE_FRAME_REVISION,
        "retrieval_date": retrieval_date,
        "query_sha256s": query_hashes,
        "raw_cache_sha256s": dict(sorted(cache_hashes.items())),
        "tokenizer_revision": tokenizer_revision,
        "model_id": model_id,
        "chat_template_sha256": chat_template_sha256,
        "matchability_policy_sha256": matchability_audit["policy_sha256"],
        "code_sha256s": _behavior_code_hashes(),
    }
    frame = {
        "schema_version": 1,
        "source_revision": SOURCE_FRAME_REVISION,
        "claim_scope": "open_instrument_development_only",
        "design": requested_design,
        "retrieval_date": retrieval_date,
        "query_limit": query_limit,
        "required_per_domain": required_per_domain,
        "excluded_source_v5_qids": sorted(excluded_qids),
        "ambiguous_cross_domain_qids": sorted(ambiguous_qids),
        "ambiguous_cross_domain_names": sorted(ambiguous_names),
        "query_sha256s": query_hashes,
        "raw_eligible_counts": {
            domain: len(raw_records[domain]) for domain in REGISTERED_DOMAINS
        },
        "collision_free_counts": {
            domain: len(collision_free[domain]) for domain in REGISTERED_DOMAINS
        },
        "matchability_audit": matchability_audit,
        "records_by_domain": {
            domain: [asdict(record) for record in selected[domain]]
            for domain in REGISTERED_DOMAINS
        },
        "provenance": provenance,
        "provenance_sha256": _canonical_sha256(provenance),
    }
    frame["frame_payload_sha256"] = _canonical_sha256(frame)
    _write_json_immutable(frame_path, frame)
    return _result(frame_path, frame)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the bounded open FA Source-v6 source frame."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query-limit", type=int, default=1200)
    parser.add_argument("--required-per-domain", type=int, required=True)
    parser.add_argument("--retrieval-date", default=date.today().isoformat())
    parser.add_argument(
        "--exclude-candidates",
        type=Path,
        action="append",
        default=[],
    )
    args = parser.parse_args(argv)

    config = FAConfig.from_json(args.config)
    prepared = load_pinned_tokenizer(config)
    result = build_development_frame(
        output_dir=args.output_dir,
        tokenizer=prepared.tokenizer,
        tokenizer_revision=config.tokenizer_revision,
        query_limit=args.query_limit,
        required_per_domain=args.required_per_domain,
        excluded_qids=_load_excluded_qids(args.exclude_candidates),
        retrieval_date=args.retrieval_date,
        model_id=config.model_id,
        chat_template_sha256=prepared.chat_template_sha256,
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "source_frame": str(result["source_frame"]),
                "source_frame_sha256": result["source_frame_sha256"],
                "counts_by_domain": result["counts_by_domain"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_design(
    *,
    tokenizer_revision: str,
    query_limit: int,
    required_per_domain: int,
    excluded_qids: frozenset[str],
    retrieval_date: str,
) -> None:
    if type(query_limit) is not int or query_limit <= 0:
        raise ValueError("query_limit must be a positive integer")
    if type(required_per_domain) is not int or required_per_domain <= 0:
        raise ValueError("required_per_domain must be a positive integer")
    if not tokenizer_revision:
        raise ValueError("tokenizer_revision must be pinned")
    if not retrieval_date:
        raise ValueError("retrieval_date must be nonempty")
    if not isinstance(excluded_qids, frozenset):
        raise TypeError("excluded_qids must be a frozenset")


def _behavior_code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "fa_development_frame.py",
        "fa_development_source.py",
        "fa_confirmatory_source.py",
        "fa_confirmatory_synthetics.py",
        "fa_entities.py",
    )
    return {name: _sha256_file(package / name) for name in names}


def _design_payload(
    *,
    tokenizer_revision: str,
    query_limit: int,
    required_per_domain: int,
    excluded_qids: frozenset[str],
    retrieval_date: str,
    model_id: str,
    chat_template_sha256: str | None,
) -> dict[str, Any]:
    return {
        "query_limit": query_limit,
        "required_per_domain": required_per_domain,
        "excluded_source_v5_qids": sorted(excluded_qids),
        "retrieval_date": retrieval_date,
        "model_id": model_id,
        "tokenizer_revision": tokenizer_revision,
        "chat_template_sha256": chat_template_sha256,
    }


def _load_excluded_qids(paths: list[Path]) -> frozenset[str]:
    qids = set()
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"excluded candidate manifest must be a list: {path}")
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("qid"), str):
                qids.add(row["qid"])
    return frozenset(qids)


def _result(frame_path: Path, frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_frame": frame_path,
        "source_frame_sha256": _sha256_file(frame_path),
        "counts_by_domain": {
            domain: len(frame["records_by_domain"][domain])
            for domain in REGISTERED_DOMAINS
        },
        "provenance_sha256": frame["provenance_sha256"],
    }


def _write_json_immutable(path: Path, value: Any) -> None:
    normalized = json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    if path.exists():
        if _read_json_object(path) == normalized:
            return
        raise FileExistsError(f"refusing to overwrite immutable JSON: {path}")
    serialized = (
        json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale immutable temporary file: {temporary}")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"cached JSON payload must be an object: {path}")
    return payload


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
