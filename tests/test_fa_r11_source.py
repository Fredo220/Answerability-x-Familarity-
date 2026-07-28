from trajectory_extractor.fa_r11_source import (
    R11_RELATION_FIELDS,
    RankedEntity,
    build_r11_domain_query,
    materialize_r11_rows,
)


def _linked(qid: str, label: str) -> tuple[str, dict[str, object]]:
    return (
        qid,
        {
            "labels": {"en": {"value": label}},
            "aliases": {"en": []},
        },
    )


def _claim(qid: str) -> dict[str, object]:
    return {
        "rank": "normal",
        "mainsnak": {"datavalue": {"value": {"id": qid}}},
    }


def _year(value: str) -> dict[str, object]:
    return {
        "rank": "normal",
        "mainsnak": {"datavalue": {"value": {"time": f"+{value}-01-01T00:00:00Z"}}},
    }


def _entity(label: str, domain: str, index: int) -> dict[str, object]:
    fields = R11_RELATION_FIELDS[domain]
    claims = {}
    for offset, field in enumerate(fields):
        if field.value_kind == "year":
            claims[field.property_id] = [_year(str(1900 + index + offset))]
        else:
            claims[field.property_id] = [_claim(f"Q{9000 + index * 10 + offset}")]
    return {
        "labels": {"en": {"value": label}},
        "aliases": {"en": []},
        "claims": claims,
    }


def test_domain_query_is_ranked_and_relation_blind() -> None:
    query = build_r11_domain_query("person", limit=200)

    assert "wdt:P31 wd:Q5" in query
    assert "ORDER BY DESC(?sitelinks) ?item" in query
    assert "P569" not in query


def test_materialize_rows_is_balanced_and_entity_disjoint() -> None:
    ranked = {}
    entities = {}
    linked = {}
    for domain_index, domain in enumerate(R11_RELATION_FIELDS):
        domain_ranked = []
        for index in range(4):
            qid = f"Q{100 + domain_index * 10 + index}"
            domain_ranked.append(RankedEntity(qid=qid, sitelinks=100 - index))
            entities[qid] = _entity(
                f"{domain.replace('_', ' ').title()} Name {index}",
                domain,
                index,
            )
            for offset, field in enumerate(R11_RELATION_FIELDS[domain]):
                if field.value_kind != "year":
                    linked.update(
                        [
                            _linked(
                                f"Q{9000 + index * 10 + offset}",
                                f"Value {domain_index} {index} {offset}",
                            )
                        ]
                    )
        ranked[domain] = tuple(domain_ranked)
    entities.update(linked)

    rows = materialize_r11_rows(
        ranked,
        entities,
        development_candidates_per_domain=2,
        validation_candidates_per_domain=2,
        split_seed=20260728,
        excluded_qids=frozenset(),
        exclusion_manifest_sha256="a" * 64,
        exclusion_parent_sha256s=("b" * 64,),
        retrieval_date="2026-07-28",
    )

    development_qids = {
        row["qid"] for row in rows if row["split"] == "instrument_development"
    }
    validation_qids = {
        row["qid"] for row in rows if row["split"] == "construction_validation"
    }
    assert development_qids.isdisjoint(validation_qids)
    for split in ("instrument_development", "construction_validation"):
        for domain in R11_RELATION_FIELDS:
            entity_ids = {
                row["entity_id"]
                for row in rows
                if row["split"] == split and row["domain"] == domain
            }
            assert len(entity_ids) == 2
            assert all(
                len(
                    {
                        row["relation_id"]
                        for row in rows
                        if row["entity_id"] == entity_id
                    }
                )
                == 5
                for entity_id in entity_ids
            )


def test_materialization_excludes_previously_seen_qids() -> None:
    ranked = {
        domain: tuple(
            RankedEntity(qid=f"Q{1000 + domain_index * 10 + index}", sitelinks=10)
            for index in range(3)
        )
        for domain_index, domain in enumerate(R11_RELATION_FIELDS)
    }
    entities = {}
    linked = {}
    for domain_index, (domain, domain_rows) in enumerate(ranked.items()):
        for index, ranked_entity in enumerate(domain_rows):
            entities[ranked_entity.qid] = _entity(
                f"{domain.replace('_', ' ').title()} Entity {index}",
                domain,
                index,
            )
            for offset, field in enumerate(R11_RELATION_FIELDS[domain]):
                if field.value_kind != "year":
                    linked.update(
                        [
                            _linked(
                                f"Q{9000 + index * 10 + offset}",
                                f"Value {domain_index} {index} {offset}",
                            )
                        ]
                    )
    entities.update(linked)
    excluded = {next(iter(ranked.values()))[0].qid}

    rows = materialize_r11_rows(
        ranked,
        entities,
        development_candidates_per_domain=1,
        validation_candidates_per_domain=1,
        split_seed=20260728,
        excluded_qids=frozenset(excluded),
        exclusion_manifest_sha256="a" * 64,
        exclusion_parent_sha256s=("b" * 64,),
        retrieval_date="2026-07-28",
    )

    assert excluded.isdisjoint({row["qid"] for row in rows})
