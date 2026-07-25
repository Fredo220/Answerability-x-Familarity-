from __future__ import annotations

import json

import pytest

from trajectory_extractor.fa_confirmatory_source import REGISTERED_DOMAINS
from trajectory_extractor.fa_development_frame import (
    build_development_domain_query,
    build_development_frame,
)


class _WordTokenizer:
    name_or_path = "pinned-test-tokenizer"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def _qlever_payload(domain: str, *, include_collision: bool = False):
    domain_offset = REGISTERED_DOMAINS.index(domain) * 1_000
    qids = [f"Q{10_000 + domain_offset + index}" for index in range(1, 4)]
    if include_collision:
        qids.append("Q99999")
    bindings = []
    for index, qid in enumerate(qids, start=1):
        values = []
        for slot in range(1, 4):
            value_qid = 200_000 + domain_offset + index * 10 + slot
            values.append(f"http://www.wikidata.org/entity/Q{value_qid}")
        if domain in {"organization", "creative_work"}:
            values[2 if domain == "organization" else 1] = "+2001-01-01T00:00:00Z"
        bindings.append(
            {
                "item": {
                    "type": "uri",
                    "value": f"http://www.wikidata.org/entity/{qid}",
                },
                "sitelinks": {"type": "literal", "value": str(500 - index)},
                **{
                    f"value_{slot}": {"type": "literal", "value": value}
                    for slot, value in enumerate(values, start=1)
                },
            }
        )
    return {"results": {"bindings": bindings}}


def _entities_for_payloads(payloads):
    entities = {}
    for domain, payload in payloads.items():
        for binding in payload["results"]["bindings"]:
            qid = binding["item"]["value"].rsplit("/", 1)[-1]
            domain_label = domain.replace("_", " ").title()
            entities[qid] = {
                "labels": {"en": {"value": f"{domain_label} Entity {qid}"}},
                "aliases": {"en": []},
            }
            for slot in range(1, 4):
                value = binding[f"value_{slot}"]["value"]
                if "/Q" not in value:
                    continue
                linked_qid = value.rsplit("/", 1)[-1]
                entities[linked_qid] = {
                    "labels": {"en": {"value": f"Value {linked_qid}"}},
                    "aliases": {"en": []},
                }
    return entities


def _install_remote_fakes(monkeypatch):
    payloads = {
        domain: _qlever_payload(
            domain,
            include_collision=domain in {"person", "place"},
        )
        for domain in REGISTERED_DOMAINS
    }
    entities = _entities_for_payloads(payloads)
    queries = []

    def fake_post(query):
        queries.append(query)
        for domain, marker in {
            "person": "wd:Q5;",
            "place": "wd:Q515;",
            "organization": "wd:Q43229;",
            "creative_work": "wd:Q11424;",
        }.items():
            if marker in query:
                return payloads[domain]
        raise AssertionError("unrecognized domain query")

    def fake_fetch(qids, *, cache_dir=None, props="labels|aliases"):
        del cache_dir, props
        return {qid: entities[qid] for qid in qids if qid in entities}

    monkeypatch.setattr(
        "trajectory_extractor.fa_development_frame._post_sparql",
        fake_post,
    )
    monkeypatch.setattr(
        "trajectory_extractor.fa_development_frame._fetch_entities",
        fake_fetch,
    )
    return queries


def test_place_query_replaces_timezone_with_single_continent_current_facts():
    query = build_development_domain_query("place", limit=17)

    assert "wdt:P421" not in query
    assert "?countryStatement ps:P17 ?country" in query
    assert "?adminStatement ps:P131 ?admin" in query
    assert "?admin wdt:P31/wdt:P279* wd:Q56061" in query
    assert "?raw_1 wdt:P30 ?continent" in query
    assert "HAVING(COUNT(DISTINCT ?country) = 1)" in query
    assert "HAVING(COUNT(DISTINCT ?admin) = 1)" in query
    assert "HAVING(COUNT(DISTINCT ?continent) = 1)" in query
    assert "FILTER(?raw_2 != ?raw_1)" in query
    assert "rdfs:label ?itemLabel" in query
    assert "PREFIX rdfs:" in query
    assert "?other rdfs:label ?itemLabel" in query
    assert "?other wdt:P31/wdt:P279* wd:Q515" not in query
    assert "FILTER(?other != ?item)" in query
    assert "pq:P582" in query
    assert "FILTER NOT EXISTS" in query
    assert "LIMIT 17" in query


def test_organization_query_requires_single_targets_and_settlement_headquarters():
    query = build_development_domain_query("organization", limit=17)

    assert "?item wdt:P31/wdt:P279* wd:Q56061" in query
    assert "?raw_2 wdt:P31/wdt:P279* wd:Q486972" in query
    assert "?item p:P17 ?countryStatement" in query
    assert "?countryStatement ps:P17 ?country" in query
    assert "?item p:P159 ?headquartersStatement" in query
    assert "?headquartersStatement ps:P159 ?headquarters" in query
    assert "?item p:P571 ?inceptionStatement" in query
    assert "?inceptionStatement ps:P571 ?inception" in query
    assert "HAVING(COUNT(DISTINCT ?country) = 1)" in query
    assert "HAVING(COUNT(DISTINCT ?headquarters) = 1)" in query
    assert "HAVING(COUNT(DISTINCT ?inception) = 1)" in query
    assert "FILTER NOT EXISTS" in query
    assert "LIMIT 17" in query


def test_creative_work_query_excludes_duplicate_english_titles():
    query = build_development_domain_query("creative_work", limit=17)

    assert "rdfs:label ?itemLabel" in query
    assert "?other wdt:P31/wdt:P279* wd:Q11424" in query
    assert "FILTER(?other != ?item)" in query
    assert "LIMIT 17" in query


def test_builds_bounded_frame_with_exclusions_collisions_and_provenance(
    tmp_path,
    monkeypatch,
):
    queries = _install_remote_fakes(monkeypatch)

    result = build_development_frame(
        output_dir=tmp_path,
        tokenizer=_WordTokenizer(),
        tokenizer_revision="a" * 40,
        query_limit=17,
        required_per_domain=2,
        excluded_qids=frozenset({"Q10001"}),
        retrieval_date="2026-07-25",
    )

    assert len(queries) == 4
    assert all("LIMIT 17" in query for query in queries)
    frame = json.loads(result["source_frame"].read_text())
    assert frame["claim_scope"] == "open_instrument_development_only"
    assert frame["query_limit"] == 17
    assert frame["required_per_domain"] == 2
    assert frame["excluded_prior_qids"] == ["Q10001"]
    assert frame["ambiguous_cross_domain_qids"] == ["Q99999"]
    assert len(result["source_frame_sha256"]) == 64
    assert len(frame["provenance_sha256"]) == 64
    assert set(frame["query_sha256s"]) == set(REGISTERED_DOMAINS)
    assert {
        domain: len(frame["records_by_domain"][domain]) for domain in REGISTERED_DOMAINS
    } == {domain: 2 for domain in REGISTERED_DOMAINS}
    assert all(
        "Q10001" not in {row["qid"] for row in rows}
        for rows in frame["records_by_domain"].values()
    )


def test_place_query_limit_changes_only_place_and_is_bound_in_design(
    tmp_path,
    monkeypatch,
):
    queries = _install_remote_fakes(monkeypatch)

    result = build_development_frame(
        output_dir=tmp_path,
        tokenizer=_WordTokenizer(),
        tokenizer_revision="a" * 40,
        query_limit=17,
        place_query_limit=23,
        required_per_domain=2,
        excluded_qids=frozenset(),
        retrieval_date="2026-07-25",
    )

    place_queries = [query for query in queries if "wd:Q515;" in query]
    other_queries = [query for query in queries if "wd:Q515;" not in query]
    assert len(place_queries) == 1
    assert "LIMIT 23" in place_queries[0]
    assert len(other_queries) == 3
    assert all("LIMIT 17" in query for query in other_queries)

    frame = json.loads(result["source_frame"].read_text())
    assert frame["design"]["query_limit"] == 17
    assert frame["design"]["place_query_limit"] == 23
    assert frame["query_limits_by_domain"] == {
        "creative_work": 17,
        "organization": 17,
        "person": 17,
        "place": 23,
    }


def test_replays_immutable_frame_and_cached_queries_without_network(
    tmp_path,
    monkeypatch,
):
    _install_remote_fakes(monkeypatch)
    arguments = {
        "output_dir": tmp_path,
        "tokenizer": _WordTokenizer(),
        "tokenizer_revision": "a" * 40,
        "query_limit": 9,
        "required_per_domain": 2,
        "excluded_qids": frozenset(),
        "retrieval_date": "2026-07-25",
    }
    first = build_development_frame(**arguments)

    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError("immutable replay must not access the network")

    monkeypatch.setattr(
        "trajectory_extractor.fa_development_frame._post_sparql",
        fail,
    )
    monkeypatch.setattr(
        "trajectory_extractor.fa_development_frame._fetch_entities",
        fail,
    )
    second = build_development_frame(**arguments)

    assert second["source_frame_sha256"] == first["source_frame_sha256"]
    assert len(list((tmp_path / "source_cache").glob("qlever_*.json"))) == 4


def test_existing_frame_rejects_a_changed_design(tmp_path, monkeypatch):
    _install_remote_fakes(monkeypatch)
    common = {
        "output_dir": tmp_path,
        "tokenizer": _WordTokenizer(),
        "tokenizer_revision": "a" * 40,
        "required_per_domain": 2,
        "excluded_qids": frozenset(),
        "retrieval_date": "2026-07-25",
    }
    build_development_frame(query_limit=9, **common)

    with pytest.raises(ValueError, match="does not match the requested design"):
        build_development_frame(query_limit=10, **common)


def test_existing_frame_rejects_changed_place_query_limit(tmp_path, monkeypatch):
    _install_remote_fakes(monkeypatch)
    common = {
        "output_dir": tmp_path,
        "tokenizer": _WordTokenizer(),
        "tokenizer_revision": "a" * 40,
        "query_limit": 9,
        "required_per_domain": 2,
        "excluded_qids": frozenset(),
        "retrieval_date": "2026-07-25",
    }
    build_development_frame(place_query_limit=12, **common)

    with pytest.raises(ValueError, match="does not match the requested design"):
        build_development_frame(place_query_limit=13, **common)


@pytest.mark.parametrize("bad_limit", [0, -1, True, 1.5, "20"])
def test_rejects_invalid_place_query_limit(tmp_path, bad_limit):
    with pytest.raises(ValueError, match="place_query_limit"):
        build_development_frame(
            output_dir=tmp_path,
            tokenizer=_WordTokenizer(),
            tokenizer_revision="a" * 40,
            query_limit=9,
            place_query_limit=bad_limit,
            required_per_domain=2,
            excluded_qids=frozenset(),
            retrieval_date="2026-07-25",
        )


def test_existing_frame_rejects_stale_source_revision(tmp_path, monkeypatch):
    _install_remote_fakes(monkeypatch)
    arguments = {
        "output_dir": tmp_path,
        "tokenizer": _WordTokenizer(),
        "tokenizer_revision": "a" * 40,
        "query_limit": 9,
        "required_per_domain": 2,
        "excluded_qids": frozenset(),
        "retrieval_date": "2026-07-25",
    }
    result = build_development_frame(**arguments)
    frame = json.loads(result["source_frame"].read_text())
    frame["source_revision"] = "fa-development-source-frame-v6-r3"
    frame_without_hash = dict(frame)
    frame_without_hash.pop("frame_payload_sha256")
    frame["frame_payload_sha256"] = _canonical_hash(frame_without_hash)
    result["source_frame"].write_text(json.dumps(frame), encoding="utf-8")

    with pytest.raises(ValueError, match="stale source revision"):
        build_development_frame(**arguments)


def test_existing_frame_rejects_stale_behavior_code(tmp_path, monkeypatch):
    _install_remote_fakes(monkeypatch)
    arguments = {
        "output_dir": tmp_path,
        "tokenizer": _WordTokenizer(),
        "tokenizer_revision": "a" * 40,
        "query_limit": 9,
        "required_per_domain": 2,
        "excluded_qids": frozenset(),
        "retrieval_date": "2026-07-25",
    }
    result = build_development_frame(**arguments)
    frame = json.loads(result["source_frame"].read_text())
    frame["provenance"]["code_sha256s"]["fa_development_source.py"] = "0" * 64
    frame["provenance_sha256"] = _canonical_hash(frame["provenance"])
    frame_without_hash = dict(frame)
    frame_without_hash.pop("frame_payload_sha256")
    frame["frame_payload_sha256"] = _canonical_hash(frame_without_hash)
    result["source_frame"].write_text(json.dumps(frame), encoding="utf-8")

    with pytest.raises(ValueError, match="stale behavior code"):
        build_development_frame(**arguments)


def test_existing_frame_rejects_tampered_provenance(tmp_path, monkeypatch):
    _install_remote_fakes(monkeypatch)
    arguments = {
        "output_dir": tmp_path,
        "tokenizer": _WordTokenizer(),
        "tokenizer_revision": "a" * 40,
        "query_limit": 9,
        "required_per_domain": 2,
        "excluded_qids": frozenset(),
        "retrieval_date": "2026-07-25",
    }
    result = build_development_frame(**arguments)
    frame = json.loads(result["source_frame"].read_text())
    frame["provenance"]["model_id"] = "tampered/model"
    result["source_frame"].write_text(json.dumps(frame), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance hash"):
        build_development_frame(**arguments)


def _canonical_hash(payload):
    return __import__("hashlib").sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def test_fails_closed_when_matchable_domain_misses_exact_gate(
    tmp_path,
    monkeypatch,
):
    _install_remote_fakes(monkeypatch)

    with pytest.raises(ValueError, match="requires 4"):
        build_development_frame(
            output_dir=tmp_path,
            tokenizer=_WordTokenizer(),
            tokenizer_revision="a" * 40,
            query_limit=20,
            required_per_domain=4,
            excluded_qids=frozenset(),
            retrieval_date="2026-07-25",
        )


def test_refuses_nonidentical_cached_query(tmp_path, monkeypatch):
    _install_remote_fakes(monkeypatch)
    build_development_frame(
        output_dir=tmp_path,
        tokenizer=_WordTokenizer(),
        tokenizer_revision="a" * 40,
        query_limit=8,
        required_per_domain=2,
        excluded_qids=frozenset(),
        retrieval_date="2026-07-25",
    )
    frame_path = tmp_path / "development_source_frame_v1.json"
    frame_path.unlink()
    cached = next((tmp_path / "source_cache").glob("qlever_person_*.json"))
    cached.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="QLever response is missing"):
        build_development_frame(
            output_dir=tmp_path,
            tokenizer=_WordTokenizer(),
            tokenizer_revision="a" * 40,
            query_limit=8,
            required_per_domain=2,
            excluded_qids=frozenset(),
            retrieval_date="2026-07-25",
        )
