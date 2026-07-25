from __future__ import annotations

import json

import pytest

from trajectory_extractor.fa_confirmatory_source import REGISTERED_DOMAINS
from trajectory_extractor.fa_development_frame import build_development_frame


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
    assert frame["excluded_source_v5_qids"] == ["Q10001"]
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
