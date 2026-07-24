import json

import pytest

from trajectory_extractor.fa_confirmatory_source import (
    REGISTERED_DOMAINS,
    REGISTERED_SPLIT_COUNTS,
    RankedSource,
    SCREENING_POOL_MULTIPLIER,
    SourceRecord,
    _fetch_entities,
    _sha256_file,
    _write_json,
    audit_materialized_source,
    assign_split_pools,
    build_domain_query,
    build_source_records_from_ranked_values,
    build_source_records,
    exclude_cross_domain_source_collisions,
    filter_matchable_source_records,
    materialize_manifests,
    parse_qlever_candidates,
    source_matching_policy_sha256,
)


def _record(domain: str, index: int) -> SourceRecord:
    qid = f"Q{100000 + tuple(REGISTERED_DOMAINS).index(domain) * 1000 + index}"
    return SourceRecord(
        qid=qid,
        label=f"{domain.title()} Name {index}",
        domain=domain,
        sitelinks=500 - index,
        source_rank=index + 1,
        property_values=(
            ("P1", ("First", "First.")),
            ("P2", ("Second", "Second.")),
            ("P3", ("Third", "Third.")),
        ),
    )


def _entity(label, claims):
    return {
        "labels": {"en": {"language": "en", "value": label}},
        "claims": {
            property_id: [
                {
                    "rank": "normal",
                    "mainsnak": {"datavalue": {"value": value}},
                }
            ]
            for property_id, value in claims.items()
        },
    }


def _linked(label):
    return {
        "labels": {"en": {"language": "en", "value": label}},
        "aliases": {"en": []},
    }


def test_domain_queries_are_qid_only_and_property_complete():
    for domain in REGISTERED_DOMAINS:
        query = build_domain_query(domain, limit=123)
        assert "SELECT ?item ?sitelinks" in query
        assert "LIMIT 123" in query
        assert "SERVICE wikibase:label" not in query


def test_domain_queries_collect_all_truthy_values():
    creative_work = build_domain_query("creative_work", limit=123)
    organization = build_domain_query("organization", limit=123)

    assert "GROUP_CONCAT(DISTINCT STR(?raw_2); separator=\"|\")" in creative_work
    assert "GROUP_CONCAT(DISTINCT STR(?raw_3); separator=\"|\")" in organization


def test_qlever_parser_preserves_rank_and_deduplicates_qids():
    payload = {
        "results": {
            "bindings": [
                {
                    "item": {"value": "http://www.wikidata.org/entity/Q10"},
                    "sitelinks": {"value": "200"},
                },
                {
                    "item": {"value": "http://www.wikidata.org/entity/Q10"},
                    "sitelinks": {"value": "200"},
                },
                {
                    "item": {"value": "http://www.wikidata.org/entity/Q20"},
                    "sitelinks": {"value": "100"},
                },
            ]
        }
    }
    raw_values = (
        ("http://www.wikidata.org/entity/Q30", "alpha", "beta"),
        ("http://www.wikidata.org/entity/Q30", "alpha", "beta"),
        ("http://www.wikidata.org/entity/Q40", "gamma", "delta"),
    )
    for binding, values in zip(
        payload["results"]["bindings"],
        raw_values,
        strict=True,
    ):
        for index, value in enumerate(values, start=1):
            binding[f"value_{index}"] = {"value": value}
    assert parse_qlever_candidates(payload) == (
        RankedSource("Q10", 200, raw_values[0]),
        RankedSource("Q20", 100, raw_values[2]),
    )


def test_source_records_use_exact_claim_aliases_and_exclusions():
    entities = {
        "Q10": _entity("Ada Example", {"P27": "Q20", "P106": "Q30", "P19": "Q40"}),
        "Q11": _entity("Pilot Example", {"P27": "Q20", "P106": "Q30", "P19": "Q40"}),
        "Q20": _linked("Exampleland"),
        "Q30": _linked("physicist"),
        "Q40": _linked("Sample City"),
    }
    records = build_source_records(
        "person",
        (
            RankedSource(
                "Q10",
                200,
                tuple(f"http://www.wikidata.org/entity/Q{value}" for value in (20, 30, 40)),
            ),
            RankedSource(
                "Q11",
                190,
                tuple(f"http://www.wikidata.org/entity/Q{value}" for value in (20, 30, 40)),
            ),
        ),
        entities,
        excluded_qids=frozenset({"Q11"}),
    )
    assert len(records) == 1
    assert records[0].qid == "Q10"
    assert records[0].property_values[0][1] == ("Exampleland", "Exampleland.")


def test_ranked_value_records_need_only_labels_and_aliases():
    entities = {
        "Q10": _linked("Example Film"),
        "Q20": _linked("Example Director"),
        "Q30": _linked("Exampleland"),
    }
    records = build_source_records_from_ranked_values(
        "creative_work",
        (
            RankedSource(
                "Q10",
                200,
                (
                    "http://www.wikidata.org/entity/Q20",
                    "1999-01-01T00:00:00Z",
                    "http://www.wikidata.org/entity/Q30",
                ),
            ),
        ),
        entities,
    )

    assert records == (
        SourceRecord(
            qid="Q10",
            label="Example Film",
            domain="creative_work",
            sitelinks=200,
            source_rank=1,
            property_values=(
                ("P57", ("Example Director", "Example Director.")),
                ("P577", ("1999", "1999.")),
                ("P495", ("Exampleland", "Exampleland.")),
            ),
        ),
    )


def test_ranked_value_records_accept_every_collected_entity_value():
    entities = {
        "Q10": _linked("Ada Example"),
        "Q20": _linked("Exampleland"),
        "Q21": _linked("Otherland"),
        "Q30": _linked("physicist"),
        "Q31": _linked("writer"),
        "Q40": _linked("Sample City"),
    }
    records = build_source_records_from_ranked_values(
        "person",
        (
            RankedSource(
                "Q10",
                200,
                (
                    "http://www.wikidata.org/entity/Q20|"
                    "http://www.wikidata.org/entity/Q21",
                    "http://www.wikidata.org/entity/Q30|"
                    "http://www.wikidata.org/entity/Q31",
                    "http://www.wikidata.org/entity/Q40",
                ),
            ),
        ),
        entities,
    )

    assert records[0].property_values[0][1] == (
        "Exampleland",
        "Exampleland.",
        "Otherland",
        "Otherland.",
    )
    assert "physicist" in records[0].property_values[1][1]
    assert "writer" in records[0].property_values[1][1]

    reversed_records = build_source_records_from_ranked_values(
        "person",
        (
            RankedSource(
                "Q10",
                200,
                (
                    "http://www.wikidata.org/entity/Q21|"
                    "http://www.wikidata.org/entity/Q20",
                    "http://www.wikidata.org/entity/Q31|"
                    "http://www.wikidata.org/entity/Q30",
                    "http://www.wikidata.org/entity/Q40",
                ),
            ),
        ),
        entities,
    )

    assert reversed_records == records


def test_year_values_accept_qlever_datetime_literals():
    entities = {
        "Q10": _entity(
            "Example Film",
            {
                "P57": "Q20",
                "P577": {"time": "1999-01-01T00:00:00Z"},
                "P495": "Q30",
            },
        ),
        "Q20": _linked("Example Director"),
        "Q30": _linked("Exampleland"),
    }
    records = build_source_records(
        "creative_work",
        (
            RankedSource(
                "Q10",
                200,
                (
                    "http://www.wikidata.org/entity/Q20",
                    "1999-01-01T00:00:00Z",
                    "http://www.wikidata.org/entity/Q30",
                ),
            ),
        ),
        entities,
    )

    assert records[0].property_values[1][1] == ("1999", "1999.")


def test_year_values_keep_only_the_earliest_non_deprecated_year():
    source = _entity(
        "Example Film",
        {
            "P57": "Q20",
            "P577": {"time": "2001-01-01T00:00:00Z"},
            "P495": "Q30",
        },
    )
    source["claims"]["P577"].extend(
        [
            {
                "rank": "normal",
                "mainsnak": {
                    "datavalue": {
                        "value": {"time": "1999-01-01T00:00:00Z"}
                    }
                },
            },
            {
                "rank": "deprecated",
                "mainsnak": {
                    "datavalue": {
                        "value": {"time": "1998-01-01T00:00:00Z"}
                    }
                },
            },
        ]
    )
    entities = {
        "Q10": source,
        "Q20": _linked("Example Director"),
        "Q30": _linked("Exampleland"),
    }

    records = build_source_records(
        "creative_work",
        (
            RankedSource(
                "Q10",
                200,
                (
                    "http://www.wikidata.org/entity/Q20",
                    "2001-01-01T00:00:00Z",
                    "http://www.wikidata.org/entity/Q30",
                ),
            ),
        ),
        entities,
    )

    assert records[0].property_values[1][1] == ("1999", "1999.")


def test_source_records_accept_every_non_deprecated_multivalued_fact():
    source = _entity(
        "Ada Example",
        {"P27": "Q20", "P106": "Q30", "P19": "Q40"},
    )
    source["claims"]["P106"].append(
        {
            "rank": "normal",
            "mainsnak": {"datavalue": {"value": "Q31"}},
        }
    )
    source["claims"]["P106"].append(
        {
            "rank": "deprecated",
            "mainsnak": {"datavalue": {"value": "Q32"}},
        }
    )
    entities = {
        "Q10": source,
        "Q20": _linked("Exampleland"),
        "Q30": _linked("physicist"),
        "Q31": _linked("writer"),
        "Q32": _linked("deprecated role"),
        "Q40": _linked("Sample City"),
    }
    records = build_source_records(
        "person",
        (
            RankedSource(
                "Q10",
                200,
                tuple(
                    f"http://www.wikidata.org/entity/Q{value}"
                    for value in (20, 30, 40)
                ),
            ),
        ),
        entities,
    )

    occupation_aliases = records[0].property_values[1][1]
    assert "physicist" in occupation_aliases
    assert "writer" in occupation_aliases
    assert "deprecated role" not in occupation_aliases


def test_split_assignment_is_exact_deterministic_and_isolated():
    needed_per_domain = (
        sum(REGISTERED_SPLIT_COUNTS.values())
        // len(REGISTERED_DOMAINS)
        * SCREENING_POOL_MULTIPLIER
    )
    records = {
        domain: tuple(_record(domain, index) for index in range(needed_per_domain))
        for domain in REGISTERED_DOMAINS
    }
    first = assign_split_pools(records, seed=20260722)
    second = assign_split_pools(records, seed=20260722)
    assert first == second
    all_qids = []
    for split, split_count in REGISTERED_SPLIT_COUNTS.items():
        expected = split_count * SCREENING_POOL_MULTIPLIER
        assert len(first[split]) == expected
        for domain in REGISTERED_DOMAINS:
            assert sum(row.domain == domain for row in first[split]) == (
                split_count // len(REGISTERED_DOMAINS)
            ) * SCREENING_POOL_MULTIPLIER
        all_qids.extend(row.qid for row in first[split])
    assert len(all_qids) == len(set(all_qids))


def test_matchability_filter_runs_before_split_assignment():
    class WordTokenizer:
        name_or_path = "registered-word-tokenizer"

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

    records = {
        domain: tuple(_record(domain, index) for index in range(6))
        for domain in REGISTERED_DOMAINS
    }

    selected, audit = filter_matchable_source_records(
        records,
        WordTokenizer(),
        required_per_domain=4,
    )

    assert {
        domain: len(rows) for domain, rows in selected.items()
    } == {
        domain: 4 for domain in REGISTERED_DOMAINS
    }
    assert audit["generator_attempt_limit"] == 5000
    assert audit["eligible_counts"] == {
        domain: 6 for domain in REGISTERED_DOMAINS
    }
    assert audit["selected_matchable_counts"] == {
        domain: 4 for domain in REGISTERED_DOMAINS
    }
    assert len(audit["attrition_rows"]) == 24
    assert {
        row["status"] for row in audit["attrition_rows"]
    } == {
        "selected",
        "matchable_not_selected",
    }
    assert all(row["source_rank"] >= 1 for row in audit["attrition_rows"])
    assert all(row["sentence_frame_token_count"] >= 1 for row in audit["attrition_rows"])
    assert len(source_matching_policy_sha256()) == 64


def test_cross_domain_qid_and_name_collisions_are_removed_before_matching():
    records = {
        domain: [_record(domain, index) for index in range(2)]
        for domain in REGISTERED_DOMAINS
    }
    records["place"][0] = SourceRecord(
        qid=records["person"][0].qid,
        label=records["place"][0].label,
        domain="place",
        sitelinks=records["place"][0].sitelinks,
        source_rank=records["place"][0].source_rank,
        property_values=records["place"][0].property_values,
    )
    records["organization"][1] = SourceRecord(
        qid=records["organization"][1].qid,
        label=records["creative_work"][1].label,
        domain="organization",
        sitelinks=records["organization"][1].sitelinks,
        source_rank=records["organization"][1].source_rank,
        property_values=records["organization"][1].property_values,
    )

    filtered, ambiguous_qids, ambiguous_names = (
        exclude_cross_domain_source_collisions(records)
    )

    assert records["person"][0].qid in ambiguous_qids
    assert records["creative_work"][1].label.casefold() in ambiguous_names
    assert all(
        row.qid != records["person"][0].qid
        for rows in filtered.values()
        for row in rows
    )
    assert all(
        row.label.casefold() != records["creative_work"][1].label.casefold()
        for rows in filtered.values()
        for row in rows
    )


def test_matchability_filter_fails_closed_on_domain_shortage():
    class NeverMatchingTokenizer:
        name_or_path = "registered-never-matching-tokenizer"

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return ["known"] if "Person Name 0" in text else list(text)

    records = {
        domain: (_record(domain, 0),)
        for domain in REGISTERED_DOMAINS
    }

    with pytest.raises(ValueError, match="no_match_under_frozen_generator"):
        filter_matchable_source_records(
            records,
            NeverMatchingTokenizer(),
            required_per_domain=1,
        )


def test_materialized_manifests_bind_qids_questions_and_provenance():
    assigned = {
        split: tuple(_record(domain, index + 1) for index, domain in enumerate(REGISTERED_DOMAINS))
        for split in REGISTERED_SPLIT_COUNTS
    }
    manifests = materialize_manifests(
        assigned,
        retrieval_date="2026-07-24",
        query_hashes={domain: "a" * 64 for domain in REGISTERED_DOMAINS},
    )
    candidates, questions = manifests["mechanism_train"]
    assert len(candidates) == 4
    assert len(questions) == 12
    assert {question.qid for question in questions} == {
        candidate.qid for candidate in candidates
    }
    assert all("CC0" in candidate.source_provenance for candidate in candidates)


def test_source_audit_rejects_cross_split_qid_and_name_collisions():
    records = {
        domain: tuple(_record(domain, index) for index in range(96))
        for domain in REGISTERED_DOMAINS
    }
    assigned = assign_split_pools(records, seed=20260722)
    manifests = materialize_manifests(
        assigned,
        retrieval_date="2026-07-24",
        query_hashes={domain: "a" * 64 for domain in REGISTERED_DOMAINS},
    )
    assert audit_materialized_source(manifests) == {
        "candidate_count": 384,
        "question_count": 1152,
        "split_count": 5,
    }

    duplicate = dict(manifests)
    candidates, questions = duplicate["behavior_test"]
    other_candidates, other_questions = duplicate["mechanism_train"]
    replacement = other_candidates[0]
    duplicate["behavior_test"] = (
        (replacement, *candidates[1:]),
        (
            *(
                question
                for question in other_questions
                if question.qid == replacement.qid
            ),
            *(question for question in questions if question.qid != candidates[0].qid),
        ),
    )
    with pytest.raises(
        ValueError,
        match="duplicate question IDs|cross-split QID leakage",
    ):
        audit_materialized_source(duplicate)


def test_entity_fetch_resumes_from_batch_caches(tmp_path, monkeypatch):
    requested = []

    def fake_request(request):
        query = request.full_url.split("?", 1)[1]
        qids = query.split("ids=", 1)[1].split("&", 1)[0].replace("%7C", "|").split("|")
        requested.append(tuple(qids))
        return {
            "entities": {
                qid: _linked(f"Entity {qid}")
                for qid in qids
            }
        }

    monkeypatch.setattr(
        "trajectory_extractor.fa_confirmatory_source._request_json",
        fake_request,
    )
    qids = tuple(f"Q{index}" for index in range(1, 53))
    cache_dir = tmp_path / "entity_batches"
    first = _fetch_entities(qids, cache_dir=cache_dir)
    assert len(first) == 52
    assert len(requested) == 2
    assert len(list(cache_dir.glob("entities_*.json"))) == 2

    requested.clear()
    second = _fetch_entities(qids, cache_dir=cache_dir)
    assert first == second
    assert requested == []
    for cache_path in cache_dir.glob("entities_*.json"):
        assert isinstance(json.loads(cache_path.read_text()), dict)


def test_source_writer_resumes_identically_and_refuses_overwrite(tmp_path):
    path = tmp_path / "source.json"
    _write_json(path, {"revision": 1})
    _write_json(path, {"revision": 1})

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json(path, {"revision": 2})

    assert _sha256_file(path) == (
        "b3e94f06083e92373724ff153d1c5b022fbbd041a40daa18458a65e8419b0951"
    )
