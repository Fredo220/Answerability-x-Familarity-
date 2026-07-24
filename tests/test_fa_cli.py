import argparse
import csv
import hashlib
import inspect
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trajectory_extractor import cli
from trajectory_extractor.fa_artifacts import FAArtifactStore, UnlockReceipt
import trajectory_extractor.fa_cli as fa_cli
import trajectory_extractor.fa_probes as fa_probes
from trajectory_extractor.fa_cli import dispatch_fa, register_fa_subcommands
from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_data import (
    CONFIRMATORY_POWER_SIMULATIONS,
    REGISTERED_POWER_GRID,
    PowerAudit,
    PowerCell,
    build_factorial_examples,
)
from trajectory_extractor.fa_entities import (
    CandidateEntity,
    EntityMatch,
    NaturalnessAudit,
    NaturalnessRating,
    ScreeningQuestion,
    SyntheticCandidate,
)
from trajectory_extractor.fa_features import OutputEvidence
from trajectory_extractor.fa_naturalness import (
    compile_initial_responses_from_issuance,
    naturalness_matches_sha256,
    packet_issuance_record,
    prepare_initial_rating_packets,
    submission_record,
)
from trajectory_extractor.fa_runtime import run_generation_shard
from trajectory_extractor.fa_probes import (
    OUTPUT_CONTROL_SCHEMA_SHA256,
    ProbeRow,
    ProbeTestAuthorization,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "familiarity_answerability_qwen06b_smoke.json"
)

MATCH = {
    "pair_id": "Q1--syn-1",
    "real_entity_id": "Q1",
    "real_qid": "Q1",
    "synthetic_candidate_id": "syn-1",
    "real_name": "Old Vale",
    "synthetic_name": "New Vale",
    "coarse_type": "place",
    "split": "pilot",
    "generator_revision": "names-v1",
    "tokenizer_revision": "tokenizer-v1",
    "real_token_count": 2,
    "synthetic_token_count": 2,
    "real_word_count": 2,
    "synthetic_word_count": 2,
    "real_character_count": 8,
    "synthetic_character_count": 8,
    "character_length_delta": 0,
    "character_tolerance": 2,
    "capitalization_pattern_equal": True,
}


def smoke_pilot_matches():
    rows = []
    domains = ("person", "place", "organization", "creative_work")
    for index in range(8):
        real_name = f"Old Vale {index}"
        synthetic_name = f"New Vale {index}"
        rows.append(
            {
                **MATCH,
                "pair_id": f"Q{index + 1}--syn-{index + 1}",
                "real_entity_id": f"Q{index + 1}",
                "real_qid": f"Q{index + 1}",
                "synthetic_candidate_id": f"syn-{index + 1}",
                "real_name": real_name,
                "synthetic_name": synthetic_name,
                "coarse_type": domains[index % len(domains)],
                "real_token_count": 3,
                "synthetic_token_count": 3,
                "real_word_count": 3,
                "synthetic_word_count": 3,
                "real_character_count": len(real_name),
                "synthetic_character_count": len(synthetic_name),
            }
        )
    return rows


def screened_matches_manifest(
    tmp_path,
    config,
    rows,
    *,
    shard_id="screened-matches-test",
    audited_entity_ids=None,
    namespace="pilot",
):
    model_hash, tokenizer_hash = fa_cli._config_runtime_hashes(config)
    template_hash = config.chat_template_sha256 or CHAT_TEMPLATE_SHA256
    store = FAArtifactStore(tmp_path)
    completion = store.write_completed_shard(
        config.run_id,
        namespace,
        f"{shard_id}-completion",
        [],
        {
            "config_sha256": config.config_hash,
            "candidate_manifest_sha256": "a" * 64,
            "questions_manifest_sha256": "b" * 64,
            "model_sha256": model_hash,
            "tokenizer_sha256": tokenizer_hash,
            "chat_template_sha256": template_hash,
        },
        record_kind="screening_completion",
    )
    audit = store.write_completed_shard(
        config.run_id,
        namespace,
        f"{shard_id}-audit",
        [
            {
                "kind": "screening_audit",
                "decision": "passed",
                "screening_completion_sha256": completion.sha256,
                "selected_entity_ids": (
                    [row["real_entity_id"] for row in rows]
                    if audited_entity_ids is None
                    else list(audited_entity_ids)
                ),
                "required_count": config.split_counts[namespace],
                "reserve_per_domain": (
                    0
                    if config.profile == "smoke"
                    else fa_cli._CONFIRMATORY_RESERVE_PER_DOMAIN[namespace]
                ),
                "selected_count": len(rows),
            }
        ],
        {
            "config_sha256": config.config_hash,
            "candidate_manifest_sha256": "a" * 64,
            "questions_manifest_sha256": "b" * 64,
            "synthetic_manifest_sha256": "c" * 64,
            "screening_completion_sha256": completion.sha256,
            "screening_parser_sha256": "f" * 64,
        },
        record_kind="screening_audit",
    )
    return store.write_completed_shard(
        config.run_id,
        namespace,
        shard_id,
        [{"kind": "screened_match", **row} for row in rows],
        {
            "config_sha256": config.config_hash,
            "model_sha256": model_hash,
            "tokenizer_sha256": tokenizer_hash,
            "chat_template_sha256": template_hash,
            "candidate_manifest_sha256": "a" * 64,
            "questions_manifest_sha256": "b" * 64,
            "synthetic_manifest_sha256": "c" * 64,
            "screening_completion_sha256": completion.sha256,
            "screening_audit_sha256": audit.sha256,
            "screening_parser_sha256": "f" * 64,
            "matching_policy_sha256": fa_cli._matching_policy_sha256(),
            "screening_completion_manifest": str(
                completion.manifest_path.relative_to(store.root)
            ),
            "screening_audit_manifest": str(
                audit.manifest_path.relative_to(store.root)
            ),
        },
        record_kind="screened_match",
    ).manifest_path


class FakeTokenizer:
    chat_template = "fake qwen template"
    all_special_ids = ()

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=None,
    ):
        del enable_thinking
        rendered = messages[0]["content"] + " <assistant>"
        return self.encode(rendered) if tokenize else rendered


CHAT_TEMPLATE_BYTES = FakeTokenizer.chat_template.encode("utf-8")
CHAT_TEMPLATE_SHA256 = hashlib.sha256(CHAT_TEMPLATE_BYTES).hexdigest()


@pytest.fixture(autouse=True)
def register_fake_smoke_template(monkeypatch):
    monkeypatch.setattr(fa_cli, "_SMOKE_CHAT_TEMPLATE_SHA256", CHAT_TEMPLATE_SHA256)
    monkeypatch.setattr(fa_cli, "_SMOKE_CONFIG_PATH", CONFIG_PATH)


def install_fake_tokenizer(monkeypatch):
    monkeypatch.setattr(
        fa_cli,
        "_TOKENIZER_LOADER",
        lambda model_id, *, revision: FakeTokenizer(),
    )


def confirmatory_reserve_matches(config, *, reserve_per_cell=1):
    base = EntityMatch(**MATCH)
    domains = ("person", "place", "organization", "creative_work")
    rows = []
    index = 100
    for split, split_count in config.split_counts.items():
        quota = split_count // len(domains)
        for domain in domains:
            for _ in range(quota + reserve_per_cell):
                real_name = f"Old Vale {index}"
                synthetic_name = f"New Vale {index}"
                rows.append(
                    replace(
                        base,
                        pair_id=f"Q{index}--syn-{index}",
                        real_entity_id=f"Q{index}",
                        real_qid=f"Q{index}",
                        synthetic_candidate_id=f"syn-{index}",
                        real_name=real_name,
                        synthetic_name=synthetic_name,
                        coarse_type=domain,
                        split=split,
                        real_word_count=3,
                        synthetic_word_count=3,
                        real_character_count=len(real_name),
                        synthetic_character_count=len(synthetic_name),
                    )
                )
                index += 1
    return tuple(rows)


def confirmatory_registered_match_pool(config):
    base = EntityMatch(**MATCH)
    domains = ("person", "place", "organization", "creative_work")
    rows = []
    index = 1000
    for split, split_count in config.split_counts.items():
        quota = (
            split_count // len(domains)
            + fa_cli._CONFIRMATORY_RESERVE_PER_DOMAIN[split]
        )
        for domain in domains:
            for _ in range(quota):
                real_name = f"Old Vale {index}"
                synthetic_name = f"New Vale {index}"
                rows.append(
                    replace(
                        base,
                        pair_id=f"Q{index}--syn-{index}",
                        real_entity_id=f"Q{index}",
                        real_qid=f"Q{index}",
                        synthetic_candidate_id=f"syn-{index}",
                        real_name=real_name,
                        synthetic_name=synthetic_name,
                        coarse_type=domain,
                        split=split,
                        real_word_count=3,
                        synthetic_word_count=3,
                        real_character_count=len(real_name),
                        synthetic_character_count=len(synthetic_name),
                    )
                )
                index += 1
    return tuple(rows)


def confirmatory_screened_collection(tmp_path, config):
    manifests = []
    rows = confirmatory_registered_match_pool(config)
    for split in config.split_counts:
        split_rows = [
            asdict(row)
            for row in rows
            if row.split == split
        ]
        manifests.append(
            screened_matches_manifest(
                tmp_path,
                config,
                split_rows,
                shard_id=f"screened-{split}",
                namespace=split,
            )
        )
    payload = fa_cli._assemble_screened_matches(
        config,
        tmp_path,
        SimpleNamespace(
            screened_matches_manifest=manifests,
            shard_id="confirmatory-screened-collection",
        ),
    )
    return Path(payload["manifest"]), rows


def test_confirmatory_screened_collection_binds_every_split_and_reserve(tmp_path):
    config = FAConfig.from_json(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    manifest, expected = confirmatory_screened_collection(tmp_path, config)

    observed = fa_cli._load_verified_screened_match_collection(
        FAArtifactStore(tmp_path),
        manifest,
        config,
    )

    assert len(observed) == 244
    assert set(observed) == set(expected)
    assert {row.split for row in observed} == set(config.split_counts)


def test_confirmatory_naturalness_rejects_raw_match_files(tmp_path):
    config = FAConfig.from_json(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    raw = tmp_path / "raw-matches.json"
    raw.write_text(json.dumps([MATCH]), encoding="utf-8")

    with pytest.raises(ValueError, match="verified screened-match collection"):
        fa_cli._load_naturalness_matches(
            config,
            tmp_path,
            SimpleNamespace(
                screened_matches_manifest=None,
                matches_manifest=raw,
            ),
        )


def test_confirmatory_naturalness_rejects_single_screened_child_shard(tmp_path):
    config = FAConfig.from_json(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    rows = [
        asdict(row)
        for row in confirmatory_registered_match_pool(config)
        if row.split == "mechanism_train"
    ]
    child = screened_matches_manifest(
        tmp_path,
        config,
        rows,
        shard_id="screened-mechanism-train",
        namespace="mechanism_train",
    )

    with pytest.raises(ValueError, match="verified screened-match collection"):
        fa_cli._load_naturalness_matches(
            config,
            tmp_path,
            SimpleNamespace(
                screened_matches_manifest=child,
                matches_manifest=None,
            ),
        )


def test_screening_matching_uses_the_pinned_model_tokenizer(tmp_path, monkeypatch):
    config = FAConfig.from_json(CONFIG_PATH)

    class SentinelTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

    sentinel = SentinelTokenizer()
    read_json_rows = fa_cli._read_json_rows
    domains = ("person", "place", "organization", "creative_work")
    candidates = []
    synthetics = []
    for index in range(8):
        domain = domains[index % len(domains)]
        candidates.append(
            {
                "entity_id": f"entity-{index}",
                "qid": f"Q{index + 1}",
                "name": f"Old Vale {index}",
                "coarse_type": domain,
                "split": "pilot",
                "source_query": "registered-query-v1",
                "source_provenance": "CC0-1.0",
                "screening_aliases": [["alpha"], ["beta"], ["gamma"]],
            }
        )
        synthetics.append(
            {
                "candidate_id": f"syn-{index}",
                "name": f"New Vale {index}",
                "coarse_type": domain,
                "split": "pilot",
                "generator_revision": "names-v1",
            }
        )
    questions = [
        {
            "question_id": f"entity-{index}-{question_index}",
            "qid": f"Q{index + 1}",
            "prompt": f"Question {question_index}",
            "accepted_aliases": [answer],
            "source_provenance": "CC0-1.0",
        }
        for index in range(8)
        for question_index, answer in enumerate(
            ("alpha", "beta", "gamma"), start=1
        )
    ]
    monkeypatch.setattr(
        fa_cli,
        "_read_json_rows",
        lambda path: (
            candidates
            if path.name == "candidates.json"
            else questions
            if path.name == "questions.json"
            else synthetics
            if path.name == "synthetic.json"
            else read_json_rows(path)
        ),
    )
    monkeypatch.setattr(
        fa_cli,
        "_load_verified_screening_completions",
        lambda *args, **kwargs: {
            candidate["entity_id"]: ("alpha", "beta", "gamma")
            for candidate in candidates
        },
    )
    monkeypatch.setattr(
        fa_cli,
        "_require_verified_shard_kind",
        lambda *args, **kwargs: SimpleNamespace(
            namespace="pilot",
            shard_id="screening-0001",
            sha256="a" * 64,
            manifest_path=tmp_path / "screening.jsonl.manifest.json",
        ),
    )
    monkeypatch.setattr(
        fa_cli,
        "load_pinned_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(tokenizer=sentinel),
    )
    observed = []
    monkeypatch.setattr(
        fa_cli,
        "match_synthetic_entities",
        lambda qualified, candidates, tokenizer: observed.append(tokenizer) or (),
    )

    payload = fa_cli._screen_entities(
        config,
        tmp_path,
        SimpleNamespace(
            candidates_manifest=Path("candidates.json"),
            questions_manifest=Path("questions.json"),
            synthetic_manifest=Path("synthetic.json"),
            screening_manifest=Path("screening.json"),
        ),
    )
    resumed = fa_cli._screen_entities(
        config,
        tmp_path,
        SimpleNamespace(
            candidates_manifest=Path("candidates.json"),
            questions_manifest=Path("questions.json"),
            synthetic_manifest=Path("synthetic.json"),
            screening_manifest=Path("screening.json"),
        ),
    )

    assert payload["status"] == "screened"
    assert resumed == payload
    assert observed == [sentinel, sentinel]
    store = FAArtifactStore(tmp_path)
    match_shard = store.verify_shard(payload["manifest"])
    audit_shard = store.verify_shard(payload["audit_manifest"])
    match_lineage = json.loads(
        match_shard.manifest_path.read_text(encoding="utf-8")
    )["lineage"]
    assert match_shard.record_kind == "screened_match"
    assert audit_shard.record_kind == "screening_audit"
    assert match_lineage["screening_audit_sha256"] == audit_shard.sha256
    assert len(match_lineage["screening_parser_sha256"]) == 64
    assert match_lineage["matching_policy_sha256"] == fa_cli._matching_policy_sha256()
    audit_lineage = json.loads(
        audit_shard.manifest_path.read_text(encoding="utf-8")
    )["lineage"]
    assert audit_shard.shard_id == (
        f"screening-audit-screening-0001-{fa_cli._sha256_json(audit_lineage)[:12]}"
    )
    assert match_shard.shard_id.endswith(
        f"-{fa_cli._matching_policy_sha256()[:12]}"
    )


def test_screening_failure_writes_a_machine_readable_audit(tmp_path, monkeypatch):
    config = FAConfig.from_json(CONFIG_PATH)

    class WordTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

    domains = ("person", "place", "organization", "creative_work")
    candidates = [
        {
            "entity_id": f"entity-{index}",
            "qid": f"Q{index}",
            "name": f"Entity {index}",
            "coarse_type": domain,
            "split": "pilot",
            "source_query": "registered-query-v1",
            "source_provenance": "CC0-1.0",
            "screening_aliases": [["alpha"], ["beta"], ["gamma"]],
        }
        for index, domain in enumerate(domains, start=1)
    ]
    questions = [
        {
            "question_id": f"entity-{index}-{question_index}",
            "qid": f"Q{index}",
            "prompt": f"Question {question_index}",
            "accepted_aliases": [answer],
            "source_provenance": "CC0-1.0",
        }
        for index in range(1, 5)
        for question_index, answer in enumerate(
            ("alpha", "beta", "gamma"), start=1
        )
    ]
    synthetics = [
        {
            "candidate_id": f"syn-{index}",
            "name": f"Synthetic {index}",
            "coarse_type": domain,
            "split": "pilot",
            "generator_revision": "names-v1",
        }
        for index, domain in enumerate(domains, start=1)
    ]
    monkeypatch.setattr(
        fa_cli,
        "_read_json_rows",
        lambda path: (
            candidates
            if path.name == "candidates.json"
            else questions
            if path.name == "questions.json"
            else synthetics
        ),
    )
    monkeypatch.setattr(
        fa_cli,
        "_load_verified_screening_completions",
        lambda *args, **kwargs: {
            candidate["entity_id"]: ("alpha", "beta", "gamma")
            for candidate in candidates
        },
    )
    monkeypatch.setattr(
        fa_cli,
        "_require_verified_shard_kind",
        lambda *args, **kwargs: SimpleNamespace(
            namespace="pilot",
            shard_id="screening-failed",
            sha256="a" * 64,
        ),
    )
    monkeypatch.setattr(
        fa_cli,
        "load_pinned_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(tokenizer=WordTokenizer()),
    )

    with pytest.raises(ValueError, match="screening audit"):
        fa_cli._screen_entities(
            config,
            tmp_path,
            SimpleNamespace(
                candidates_manifest=Path("candidates.json"),
                questions_manifest=Path("questions.json"),
                synthetic_manifest=Path("synthetic.json"),
                screening_manifest=Path("screening.json"),
            ),
        )

    audit = next(
        shard
        for shard in FAArtifactStore(tmp_path).resume_verified_shards(
            config.run_id, "pilot"
        )
        if shard.record_kind == "screening_audit"
    )
    row = json.loads(audit.data_path.read_text(encoding="utf-8"))
    assert row["decision"] == "stopped"
    assert row["selected_entity_ids"] == []
    assert "exact domain balance" in row["stop_reason"]


def test_screening_selection_is_manifest_ordered_and_exactly_domain_balanced():
    domains = ("person", "place", "organization", "creative_work")
    aliases = (("alpha",), ("beta",), ("gamma",))
    candidates = []
    for round_index in range(3):
        for domain_index, domain in enumerate(domains):
            index = round_index * len(domains) + domain_index + 1
            candidates.append(
                CandidateEntity(
                    entity_id=f"entity-{index}",
                    qid=f"Q{index}",
                    name=f"Entity {index}",
                    coarse_type=domain,
                    split="pilot",
                    source_query="registered-query-v1",
                    source_provenance="CC0-1.0",
                    screening_aliases=aliases,
                )
            )

    selected = fa_cli._select_domain_balanced_candidates(
        candidates,
        required_count=8,
    )

    assert tuple(candidate.entity_id for candidate in selected) == tuple(
        f"entity-{index}" for index in range(1, 9)
    )
    assert {
        domain: sum(candidate.coarse_type == domain for candidate in selected)
        for domain in domains
    } == {domain: 2 for domain in domains}


def test_confirmatory_screening_selection_retains_registered_reserves():
    aliases = (("alpha",), ("beta",), ("gamma",))
    domains = ("person", "place", "organization", "creative_work")
    candidates = tuple(
        CandidateEntity(
            entity_id=f"entity-{index}",
            qid=f"Q{index}",
            name=f"Entity {index}",
            coarse_type=domains[(index - 1) % len(domains)],
            split="behavior_test",
            source_query="registered-query-v1",
            source_provenance="Wikidata CC0",
            screening_aliases=aliases,
        )
        for index in range(1, 61)
    )

    selected = fa_cli._select_domain_balanced_candidates(
        candidates,
        required_count=48,
        reserve_per_domain=3,
    )

    assert len(selected) == 60
    assert {
        domain: sum(candidate.coarse_type == domain for candidate in selected)
        for domain in domains
    } == {domain: 15 for domain in domains}


def test_confirmatory_reserve_table_is_frozen_per_split():
    config = FAConfig.from_json(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    aliases = (("alpha",), ("beta",), ("gamma",))
    expected = {
        "mechanism_train": 4,
        "locked_validation": 2,
        "behavior_test": 3,
        "probe_test": 2,
        "intervention_test": 2,
    }
    for split, reserve in expected.items():
        candidate = CandidateEntity(
            entity_id=f"entity-{split}",
            qid=f"Q{len(split) + 1}",
            name="Entity Name",
            coarse_type="person",
            split=split,
            source_query="registered-query-v1",
            source_provenance="Wikidata CC0",
            screening_aliases=aliases,
        )
        assert fa_cli._screening_reserve_per_domain((candidate,), config) == reserve


def test_screening_selection_fails_closed_on_domain_shortage_and_unknown_domain():
    aliases = (("alpha",), ("beta",), ("gamma",))

    def candidate(index, domain):
        return CandidateEntity(
            entity_id=f"entity-{index}",
            qid=f"Q{index}",
            name=f"Entity {index}",
            coarse_type=domain,
            split="pilot",
            source_query="registered-query-v1",
            source_provenance="CC0-1.0",
            screening_aliases=aliases,
        )

    shortage = [
        candidate(1, "person"),
        candidate(2, "place"),
        candidate(3, "organization"),
        candidate(4, "creative_work"),
    ]
    with pytest.raises(ValueError, match="exact domain balance"):
        fa_cli._select_domain_balanced_candidates(shortage, required_count=8)

    with pytest.raises(ValueError, match="unregistered entity domain"):
        fa_cli._select_domain_balanced_candidates(
            [candidate(5, "unknown")],
            required_count=4,
        )


def test_screening_required_count_rejects_mixed_splits_and_duplicate_identities():
    config = FAConfig.from_json(CONFIG_PATH)
    aliases = (("alpha",), ("beta",), ("gamma",))

    def candidate(entity_id, qid, split):
        return CandidateEntity(
            entity_id=entity_id,
            qid=qid,
            name=f"Entity {qid}",
            coarse_type="person",
            split=split,
            source_query="registered-query-v1",
            source_provenance="CC0-1.0",
            screening_aliases=aliases,
        )

    with pytest.raises(ValueError, match="exactly one split"):
        fa_cli._screening_required_count(
            [
                candidate("entity-1", "Q1", "pilot"),
                candidate("entity-2", "Q2", "circuit_dev"),
            ],
            config,
        )
    with pytest.raises(ValueError, match="duplicate entity IDs"):
        fa_cli._screening_required_count(
            [
                candidate("entity-1", "Q1", "pilot"),
                candidate("entity-1", "Q2", "pilot"),
            ],
            config,
        )
    with pytest.raises(ValueError, match="duplicate QIDs"):
        fa_cli._screening_required_count(
            [
                candidate("entity-1", "Q1", "pilot"),
                candidate("entity-2", "Q1", "pilot"),
            ],
            config,
        )


def test_confirmatory_screening_requires_exact_frozen_2x_source_pool():
    config = FAConfig.from_json(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    aliases = (("alpha",), ("beta",), ("gamma",))
    rows = []
    index = 1
    for domain in ("person", "place", "organization", "creative_work"):
        for _ in range(32):
            rows.append(
                CandidateEntity(
                    entity_id=f"entity-{index}",
                    qid=f"Q{index}",
                    name=f"Entity {index}",
                    coarse_type=domain,
                    split="mechanism_train",
                    source_query="registered-query-v2",
                    source_provenance="Wikidata CC0 v2",
                    screening_aliases=aliases,
                )
            )
            index += 1

    assert fa_cli._screening_required_count(rows, config) == 64
    with pytest.raises(ValueError, match="exact registered 2x source pool"):
        fa_cli._screening_required_count(rows[:-1], config)


def test_screening_excludes_candidates_without_complete_surface_match():
    aliases = (("alpha",), ("beta",), ("gamma",))
    candidates = (
        CandidateEntity(
            entity_id="entity-1",
            qid="Q1",
            name="Old Vale",
            coarse_type="place",
            split="mechanism_train",
            source_query="registered-query-v3",
            source_provenance="Wikidata CC0 v3",
            screening_aliases=aliases,
        ),
        CandidateEntity(
            entity_id="entity-2",
            qid="Q2",
            name="Oppenheimer",
            coarse_type="creative_work",
            split="mechanism_train",
            source_query="registered-query-v3",
            source_provenance="Wikidata CC0 v3",
            screening_aliases=aliases,
        ),
    )
    synthetic = tuple(
        SyntheticCandidate(
            candidate_id=f"syn-entity-1-v{index:02d}",
            name=name,
            coarse_type="place",
            split="mechanism_train",
            generator_revision="fa-confirmatory-pseudonyms-v2",
        )
        for index, name in enumerate(
            ("New Vale", "Red Vale", "Sun Vale"),
            start=1,
        )
    )

    assert fa_cli._matchable_screening_candidates(
        candidates,
        synthetic,
        FakeTokenizer(),
        required_variants=3,
        generator_revision="fa-confirmatory-pseudonyms-v2",
    ) == (candidates[0],)
    assert fa_cli._matchable_screening_candidates(
        candidates,
        synthetic[:1],
        FakeTokenizer(),
        required_variants=3,
        generator_revision="fa-confirmatory-pseudonyms-v2",
    ) == ()


def test_confirmatory_screening_binds_the_v4_source_revision(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = FAConfig.from_json(
        root / "configs" / "familiarity_answerability_gemma2_2b.json"
    )
    source = tmp_path / "confirmatory_source_v4"
    source.mkdir()
    candidates = source / "candidate_entities_mechanism_train_v1.json"
    questions = source / "screening_questions_mechanism_train_v1.json"
    snapshot = source / "source_snapshot_v1.json"
    synthetic = source / "synthetic_candidates_mechanism_train_v1.json"
    synthetic_snapshot = source / "synthetic_source_snapshot_v1.json"
    candidates.write_text("[]\n", encoding="utf-8")
    questions.write_text("[]\n", encoding="utf-8")
    synthetic.write_text("[]\n", encoding="utf-8")
    synthetic_snapshot.write_text(
        json.dumps({"generator_revision": "test-generator"}),
        encoding="utf-8",
    )
    snapshot.write_text(
        json.dumps({"source_revision": fa_cli.CONFIRMATORY_SOURCE_REVISION}),
        encoding="utf-8",
    )
    integrity = source / "source_integrity_v1.json"
    integrity.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_revision": fa_cli.CONFIRMATORY_SOURCE_REVISION,
                "source_matching_policy_sha256": (
                    fa_cli.fa_confirmatory_source.source_matching_policy_sha256()
                ),
                "source_snapshot": str(snapshot),
                "source_snapshot_sha256": hashlib.sha256(
                    snapshot.read_bytes()
                ).hexdigest(),
                "synthetic_snapshot": str(synthetic_snapshot),
                "synthetic_snapshot_sha256": hashlib.sha256(
                    synthetic_snapshot.read_bytes()
                ).hexdigest(),
                "synthetic_files": {
                    "mechanism_train": {
                        "path": str(synthetic),
                        "sha256": hashlib.sha256(
                            synthetic.read_bytes()
                        ).hexdigest(),
                    }
                },
                "materialized_files": {
                    "mechanism_train": {
                        "candidate_manifest": str(candidates),
                        "candidate_sha256": hashlib.sha256(
                            candidates.read_bytes()
                        ).hexdigest(),
                        "question_manifest": str(questions),
                        "question_sha256": hashlib.sha256(
                            questions.read_bytes()
                        ).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    integrity_hash = fa_cli._verify_confirmatory_source_inputs(
        config,
        tmp_path,
        "mechanism_train",
        candidates,
        questions,
        integrity,
    )

    assert isinstance(integrity_hash, str)
    assert len(integrity_hash) == 64

    tampered = json.loads(integrity.read_text(encoding="utf-8"))
    tampered["source_matching_policy_sha256"] = "0" * 64
    integrity.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="matching policy hash"):
        fa_cli._verify_confirmatory_source_inputs(
            config,
            tmp_path,
            "mechanism_train",
            candidates,
            questions,
            integrity,
        )


def test_candidate_manifest_hash_binds_selection_order():
    aliases = (("alpha",), ("beta",), ("gamma",))
    candidates = tuple(
        CandidateEntity(
            entity_id=f"entity-{index}",
            qid=f"Q{index}",
            name=f"Entity {index}",
            coarse_type="person",
            split="pilot",
            source_query="registered-query-v1",
            source_provenance="CC0-1.0",
            screening_aliases=aliases,
        )
        for index in (1, 2)
    )

    assert fa_cli._candidate_manifest_sha256(candidates) != (
        fa_cli._candidate_manifest_sha256(tuple(reversed(candidates)))
    )


def test_screening_loader_rejects_namespace_mismatched_to_candidate_split(
    monkeypatch,
):
    config = FAConfig.from_json(CONFIG_PATH)
    candidate = CandidateEntity(
        entity_id="entity-1",
        qid="Q1",
        name="Entity One",
        coarse_type="person",
        split="pilot",
        source_query="registered-query-v1",
        source_provenance="CC0-1.0",
        screening_aliases=(("alpha",), ("beta",), ("gamma",)),
    )
    shard = SimpleNamespace(namespace="circuit_dev")
    monkeypatch.setattr(
        fa_cli,
        "_require_verified_shard_kind",
        lambda *args, **kwargs: shard,
    )
    questions = tuple(
        ScreeningQuestion(
            question_id=f"entity-1-{index}",
            qid="Q1",
            prompt=f"Question {index}",
            accepted_aliases=(answer,),
            source_provenance="CC0-1.0",
        )
        for index, answer in enumerate(("alpha", "beta", "gamma"), start=1)
    )

    with pytest.raises(ValueError, match="namespace does not match"):
        fa_cli._load_verified_screening_completions(
            SimpleNamespace(),
            Path("screening.json"),
            [candidate],
            questions,
            config,
        )


def test_screening_loader_rejects_a_tampered_question_manifest(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    candidate = CandidateEntity(
        entity_id="entity-1",
        qid="Q1",
        name="Entity One",
        coarse_type="person",
        split="pilot",
        source_query="registered-query-v1",
        source_provenance="CC0-1.0",
        screening_aliases=(("alpha",), ("beta",), ("gamma",)),
    )
    questions = tuple(
        ScreeningQuestion(
            question_id=f"entity-1-{index}",
            qid="Q1",
            prompt=f"Question {index}",
            accepted_aliases=(answer,),
            source_provenance="CC0-1.0",
        )
        for index, answer in enumerate(("alpha", "beta", "gamma"), start=1)
    )
    model_hash, tokenizer_hash = fa_cli._config_runtime_hashes(config)
    template_hash = config.chat_template_sha256 or CHAT_TEMPLATE_SHA256
    rows = [
        {
            "kind": "screening_completion",
            "entity_id": candidate.entity_id,
            "qid": candidate.qid,
            "question_id": question.question_id,
            "question_index": index,
            "prompt": question.prompt,
            "accepted_aliases": list(question.accepted_aliases),
            "source_provenance": question.source_provenance,
            "raw_output": answer,
            "answer_text": answer,
            "status": "completed",
            "exception_class": None,
            "generation": dict(config.generation),
            "config_sha256": config.config_hash,
            "model_sha256": model_hash,
            "tokenizer_sha256": tokenizer_hash,
            "chat_template_sha256": template_hash,
        }
        for index, (question, answer) in enumerate(
            zip(questions, ("alpha", "beta", "gamma"), strict=True)
        )
    ]
    shard = FAArtifactStore(tmp_path).write_completed_shard(
        config.run_id,
        "pilot",
        "screening-question-lineage",
        rows,
        {
            "config_sha256": config.config_hash,
            "candidate_manifest_sha256": fa_cli._candidate_manifest_sha256(
                [candidate]
            ),
            "questions_manifest_sha256": fa_cli._screening_question_manifest_sha256(
                questions
            ),
            "model_sha256": model_hash,
            "tokenizer_sha256": tokenizer_hash,
            "chat_template_sha256": template_hash,
        },
        record_kind="screening_completion",
    )
    tampered = (
        replace(questions[0], prompt="Tampered question"),
        questions[1],
        questions[2],
    )

    with pytest.raises(ValueError, match="questions_manifest_sha256"):
        fa_cli._load_verified_screening_completions(
            FAArtifactStore(tmp_path),
            shard.manifest_path,
            [candidate],
            tampered,
            config,
        )


def test_screening_loader_rejects_a_tampered_rendered_prompt_hash(
    tmp_path, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    candidate = CandidateEntity(
        entity_id="entity-1",
        qid="Q1",
        name="Entity One",
        coarse_type="person",
        split="pilot",
        source_query="registered-query-v1",
        source_provenance="CC0-1.0",
        screening_aliases=(("alpha",), ("beta",), ("gamma",)),
    )
    questions = tuple(
        ScreeningQuestion(
            question_id=f"entity-1-{index}",
            qid="Q1",
            prompt=f"Question {index}",
            accepted_aliases=(answer,),
            source_provenance="CC0-1.0",
        )
        for index, answer in enumerate(("alpha", "beta", "gamma"), start=1)
    )
    model_hash, tokenizer_hash = fa_cli._config_runtime_hashes(config)
    template_hash = config.chat_template_sha256 or CHAT_TEMPLATE_SHA256
    rows = [
        {
            "kind": "screening_completion",
            "entity_id": candidate.entity_id,
            "qid": candidate.qid,
            "question_id": question.question_id,
            "question_index": index,
            "prompt": question.prompt,
            "accepted_aliases": list(question.accepted_aliases),
            "source_provenance": question.source_provenance,
            "rendered_prompt_sha256": (
                "0" * 64
                if index == 0
                else fa_cli._screening_rendered_prompt_sha256(
                    config, FakeTokenizer(), question.prompt
                )
            ),
            "raw_output": answer,
            "answer_text": answer,
            "status": "completed",
            "exception_class": None,
            "generation": dict(config.generation),
            "config_sha256": config.config_hash,
            "model_sha256": model_hash,
            "tokenizer_sha256": tokenizer_hash,
            "chat_template_sha256": template_hash,
        }
        for index, (question, answer) in enumerate(
            zip(questions, ("alpha", "beta", "gamma"), strict=True)
        )
    ]
    shard = SimpleNamespace(
        namespace="pilot",
        manifest_path=tmp_path / "screening.jsonl.manifest.json",
        data_path=tmp_path / "screening.jsonl",
    )
    monkeypatch.setattr(
        fa_cli,
        "_require_verified_shard_kind",
        lambda *args, **kwargs: shard,
    )
    monkeypatch.setattr(fa_cli, "_verify_artifact_run_id", lambda *args: None)
    monkeypatch.setattr(fa_cli, "_verify_shard_lineage", lambda *args: None)
    monkeypatch.setattr(fa_cli, "_read_json_rows", lambda *args: rows)

    with pytest.raises(ValueError, match="invalid data or provenance"):
        fa_cli._load_verified_screening_completions(
            SimpleNamespace(),
            shard.manifest_path,
            [candidate],
            questions,
            config,
            tokenizer=FakeTokenizer(),
        )


def test_screening_generation_writes_provenance_bound_completion_shard(
    tmp_path, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    candidates = tmp_path / "candidates.json"
    questions = tmp_path / "questions.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "entity_id": "Q90",
                    "qid": "Q90",
                    "name": "Old Vale",
                    "coarse_type": "place",
                    "split": "pilot",
                    "source_query": "registered-query-v1",
                    "source_provenance": "CC0-1.0",
                    "screening_aliases": [["alpha"], ["beta"], ["gamma"]],
                }
            ]
        ),
        encoding="utf-8",
    )
    questions.write_text(
        json.dumps(
            [
                {
                    "question_id": f"Q90-{index}",
                    "qid": "Q90",
                    "prompt": f"Question {index}",
                    "accepted_aliases": [answer],
                    "source_provenance": "CC0-1.0",
                }
                for index, answer in enumerate(("alpha", "beta", "gamma"), start=1)
            ]
        ),
        encoding="utf-8",
    )

    class FakeScreeningRunner:
        model_id = config.model_id
        model_revision = config.model_revision
        tokenizer_revision = config.tokenizer_revision
        chat_template_sha256 = CHAT_TEMPLATE_SHA256

        def render_prompt(self, value):
            return f"rendered:{value}"

        def generate(self, prompts, generation):
            assert prompts == (
                "rendered:Question 1",
                "rendered:Question 2",
                "rendered:Question 3",
            )
            assert generation == dict(config.generation)
            return ("<think>\n</think>\nalpha", "beta", "gamma")

    monkeypatch.setattr(fa_cli, "HFModelRunner", lambda _config: FakeScreeningRunner())
    payload = fa_cli._run_screening(
        config,
        tmp_path,
        SimpleNamespace(
            candidates_manifest=candidates,
            questions_manifest=questions,
            shard_id="screening-0001",
            namespace="pilot",
        ),
    )

    shard = FAArtifactStore(tmp_path).verify_shard(payload["shard_manifest"])
    rows = fa_cli._read_json_rows(shard.data_path)
    assert shard.record_kind == "screening_completion"
    assert payload["count"] == 3
    assert tuple(row["answer_text"] for row in rows) == ("alpha", "beta", "gamma")
    assert all(row["config_sha256"] == config.config_hash for row in rows)


@pytest.mark.parametrize(
    ("raw_output", "answer"),
    [
        ("<think>\nwork\n</think>\nalpha", "alpha"),
        (": Radioactivity", "Radioactivity"),
        ("Answer: New York", "New York"),
        ("first line\nFinal: 'physics'", "physics"),
        ("plain answer.", "plain answer."),
        ("", ""),
    ],
)
def test_screening_answer_extraction_is_deterministic(raw_output, answer):
    assert fa_cli._screening_answer_text(raw_output) == answer


def test_screening_parser_hash_binds_the_parser_implementation():
    expected = fa_cli._sha256_json(
        {
            "revision": "fa-screening-answer-v1",
            "implementation": inspect.getsource(fa_cli._screening_answer_text),
            "rules": (
                "strip",
                "suffix-after-final-think-close",
                "last-nonempty-line",
                "suffix-after-final-colon",
                "single-matching-quote-pair",
            ),
        }
    )

    assert fa_cli._screening_parser_sha256() == expected


def test_matching_policy_hash_binds_surface_and_assignment_implementation():
    expected = fa_cli._sha256_json(
        {
            "revision": "fa-entity-matching-v5",
            "source_matching_policy_sha256": (
                fa_cli.fa_confirmatory_source.source_matching_policy_sha256()
            ),
            "character_tolerance": fa_cli.fa_entities.CHARACTER_TOLERANCE,
            "sentence_frame": fa_cli.fa_entities.TOKENIZER_SENTENCE_FRAME,
            "same_string_facts": fa_cli.fa_entities.SAME_STRING_EXPOSURE_FACTS,
            "confirmatory_reserve_per_domain": (
                fa_cli._CONFIRMATORY_RESERVE_PER_DOMAIN
            ),
            "implementations": {
                "source_matchability_filter": inspect.getsource(
                    fa_cli.fa_confirmatory_source.filter_matchable_source_records
                ),
                "pseudonym_generator": inspect.getsource(
                    fa_cli.fa_confirmatory_synthetics.generate_synthetic_candidates
                ),
                "pseudonym_proposal": inspect.getsource(
                    fa_cli.fa_confirmatory_synthetics._pseudonym
                ),
                "matchability_filter": inspect.getsource(
                    fa_cli._matchable_screening_candidates
                ),
                "selection": inspect.getsource(
                    fa_cli._select_domain_balanced_candidates
                ),
                "match": inspect.getsource(
                    fa_cli.fa_entities.match_synthetic_entities
                ),
                "assignment": inspect.getsource(
                    fa_cli.fa_entities._deterministic_assignment
                ),
                "make_match": inspect.getsource(fa_cli.fa_entities._make_match),
                "surface": inspect.getsource(
                    fa_cli.fa_entities._surface_compatible
                ),
                "token_count": inspect.getsource(fa_cli.fa_entities._token_count),
                "same_string_prefix": inspect.getsource(
                    fa_cli.fa_entities.render_same_string_exposure_prefix
                ),
                "same_string_token_count": inspect.getsource(
                    fa_cli.fa_entities._same_string_token_count
                ),
            },
        }
    )

    assert fa_cli._matching_policy_sha256() == expected


def test_screening_artifact_write_is_idempotent_and_fail_closed(tmp_path):
    store = FAArtifactStore(tmp_path)
    rows = [{"kind": "screening_audit", "decision": "passed"}]
    lineage = {"config_sha256": "a" * 64}

    first = fa_cli._write_or_verify_screening_shard(
        store,
        "smoke-v1",
        "pilot",
        "idempotent-screening-audit",
        rows,
        lineage,
        record_kind="screening_audit",
    )
    second = fa_cli._write_or_verify_screening_shard(
        store,
        "smoke-v1",
        "pilot",
        "idempotent-screening-audit",
        rows,
        lineage,
        record_kind="screening_audit",
    )

    assert second.manifest_path == first.manifest_path
    assert second.sha256 == first.sha256
    with pytest.raises(ValueError, match="rows do not match"):
        fa_cli._write_or_verify_screening_shard(
            store,
            "smoke-v1",
            "pilot",
            "idempotent-screening-audit",
            [{"kind": "screening_audit", "decision": "stopped"}],
            lineage,
            record_kind="screening_audit",
        )


def test_dataset_audit_uses_the_pinned_model_tokenizer(tmp_path, monkeypatch):
    config = FAConfig.from_json(CONFIG_PATH)
    sentinel = object()
    rows = (
        SimpleNamespace(block="factorial"),
        SimpleNamespace(block="same_string"),
    )
    monkeypatch.setattr(
        fa_cli,
        "_load_manifest",
        lambda *args, **kwargs: SimpleNamespace(examples=rows),
    )
    monkeypatch.setattr(
        fa_cli,
        "load_pinned_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(tokenizer=sentinel),
    )
    observed = []
    monkeypatch.setattr(
        fa_cli,
        "audit_dataset",
        lambda factorial, same_string, *, tokenizer: observed.append(tokenizer)
        or SimpleNamespace(passed=True, checks={"tokenizer": True}, violations=()),
    )

    payload = fa_cli._audit_manifest(
        config,
        tmp_path,
        SimpleNamespace(manifest=tmp_path / "manifest.json"),
    )

    assert payload["status"] == "passed"
    assert observed == [sentinel]


def test_confirmatory_reserve_selection_is_balanced_deterministic_and_excludes_rejected():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    config = FAConfig.from_json(config_path)
    matches = confirmatory_reserve_matches(config)
    rejected = matches[0].pair_id
    accepted = tuple(match.pair_id for match in matches if match.pair_id != rejected)
    audit = NaturalnessAudit(
        accepted_pair_ids=accepted,
        excluded_pair_ids=(rejected,),
        third_rater_pair_ids=(),
        decisions={
            match.pair_id: "excluded_malformed" if match.pair_id == rejected else "accepted"
            for match in matches
        },
    )

    selected = fa_cli._select_confirmatory_matches(config, matches, audit)
    reversed_selected = fa_cli._select_confirmatory_matches(config, tuple(reversed(matches)), audit)

    assert selected == reversed_selected
    assert len(selected) == sum(config.split_counts.values())
    assert rejected not in {match.pair_id for match in selected}
    for split, split_count in config.split_counts.items():
        expected = split_count // 4
        assert {
            domain: sum(
                match.split == split and match.coarse_type == domain for match in selected
            )
            for domain in ("person", "place", "organization", "creative_work")
        } == {
            "person": expected,
            "place": expected,
            "organization": expected,
            "creative_work": expected,
        }


def sha256_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def probe_rows_for_examples(examples):
    rows = []
    domains = ("person", "place", "organization", "creative_work")
    for index, example in enumerate(sorted(examples, key=lambda row: row.example_id)):
        residual = np.zeros((3, 26, 4), dtype=np.float64)
        residual[:, :, 0] = float(index % 2)
        for task in ("familiarity", "answerability", "unsupported_answer"):
            if task == "unsupported_answer" and example.answerability == "target_bound":
                continue
            if task == "familiarity":
                label = int(example.target_familiarity == "screened_real")
            elif task == "answerability":
                label = example.answerability
            else:
                label = index % 2
            rows.append(
                ProbeRow(
                    example_id=example.example_id,
                    split=example.split,
                    task=task,
                    label=label,
                    entity_id=f"{example.split}-entity-{index}",
                    template_id=f"{example.split}-template-{index}",
                    relation_id=f"{example.split}-relation-{index}",
                    domain=domains[index % len(domains)],
                    condition=f"condition-{index}",
                    answerability_condition=example.answerability,
                    target_familiarity_condition=example.target_familiarity,
                    distractor_familiarity_condition=example.distractor_familiarity,
                    surface_features=(float(index),),
                    output_margin_features=tuple(float(index) for _ in range(11)),
                    residual_features=residual,
                    sae_features=None,
                    outcome_status="valid",
                    source_sha256=example.canonical_payload_sha256,
                    activation_sha256=sha256_json(
                        {"activation": example.example_id}
                    ),
                    metadata_manifest_sha256=sha256_json(
                        {"metadata": example.split}
                    ),
                    metadata_row_sha256=sha256_json(
                        {"metadata-row": example.example_id}
                    ),
                    output_control_schema_sha256=OUTPUT_CONTROL_SCHEMA_SHA256,
                    output_evidence_sha256=sha256_json(
                        {"output": example.example_id}
                    ),
                )
            )
    return tuple(rows)


def write_probe_rows_artifact(root, config, split, rows, *, lineage=None):
    return FAArtifactStore(root).write_completed_shard(
        config.run_id,
        split,
        f"probe-rows-{split}",
        [{"kind": "probe_rows", "row": row.to_record()} for row in rows],
        {"config_sha256": config.config_hash, **(lineage or {})},
        record_kind="probe_rows",
    )


def naturalness_ratings_manifest(root, config, *, rating_schema_version=1):
    store = FAArtifactStore(root)
    preregistration = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "familiarity_answerability_preregistration.md"
    )
    protocol_sha256 = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    rating_protocol = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "fa_naturalness_rating_protocol.md"
    )
    rating_protocol_sha256 = hashlib.sha256(
        rating_protocol.read_bytes()
    ).hexdigest()
    matches = (EntityMatch(**MATCH),)
    packet_dir = Path(root) / f"fixture-packets-{rating_schema_version}"
    prepared = prepare_initial_rating_packets(
        matches,
        config_sha256=config.config_hash,
        protocol_sha256=protocol_sha256,
        rating_protocol_sha256=rating_protocol_sha256,
        output_dir=packet_dir,
        rater_ids=("rater-a", "rater-b"),
    )
    issuance_row, issuance_lineage = packet_issuance_record(
        prepared["private_key"]
    )
    issuance = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        f"rating-issuance-{rating_schema_version}",
        [issuance_row],
        issuance_lineage,
        record_kind="naturalness_packet_issuance",
    )
    fa_cli._restrict_private_naturalness_shard(issuance)
    response_paths = tuple(
        packet_dir / "public" / f"{rater_id}-response.csv"
        for rater_id in ("rater-a", "rater-b")
    )
    for response_path in response_paths:
        with response_path.open(newline="", encoding="utf-8") as handle:
            response_rows = list(csv.DictReader(handle))
            fieldnames = tuple(response_rows[0])
        for response in response_rows:
            for candidate in ("a", "b"):
                response[f"candidate_{candidate}_naturalness"] = "4"
                response[f"candidate_{candidate}_type_fit"] = "4"
                response[f"candidate_{candidate}_malformed"] = "false"
            response["independence_attested"] = "true"
        with response_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(response_rows)
    rating_values, assignments, disagreements, responses = (
        compile_initial_responses_from_issuance(
            matches,
            issuance=issuance_row,
            response_paths=response_paths,
            config_sha256=config.config_hash,
            protocol_sha256=protocol_sha256,
            rating_protocol_sha256=rating_protocol_sha256,
        )
    )
    submission_row, submission_lineage = submission_record(
        rating_values,
        assignments,
        responses,
        config_sha256=config.config_hash,
        issuance_manifest=str(issuance.manifest_path.relative_to(store.root)),
        issuance_sha256=issuance.sha256,
        disagreement_pair_ids=disagreements,
    )
    submission = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        f"rating-submission-{rating_schema_version}",
        [submission_row],
        submission_lineage,
        record_kind="naturalness_submission",
    )
    ratings = [asdict(value) for value in rating_values]
    ratings[0]["schema_version"] = rating_schema_version
    blinding_sha256 = sha256_json(assignments)
    row = {
        "kind": "naturalness_ratings",
        "schema_version": 1,
        "config_sha256": config.config_hash,
        "protocol_sha256": protocol_sha256,
        "blinding_manifest_sha256": blinding_sha256,
        "assignments": assignments,
        "ratings": ratings,
    }
    shard = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        f"ratings-{rating_schema_version}",
        [row],
        {
            "config_sha256": config.config_hash,
            "protocol_sha256": protocol_sha256,
            "rating_protocol_sha256": rating_protocol_sha256,
            "blinding_manifest_sha256": blinding_sha256,
            "matches_sha256": naturalness_matches_sha256(
                (EntityMatch(**MATCH),)
            ),
            "initial_submission_manifest": str(
                submission.manifest_path.relative_to(store.root)
            ),
            "initial_submission_sha256": submission.sha256,
        },
        record_kind="naturalness_ratings",
    )
    return shard.manifest_path


def valid_examples(config, split):
    match = dict(MATCH)
    match["split"] = split
    return build_factorial_examples(
        config, (EntityMatch(**match),), tokenizer=FakeTokenizer()
    )


def prompt_capability(
    root, config, split, *, full_hash="b" * 64, template_bytes=CHAT_TEMPLATE_BYTES
):
    examples = valid_examples(config, split)
    store = FAArtifactStore(root)
    template_hash = hashlib.sha256(template_bytes).hexdigest()
    prepared = SimpleNamespace(
        chat_template_bytes=template_bytes,
        chat_template_sha256=template_hash,
    )
    try:
        tokenizer_pin = fa_cli._write_tokenizer_pin(
            store, config, prepared, full_hash
        )
    except FileExistsError:
        tokenizer_pin = store.verify_shard(
            store.root
            / "runs"
            / "familiarity_answerability"
            / config.run_id
            / "shards"
            / ("mechanism_train" if config.profile == "confirmatory" else "pilot")
            / (
                f"tokenizer-pin-{template_hash[:12]}-"
                f"{full_hash[:12]}.jsonl.manifest.json"
            )
        )
    shard = fa_cli._write_prompt_capability(
        store,
        config,
        full_hash,
        split,
        examples,
        template_hash,
        tokenizer_pin,
    )
    return examples, shard


def test_fa_commands_are_registered_with_explicit_config_and_root(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    matches = tmp_path / "matches.json"
    matches.write_text(json.dumps(smoke_pilot_matches()), encoding="utf-8")
    args = parser.parse_args(
        [
            "fa-build-pilot",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--matches-manifest",
            str(matches),
        ]
    )

    assert args.command == "fa-build-pilot"
    assert args.config == str(CONFIG_PATH)
    assert args.root == str(tmp_path)


def test_tokenizer_pin_identity_binds_source_manifest(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    prepared = SimpleNamespace(
        chat_template_bytes=CHAT_TEMPLATE_BYTES,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
    )
    store = FAArtifactStore(tmp_path)

    first = fa_cli._write_tokenizer_pin(store, config, prepared, "a" * 64)
    second = fa_cli._write_tokenizer_pin(store, config, prepared, "b" * 64)

    assert first.manifest_path != second.manifest_path
    assert first.sha256 == second.sha256


def test_screening_parser_requires_explicit_inputs_namespace_and_shard(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    args = parser.parse_args(
        [
            "fa-run-screening",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--candidates-manifest",
            str(tmp_path / "candidates.json"),
            "--questions-manifest",
            str(tmp_path / "questions.json"),
            "--namespace",
            "pilot",
            "--shard-id",
            "screening-0001",
        ]
    )

    assert args.command == "fa-run-screening"
    assert args.namespace == "pilot"
    assert args.shard_id == "screening-0001"


def test_screening_scoring_parser_requires_the_question_manifest(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "fa-screen-entities",
                "--config",
                str(CONFIG_PATH),
                "--root",
                str(tmp_path),
                "--candidates-manifest",
                str(tmp_path / "candidates.json"),
                "--screening-manifest",
                str(tmp_path / "screening.json"),
                "--synthetic-manifest",
                str(tmp_path / "synthetic.json"),
            ]
        )


def test_generation_parser_accepts_explicit_resume_flag(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    args = parser.parse_args(
        [
            "fa-run-generation",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "prompts.jsonl.manifest.json"),
            "--shard-id",
            "0001",
            "--namespace",
            "pilot",
            "--resume",
        ]
    )

    assert args.resume is True


def test_activation_parser_requires_explicit_manifest_namespace_and_shard(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    args = parser.parse_args(
        [
            "fa-extract-activations",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "prompts.jsonl.manifest.json"),
            "--namespace",
            "pilot",
            "--shard-id",
            "0001",
            "--layers",
            "0,4,8",
            "--resume",
        ]
    )

    assert args.namespace == "pilot"
    assert args.shard_id == "0001"
    assert args.layers == "0,4,8"
    assert args.resume is True


def test_probe_materialization_parser_requires_both_capabilities(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    args = parser.parse_args(
        [
            "fa-materialize-probe-rows",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "prompts.jsonl.manifest.json"),
            "--metadata-manifest",
            str(tmp_path / "metadata.jsonl.manifest.json"),
            "--namespace",
            "locked_validation",
            "--shard-id",
            "probe-evidence",
            "--resume",
        ]
    )

    assert args.namespace == "locked_validation"
    assert args.shard_id == "probe-evidence"
    assert args.resume is True


def test_activation_cli_wires_verified_prompt_to_resumable_writer(
    tmp_path, capsys, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, "pilot")
    calls = {}

    class FakeModelRunner:
        def __init__(self, supplied):
            assert supplied == config
            self.model = object()
            self.tokenizer = object()
            self.model_id = supplied.model_id
            self.model_revision = supplied.model_revision
            self.tokenizer_revision = supplied.tokenizer_revision
            self.chat_template_sha256 = CHAT_TEMPLATE_SHA256

        def generate(self, prompts, generation):
            raise AssertionError("activation extraction must not generate completions")

    class FakeSelectedRunner:
        def __init__(self, model, tokenizer, **pins):
            calls["runner"] = (model, tokenizer, pins)

    def fake_write(runner, examples, registered_layers, *, destination):
        calls["write"] = (runner, tuple(examples), tuple(registered_layers), destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        for path in (
            destination,
            destination.with_suffix(".jsonl"),
            destination.with_suffix(".manifest.json"),
        ):
            path.write_bytes(b"sealed")
        return SimpleNamespace(
            manifest_path=destination.with_suffix(".manifest.json"),
            request_sha256="a" * 64,
            row_count=len(examples),
        )

    monkeypatch.setattr(fa_cli, "HFModelRunner", FakeModelRunner)
    monkeypatch.setattr(fa_cli, "_ACTIVATION_RUNNER_FACTORY", FakeSelectedRunner)
    monkeypatch.setattr(fa_cli, "_ACTIVATION_SHARD_WRITER", fake_write)

    exit_code = cli.main(
        [
            "fa-extract-activations",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--namespace",
            "pilot",
            "--shard-id",
            "0001",
            "--layers",
            "0,4,8",
            "--resume",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "extracted"
    assert payload["request_sha256"] == "a" * 64
    assert calls["write"][2] == (0, 4, 8)
    assert calls["write"][3] == (
        tmp_path
        / "runs"
        / "familiarity_answerability"
        / config.run_id
        / "activations"
        / "pilot"
        / "0001.npz"
    )


def test_generic_activation_cli_rejects_protected_namespace_before_model_load(
    tmp_path, capsys, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, "probe_test")

    class MustNotConstruct:
        def __init__(self, _config):
            raise AssertionError("protected extraction must use its endpoint transaction")

    monkeypatch.setattr(fa_cli, "HFModelRunner", MustNotConstruct)
    exit_code = cli.main(
        [
            "fa-extract-activations",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--namespace",
            "probe_test",
            "--shard-id",
            "0001",
            "--layers",
            "0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "error"
    assert "protected test namespaces" in payload["error"]["message"]


def test_probe_materialization_writes_compact_provenance_bound_evidence(
    tmp_path, capsys, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    examples, prompts = prompt_capability(tmp_path, config, "locked_validation")
    metadata = fa_cli._write_probe_metadata(
        FAArtifactStore(tmp_path),
        config,
        "b" * 64,
        (EntityMatch(**MATCH),),
        examples,
    )

    class FakeModelRunner:
        def __init__(self, supplied):
            self.model = object()
            self.tokenizer = object()
            self.model_id = supplied.model_id
            self.model_revision = supplied.model_revision
            self.tokenizer_revision = supplied.tokenizer_revision
            self.chat_template_sha256 = CHAT_TEMPLATE_SHA256

        def generate(self, rendered, generation):
            return ["UNKNOWN" for _ in rendered]

    def fake_activation_writer(runner, supplied, layers, *, destination):
        assert tuple(layers) == tuple(range(26))
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = destination.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "request_sha256": "1" * 64,
                    "npz_sha256": "2" * 64,
                    "index_sha256": "3" * 64,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(
            manifest_path=manifest_path,
            request_sha256="1" * 64,
            npz_sha256="2" * 64,
            index_sha256="3" * 64,
        )

    activation_records = tuple(
        SimpleNamespace(
            example_id=example.example_id,
            activation_sha256=sha256_json({"activation": example.example_id}),
        )
        for example in examples
    )
    output_evidence = tuple(
        OutputEvidence(
            example_id=example.example_id,
            source_sha256=example.canonical_payload_sha256,
            target_code=example.registry_code,
            unknown_suffix="UNKNOWN",
            target_logp=-2.0,
            unknown_logp=-1.0,
            prompt_bytes=f"prompt:{example.example_id}".encode("utf-8"),
            rendered_prompt_sha256=hashlib.sha256(
                f"prompt:{example.example_id}".encode("utf-8")
            ).hexdigest(),
            prompt_input_ids=(1,),
            target_token_ids=(2,),
            unknown_token_ids=(3,),
            model_id=config.model_id,
            model_revision=config.model_revision,
            tokenizer_id=config.model_id,
            tokenizer_revision=config.tokenizer_revision,
            tokenizer_config_sha256="4" * 64,
            chat_template_sha256=CHAT_TEMPLATE_SHA256,
            config_sha256=config.config_hash,
        )
        for example in sorted(examples, key=lambda row: row.example_id)
    )

    monkeypatch.setattr(fa_cli, "HFModelRunner", FakeModelRunner)
    monkeypatch.setattr(
        fa_cli, "_ACTIVATION_RUNNER_FACTORY", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(fa_cli, "_ACTIVATION_SHARD_WRITER", fake_activation_writer)
    monkeypatch.setattr(fa_cli, "load_activation_records", lambda path: activation_records)
    monkeypatch.setattr(fa_cli, "_PROBE_SCORER_FACTORY", lambda *args: object())
    monkeypatch.setattr(
        fa_cli,
        "_PROBE_ROW_MATERIALIZER",
        lambda supplied, activations, scorer, metadata, *, unsupported_outcomes: (
            probe_rows_for_examples(supplied),
            output_evidence,
        ),
    )

    exit_code = cli.main(
        [
            "fa-materialize-probe-rows",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--metadata-manifest",
            str(metadata.manifest_path),
            "--namespace",
            "locked_validation",
            "--shard-id",
            "probe-evidence",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "materialized"
    assert payload["compact_evidence_count"] == len(examples)
    shard = FAArtifactStore(tmp_path).verify_shard(payload["probe_rows_manifest"])
    compact = fa_cli._read_json_rows(shard.data_path)
    assert all(row["schema_version"] == 2 for row in compact)
    assert all("residual_features" not in row for row in compact)
    lineage = json.loads(shard.manifest_path.read_text(encoding="utf-8"))["lineage"]
    assert lineage["prompt_manifest_sha256"] == prompts.sha256
    assert lineage["metadata_manifest_sha256"] == metadata.sha256
    assert lineage["materialization_schema_sha256"] == (
        fa_cli._probe_materialization_schema_sha256()
    )
    reconstructed, _ = fa_cli._load_probe_rows_manifest(
        FAArtifactStore(tmp_path),
        shard.manifest_path,
        config,
        expected_namespace="locked_validation",
    )
    assert len(reconstructed) == len(probe_rows_for_examples(examples))

    class MustNotReloadModel:
        def __init__(self, supplied):
            raise AssertionError("verified probe evidence resume must not reload the model")

    monkeypatch.setattr(fa_cli, "HFModelRunner", MustNotReloadModel)
    resume_code = cli.main(
        [
            "fa-materialize-probe-rows",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--metadata-manifest",
            str(metadata.manifest_path),
            "--namespace",
            "locked_validation",
            "--shard-id",
            "probe-evidence",
            "--resume",
        ]
    )
    resumed = json.loads(capsys.readouterr().out)
    assert resume_code == 0
    assert resumed["status"] == "recovered"
    assert resumed["probe_rows_manifest"] == payload["probe_rows_manifest"]


def test_public_probe_materialization_rejects_probe_test_before_model_load(
    tmp_path, capsys, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompt = prompt_capability(tmp_path, config, "probe_test")

    class MustNotReloadModel:
        def __init__(self, supplied):
            raise AssertionError("protected evidence must not load before endpoint unlock")

    monkeypatch.setattr(fa_cli, "HFModelRunner", MustNotReloadModel)
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "fa-materialize-probe-rows",
                "--config",
                str(CONFIG_PATH),
                "--root",
                str(tmp_path),
                "--manifest",
                str(prompt.manifest_path),
                "--metadata-manifest",
                str(tmp_path / "unused.json"),
                "--namespace",
                "probe_test",
                "--shard-id",
                "forbidden",
            ]
        )
    payload = json.loads(capsys.readouterr().out)
    assert raised.value.code == 2
    assert payload["error"]["type"] == "ArgumentError"


def test_behavior_test_command_closes_one_use_endpoint_with_canonical_metrics(
    tmp_path, capsys, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, "behavior_test")
    preregistration_hash = hashlib.sha256(
        (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "familiarity_answerability_preregistration.md"
        ).read_bytes()
    ).hexdigest()
    selection_hash = "d" * 64
    store = FAArtifactStore(tmp_path)
    store.seal_endpoint(
        "behavior_test",
        (prompts,),
        {
            "preregistration": preregistration_hash,
            "selection_manifest": selection_hash,
        },
    )

    class CanonicalRecord:
        def __init__(self, value):
            self.value = value

        def to_record(self):
            return dict(self.value)

    monkeypatch.setattr(fa_cli, "HFModelRunner", FakeRunner)
    monkeypatch.setattr(
        fa_cli,
        "_BEHAVIOR_BOOTSTRAP",
        lambda rows, replicates, seed: CanonicalRecord(
            {"replicates": replicates, "rows": len(rows), "seed": seed}
        ),
    )
    monkeypatch.setattr(
        fa_cli,
        "_BEHAVIOR_GATE",
        lambda metrics, bootstrap, **kwargs: CanonicalRecord(
            {"status": "not_evaluable", "config_hash": kwargs["config_hash"]}
        ),
    )

    exit_code = cli.main(
        [
            "fa-evaluate-behavior-test",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--shard-id",
            "confirmatory-0001",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "evaluated"
    assert payload["endpoint_state"] == "closed"
    assert store.endpoint_state("behavior_test", prompts.manifest_path) == "closed"
    metrics = store.verify_shard(payload["metrics_manifest"])
    assert metrics.record_kind == "metrics"
    assert metrics.namespace == "behavior_test"


def test_fa_dispatch_is_isolated_and_cli_routes_fa_commands(tmp_path, capsys, monkeypatch):
    install_fake_tokenizer(monkeypatch)
    args = argparse.Namespace(command="rlmf-prepare-data")
    assert dispatch_fa(args) is None

    config = FAConfig.from_json(CONFIG_PATH)
    matches = screened_matches_manifest(tmp_path, config, smoke_pilot_matches())
    exit_code = cli.main(
        [
            "fa-build-pilot",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--matches-manifest",
            str(matches),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["command"] == "fa-build-pilot"
    assert payload["status"] == "built"


def test_pilot_prompt_capability_references_verified_tokenizer_pin(tmp_path, monkeypatch):
    install_fake_tokenizer(monkeypatch)
    config = FAConfig.from_json(CONFIG_PATH)
    matches = screened_matches_manifest(tmp_path, config, smoke_pilot_matches())

    payload = fa_cli._build_manifest(
        config,
        tmp_path,
        SimpleNamespace(matches_manifest=matches),
        confirmatory=False,
    )
    resumed = fa_cli._build_manifest(
        config,
        tmp_path,
        SimpleNamespace(matches_manifest=matches),
        confirmatory=False,
    )
    store = FAArtifactStore(tmp_path)
    prompt = store.verify_shard(payload["manifest"])
    pin = store.verify_shard(payload["tokenizer_pin_manifest"])
    prompt_row = fa_cli._read_json_rows(prompt.data_path)[0]
    prompt_lineage = json.loads(prompt.manifest_path.read_text(encoding="utf-8"))[
        "lineage"
    ]
    pin_row = fa_cli._read_json_rows(pin.data_path)[0]
    block_counts = Counter(
        example["block"] for example in prompt_row["examples"]
    )

    assert resumed == payload
    assert payload["count"] == 320
    assert block_counts == Counter({"factorial": 288, "same_string": 32})
    assert prompt_row["tokenizer_pin_manifest"] == str(
        pin.manifest_path.relative_to(store.root)
    )
    assert prompt_row["tokenizer_pin_sha256"] == pin.sha256
    assert prompt_lineage["tokenizer_pin_sha256"] == pin.sha256
    assert hashlib.sha256(bytes.fromhex(pin_row["chat_template_utf8_hex"])).hexdigest() == (
        pin_row["chat_template_sha256"]
    )


def test_pilot_accepts_audited_selection_independent_of_match_row_order(
    tmp_path, monkeypatch
):
    install_fake_tokenizer(monkeypatch)
    config = FAConfig.from_json(CONFIG_PATH)
    rows = smoke_pilot_matches()
    matches = screened_matches_manifest(
        tmp_path,
        config,
        rows,
        audited_entity_ids=reversed(
            [row["real_entity_id"] for row in rows]
        ),
    )

    payload = fa_cli._build_manifest(
        config,
        tmp_path,
        SimpleNamespace(matches_manifest=matches),
        confirmatory=False,
    )

    assert payload["status"] == "built"


def test_pilot_rejects_audited_selection_membership_mismatch(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    rows = smoke_pilot_matches()
    audited_ids = [row["real_entity_id"] for row in rows]
    audited_ids[-1] = "entity-not-in-match-shard"
    matches = screened_matches_manifest(
        tmp_path,
        config,
        rows,
        audited_entity_ids=audited_ids,
    )

    with pytest.raises(ValueError, match="audited selection"):
        fa_cli._build_manifest(
            config,
            tmp_path,
            SimpleNamespace(matches_manifest=matches),
            confirmatory=False,
        )


def test_pilot_construction_requires_the_registered_match_count(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    matches = screened_matches_manifest(tmp_path, config, [MATCH])

    with pytest.raises(ValueError, match="exactly 8"):
        fa_cli._build_manifest(
            config,
            tmp_path,
            SimpleNamespace(matches_manifest=matches),
            confirmatory=False,
        )


def test_pilot_rejects_a_stale_matching_policy(tmp_path, monkeypatch):
    config = FAConfig.from_json(CONFIG_PATH)
    matches = screened_matches_manifest(tmp_path, config, smoke_pilot_matches())
    monkeypatch.setattr(fa_cli, "_matching_policy_sha256", lambda: "0" * 64)

    with pytest.raises(ValueError, match="matching policy"):
        fa_cli._build_manifest(
            config,
            tmp_path,
            SimpleNamespace(matches_manifest=matches),
            confirmatory=False,
        )


def test_pilot_construction_rejects_raw_screened_matches(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    raw_matches = tmp_path / "matches.json"
    raw_matches.write_text(json.dumps(smoke_pilot_matches()), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable shard manifest"):
        fa_cli._build_manifest(
            config,
            tmp_path,
            SimpleNamespace(matches_manifest=raw_matches),
            confirmatory=False,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("real_entity", "duplicate real entity IDs"),
        ("real_qid", "duplicate real QIDs"),
        ("synthetic", "duplicate synthetic candidate IDs"),
        ("domain", "not exactly domain balanced"),
    ],
)
def test_pilot_construction_rejects_invalid_screened_match_structure(
    tmp_path, mutation, message
):
    config = FAConfig.from_json(CONFIG_PATH)
    rows = smoke_pilot_matches()
    if mutation == "real_entity":
        rows[1]["real_entity_id"] = rows[0]["real_entity_id"]
        rows[1]["pair_id"] = (
            f"{rows[1]['real_entity_id']}--{rows[1]['synthetic_candidate_id']}"
        )
    elif mutation == "real_qid":
        rows[1]["real_qid"] = rows[0]["real_qid"]
    elif mutation == "synthetic":
        rows[1]["synthetic_candidate_id"] = rows[0]["synthetic_candidate_id"]
        rows[1]["synthetic_name"] = rows[0]["synthetic_name"]
        rows[1]["synthetic_token_count"] = rows[0]["synthetic_token_count"]
        rows[1]["synthetic_word_count"] = rows[0]["synthetic_word_count"]
        rows[1]["synthetic_character_count"] = rows[0][
            "synthetic_character_count"
        ]
        rows[1]["character_length_delta"] = (
            rows[1]["synthetic_character_count"]
            - rows[1]["real_character_count"]
        )
        rows[1]["pair_id"] = (
            f"{rows[1]['real_entity_id']}--{rows[1]['synthetic_candidate_id']}"
        )
    else:
        rows[1]["coarse_type"] = "person"
    matches = screened_matches_manifest(tmp_path, config, rows)

    with pytest.raises(ValueError, match=message):
        fa_cli._build_manifest(
            config,
            tmp_path,
            SimpleNamespace(matches_manifest=matches),
            confirmatory=False,
        )


def test_probe_selection_reads_only_hash_bound_prompt_identities(
    tmp_path, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    examples, prompt = prompt_capability(tmp_path, config, "probe_test")

    def forbidden_prompt_payload_read(*args, **kwargs):
        raise AssertionError("selection must not parse protected prompt payloads")

    monkeypatch.setattr(fa_cli, "_read_json_rows", forbidden_prompt_payload_read)
    identities, verified = fa_cli._load_prompt_source_identities(
        FAArtifactStore(tmp_path),
        prompt.manifest_path,
        config,
        expected_namespace="probe_test",
    )

    assert verified.sha256 == prompt.sha256
    assert [identity.example_id for identity in identities] == sorted(
        example.example_id for example in examples
    )
    assert all(len(identity.canonical_payload_sha256) == 64 for identity in identities)


def test_protected_probe_rows_open_only_after_authorization_and_match_task_identities(
    tmp_path,
):
    config = FAConfig.from_json(CONFIG_PATH)
    examples, prompt = prompt_capability(tmp_path, config, "probe_test")
    store = FAArtifactStore(tmp_path)
    task_identities, _ = fa_cli._load_prompt_task_source_identities(
        store,
        prompt.manifest_path,
        config,
        expected_namespace="probe_test",
    )
    rows = probe_rows_for_examples(examples)
    probe_rows = store.write_completed_shard(
        config.run_id,
        "probe_test",
        "probe-rows",
        [{"kind": "probe_rows", "row": row.to_record()} for row in rows],
        {
            "config_sha256": config.config_hash,
            "prompt_manifest_sha256": prompt.sha256,
            "task_source_identities_sha256": fa_cli._task_source_identities_sha256(
                task_identities
            ),
        },
        record_kind="probe_rows",
    )

    with pytest.raises(ValueError, match="require a probe_test authorization"):
        fa_cli._load_probe_rows_manifest(
            store,
            probe_rows.manifest_path,
            config,
            expected_namespace="probe_test",
        )

    authorization = ProbeTestAuthorization.from_unlock_receipt(
        UnlockReceipt(
            endpoint="probe_test",
            lease_id="a" * 32,
            state="unlocked_once",
            preregistration_hash="b" * 64,
            selection_manifest_hash="c" * 64,
        )
    )
    loaded, verified = fa_cli._load_probe_rows_manifest(
        store,
        probe_rows.manifest_path,
        config,
        expected_namespace="probe_test",
        authorization=authorization,
        expected_prompt_sha256=prompt.sha256,
        expected_task_identities=task_identities,
    )

    assert verified.sha256 == probe_rows.sha256
    assert [row.to_record() for row in loaded] == [row.to_record() for row in rows]


def test_core_cli_closes_f1_and_f2a_reports_and_recovers_atomically(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fa_probes, "_TRAIN_ONLY_CV_FAST_PATH_FOR_TESTS", True)
    monkeypatch.setattr(fa_probes, "_BOOTSTRAP_DRAW_OVERRIDE_FOR_TESTS", 80)
    config = FAConfig.from_json(CONFIG_PATH)
    _, behavior_prompt = prompt_capability(tmp_path, config, "behavior_test")
    monkeypatch.setattr(fa_cli, "HFModelRunner", FakeRunner)

    def smoke_behavior_gate(metrics, bootstrap, **kwargs):
        from trajectory_extractor.fa_scoring import (
            CONFIRMATORY_THRESHOLDS,
            behavioral_gate,
        )

        return behavioral_gate(
            metrics,
            bootstrap,
            **{**kwargs, "thresholds": CONFIRMATORY_THRESHOLDS},
        )

    monkeypatch.setattr(fa_cli, "_BEHAVIOR_GATE", smoke_behavior_gate)
    behavior_sealed = fa_cli._seal_behavior_test(
        config,
        tmp_path,
        SimpleNamespace(behavior_test_manifest=behavior_prompt.manifest_path),
    )
    behavior_evaluated = fa_cli._evaluate_behavior_test(
        config,
        tmp_path,
        SimpleNamespace(
            manifest=behavior_prompt.manifest_path,
            shard_id="behavior-smoke",
        ),
    )

    train = write_probe_rows_artifact(
        tmp_path,
        config,
        "mechanism_train",
        probe_rows_for_examples(valid_examples(config, "mechanism_train")),
    )
    validation = write_probe_rows_artifact(
        tmp_path,
        config,
        "locked_validation",
        probe_rows_for_examples(valid_examples(config, "locked_validation")),
    )
    test_examples, prompt = prompt_capability(tmp_path, config, "probe_test")

    def smoke_nulls(
        train_rows,
        validation_rows,
        *,
        seeds,
        protected_test_ids,
        probe_test_source_identities,
        _allow_test_seed_override,
    ):
        del protected_test_ids
        base = fa_probes.fit_selection(train_rows, validation_rows)
        results = []
        for kind in (
            "label_permutation",
            "layer_order",
            "random_map",
            "output_aligned_11d",
        ):
            seed = seeds[0]
            provenance = {"kind": kind, "seed": seed, "config": {"smoke": True}}
            selection = replace(base, null_provenance=provenance)
            frozen = {"kind": kind, "seed": seed, "transform": {"smoke": True}}
            results.append(
                fa_probes.NullSelectionResult(
                    kind=kind,
                    seed=seed,
                    config=frozen,
                    config_sha256=sha256_json(frozen),
                    selection=selection,
                    max_norm_error=0.0,
                    test_source_identities=tuple(probe_test_source_identities),
                    test_transform={
                        "seed": seed,
                        "row_count": len(probe_test_source_identities),
                    },
                )
            )
        return tuple(results)

    monkeypatch.setattr(fa_cli, "_PROBE_NULL_SELECTOR", smoke_nulls)
    payload = fa_cli._fit_probes(
        config,
        tmp_path,
        SimpleNamespace(
            train_rows_manifest=train.manifest_path,
            validation_rows_manifest=validation.manifest_path,
            probe_test_manifest=prompt.manifest_path,
            shard_id="selection-smoke",
        ),
    )
    bundle = fa_cli._load_f2a_selection_bundle(
        FAArtifactStore(tmp_path), payload["selection_manifest"], config
    )

    assert payload["status"] == "selected"
    assert bundle.selection_bundle_hash == payload["selection_bundle_hash"]
    assert all(len(bundle.null_selections[task]) == 4 for task in fa_probes.TASKS)

    task_identities, _ = fa_cli._load_prompt_task_source_identities(
        FAArtifactStore(tmp_path),
        prompt.manifest_path,
        config,
        expected_namespace="probe_test",
    )
    test_rows = write_probe_rows_artifact(
        tmp_path,
        config,
        "probe_test",
        probe_rows_for_examples(test_examples),
        lineage={
            "prompt_manifest_sha256": prompt.sha256,
            "task_source_identities_sha256": fa_cli._task_source_identities_sha256(
                task_identities
            ),
        },
    )

    def materialize_after_unlock(supplied_config, supplied_root, args, *, authorization):
        assert supplied_config == config
        assert supplied_root == tmp_path
        assert isinstance(authorization, ProbeTestAuthorization)
        assert args.namespace == "probe_test"
        assert args.manifest == str(prompt.manifest_path)
        assert FAArtifactStore(tmp_path).endpoint_state(
            "probe_test", prompt.manifest_path
        ) == "unlocked_once"
        return {
            "status": "materialized",
            "probe_rows_manifest": str(test_rows.manifest_path),
        }

    monkeypatch.setattr(fa_cli, "_materialize_probe_rows", materialize_after_unlock)
    sealed = fa_cli._seal_probe_selection(
        config,
        tmp_path,
        SimpleNamespace(
            selection_manifest=payload["selection_manifest"],
            probe_test_manifest=prompt.manifest_path,
        ),
    )
    evaluated = fa_cli._evaluate_probe_test(
        config,
        tmp_path,
        SimpleNamespace(
            selection_manifest=payload["selection_manifest"],
            probe_test_manifest=prompt.manifest_path,
            metadata_manifest="unused-by-smoke-materializer",
            shard_id="probe-test-smoke",
        ),
    )
    recovered = fa_cli._evaluate_probe_test(
        config,
        tmp_path,
        SimpleNamespace(
            selection_manifest=payload["selection_manifest"],
            probe_test_manifest=prompt.manifest_path,
            metadata_manifest="unused-on-closed-recovery",
            shard_id="probe-test-smoke",
        ),
    )
    report = fa_cli._build_evidence_report(
        config,
        tmp_path,
        SimpleNamespace(
            behavior_test_manifest=behavior_prompt.manifest_path,
            probe_test_manifest=prompt.manifest_path,
            selection_manifest=payload["selection_manifest"],
            output="reports/f2a-smoke.md",
        ),
    )

    assert behavior_sealed["endpoint_state"] == "sealed"
    assert behavior_evaluated["endpoint_state"] == "closed"
    assert sealed["endpoint_state"] == "sealed"
    assert evaluated["status"] == "evaluated"
    assert evaluated["endpoint_state"] == "closed"
    assert recovered["status"] == "recovered"
    assert recovered["metrics_manifest"] == evaluated["metrics_manifest"]
    assert report["status"] == "reported"
    report_text = (tmp_path / "reports" / "f2a-smoke.md").read_text(
        encoding="utf-8"
    )
    assert "F1: evaluated" in report_text
    assert "F2A: evaluated" in report_text
    assert "F2B: skipped" in report_text


def test_behavior_seal_is_config_bound_and_idempotently_rejected(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompt = prompt_capability(tmp_path, config, "behavior_test")
    args = SimpleNamespace(behavior_test_manifest=prompt.manifest_path)

    sealed = fa_cli._seal_behavior_test(config, tmp_path, args)

    assert sealed["status"] == "sealed"
    assert sealed["endpoint"] == "behavior_test"
    assert len(sealed["selection_sha256"]) == 64
    assert FAArtifactStore(tmp_path).endpoint_state(
        "behavior_test", prompt.manifest_path
    ) == "sealed"
    with pytest.raises(ValueError, match="already sealed"):
        fa_cli._seal_behavior_test(config, tmp_path, args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "attacker/model", "model identity"),
        ("model_revision", "0" * 40, "model identity"),
        ("tokenizer_revision", "1" * 40, "model identity"),
        ("chat_template_utf8_hex", b"alternate template".hex(), "claimed hash"),
    ],
)
def test_prompt_verifier_rejects_forged_tokenizer_pin_identity_or_template_bytes(
    tmp_path, field, value, message
):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompt = prompt_capability(tmp_path, config, "pilot")
    store = FAArtifactStore(tmp_path)
    prompt_row = fa_cli._read_json_rows(prompt.data_path)[0]
    prompt_lineage = json.loads(prompt.manifest_path.read_text(encoding="utf-8"))[
        "lineage"
    ]
    pin_path = store.root / prompt_row["tokenizer_pin_manifest"]
    pin = store.verify_shard(pin_path)
    pin_row = fa_cli._read_json_rows(pin.data_path)[0]
    pin_lineage = json.loads(pin.manifest_path.read_text(encoding="utf-8"))["lineage"]
    pin_row[field] = value
    forged_pin = store.write_completed_shard(
        config.run_id,
        "pilot",
        f"forged-pin-{field}",
        [pin_row],
        pin_lineage,
        record_kind="tokenizer_pin",
    )
    prompt_row["tokenizer_pin_manifest"] = str(
        forged_pin.manifest_path.relative_to(store.root)
    )
    prompt_row["tokenizer_pin_sha256"] = forged_pin.sha256
    prompt_row["subset_manifest_sha256"] = fa_cli._prompt_subset_sha256(
        prompt_row["config_hash"],
        prompt_row["full_manifest_sha256"],
        prompt_row["namespace"],
        prompt_row["chat_template_sha256"],
        forged_pin.sha256,
        None,
        tuple(prompt_row["examples"]),
    )
    prompt_lineage["subset_manifest_sha256"] = prompt_row[
        "subset_manifest_sha256"
    ]
    prompt_lineage["tokenizer_pin_sha256"] = forged_pin.sha256
    forged_prompt = store.write_completed_shard(
        config.run_id,
        "pilot",
        f"forged-prompt-{field}",
        [prompt_row],
        prompt_lineage,
        record_kind="prompt_manifest",
    )

    with pytest.raises(ValueError, match=message):
        fa_cli._load_manifest(store, forged_prompt.manifest_path, config)


def test_fa_commands_require_explicit_input_manifests_and_restrict_generation_namespaces(tmp_path, capsys):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(["fa-run-generation", "--config", str(CONFIG_PATH)])
    argument_error = json.loads(capsys.readouterr().out)
    assert argument_error["status"] == "error"
    assert argument_error["error"]["type"] == "ArgumentError"
    args = parser.parse_args(
        [
            "fa-run-generation",
            "--config",
            str(CONFIG_PATH),
            "--manifest",
                str(tmp_path / "examples.json"),
            "--shard-id",
            "0001",
            "--namespace",
            "behavior_test",
        ]
    )
    (tmp_path / "examples.json").write_text(
        json.dumps(
            {
                "config_hash": FAConfig.from_json(CONFIG_PATH).config_hash,
                "manifest_sha256": "a" * 64,
                "examples": [],
            }
        ),
        encoding="utf-8",
    )
    exit_code = dispatch_fa(args)
    payload = json.loads(capsys.readouterr().out)
    assert exit_code != 0
    assert payload == {
        "command": "fa-run-generation",
        "error": {
            "message": (
                "fa-run-generation is generic-only and cannot evaluate protected "
                "test namespaces"
            ),
            "type": "ValueError",
        },
        "status": "error",
    }


def test_pilot_gate_json_contract_stops_confirmatory_construction(tmp_path, capsys):
    config_payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_payload["split_counts"] = {"pilot": 1, "circuit_dev": 1}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    template_hash = CHAT_TEMPLATE_SHA256
    active_config = FAConfig.from_json(config_path)
    _, prompts = prompt_capability(tmp_path, active_config, "pilot")

    class BlockingRunner:
        model_id = active_config.model_id
        model_revision = active_config.model_revision
        tokenizer_revision = active_config.tokenizer_revision
        chat_template_sha256 = template_hash

        def generate(self, prompts, generation):
            return ["UNKNOWN"] * len(prompts)

    manifest = fa_cli._load_manifest(FAArtifactStore(tmp_path), prompts.manifest_path, active_config)
    generation = run_generation_shard(
        BlockingRunner(), manifest, FAArtifactStore(tmp_path), "responses", config=active_config
    )

    exit_code = cli.main(
        [
            "fa-score-behavior",
            "--config",
            str(config_path),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--generation-manifest",
            str(generation.manifest_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["command"] == "fa-score-behavior"
    assert payload["pilot_gate"]["status"] == "blocked"
    assert "target_bound_accuracy_below_70_percent" in payload["pilot_gate"]["reasons"]
    assert Path(payload["pilot_gate_manifest"]).exists()
    with pytest.raises(ValueError, match="exact registered smoke config"):
        fa_cli._load_verified_pilot_gate(
            FAArtifactStore(tmp_path),
            payload["pilot_gate_manifest"],
            FAConfig.from_json(CONFIG_PATH),
        )


def test_pilot_gate_rejects_self_consistent_template_without_tokenizer_pin_chain(
    tmp_path,
):
    config = FAConfig.from_json(CONFIG_PATH)
    alternate_template = b"arbitrary alternate self-consistent template"
    alternate_hash = hashlib.sha256(alternate_template).hexdigest()
    _, prompts = prompt_capability(
        tmp_path, config, "pilot", template_bytes=alternate_template
    )
    store = FAArtifactStore(tmp_path)
    assert alternate_hash != CHAT_TEMPLATE_SHA256
    with pytest.raises(ValueError, match="registered smoke tokenizer revision"):
        fa_cli._load_manifest(store, prompts.manifest_path, config)


class FakeRunner:
    def __init__(self, config):
        self.model_id = config.model_id
        self.model_revision = config.model_revision
        self.tokenizer_revision = config.tokenizer_revision
        self.chat_template_sha256 = CHAT_TEMPLATE_SHA256

    def generate(self, prompts, generation):
        return ["K7M2Q" for _ in prompts]


def test_run_generation_uses_fake_runner_for_generic_namespace(tmp_path, capsys, monkeypatch):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, "pilot")
    monkeypatch.setattr(fa_cli, "HFModelRunner", FakeRunner)

    exit_code = cli.main(
        [
            "fa-run-generation",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--shard-id",
            "0001",
            "--namespace",
            "pilot",
            "--resume",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "generated"
    assert Path(payload["shard_manifest"]).exists()
    assert any(
        shard.record_kind == "generation_checkpoint"
        for shard in FAArtifactStore(tmp_path).resume_verified_shards(
            config.run_id, "pilot"
        )
    )


def test_behavior_scoring_rejects_mutable_raw_generation_rows(tmp_path, capsys):
    config = FAConfig.from_json(CONFIG_PATH)
    _, manifest = prompt_capability(tmp_path, config, "pilot")
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")

    exit_code = cli.main(
        [
            "fa-score-behavior",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(manifest.manifest_path),
            "--generation-manifest",
            str(raw),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code != 0
    assert payload["status"] == "error"
    assert "verified generation sidecar manifest" in payload["error"]["message"]


@pytest.mark.parametrize(
    "namespace", ["behavior_test", "probe_test", "intervention_test"]
)
def test_generic_generation_rejects_protected_namespaces_before_runner_construction(
    tmp_path, capsys, monkeypatch, namespace
):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, namespace)

    class MustNotConstruct:
        def __init__(self, config):
            raise AssertionError("protected generation must use a dedicated later command")

    monkeypatch.setattr(fa_cli, "HFModelRunner", MustNotConstruct)
    exit_code = cli.main(
        [
            "fa-run-generation",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
            "--manifest",
            str(prompts.manifest_path),
            "--shard-id",
            "0001",
            "--namespace",
            namespace,
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["error"]["message"] == (
        "fa-run-generation is generic-only and cannot evaluate protected test namespaces"
    )


@pytest.mark.parametrize(
    ("command", "required_option"),
    [
        ("fa-analyze-pilot-activations", "--manifest"),
        ("fa-fit-probes", "--train-rows-manifest"),
        ("fa-seal-behavior-test", "--behavior-test-manifest"),
        ("fa-seal-selection", "--selection-manifest"),
        ("fa-evaluate-probe-test", "--selection-manifest"),
        ("fa-build-report", "--output"),
    ],
)
def test_f2a_commands_require_explicit_artifacts_before_dispatch(
    capsys, command, required_option
):
    with pytest.raises(SystemExit) as raised:
        cli.main([command, "--config", str(CONFIG_PATH)])
    payload = json.loads(capsys.readouterr().out)

    assert raised.value.code == 2
    assert payload["command"] == command
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "ArgumentError"
    assert required_option in payload["error"]["message"]


def test_intervention_test_command_remains_not_implemented(capsys):
    exit_code = cli.main(
        ["fa-evaluate-intervention-test", "--config", str(CONFIG_PATH)]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "not_implemented"


def test_standalone_unlock_command_is_fail_closed_and_cannot_create_a_lease(
    tmp_path, capsys
):
    config = FAConfig.from_json(CONFIG_PATH)

    exit_code = cli.main(
        [
            "fa-unlock-endpoint",
            "--config",
            str(CONFIG_PATH),
            "--root",
            str(tmp_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["error"]["message"] == (
        "standalone endpoint unlock is disabled; a dedicated protected evaluation "
        "command must acquire and close its lease atomically"
    )
    endpoint_root = (
        tmp_path
        / "runs"
        / "familiarity_answerability"
        / config.run_id
        / "endpoints"
    )
    assert not endpoint_root.exists()


def exhaustive_power_audit(rows, *, power=0.8):
    cells = tuple(
        PowerCell(
            absent_attempt_rate=absent,
            entity_icc=entity,
            template_icc=template,
            invalid_format_rate=invalid,
            interaction=interaction,
            estimated_power=power,
            monte_carlo_standard_error=math.sqrt(
                power * (1 - power) / CONFIRMATORY_POWER_SIMULATIONS
            ),
            simulations=CONFIRMATORY_POWER_SIMULATIONS,
        )
        for absent, entity, template, invalid, interaction in product(
            REGISTERED_POWER_GRID.absent_attempt_rates,
            REGISTERED_POWER_GRID.entity_iccs,
            REGISTERED_POWER_GRID.template_iccs,
            REGISTERED_POWER_GRID.invalid_format_rates,
            REGISTERED_POWER_GRID.interactions,
        )
    )
    return PowerAudit(
        design_sha256=fa_cli._design_sha256(rows),
        seed=20260722,
        simulations=CONFIRMATORY_POWER_SIMULATIONS,
        cells=cells,
        registered_grid=True,
    )


def test_registered_power_preparation_uses_exact_fa_data_signature_and_typed_artifact(
    tmp_path, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    rows = valid_examples(config, "behavior_test")
    audit = exhaustive_power_audit(rows)
    calls = []

    def execute(design, effects, correlations, seed, *, simulations):
        calls.append((tuple(design), effects, correlations, seed, simulations))
        return audit

    monkeypatch.setattr(fa_cli, "_POWER_EXECUTOR", execute)
    loaded, shard = fa_cli._prepare_power_audit(
        FAArtifactStore(tmp_path),
        config,
        rows,
        None,
        run_registered=True,
    )

    assert loaded == audit
    assert shard.record_kind == "power_audit"
    assert calls == [
        (
            tuple(rows),
            REGISTERED_POWER_GRID.interactions,
            {
                "entity_icc": REGISTERED_POWER_GRID.entity_iccs,
                "template_icc": REGISTERED_POWER_GRID.template_iccs,
                "invalid_format_rate": REGISTERED_POWER_GRID.invalid_format_rates,
            },
            20260722,
            2000,
        )
    ]


def test_confirmatory_power_modes_are_explicit_and_mutually_exclusive(tmp_path, capsys):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)
    common = [
        "fa-build-confirmatory",
        "--config",
        str(CONFIG_PATH),
        "--matches-manifest",
        str(tmp_path / "matches.json"),
        "--pilot-gate-manifest",
        str(tmp_path / "gate.json"),
        "--naturalness-ratings-manifest",
        str(tmp_path / "ratings.json"),
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(common)
    capsys.readouterr()
    with pytest.raises(SystemExit):
        parser.parse_args(
            common
            + [
                "--power-audit-manifest",
                str(tmp_path / "power.json"),
                "--run-registered-power-audit",
            ]
        )


def test_prompt_loader_rejects_raw_or_self_attested_manifests(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    raw = tmp_path / "manifest.json"
    raw.write_text(json.dumps({"manifest_sha256": "a" * 64}), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable shard manifest"):
        fa_cli._load_manifest(FAArtifactStore(tmp_path), raw, config)

    fake = FAArtifactStore(tmp_path).write_completed_shard(
        config.run_id,
        "pilot",
        "self-attested",
        [
            {
                "kind": "prompt_manifest",
                "config_hash": config.config_hash,
                "full_manifest_sha256": "a" * 64,
                "subset_manifest_sha256": "b" * 64,
                "chat_template_sha256": "d" * 64,
                "namespace": "pilot",
                "model_sha256": "e" * 64,
                "tokenizer_sha256": "f" * 64,
                "generation": dict(config.generation),
                "examples": [{"example_id": "self-attested"}],
            }
        ],
        {
            "config_sha256": config.config_hash,
            "source_manifest_sha256": "a" * 64,
            "subset_manifest_sha256": "b" * 64,
            "chat_template_sha256": "d" * 64,
        },
        record_kind="prompt_manifest",
    )
    with pytest.raises(ValueError, match="invalid schema"):
        fa_cli._load_manifest(FAArtifactStore(tmp_path), fake.manifest_path, config)


def test_confirmatory_index_contains_only_ids_hashes_and_capability_paths(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    pilot_rows, pilot = prompt_capability(tmp_path, config, "pilot")
    protected_rows, protected = prompt_capability(tmp_path, config, "behavior_test")
    store = FAArtifactStore(tmp_path)
    power = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        "power-index-parent",
        [{"kind": "power_audit", "audit": {}}],
        {"config_sha256": config.config_hash},
        record_kind="power_audit",
    )
    pin = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        "pin-index-parent",
        [{"kind": "tokenizer_pin"}],
        {"config_sha256": config.config_hash},
        record_kind="tokenizer_pin",
    )

    index = fa_cli._confirmatory_index_record(
        store,
        config,
        "f" * 64,
        pilot_rows + protected_rows,
        {"pilot": pilot, "behavior_test": protected},
        power,
        pin,
    )
    encoded = json.dumps(index, sort_keys=True)

    assert "user_text" not in encoded
    assert "target_text" not in encoded
    assert "expected_output" not in encoded
    assert all(row.user_text not in encoded for row in pilot_rows + protected_rows)
    assert set(index["capabilities"]) == {"pilot", "behavior_test"}


def test_confirmatory_build_prepares_capabilities_without_sealing_endpoints(
    tmp_path, monkeypatch
):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    config = FAConfig.from_json(config_path)
    store = FAArtifactStore(tmp_path)
    matches = tmp_path / "matches.json"
    matches.write_text(json.dumps([MATCH]), encoding="utf-8")
    ratings = naturalness_ratings_manifest(tmp_path, config)
    rows = tuple(
        SimpleNamespace(
            example_id=sha256_json({"namespace": namespace}),
                canonical_payload_sha256=sha256_json({"namespace": namespace}),
                split=namespace,
                answerability="code_absent",
                entity_unit_id="Q1--syn-1",
                template_family={
                    "mechanism_train": "train_registry_direct",
                    "locked_validation": "validation_archive_direct",
                    "behavior_test": "behavior_catalog_direct",
                    "probe_test": "probe_index_direct",
                    "intervention_test": "intervention_register_direct",
                }[namespace],
                block="factorial",
            )
        for namespace in config.split_counts
    )
    power = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        "prepared-power",
        [{"kind": "power_audit", "audit": {}}],
        {"config_sha256": config.config_hash},
        record_kind="power_audit",
    )
    observed_smoke_configs = []

    def load_gate(artifact_store, path, expected_config):
        observed_smoke_configs.append(expected_config)
        return {"status": "passed", "evidence_sha256": "a" * 64}

    monkeypatch.setattr(fa_cli, "_load_verified_pilot_gate", load_gate)
    monkeypatch.setattr(
        fa_cli,
        "load_pinned_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(
            tokenizer=FakeTokenizer(),
            chat_template_bytes=b"registered confirmatory template",
            chat_template_sha256=config.chat_template_sha256,
        ),
    )
    monkeypatch.setattr(fa_cli, "build_factorial_examples", lambda *args, **kwargs: rows)
    monkeypatch.setattr(fa_cli, "build_same_string_examples", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        fa_cli,
        "_prepare_power_audit",
        lambda *args, **kwargs: (SimpleNamespace(), power),
    )
    monkeypatch.setattr(
        fa_cli,
        "build_manifest",
        lambda *args, **kwargs: SimpleNamespace(manifest_sha256="f" * 64),
    )
    monkeypatch.setattr(
        fa_cli,
        "_select_confirmatory_matches",
        lambda _config, selected, _audit: tuple(selected),
    )
    monkeypatch.setattr(
        fa_cli,
        "_load_verified_screened_match_collection",
        lambda *args, **kwargs: (EntityMatch(**MATCH),),
    )

    def must_not_seal(*args, **kwargs):
        raise AssertionError("confirmatory construction must not seal an endpoint")

    monkeypatch.setattr(FAArtifactStore, "seal_endpoint", must_not_seal)
    payload = fa_cli._build_manifest(
        config,
        tmp_path,
        SimpleNamespace(
            matches_manifest=matches,
            pilot_gate_manifest=tmp_path / "gate.json",
            power_audit_manifest=None,
            run_registered_power_audit=True,
            naturalness_ratings_manifest=ratings,
        ),
        confirmatory=True,
    )

    assert observed_smoke_configs == [FAConfig.from_json(CONFIG_PATH)]
    assert set(payload["namespace_manifests"]) == set(config.split_counts)
    assert "protected_endpoint_manifests" not in payload
    audit = store.verify_shard(payload["naturalness_audit_manifest"])
    assert audit.record_kind == "naturalness_audit"
    assert audit.sha256 == payload["naturalness_audit_sha256"]
    for manifest_path in payload["namespace_manifests"].values():
        prompt = store.verify_shard(manifest_path)
        row = fa_cli._read_json_rows(prompt.data_path)[0]
        assert row["naturalness_audit_sha256"] == audit.sha256
    assert not (
        tmp_path
        / "runs"
        / "familiarity_answerability"
        / config.run_id
        / "endpoints"
    ).exists()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_generation_sidecar_requires_exact_example_multiset(tmp_path, mutation):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, "pilot")
    store = FAArtifactStore(tmp_path)
    manifest = fa_cli._load_manifest(store, prompts.manifest_path, config)
    generated = run_generation_shard(
        FakeRunner(config), manifest, store, "valid", config=config, namespace="pilot"
    )
    rows = list(fa_cli._read_json_rows(generated.data_path))
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        extra = dict(rows[0])
        extra["example_id"] = "0" * 64
        rows.append(extra)
    lineage = json.loads(generated.manifest_path.read_text(encoding="utf-8"))["lineage"]
    forged = store.write_completed_shard(
        config.run_id,
        "pilot",
        f"forged-{mutation}",
        rows,
        lineage,
        record_kind="generation",
    )

    with pytest.raises(ValueError, match="every expected example exactly once"):
        fa_cli._load_verified_generation_sidecar(
            store, forged.manifest_path, manifest, config
        )


def test_power_audit_rejects_duplicate_registered_cells(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    rows = valid_examples(config, "behavior_test")
    audit = exhaustive_power_audit(rows)
    forged = PowerAudit(
        design_sha256=audit.design_sha256,
        seed=audit.seed,
        simulations=audit.simulations,
        cells=audit.cells[:-1] + (audit.cells[0],),
        registered_grid=True,
    )
    shard = FAArtifactStore(tmp_path).write_completed_shard(
        config.run_id,
        "mechanism_train",
        "forged-power",
        [{"kind": "power_audit", "audit": asdict(forged)}],
        {"config_sha256": config.config_hash, "design_sha256": audit.design_sha256},
        record_kind="power_audit",
    )

    with pytest.raises(ValueError, match="180 unique"):
        fa_cli._prepare_power_audit(
            FAArtifactStore(tmp_path),
            config,
            rows,
            shard.manifest_path,
            run_registered=False,
        )


def test_pilot_gate_recomputes_and_rejects_forged_stored_pass(tmp_path):
    config = FAConfig.from_json(CONFIG_PATH)
    _, prompts = prompt_capability(tmp_path, config, "pilot")
    store = FAArtifactStore(tmp_path)
    manifest = fa_cli._load_manifest(store, prompts.manifest_path, config)

    class BlockingRunner(FakeRunner):
        def generate(self, prompts, generation):
            return ["INVALID"] * len(prompts)

    generation = run_generation_shard(
        BlockingRunner(config), manifest, store, "blocked", config=config
    )
    forged_gate = {"status": "passed", "reasons": []}
    metrics = {"forged": True}
    evidence_hash = sha256_json({"metrics": metrics, "pilot_gate": forged_gate})
    gate = store.write_completed_shard(
        config.run_id,
        "pilot",
        "forged-gate",
        [
            {
                "kind": "pilot_gate",
                "config_sha256": config.config_hash,
                "source_manifest_sha256": manifest.manifest_sha256,
                "prompt_manifest": str(prompts.manifest_path.relative_to(store.root)),
                "prompt_manifest_sha256": prompts.sha256,
                "tokenizer_pin_manifest": str(
                    manifest.tokenizer_pin_manifest_path.relative_to(store.root)
                ),
                "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
                "chat_template_sha256": manifest.chat_template_sha256,
                "generation_sidecar_manifest": str(
                    generation.manifest_path.relative_to(store.root)
                ),
                "generation_sidecar_sha256": generation.sha256,
                "pilot_gate": forged_gate,
                "metrics": metrics,
                "evidence_sha256": evidence_hash,
            }
        ],
        {
            "config_sha256": config.config_hash,
            "source_manifest_sha256": manifest.manifest_sha256,
            "generation_sidecar_sha256": generation.sha256,
            "prompt_manifest_sha256": prompts.sha256,
            "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
            "chat_template_sha256": manifest.chat_template_sha256,
        },
        record_kind="pilot_gate",
    )
    gate_row = fa_cli._read_json_rows(gate.data_path)[0]
    gate_lineage = json.loads(gate.manifest_path.read_text(encoding="utf-8"))[
        "lineage"
    ]
    unrelated_run_gate = store.write_completed_shard(
        "unrelated-run",
        "pilot",
        "forged-gate",
        [gate_row],
        gate_lineage,
        record_kind="pilot_gate",
    )

    with pytest.raises(ValueError, match="registered smoke run"):
        fa_cli._load_verified_pilot_gate(
            store, unrelated_run_gate.manifest_path, config
        )

    with pytest.raises(ValueError, match="deterministic recomputation"):
        fa_cli._load_verified_pilot_gate(store, gate.manifest_path, config)


def test_build_confirmatory_rejects_raw_passed_gate_before_tokenizer_loading(
    tmp_path, capsys, monkeypatch
):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    matches = tmp_path / "matches.json"
    matches.write_text(json.dumps([MATCH]), encoding="utf-8")
    gate = tmp_path / "raw-gate.json"
    gate.write_text(json.dumps({"status": "passed"}), encoding="utf-8")

    def must_not_load(*args, **kwargs):
        raise AssertionError("tokenizer must not load before gate verification")

    monkeypatch.setattr(fa_cli, "load_pinned_tokenizer", must_not_load)
    exit_code = cli.main(
        [
            "fa-build-confirmatory",
            "--config",
            str(config_path),
            "--root",
            str(tmp_path),
            "--matches-manifest",
            str(matches),
            "--pilot-gate-manifest",
            str(gate),
            "--naturalness-ratings-manifest",
            str(tmp_path / "ratings.json"),
            "--run-registered-power-audit",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code != 0
    assert payload["status"] == "error"
    assert "verified pilot gate sidecar manifest" in payload["error"]["message"]


def test_confirmatory_build_requires_human_naturalness_ratings_at_parse_time(
    tmp_path, capsys
):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "fa-build-confirmatory",
                "--config",
                str(
                    Path(__file__).resolve().parents[1]
                    / "configs"
                    / "familiarity_answerability_gemma2_2b.json"
                ),
                "--matches-manifest",
                str(tmp_path / "matches.json"),
                "--pilot-gate-manifest",
                str(tmp_path / "gate.json"),
                "--run-registered-power-audit",
            ]
        )

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["type"] == "ArgumentError"
    assert "naturalness-ratings-manifest" in payload["error"]["message"]


def test_naturalness_ratings_require_verified_blinded_input_and_current_schema(
    tmp_path,
):
    config = FAConfig.from_json(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    raw = tmp_path / "ratings.json"
    raw.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable shard manifest"):
        fa_cli._load_verified_naturalness_ratings(
            FAArtifactStore(tmp_path), raw, config
        )

    unsupported = naturalness_ratings_manifest(
        tmp_path, config, rating_schema_version=2
    )
    with pytest.raises(ValueError, match="schema_version"):
        fa_cli._load_verified_naturalness_ratings(
            FAArtifactStore(tmp_path), unsupported, config
        )


def test_prompt_capability_writer_enforces_profile_specific_human_audit(tmp_path):
    confirmatory = FAConfig.from_json(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "familiarity_answerability_gemma2_2b.json"
    )
    store = FAArtifactStore(tmp_path)
    pin = store.write_completed_shard(
        confirmatory.run_id,
        "mechanism_train",
        "pin-for-profile-check",
        [{"kind": "tokenizer_pin"}],
        {"config_sha256": confirmatory.config_hash},
        record_kind="tokenizer_pin",
    )
    example = SimpleNamespace(
        example_id="a" * 64,
        canonical_payload_sha256="b" * 64,
        split="mechanism_train",
    )
    with pytest.raises(ValueError, match="confirmatory prompt capability requires"):
        fa_cli._write_prompt_capability(
            store,
            confirmatory,
            "c" * 64,
            "mechanism_train",
            (example,),
            confirmatory.chat_template_sha256,
            pin,
        )


def test_report_rejects_evidence_sidecar_from_another_run_before_loading(
    tmp_path, monkeypatch
):
    config = FAConfig.from_json(CONFIG_PATH)
    foreign_manifest = tmp_path / "foreign.manifest.json"
    foreign_manifest.write_text(
        json.dumps({"run_id": "another-run"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        fa_cli,
        "load_closed_f1_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("foreign evidence must be rejected before loading")
        ),
    )

    with pytest.raises(ValueError, match="does not belong to the registered smoke run"):
        fa_cli._build_evidence_report(
            config,
            tmp_path,
            SimpleNamespace(
                behavior_test_manifest=str(foreign_manifest),
                probe_test_manifest=None,
                selection_manifest=None,
                output="report.md",
            ),
        )
