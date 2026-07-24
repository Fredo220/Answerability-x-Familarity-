import hashlib
import json
from types import SimpleNamespace

import pytest

from trajectory_extractor.fa_confirmatory_synthetics import (
    generate_synthetic_candidates,
    generate_synthetic_manifests,
)
from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_entities import CandidateEntity, match_synthetic_entities


class WordTokenizer:
    name_or_path = "registered-test-tokenizer"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def _candidate(entity_id, name, domain):
    return CandidateEntity(
        entity_id=entity_id,
        qid=f"Q{int(entity_id.rsplit('-', 1)[1]) + 1}",
        name=name,
        coarse_type=domain,
        split="mechanism_train",
        source_query="https://www.wikidata.org/",
        source_provenance="Wikidata CC0, retrieved 2026-07-24",
        screening_aliases=(("alpha",), ("beta",), ("gamma",)),
    )


def test_synthetic_builder_is_deterministic_compatible_and_collision_free():
    candidates = (
        _candidate("entity-1", "Ada Norton", "person"),
        _candidate("entity-2", "Cedar Valley", "place"),
        _candidate("entity-3", "Global Council", "organization"),
        _candidate("entity-4", "Silent River", "creative_work"),
    )
    tokenizer = WordTokenizer()

    first = generate_synthetic_candidates(
        candidates,
        tokenizer,
        variants_per_entity=2,
    )
    second = generate_synthetic_candidates(
        candidates,
        tokenizer,
        variants_per_entity=2,
    )

    assert first == second
    assert len(first) == 8
    assert len({row.name.casefold() for row in first}) == len(first)
    assert not {
        row.name.casefold() for row in first
    } & {
        candidate.name.casefold() for candidate in candidates
    }
    assert len(match_synthetic_entities(candidates, first, tokenizer)) == 4


def test_synthetic_surface_generation_is_invariant_to_split_assignment():
    first = _candidate("entity-1", "Ada Norton", "person")
    second = CandidateEntity(
        entity_id="confirmatory-probe_test-person-q2",
        qid=first.qid,
        name=first.name,
        coarse_type=first.coarse_type,
        split="probe_test",
        source_query=first.source_query,
        source_provenance=first.source_provenance,
        screening_aliases=first.screening_aliases,
    )

    first_names = [
        row.name
        for row in generate_synthetic_candidates(
            (first,),
            WordTokenizer(),
            variants_per_entity=3,
        )
    ]
    second_names = [
        row.name
        for row in generate_synthetic_candidates(
            (second,),
            WordTokenizer(),
            variants_per_entity=3,
        )
    ]

    assert first_names == second_names


def test_synthetic_builder_rejects_unregistered_domain():
    candidate = _candidate("entity-1", "Example Name", "unknown")
    with pytest.raises(KeyError):
        generate_synthetic_candidates((candidate,), WordTokenizer())


def test_synthetic_builder_can_omit_an_unmatchable_source_candidate():
    class KnownNameTokenizer:
        name_or_path = "registered-known-name-tokenizer"

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            if "Oppenheimer" in text:
                return ["known"]
            return list(text)

    candidate = _candidate(
        "entity-1",
        "Oppenheimer",
        "creative_work",
    )

    assert generate_synthetic_candidates(
        (candidate,),
        KnownNameTokenizer(),
        variants_per_entity=3,
        allow_incomplete=True,
    ) == ()


def test_synthetic_builder_preserves_mixed_case_pattern():
    candidate = _candidate("entity-1", "YouTube", "organization")
    synthetic = generate_synthetic_candidates(
        (candidate,),
        WordTokenizer(),
        variants_per_entity=1,
    )

    assert len(synthetic) == 1
    assert [
        character.isupper() for character in synthetic[0].name
    ] == [
        character.isupper() for character in candidate.name
    ]


def test_synthetic_builder_preserves_intraword_punctuation_for_token_matching():
    class PunctuationTokenizer:
        name_or_path = "registered-punctuation-tokenizer"

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return text.replace("-", " - ").split()

    candidate = _candidate("entity-1", "WALL-E", "creative_work")
    synthetic = generate_synthetic_candidates(
        (candidate,),
        PunctuationTokenizer(),
        variants_per_entity=1,
    )

    assert len(synthetic) == 1
    assert synthetic[0].name[4] == "-"


def test_synthetic_snapshot_hashes_every_materialized_file(
    tmp_path,
    monkeypatch,
):
    config = FAConfig.from_json(
        "configs/familiarity_answerability_gemma2_2b.json"
    )
    candidate = _candidate("entity-1", "Ada Norton", "person")
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    **candidate.__dict__,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "trajectory_extractor.fa_confirmatory_synthetics.load_pinned_tokenizer",
        lambda _config: SimpleNamespace(
            tokenizer=WordTokenizer(),
            chat_template_sha256=config.chat_template_sha256,
        ),
    )

    manifest = generate_synthetic_manifests(
        (candidate_path,),
        output_dir=tmp_path,
        config=config,
        variants_per_entity=3,
    )

    synthetic_path = tmp_path / "synthetic_candidates_mechanism_train_v1.json"
    assert manifest["files"]["mechanism_train"]["sha256"] == hashlib.sha256(
        synthetic_path.read_bytes()
    ).hexdigest()
    snapshot_path = tmp_path / "synthetic_source_snapshot_v1.json"
    assert manifest["source_snapshot_sha256"] == hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()


def test_complete_manifest_build_fails_closed_before_writing_partial_files(
    tmp_path,
    monkeypatch,
):
    class KnownNameTokenizer:
        name_or_path = "registered-known-name-tokenizer"

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            if "Oppenheimer" in text:
                return ["known"]
            return list(text)

    config = FAConfig.from_json(
        "configs/familiarity_answerability_gemma2_2b.json"
    )
    candidate = _candidate("entity-1", "Oppenheimer", "creative_work")
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(
        json.dumps(
            [{"schema_version": 1, **candidate.__dict__}]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "trajectory_extractor.fa_confirmatory_synthetics.load_pinned_tokenizer",
        lambda _config: SimpleNamespace(
            tokenizer=KnownNameTokenizer(),
            chat_template_sha256=config.chat_template_sha256,
        ),
    )

    with pytest.raises(
        ValueError,
        match="lost complete pseudonym reserves",
    ):
        generate_synthetic_manifests(
            (candidate_path,),
            output_dir=tmp_path,
            config=config,
            require_complete=True,
        )

    assert not list(tmp_path.glob("synthetic_candidates_*"))
    assert not (tmp_path / "synthetic_source_snapshot_v1.json").exists()
