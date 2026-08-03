from __future__ import annotations

import json

import pytest

from trajectory_extractor.fa_same_string_replication_v3 import (
    REP_V3_SPLIT_COUNTS,
    build_replication_corpus,
    verify_replication_corpus,
    write_replication_corpus,
)


class _CharacterTokenizer:
    bos_token_id = 1
    name_or_path = "test/character-tokenizer"
    chat_template = "test-template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "<BOS>user\n" + messages[0]["content"] + "\nassistant\n"

    def __call__(self, text, *, add_special_tokens, **_kwargs):
        assert add_special_tokens is False
        return {"input_ids": [self.bos_token_id] + [ord(char) + 10 for char in text[5:]]}


def test_v3_corpus_has_registered_splits_templates_and_complete_units():
    tokenizer = _CharacterTokenizer()
    corpus = build_replication_corpus(tokenizer)

    assert len(corpus.prompts) == 320
    for split, expected_units in REP_V3_SPLIT_COUNTS.items():
        rows = [row for row in corpus.prompts if row.split == split]
        assert len(rows) == 4 * expected_units
        assert len({row.entity_unit_id for row in rows}) == expected_units
    assert corpus.audit.passed
    assert corpus.audit.checks["complete_2x2_units"]
    assert corpus.audit.checks["template_holdout"]
    assert corpus.audit.checks["split_identity_disjointness"]


def test_v3_pairs_have_exact_rendered_token_multisets():
    corpus = build_replication_corpus(_CharacterTokenizer())

    assert corpus.audit.checks["answerability_token_multisets"]
    assert corpus.audit.checks["exposure_token_multisets"]
    assert corpus.audit.checks["single_bos"]


def test_v3_corpus_is_deterministic_and_round_trips(tmp_path):
    tokenizer = _CharacterTokenizer()
    first = build_replication_corpus(tokenizer)
    second = build_replication_corpus(tokenizer)
    assert first.manifest_sha256 == second.manifest_sha256

    paths = write_replication_corpus(first, tmp_path)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["manifest_sha256"] == first.manifest_sha256
    assert manifest["row_count"] == 320
    assert verify_replication_corpus(paths.manifest, tokenizer).manifest_sha256 == first.manifest_sha256


def test_v3_verifier_rejects_prompt_tampering(tmp_path):
    tokenizer = _CharacterTokenizer()
    paths = write_replication_corpus(build_replication_corpus(tokenizer), tmp_path)
    rows = paths.prompts.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0])
    payload["user_text"] += " tampered"
    rows[0] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    paths.prompts.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash|reconstruct|canonical"):
        verify_replication_corpus(paths.manifest, tokenizer)
