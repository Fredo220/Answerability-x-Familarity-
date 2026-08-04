from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from trajectory_extractor.fa_answerability_causal import (
    CAUSAL_SPLIT_COUNTS,
    CAUSAL_VALIDATION_LAYERS,
    CAUSAL_VALIDATION_MULTIPLIERS,
    ValidationCandidate,
    audit_causal_corpus,
    build_causal_corpus,
    fit_train_only_directions,
    select_validation_candidate,
    verify_causal_corpus,
    write_causal_corpus,
)
from trajectory_extractor.fa_same_string_replication_v3 import build_replication_corpus


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


def _v3_train_activations(prompts):
    rows = [row for row in prompts if row.split == "representation_train"]
    values = {}
    for row in rows:
        answerability = 1.0 if row.answerability == "target_bound" else -1.0
        exposure = 0.2 if row.exposure == "high_exposure" else -0.2
        activation = np.zeros((len(CAUSAL_VALIDATION_LAYERS), 4), dtype=np.float64)
        activation[:, 0] = answerability
        activation[:, 1] = exposure
        values[row.example_id] = activation
    return values


def _candidate(*, layer, multiplier, effect, unit_ids):
    return ValidationCandidate(
        layer_id=layer,
        multiplier=multiplier,
        unit_effects=tuple((unit_id, effect) for unit_id in unit_ids),
        invalid_output_rate=0.0,
        bound_accuracy_drop=0.0,
        prompts_sha256="a" * 64,
        prefixes_sha256="b" * 64,
        model_sha256="c" * 64,
        tokenizer_sha256="d" * 64,
        direction_sha256="e" * 64,
    )


def test_causal_corpus_is_fresh_complete_and_deterministic(tmp_path):
    tokenizer = _CharacterTokenizer()
    v3 = build_replication_corpus(tokenizer)
    first = build_causal_corpus(tokenizer, v3_prompts=v3.prompts)
    second = build_causal_corpus(tokenizer, v3_prompts=v3.prompts)

    assert first.manifest_sha256 == second.manifest_sha256
    assert len(first.prompts) == 192
    assert first.audit.passed
    for split, count in CAUSAL_SPLIT_COUNTS.items():
        assert len({row.entity_unit_id for row in first.prompts if row.split == split}) == count

    paths = write_causal_corpus(first, tmp_path)
    assert verify_causal_corpus(paths.manifest, tokenizer, v3_prompts=v3.prompts).manifest_sha256 == first.manifest_sha256


def test_causal_corpus_rejects_tampered_label_incomplete_unit_and_token_mismatch(tmp_path):
    tokenizer = _CharacterTokenizer()
    corpus = build_causal_corpus(tokenizer)
    paths = write_causal_corpus(corpus, tmp_path)
    records = paths.prompts.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(records[0])
    tampered["answerability"] = "target_bound" if tampered["answerability"] == "target_unbound" else "target_unbound"
    records[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    paths.prompts.write_text("\n".join(records) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash|canonical|reconstruct"):
        verify_causal_corpus(paths.manifest, tokenizer)

    incomplete = audit_causal_corpus(corpus.prompts[:-1], tokenizer)
    assert not incomplete.passed
    assert "complete_2x2_units" in incomplete.violations

    mismatch = replace(corpus.prompts[0], rendered_token_ids=(99,))
    token_audit = audit_causal_corpus((mismatch, *corpus.prompts[1:]), tokenizer)
    assert not token_audit.passed
    assert "tokenizer_replay" in token_audit.violations


def test_causal_corpus_rejects_reused_v3_test_identity():
    tokenizer = _CharacterTokenizer()
    corpus = build_causal_corpus(tokenizer)
    v3_test_identity = corpus.prompts[0].entity_unit_id

    audit = audit_causal_corpus(
        corpus.prompts,
        tokenizer,
        v3_test_identity_sets={"entity_unit_ids": frozenset({v3_test_identity})},
    )

    assert not audit.passed
    assert "v3_test_identity_isolation" in audit.violations


def test_direction_fitting_uses_only_complete_v3_representation_train_rows():
    tokenizer = _CharacterTokenizer()
    v3 = build_replication_corpus(tokenizer)
    activations = _v3_train_activations(v3.prompts)

    directions = fit_train_only_directions(
        v3.prompts,
        activations,
        prompts_sha256="a" * 64,
        activations_sha256="b" * 64,
    )

    assert tuple(direction.layer_id for direction in directions.directions) == CAUSAL_VALIDATION_LAYERS
    assert all(np.isclose(np.linalg.norm(direction.vector), 1.0) for direction in directions.directions)
    assert all(direction.natural_scale > 0.0 for direction in directions.directions)

    test_rows = tuple(row for row in v3.prompts if row.split == "entity_test")
    with pytest.raises(ValueError, match="representation_train"):
        fit_train_only_directions(
            test_rows,
            {row.example_id: np.zeros((5, 4)) for row in test_rows},
            prompts_sha256="a" * 64,
            activations_sha256="b" * 64,
        )

    with pytest.raises(ValueError, match="complete 2x2"):
        fit_train_only_directions(
            tuple(row for row in v3.prompts if row.split == "representation_train")[:-1],
            activations,
            prompts_sha256="a" * 64,
            activations_sha256="b" * 64,
        )


def test_validation_selection_is_fixed_to_causal_validation_and_deterministic():
    corpus = build_causal_corpus(_CharacterTokenizer())
    units = tuple(
        sorted({row.entity_unit_id for row in corpus.prompts if row.split == "causal_validation"})
    )
    candidates = (
        _candidate(layer=6, multiplier=1.0, effect=0.4, unit_ids=units),
        _candidate(layer=12, multiplier=0.5, effect=0.4, unit_ids=units),
        _candidate(layer=0, multiplier=0.5, effect=0.4, unit_ids=units),
    )

    first = select_validation_candidate(candidates, corpus)
    second = select_validation_candidate(tuple(reversed(candidates)), corpus)

    assert first == second
    assert first.layer_id == 0
    assert first.multiplier == 0.5
    assert first.direction_sha256 == "e" * 64
    assert set(CAUSAL_VALIDATION_MULTIPLIERS) == {0.25, 0.5, 1.0}

    with pytest.raises(ValueError, match="validation units"):
        select_validation_candidate(
            (_candidate(layer=0, multiplier=0.25, effect=0.5, unit_ids=units[:-1]),),
            corpus,
        )
