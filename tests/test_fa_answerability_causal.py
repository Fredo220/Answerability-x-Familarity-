from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from trajectory_extractor.fa_answerability_causal import (
    CAUSAL_SPLIT_COUNTS,
    CAUSAL_VALIDATION_LAYERS,
    CAUSAL_VALIDATION_MULTIPLIERS,
    CausalExpectedProvenance,
    ValidationCandidate,
    audit_causal_corpus,
    build_causal_corpus,
    fit_train_only_directions,
    load_v3_training_direction_inputs,
    select_causal_intervention,
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


def _selection_provenance(corpus):
    return CausalExpectedProvenance(
        corpus_sha256=corpus.manifest_sha256,
        direction_bundle_sha256="a" * 64,
        direction_hashes={
            layer: f"{index + 1:x}" * 64
            for index, layer in enumerate(CAUSAL_VALIDATION_LAYERS)
        },
        model_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
    )


def _candidate(*, layer, multiplier, effect, unit_ids, provenance):
    return ValidationCandidate(
        layer_id=layer,
        multiplier=multiplier,
        unit_effects=tuple((unit_id, effect) for unit_id in unit_ids),
        invalid_output_rate=0.0,
        bound_accuracy_drop=0.0,
        corpus_sha256=provenance.corpus_sha256,
        direction_bundle_sha256=provenance.direction_bundle_sha256,
        model_sha256=provenance.model_sha256,
        tokenizer_sha256=provenance.tokenizer_sha256,
        direction_sha256=provenance.direction_hashes[layer],
    )


def _full_grid(*, unit_ids, provenance):
    return tuple(
        _candidate(
            layer=layer,
            multiplier=multiplier,
            effect=0.4,
            unit_ids=unit_ids,
            provenance=provenance,
        )
        for layer in CAUSAL_VALIDATION_LAYERS
        for multiplier in CAUSAL_VALIDATION_MULTIPLIERS
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
    v3 = build_replication_corpus(tokenizer)
    corpus = build_causal_corpus(tokenizer, v3_prompts=v3.prompts)
    paths = write_causal_corpus(corpus, tmp_path)
    records = paths.prompts.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(records[0])
    tampered["answerability"] = "target_bound" if tampered["answerability"] == "target_unbound" else "target_unbound"
    records[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    paths.prompts.write_text("\n".join(records) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash|canonical|reconstruct"):
        verify_causal_corpus(paths.manifest, tokenizer, v3_prompts=v3.prompts)

    incomplete = audit_causal_corpus(corpus.prompts[:-1], tokenizer, v3_prompts=v3.prompts)
    assert not incomplete.passed
    assert "complete_2x2_units" in incomplete.violations

    mismatch = replace(corpus.prompts[0], rendered_token_ids=(99,))
    token_audit = audit_causal_corpus(
        (mismatch, *corpus.prompts[1:]), tokenizer, v3_prompts=v3.prompts
    )
    assert not token_audit.passed
    assert "tokenizer_replay" in token_audit.violations


def test_causal_corpus_requires_v3_exclusions_and_isolates_all_causal_identities():
    tokenizer = _CharacterTokenizer()
    v3 = build_replication_corpus(tokenizer)
    with pytest.raises(ValueError, match="v3 exclusions"):
        build_causal_corpus(tokenizer)

    corpus = build_causal_corpus(tokenizer, v3_prompts=v3.prompts)
    with pytest.raises(ValueError, match="v3 exclusions"):
        audit_causal_corpus(corpus.prompts, tokenizer)
    with pytest.raises(ValueError, match="complete v3 exclusions"):
        audit_causal_corpus(
            corpus.prompts,
            tokenizer,
            v3_prompts=tuple(
                row
                for row in v3.prompts
                if row.example_id
                != next(item.example_id for item in v3.prompts if item.split == "entity_test")
            ),
        )

    v3_test_prompt = next(row for row in v3.prompts if row.split == "entity_test")
    reused_v3 = corpus.prompts[0]
    object.__setattr__(reused_v3, "target_text", v3_test_prompt.target_text)
    audit = audit_causal_corpus(corpus.prompts, tokenizer, v3_prompts=v3.prompts)
    assert not audit.passed
    assert "v3_test_identity_isolation" in audit.violations

    corpus = build_causal_corpus(tokenizer, v3_prompts=v3.prompts)
    validation_name = next(
        row.target_text for row in corpus.prompts if row.split == "causal_validation"
    )
    entity_prompt = next(
        row for row in corpus.prompts if row.split == "causal_entity_test"
    )
    object.__setattr__(entity_prompt, "target_text", validation_name)
    audit = audit_causal_corpus(corpus.prompts, tokenizer, v3_prompts=v3.prompts)
    assert not audit.passed
    assert "causal_identity_disjointness" in audit.violations


def test_direction_fitting_uses_verified_v3_manifests_and_preserves_provenance():
    root = "release/familiarity_answerability/representation_replication_v3"
    source = load_v3_training_direction_inputs(
        v3_manifest_path=f"{root}/same_string_replication_v3_manifest.json",
        activation_manifest_path=f"{root}/activations/activations-representation_train.manifest.json",
        expected_model_id="google/gemma-2-2b-it",
        expected_model_revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        expected_tokenizer_id="google/gemma-2-2b-it",
        expected_tokenizer_revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        expected_chat_template_sha256="ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6",
    )
    directions = fit_train_only_directions(source)

    assert tuple(direction.layer_id for direction in directions.directions) == CAUSAL_VALIDATION_LAYERS
    assert all(np.isclose(np.linalg.norm(direction.vector), 1.0) for direction in directions.directions)
    assert all(direction.natural_scale > 0.0 for direction in directions.directions)
    assert directions.source.activation_index_sha256 == source.source.activation_index_sha256
    assert directions.source.v3_prompts_sha256 == source.source.v3_prompts_sha256

    with pytest.raises(ValueError, match="model ID"):
        load_v3_training_direction_inputs(
            v3_manifest_path=f"{root}/same_string_replication_v3_manifest.json",
            activation_manifest_path=f"{root}/activations/activations-representation_train.manifest.json",
            expected_model_id="wrong/model",
            expected_model_revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
            expected_tokenizer_id="google/gemma-2-2b-it",
            expected_tokenizer_revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
            expected_chat_template_sha256="ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6",
        )


def test_validation_selection_is_fixed_to_causal_validation_and_deterministic():
    tokenizer = _CharacterTokenizer()
    corpus = build_causal_corpus(tokenizer, v3_prompts=build_replication_corpus(tokenizer).prompts)
    units = tuple(
        sorted({row.entity_unit_id for row in corpus.prompts if row.split == "causal_validation"})
    )
    provenance = _selection_provenance(corpus)
    candidates = _full_grid(unit_ids=units, provenance=provenance)

    first = select_causal_intervention(candidates, corpus, provenance)
    second = select_causal_intervention(tuple(reversed(candidates)), corpus, provenance)

    assert first == second
    assert first.layer_id == 0
    assert first.multiplier == 0.25
    assert first.direction_sha256 == provenance.direction_hashes[0]
    assert set(CAUSAL_VALIDATION_MULTIPLIERS) == {0.25, 0.5, 1.0}

    with pytest.raises(ValueError, match="complete fixed grid"):
        select_causal_intervention(candidates[:-1], corpus, provenance)

    with pytest.raises(ValueError, match="direction hash"):
        select_causal_intervention(
            (
                replace(candidates[0], direction_sha256="f" * 64),
                *candidates[1:],
            ),
            corpus,
            provenance,
        )

    with pytest.raises(ValueError, match="validation units"):
        select_causal_intervention(
            (
                _candidate(
                    layer=0,
                    multiplier=0.25,
                    effect=0.5,
                    unit_ids=units[:-1],
                    provenance=provenance,
                ),
                *candidates[1:],
            ),
            corpus,
            provenance,
        )
