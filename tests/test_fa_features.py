from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trajectory_extractor.fa_activations import ActivationRecord, resolve_registered_anchors
from trajectory_extractor.fa_data import FAExample, _structured_user_text
from trajectory_extractor.fa_features import (
    OUTPUT_FEATURE_DIM,
    OUTPUT_FEATURE_NAMES,
    SURFACE_FEATURE_NAMES,
    FeatureEvidence,
    HFTeacherForcedScorer,
    OutputEvidence,
    UnsupportedAnswerOutcome,
    VerifiedDomainRelation,
    build_probe_row,
    build_probe_rows,
    output_feature_vector,
    surface_feature_vector,
)
from trajectory_extractor.fa_probes import OUTPUT_CONTROL_SCHEMA_SHA256


MODEL_REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40
CONFIG_SHA256 = "c" * 64
FAKE_VOCAB_SIZE = 1200
FAKE_TOKEN_LOGIT_STEP = 0.001
EXPECTED_SURFACE_FEATURE_NAMES = (
    "target_character_count",
    "distractor_character_count",
    "prompt_character_count",
    "target_word_count",
    "distractor_word_count",
    "prompt_word_count",
    "target_uppercase_character_count",
    "distractor_uppercase_character_count",
    "prompt_uppercase_character_count",
    "rendered_prompt_token_count",
    "target_is_first",
    "target_is_second",
    "code_is_first",
    "code_is_second",
    "code_is_absent",
    "entity_domain_person",
    "entity_domain_place",
    "entity_domain_organization",
    "entity_domain_creative_work",
    "prompt_template_train_registry_direct",
    "prompt_template_train_registry_possessive",
    "prompt_template_train_registry_query",
    "prompt_template_validation_archive_direct",
    "prompt_template_validation_archive_possessive",
    "prompt_template_validation_archive_query",
    "prompt_template_behavior_catalog_direct",
    "prompt_template_behavior_catalog_inverse",
    "prompt_template_behavior_ledger_direct",
    "prompt_template_behavior_ledger_query",
    "prompt_template_probe_index_direct",
    "prompt_template_probe_index_inverse",
    "prompt_template_probe_file_direct",
    "prompt_template_probe_file_query",
    "prompt_template_intervention_register_direct",
    "prompt_template_intervention_register_inverse",
    "prompt_template_intervention_dossier_direct",
    "prompt_template_intervention_dossier_query",
)


def _expected_token_logp(token_id: int) -> float:
    normalizer = math.log(
        sum(math.exp(index * FAKE_TOKEN_LOGIT_STEP) for index in range(FAKE_VOCAB_SIZE))
    )
    return token_id * FAKE_TOKEN_LOGIT_STEP - normalizer


def _expected_sequence_logp(text: str) -> float:
    return sum(_expected_token_logp(1000 + ord(character)) for character in text)


class FakeTokenizer:
    chat_template = "fake-chat-template-v1"
    name_or_path = "fake/tokenizer"
    all_special_ids = (1, 2, 3)
    init_kwargs = {"revision": TOKENIZER_REVISION, "use_fast": True}
    _specials = {"<bos>": 1, "<user>": 2, "<assistant>": 3}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        rendered = f"<bos><user>{messages[0]['content']}"
        if add_generation_prompt:
            rendered += "<assistant>"
        if not tokenize:
            return rendered
        return self(rendered, add_special_tokens=False)["input_ids"]

    def __call__(self, text, *, add_special_tokens, **kwargs):
        assert add_special_tokens is False
        values = []
        offsets = []
        index = 0
        while index < len(text):
            special = next((item for item in self._specials if text.startswith(item, index)), None)
            if special is None:
                values.append(1000 + ord(text[index]))
                offsets.append((index, index + 1))
                index += 1
            else:
                values.append(self._specials[special])
                offsets.append((index, index + len(special)))
                index += len(special)
        result = {"input_ids": values}
        if kwargs.get("return_special_tokens_mask"):
            result["special_tokens_mask"] = [int(value in self.all_special_ids) for value in values]
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result


class FakeTeacherForcedModel(torch.nn.Module):
    """Stationary causal LM whose exact per-token probabilities are known to tests."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = 0

    def forward(self, input_ids):
        assert not torch.is_grad_enabled()
        self.calls += 1
        batch, sequence = input_ids.shape
        vocabulary_logits = (
            torch.arange(FAKE_VOCAB_SIZE, dtype=torch.float64, device=input_ids.device)
            * FAKE_TOKEN_LOGIT_STEP
        )
        logits = vocabulary_logits.view(1, 1, -1).expand(batch, sequence, -1)
        return SimpleNamespace(logits=logits)


class EvalTrackingTeacherForcedModel(FakeTeacherForcedModel):
    def __init__(self, *, sticky_training: bool = False):
        super().__init__()
        self.eval_calls = 0
        self.forward_training_states: list[bool] = []
        self.sticky_training = sticky_training
        self.train()

    def eval(self):
        self.eval_calls += 1
        if not self.sticky_training:
            return super().eval()
        return self

    def forward(self, input_ids):
        self.forward_training_states.append(self.training)
        return super().forward(input_ids)


def _payload(**changes):
    base = {
        "entity_unit_id": "entity-unit-1",
        "split": "mechanism_train",
        "template_family": "train_registry_direct",
        "target_familiarity": "screened_real",
        "distractor_familiarity": "matched_synthetic",
        "answerability": "code_absent",
        "target_text": "Ada",
        "distractor_text": "Bea",
        "registry_code": "K7M2Q",
        "expected_output": "UNKNOWN",
        "user_text": (
            "In the Alder Registry, Ada has archive color amber. Bea has archive shape oval. "
            "What is Ada's archive code? Answer with the code, or UNKNOWN if it is not stated."
        ),
        "target_intro_span": (0, 0),
        "target_query_span": (0, 0),
        "target_entity_id": "target-1",
        "distractor_entity_id": "distractor-1",
        "entity_order": "target_first",
        "query_role": "first",
        "relation_order": "code_first",
        "code_position": "first",
        "block": "factorial",
        "exposure": None,
    }
    base.update(changes)
    return base


def _example(tokenizer: FakeTokenizer, **changes) -> FAExample:
    payload = _payload(**changes)
    (
        payload["user_text"],
        payload["target_intro_span"],
        payload["target_query_span"],
    ) = _structured_user_text(
        family=payload["template_family"],
        target=payload["target_text"],
        distractor=payload["distractor_text"],
        answerability=payload["answerability"],
        entity_order=payload["entity_order"],
        registry_code=payload["registry_code"],
        block=payload["block"],
        exposure=payload["exposure"],
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": payload["user_text"]}],
        tokenize=True,
        add_generation_prompt=True,
    )
    payload["rendered_token_count"] = len(rendered)
    payload["rendered_token_ids"] = tuple(rendered)
    payload["special_token_sequence"] = (1, 2, 3)
    canonical = {
        key: payload[key]
        for key in (
            "entity_unit_id", "split", "template_family", "target_familiarity",
            "distractor_familiarity", "answerability", "target_text", "distractor_text",
            "registry_code", "expected_output", "user_text", "target_intro_span",
            "target_query_span", "target_entity_id", "distractor_entity_id", "entity_order",
            "query_role", "relation_order", "code_position", "rendered_token_count",
            "rendered_token_ids", "special_token_sequence", "block", "exposure",
        )
    }
    canonical["target_intro_span"] = list(canonical["target_intro_span"])
    canonical["target_query_span"] = list(canonical["target_query_span"])
    canonical["rendered_token_ids"] = list(canonical["rendered_token_ids"])
    canonical["special_token_sequence"] = list(canonical["special_token_sequence"])
    digest = hashlib.sha256(
        __import__("json").dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return FAExample(example_id=digest, canonical_payload_sha256=digest, **payload)


def _activation(example: FAExample, tokenizer: FakeTokenizer) -> ActivationRecord:
    anchors = resolve_registered_anchors(example, tokenizer)
    values = np.arange(3 * 26 * 2, dtype=np.float64).reshape(3, 26, 2)
    payload = __import__("trajectory_extractor.fa_activations", fromlist=["_activation_hash_payload"])
    digest = hashlib.sha256(
        payload._activation_hash_payload(example.example_id, tuple(range(26)), anchors.anchor_names, values)
    ).hexdigest()
    return ActivationRecord(
        example_id=example.example_id,
        anchors=anchors,
        layer_ids=tuple(range(26)),
        activations=values,
        dtype="float64",
        shape=values.shape,
        activation_sha256=digest,
        model_id="fake/model",
        model_revision=MODEL_REVISION,
    )


def _metadata_row(example: FAExample, **changes) -> dict[str, str]:
    row = {
        "example_id": example.example_id,
        "entity_id": example.entity_unit_id,
        "template_id": example.template_family,
        "relation_id": "archive_code",
        "domain": "person",
        "condition": "factorial",
    }
    row.update(changes)
    return row


def _metadata(example: FAExample, **changes) -> VerifiedDomainRelation:
    row = _metadata_row(example, **changes)
    return VerifiedDomainRelation.from_manifest(
        {
            "manifest_revision": "2026-07-22",
            "rows": [row],
        },
        example_id=example.example_id,
        entity_id=example.entity_unit_id,
        template_id=example.template_family,
    )


def _evidence(example: FAExample, tokenizer: FakeTokenizer, activation: ActivationRecord) -> OutputEvidence:
    model = FakeTeacherForcedModel()
    scorer = HFTeacherForcedScorer(
        model,
        tokenizer,
        model_id="fake/model",
        model_revision=MODEL_REVISION,
        config_sha256=CONFIG_SHA256,
    )
    evidence = scorer.score(example)
    assert model.calls == 2
    assert not torch.is_grad_enabled() or all(parameter.grad is None for parameter in model.parameters())
    return evidence


def test_teacher_forced_scoring_and_output_vector_are_exact_frozen_and_label_independent():
    tokenizer = FakeTokenizer()
    example = _example(tokenizer)
    activation = _activation(example, tokenizer)
    evidence = _evidence(example, tokenizer, activation)

    assert evidence.target_code == example.registry_code
    assert evidence.unknown_suffix == "UNKNOWN"
    assert evidence.prompt_input_ids == activation.anchors.input_ids
    expected_target_token_ids = tuple(1000 + ord(character) for character in "K7M2Q")
    expected_unknown_token_ids = tuple(1000 + ord(character) for character in "UNKNOWN")
    expected_target_token_logps = tuple(
        _expected_token_logp(token_id) for token_id in expected_target_token_ids
    )
    expected_unknown_token_logps = tuple(
        _expected_token_logp(token_id) for token_id in expected_unknown_token_ids
    )
    expected_target_logp = _expected_sequence_logp("K7M2Q")
    expected_unknown_logp = _expected_sequence_logp("UNKNOWN")
    assert evidence.target_token_ids == expected_target_token_ids
    assert evidence.unknown_token_ids == expected_unknown_token_ids
    assert evidence.target_logp == pytest.approx(expected_target_logp)
    assert evidence.unknown_logp == pytest.approx(expected_unknown_logp)
    assert evidence.target_logp == pytest.approx(sum(expected_target_token_logps))
    assert evidence.unknown_logp == pytest.approx(sum(expected_unknown_token_logps))
    assert evidence.target_logp != pytest.approx(
        sum(expected_target_token_logps) / len(expected_target_token_logps)
    )
    assert evidence.unknown_logp != pytest.approx(
        sum(expected_unknown_token_logps) / len(expected_unknown_token_logps)
    )
    assert OUTPUT_FEATURE_DIM == 11
    assert len(OUTPUT_FEATURE_NAMES) == OUTPUT_FEATURE_DIM
    vector = output_feature_vector(evidence)
    assert vector.shape == (11,)
    assert vector.flags.writeable is False
    maximum = max(expected_target_logp, expected_unknown_logp)
    logsumexp = maximum + math.log(
        math.exp(expected_target_logp - maximum)
        + math.exp(expected_unknown_logp - maximum)
    )
    target_probability = math.exp(expected_target_logp - logsumexp)
    unknown_probability = math.exp(expected_unknown_logp - logsumexp)
    expected = (
        expected_target_logp,
        expected_unknown_logp,
        expected_target_logp - expected_unknown_logp,
        maximum,
        logsumexp,
        target_probability,
        unknown_probability,
        -sum(
            probability * math.log(probability)
            for probability in (target_probability, unknown_probability)
            if probability > 0.0
        ),
        abs(target_probability - unknown_probability),
        target_probability - unknown_probability,
        max(target_probability, unknown_probability),
    )
    assert tuple(vector.tolist()) == pytest.approx(expected)
    assert vector[2] == pytest.approx(vector[0] - vector[1])
    assert vector[8] == pytest.approx(abs(vector[5] - vector[6]))
    assert vector[9] == pytest.approx(vector[5] - vector[6])

    feature = FeatureEvidence.from_records(example, activation, evidence, CONFIG_SHA256)
    row = build_probe_row(example, activation, feature, _metadata(example), task="familiarity")
    assert row.output_margin_features == pytest.approx(expected)
    assert row.output_control_schema_sha256 == OUTPUT_CONTROL_SCHEMA_SHA256
    assert row.output_evidence_sha256 == evidence.sha256
    assert row.answerability_condition == example.answerability
    assert row.target_familiarity_condition == example.target_familiarity
    assert row.distractor_familiarity_condition == example.distractor_familiarity
    changed = replace(row, label=0)
    assert changed.surface_features == row.surface_features
    assert changed.output_margin_features == row.output_margin_features
    assert changed.residual_features.tobytes() == row.residual_features.tobytes()
    with pytest.raises(ValueError):
        vector.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        evidence.target_logp = 0.0
    with pytest.raises(TypeError):
        feature.canonical_payload["output_evidence"]["prompt_input_ids"][0] = -1
    with pytest.raises(TypeError):
        feature.canonical_payload["output_evidence"]["model_id"] = "other/model"


def test_probe_surface_baseline_has_frozen_verified_domain_and_template_controls():
    tokenizer = FakeTokenizer()
    example = _example(tokenizer)
    activation = _activation(example, tokenizer)
    evidence = _evidence(example, tokenizer, activation)
    feature = FeatureEvidence.from_records(example, activation, evidence, CONFIG_SHA256)
    metadata = _metadata(example)

    familiarity = build_probe_row(example, activation, feature, metadata, task="familiarity")
    answerability = build_probe_row(example, activation, feature, metadata, task="answerability")
    unsupported = build_probe_row(
        example,
        activation,
        feature,
        metadata,
        task="unsupported_answer",
        outcome=UnsupportedAnswerOutcome(example.example_id, 1, "invalid"),
    )
    expected = (
        *surface_feature_vector(example),
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        *([0.0] * 17),
    )
    assert SURFACE_FEATURE_NAMES == EXPECTED_SURFACE_FEATURE_NAMES
    assert len(SURFACE_FEATURE_NAMES) == len(expected) == 37
    assert feature.surface_features == surface_feature_vector(example)
    assert familiarity.surface_features == expected
    assert answerability.surface_features == expected
    assert unsupported.surface_features == expected
    assert familiarity.label != answerability.label
    assert unsupported.outcome_status == "invalid"
    assert not any(
        forbidden in name
        for name in SURFACE_FEATURE_NAMES
        for forbidden in ("familiarity", "answerability", "outcome", "completion", "expected_output")
    )

    place = build_probe_row(
        example,
        activation,
        feature,
        _metadata(example, domain="place"),
        task="familiarity",
    )
    assert place.surface_features[:15] == familiarity.surface_features[:15]
    assert place.surface_features[15:19] == (0.0, 1.0, 0.0, 0.0)
    assert place.surface_features[19:] == familiarity.surface_features[19:]

    with pytest.raises(ValueError, match="registered entity domain"):
        build_probe_row(
            example,
            activation,
            feature,
            _metadata(example, domain="unregistered_domain"),
            task="familiarity",
        )


def test_teacher_forced_scoring_forces_eval_mode_and_rejects_models_stuck_in_training():
    tokenizer = FakeTokenizer()
    example = _example(tokenizer)

    tracking_model = EvalTrackingTeacherForcedModel()
    scorer = HFTeacherForcedScorer(
        tracking_model,
        tokenizer,
        model_id="fake/model",
        model_revision=MODEL_REVISION,
        config_sha256=CONFIG_SHA256,
    )
    scorer.score(example)
    assert tracking_model.eval_calls == 1
    assert tracking_model.forward_training_states == [False, False]

    stuck_model = EvalTrackingTeacherForcedModel(sticky_training=True)
    stuck_scorer = HFTeacherForcedScorer(
        stuck_model,
        tokenizer,
        model_id="fake/model",
        model_revision=MODEL_REVISION,
        config_sha256=CONFIG_SHA256,
    )
    with pytest.raises(ValueError, match="training mode"):
        stuck_scorer.score(example)


def test_features_exclude_labels_completion_and_unregistered_categorical_hashes():
    tokenizer = FakeTokenizer()
    example = _example(tokenizer)
    activation = _activation(example, tokenizer)
    evidence = _evidence(example, tokenizer, activation)
    feature = FeatureEvidence.from_records(example, activation, evidence, CONFIG_SHA256)

    assert surface_feature_vector(example) == feature.surface_features
    assert all(np.isfinite(feature.surface_features))
    assert example.target_familiarity not in feature.canonical_payload
    assert example.answerability not in feature.canonical_payload
    assert example.expected_output not in feature.canonical_payload
    assert "completion" not in feature.canonical_payload

    with pytest.raises(ValueError, match="membership"):
        VerifiedDomainRelation.from_manifest(
            {"rows": []},
            example_id=example.example_id,
            entity_id=example.entity_unit_id,
            template_id=example.template_family,
        )


@pytest.mark.parametrize("answerability", ("target_bound", "distractor_bound", "code_absent"))
def test_answerability_rows_cover_every_registered_class(answerability: str):
    tokenizer = FakeTokenizer()
    expected_output = "K7M2Q" if answerability == "target_bound" else "UNKNOWN"
    example = _example(tokenizer, answerability=answerability, expected_output=expected_output)
    activation = _activation(example, tokenizer)
    evidence = _evidence(example, tokenizer, activation)
    feature = FeatureEvidence.from_records(example, activation, evidence, CONFIG_SHA256)
    row = build_probe_row(example, activation, feature, _metadata(example), task="answerability")
    assert row.label == answerability


@pytest.mark.parametrize("answerability", ("distractor_bound", "code_absent"))
@pytest.mark.parametrize("outcome_status", ("valid", "missing", "invalid"))
def test_unsupported_answer_rows_cover_every_registered_status(
    answerability: str,
    outcome_status: str,
):
    tokenizer = FakeTokenizer()
    example = _example(tokenizer, answerability=answerability)
    activation = _activation(example, tokenizer)
    evidence = _evidence(example, tokenizer, activation)
    feature = FeatureEvidence.from_records(example, activation, evidence, CONFIG_SHA256)
    row = build_probe_row(
        example,
        activation,
        feature,
        _metadata(example),
        task="unsupported_answer",
        outcome=UnsupportedAnswerOutcome(example.example_id, 1, outcome_status),
    )
    assert row.label == 1
    assert row.outcome_status == outcome_status


def test_metadata_membership_is_hash_bound_and_exactly_matches_the_example():
    tokenizer = FakeTokenizer()
    example = _example(tokenizer)
    activation = _activation(example, tokenizer)
    evidence = _evidence(example, tokenizer, activation)
    feature = FeatureEvidence.from_records(example, activation, evidence, CONFIG_SHA256)
    metadata = _metadata(example)

    row = build_probe_row(example, activation, feature, metadata, task="familiarity")
    assert row.metadata_manifest_sha256 == metadata.metadata_manifest_sha256
    assert row.metadata_row_sha256 == metadata.metadata_row_sha256

    swapped_entity_metadata = VerifiedDomainRelation.from_manifest(
        {
            "rows": [
                {
                    "example_id": example.example_id,
                    "entity_id": "entity-unit-swapped",
                    "template_id": example.template_family,
                    "relation_id": "archive_code",
                    "domain": "person",
                    "condition": "factorial",
                }
            ]
        },
        example_id=example.example_id,
        entity_id="entity-unit-swapped",
        template_id=example.template_family,
    )
    with pytest.raises(ValueError, match="exact example metadata"):
        build_probe_row(example, activation, feature, swapped_entity_metadata, task="familiarity")

    swapped_template_metadata = VerifiedDomainRelation.from_manifest(
        {
            "rows": [
                {
                    "example_id": example.example_id,
                    "entity_id": example.entity_unit_id,
                    "template_id": "other_template_family",
                    "relation_id": "archive_code",
                    "domain": "person",
                    "condition": "factorial",
                }
            ]
        },
        example_id=example.example_id,
        entity_id=example.entity_unit_id,
        template_id="other_template_family",
    )
    with pytest.raises(ValueError, match="exact example metadata"):
        build_probe_row(example, activation, feature, swapped_template_metadata, task="familiarity")


@pytest.mark.parametrize(
    ("variant", "changes"),
    (
        ("exact duplicate", {}),
        ("relation ambiguity", {"relation_id": "archive_color"}),
        ("domain ambiguity", {"domain": "place"}),
        ("condition ambiguity", {"condition": "same_string"}),
    ),
)
def test_metadata_lookup_rejects_ambiguous_duplicate_exact_key_matches(
    variant: str,
    changes: dict[str, str],
):
    tokenizer = FakeTokenizer()
    example = _example(tokenizer)
    row = _metadata_row(example)
    duplicate = {**row, **changes}

    with pytest.raises(ValueError, match="ambiguous"):
        VerifiedDomainRelation.from_manifest(
            {"rows": [row, duplicate], "variant": variant},
            example_id=example.example_id,
            entity_id=example.entity_unit_id,
            template_id=example.template_family,
        )


@pytest.mark.parametrize(
    ("left", "right"),
    (
        ("relation_id", "domain"),
        ("relation_id", "condition"),
        ("domain", "condition"),
    ),
)
def test_verified_metadata_rejects_relation_domain_condition_field_swaps(
    left: str,
    right: str,
):
    tokenizer = FakeTokenizer()
    example = _example(tokenizer)
    row = _metadata_row(example)
    fields = {
        "relation_id": row["relation_id"],
        "domain": row["domain"],
        "condition": row["condition"],
    }
    fields[left], fields[right] = fields[right], fields[left]

    with pytest.raises(ValueError, match="does not match"):
        VerifiedDomainRelation(
            example_id=example.example_id,
            entity_id=example.entity_unit_id,
            template_id=example.template_family,
            relation_id=fields["relation_id"],
            domain=fields["domain"],
            condition=fields["condition"],
            metadata_manifest={"rows": [row]},
            metadata_row=row,
        )


def test_provenance_binding_labels_and_unsupported_rows_are_strict():
    tokenizer = FakeTokenizer()
    example = _example(tokenizer)
    activation = _activation(example, tokenizer)
    evidence = _evidence(example, tokenizer, activation)
    feature = FeatureEvidence.from_records(example, activation, evidence, CONFIG_SHA256)
    metadata = _metadata(example)

    familiarity = build_probe_row(example, activation, feature, metadata, task="familiarity")
    answerability = build_probe_row(example, activation, feature, metadata, task="answerability")
    unsupported = build_probe_row(
        example,
        activation,
        feature,
        metadata,
        task="unsupported_answer",
        outcome=UnsupportedAnswerOutcome(example.example_id, 1, "valid"),
    )
    assert familiarity.label == 1
    assert answerability.label == "code_absent"
    assert unsupported.label == 1
    assert unsupported.outcome_status == "valid"
    assert familiarity.residual_features.shape == (3, 26, 2)

    with pytest.raises(ValueError, match="provenance"):
        FeatureEvidence.from_records(example, activation, replace(evidence, model_revision="d" * 40), CONFIG_SHA256)
    with pytest.raises(ValueError, match="evidence-absent"):
        target_bound = _example(tokenizer, answerability="target_bound", expected_output="K7M2Q")
        target_activation = _activation(target_bound, tokenizer)
        target_feature = FeatureEvidence.from_records(
            target_bound,
            target_activation,
            _evidence(target_bound, tokenizer, target_activation),
            CONFIG_SHA256,
        )
        build_probe_row(
            target_bound,
            target_activation,
            target_feature,
            _metadata(target_bound),
            task="unsupported_answer",
            outcome=UnsupportedAnswerOutcome(target_bound.example_id, 1, "valid"),
        )


def test_batch_binding_is_a_deterministic_exact_id_multiset_without_invented_groups():
    tokenizer = FakeTokenizer()
    first = _example(tokenizer)
    second = _example(tokenizer, entity_unit_id="entity-unit-2", target_entity_id="target-2", distractor_entity_id="distractor-2")
    # The upstream example contract makes IDs content-addressed; rebuild its text with a distinct code.
    second = _example(
        tokenizer,
        entity_unit_id="entity-unit-2",
        target_entity_id="target-2",
        distractor_entity_id="distractor-2",
        registry_code="K8N3R",
    )
    records = []
    for example in (first, second):
        activation = _activation(example, tokenizer)
        feature = FeatureEvidence.from_records(example, activation, _evidence(example, tokenizer, activation), CONFIG_SHA256)
        records.append((example, activation, feature, _metadata(example)))

    expected_ids = [first.example_id, second.example_id]
    rows = build_probe_rows(records[::-1], task="familiarity", expected_example_ids=expected_ids)
    assert [row.example_id for row in rows] == sorted(example.example_id for example, *_ in records)
    with pytest.raises(ValueError, match="exact ID multiset"):
        build_probe_rows(records[:1], task="familiarity", expected_example_ids=expected_ids)
    with pytest.raises(ValueError, match="exact ID multiset"):
        build_probe_rows(
            [records[0], records[0]],
            task="familiarity",
            expected_example_ids=expected_ids,
        )
