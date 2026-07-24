from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, replace
from itertools import product
import hashlib
import json
import math
import unicodedata

import pytest

import trajectory_extractor.fa_data as fa_data
from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_data import (
    CONFIRMATORY_POWER_SIMULATIONS,
    REGISTERED_POWER_GRID,
    TEST_TEMPLATE_FAMILIES,
    TRAIN_TEMPLATE_FAMILIES,
    VALIDATION_TEMPLATE_FAMILIES,
    FAExample,
    FAManifest,
    PowerAudit,
    PowerCell,
    audit_dataset,
    build_factorial_examples,
    build_same_string_examples,
    build_manifest,
    simulate_interaction_power,
)
from trajectory_extractor.fa_entities import EntityMatch


class FakeTokenizer:
    all_special_ids = (0,)
    chat_template = "test-chat-template-v1"

    def encode(self, text, add_special_tokens=False):
        tokens = text.replace(".", " .").replace("?", " ?").split()
        return ([0] if add_special_tokens else []) + tokens + ([0] if add_special_tokens else [])

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        assert messages == [{"role": "user", "content": messages[0]["content"]}]
        return self.encode(messages[0]["content"], add_special_tokens=True)


class LegacyFakeTokenizer:
    all_special_ids = (0,)

    def encode(self, text, add_special_tokens=False):
        tokens = text.split()
        return ([0] if add_special_tokens else []) + tokens + ([0] if add_special_tokens else [])


class ChatTemplateTokenizer(FakeTokenizer):
    all_special_ids = (0, 91, 92)

    def __init__(self):
        self.template_revision = 1

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        raw = self.encode(messages[0]["content"], add_special_tokens=False)
        suffix = [92] * self.template_revision
        return [0, 91, *raw, *suffix, 0]


class VariableCodeTokenizer(FakeTokenizer):
    def encode(self, text, add_special_tokens=False):
        tokens = super().encode(text, add_special_tokens=False)
        split_tokens = []
        for token in tokens:
            bare = token.strip(".,:;?!")
            if fa_data._CODE.fullmatch(bare) and bare[-1] in "23456789":
                split_tokens.extend((bare[:3], bare[3:]))
            else:
                split_tokens.append(token)
        return ([0] if add_special_tokens else []) + split_tokens + ([0] if add_special_tokens else [])


def config() -> FAConfig:
    result = FAConfig(
        schema_version=1,
        profile="confirmatory",
        study_id="familiarity-answerability-gemma2-2b-v1",
        run_id="confirmatory-v1",
        model_id="google/gemma-2-2b-it",
        model_revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        tokenizer_revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        chat_template_sha256="ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6",
        split_seed=20260722,
        split_counts={
            "mechanism_train": 64,
            "locked_validation": 32,
            "behavior_test": 48,
            "probe_test": 24,
            "intervention_test": 24,
        },
        generation={"do_sample": False, "max_new_tokens": 16, "temperature": 0.0},
        bootstrap_replicates=10000,
        bootstrap_seed=20260722,
        thresholds={
            "format_validity_min": 0.95,
            "h1_min_interaction": 0.05,
            "h2_noninferiority_margin": 0.05,
            "h5_relative_log_loss_min": 0.02,
            "h6_relative_log_loss_min": 0.01,
            "h7_average_effect_min": 0.05,
            "h7_control_margin_min": 0.02,
            "intervention_accuracy_drop_max": 0.05,
            "intervention_control_rate_change_max": 0.03,
            "probe_auroc_min": 0.65,
            "probe_balanced_accuracy_min": 0.55,
            "sae_loss_recovery_min": 0.70,
            "sae_finite_fraction_min": 0.95,
            "circuit_proxy_spearman_min": 0.80,
            "circuit_distribution_spearman_min": 0.80,
            "circuit_perturbation_spearman_min": 0.60,
            "circuit_sign_concordance_min": 0.75,
        },
        anchors=("target_intro_end", "user_prompt_end", "assistant_prefix_end"),
    )
    # Keep confirmatory validation intact in production while pinning this fake
    # tokenizer's actual template bytes inside this test module.
    object.__setattr__(
        result,
        "chat_template_sha256",
        hashlib.sha256(FakeTokenizer.chat_template.encode("utf-8")).hexdigest(),
    )
    return result


def entity_unit(
    index: int = 0,
    *,
    split: str = "mechanism_train",
    coarse_type: str = "place",
) -> EntityMatch:
    suffix = str(index + 10)
    real_name = f"Old Vale{suffix}"
    synthetic_name = f"New Hill{suffix}"
    return EntityMatch(
        pair_id=f"Q{suffix}--syn-{suffix}",
        real_entity_id=f"Q{suffix}",
        real_qid=f"Q{suffix}",
        synthetic_candidate_id=f"syn-{suffix}",
        real_name=real_name,
        synthetic_name=synthetic_name,
        coarse_type=coarse_type,
        split=split,
        generator_revision="names-v1",
        tokenizer_revision="test-tokenizer-v1",
        real_token_count=2,
        synthetic_token_count=2,
        real_word_count=2,
        synthetic_word_count=2,
        real_character_count=len(real_name),
        synthetic_character_count=len(synthetic_name),
        character_length_delta=0,
        character_tolerance=2,
        capitalization_pattern_equal=True,
    )


def accented_entity_unit(index: int = 0, *, normalization: str) -> EntityMatch:
    real_name = unicodedata.normalize(normalization, "Jose\u0301 Vale10")
    synthetic_name = unicodedata.normalize(normalization, "Rene\u0301 Hill10")
    return replace(
        entity_unit(index),
        real_name=real_name,
        synthetic_name=synthetic_name,
        real_character_count=len(real_name),
        synthetic_character_count=len(synthetic_name),
        character_length_delta=len(synthetic_name) - len(real_name),
    )


def registered_examples(tokenizer=FakeTokenizer()):
    matches = (
        entity_unit(0, split="mechanism_train", coarse_type="place"),
        entity_unit(1, split="mechanism_train", coarse_type="person"),
        entity_unit(2, split="locked_validation", coarse_type="place"),
        entity_unit(3, split="behavior_test", coarse_type="organization"),
        entity_unit(4, split="probe_test", coarse_type="creative_work"),
        entity_unit(5, split="intervention_test", coarse_type="person"),
    )
    return build_factorial_examples(config(), matches, tokenizer=tokenizer), matches


@pytest.fixture(scope="module")
def full_confirmatory_design():
    matches = tuple(
        entity_unit(
            index,
            split=split,
            coarse_type=("place", "person", "organization", "creative_work")[index % 4],
        )
        for index, split in enumerate(
            ["mechanism_train"] * 64
            + ["locked_validation"] * 32
            + ["behavior_test"] * 48
            + ["probe_test"] * 24
            + ["intervention_test"] * 24
        )
    )
    tokenizer = FakeTokenizer()
    factorial = build_factorial_examples(config(), matches, tokenizer=tokenizer)
    same_string = build_same_string_examples(config(), matches, tokenizer=tokenizer)
    return factorial, same_string


def exhaustive_power_audit(rows, *, power=0.80):
    design_hash = simulate_interaction_power(
        rows,
        effect_grid=(0.0,),
        within_entity_correlations={
            "entity_icc": (0.05,),
            "template_icc": (0.02,),
            "invalid_format_rate": (0.0,),
        },
        simulations=1,
    ).design_sha256
    cells = tuple(
        PowerCell(
            absent_attempt_rate=absent_rate,
            entity_icc=entity_icc,
            template_icc=template_icc,
            invalid_format_rate=invalid_rate,
            interaction=interaction,
            estimated_power=power,
            monte_carlo_standard_error=math.sqrt(
                power * (1.0 - power) / CONFIRMATORY_POWER_SIMULATIONS
            ),
            simulations=CONFIRMATORY_POWER_SIMULATIONS,
        )
        for absent_rate, entity_icc, template_icc, invalid_rate, interaction in product(
            REGISTERED_POWER_GRID.absent_attempt_rates,
            REGISTERED_POWER_GRID.entity_iccs,
            REGISTERED_POWER_GRID.template_iccs,
            REGISTERED_POWER_GRID.invalid_format_rates,
            REGISTERED_POWER_GRID.interactions,
        )
    )
    return PowerAudit(
        design_sha256=design_hash,
        seed=20260722,
        simulations=CONFIRMATORY_POWER_SIMULATIONS,
        cells=cells,
        registered_grid=True,
    )


def stub_confirmatory_recomputation(monkeypatch, expected):
    def recompute(rows):
        assert fa_data._design_sha256(rows) == expected.design_sha256
        return expected

    monkeypatch.setattr(fa_data, "_recompute_confirmatory_power_audit", recompute)


def test_each_entity_unit_expands_over_its_registered_template_families():
    tokenizer = FakeTokenizer()
    rows, _ = registered_examples(tokenizer)

    expected_families = {
        "mechanism_train": TRAIN_TEMPLATE_FAMILIES,
        "locked_validation": VALIDATION_TEMPLATE_FAMILIES,
        "behavior_test": TEST_TEMPLATE_FAMILIES,
        "probe_test": fa_data.PROBE_TEMPLATE_FAMILIES,
        "intervention_test": fa_data.INTERVENTION_TEMPLATE_FAMILIES,
    }
    for split, families in expected_families.items():
        split_rows = [row for row in rows if row.split == split]
        unit_count = len({row.entity_unit_id for row in split_rows})
        assert len(split_rows) == 12 * len(families) * unit_count
        assert {row.template_family for row in split_rows} == set(families)
        assert Counter(
            (row.target_familiarity, row.distractor_familiarity, row.answerability)
            for row in split_rows
        ) == Counter(
            {
                (target, distractor, answerability): len(families) * unit_count
                for target in ("screened_real", "matched_synthetic")
                for distractor in ("screened_real", "matched_synthetic")
                for answerability in ("target_bound", "distractor_bound", "code_absent")
            }
        )


def test_target_and_distractor_bound_pairs_preserve_lexical_multiset_and_code():
    rows, _ = registered_examples()
    target = next(row for row in rows if row.answerability == "target_bound")
    distractor = next(
        row
        for row in rows
        if row.entity_unit_id == target.entity_unit_id
        and row.template_family == target.template_family
        and row.target_familiarity == target.target_familiarity
        and row.distractor_familiarity == target.distractor_familiarity
        and row.answerability == "distractor_bound"
    )

    tokenizer = FakeTokenizer()
    assert Counter(tokenizer.encode(target.user_text)) == Counter(tokenizer.encode(distractor.user_text))
    assert target.registry_code == distractor.registry_code
    assert target.rendered_token_count == distractor.rendered_token_count
    assert target.special_token_sequence == distractor.special_token_sequence


def test_same_string_rows_fix_target_and_token_budget_and_use_one_balanced_family():
    factorial, matches = registered_examples()
    rows = build_same_string_examples(config(), matches, tokenizer=FakeTokenizer())

    assert len(rows) == 4 * len(matches)
    for match in matches:
        unit_rows = [row for row in rows if row.entity_unit_id == match.pair_id]
        assert {row.target_text for row in unit_rows} == {match.synthetic_name}
        assert {(row.exposure, row.answerability) for row in unit_rows} == {
            ("high_exposure", "target_bound"),
            ("high_exposure", "code_absent"),
            ("low_exposure", "target_bound"),
            ("low_exposure", "code_absent"),
        }
        assert {row.target_familiarity for row in unit_rows} == {"matched_synthetic"}
        assert {row.distractor_familiarity for row in unit_rows} == {"screened_real"}
        assert len({row.template_family for row in unit_rows}) == 1
        by_answerability = {
            answerability: [row for row in unit_rows if row.answerability == answerability]
            for answerability in ("target_bound", "code_absent")
        }
        for pair in by_answerability.values():
            assert pair[0].rendered_token_count == pair[1].rendered_token_count
            assert pair[0].special_token_sequence == pair[1].special_token_sequence

    assert not {row.example_id for row in factorial} & {row.example_id for row in rows}


def test_complete_confirmatory_construction_requires_exact_domain_balance_in_every_split():
    split_sequence = (
        ["mechanism_train"] * 64
        + ["locked_validation"] * 32
        + ["behavior_test"] * 48
        + ["probe_test"] * 24
        + ["intervention_test"] * 24
    )
    domains = ("person", "place", "organization", "creative_work")
    matches = tuple(
        entity_unit(
            index,
            split=split,
            coarse_type=(
                "place"
                if split == "behavior_test" and index == 96
                else domains[index % 4]
            ),
        )
        for index, split in enumerate(split_sequence)
    )

    with pytest.raises(ValueError, match="exactly balanced across the four"):
        build_factorial_examples(config(), matches, tokenizer=FakeTokenizer())


def test_same_string_exposure_prefixes_are_neutral_unrelated_and_target_matched():
    _factorial, matches = registered_examples()
    tokenizer = FakeTokenizer()
    rows = build_same_string_examples(config(), matches, tokenizer=tokenizer)
    prohibited = {
        "archive",
        "registry",
        "catalog",
        "ledger",
        "code",
        "answer",
        "answerability",
        "unknown",
        "uncertain",
        "uncertainty",
        "abstain",
        "abstention",
    }

    for match in matches:
        unit_rows = [row for row in rows if row.entity_unit_id == match.pair_id]
        for row in unit_rows:
            prefix, separator, task = row.user_text.partition(" Task: ")
            assert separator
            prefix_words = {word.casefold().strip(".,:;?!") for word in prefix.split()}
            assert prefix_words.isdisjoint(prohibited)
            assert row.target_text in task
            if row.exposure == "high_exposure":
                assert prefix.count(row.target_text) == 4
            else:
                assert row.target_text not in prefix

        for answerability in ("target_bound", "code_absent"):
            high, low = sorted(
                (row for row in unit_rows if row.answerability == answerability),
                key=lambda row: row.exposure,
            )
            high_prefix = high.user_text.partition(" Task: ")[0]
            low_prefix = low.user_text.partition(" Task: ")[0]
            assert len(tokenizer.encode(high_prefix)) == len(tokenizer.encode(low_prefix))
            assert high.user_text.partition(" Task: ")[2] == low.user_text.partition(" Task: ")[2]


def test_template_families_have_distinct_labels_and_registered_prompt_bytes():
    family_sets = tuple(fa_data._FAMILIES_BY_SPLIT[split] for split in config().split_counts)
    flattened = [family for families in family_sets for family in families]
    assert len(flattened) == len(set(flattened))

    rendered = {
        fa_data._TEMPLATE_TEXT[family].format(
            first="First Entity",
            second="Second Entity",
            first_relation="code",
            first_value="KABCD",
            second_relation="color",
            second_value="amber",
            query="First Entity",
        ).encode("utf-8")
        for family in flattened
    }
    assert len(rendered) == len(flattened)


def test_every_rendered_prompt_ends_with_the_exact_output_contract():
    rows, matches = registered_examples()
    same_string = build_same_string_examples(
        config(), matches, tokenizer=FakeTokenizer()
    )

    assert all(
        row.user_text.endswith(fa_data._EXACT_OUTPUT_INSTRUCTION)
        for row in (*rows, *same_string)
    )


def test_template_audit_rejects_duplicate_registered_content(monkeypatch):
    monkeypatch.setitem(
        fa_data._TEMPLATE_TEXT,
        "train_registry_possessive",
        fa_data._TEMPLATE_TEXT["train_registry_direct"],
    )
    rows, matches = registered_examples()
    same_string = build_same_string_examples(config(), matches, tokenizer=FakeTokenizer())

    assert not audit_dataset(rows, same_string, tokenizer=FakeTokenizer()).checks["template_overlap"]


@pytest.mark.parametrize("replacement", ["archive", "records"])
def test_same_string_audit_rejects_unregistered_prefix_concepts_without_length_drift(replacement):
    _factorial, matches = registered_examples()
    tokenizer = FakeTokenizer()
    rows = list(build_same_string_examples(config(), matches, tokenizer=tokenizer))
    target = next(row for row in rows if row.exposure == "high_exposure")
    tampered_text = target.user_text.replace("visits", replacement, 1)
    token_ids, special_tokens = fa_data._token_metadata(tampered_text, tokenizer)
    object.__setattr__(target, "user_text", tampered_text)
    object.__setattr__(target, "rendered_token_ids", token_ids)
    object.__setattr__(target, "rendered_token_count", len(token_ids))
    object.__setattr__(target, "special_token_sequence", special_tokens)

    audit = audit_dataset(registered_examples()[0], rows, tokenizer=tokenizer)
    assert not audit.checks["same_string_token_budget"]


def test_code_vocabulary_is_tokenizer_filtered_single_class_and_deterministic():
    _rows, matches = registered_examples()
    tokenizer = VariableCodeTokenizer()
    forward = build_factorial_examples(config(), matches, tokenizer=tokenizer)
    reverse = build_factorial_examples(config(), tuple(reversed(matches)), tokenizer=tokenizer)

    code_by_unit = {row.entity_unit_id: row.registry_code for row in forward}
    assert code_by_unit == {row.entity_unit_id: row.registry_code for row in reverse}
    assert len({len(tokenizer.encode(code)) for code in code_by_unit.values()}) == 1

    for unit_id, code in code_by_unit.items():
        unit_rows = [row for row in forward if row.entity_unit_id == unit_id]
        assert {row.registry_code for row in unit_rows} == {code}
        assert len({row.split for row in unit_rows}) == 1
        assert Counter(row.target_familiarity for row in unit_rows).most_common()[0][1] == len(
            unit_rows
        ) // 2
        assert Counter(row.distractor_familiarity for row in unit_rows).most_common()[0][1] == len(
            unit_rows
        ) // 2
        assert Counter(row.entity_order for row in unit_rows) == Counter(
            {"target_first": len(unit_rows) // 2, "target_second": len(unit_rows) // 2}
        )

    same_string = build_same_string_examples(config(), matches, tokenizer=tokenizer)
    assert audit_dataset(forward, same_string, tokenizer=tokenizer).checks["code_vocabulary"]


def test_code_vocabulary_audit_rejects_a_regex_valid_code_from_another_token_class():
    tokenizer = VariableCodeTokenizer()
    rows, matches = registered_examples(tokenizer)
    same_string = build_same_string_examples(config(), matches, tokenizer=tokenizer)
    target_unit = rows[0].entity_unit_id
    replacement_code = next(
        code
        for code in ("KAAA2", "KAAAA")
        if len(tokenizer.encode(code)) != len(tokenizer.encode(rows[0].registry_code))
    )
    assert len(tokenizer.encode(replacement_code)) != len(tokenizer.encode(rows[0].registry_code))
    for row in (*rows, *same_string):
        if row.entity_unit_id == target_unit:
            object.__setattr__(row, "registry_code", replacement_code)

    audit = audit_dataset(rows, same_string, tokenizer=tokenizer)
    assert not audit.checks["code_vocabulary"]


def test_code_vocabulary_audit_rejects_globally_consistent_but_unregistered_token_class():
    tokenizer = VariableCodeTokenizer()
    rows, matches = registered_examples(tokenizer)
    same_string = build_same_string_examples(config(), matches, tokenizer=tokenizer)
    registered_length = len(tokenizer.encode(rows[0].registry_code))
    replacements = {}
    used = set()
    for row in rows:
        if row.entity_unit_id in replacements:
            continue
        for attempt in range(256, 1024):
            candidate = fa_data._code_for(row.split, row.entity_unit_id, attempt)
            if (
                candidate not in used
                and len(tokenizer.encode(candidate)) != registered_length
            ):
                replacements[row.entity_unit_id] = candidate
                used.add(candidate)
                break

    assert len(replacements) == len({row.entity_unit_id for row in rows})
    assert len({len(tokenizer.encode(code)) for code in replacements.values()}) == 1
    for row in (*rows, *same_string):
        object.__setattr__(row, "registry_code", replacements[row.entity_unit_id])

    audit = audit_dataset(rows, same_string, tokenizer=tokenizer)
    assert not audit.checks["code_vocabulary"]


def test_rendered_token_metadata_uses_actual_chat_template_and_generation_prompt():
    tokenizer = ChatTemplateTokenizer()
    rows, _ = registered_examples(tokenizer)
    row = rows[0]
    expected = tokenizer.apply_chat_template(
        [{"role": "user", "content": row.user_text}],
        tokenize=True,
        add_generation_prompt=True,
    )

    assert row.rendered_token_ids == tuple(expected)
    assert row.rendered_token_count == len(expected)
    assert row.special_token_sequence == (0, 91, 92, 0)
    assert row.rendered_token_ids != tuple(tokenizer.encode(row.user_text, add_special_tokens=True))


def test_rendered_token_audit_detects_chat_template_only_changes():
    tokenizer = ChatTemplateTokenizer()
    rows, matches = registered_examples(tokenizer)
    same_string = build_same_string_examples(config(), matches, tokenizer=tokenizer)
    tokenizer.template_revision = 2

    audit = audit_dataset(rows, same_string, tokenizer=tokenizer)
    assert not audit.checks["rendered_token_length"]
    assert not audit.checks["special_token_sequence"]


def test_confirmatory_construction_requires_apply_chat_template_but_fake_adapter_is_conservative():
    legacy = LegacyFakeTokenizer()
    _rows, matches = registered_examples()

    with pytest.raises(ValueError, match="apply_chat_template"):
        build_factorial_examples(config(), matches, tokenizer=legacy)

    token_ids, special_tokens = fa_data._token_metadata("plain fake prompt", legacy)
    assert token_ids == (0, "plain", "fake", "prompt", 0)
    assert special_tokens == (0, 0)


def test_confirmatory_construction_rejects_wrong_template_bytes_before_rendering():
    class WrongTemplateTokenizer(FakeTokenizer):
        chat_template = "test-chat-template-v2"

        def __init__(self):
            self.render_calls = 0

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            self.render_calls += 1
            raise AssertionError("wrong template must be rejected before rendering")

    _rows, matches = registered_examples()
    tokenizer = WrongTemplateTokenizer()

    with pytest.raises(ValueError, match="chat_template_sha256"):
        build_factorial_examples(config(), matches, tokenizer=tokenizer)

    assert tokenizer.render_calls == 0


def test_examples_and_manifests_are_immutable_and_content_addressed():
    rows, _ = registered_examples()
    test_config = config()
    ordered = tuple(sorted(rows, key=lambda row: row.example_id))
    manifest = FAManifest(
        config_hash=test_config.config_hash,
        examples=ordered,
        manifest_sha256=fa_data._manifest_sha256(test_config.config_hash, ordered),
    )

    assert isinstance(manifest, FAManifest)
    assert len(manifest.manifest_sha256) == 64
    assert rows[0].canonical_payload_sha256 in rows[0].example_id
    with pytest.raises(FrozenInstanceError):
        rows[0].user_text = "changed"
    with pytest.raises(ValueError, match="canonical_payload_sha256"):
        replace(rows[0], canonical_payload_sha256="0" * 64)


def test_examples_commit_to_immutable_structured_target_role_spans():
    rows, matches = registered_examples()
    row = rows[0]
    payload = fa_data._example_payload(row)

    assert row.user_text[slice(*row.target_intro_span)] == row.target_text
    assert row.user_text[slice(*row.target_query_span)] == row.target_text
    assert row.target_intro_span[1] <= row.target_query_span[0]
    assert payload["target_intro_span"] == list(row.target_intro_span)
    assert payload["target_query_span"] == list(row.target_query_span)
    altered_payload = {**payload, "target_intro_span": list(row.target_query_span)}
    assert fa_data._payload_sha256(altered_payload) != row.example_id
    with pytest.raises(FrozenInstanceError):
        row.target_intro_span = row.target_query_span

    same_string = next(
        item
        for item in build_same_string_examples(config(), matches, tokenizer=FakeTokenizer())
        if item.exposure == "high_exposure"
    )
    assert same_string.user_text[: same_string.target_intro_span[0]].count(
        same_string.target_text
    ) == 4
    assert same_string.target_intro_span[0] > same_string.user_text.index(" Task: ")


def test_exactly_two_target_substrings_cannot_override_structured_template_roles():
    rows, _ = registered_examples()
    row = next(item for item in rows if item.user_text.count(item.target_text) == 2)

    with pytest.raises(ValueError, match="structured target spans"):
        replace(
            row,
            target_intro_span=row.target_query_span,
            target_query_span=row.target_intro_span,
        )


def test_same_string_prefix_occurrence_cannot_impersonate_target_introduction():
    _rows, matches = registered_examples()
    row = next(
        item
        for item in build_same_string_examples(config(), matches, tokenizer=FakeTokenizer())
        if item.exposure == "high_exposure"
    )
    prefix_start = row.user_text.index(row.target_text)

    with pytest.raises(ValueError, match="template semantics"):
        replace(
            row,
            target_intro_span=(prefix_start, prefix_start + len(row.target_text)),
        )


@pytest.mark.parametrize(
    "field",
    ("target_text", "distractor_text", "expected_output", "user_text"),
)
def test_fa_example_constructor_rejects_non_nfc_text_fields_from_loaded_records(field):
    row = build_factorial_examples(
        config(),
        (accented_entity_unit(normalization="NFC"),),
        tokenizer=FakeTokenizer(),
    )[0]
    payload = {
        "example_id": row.example_id,
        "canonical_payload_sha256": row.canonical_payload_sha256,
        **fa_data._example_payload(row),
    }
    payload[field] = unicodedata.normalize("NFD", "Jos\u00e9")
    loaded_payload = json.loads(json.dumps(payload, ensure_ascii=False))

    with pytest.raises(ValueError, match=rf"{field}.*Unicode NFC"):
        FAExample(**loaded_payload)


@pytest.mark.parametrize(
    "field",
    ("rendered_token_ids", "special_token_sequence"),
)
def test_fa_example_constructor_rejects_non_nfc_strings_in_token_sequences(field):
    row = build_factorial_examples(
        config(),
        (accented_entity_unit(normalization="NFC"),),
        tokenizer=FakeTokenizer(),
    )[0]
    payload = fa_data._example_payload(row)
    payload[field][0] = unicodedata.normalize("NFD", "Jos\u00e9")
    digest = fa_data._payload_sha256(payload)
    loaded_payload = json.loads(
        json.dumps(
            {
                "example_id": digest,
                "canonical_payload_sha256": digest,
                **payload,
            },
            ensure_ascii=False,
        )
    )

    with pytest.raises(ValueError, match=rf"{field}.*Unicode NFC"):
        FAExample(**loaded_payload)


@pytest.mark.parametrize(
    "exposure,expected_subject",
    (("high_exposure", "target"), ("low_exposure", "distractor")),
)
def test_same_string_reconstruction_normalizes_nfd_subject_target_and_distractor(
    exposure, expected_subject
):
    target = unicodedata.normalize("NFD", "Ren\u00e9 Hill10")
    distractor = unicodedata.normalize("NFD", "Jos\u00e9 Vale10")

    user_text, intro_span, query_span = fa_data._structured_user_text(
        family="train_registry_direct",
        target=target,
        distractor=distractor,
        answerability="target_bound",
        entity_order="target_second",
        registry_code="K7M2Q",
        block="same_string",
        exposure=exposure,
    )

    nfc_target = unicodedata.normalize("NFC", target)
    nfc_distractor = unicodedata.normalize("NFC", distractor)
    prefix, separator, task = user_text.partition(" Task: ")
    subject = nfc_target if expected_subject == "target" else nfc_distractor
    assert separator
    assert unicodedata.is_normalized("NFC", user_text)
    assert prefix.count(subject) == 4
    assert nfc_target in task and nfc_distractor in task
    assert user_text[slice(*intro_span)] == nfc_target
    assert user_text[slice(*query_span)] == nfc_target


def test_decomposed_entity_names_are_normalized_before_factorial_spans_and_hashes():
    decomposed = (accented_entity_unit(normalization="NFD"),)
    precomposed = (accented_entity_unit(normalization="NFC"),)

    actual = build_factorial_examples(config(), decomposed, tokenizer=FakeTokenizer())
    expected = build_factorial_examples(config(), precomposed, tokenizer=FakeTokenizer())

    assert actual == expected
    for row in actual:
        assert unicodedata.is_normalized("NFC", row.target_text)
        assert unicodedata.is_normalized("NFC", row.distractor_text)
        assert unicodedata.is_normalized("NFC", row.user_text)
        assert row.user_text[slice(*row.target_intro_span)] == row.target_text
        assert row.user_text[slice(*row.target_query_span)] == row.target_text
        assert row.canonical_payload_sha256 == fa_data._example_sha256(row)


def test_decomposed_entity_names_are_normalized_before_same_string_spans_and_hashes():
    decomposed = (accented_entity_unit(normalization="NFD"),)
    precomposed = (accented_entity_unit(normalization="NFC"),)

    actual = build_same_string_examples(config(), decomposed, tokenizer=FakeTokenizer())
    expected = build_same_string_examples(config(), precomposed, tokenizer=FakeTokenizer())

    assert actual == expected
    for row in actual:
        assert unicodedata.is_normalized("NFC", row.target_text)
        assert unicodedata.is_normalized("NFC", row.distractor_text)
        assert unicodedata.is_normalized("NFC", row.user_text)
        assert row.user_text[slice(*row.target_intro_span)] == row.target_text
        assert row.user_text[slice(*row.target_query_span)] == row.target_text
        assert row.canonical_payload_sha256 == fa_data._example_sha256(row)


def test_audit_covers_registered_controls_and_detects_lexical_tampering():
    rows, _ = registered_examples()
    same_string = build_same_string_examples(config(), _, tokenizer=FakeTokenizer())
    audit = audit_dataset(rows, same_string, tokenizer=FakeTokenizer())

    assert audit.passed
    assert set(audit.checks) == {
        "independent_target_distractor_variation",
        "entity_order",
        "query_role",
        "relation_order",
        "code_position",
        "code_vocabulary",
        "template_overlap",
        "entity_overlap",
        "rendered_token_length",
        "special_token_sequence",
        "lexical_multiset",
        "same_string_token_budget",
    }
    tampered = next(row for row in rows if row.answerability == "distractor_bound")
    object.__setattr__(tampered, "user_text", "tampered")
    object.__setattr__(tampered, "rendered_token_count", 1)
    object.__setattr__(tampered, "rendered_token_ids", ("tampered",))
    broken = audit_dataset(rows, same_string, tokenizer=FakeTokenizer())
    assert not broken.passed
    assert not broken.checks["lexical_multiset"]


def test_power_simulation_uses_registered_grid_is_deterministic_and_has_fast_override():
    rows, _ = registered_examples()
    first = simulate_interaction_power(
        rows,
        effect_grid=(0.0, 0.05),
        within_entity_correlations={"entity_icc": (0.30,), "template_icc": (0.10,), "invalid_format_rate": (0.05,)},
        seed=20260722,
        simulations=25,
    )
    second = simulate_interaction_power(
        rows,
        effect_grid=(0.0, 0.05),
        within_entity_correlations={"entity_icc": (0.30,), "template_icc": (0.10,), "invalid_format_rate": (0.05,)},
        seed=20260722,
        simulations=25,
    )

    assert first == second
    assert len(first.cells) == 3 * 1 * 1 * 1 * 2
    assert all(0.0 <= cell.estimated_power <= 1.0 for cell in first.cells)
    assert all(cell.monte_carlo_standard_error >= 0.0 for cell in first.cells)
    assert REGISTERED_POWER_GRID.interactions[2] == 0.05
    assert CONFIRMATORY_POWER_SIMULATIONS == 2000


def test_power_simulation_samples_binary_and_invalid_events_not_analytic_gaussians(monkeypatch):
    rows, _ = registered_examples()

    def analytic_approximation_is_forbidden(*_args, **_kwargs):
        raise AssertionError("analytic Gaussian approximation was called")

    monkeypatch.setattr(
        fa_data,
        "_conservative_interaction_se",
        analytic_approximation_is_forbidden,
        raising=False,
    )
    audit = simulate_interaction_power(
        rows,
        effect_grid=(0.10,),
        within_entity_correlations={
            "entity_icc": (0.30,),
            "template_icc": (0.10,),
            "invalid_format_rate": (1.0,),
        },
        simulations=12,
    )

    assert len(audit.cells) == 3
    assert all(cell.estimated_power == 0.0 for cell in audit.cells)
    assert all(cell.monte_carlo_standard_error == 0.0 for cell in audit.cells)


def test_power_simulation_draws_invalid_events_independently_for_each_simulation(monkeypatch):
    rows, _ = registered_examples()
    original_default_rng = fa_data.np.random.default_rng
    observed_sizes = []

    class RecordingGenerator:
        def __init__(self, seed):
            self._generator = original_default_rng(seed)

        def normal(self, *args, **kwargs):
            return self._generator.normal(*args, **kwargs)

        def binomial(self, n, p, size=None):
            observed_sizes.append(size)
            return self._generator.binomial(n, p, size=size)

    monkeypatch.setattr(fa_data.np.random, "default_rng", RecordingGenerator)
    simulations = 7
    simulate_interaction_power(
        rows,
        effect_grid=(0.05,),
        within_entity_correlations={
            "entity_icc": (0.05,),
            "template_icc": (0.02,),
            "invalid_format_rate": (0.05,),
        },
        simulations=simulations,
    )

    grouped_rows = len(fa_data._prepare_power_design(rows)["repetitions"])
    assert observed_sizes[0] == (simulations, grouped_rows)


def test_power_design_registers_entity_template_intersections_for_two_way_clustering():
    rows, _ = registered_examples()
    prepared = fa_data._prepare_power_design(rows)

    assert prepared is not None
    expected_intersections = {
        (row.entity_unit_id, row.split, row.template_family) for row in rows
    }
    assert prepared["intersection_membership"].shape[1] == len(expected_intersections)
    assert (prepared["intersection_membership"].sum(axis=1) == 1.0).all()


def test_logistic_random_effects_preserve_registered_marginal_probabilities():
    entity_variance, template_variance = fa_data._joint_logit_random_effect_variances(0.30, 0.10)
    total_variance = entity_variance + template_variance

    for probability in (0.10, 0.25, 0.50, 0.55, 0.80):
        intercept = fa_data._calibrated_logit_intercept(probability, total_variance)
        assert fa_data._logistic_normal_mean(intercept, total_variance) == pytest.approx(
            probability,
            abs=1e-10,
        )


@pytest.mark.parametrize("entity_icc,template_icc", [(0.0, 0.0), (0.05, 0.02), (0.30, 0.10)])
def test_joint_logistic_random_effect_variances_realize_both_iccs(entity_icc, template_icc):
    entity_variance, template_variance = fa_data._joint_logit_random_effect_variances(
        entity_icc,
        template_icc,
    )
    latent_total = entity_variance + template_variance + math.pi**2 / 3.0

    assert entity_variance / latent_total == pytest.approx(entity_icc)
    assert template_variance / latent_total == pytest.approx(template_icc)


@pytest.mark.parametrize("entity_icc,template_icc", [(-0.01, 0.10), (0.30, -0.01), (0.60, 0.40)])
def test_joint_logistic_random_effect_variances_reject_invalid_pairs(entity_icc, template_icc):
    with pytest.raises(ValueError, match="ICCs"):
        fa_data._joint_logit_random_effect_variances(entity_icc, template_icc)


def test_confirmatory_power_recomputation_uses_exact_registered_execution(monkeypatch):
    rows, _ = registered_examples()
    sentinel = exhaustive_power_audit(rows)
    calls = []

    def simulate(design, effect_grid, within_entity_correlations, seed, *, simulations):
        calls.append(
            (tuple(design), effect_grid, within_entity_correlations, seed, simulations)
        )
        return sentinel

    monkeypatch.setattr(fa_data, "simulate_interaction_power", simulate)

    assert fa_data._recompute_confirmatory_power_audit(rows) is sentinel
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
    assert len(sentinel.cells) == 180


def test_confirmatory_manifest_fails_closed_without_a_registered_power_audit():
    matches = tuple(
        entity_unit(index, split=split, coarse_type=("place", "person", "organization", "creative_work")[index % 4])
        for index, split in enumerate(
            ["mechanism_train"] * 64
            + ["locked_validation"] * 32
            + ["behavior_test"] * 48
            + ["probe_test"] * 24
            + ["intervention_test"] * 24
        )
    )
    tokenizer = FakeTokenizer()
    rows = build_factorial_examples(config(), matches, tokenizer=tokenizer)
    same_string = build_same_string_examples(config(), matches, tokenizer=tokenizer)

    with pytest.raises(ValueError, match="power audit"):
        build_manifest(config(), rows + same_string)


def test_confirmatory_gate_recognizes_factorial_corpus_when_same_string_rows_are_included(
    full_confirmatory_design,
    monkeypatch,
):
    factorial, same_string = full_confirmatory_design

    with pytest.raises(ValueError, match="power audit"):
        build_manifest(config(), factorial + same_string)

    audit = exhaustive_power_audit(factorial)
    stub_confirmatory_recomputation(monkeypatch, audit)
    manifest = build_manifest(config(), factorial + same_string, power_audit=audit)
    assert manifest.power_audit == audit


def test_confirmatory_gate_rejects_structurally_valid_fabricated_power_audit(
    full_confirmatory_design,
    monkeypatch,
):
    factorial, same_string = full_confirmatory_design
    recomputed = exhaustive_power_audit(factorial, power=0.81)
    fabricated = exhaustive_power_audit(factorial, power=0.95)
    stub_confirmatory_recomputation(monkeypatch, recomputed)

    with pytest.raises(ValueError, match="exact registered recomputation"):
        build_manifest(config(), factorial + same_string, power_audit=fabricated)


def test_manifest_hash_commits_to_canonical_power_audit_content(
    full_confirmatory_design,
    monkeypatch,
):
    factorial, same_string = full_confirmatory_design
    first_audit = exhaustive_power_audit(factorial, power=0.80)
    stub_confirmatory_recomputation(monkeypatch, first_audit)
    first = build_manifest(config(), factorial + same_string, power_audit=first_audit)

    second_audit = exhaustive_power_audit(factorial, power=0.81)
    stub_confirmatory_recomputation(monkeypatch, second_audit)
    second = build_manifest(config(), factorial + same_string, power_audit=second_audit)

    assert first.manifest_sha256 != second.manifest_sha256


@pytest.mark.parametrize("missing_block", ["factorial", "same_string"])
def test_confirmatory_manifest_rejects_any_missing_design_row(
    full_confirmatory_design,
    monkeypatch,
    missing_block,
):
    factorial, same_string = full_confirmatory_design
    rows = list(factorial + same_string)
    rows.remove(next(row for row in rows if row.block == missing_block))

    def power_must_not_run(_rows):
        raise AssertionError("incomplete design must fail before power recomputation")

    monkeypatch.setattr(fa_data, "_recompute_confirmatory_power_audit", power_must_not_run, raising=False)

    with pytest.raises(ValueError, match="complete factorial and same-string design"):
        build_manifest(
            config(),
            rows,
            power_audit=exhaustive_power_audit(factorial),
        )


@pytest.mark.parametrize(
    "tamper",
    [
        lambda audit: replace(audit, cells=audit.cells[:1]),
        lambda audit: replace(audit, cells=audit.cells[:-1] + (audit.cells[0],)),
        lambda audit: replace(
            audit,
            cells=(
                replace(audit.cells[0], monte_carlo_standard_error=float("nan")),
                *audit.cells[1:],
            ),
        ),
        lambda audit: replace(
            audit,
            cells=(replace(audit.cells[0], simulations=1), *audit.cells[1:]),
        ),
        lambda audit: replace(audit, design_sha256="0" * 64),
    ],
)
def test_confirmatory_gate_rejects_partial_fabricated_or_invalid_power_audits(
    full_confirmatory_design,
    tamper,
):
    factorial, same_string = full_confirmatory_design
    audit = tamper(exhaustive_power_audit(factorial))

    with pytest.raises(ValueError, match="passing registered power audit"):
        build_manifest(config(), factorial + same_string, power_audit=audit)


def test_confirmatory_gate_requires_every_conservative_five_point_cell_to_reach_power(
    full_confirmatory_design,
):
    factorial, same_string = full_confirmatory_design
    audit = exhaustive_power_audit(factorial)
    index = next(index for index, cell in enumerate(audit.cells) if cell.interaction == 0.05)
    cells = list(audit.cells)
    cells[index] = replace(cells[index], estimated_power=0.799)

    with pytest.raises(ValueError, match="passing registered power audit"):
        build_manifest(
            config(),
            factorial + same_string,
            power_audit=replace(audit, cells=tuple(cells)),
        )
