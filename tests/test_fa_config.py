import json
from pathlib import Path

import pytest

from trajectory_extractor.fa_config import (
    CONFIRMATORY_SPLIT_COUNTS,
    NON_CONFIRMATORY_NAMESPACES,
    FAConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_CONFIG = REPO_ROOT / "configs" / "familiarity_answerability_gemma2_2b.json"
SMOKE_CONFIG = REPO_ROOT / "configs" / "familiarity_answerability_qwen06b_smoke.json"
ACTIVE_SMOKE_CONFIG = (
    REPO_ROOT / "configs" / "familiarity_answerability_qwen17b_smoke.json"
)
SOURCE_PINS = REPO_ROOT / "data" / "fa" / "source_pins.json"
PROTOCOL_AMENDMENT = (
    REPO_ROOT / "docs" / "familiarity_answerability_protocol_amendment_2026-07-22.md"
)


def test_confirmatory_config_is_canonical_and_pinned():
    config = FAConfig.from_json(CONFIRMATORY_CONFIG)

    assert config.profile == "confirmatory"
    assert config.model_id == "google/gemma-2-2b-it"
    assert config.model_revision == "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
    assert config.tokenizer_revision == "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
    assert (
        config.chat_template_sha256
        == "ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6"
    )
    assert dict(config.split_counts) == CONFIRMATORY_SPLIT_COUNTS
    assert config.split_seed == 20260722
    assert config.bootstrap_replicates == 10000
    assert config.bootstrap_seed == 20260722
    assert config.anchors == (
        "target_intro_end",
        "user_prompt_end",
        "assistant_prefix_end",
    )
    assert dict(config.generation) == {
        "do_sample": False,
        "max_new_tokens": 16,
        "temperature": 0.0,
    }
    assert dict(config.thresholds) == {
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
    }
    assert set(config.split_counts).isdisjoint(NON_CONFIRMATORY_NAMESPACES)
    assert len(config.config_hash) == 64
    assert config.canonical_bytes == json.dumps(
        json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8")),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_same_string_confirmatory_config_reuses_registered_model_and_counts():
    config = FAConfig.from_json(
        "configs/familiarity_answerability_same_string_gemma2_2b.json"
    )
    assert config.study_id == "familiarity-answerability-same-string-gemma2-2b-v1"
    assert config.run_id == "same-string-primary-v1"
    assert config.model_revision == "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
    assert config.split_counts["behavior_test"] == 48
    assert config.thresholds["h1_min_interaction"] == 0.05


def test_config_rejects_mutable_revision_and_unknown_split(tmp_path):
    payload = json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8"))
    payload["model_revision"] = "main"
    path = tmp_path / "mutable.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable revision"):
        FAConfig.from_json(path)

    payload = json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8"))
    payload["split_counts"]["unregistered"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="registered split"):
        FAConfig.from_json(path)


def test_direct_construction_rejects_invalid_confirmatory_pin():
    payload = json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8"))
    payload["model_revision"] = "a" * 40

    with pytest.raises(ValueError, match="official pin"):
        FAConfig(**payload)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_config_rejects_schema_version_alias_types(tmp_path, schema_version):
    payload = json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version

    with pytest.raises(ValueError, match="schema_version"):
        FAConfig(**payload)

    path = tmp_path / "schema-version-alias.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        FAConfig.from_json(path)


def test_direct_construction_rejects_non_exact_generation_types():
    class IntAlias(int):
        pass

    payload = json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8"))
    payload["generation"]["max_new_tokens"] = IntAlias(16)

    with pytest.raises(ValueError, match="generation"):
        FAConfig(**payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "a" * 40),
        ("tokenizer_revision", "a" * 40),
        ("chat_template_sha256", "a" * 64),
        ("split_seed", 1),
        ("bootstrap_replicates", 1),
        ("bootstrap_seed", 1),
        ("anchors", ["user_prompt_end", "target_intro_end", "assistant_prefix_end"]),
        ("generation", {"do_sample": False, "max_new_tokens": 17, "temperature": 0.0}),
    ],
)
def test_confirmatory_config_rejects_mutated_registered_values(tmp_path, field, value):
    payload = json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="confirmatory"):
        FAConfig.from_json(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("do_sample", 0),
        ("max_new_tokens", True),
        ("temperature", False),
    ],
)
def test_confirmatory_config_rejects_generation_alias_types(tmp_path, field, value):
    payload = json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8"))
    payload["generation"][field] = value
    path = tmp_path / "generation-alias.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="generation"):
        FAConfig.from_json(path)


@pytest.mark.parametrize("field", ["model_revision", "tokenizer_revision"])
def test_smoke_config_rejects_unregistered_qwen_revision(tmp_path, field):
    payload = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    payload[field] = "a" * 40
    path = tmp_path / "smoke-revision.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="smoke .*revision"):
        FAConfig.from_json(path)


@pytest.mark.parametrize("field", ["split_counts", "generation", "thresholds"])
def test_config_mapping_fields_are_structurally_immutable(field):
    config = FAConfig.from_json(CONFIRMATORY_CONFIG)

    with pytest.raises(TypeError):
        getattr(config, field)["changed"] = "value"


def test_config_rejects_nonfinite_threshold_and_confirmatory_qwen(tmp_path):
    payload = json.loads(CONFIRMATORY_CONFIG.read_text(encoding="utf-8"))
    payload["thresholds"]["h1_min_interaction"] = float("nan")
    path = tmp_path / "nonfinite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        FAConfig.from_json(path)

    payload = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    payload["profile"] = "confirmatory"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Qwen"):
        FAConfig.from_json(path)


def test_smoke_config_uses_only_nonconfirmatory_namespaces():
    config = FAConfig.from_json(SMOKE_CONFIG)

    assert config.profile == "smoke"
    assert set(config.split_counts) == NON_CONFIRMATORY_NAMESPACES
    assert config.thresholds == {}
    assert config.anchors == ()


def test_active_smoke_config_is_pinned_to_qwen17b():
    config = FAConfig.from_json(ACTIVE_SMOKE_CONFIG)

    assert config.model_id == "Qwen/Qwen3-1.7B"
    assert config.model_revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert config.tokenizer_revision == config.model_revision
    assert (
        config.chat_template_sha256
        == "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    )
    assert config.run_id == "smoke-qwen17b-v1"


def test_active_smoke_config_rejects_unregistered_template_hash(tmp_path):
    payload = json.loads(ACTIVE_SMOKE_CONFIG.read_text(encoding="utf-8"))
    payload["chat_template_sha256"] = "a" * 64
    path = tmp_path / "active-smoke-template.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="active smoke chat_template"):
        FAConfig.from_json(path)


def test_source_pins_use_the_official_gemma_scope_repository_and_revision():
    pins = json.loads(SOURCE_PINS.read_text(encoding="utf-8"))

    assert pins["gemma_scope"] == {
        "repository": "google/gemma-scope-2b-pt-res",
        "revision": "fd571b47c1c64851e9b1989792367b9babb4af63",
    }
    assert pins["qwen_active_smoke_model_and_tokenizer"] == {
        "repository": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "chat_template_sha256": (
            "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
        ),
    }


def test_pre_outcome_amendment_freezes_the_exact_output_control_coordinates():
    amendment = PROTOCOL_AMENDMENT.read_text(encoding="utf-8")
    coordinates = (
        "target sequence log-likelihood",
        "UNKNOWN sequence log-likelihood",
        "target-minus-UNKNOWN log-likelihood",
        "maximum registered sequence log-likelihood",
        "registered-sequence log-sum-exp",
        "normalized target probability",
        "normalized UNKNOWN probability",
        "binary target-versus-UNKNOWN entropy",
        "absolute normalized probability margin",
        "signed normalized probability margin",
        "maximum normalized registered-sequence probability",
    )

    positions = [amendment.index(f"{index}. {name}") for index, name in enumerate(coordinates, 1)]
    assert positions == sorted(positions)
    assert "teacher-forced" in amendment
    assert "no generated completion" in amendment


def test_pre_outcome_amendment_freezes_f2a_selection_and_null_contract():
    amendment = PROTOCOL_AMENDMENT.read_text(encoding="utf-8")

    required_contract = (
        "Grouped cross-validation entirely within `mechanism_train` selects PCA dimensionality and logistic `C`",
        "The same four registered domains occur in every confirmatory split",
        "each of the three answerability states in turn",
        "each target-familiarity condition in turn",
        "exactly 10,000 bootstrap replicates",
        "2026072201` through `2026072299",
        "7aee4f4ee03201f4a8b7bee296294bc5c6a14a5251dfa71bb8cff15ce3d4e07f",
        "adjacent difference is `delta_l = h_l - h_(l-1)`",
        "`1 - cosine(delta_(l-1), delta_l)`",
        "exclude transformer layer 25 and `assistant_prefix_end`",
    )

    for clause in required_contract:
        assert clause in amendment
