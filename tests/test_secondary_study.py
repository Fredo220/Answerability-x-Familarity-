import numpy as np
import pytest

import trajectory_extractor.secondary_study as secondary_study
from trajectory_extractor.secondary_study import (
    benjamini_hochberg,
    causal_output_uncertainty,
    evaluate_concept_secondary,
    paired_entity_family_permutation_p,
    secondary_endpoint_status,
)
from trajectory_extractor.types import TrajectoryBatch


def make_batch() -> TrajectoryBatch:
    rng = np.random.default_rng(17)
    n_train, n_val, n_test = 40, 20, 40
    n_examples = n_train + n_val + n_test
    labels = np.arange(n_examples) % 2
    splits = np.array(
        ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    )
    hidden = rng.normal(0, 0.15, size=(n_examples, 2, 4, 6)).astype(np.float32)
    for layer in range(4):
        hidden[:, :, layer, 0] += labels[:, None] * (0.4 + 0.2 * layer)
        hidden[:, 1, layer, 1] += labels * (0.1 * layer)
    provenance = tuple(
        {
            "entity_family": (
                f"test-family-{index // 2}" if splits[index] == "test" else f"fit-{index // 2}"
            )
        }
        for index in range(n_examples)
    )
    return TrajectoryBatch(
        example_ids=tuple(f"e{index:03d}" for index in range(n_examples)),
        labels=labels.astype(np.int64),
        splits=splits,
        hidden_states=hidden,
        token_mask=np.ones((n_examples, 2), dtype=bool),
        token_logprobs=(-1.0 + 0.05 * rng.normal(size=(n_examples, 2))).astype(np.float32),
        token_entropies=(1.0 + 0.05 * rng.normal(size=(n_examples, 2))).astype(np.float32),
        provenance=provenance,
    )


def test_secondary_evaluation_keeps_registered_comparison_and_fit_ids(monkeypatch):
    batch = make_batch()
    permutation_call: dict[str, object] = {}
    real_permutation = secondary_study.paired_entity_family_permutation_p

    def capture_permutation(
        labels,
        candidate,
        baseline,
        *,
        groups,
        n_permutations,
        seed,
    ):
        permutation_call.update(
            groups=np.asarray(groups).copy(),
            n_permutations=n_permutations,
            seed=seed,
        )
        return real_permutation(
            labels,
            candidate,
            baseline,
            groups=groups,
            n_permutations=n_permutations,
            seed=seed,
        )

    monkeypatch.setattr(
        secondary_study,
        "paired_entity_family_permutation_p",
        capture_permutation,
    )

    result = evaluate_concept_secondary(
        batch,
        pca_dims=3,
        ridge_alpha=1e-3,
        n_bootstrap=50,
    )
    comparison = result["registered_comparison"]

    assert set(result["methods"]) == {
        "contrastive_vector",
        "contrastive_plus_dynamics",
        "full_metacognitive_monitor",
    }
    assert comparison["candidate"] == "contrastive_plus_dynamics"
    assert comparison["baseline"] == "contrastive_vector"
    assert comparison["fdr_family"] == [
        "detection_vector_dynamics",
        "intervention_capping_vs_triggered_pending",
    ]
    test = np.flatnonzero(batch.splits == "test")
    expected_groups = np.asarray(
        [batch.provenance[index]["entity_family"] for index in test]
    )
    np.testing.assert_array_equal(permutation_call["groups"], expected_groups)
    assert permutation_call["seed"] == 42
    assert permutation_call["n_permutations"] == 2_000
    assert comparison["permutation_seed"] == permutation_call["seed"]
    assert comparison["n_permutations"] == permutation_call["n_permutations"]
    expected_adjusted_p = float(
        benjamini_hochberg(np.array([comparison["raw_p"], 1.0]))[0]
    )
    assert comparison["bh_adjusted_p"] == pytest.approx(expected_adjusted_p)
    expected_supported = bool(
        result["endpoint_status"]["evaluable"]
        and comparison["delta_auroc"] >= comparison["minimum_effect"]
        and comparison["lower"] > 0.0
        and comparison["bh_adjusted_p"] < 0.05
    )
    assert comparison["supported"] is expected_supported
    expected_claim_status = (
        "not_evaluable"
        if not result["endpoint_status"]["evaluable"]
        else "provisional_supported" if expected_supported else "not_supported"
    )
    assert result["claim_status"] == expected_claim_status
    assert result["endpoint_status"]["evaluable"] is True
    assert result["endpoint_status"]["positive_examples"] == 20
    assert result["endpoint_status"]["positive_clusters"] == 20
    train = np.flatnonzero(batch.splits == "train")
    stable_train = train[batch.labels[train] == 0]
    assert result["fit_example_ids"]["direction"] == [batch.example_ids[index] for index in train]
    assert result["fit_example_ids"]["vector_standardization"] == [
        batch.example_ids[index] for index in train
    ]
    assert result["fit_example_ids"]["operator"] == [
        batch.example_ids[index] for index in stable_train
    ]
    assert result["artifacts"]["metacognitive_risk_probability"].shape == (40,)
    assert "metacognitive_risk_surface" not in result["artifacts"]
    np.testing.assert_array_equal(result["artifacts"]["validation_indices"], np.arange(40, 60))
    np.testing.assert_array_equal(result["artifacts"]["validation_labels"], batch.labels[40:60])
    assert result["artifacts"]["validation_metacognitive_risk_surface"].shape == (20, 2, 4)
    assert result["artifacts"]["directions"].shape == (4, 6)
    assert result["methods"]["full_metacognitive_monitor"]["metadata"]["analysis_role"] == "exploratory"
    assert result["methods"]["contrastive_plus_dynamics"]["metadata"]["analysis_role"] == "confirmatory"
    assert (
        result["methods"]["contrastive_plus_dynamics"]["test_false_positive_rate"]
        == result["methods"]["contrastive_plus_dynamics"]["test"]["false_positive_rate"]
    )
    assert "validation_diagnostics" in result["methods"]["contrastive_plus_dynamics"]
    assert "median_positive_crossing" not in result["methods"]["contrastive_plus_dynamics"]
    for value in result["artifacts"].values():
        if np.issubdtype(np.asarray(value).dtype, np.floating):
            assert np.isfinite(value).all()


def test_secondary_evaluation_only_materializes_validation_surface(monkeypatch):
    batch = make_batch()
    validation = np.flatnonzero(batch.splits == "val")
    surface_calls: list[np.ndarray] = []
    original_evaluate = secondary_study.evaluate_prefix_surface
    original_predict_surface = secondary_study.predict_prefix_probability_surface

    def evaluate_validation_surface(*args, **kwargs):
        surface_calls.append(kwargs["test_indices"])
        return original_evaluate(*args, **kwargs)

    def predict_validation_surface(*args, **kwargs):
        surface_calls.append(kwargs["predict_indices"])
        return original_predict_surface(*args, **kwargs)

    monkeypatch.setattr(secondary_study, "evaluate_prefix_surface", evaluate_validation_surface)
    monkeypatch.setattr(
        secondary_study,
        "predict_prefix_probability_surface",
        predict_validation_surface,
    )

    evaluate_concept_secondary(batch, pca_dims=3, n_bootstrap=10)

    assert surface_calls
    assert all(np.array_equal(indices, validation) for indices in surface_calls)


def test_endpoint_is_not_evaluable_when_positive_cluster_count_is_too_small():
    batch = make_batch()
    provenance = tuple({"entity_family": "one-family"} for _ in batch.provenance)
    reduced = TrajectoryBatch(
        example_ids=batch.example_ids,
        labels=batch.labels,
        splits=batch.splits,
        hidden_states=batch.hidden_states,
        token_mask=batch.token_mask,
        token_logprobs=batch.token_logprobs,
        token_entropies=batch.token_entropies,
        provenance=provenance,
    )
    test = np.flatnonzero(reduced.splits == "test")

    status = secondary_endpoint_status(reduced, test)

    assert status["evaluable"] is False
    assert status["positive_examples"] == 20
    assert status["positive_clusters"] == 1
    assert "positive_clusters<10" in status["reasons"]


def test_benjamini_hochberg_preserves_order_and_monotonicity():
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.20]))

    np.testing.assert_allclose(adjusted, np.array([0.03, 0.06, 0.20]))


def test_cluster_permutation_p_value_swaps_entity_families_together():
    labels = np.array([0, 1, 0, 1])
    candidate = np.array([0.1, 0.9, 0.2, 0.8])
    baseline = np.array([0.8, 0.2, 0.7, 0.3])
    groups = np.array(["family-a", "family-a", "family-b", "family-b"])

    p_value = paired_entity_family_permutation_p(
        labels,
        candidate,
        baseline,
        groups=groups,
        n_permutations=8,
    )

    assert p_value == pytest.approx(5 / 9)


def test_secondary_evaluation_rejects_test_example_without_entity_family():
    batch = make_batch()
    provenance = list(batch.provenance)
    provenance[-1] = {"entity_family": ""}
    missing_family = TrajectoryBatch(
        example_ids=batch.example_ids,
        labels=batch.labels,
        splits=batch.splits,
        hidden_states=batch.hidden_states,
        token_mask=batch.token_mask,
        token_logprobs=batch.token_logprobs,
        token_entropies=batch.token_entropies,
        provenance=tuple(provenance),
    )

    with pytest.raises(ValueError, match="test example.*entity_family"):
        evaluate_concept_secondary(missing_family, pca_dims=3, n_bootstrap=10)


def test_output_uncertainty_is_shifted_to_prior_tokens():
    batch = make_batch()

    logprobs, entropies = causal_output_uncertainty(batch)

    assert np.all(logprobs[:, 0] == 0.0)
    assert np.all(entropies[:, 0] == 0.0)
    np.testing.assert_allclose(logprobs[:, 1], batch.token_logprobs[:, 0])
    np.testing.assert_allclose(entropies[:, 1], batch.token_entropies[:, 0])
