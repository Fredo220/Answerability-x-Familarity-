import numpy as np

from trajectory_extractor.secondary_study import (
    benjamini_hochberg,
    causal_output_uncertainty,
    evaluate_concept_secondary,
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


def test_secondary_evaluation_keeps_registered_comparison_and_fit_ids():
    batch = make_batch()

    result = evaluate_concept_secondary(
        batch,
        pca_dims=3,
        ridge_alpha=1e-3,
        n_bootstrap=50,
    )

    assert set(result["methods"]) == {
        "contrastive_vector",
        "contrastive_plus_dynamics",
        "full_metacognitive_monitor",
    }
    assert result["registered_comparison"]["candidate"] == "contrastive_plus_dynamics"
    assert result["registered_comparison"]["baseline"] == "contrastive_vector"
    assert result["endpoint_status"]["evaluable"] is True
    assert result["endpoint_status"]["positive_examples"] == 20
    assert result["endpoint_status"]["positive_clusters"] == 20
    assert all(identifier.startswith("e0") for identifier in result["fit_example_ids"]["direction"])
    assert all(
        batch.labels[batch.example_ids.index(identifier)] == 0
        for identifier in result["fit_example_ids"]["operator"]
    )
    assert result["artifacts"]["metacognitive_risk_probability"].shape == (40,)
    assert result["artifacts"]["metacognitive_risk_surface"].shape == (40, 2, 4)
    assert result["artifacts"]["directions"].shape == (4, 6)


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


def test_output_uncertainty_is_shifted_to_prior_tokens():
    batch = make_batch()

    logprobs, entropies = causal_output_uncertainty(batch)

    assert np.all(logprobs[:, 0] == 0.0)
    assert np.all(entropies[:, 0] == 0.0)
    np.testing.assert_allclose(logprobs[:, 1], batch.token_logprobs[:, 0])
    np.testing.assert_allclose(entropies[:, 1], batch.token_entropies[:, 0])
