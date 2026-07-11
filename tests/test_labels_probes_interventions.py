import numpy as np
import torch

from trajectory_extractor.interventions import InterventionArm, build_intervention_plans
from trajectory_extractor.labels import (
    ConceptOutcome,
    classify_concept_response,
    concept_error_label,
    is_refusal,
    safety_rates,
)
from trajectory_extractor.probes import LayerwiseStaticProbe
from trajectory_extractor.types import TrajectoryBatch


def make_batch():
    rng = np.random.default_rng(7)
    return TrajectoryBatch(
        example_ids=("a", "b", "c", "d"),
        labels=np.array([0, 1, 0, 1]),
        splits=np.array(["train", "train", "test", "test"]),
        hidden_states=rng.normal(size=(4, 2, 3, 4)).astype(np.float32),
        token_mask=np.ones((4, 2), dtype=bool),
        token_logprobs=np.zeros((4, 2), dtype=np.float32),
        token_entropies=np.zeros((4, 2), dtype=np.float32),
    )


def test_static_probe_scores_each_causal_token_and_fits_train_only():
    batch = make_batch()
    probe = LayerwiseStaticProbe().fit(batch, np.array([0, 1]))
    scores = probe.predict_scores(batch)
    assert probe.fit_example_ids == ("a", "b")
    assert scores.shape == (4, 2, 3)
    assert np.isfinite(scores).all()


def test_concept_label_normalizes_case_and_punctuation():
    assert concept_error_label(" Zephyroscope. ", "zephyroscope") == 0
    assert concept_error_label("Valtirium", "Zephyroscope") == 1


def test_concept_decision_separates_binding_from_other_errors():
    binding = classify_concept_response("Valtirium", "Zephyroscope", ("Valtirium",))
    formatting = classify_concept_response("The Zephyroscope", "Zephyroscope", ("Valtirium",))
    refusal = classify_concept_response("I cannot assist.", "Zephyroscope", ("Valtirium",))
    assert binding.outcome == ConceptOutcome.DISTRACTOR_BINDING
    assert binding.binding_error == 1
    assert formatting.outcome == ConceptOutcome.FORMAT_ONLY
    assert formatting.binding_error == 0
    assert refusal.outcome == ConceptOutcome.REFUSAL
    assert is_refusal("I can't assist with that.")


def test_safety_rates_keep_harmful_and_benign_controls_separate():
    rates = safety_rates(
        unsafe=[True, False, False, False],
        refused=[False, True, True, False],
        benign=[False, False, True, True],
    )
    assert rates.unsafe_response_rate == 0.5
    assert rates.refusal_rate == 0.5
    assert rates.benign_over_refusal_rate == 0.5


def test_all_preregistered_intervention_arms_are_built():
    positive = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
    negative = torch.tensor([[0.0, 2.0], [0.0, 1.0]])
    plans = build_intervention_plans(positive, negative, strength=1.5, threshold=0.4)
    assert {plan.arm for plan in plans} == set(InterventionArm)
    for plan in plans[1:]:
        assert torch.isclose(torch.linalg.vector_norm(plan.direction), torch.tensor(1.0))
