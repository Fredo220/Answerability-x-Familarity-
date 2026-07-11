from types import SimpleNamespace

import torch

from trajectory_extractor.artifacts import RunStore
from trajectory_extractor.datasets.concept_mixing import ConceptMixingExample
from trajectory_extractor.intervention_study import _generate_concept_errors
from trajectory_extractor.interventions import InterventionArm, InterventionPlan
from trajectory_extractor.types import ExperimentConfig


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, text, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "Correct Object"


class FakeModel:
    device = torch.device("cpu")
    config = SimpleNamespace(_commit_hash="fake-revision")

    def __init__(self):
        self.generate_calls = 0

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls += 1
        return torch.tensor([[1, 2, 3, 4]])


def test_concept_tuning_candidate_is_persisted_and_resumed(tmp_path):
    example = ConceptMixingExample(
        example_id="example",
        split="val",
        entity_family="family",
        template_group="template",
        context="context",
        question="question",
        prompt="prompt",
        answer="Correct Object",
        relation="invented",
        distractor_count=1,
        name_similarity="high",
        answer_position=0,
        target_entity="entity",
        distractor_answers=("Wrong Object",),
        entity_rarity="synthetic",
    )
    model = FakeModel()
    store = RunStore(tmp_path)
    kwargs = dict(
        model=model,
        tokenizer=FakeTokenizer(),
        examples=[example],
        config=ExperimentConfig(model_id="fake/model", max_new_tokens=1),
        run_id="tuning",
        store=store,
        plan=InterventionPlan(InterventionArm.NONE, None, 0.0, None),
        layer=1,
        operator_model=None,
        candidate="candidate-1",
    )
    first = _generate_concept_errors(**kwargs)
    second = _generate_concept_errors(**kwargs)
    assert first.tolist() == [0]
    assert second.tolist() == [0]
    assert model.generate_calls == 1
    assert store.response_ids("tuning") == ["candidate-1__none__example"]
