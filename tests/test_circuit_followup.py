import json

import numpy as np

from trajectory_extractor.artifacts import RunStore
from trajectory_extractor.circuit_followup import (
    prepare_circuit_followup,
    select_circuit_followup_cases,
)
from trajectory_extractor.types import ActivationRun


def test_circuit_followup_selection_is_balanced_and_deterministic():
    rows = [
        {"example_id": "tp", "label": 1, "selected_dynamics_score": 0.9},
        {"example_id": "fp", "label": 0, "selected_dynamics_score": 0.8},
        {"example_id": "fn", "label": 1, "selected_dynamics_score": 0.2},
        {"example_id": "tn", "label": 0, "selected_dynamics_score": 0.1},
    ]
    selected = select_circuit_followup_cases(rows, threshold=0.5, per_stratum=1)
    assert [row["stratum"] for row in selected] == [
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]


def test_prepare_circuit_followup_enriches_frozen_test_cases(tmp_path):
    store = RunStore(tmp_path)
    run_id = "concept"
    store.write_manifest(run_id, {"track": "concept_mixing"})
    for example_id, label, score in (("tp", 1, 0.9), ("tn", 0, 0.1)):
        store.write(
            ActivationRun(
                run_id=run_id,
                example_id=example_id,
                track="concept_mixing",
                split="test",
                prompt=f"prompt {example_id}",
                response="answer",
                label=label,
                input_token_count=2,
                response_token_ids=np.array([3]),
                hidden_states=np.zeros((1, 2, 4), dtype=np.float16),
                token_logprobs=np.array([-1.0], dtype=np.float32),
                token_entropies=np.array([1.0], dtype=np.float32),
                provenance={
                    "response_token_start": 2,
                    "response_token_end": 3,
                    "expected_answer": "answer",
                    "entity_family": "family",
                },
            )
        )
    store.write_json(
        run_id,
        "metrics",
        "detection_exact_error",
        {
            "selected_dynamics_method": "combined",
            "methods": {"combined": {"test": {"threshold": 0.5}}},
        },
    )
    store.write_json(
        run_id,
        "labels",
        "detection_predictions_exact_error",
        [
            {"example_id": "tp", "label": 1, "selected_dynamics_score": 0.9},
            {"example_id": "tn", "label": 0, "selected_dynamics_score": 0.1},
        ],
    )
    result = prepare_circuit_followup(store, run_id=run_id, per_stratum=1)
    assert result["primary_endpoint_impact"] == "none"
    assert {case["example_id"] for case in result["cases"]} == {"tp", "tn"}
    saved = json.loads(
        (tmp_path / run_id / "labels" / "circuit_followup_exact_error.json").read_text()
    )
    assert saved["published_reference_model"] == "meta-llama/Llama-3.2-1B"
