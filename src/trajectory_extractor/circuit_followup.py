from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trajectory_extractor.artifacts import RunStore


STRATA = ("true_positive", "false_positive", "false_negative", "true_negative")


def select_circuit_followup_cases(
    prediction_rows: list[dict],
    *,
    threshold: float,
    per_stratum: int = 5,
) -> list[dict]:
    """Select archetypal held-out cases without manual cherry-picking."""
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    if len({str(row["example_id"]) for row in prediction_rows}) != len(prediction_rows):
        raise ValueError("prediction rows must have unique example IDs")
    grouped = {name: [] for name in STRATA}
    for row in prediction_rows:
        label = int(row["label"])
        score = float(row["selected_dynamics_score"])
        if label not in (0, 1) or not np.isfinite(score):
            raise ValueError("labels must be binary and dynamics scores finite")
        predicted = int(score >= threshold)
        stratum = {
            (1, 1): "true_positive",
            (0, 1): "false_positive",
            (1, 0): "false_negative",
            (0, 0): "true_negative",
        }[(label, predicted)]
        grouped[stratum].append(
            {
                **row,
                "prediction": predicted,
                "stratum": stratum,
                "threshold_margin": abs(score - threshold),
            }
        )
    selected = []
    for stratum in STRATA:
        ranked = sorted(
            grouped[stratum],
            key=lambda row: (-float(row["threshold_margin"]), str(row["example_id"])),
        )
        selected.extend(ranked[:per_stratum])
    return selected


def prepare_circuit_followup(
    store: RunStore,
    *,
    run_id: str,
    endpoint: str = "exact_error",
    per_stratum: int = 5,
) -> dict:
    metrics_path = store.root / run_id / "metrics" / f"detection_{endpoint}.json"
    predictions_path = store.root / run_id / "labels" / f"detection_predictions_{endpoint}.json"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("Run detection evaluation before preparing circuit follow-up cases")
    metrics = json.loads(metrics_path.read_text())
    predictions = json.loads(predictions_path.read_text())
    method = str(metrics["selected_dynamics_method"])
    threshold = float(metrics["methods"][method]["test"]["threshold"])
    selected = select_circuit_followup_cases(
        predictions, threshold=threshold, per_stratum=per_stratum
    )
    cases = []
    for row in selected:
        run = store.read(run_id, str(row["example_id"]))
        cases.append(
            {
                **row,
                "prompt": run.prompt,
                "response": run.response,
                "expected_answer": run.provenance.get("expected_answer"),
                "distractor_answers": run.provenance.get("distractor_answers", []),
                "concept_outcome": run.provenance.get("concept_outcome"),
                "entity_family": run.provenance.get("entity_family"),
                "template_group": run.provenance.get("template_group"),
            }
        )
    summary = {
        "status": "exploratory_mechanistic_followup",
        "primary_endpoint_impact": "none",
        "source_run_id": run_id,
        "endpoint": endpoint,
        "selected_method": method,
        "frozen_threshold": threshold,
        "per_stratum": per_stratum,
        "selection_rule": "largest absolute frozen-threshold margin, example_id tie-break",
        "published_reference_model": "meta-llama/Llama-3.2-1B",
        "primary_study_model": "meta-llama/Llama-3.2-1B-Instruct",
        "checkpoint_compatibility": (
            "Published Llama transcoders target the base checkpoint. Results on the base model are "
            "an external mechanistic replication, not a direct explanation of the Instruct checkpoint."
        ),
        "counts": {
            stratum: sum(case["stratum"] == stratum for case in cases) for stratum in STRATA
        },
        "cases": cases,
    }
    store.write_json(run_id, "labels", f"circuit_followup_{endpoint}", summary)
    return summary
