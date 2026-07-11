from __future__ import annotations

import json
from pathlib import Path

from trajectory_extractor.artifacts import RunStore


def write_study_report(
    store: RunStore,
    output: str | Path,
    *,
    concept_run: str = "concept-main",
    transfer_run: str = "real-transfer",
    jailbreak_run: str = "jailbreak-main",
    concept_intervention_run: str = "concept-intervention",
    jailbreak_intervention_run: str = "jailbreak-intervention-test",
) -> Path:
    concept = _read(store, concept_run, "metrics/detection_exact_error.json")
    binding = _read(store, concept_run, "metrics/detection_binding_error.json")
    transfer = _read(store, transfer_run, "metrics/detection.json")
    jailbreak = _read(store, jailbreak_run, "metrics/grouped_detection.json")
    concept_intervention = _read(
        store, concept_intervention_run, "metrics/intervention.json"
    )
    jailbreak_intervention = _read(
        store, jailbreak_intervention_run, "metrics/intervention.json"
    )
    lines = [
        "# Feature-Dynamics Study Results",
        "",
        "This report is generated from frozen run artifacts. Missing stages are marked as not run.",
        "",
        "## Detection",
        "",
        _detection_line("Concept exact-error endpoint", concept),
        _detection_line("Concept distractor-binding endpoint", binding),
        _detection_line("Real-transfer endpoint", transfer),
        _jailbreak_line(jailbreak),
        "",
        "## Intervention",
        "",
        _intervention_line("Concept intervention", concept_intervention),
        _intervention_line("Jailbreak intervention", jailbreak_intervention),
        "",
        "## Audit Status",
        "",
        (
            "- Jailbreak Guard audit: completed."
            if (store.root / jailbreak_run / "labels" / "manual_audit_completed.json").exists()
            else "- Jailbreak Guard audit: not completed. Jailbreak conclusions are provisional."
        ),
        "",
        "## Interpretation Limits",
        "",
        "- Results are local to the recorded Llama 3.2 1B model revision.",
        "- Early warning refers to causal pre-token and layer prefixes, not an exact failure point.",
        "- PCA coordinates do not establish monosemanticity or reverse superposition.",
        "- Remizov/Chernoff is mathematical inspiration, not an LLM operator-semigroup theorem.",
    ]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _read(store: RunStore, run_id: str, relative: str):
    path = store.root / run_id / relative
    return json.loads(path.read_text()) if path.exists() else None


def _detection_line(label: str, result) -> str:
    if result is None:
        return f"- {label}: **not run**."
    delta = result["bootstrap_delta"]
    return (
        f"- {label}: **{result['decision']['outcome']}**; selected dynamics "
        f"`{result['selected_dynamics_method']}` vs `{result['selected_simple_baseline']}`, "
        f"AUROC delta {delta['delta']:.3f} (95% CI {delta['lower']:.3f}, {delta['upper']:.3f})."
    )


def _jailbreak_line(result) -> str:
    if result is None or result.get("aggregate_oof") is None:
        return "- Jailbreak grouped detection: **not run or not evaluable**."
    aggregate = result["aggregate_oof"]
    return (
        f"- Jailbreak grouped detection: **{aggregate['decision']['outcome']}**; "
        f"OOF AUROC delta {aggregate['delta']:.3f} "
        f"(95% CI {aggregate['lower']:.3f}, {aggregate['upper']:.3f})."
    )


def _intervention_line(label: str, result) -> str:
    if result is None:
        return f"- {label}: **not run**."
    comparison = result["vs_random"]
    return (
        f"- {label}: **{result['decision']['outcome']}**; relative failure reduction "
        f"{result['relative_reduction']:.1%}, control loss {result['control_loss_points']:.1%}, "
        f"delta vs random {comparison['delta_success_rate']:.3f} "
        f"(95% CI {comparison['lower']:.3f}, {comparison['upper']:.3f})."
    )
