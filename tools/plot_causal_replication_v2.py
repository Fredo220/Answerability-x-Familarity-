"""Plot the published causal-replication v2 primary and control effects."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "release/familiarity_answerability/answerability_causal_replication_v2"
    / "results/result.json"
)
OUTPUT = ROOT / "docs/assets/familiarity_answerability_causal_v2.png"

SPLITS = ("causal_entity_test", "causal_template_test")
SPLIT_LABELS = ("Unseen entities", "Unseen templates")
SERIES = (
    ("primary", "Primary"),
    ("wrong_layer", "Wrong layer"),
    ("wrong_anchor", "Wrong anchor"),
    ("label_shuffled_direction", "Shuffled labels"),
    ("norm_matched_random", "Random mean"),
    ("sign_reversed", "Sign reversed"),
)


def load_result() -> dict:
    with RESULT.open(encoding="utf-8") as handle:
        return json.load(handle)


def effect(result: dict, split: str, series: str) -> float:
    decision = result["split_decisions"][split]
    if series == "primary":
        return float(decision["mean_effect"])
    return float(decision["control_effects"][series]["mean_effect"])


def main() -> None:
    result = load_result()
    values = np.array(
        [[effect(result, split, series) for series, _ in SERIES] for split in SPLITS]
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )

    figure, axis = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    figure.patch.set_facecolor("white")
    y = np.arange(len(SERIES))
    height = 0.34
    colors = ("#0F766E", "#B45309")
    hatches = (None, "//")

    for index, (label, color, hatch) in enumerate(
        zip(SPLIT_LABELS, colors, hatches, strict=True)
    ):
        bars = axis.barh(
            y + (index - 0.5) * height,
            values[index],
            height,
            color=color,
            edgecolor="#334155" if hatch else color,
            linewidth=0.7,
            hatch=hatch,
            label=label,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:+.3f}" for value in values[index]],
            padding=4,
            fontsize=9,
        )

    axis.axvline(0, color="#475569", linewidth=1)
    axis.set_yticks(y, [label for _, label in SERIES])
    axis.invert_yaxis()
    axis.set_xlim(-0.42, 0.43)
    axis.set_xlabel("Mean bidirectional code-vs-UNKNOWN margin effect")
    axis.set_title("Causal replication v2: primary effect and controls", fontsize=15)
    axis.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    axis.text(
        0.0,
        -0.15,
        "Overall preregistered decision: not supported. The unseen-entity "
        "split failed because the wrong-layer control was stronger than the primary effect.",
        transform=axis.transAxes,
        color="#64748B",
        fontsize=9,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
