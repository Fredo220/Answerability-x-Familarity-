"""Generate the README summary figure from published result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_RESULT = (
    ROOT / "docs/results/same_string_feasibility_v2_behavior_result.json"
)
REPRESENTATION_RESULT = (
    ROOT
    / "release/familiarity_answerability/representation_replication_v3"
    / "analysis/result.json"
)
OUTPUT = ROOT / "docs/assets/familiarity_answerability_summary.png"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    behavior = load_json(BEHAVIOR_RESULT)
    representation = load_json(REPRESENTATION_RESULT)

    cells = behavior["cell_metrics"]
    high = np.array(
        [
            cells["high_exposure_code_absent"]["attempt_rate"],
            cells["high_exposure_target_bound"]["attempt_rate"],
        ]
    )
    low = np.array(
        [
            cells["low_exposure_code_absent"]["attempt_rate"],
            cells["low_exposure_target_bound"]["attempt_rate"],
        ]
    )

    primary = {row["test_split"]: row for row in representation["primary"]}
    auroc_gain = np.array(
        [
            primary["entity_test"]["mean_auroc_improvement"],
            primary["template_test"]["mean_auroc_improvement"],
        ]
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

    figure, (behavior_ax, representation_ax) = plt.subplots(
        1, 2, figsize=(11.5, 4.4), constrained_layout=True
    )
    figure.patch.set_facecolor("white")

    x = np.arange(2)
    width = 0.34
    high_bars = behavior_ax.bar(
        x - width / 2,
        high * 100,
        width,
        color="#0F766E",
        label="High exposure",
    )
    low_bars = behavior_ax.bar(
        x + width / 2,
        low * 100,
        width,
        color="#475569",
        label="Low exposure",
    )
    behavior_ax.set_title("Behavioral pilot", fontsize=14, pad=12)
    behavior_ax.text(
        0.98,
        0.96,
        "Registered interaction:\nnot supported",
        transform=behavior_ax.transAxes,
        ha="right",
        va="top",
        color="#64748B",
        fontsize=9,
    )
    behavior_ax.set_xticks(x, ["Evidence absent", "Evidence present"])
    behavior_ax.set_ylabel("Answer-attempt rate (%)")
    behavior_ax.set_ylim(0, 108)
    behavior_ax.legend(frameon=False, loc="upper left")
    behavior_ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    behavior_ax.set_axisbelow(True)
    behavior_ax.spines[["top", "right"]].set_visible(False)
    behavior_ax.bar_label(high_bars, fmt="%.1f%%", padding=3, fontsize=9)
    behavior_ax.bar_label(low_bars, fmt="%.1f%%", padding=3, fontsize=9)

    representation_bars = representation_ax.bar(
        ["Unseen entities", "Unseen templates"],
        auroc_gain,
        width=0.58,
        color=["#B45309", "#D97706"],
    )
    representation_ax.axhline(0, color="#64748B", linewidth=0.9)
    representation_ax.set_title("Representation replication", fontsize=14, pad=12)
    representation_ax.set_ylabel("Mean AUROC improvement")
    representation_ax.set_ylim(0, 0.48)
    representation_ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    representation_ax.set_axisbelow(True)
    representation_ax.spines[["top", "right"]].set_visible(False)
    representation_ax.bar_label(
        representation_bars,
        labels=[f"+{value:.3f}" for value in auroc_gain],
        padding=4,
        fontsize=10,
        fontweight="bold",
    )
    representation_ax.text(
        0.5,
        0.025,
        "Permutation p = 0.001 for both splits",
        transform=representation_ax.transAxes,
        ha="center",
        color="#64748B",
        fontsize=9,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
