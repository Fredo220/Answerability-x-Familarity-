from __future__ import annotations

import os
from pathlib import Path

_CACHE_ROOT = Path.cwd() / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_prefix_surface(surface: np.ndarray, output: str | Path, *, title: str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(surface, origin="lower", aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
    axis.set_xlabel("Layer prefix")
    axis.set_ylabel("Answer-token prefix")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="AUROC")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_method_comparison(metrics: dict[str, float], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    names = list(metrics)
    axis.bar(names, [metrics[name] for name in names], color="#287271")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Test AUROC")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_class_risk_gap(
    probabilities: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    values = np.asarray(probabilities, dtype=float)
    target = np.asarray(labels, dtype=int)
    if values.ndim != 3 or values.shape[0] != target.size:
        raise ValueError("probabilities must have shape [example, token, layer]")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("risk-gap plot requires both classes")
    gap = np.median(values[target == 1], axis=0) - np.median(values[target == 0], axis=0)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(gap, aspect="auto", origin="lower", cmap="coolwarm")
    axis.set_xlabel("Layer prefix")
    axis.set_ylabel("Answer-token prefix")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Median risk gap")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path
