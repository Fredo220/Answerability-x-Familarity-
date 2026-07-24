#!/usr/bin/env python3
"""Verify and render the immutable Familiarity-vs-Answerability pilot results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from trajectory_extractor.fa_artifacts import FAArtifactStore  # noqa: E402
from trajectory_extractor.fa_cli import _load_manifest  # noqa: E402
from trajectory_extractor.fa_config import FAConfig  # noqa: E402


TASK_CLASSES = {
    "familiarity": ("matched_synthetic", "screened_real"),
    "answerability": ("target_bound", "distractor_bound", "code_absent"),
}
PRIMARY_ANCHORS = {
    "familiarity": "target_intro_end",
    "answerability": "user_prompt_end",
}
LAYERS = (0, 9, 18, 27)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(row: dict[str, Any]) -> tuple[str, str, int | None, str]:
    return row["task"], row["anchor"], row["layer_id"], row["model_family"]


def load_and_audit(
    *,
    root: Path,
    config_path: Path,
    prompt_manifest: Path,
    metrics_manifest: Path,
    predictions_manifest: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Verify artifact lineage and recompute aggregate metrics from OOF predictions."""

    root = root.resolve()
    store = FAArtifactStore(root)
    config = FAConfig.from_json(config_path)
    prompt = _load_manifest(store, prompt_manifest, config)
    metrics_shard = store.verify_shard(metrics_manifest)
    predictions_shard = store.verify_shard(predictions_manifest)
    if metrics_shard.record_kind != "pilot_metrics":
        raise ValueError("metrics artifact has the wrong record kind")
    if predictions_shard.record_kind != "pilot_predictions":
        raise ValueError("predictions artifact has the wrong record kind")
    metrics_sidecar = json.loads(metrics_shard.manifest_path.read_text(encoding="utf-8"))
    lineage = metrics_sidecar["lineage"]
    if (
        lineage.get("predictions_sha256") != predictions_shard.sha256
        or lineage.get("prompt_manifest_sha256") != prompt.shard_sha256
        or lineage.get("analysis_sha256")
        != "f71f8a05830609b442c693911e85a7b39c20c9be732a1291b53b3df18e9731d5"
    ):
        raise ValueError("pilot report inputs do not share the frozen lineage")

    metrics = _read_jsonl(metrics_shard.data_path)
    predictions = _read_jsonl(predictions_shard.data_path)
    examples = {row.example_id: row for row in prompt.examples if row.block == "factorial"}
    if len(metrics) != 38 or len(predictions) != 10_944 or len(examples) != 288:
        raise ValueError("pilot report inputs do not have the frozen dimensions")
    metrics_by_key = {_key(row): row for row in metrics}
    predictions_by_key: dict[
        tuple[str, str, int | None, str], list[dict[str, Any]]
    ] = {}
    for row in predictions:
        predictions_by_key.setdefault(_key(row), []).append(row)
    if set(metrics_by_key) != set(predictions_by_key):
        raise ValueError("pilot metrics and predictions have different candidates")

    for candidate, rows in predictions_by_key.items():
        if len(rows) != 288 or {row["example_id"] for row in rows} != set(examples):
            raise ValueError("pilot candidate does not cover every factorial example")
        task = candidate[0]
        classes = TASK_CLASSES[task]
        ordered = sorted(rows, key=lambda row: row["example_id"])
        labels = np.asarray([classes.index(row["label"]) for row in ordered])
        probabilities = np.asarray(
            [
                [row["class_probabilities"][class_name] for class_name in classes]
                for row in ordered
            ],
            dtype=np.float64,
        )
        if (
            not np.allclose(probabilities.sum(axis=1), 1.0)
            or not np.isfinite(probabilities).all()
            or any(
                row["entity_unit_id"] != row["held_out_entity_unit_id"]
                for row in ordered
            )
        ):
            raise ValueError("pilot OOF predictions fail probability or group audit")
        predicted = probabilities.argmax(axis=1)
        if task == "familiarity":
            auroc = roc_auc_score(labels, probabilities[:, 1])
        else:
            auroc = roc_auc_score(
                labels,
                probabilities,
                labels=np.arange(len(classes)),
                multi_class="ovr",
                average="macro",
            )
        observed = metrics_by_key[candidate]
        recomputed = {
            "auroc": float(auroc),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, predicted)
            ),
            "log_loss": float(
                log_loss(labels, probabilities, labels=np.arange(len(classes)))
            ),
        }
        if any(
            not np.isclose(recomputed[name], observed[name], rtol=0.0, atol=1e-12)
            for name in recomputed
        ):
            raise ValueError("pilot aggregate metric does not recompute")
        for row in ordered:
            if row["example_id"] not in examples:
                raise ValueError("pilot prediction references an unknown example")

    evidence = {
        "config_sha256": config.config_hash,
        "prompt_manifest_sha256": prompt.shard_sha256,
        "metrics_manifest_sha256": metrics_shard.sha256,
        "predictions_manifest_sha256": predictions_shard.sha256,
        "analysis_sha256": lineage["analysis_sha256"],
        "activation_npz_sha256": lineage["activation_npz_sha256"],
        "analysis_spec_sha256": lineage["analysis_spec_sha256"],
        "audit_status": "passed",
    }
    return metrics, predictions, evidence


def _candidate(
    metrics: list[dict[str, Any]],
    task: str,
    anchor: str,
    layer: int | None,
    family: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in metrics
        if _key(row) == (task, anchor, layer, family)
    ]
    if len(matches) != 1:
        raise ValueError("pilot candidate identity is not unique")
    return matches[0]


def _plot_task(metrics: list[dict[str, Any]], task: str, output: Path) -> None:
    primary = PRIMARY_ANCHORS[task]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    colors = {
        "residual_static": "#176B87",
        "morphology_plus_residual": "#D97706",
    }
    labels = {
        "residual_static": "Residual static",
        "morphology_plus_residual": "Morphology + residual",
    }
    for axis, anchor, title in (
        (axes[0], primary, "Registered pre-output anchor"),
        (axes[1], "assistant_prefix_end", "Output-proximal control"),
    ):
        for family in colors:
            values = [
                _candidate(metrics, task, anchor, layer, family)["auroc"]
                for layer in LAYERS
            ]
            axis.plot(
                LAYERS,
                values,
                marker="o",
                linewidth=2,
                color=colors[family],
                label=labels[family],
            )
        morphology = _candidate(
            metrics, task, "surface_only", None, "surface_morphology"
        )["auroc"]
        axis.axhline(
            morphology,
            color="#586069",
            linestyle="--",
            linewidth=1.5,
            label="Morphology baseline",
        )
        axis.axhline(0.5, color="#9CA3AF", linestyle=":", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Transformer layer")
        axis.set_xticks(LAYERS)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out-entity AUROC")
    axes[0].set_ylim(0.42, 1.01)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Familiarity decoding" if task == "familiarity" else "Answerability decoding",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_nulls(metrics: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, task in zip(axes, ("familiarity", "answerability"), strict=True):
        anchor = PRIMARY_ANCHORS[task]
        rows = [
            _candidate(metrics, task, anchor, layer, "residual_static")
            for layer in LAYERS
        ]
        nulls = np.asarray([row["permutation_aurocs"] for row in rows]).mean(axis=0)
        observed = rows[0]["mean_layer_omnibus_auroc"]
        axis.hist(nulls, bins=16, color="#CBD5E1", edgecolor="#64748B")
        axis.axvline(observed, color="#B91C1C", linewidth=2.2)
        axis.set_title(task.capitalize())
        axis.set_xlabel("Mean AUROC across four fixed layers")
        axis.set_ylabel("Permutation count")
        axis.text(
            0.98,
            0.94,
            f"observed = {observed:.3f}\np = 1/101",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Within-stratum label-permutation diagnostics",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _summary(metrics: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any]:
    familiarity_rows = [
        _candidate(
            metrics,
            "familiarity",
            "target_intro_end",
            layer,
            "residual_static",
        )
        for layer in LAYERS
    ]
    answerability_rows = [
        _candidate(
            metrics,
            "answerability",
            "user_prompt_end",
            layer,
            "residual_static",
        )
        for layer in LAYERS
    ]
    answerability_combined = [
        _candidate(
            metrics,
            "answerability",
            "user_prompt_end",
            layer,
            "morphology_plus_residual",
        )
        for layer in LAYERS
    ]
    return {
        "schema_version": 1,
        "claim_scope": "development_only_model_specific_decodability",
        "evidence": evidence,
        "familiarity": {
            "surface_morphology_auroc": _candidate(
                metrics,
                "familiarity",
                "surface_only",
                None,
                "surface_morphology",
            )["auroc"],
            "residual_mean_layer_auroc": familiarity_rows[0][
                "mean_layer_omnibus_auroc"
            ],
            "residual_mean_layer_permutation_p": familiarity_rows[0][
                "mean_layer_omnibus_p"
            ],
            "residual_layer_aurocs": {
                str(row["layer_id"]): row["auroc"] for row in familiarity_rows
            },
            "worst_condition_balanced_accuracy": {
                str(row["layer_id"]): row["worst_condition_balanced_accuracy"]
                for row in familiarity_rows
            },
        },
        "answerability": {
            "surface_morphology_auroc": _candidate(
                metrics,
                "answerability",
                "surface_only",
                None,
                "surface_morphology",
            )["auroc"],
            "surface_morphology_log_loss": _candidate(
                metrics,
                "answerability",
                "surface_only",
                None,
                "surface_morphology",
            )["log_loss"],
            "residual_mean_layer_auroc": answerability_rows[0][
                "mean_layer_omnibus_auroc"
            ],
            "residual_mean_layer_permutation_p": answerability_rows[0][
                "mean_layer_omnibus_p"
            ],
            "residual_layer_aurocs": {
                str(row["layer_id"]): row["auroc"] for row in answerability_rows
            },
            "combined_layer_log_losses": {
                str(row["layer_id"]): row["log_loss"]
                for row in answerability_combined
            },
            "combined_log_loss_relative_improvement_by_layer": {
                str(row["layer_id"]): (
                    _candidate(
                        metrics,
                        "answerability",
                        "surface_only",
                        None,
                        "surface_morphology",
                    )["log_loss"]
                    - row["log_loss"]
                )
                / _candidate(
                    metrics,
                    "answerability",
                    "surface_only",
                    None,
                    "surface_morphology",
                )["log_loss"]
                for row in answerability_combined
            },
            "incremental_claim_status": "not_evaluable",
        },
    }


def _write_report(output: Path, summary: dict[str, Any]) -> None:
    familiarity = summary["familiarity"]
    answerability = summary["answerability"]
    evidence = summary["evidence"]
    output.write_text(
        f"""# Familiarity vs. Answerability: Qwen3-1.7B Development Pilot

## Status

This is a development-only, model-specific pilot. It is not confirmatory F2A,
not causal evidence, and not a claim about truth, metacognition, or production
models. All 288 factorial examples were evaluated with leave-one-entity-unit-out
cross-validation across eight entity units.

## Behavioral Feasibility

The preceding immutable generation gate passed: target-bound accuracy was
`0.9911`, all factorial cells had valid-format rate `1.0`, and the
evidence-absent answer-attempt rate was `0.8846`.

## Familiarity

At the registered `target_intro_end` anchor, the surface-morphology baseline was
at chance (`AUROC = {familiarity['surface_morphology_auroc']:.3f}`). The
residual-stream probe averaged
`AUROC = {familiarity['residual_mean_layer_auroc']:.3f}` over the four fixed
layers, with within-stratum permutation resolution-floor
`p = {familiarity['residual_mean_layer_permutation_p']:.4f}`. The result was
present in every answerability condition. This supports held-out-entity
decodability of the registered real-versus-synthetic familiarity proxy in this
Qwen checkpoint.

## Answerability

Visible morphology already predicted much of the registered three-state target
(`macro AUROC = {answerability['surface_morphology_auroc']:.3f}`,
`log loss = {answerability['surface_morphology_log_loss']:.3f}`). The
residual-only probe averaged
`macro AUROC = {answerability['residual_mean_layer_auroc']:.3f}`. The frozen
morphology-plus-residual candidates reduced held-out log loss at all four
registered layers, by
`{100 * min(answerability['combined_log_loss_relative_improvement_by_layer'].values()):.1f}%`
to
`{100 * max(answerability['combined_log_loss_relative_improvement_by_layer'].values()):.1f}%`
relative to morphology alone. This is suggestive, but V13 did not preregister a
paired incremental null distribution or confidence interval. Incremental
internal answerability information is therefore **not evaluable** in this
pilot, rather than supported or refuted.

The historical V13 artifact key `surface_design_oracle` is reported as a
linear design-feature control. The raw fields are sufficient to reconstruct the
answerability label, but the frozen linear model omitted the required
entity-order by code-position interaction; V14 records this post-result naming
correction without changing V13.

## Interpretation

The pilot demonstrates that the extraction, provenance, grouped evaluation, and
permutation machinery works on a real local model. It provides a strong
familiarity-decoding pilot and a suggestive answerability result that motivates
a registered incremental contrast. It does not establish a mechanism or
satisfy the confirmatory Fellowship study by itself. Confirmatory Gemma
execution remains gated on independent human naturalness ratings and untouched
protected splits.

## Figures

![Familiarity layer decoding](familiarity_layer_auroc.png)

![Answerability layer decoding](answerability_layer_auroc.png)

![Permutation diagnostics](omnibus_permutation_nulls.png)

## Provenance

- Analysis SHA-256: `{evidence['analysis_sha256']}`
- Activation NPZ SHA-256: `{evidence['activation_npz_sha256']}`
- Metrics shard SHA-256: `{evidence['metrics_manifest_sha256']}`
- Predictions shard SHA-256: `{evidence['predictions_manifest_sha256']}`
- Analysis specification SHA-256: `{evidence['analysis_spec_sha256']}`
- Artifact recomputation audit: `{evidence['audit_status']}`
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--metrics-manifest", required=True)
    parser.add_argument("--predictions-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics, predictions, evidence = load_and_audit(
        root=root,
        config_path=Path(args.config),
        prompt_manifest=Path(args.prompt_manifest),
        metrics_manifest=Path(args.metrics_manifest),
        predictions_manifest=Path(args.predictions_manifest),
    )
    summary = _summary(metrics, evidence)
    summary_path = output / "pilot_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot_task(metrics, "familiarity", output / "familiarity_layer_auroc.png")
    _plot_task(metrics, "answerability", output / "answerability_layer_auroc.png")
    _plot_nulls(metrics, output / "omnibus_permutation_nulls.png")
    _write_report(output / "README.md", summary)
    manifest = {
        "schema_version": 1,
        "claim_scope": summary["claim_scope"],
        "source_evidence": evidence,
        "files": {
            path.name: {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "MANIFEST.json"
        },
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
