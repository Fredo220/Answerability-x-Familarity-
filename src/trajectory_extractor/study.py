from __future__ import annotations

import json
import csv
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from trajectory_extractor.ablations import last_token_only, random_projection, shuffled_layers
from trajectory_extractor.artifacts import RunStore
from trajectory_extractor.datasets.concept_mixing import ConceptMixingExample
from trajectory_extractor.datasets.jailbreak import JailbreakExample
from trajectory_extractor.datasets.real_transfer import FactualTriple
from trajectory_extractor.evaluation import (
    binary_metrics,
    evaluate_and_predict_prefix_surfaces,
    paired_bootstrap_auc_delta,
    select_threshold,
)
from trajectory_extractor.extraction import generate_and_extract
from trajectory_extractor.features import compute_raw_dynamics
from trajectory_extractor.labels import classify_concept_response
from trajectory_extractor.labels import is_refusal
from trajectory_extractor.judge import judge_with_llama_guard, stratified_audit_indices
from trajectory_extractor.operator_residual import LayerwiseOperatorResidual
from trajectory_extractor.probes import LayerwiseStaticProbe
from trajectory_extractor.reporting import classify_detection
from trajectory_extractor.types import ExperimentConfig, TrajectoryBatch


def load_concept_examples(path: str | Path) -> list[ConceptMixingExample]:
    return [ConceptMixingExample(**json.loads(line)) for line in Path(path).read_text().splitlines() if line]


def extract_concept_run(
    model,
    tokenizer,
    examples: list[ConceptMixingExample],
    *,
    config: ExperimentConfig,
    run_id: str,
    store: RunStore,
    dataset_provenance: dict | None = None,
) -> None:
    manifest = {"config": config.to_dict(), "track": "concept_mixing"}
    if dataset_provenance is not None:
        manifest["dataset"] = dataset_provenance
    _prepare_run(store, run_id, manifest)
    completed = sum(store.has_example(run_id, example.example_id) for example in examples)
    session_started = perf_counter()
    session_completed = 0
    for index, example in enumerate(examples):
        if store.has_example(run_id, example.example_id):
            continue
        example_started = perf_counter()
        run = generate_and_extract(
            model,
            tokenizer,
            example.prompt,
            config=config,
            run_id=run_id,
            example_id=example.example_id,
            track="concept_mixing",
            split=example.split,
            label=0,
            provenance={
                "expected_answer": example.answer,
                "entity_family": example.entity_family,
                "template_group": example.template_group,
                "distractor_count": example.distractor_count,
                "name_similarity": example.name_similarity,
                "answer_position": example.answer_position,
                "entity_rarity": example.entity_rarity,
            },
        )
        decision = classify_concept_response(run.response, example.answer, example.distractor_answers)
        run.label = decision.exact_error
        run.provenance.update(
            {
                "concept_outcome": decision.outcome.value,
                "binding_error": decision.binding_error,
                "distractor_answers": list(example.distractor_answers),
                "target_entity": example.target_entity,
                "extraction_runtime_seconds": perf_counter() - example_started,
            }
        )
        store.write(run)
        completed += 1
        session_completed += 1
        _report_extraction_progress(
            "concept", completed, len(examples), session_completed, session_started, every=10
        )


def extract_real_transfer_run(
    model,
    tokenizer,
    examples: list[FactualTriple],
    *,
    config: ExperimentConfig,
    run_id: str,
    store: RunStore,
    dataset_provenance: dict | None = None,
) -> None:
    manifest = {"config": config.to_dict(), "track": "real_transfer"}
    if dataset_provenance is not None:
        manifest["dataset"] = dataset_provenance
    _prepare_run(store, run_id, manifest)
    completed = sum(store.has_example(run_id, example.example_id) for example in examples)
    session_started = perf_counter()
    session_completed = 0
    for index, example in enumerate(examples):
        if store.has_example(run_id, example.example_id):
            continue
        example_started = perf_counter()
        run = generate_and_extract(
            model,
            tokenizer,
            example.prompt,
            config=config,
            run_id=run_id,
            example_id=example.example_id,
            track="real_transfer",
            split=example.split,
            label=0,
            provenance={
                "expected_answer": example.object,
                "source_url": example.source_url,
                "subject": example.subject,
                "relation": example.relation,
                "distractor_answers": list(example.distractor_answers),
            },
        )
        decision = classify_concept_response(run.response, example.object, example.distractor_answers)
        run.label = decision.exact_error
        run.provenance.update(
            {
                "concept_outcome": decision.outcome.value,
                "binding_error": decision.binding_error,
                "extraction_runtime_seconds": perf_counter() - example_started,
            }
        )
        store.write(run)
        completed += 1
        session_completed += 1
        _report_extraction_progress(
            "transfer", completed, len(examples), session_completed, session_started, every=10
        )


def extract_jailbreak_run(
    model,
    tokenizer,
    examples: list[JailbreakExample],
    *,
    config: ExperimentConfig,
    run_id: str,
    store: RunStore,
    dataset_provenance: dict | None = None,
) -> None:
    manifest = {"config": config.to_dict(), "track": "jailbreak"}
    if dataset_provenance is not None:
        manifest["dataset"] = dataset_provenance
    _prepare_run(store, run_id, manifest)
    completed = sum(store.has_example(run_id, example.example_id) for example in examples)
    session_started = perf_counter()
    session_completed = 0
    for index, example in enumerate(examples):
        if store.has_example(run_id, example.example_id):
            continue
        prompt = example.artifact or example.behavior
        example_started = perf_counter()
        run = generate_and_extract(
            model,
            tokenizer,
            prompt,
            config=config,
            run_id=run_id,
            example_id=example.example_id,
            track="jailbreak",
            split="unjudged",
            label=-1,
            provenance={
                "behavior": example.behavior,
                "behavior_name": example.behavior_name,
                "goal": example.goal,
                "category": example.category,
                "benign": example.benign,
                "source": example.source,
                "used_frozen_artifact": example.artifact is not None,
                "artifact_source": example.artifact_source,
                "pair_id": example.pair_id,
                "extraction_runtime_seconds": 0.0,
            },
        )
        run.provenance["extraction_runtime_seconds"] = perf_counter() - example_started
        store.write(run)
        completed += 1
        session_completed += 1
        _report_extraction_progress(
            "jailbreak", completed, len(examples), session_completed, session_started, every=5
        )


def judge_jailbreak_run(
    guard_model,
    guard_tokenizer,
    *,
    run_id: str,
    store: RunStore,
    audit_fraction: float = 0.20,
) -> dict:
    example_ids = store.judgable_ids(run_id)
    labels = []
    decisions = []
    for example_id in example_ids:
        run = store.read_judgable(run_id, example_id)
        if not run.track.startswith("jailbreak"):
            raise ValueError("Guard judging is restricted to jailbreak runs")
        decision = judge_with_llama_guard(
            guard_model,
            guard_tokenizer,
            str(run.provenance["behavior"]),
            run.response,
        )
        run.label = int(decision.unsafe)
        run.split = str(run.provenance.get("category", "unknown"))
        run.provenance.update(
            {
                "guard_unsafe": decision.unsafe,
                "guard_category": decision.category,
                "guard_raw": decision.raw_text,
                "refused": is_refusal(run.response),
            }
        )
        store.write_judgable(run)
        labels.append(run.label)
        decisions.append(
            {
                "example_id": run.example_id,
                "guard_unsafe": decision.unsafe,
                "guard_category": decision.category,
                "benign": bool(run.provenance.get("benign", False)),
                "category": str(run.provenance.get("category", "unknown")),
                "behavior": run.provenance.get("behavior"),
                "response": run.response,
            }
        )
    audit_strata = [
        (row["category"], bool(row["benign"]), bool(row["guard_unsafe"]))
        for row in decisions
    ]
    audit = stratified_audit_indices(audit_strata, fraction=audit_fraction)
    audit_rows = [
        {**decisions[index], "human_unsafe": None, "reviewer": None, "notes": ""}
        for index in audit
    ]
    store.write_json(run_id, "labels", "guard_decisions", decisions)
    store.write_json(run_id, "labels", "manual_audit_sample", audit_rows)
    return {"judged": len(decisions), "manual_audit_sample": len(audit_rows)}


def record_manual_audit(
    path: str | Path,
    *,
    run_id: str,
    store: RunStore,
) -> dict:
    source = Path(path)
    if source.suffix == ".json":
        rows = json.loads(source.read_text())
    else:
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    expected = json.loads((store.root / run_id / "labels" / "manual_audit_sample.json").read_text())
    expected_ids = {str(row["example_id"]) for row in expected}
    received_ids = {str(row.get("example_id", "")) for row in rows}
    if received_ids != expected_ids:
        raise ValueError("Completed audit must contain exactly the frozen audit sample IDs")
    completed = []
    agreements = []
    for row in rows:
        value = str(row.get("human_unsafe", "")).strip().casefold()
        if value not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(f"Invalid human_unsafe value for {row.get('example_id')}")
        human = value in {"1", "true", "yes"}
        guard = bool(
            store.read_judgable(run_id, str(row["example_id"])).provenance["guard_unsafe"]
        )
        agreements.append(human == guard)
        completed.append({**row, "human_unsafe": human, "guard_unsafe": guard})
    result = {
        "audited": len(completed),
        "agreement": float(np.mean(agreements)),
        "rows": completed,
    }
    store.write_json(run_id, "labels", "manual_audit_completed", result)
    return {"audited": result["audited"], "agreement": result["agreement"]}


def evaluate_detection_methods(
    batch: TrajectoryBatch,
    *,
    pca_dims: int = 32,
    ridge_alpha: float = 1e-3,
    n_bootstrap: int = 2000,
    return_predictions: bool = False,
) -> dict:
    train = np.flatnonzero(batch.splits == "train")
    val = np.flatnonzero(batch.splits == "val")
    test = np.flatnonzero(batch.splits == "test")
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Expected non-empty train, val, and test splits")

    static_model = LayerwiseStaticProbe().fit(batch, train)
    static = static_model.predict_scores(batch)[..., None]
    stable_train = train[batch.labels[train] == 0]
    if stable_train.size == 0:
        raise ValueError("Operator baseline requires correct/safe training examples")
    operator_model = LayerwiseOperatorResidual(pca_dims, ridge_alpha).fit(batch, stable_train)
    operator = _pad(operator_model.transform(batch), batch.hidden_states.shape[2])[..., None]
    dynamics = compute_raw_dynamics(batch.hidden_states)
    raw = np.stack(
        [
            _pad(dynamics.velocity, batch.hidden_states.shape[2]),
            _pad(dynamics.curvature, batch.hidden_states.shape[2]),
            _pad(dynamics.direction_change, batch.hidden_states.shape[2]),
        ],
        axis=-1,
    )
    output = np.stack([batch.token_logprobs, batch.token_entropies], axis=-1)
    output = np.broadcast_to(output[:, :, None, :], (*output.shape[:2], batch.hidden_states.shape[2], 2))
    methods = {
        "output": output,
        "static": static,
        "raw_dynamics": raw,
        "operator_residual": operator,
        "combined": np.concatenate([static, raw, operator], axis=-1),
    }

    method_results = {}
    test_predictions = {}
    for name, features in methods.items():
        validation, validation_probabilities, test_surface = evaluate_and_predict_prefix_surfaces(
            features,
            batch.labels,
            token_mask=batch.token_mask,
            train_indices=train,
            validation_indices=val,
            test_indices=test,
        )
        best_index = np.unravel_index(int(np.nanargmax(validation.auroc)), validation.auroc.shape)
        val_selected = validation_probabilities[:, best_index[0], best_index[1]]
        probabilities = test_surface[:, best_index[0], best_index[1]]
        threshold = select_threshold(batch.labels[val], val_selected)
        metrics = binary_metrics(batch.labels[test], probabilities, threshold=threshold)
        test_predictions[name] = probabilities
        method_results[name] = {
            "selected_token": int(best_index[0]),
            "selected_layer": int(best_index[1]),
            "validation_auroc": float(validation.auroc[best_index]),
            "validation_auprc": float(validation.auprc[best_index]),
            "test": metrics,
            "test_auroc": metrics["auroc"],
            "test_auprc": metrics["auprc"],
            "test_calibration_error": metrics["calibration_error"],
            "test_false_positive_rate": metrics["false_positive_rate"],
            "validation_diagnostics": {
                "status": "not_interpretable",
                "reason": (
                    "independently_fitted_prefix_classifiers_do_not_share_a_transferable_threshold"
                ),
            },
            "validation_surface": {
                "auroc": validation.auroc.tolist(),
                "auprc": validation.auprc.tolist(),
            },
        }

    simple = max(("output", "static"), key=lambda name: method_results[name]["validation_auroc"])
    dynamic = max(
        ("raw_dynamics", "operator_residual", "combined"),
        key=lambda name: method_results[name]["validation_auroc"],
    )
    bootstrap_groups, bootstrap_unit = _bootstrap_groups(batch, test)
    bootstrap = paired_bootstrap_auc_delta(
        batch.labels[test],
        test_predictions[dynamic],
        test_predictions[simple],
        n_bootstrap=n_bootstrap,
        groups=bootstrap_groups,
    )
    decision = classify_detection(bootstrap.delta, bootstrap.lower, bootstrap.upper)
    result = {
        "methods": method_results,
        "selected_simple_baseline": simple,
        "selected_dynamics_method": dynamic,
        "bootstrap_delta": {
            "delta": bootstrap.delta,
            "lower": bootstrap.lower,
            "upper": bootstrap.upper,
        },
        "bootstrap_samples": bootstrap.samples.tolist(),
        "bootstrap_unit": bootstrap_unit,
        "decision": asdict(decision),
        "fit_example_ids": {
            "operator": list(operator_model.fit_example_ids),
            "static_probe": list(static_model.fit_example_ids),
        },
        "operator_reference_class": 0,
    }
    if return_predictions:
        result["_test_indices"] = test.tolist()
        result["_test_predictions"] = {name: values.tolist() for name, values in test_predictions.items()}
    return result


def evaluate_grouped_jailbreak_methods(
    batch: TrajectoryBatch,
    categories: list[str],
    *,
    pca_dims: int = 32,
    ridge_alpha: float = 1e-3,
    n_bootstrap: int = 2000,
) -> dict:
    """Rotate held-out category groups through validation and test roles."""
    category_array = np.asarray(categories)
    if category_array.shape != (len(batch.example_ids),):
        raise ValueError("categories must align with batch examples")
    unique = sorted(set(categories))
    if len(unique) < 3:
        raise ValueError("Grouped evaluation requires at least three behavior categories")
    folds = []
    oof_simple = np.full(len(categories), np.nan, dtype=np.float32)
    oof_dynamic = np.full(len(categories), np.nan, dtype=np.float32)
    for index, test_category in enumerate(unique):
        val_category = unique[(index + 1) % len(unique)]
        splits = np.full(len(categories), "train", dtype="<U5")
        splits[category_array == val_category] = "val"
        splits[category_array == test_category] = "test"
        fold_batch = TrajectoryBatch(
            example_ids=batch.example_ids,
            labels=batch.labels,
            splits=splits,
            hidden_states=batch.hidden_states,
            token_mask=batch.token_mask,
            token_logprobs=batch.token_logprobs,
            token_entropies=batch.token_entropies,
            provenance=batch.provenance,
        )
        try:
            result = evaluate_detection_methods(
                fold_batch,
                pca_dims=pca_dims,
                ridge_alpha=ridge_alpha,
                n_bootstrap=n_bootstrap,
                return_predictions=True,
            )
        except ValueError as error:
            folds.append(
                {
                    "test_category": test_category,
                    "validation_category": val_category,
                    "skipped": str(error),
                }
            )
            continue
        test_indices = np.asarray(result.pop("_test_indices"), dtype=int)
        predictions = result.pop("_test_predictions")
        simple_name = result["selected_simple_baseline"]
        dynamic_name = result["selected_dynamics_method"]
        oof_simple[test_indices] = np.asarray(predictions[simple_name], dtype=np.float32)
        oof_dynamic[test_indices] = np.asarray(predictions[dynamic_name], dtype=np.float32)
        folds.append(
            {
                "test_category": test_category,
                "validation_category": val_category,
                "result": result,
            }
        )
    completed = [fold for fold in folds if "result" in fold]
    supported = sum(fold["result"]["decision"]["outcome"] == "supported" for fold in completed)
    valid = np.isfinite(oof_simple) & np.isfinite(oof_dynamic)
    aggregate = None
    if valid.any() and np.unique(batch.labels[valid]).size == 2:
        bootstrap = paired_bootstrap_auc_delta(
            batch.labels[valid],
            oof_dynamic[valid],
            oof_simple[valid],
            n_bootstrap=n_bootstrap,
            groups=category_array[valid],
        )
        aggregate = {
            "evaluated_examples": int(valid.sum()),
            "delta": bootstrap.delta,
            "lower": bootstrap.lower,
            "upper": bootstrap.upper,
            "decision": asdict(classify_detection(bootstrap.delta, bootstrap.lower, bootstrap.upper)),
            "bootstrap_samples": bootstrap.samples.tolist(),
            "bootstrap_unit": "behavior_category",
        }
    return {
        "folds": folds,
        "supported_folds": supported,
        "completed_folds": len(completed),
        "total_folds": len(folds),
        "aggregate_oof": aggregate,
    }


def evaluate_detection_ablations(
    batch: TrajectoryBatch,
    *,
    metadata: list[dict] | None = None,
    pca_dimensions: tuple[int, ...] = (16, 32, 64),
    ridge_alpha: float = 1e-3,
    n_bootstrap: int = 500,
) -> dict:
    results = {}
    primary = evaluate_detection_methods(
        batch,
        pca_dims=32 if 32 in pca_dimensions else pca_dimensions[0],
        ridge_alpha=ridge_alpha,
        n_bootstrap=n_bootstrap,
        return_predictions=True,
    )
    test_indices = np.asarray(primary.pop("_test_indices"), dtype=int)
    predictions = primary.pop("_test_predictions")
    results["full_trajectory"] = primary
    for dimensions in pca_dimensions:
        if dimensions == 32 and 32 in pca_dimensions:
            results[f"pca_{dimensions}"] = primary
        else:
            results[f"pca_{dimensions}"] = evaluate_detection_methods(
                batch,
                pca_dims=dimensions,
                ridge_alpha=ridge_alpha,
                n_bootstrap=n_bootstrap,
            )
    results["last_token_only"] = evaluate_detection_methods(
        last_token_only(batch),
        pca_dims=32,
        ridge_alpha=ridge_alpha,
        n_bootstrap=n_bootstrap,
    )
    results["shuffled_layers"] = evaluate_detection_methods(
        shuffled_layers(batch),
        pca_dims=32,
        ridge_alpha=ridge_alpha,
        n_bootstrap=n_bootstrap,
    )
    results["random_projection_32"] = evaluate_detection_methods(
        random_projection(batch, dimensions=32),
        pca_dims=32,
        ridge_alpha=ridge_alpha,
        n_bootstrap=n_bootstrap,
    )
    subgroup_results = {}
    if metadata is not None:
        if len(metadata) != len(batch.example_ids):
            raise ValueError("metadata must align with batch examples")
        dynamic = primary["selected_dynamics_method"]
        selected_predictions = np.asarray(predictions[dynamic], dtype=float)
        test_labels = batch.labels[test_indices]
        for field in ("prompt_length_bin", "entity_rarity", "distractor_count"):
            values = np.asarray([metadata[index].get(field, "unknown") for index in test_indices])
            field_results = {}
            for value in sorted(set(values), key=str):
                selected = values == value
                if selected.sum() < 2 or np.unique(test_labels[selected]).size < 2:
                    field_results[str(value)] = {"n": int(selected.sum()), "evaluable": False}
                    continue
                field_results[str(value)] = {
                    "n": int(selected.sum()),
                    "evaluable": True,
                    "auroc": float(roc_auc_score(test_labels[selected], selected_predictions[selected])),
                    "auprc": float(
                        average_precision_score(test_labels[selected], selected_predictions[selected])
                    ),
                }
            subgroup_results[field] = field_results
    return {"runs": results, "subgroups": subgroup_results}


def build_real_transfer_evaluation_batch(
    concept_batch: TrajectoryBatch,
    transfer_batch: TrajectoryBatch,
) -> TrajectoryBatch:
    reference = np.flatnonzero(np.isin(concept_batch.splits, ["train", "val"]))
    if not len(reference):
        raise ValueError("Concept reference batch requires train and validation examples")
    if concept_batch.hidden_states.shape[2:] != transfer_batch.hidden_states.shape[2:]:
        raise ValueError("Concept and transfer activations must share layer and hidden dimensions")
    max_tokens = max(concept_batch.hidden_states.shape[1], transfer_batch.hidden_states.shape[1])
    total = len(reference) + len(transfer_batch.example_ids)
    n_layers, hidden_dim = concept_batch.hidden_states.shape[2:]
    hidden = np.zeros((total, max_tokens, n_layers, hidden_dim), dtype=np.float16)
    mask = np.zeros((total, max_tokens), dtype=bool)
    logprobs = np.zeros((total, max_tokens), dtype=np.float32)
    entropies = np.zeros((total, max_tokens), dtype=np.float32)

    def copy_rows(target_start, source_batch, source_indices):
        count = len(source_indices)
        width = source_batch.hidden_states.shape[1]
        target = slice(target_start, target_start + count)
        hidden[target, :width] = source_batch.hidden_states[source_indices]
        mask[target, :width] = source_batch.token_mask[source_indices]
        logprobs[target, :width] = source_batch.token_logprobs[source_indices]
        entropies[target, :width] = source_batch.token_entropies[source_indices]

    copy_rows(0, concept_batch, reference)
    transfer_indices = np.arange(len(transfer_batch.example_ids))
    copy_rows(len(reference), transfer_batch, transfer_indices)
    return TrajectoryBatch(
        example_ids=tuple(f"concept::{concept_batch.example_ids[index]}" for index in reference)
        + tuple(f"transfer::{example_id}" for example_id in transfer_batch.example_ids),
        labels=np.concatenate([concept_batch.labels[reference], transfer_batch.labels]),
        splits=np.concatenate(
            [concept_batch.splits[reference], np.full(len(transfer_indices), "test")]
        ),
        hidden_states=hidden,
        token_mask=mask,
        token_logprobs=logprobs,
        token_entropies=entropies,
        provenance=tuple(
            concept_batch.provenance[index] if concept_batch.provenance else {}
            for index in reference
        )
        + (
            transfer_batch.provenance
            if transfer_batch.provenance
            else tuple({} for _ in transfer_batch.example_ids)
        ),
    )


def _pad(metric: np.ndarray, n_layers: int) -> np.ndarray:
    result = np.zeros((*metric.shape[:2], n_layers), dtype=np.float32)
    result[:, :, -metric.shape[2] :] = metric
    return result


def _bootstrap_groups(batch: TrajectoryBatch, indices: np.ndarray) -> tuple[np.ndarray, str]:
    rows = [batch.provenance[index] if batch.provenance else {} for index in indices]
    for key, unit in (("entity_family", "entity_family"), ("pair_id", "matched_pair")):
        if rows and all(row.get(key) is not None for row in rows):
            return np.asarray([str(row[key]) for row in rows]), unit
    return np.asarray([batch.example_ids[index] for index in indices]), "example"


def _report_extraction_progress(
    track: str,
    completed: int,
    total: int,
    session_completed: int,
    session_started: float,
    *,
    every: int,
) -> None:
    if session_completed != 1 and completed != total and session_completed % every:
        return
    elapsed = perf_counter() - session_started
    seconds_per_example = elapsed / max(1, session_completed)
    eta_seconds = max(0.0, total - completed) * seconds_per_example
    print(
        json.dumps(
            {
                "track": track,
                "completed": completed,
                "total": total,
                "session_completed": session_completed,
                "seconds_per_example": round(seconds_per_example, 3),
                "eta_seconds": round(eta_seconds, 1),
            }
        ),
        flush=True,
    )


def _prepare_run(store: RunStore, run_id: str, manifest: dict) -> None:
    path = store.root / run_id / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        for key in ("config", "track", "dataset"):
            if key not in manifest:
                continue
            if existing.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume {run_id!r}: manifest field {key!r} changed")
        return
    store.write_manifest(run_id, manifest)
