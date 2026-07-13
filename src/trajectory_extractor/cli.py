from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from trajectory_extractor.artifacts import RunStore
from trajectory_extractor.circuit_followup import prepare_circuit_followup
from trajectory_extractor.datasets.concept_mixing import (
    ConceptMixingExample,
    generate_concept_mixing_examples,
    write_examples_jsonl,
)
from trajectory_extractor.datasets.jailbreak import (
    JailbreakExample,
    build_jailbreak_study_file,
    load_official_jailbreakbench,
    validate_jailbreak_study_set,
)
from trajectory_extractor.datasets.real_transfer import (
    fetch_wikidata_transfer_file,
    load_documented_triples,
)
from trajectory_extractor.extraction import generate_and_extract
from trajectory_extractor.intervention_study import (
    evaluate_jailbreak_intervention,
    generate_jailbreak_intervention_test,
    generate_jailbreak_intervention_validation,
    run_concept_intervention_study,
    select_jailbreak_intervention,
)
from trajectory_extractor.model_loader import load_hf_model, unload_model
from trajectory_extractor.labels import is_refusal, safety_rates
from trajectory_extractor.report_builder import write_study_report
from trajectory_extractor.rlmf_artifacts import RLMFArtifactStore
from trajectory_extractor.rlmf_data import write_popqa_snapshot
from trajectory_extractor.rlmf_types import RLMFConfig
from trajectory_extractor.secondary_artifacts import (
    SecondaryArtifactStore,
    build_analysis_provenance,
    ensure_durable_directory,
)
from trajectory_extractor.secondary_study import evaluate_concept_secondary
from trajectory_extractor.steering import SteeringHook
from trajectory_extractor.study import (
    evaluate_detection_ablations,
    build_real_transfer_evaluation_batch,
    evaluate_detection_methods,
    evaluate_grouped_jailbreak_methods,
    extract_concept_run,
    extract_jailbreak_run,
    extract_real_transfer_run,
    judge_jailbreak_run,
    load_concept_examples,
    record_manual_audit,
)
from trajectory_extractor.types import ExperimentConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="feature-dynamics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    concept = subparsers.add_parser("generate-concept-data")
    concept.add_argument("--output", default="data/processed/concept_mixing.jsonl")
    concept.add_argument("--total", type=int, default=1200)
    concept.add_argument("--seed", type=int, default=42)

    validate = subparsers.add_parser("validate-jailbreak-data")
    validate.add_argument("path")

    prepare_jbb = subparsers.add_parser("prepare-jailbreak-data")
    prepare_jbb.add_argument("harmful")
    prepare_jbb.add_argument("benign")
    prepare_jbb.add_argument("artifact")
    prepare_jbb.add_argument("--artifact-commit", required=True)
    prepare_jbb.add_argument("--output", default="data/external/jailbreakbench/study.jsonl")

    smoke = subparsers.add_parser("smoke-extract")
    smoke.add_argument("--config", default="configs/llama32_1b.json")
    smoke.add_argument("--prompt", default="Answer with one word: What color is the sky?")
    smoke.add_argument("--run-id", default="local-smoke")

    extract_concept = subparsers.add_parser("extract-concept")
    extract_concept.add_argument("--config", default="configs/llama32_1b.json")
    extract_concept.add_argument("--data", default="data/processed/concept_mixing.jsonl")
    extract_concept.add_argument("--run-id", default="concept-main")
    extract_concept.add_argument(
        "--pilot-per-split",
        type=int,
        help="Extract the first N examples from each split; rerun without this flag to resume the full run.",
    )

    evaluate = subparsers.add_parser("evaluate-concept")
    evaluate.add_argument("--config", default="configs/llama32_1b.json")
    evaluate.add_argument("--run-id", default="concept-main")
    evaluate.add_argument("--bootstrap", type=int, default=2000)
    evaluate.add_argument("--endpoint", choices=("exact_error", "binding_error"), default="exact_error")

    evaluate_secondary = subparsers.add_parser("evaluate-secondary-concept")
    evaluate_secondary.add_argument("--config", default="configs/llama32_1b.json")
    evaluate_secondary.add_argument("--run-id", default="concept-main")
    evaluate_secondary.add_argument("--bootstrap", type=int, default=2000)
    evaluate_secondary.add_argument(
        "--endpoint", choices=("exact_error", "binding_error"), default="exact_error"
    )

    ablate = subparsers.add_parser("ablate-concept")
    ablate.add_argument("--config", default="configs/llama32_1b.json")
    ablate.add_argument("--run-id", default="concept-main")
    ablate.add_argument("--bootstrap", type=int, default=500)
    ablate.add_argument("--endpoint", choices=("exact_error", "binding_error"), default="exact_error")

    transfer = subparsers.add_parser("extract-transfer")
    transfer.add_argument("path")
    transfer.add_argument("--config", default="configs/llama32_1b.json")
    transfer.add_argument("--run-id", default="real-transfer")
    transfer.add_argument("--limit", type=int, default=200)

    prepare_transfer = subparsers.add_parser("prepare-real-transfer")
    prepare_transfer.add_argument("--output", default="data/external/real_transfer.jsonl")
    prepare_transfer.add_argument("--limit", type=int, default=200)

    evaluate_transfer = subparsers.add_parser("evaluate-transfer")
    evaluate_transfer.add_argument("--config", default="configs/llama32_1b.json")
    evaluate_transfer.add_argument("--run-id", default="real-transfer")
    evaluate_transfer.add_argument("--reference-run-id", default="concept-main")
    evaluate_transfer.add_argument("--bootstrap", type=int, default=2000)

    intervene_concept = subparsers.add_parser("intervene-concept")
    intervene_concept.add_argument("--config", default="configs/llama32_1b.json")
    intervene_concept.add_argument("--data", default="data/processed/concept_mixing.jsonl")
    intervene_concept.add_argument("--baseline-run-id", default="concept-main")
    intervene_concept.add_argument("--run-id", default="concept-intervention")
    intervene_concept.add_argument("--bootstrap", type=int, default=2000)
    intervene_concept.add_argument(
        "--endpoint", choices=("exact_error", "binding_error"), default="exact_error"
    )

    extract_jailbreak = subparsers.add_parser("extract-jailbreak")
    extract_jailbreak.add_argument("path")
    extract_jailbreak.add_argument("--config", default="configs/llama32_1b_jailbreak.json")
    extract_jailbreak.add_argument("--run-id", default="jailbreak-main")
    extract_jailbreak.add_argument(
        "--pilot-pairs-per-category",
        type=int,
        help="Extract N matched harmful/benign pairs per category before the full resumable run.",
    )

    judge_jailbreak = subparsers.add_parser("judge-jailbreak")
    judge_jailbreak.add_argument("--config", default="configs/llama_guard_1b.json")
    judge_jailbreak.add_argument("--run-id", default="jailbreak-main")

    audit = subparsers.add_parser("record-manual-audit")
    audit.add_argument("path")
    audit.add_argument("--config", default="configs/llama_guard_1b.json")
    audit.add_argument("--run-id", default="jailbreak-main")

    evaluate_jailbreak = subparsers.add_parser("evaluate-jailbreak")
    evaluate_jailbreak.add_argument("--config", default="configs/llama32_1b_jailbreak.json")
    evaluate_jailbreak.add_argument("--run-id", default="jailbreak-main")
    evaluate_jailbreak.add_argument("--bootstrap", type=int, default=2000)

    jb_val = subparsers.add_parser("prepare-jailbreak-intervention-validation")
    jb_val.add_argument("path")
    jb_val.add_argument("--config", default="configs/llama32_1b_jailbreak.json")
    jb_val.add_argument("--baseline-run-id", default="jailbreak-main")
    jb_val.add_argument("--run-id", default="jailbreak-intervention-val")

    jb_select = subparsers.add_parser("select-jailbreak-intervention")
    jb_select.add_argument("--config", default="configs/llama32_1b_jailbreak.json")
    jb_select.add_argument("--baseline-run-id", default="jailbreak-main")
    jb_select.add_argument("--validation-run-id", default="jailbreak-intervention-val")

    jb_test = subparsers.add_parser("prepare-jailbreak-intervention-test")
    jb_test.add_argument("path")
    jb_test.add_argument("--config", default="configs/llama32_1b_jailbreak.json")
    jb_test.add_argument("--baseline-run-id", default="jailbreak-main")
    jb_test.add_argument("--validation-run-id", default="jailbreak-intervention-val")
    jb_test.add_argument("--run-id", default="jailbreak-intervention-test")

    jb_intervention_eval = subparsers.add_parser("evaluate-jailbreak-intervention")
    jb_intervention_eval.add_argument("--config", default="configs/llama32_1b_jailbreak.json")
    jb_intervention_eval.add_argument("--run-id", default="jailbreak-intervention-test")
    jb_intervention_eval.add_argument("--bootstrap", type=int, default=2000)

    report = subparsers.add_parser("report-study")
    report.add_argument("--config", default="configs/llama32_1b.json")
    report.add_argument("--output", default="docs/generated_study_report.md")

    circuit_followup = subparsers.add_parser("prepare-circuit-followup")
    circuit_followup.add_argument("--config", default="configs/llama32_1b.json")
    circuit_followup.add_argument("--run-id", default="concept-main")
    circuit_followup.add_argument(
        "--endpoint", choices=("exact_error", "binding_error"), default="exact_error"
    )
    circuit_followup.add_argument("--per-stratum", type=int, default=5)

    rlmf_prepare_data = subparsers.add_parser("rlmf-prepare-data")
    rlmf_prepare_data.add_argument("--config", required=True)
    rlmf_prepare_data.add_argument("--root", default=".")

    args = parser.parse_args(argv)
    if args.command == "generate-concept-data":
        examples = generate_concept_mixing_examples(total=args.total, seed=args.seed)
        write_examples_jsonl(examples, args.output, seed=args.seed)
        print(json.dumps({"output": args.output, "count": len(examples)}))
        return 0
    if args.command == "validate-jailbreak-data":
        examples = load_official_jailbreakbench(args.path)
        print(json.dumps(validate_jailbreak_study_set(examples)))
        return 0
    if args.command == "prepare-jailbreak-data":
        manifest = build_jailbreak_study_file(
            args.harmful,
            args.benign,
            args.artifact,
            args.output,
            artifact_commit=args.artifact_commit,
        )
        print(json.dumps(manifest))
        return 0
    if args.command == "smoke-extract":
        config = ExperimentConfig.from_json(args.config)
        model, tokenizer = load_hf_model(config)
        try:
            run = generate_and_extract(
                model,
                tokenizer,
                args.prompt,
                config=config,
                run_id=args.run_id,
                example_id="smoke-000",
                track="smoke",
                split="test",
                label=0,
            )
            store = RunStore(config.output_dir)
            store.write_manifest(
                args.run_id,
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "config": asdict(config),
                    "purpose": "gated-model local smoke test",
                },
            )
            path = store.write(run)
            direction = torch.ones(int(model.config.hidden_size))
            steered = generate_and_extract(
                model,
                tokenizer,
                args.prompt,
                config=config,
                run_id=args.run_id,
                example_id="smoke-steered",
                track="smoke",
                split="test",
                label=0,
                intervention=SteeringHook(
                    model,
                    layer_idx=0,
                    direction=direction,
                    strength=0.1,
                ),
            )
            steered_path = store.write(steered)
            reloaded = store.read(args.run_id, "smoke-steered")
            print(
                json.dumps(
                    {
                        "artifact": str(path),
                        "steered_artifact": str(steered_path),
                        "shape": run.hidden_states.shape,
                        "reload_verified": reloaded.hidden_states.shape == steered.hidden_states.shape,
                    }
                )
            )
        finally:
            unload_model(model)
        return 0
    if args.command == "extract-concept":
        config = ExperimentConfig.from_json(args.config)
        examples = load_concept_examples(args.data)
        examples = _concept_pilot(examples, args.pilot_per_split)
        model, tokenizer = load_hf_model(config)
        try:
            extract_concept_run(
                model,
                tokenizer,
                examples,
                config=config,
                run_id=args.run_id,
                store=RunStore(config.output_dir),
                dataset_provenance=_dataset_provenance(args.data),
            )
        finally:
            unload_model(model)
        return 0
    if args.command == "evaluate-concept":
        config = ExperimentConfig.from_json(args.config)
        store = RunStore(config.output_dir)
        batch = store.load_batch(args.run_id, label_key=args.endpoint)
        result = _timed_detection(
            batch,
            pca_dims=config.pca_dims,
            ridge_alpha=config.ridge_alpha,
            n_bootstrap=args.bootstrap,
            return_predictions=True,
        )
        _persist_detection_predictions(store, args.run_id, batch, result, args.endpoint)
        samples = result.pop("bootstrap_samples")
        store.write_json(args.run_id, "bootstrap", f"detection_auc_delta_{args.endpoint}", samples)
        path = store.write_json(args.run_id, "metrics", f"detection_{args.endpoint}", result)
        if args.endpoint == "exact_error":
            store.write_json(args.run_id, "metrics", "detection", result)
        _write_detection_figures(store, args.run_id, result, suffix=args.endpoint)
        print(json.dumps({"metrics": str(path), "decision": result["decision"]}))
        return 0
    if args.command == "evaluate-secondary-concept":
        config = ExperimentConfig.from_json(args.config)
        secondary = SecondaryArtifactStore(config.output_dir)
        claim = secondary.acquire_claim(args.run_id, args.endpoint)
        batch = RunStore(config.output_dir).load_batch(args.run_id, label_key=args.endpoint)
        started = time.perf_counter()
        result = evaluate_concept_secondary(
            batch,
            pca_dims=config.pca_dims,
            ridge_alpha=config.ridge_alpha,
            n_bootstrap=args.bootstrap,
        )
        result["runtime"] = {
            "seconds": time.perf_counter() - started,
            "max_resident_set_size": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        provenance = _secondary_analysis_provenance(
            config=config,
            config_path=args.config,
            run_id=args.run_id,
            endpoint=args.endpoint,
            bootstrap_count=args.bootstrap,
            result=result,
        )
        result["analysis_provenance"] = provenance
        result["analysis_id"] = provenance["analysis_id"]
        arrays = result.pop("artifacts")
        contrastive_path = secondary.write_npz(
            args.run_id,
            "contrastive_vectors",
            args.endpoint,
            directions=arrays["directions"],
            centers=arrays["centers"],
        )
        dynamics_path = secondary.write_npz(
            args.run_id,
            "vector_dynamics",
            args.endpoint,
            means=arrays["vector_means"],
            scales=arrays["vector_scales"],
        )
        predictions_path = secondary.write_npz(
            args.run_id,
            "comparisons",
            f"predictions_{args.endpoint}",
            validation_indices=arrays["validation_indices"],
            validation_labels=arrays["validation_labels"],
            test_indices=arrays["test_indices"],
            test_labels=arrays["test_labels"],
            contrastive_vector_probability=arrays["contrastive_vector_probability"],
            metacognitive_risk_probability=arrays["metacognitive_risk_probability"],
            validation_metacognitive_risk_surface=arrays[
                "validation_metacognitive_risk_surface"
            ],
            full_monitor_probability=arrays["full_monitor_probability"],
            bootstrap_delta_samples=arrays["bootstrap_delta_samples"],
        )
        metrics_path = secondary.write_json(
            args.run_id,
            "comparisons",
            f"detection_{args.endpoint}",
            result,
        )
        figure_paths = _write_secondary_figures(
            metrics_path.parent / "figures",
            result,
            arrays,
            endpoint=args.endpoint,
        )
        secondary.write_completion(
            args.run_id,
            args.endpoint,
            analysis_id=provenance["analysis_id"],
            artifact_paths=[
                contrastive_path,
                dynamics_path,
                predictions_path,
                metrics_path,
                *figure_paths,
            ],
        )
        secondary.release_claim(claim)
        print(
            json.dumps(
                {
                    "metrics": str(metrics_path),
                    "supported": result["registered_comparison"]["supported"],
                    "evaluable": result["endpoint_status"]["evaluable"],
                    "claim_status": result["claim_status"],
                }
            )
        )
        return 0
    if args.command == "ablate-concept":
        config = ExperimentConfig.from_json(args.config)
        store = RunStore(config.output_dir)
        batch = store.load_batch(args.run_id, label_key=args.endpoint)
        metadata = _batch_metadata(store, args.run_id, batch)
        result = evaluate_detection_ablations(
            batch,
            metadata=metadata,
            ridge_alpha=config.ridge_alpha,
            n_bootstrap=args.bootstrap,
        )
        path = store.write_json(args.run_id, "metrics", f"ablations_{args.endpoint}", result)
        if args.endpoint == "exact_error":
            store.write_json(args.run_id, "metrics", "ablations", result)
        print(json.dumps({"metrics": str(path), "runs": sorted(result["runs"])}))
        return 0
    if args.command == "extract-transfer":
        config = ExperimentConfig.from_json(args.config)
        examples = load_documented_triples(args.path, limit=args.limit)
        model, tokenizer = load_hf_model(config)
        try:
            extract_real_transfer_run(
                model,
                tokenizer,
                examples,
                config=config,
                run_id=args.run_id,
                store=RunStore(config.output_dir),
                dataset_provenance=_dataset_provenance(args.path),
            )
        finally:
            unload_model(model)
        return 0
    if args.command == "prepare-real-transfer":
        print(json.dumps(fetch_wikidata_transfer_file(args.output, limit=args.limit)))
        return 0
    if args.command == "evaluate-transfer":
        config = ExperimentConfig.from_json(args.config)
        store = RunStore(config.output_dir)
        transfer_batch = build_real_transfer_evaluation_batch(
            store.load_batch(args.reference_run_id),
            store.load_batch(args.run_id),
        )
        result = _timed_detection(
            transfer_batch,
            pca_dims=config.pca_dims,
            ridge_alpha=config.ridge_alpha,
            n_bootstrap=args.bootstrap,
            return_predictions=True,
        )
        _persist_detection_predictions(store, args.run_id, transfer_batch, result, "exact_error")
        samples = result.pop("bootstrap_samples")
        store.write_json(args.run_id, "bootstrap", "detection_auc_delta", samples)
        path = store.write_json(args.run_id, "metrics", "detection", result)
        _write_detection_figures(store, args.run_id, result)
        print(json.dumps({"metrics": str(path), "decision": result["decision"]}))
        return 0
    if args.command == "intervene-concept":
        config = ExperimentConfig.from_json(args.config)
        store = RunStore(config.output_dir)
        model, tokenizer = load_hf_model(config)
        try:
            result = run_concept_intervention_study(
                model,
                tokenizer,
                load_concept_examples(args.data),
                store.load_batch(args.baseline_run_id, label_key=args.endpoint),
                config=config,
                run_id=args.run_id,
                store=store,
                n_bootstrap=args.bootstrap,
                endpoint=args.endpoint,
            )
        finally:
            unload_model(model)
        print(json.dumps({"run_id": args.run_id, "decision": result["decision"]}))
        return 0
    if args.command == "extract-jailbreak":
        config = ExperimentConfig.from_json(args.config)
        examples = load_official_jailbreakbench(args.path)
        validate_jailbreak_study_set(examples)
        examples = _jailbreak_pilot(examples, args.pilot_pairs_per_category)
        model, tokenizer = load_hf_model(config)
        try:
            extract_jailbreak_run(
                model,
                tokenizer,
                examples,
                config=config,
                run_id=args.run_id,
                store=RunStore(config.output_dir),
                dataset_provenance=_dataset_provenance(args.path),
            )
        finally:
            unload_model(model)
        return 0
    if args.command == "judge-jailbreak":
        config = ExperimentConfig.from_json(args.config)
        guard, tokenizer = load_hf_model(config)
        try:
            result = judge_jailbreak_run(
                guard,
                tokenizer,
                run_id=args.run_id,
                store=RunStore(config.output_dir),
            )
            print(json.dumps(result))
        finally:
            unload_model(guard)
        return 0
    if args.command == "record-manual-audit":
        config = ExperimentConfig.from_json(args.config)
        result = record_manual_audit(args.path, run_id=args.run_id, store=RunStore(config.output_dir))
        print(json.dumps(result))
        return 0
    if args.command == "evaluate-jailbreak":
        config = ExperimentConfig.from_json(args.config)
        store = RunStore(config.output_dir)
        batch = store.load_batch(args.run_id)
        categories = [
            str(store.read(args.run_id, example_id).provenance["category"])
            for example_id in batch.example_ids
        ]
        result = evaluate_grouped_jailbreak_methods(
            batch,
            categories,
            pca_dims=config.pca_dims,
            ridge_alpha=config.ridge_alpha,
            n_bootstrap=args.bootstrap,
        )
        runs = [store.read(args.run_id, example_id) for example_id in batch.example_ids]
        rates = safety_rates(
            [bool(run.label) for run in runs],
            [is_refusal(run.response) for run in runs],
            [bool(run.provenance["benign"]) for run in runs],
        )
        result["operational_rates"] = asdict(rates)
        result["manual_audit_completed"] = (
            store.root / args.run_id / "labels" / "manual_audit_completed.json"
        ).exists()
        if result.get("aggregate_oof") is not None:
            store.write_json(
                args.run_id,
                "bootstrap",
                "grouped_detection_auc_delta",
                result["aggregate_oof"]["bootstrap_samples"],
            )
        path = store.write_json(args.run_id, "metrics", "grouped_detection", result)
        print(json.dumps({"metrics": str(path), "folds": result["total_folds"]}))
        return 0
    if args.command == "prepare-jailbreak-intervention-validation":
        config = ExperimentConfig.from_json(args.config)
        examples = load_official_jailbreakbench(args.path)
        validate_jailbreak_study_set(examples)
        store = RunStore(config.output_dir)
        model, tokenizer = load_hf_model(config)
        try:
            result = generate_jailbreak_intervention_validation(
                model,
                tokenizer,
                examples,
                store.load_batch(args.baseline_run_id),
                config=config,
                run_id=args.run_id,
                baseline_run_id=args.baseline_run_id,
                store=store,
            )
        finally:
            unload_model(model)
        print(json.dumps(result))
        return 0
    if args.command == "select-jailbreak-intervention":
        config = ExperimentConfig.from_json(args.config)
        result = select_jailbreak_intervention(
            validation_run_id=args.validation_run_id,
            baseline_run_id=args.baseline_run_id,
            store=RunStore(config.output_dir),
        )
        print(json.dumps(result["selected"]))
        return 0
    if args.command == "prepare-jailbreak-intervention-test":
        config = ExperimentConfig.from_json(args.config)
        examples = load_official_jailbreakbench(args.path)
        validate_jailbreak_study_set(examples)
        store = RunStore(config.output_dir)
        model, tokenizer = load_hf_model(config)
        try:
            result = generate_jailbreak_intervention_test(
                model,
                tokenizer,
                examples,
                store.load_batch(args.baseline_run_id),
                config=config,
                run_id=args.run_id,
                validation_run_id=args.validation_run_id,
                baseline_run_id=args.baseline_run_id,
                store=store,
            )
        finally:
            unload_model(model)
        print(json.dumps(result))
        return 0
    if args.command == "evaluate-jailbreak-intervention":
        config = ExperimentConfig.from_json(args.config)
        result = evaluate_jailbreak_intervention(
            run_id=args.run_id,
            store=RunStore(config.output_dir),
            n_bootstrap=args.bootstrap,
            seed=config.seed,
        )
        print(json.dumps({"run_id": args.run_id, "decision": result["decision"]}))
        return 0
    if args.command == "report-study":
        config = ExperimentConfig.from_json(args.config)
        protected_results = Path(__file__).resolve().parents[2] / "docs" / "results.md"
        if _same_output_file(Path(args.output), protected_results):
            raise ValueError("report-study must not overwrite repository docs/results.md")
        path = write_study_report(RunStore(config.output_dir), args.output)
        print(json.dumps({"report": str(path)}))
        return 0
    if args.command == "prepare-circuit-followup":
        config = ExperimentConfig.from_json(args.config)
        result = prepare_circuit_followup(
            RunStore(config.output_dir),
            run_id=args.run_id,
            endpoint=args.endpoint,
            per_stratum=args.per_stratum,
        )
        print(json.dumps({"selected": len(result["cases"]), "counts": result["counts"]}))
        return 0
    if args.command == "rlmf-prepare-data":
        config = RLMFConfig.from_json(args.config)
        paths = write_popqa_snapshot(config, RLMFArtifactStore(args.root))
        print(
            json.dumps(
                {
                    "study_id": config.study_id,
                    "count": sum(config.split_counts.values()),
                    "split_counts": dict(config.split_counts),
                    "dataset_revision": config.dataset_revision,
                    "artifacts": {name: str(path) for name, path in paths.items()},
                }
            )
        )
        return 0
    return 1


def _same_output_file(candidate: Path, protected: Path) -> bool:
    if candidate.exists() and protected.exists():
        return candidate.samefile(protected)
    return candidate.resolve() == protected.resolve()


def _timed_detection(batch, **kwargs):
    started = time.perf_counter()
    result = evaluate_detection_methods(batch, **kwargs)
    result["runtime"] = {
        "seconds": time.perf_counter() - started,
        "max_resident_set_size": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    return result


def _write_detection_figures(
    store: RunStore, run_id: str, result: dict, *, suffix: str = "primary"
) -> None:
    from trajectory_extractor.plotting import plot_method_comparison, plot_prefix_surface

    directory = store.root / run_id / "figures"
    plot_method_comparison(
        {name: values["test_auroc"] for name, values in result["methods"].items()},
        directory / f"method_comparison_{suffix}.png",
    )
    for name, values in result["methods"].items():
        plot_prefix_surface(
            np.asarray(values["validation_surface"]["auroc"]),
            directory / f"{name}_validation_auroc_{suffix}.png",
            title=f"{name}: validation AUROC",
        )


def _write_secondary_figures(
    directory: Path,
    result: dict,
    arrays: dict[str, np.ndarray],
    *,
    endpoint: str,
) -> tuple[Path, Path]:
    from trajectory_extractor.plotting import plot_class_risk_gap, plot_method_comparison

    ensure_durable_directory(directory)
    comparison = plot_method_comparison(
        {name: values["test_auroc"] for name, values in result["methods"].items()},
        directory / f"method_comparison_{endpoint}.png",
    )
    risk_gap = plot_class_risk_gap(
        arrays["validation_metacognitive_risk_surface"],
        arrays["validation_labels"],
        directory / f"validation_metacognitive_risk_gap_{endpoint}.png",
        title="Validation metacognitive risk gap",
    )
    return comparison, risk_gap


def _secondary_analysis_provenance(
    *,
    config: ExperimentConfig,
    config_path: str | Path,
    run_id: str,
    endpoint: str,
    bootstrap_count: int,
    result: dict,
) -> dict:
    comparison = result.get("registered_comparison")
    if not isinstance(comparison, dict):
        raise ValueError("secondary result is missing registered comparison provenance")
    required = ("p_value_method", "permutation_seed", "n_permutations", "fdr_family")
    if any(key not in comparison for key in required):
        raise ValueError("secondary result has incomplete permutation provenance")
    fdr_family = comparison["fdr_family"]
    if not isinstance(fdr_family, list):
        raise ValueError("secondary FDR family must be an ordered list")
    repo_root = Path(__file__).resolve().parents[2]
    return build_analysis_provenance(
        repo_root=repo_root,
        preregistration_path=repo_root / "docs" / "secondary_preregistration.md",
        config_path=Path(config_path).resolve(),
        run_root=(Path(config.output_dir) / run_id).resolve(),
        endpoint=endpoint,
        pca_dims=config.pca_dims,
        ridge_alpha=config.ridge_alpha,
        bootstrap_count=bootstrap_count,
        permutation_method=comparison["p_value_method"],
        permutation_seed=comparison["permutation_seed"],
        permutation_count=comparison["n_permutations"],
        fdr_family=fdr_family,
    )


def _persist_detection_predictions(
    store: RunStore,
    run_id: str,
    batch,
    result: dict,
    endpoint: str,
) -> None:
    test_indices = np.asarray(result.pop("_test_indices"), dtype=int)
    predictions = result.pop("_test_predictions")
    simple = str(result["selected_simple_baseline"])
    dynamic = str(result["selected_dynamics_method"])
    rows = []
    for position, batch_index in enumerate(test_indices):
        rows.append(
            {
                "example_id": batch.example_ids[batch_index],
                "label": int(batch.labels[batch_index]),
                "selected_simple_method": simple,
                "selected_dynamics_method": dynamic,
                "selected_simple_score": float(predictions[simple][position]),
                "selected_dynamics_score": float(predictions[dynamic][position]),
                "method_scores": {
                    method: float(values[position]) for method, values in predictions.items()
                },
            }
        )
    store.write_json(run_id, "labels", f"detection_predictions_{endpoint}", rows)


def _batch_metadata(store: RunStore, run_id: str, batch) -> list[dict]:
    rows = []
    lengths = []
    runs = [store.read(run_id, example_id) for example_id in batch.example_ids]
    for run in runs:
        lengths.append(len(run.prompt.split()))
    low, high = np.quantile(lengths, [1 / 3, 2 / 3])
    for run, length in zip(runs, lengths, strict=True):
        row = dict(run.provenance)
        row["prompt_length_bin"] = "short" if length <= low else "medium" if length <= high else "long"
        rows.append(row)
    return rows


def _concept_pilot(
    examples: list[ConceptMixingExample], per_split: int | None
) -> list[ConceptMixingExample]:
    if per_split is None:
        return examples
    if per_split < 1:
        raise ValueError("pilot-per-split must be positive")
    counts: dict[str, int] = {}
    selected = []
    for example in examples:
        if counts.get(example.split, 0) >= per_split:
            continue
        selected.append(example)
        counts[example.split] = counts.get(example.split, 0) + 1
    return selected


def _jailbreak_pilot(
    examples: list[JailbreakExample], pairs_per_category: int | None
) -> list[JailbreakExample]:
    if pairs_per_category is None:
        return examples
    if pairs_per_category < 1:
        raise ValueError("pilot-pairs-per-category must be positive")
    selected_pairs: dict[str, list[str]] = {}
    for example in examples:
        pair_id = str(example.pair_id)
        category_pairs = selected_pairs.setdefault(example.category, [])
        if pair_id not in category_pairs and len(category_pairs) < pairs_per_category:
            category_pairs.append(pair_id)
    allowed = {pair_id for pair_ids in selected_pairs.values() for pair_id in pair_ids}
    return [example for example in examples if str(example.pair_id) in allowed]


def _dataset_provenance(path: str | Path) -> dict:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    result = {"path": str(source), "sha256": _sha256(source)}
    manifest = source.with_suffix(source.suffix + ".manifest.json")
    if manifest.exists():
        result.update({"manifest_path": str(manifest), "manifest_sha256": _sha256(manifest)})
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
