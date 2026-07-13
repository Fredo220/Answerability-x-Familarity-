from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from dataclasses import asdict, replace
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
from trajectory_extractor.rlmf_artifacts import RLMFArtifactStore, sha256_file
from trajectory_extractor.rlmf_data import write_popqa_snapshot
from trajectory_extractor.rlmf_format import (
    NORMALIZATION_VERSION,
    PARSER_VERSION,
    REGISTERED_AUDIT_SEED,
    AuditRow,
    audit_sampling_design,
    build_judge_audit_sample,
    estimate_arm_confusion_uncertainty,
    score_blinded_judge_audit,
)
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

    rlmf_build_audit = subparsers.add_parser("rlmf-build-judge-audit")
    rlmf_build_audit.add_argument("--config", required=True)
    rlmf_build_audit.add_argument("--phase", choices=("development", "locked", "test"), required=True)
    rlmf_build_audit.add_argument("--root", default=".")
    rlmf_build_audit.add_argument(
        "--seed", type=int, choices=(REGISTERED_AUDIT_SEED,), default=REGISTERED_AUDIT_SEED
    )
    rlmf_build_audit.add_argument("--size", type=int)

    rlmf_record_audit = subparsers.add_parser("rlmf-record-judge-audit")
    rlmf_record_audit.add_argument("path")
    rlmf_record_audit.add_argument("--config", required=True)
    rlmf_record_audit.add_argument("--phase", choices=("development", "locked", "test"), required=True)
    rlmf_record_audit.add_argument("--root", default=".")
    rlmf_record_audit.add_argument("--size", type=int)

    rlmf_seal_rating = subparsers.add_parser(
        "rlmf-seal-judge-rating",
        help="Seal one independent rater source before adjudication",
    )
    rlmf_seal_rating.add_argument("path")
    rlmf_seal_rating.add_argument("--identity", required=True)
    rlmf_seal_rating.add_argument("--role", choices=("rater_a", "rater_b"), required=True)
    rlmf_seal_rating.add_argument("--config", required=True)
    rlmf_seal_rating.add_argument(
        "--phase", choices=("development", "locked", "test"), required=True
    )
    rlmf_seal_rating.add_argument("--root", default=".")
    rlmf_seal_rating.add_argument("--size", type=int)

    rlmf_seal_adjudication = subparsers.add_parser(
        "rlmf-seal-judge-adjudication",
        help="Seal disagreement-only adjudication after both rater endpoints",
    )
    rlmf_seal_adjudication.add_argument("path")
    rlmf_seal_adjudication.add_argument("--identity", required=True)
    rlmf_seal_adjudication.add_argument("--config", required=True)
    rlmf_seal_adjudication.add_argument(
        "--phase", choices=("development", "locked", "test"), required=True
    )
    rlmf_seal_adjudication.add_argument("--root", default=".")
    rlmf_seal_adjudication.add_argument("--size", type=int)

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
    if args.command == "rlmf-build-judge-audit":
        config = RLMFConfig.from_json(args.config)
        store = RLMFArtifactStore(args.root)
        return _build_rlmf_judge_audit(args, config, store)
    if args.command == "rlmf-record-judge-audit":
        config = RLMFConfig.from_json(args.config)
        store = RLMFArtifactStore(args.root)
        return _record_rlmf_judge_audit(args, config, store)
    if args.command == "rlmf-seal-judge-rating":
        config = RLMFConfig.from_json(args.config)
        return _seal_rlmf_judge_rating(args, config, RLMFArtifactStore(args.root))
    if args.command == "rlmf-seal-judge-adjudication":
        config = RLMFConfig.from_json(args.config)
        return _seal_rlmf_judge_adjudication(
            args, config, RLMFArtifactStore(args.root)
        )
    return 1


def _same_output_file(candidate: Path, protected: Path) -> bool:
    if candidate.exists() and protected.exists():
        return candidate.samefile(protected)
    return candidate.resolve() == protected.resolve()


def _registered_audit_size(phase: str) -> int:
    return {"development": 200, "locked": 400, "test": 1000}[phase]


def _audit_name_suffix(phase: str, size: int) -> str:
    del phase
    return f"_{size}"


def _rlmf_audit_path(root: str | Path, study_id: str, section: str, name: str) -> Path:
    return Path(root) / "runs" / "rlmf" / study_id / section / f"{name}.jsonl"


def _read_jsonl(path: str | Path) -> tuple[dict, ...]:
    source = Path(path)
    try:
        rows = tuple(json.loads(line) for line in source.read_text().splitlines() if line.strip())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSONL audit input: {source}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("audit JSONL must contain object rows")
    return rows


def _build_rlmf_judge_audit(args, config: RLMFConfig, store: RLMFArtifactStore) -> int:
    size = args.size or _registered_audit_size(args.phase)
    candidate_endpoint = f"audit_candidates_{args.phase}"
    candidate_relative = f"evaluation/audit_candidates_{args.phase}.jsonl"
    try:
        candidates_path, candidate_marker, candidate_record = _verified_endpoint_artifact(
            store, config.study_id, candidate_endpoint, candidate_relative
        )
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            f"verified candidate endpoint {candidate_endpoint} is required"
        ) from error

    aliases_path, aliases_marker, _aliases_record = _verified_endpoint_artifact(
        store, config.study_id, "prepare-data", "data/aliases.jsonl"
    )
    parser_path = Path(__file__).with_name("rlmf_format.py")
    parser_hash = sha256_file(parser_path)
    proxy_freeze = _proxy_freeze(
        parser_hash=parser_hash,
        aliases_path=aliases_path,
        aliases_marker=aliases_marker,
        candidate_record=candidate_record,
    )
    development_marker: Path | None = None
    if args.phase == "locked":
        development_marker, development_record, development_metadata = (
            _verified_development_audit(store, config.study_id)
        )
        development_hash = sha256_file(development_marker)
        if candidate_record["parent_hashes"].get("development_judge_audit") != development_hash:
            raise ValueError(
                "locked candidate endpoint must bind the verified development_judge_audit marker"
            )
        if _parse_timestamp(candidate_record["created_at"]) <= _parse_timestamp(
            development_record["created_at"]
        ):
            raise ValueError("locked candidate endpoint must postdate development_judge_audit")
        if development_metadata.get("proxy_freeze") != proxy_freeze:
            raise ValueError("locked sample does not match the development proxy freeze")
    locked_marker: Path | None = None
    if args.phase == "test":
        locked_marker, locked_record = _verified_locked_audit(store, config.study_id)
        locked_hash = sha256_file(locked_marker)
        if candidate_record["parent_hashes"].get("locked_judge_audit") != locked_hash:
            raise ValueError(
                "test candidate endpoint must bind the verified locked_judge_audit marker"
            )
        if _parse_timestamp(candidate_record["created_at"]) < _parse_timestamp(
            locked_record["created_at"]
        ):
            raise ValueError("test candidate endpoint predates the locked_judge_audit")

    aliases_by_example = _load_frozen_aliases(aliases_path)
    candidates = _bind_candidate_aliases(
        _read_jsonl(candidates_path), aliases_by_example
    )
    selected = build_judge_audit_sample(
        candidates, phase=args.phase, size=size, seed=args.seed
    )

    parent_sample_hash: str | None = None
    extension_request_hash: str | None = None
    pending_rows = selected
    ledger_rows = selected
    if args.phase == "test" and size > 1000:
        previous_size = size - 250
        previous_endpoint = f"test_judge_audit_evidence_{previous_size}"
        previous_relative = f"audits/test_{previous_size}_completed.jsonl"
        previous_path, previous_marker, previous_record = _verified_endpoint_artifact(
            store, config.study_id, previous_endpoint, previous_relative
        )
        request_endpoint = f"test_judge_audit_extension_request_{size}"
        request_relative = f"audits/test_{size}_extension_request.json"
        try:
            request_path, request_marker, request_record = _verified_endpoint_artifact(
                store, config.study_id, request_endpoint, request_relative
            )
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                f"verified Task 5/10 extension request {request_endpoint} is required"
            ) from error
        previous_marker_hash = sha256_file(previous_marker)
        if (
            request_record["parent_hashes"].get("test_judge_audit_evidence")
            != previous_marker_hash
        ):
            raise ValueError("extension request does not bind the prior test audit evidence")
        request = _read_json(request_path)
        required_request = {
            "status": "extension_required",
            "estimand": "delta_cMFG_star",
            "from_size": previous_size,
            "requested_size": size,
        }
        if any(request.get(key) != value for key, value in required_request.items()):
            raise ValueError("extension request has an invalid endpoint-propagation contract")
        if _parse_timestamp(request_record["created_at"]) < _parse_timestamp(
            previous_record["created_at"]
        ):
            raise ValueError("extension request predates the prior test audit evidence")
        extension_request_hash = sha256_file(request_marker)
        previous_rows = tuple(
            AuditRow.from_ledger_record(row) for row in _read_jsonl(previous_path)
        )
        selected_by_source = {row.source_id: row for row in selected}
        for previous in previous_rows:
            current = selected_by_source.get(previous.source_id)
            if current is None or not _same_audit_identity(previous, current):
                raise ValueError("test audit extension changed a sealed sample row")
        previous_sources = {row.source_id for row in previous_rows}
        pending_rows = tuple(
            row for row in selected if row.source_id not in previous_sources
        )
        if len(pending_rows) != 250:
            raise ValueError("test audit extension must append exactly 250 rows")
        ledger_rows = (*previous_rows, *pending_rows)
        parent_sample_hash = sha256_file(previous_path)

    suffix = _audit_name_suffix(args.phase, size)
    sample_name = f"{args.phase}{suffix}_sample"
    ledger_name = f"{args.phase}{suffix}_ledger"
    metadata_name = f"{args.phase}{suffix}_metadata"
    metadata_path = _rlmf_audit_path(
        args.root, config.study_id, "audits", metadata_name
    ).with_suffix(".json")
    created_at = datetime.now(timezone.utc).isoformat()
    if metadata_path.exists():
        existing_metadata = _read_json(metadata_path)
        created_at = existing_metadata.get("created_at", created_at)
    metadata = {
        "schema_version": 1,
        "created_at": created_at,
        "phase": args.phase,
        "size": size,
        "append_rows": len(pending_rows),
        "sampling_seed": args.seed,
        "parser_version": PARSER_VERSION,
        "parser_source_hash": parser_hash,
        "normalization_version": NORMALIZATION_VERSION,
        "alias_artifact_hash": sha256_file(aliases_path),
        "alias_endpoint_marker_hash": sha256_file(aliases_marker),
        "proxy_freeze": proxy_freeze,
        "sampling_design": audit_sampling_design(selected),
        "candidate_endpoint": candidate_endpoint,
        "candidate_marker_hash": sha256_file(candidate_marker),
        "parent_sample_hash": parent_sample_hash,
        "extension_request_marker_hash": extension_request_hash,
        "development_judge_audit_marker_hash": (
            sha256_file(development_marker) if development_marker is not None else None
        ),
    }
    sample_path = _write_jsonl_resumable(
        store,
        config.study_id,
        "audits",
        sample_name,
        tuple(row.rater_payload() for row in pending_rows),
    )
    ledger_path = _write_jsonl_resumable(
        store,
        config.study_id,
        "audits",
        ledger_name,
        tuple(row.to_ledger_record() for row in ledger_rows),
    )
    metadata_path = _write_json_resumable(
        store, config.study_id, "audits", metadata_name, metadata
    )
    parents = {
        "candidate_marker": sha256_file(candidate_marker),
        "alias_artifact": sha256_file(aliases_path),
        "parser_source": parser_hash,
    }
    if development_marker is not None:
        parents["development_judge_audit"] = sha256_file(development_marker)
    if parent_sample_hash is not None:
        parents["parent_sample"] = parent_sample_hash
    if extension_request_hash is not None:
        parents["extension_request"] = extension_request_hash
    sample_marker = _complete_endpoint_resumable(
        store,
        config.study_id,
        f"{args.phase}_judge_audit_sample_{size}",
        config,
        (sample_path, ledger_path, metadata_path),
        parent_hashes=parents,
    )
    print(
        json.dumps(
            {
                "phase": args.phase,
                "sample": str(sample_path),
                "ledger": str(ledger_path),
                "metadata": str(metadata_path),
                "marker": str(sample_marker),
                "rows": len(pending_rows),
                "cumulative_size": len(ledger_rows),
            }
        )
    )
    return 0


def _record_rlmf_judge_audit(
    args, config: RLMFConfig, store: RLMFArtifactStore
) -> int:
    size = args.size or _registered_audit_size(args.phase)
    suffix = _audit_name_suffix(args.phase, size)
    sample_endpoint = f"{args.phase}_judge_audit_sample_{size}"
    ledger_relative = f"audits/{args.phase}{suffix}_ledger.jsonl"
    ledger_path, sample_marker, sample_record = _verified_endpoint_artifact(
        store, config.study_id, sample_endpoint, ledger_relative
    )
    metadata_path = _study_namespace(store, config.study_id) / (
        f"audits/{args.phase}{suffix}_metadata.json"
    )
    metadata = _read_json(metadata_path)
    parser_hash = sha256_file(Path(__file__).with_name("rlmf_format.py"))
    if metadata.get("phase") != args.phase or metadata.get("size") != size:
        raise ValueError("audit metadata does not match the requested phase and size")
    if (
        metadata.get("parser_version") != PARSER_VERSION
        or metadata.get("normalization_version") != NORMALIZATION_VERSION
        or metadata.get("parser_source_hash") != parser_hash
    ):
        raise ValueError("audit parser and normalization freeze no longer verifies")
    aliases_path, _alias_marker, _alias_record = _verified_endpoint_artifact(
        store, config.study_id, "prepare-data", "data/aliases.jsonl"
    )
    if metadata.get("alias_artifact_hash") != sha256_file(aliases_path):
        raise ValueError("audit alias artifact no longer verifies")
    ledger_rows = tuple(
        AuditRow.from_ledger_record(row) for row in _read_jsonl(ledger_path)
    )
    if metadata.get("sampling_design") != audit_sampling_design(ledger_rows):
        raise ValueError("audit sampling design no longer verifies")
    development_marker: Path | None = None
    if args.phase == "locked":
        development_marker, _development_record, development_metadata = (
            _verified_development_audit(store, config.study_id)
        )
        development_hash = sha256_file(development_marker)
        if (
            metadata.get("development_judge_audit_marker_hash") != development_hash
            or sample_record["parent_hashes"].get("development_judge_audit")
            != development_hash
        ):
            raise ValueError("locked audit sample does not bind development_judge_audit")
        if metadata.get("proxy_freeze") != development_metadata.get("proxy_freeze"):
            raise ValueError("locked audit changed the development proxy freeze")
    if args.phase == "test":
        _verified_locked_audit(store, config.study_id)

    pending_rows = tuple(
        row for row in ledger_rows if row.rater_a is None and row.rater_b is None
    )
    if not pending_rows:
        raise ValueError("audit ledger contains no pending independent ratings")
    ratings, source_metadata = _load_rating_manifest(
        Path(args.path),
        pending_rows,
        store=store,
        study_id=config.study_id,
        phase=args.phase,
        size=size,
        sample_marker=sample_marker,
        sample_record=sample_record,
    )
    completed_rows = tuple(
        replace(
            row,
            rater_a=ratings["rater_a"].get(row.audit_id, row.rater_a),
            rater_b=ratings["rater_b"].get(row.audit_id, row.rater_b),
            adjudicated_label=ratings["adjudication"].get(
                row.audit_id, row.adjudicated_label
            ),
        )
        for row in ledger_rows
    )
    decision = score_blinded_judge_audit(completed_rows)

    persisted_manifest = {
        "schema_version": 2,
        "input_manifest_sha256": sha256_file(Path(args.path)),
        "sources": source_metadata,
    }
    sources_path = _write_json_resumable(
        store,
        config.study_id,
        "audits",
        f"{args.phase}{suffix}_rating_sources",
        persisted_manifest,
    )
    completed_path = _write_jsonl_resumable(
        store,
        config.study_id,
        "audits",
        f"{args.phase}{suffix}_completed",
        tuple(row.to_ledger_record() for row in completed_rows),
    )
    decision_path = _write_json_resumable(
        store,
        config.study_id,
        "audits",
        f"{args.phase}{suffix}_decision",
        decision.to_record(),
    )
    bound_paths = (
        completed_path,
        decision_path,
        sources_path,
    )
    endpoint_parents = {
        "audit_sample": sha256_file(sample_marker),
        "parser_source": parser_hash,
        "alias_artifact": sha256_file(aliases_path),
        "rating_manifest": sha256_file(Path(args.path)),
        **{
            f"{name}_endpoint": source_metadata[name]["marker_sha256"]
            for name in ("rater_a", "rater_b", "adjudication")
        },
    }
    if development_marker is not None:
        endpoint_parents["development_judge_audit"] = sha256_file(
            development_marker
        )

    if args.phase != "development" and decision.passed is not True:
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "status": "failed",
                    "passed": False,
                    "completed": str(completed_path),
                    "decision": str(decision_path),
                }
            )
        )
        return 2
    if args.phase == "development":
        marker = _complete_endpoint_resumable(
            store,
            config.study_id,
            "development_judge_audit",
            config,
            bound_paths,
            parent_hashes=endpoint_parents,
        )
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "status": decision.status,
                    "completed": str(completed_path),
                    "marker": str(marker),
                }
            )
        )
        return 0
    if args.phase == "locked":
        marker = _complete_endpoint_resumable(
            store,
            config.study_id,
            "locked_judge_audit",
            config,
            bound_paths,
            parent_hashes=endpoint_parents,
        )
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "status": decision.status,
                    "passed": True,
                    "completed": str(completed_path),
                    "marker": str(marker),
                }
            )
        )
        return 0

    confusion_path = _write_json_resumable(
        store,
        config.study_id,
        "audits",
        f"test{suffix}_confusion_uncertainty",
        estimate_arm_confusion_uncertainty(completed_rows),
    )
    evidence_marker = _complete_endpoint_resumable(
        store,
        config.study_id,
        f"test_judge_audit_evidence_{size}",
        config,
        (*bound_paths, confusion_path),
        parent_hashes=endpoint_parents,
    )
    print(
        json.dumps(
            {
                "phase": "test",
                "status": "endpoint_propagation_required",
                "completed": str(completed_path),
                "confusion_uncertainty": str(confusion_path),
                "evidence_marker": str(evidence_marker),
            }
        )
    )
    # Task 5/10 must propagate this evidence through delta_cMFG_star before the
    # audit can be finalized or an extension can be requested.
    return 3


def _seal_rlmf_judge_rating(
    args, config: RLMFConfig, store: RLMFArtifactStore
) -> int:
    size = args.size or _registered_audit_size(args.phase)
    pending_rows, sample_marker, sample_record = _pending_audit_rows(
        store, config.study_id, args.phase, size
    )
    source = _resolved_source_path(Path(args.path), args.role)
    rows = _read_rating_rows(source, allow_empty=False)
    _require_exact_rating_ids(rows, pending_rows, args.role)
    identity = _validated_identity(args.identity, args.role)
    source_hash = sha256_file(source)
    if datetime.now(timezone.utc) <= _parse_timestamp(sample_record["created_at"]):
        raise ValueError("independent rater endpoint must postdate the sample marker")

    other_role = "rater_b" if args.role == "rater_a" else "rater_a"
    try:
        _other_rows, _other_marker, _other_record, other_metadata = (
            _verified_sealed_source(
                store, config.study_id, args.phase, size, other_role
            )
        )
    except (FileNotFoundError, ValueError):
        other_metadata = None
    if other_metadata is not None:
        if identity == other_metadata["identity"]:
            raise ValueError("independent raters must have distinct identities")
        if str(source) == other_metadata["resolved_source_path"]:
            raise ValueError("independent raters must use distinct resolved source paths")
        if source_hash == other_metadata["input_sha256"]:
            raise ValueError("independent raters must use distinct source hashes")

    suffix = _audit_name_suffix(args.phase, size)
    rows_path = _write_jsonl_resumable(
        store,
        config.study_id,
        "audits",
        f"{args.phase}{suffix}_{args.role}_sealed",
        rows,
    )
    metadata = {
        "schema_version": 1,
        "role": args.role,
        "identity": identity,
        "phase": args.phase,
        "size": size,
        "resolved_source_path": str(source),
        "input_sha256": source_hash,
        "sample_marker_sha256": sha256_file(sample_marker),
    }
    metadata_path = _write_json_resumable(
        store,
        config.study_id,
        "audits",
        f"{args.phase}{suffix}_{args.role}_source",
        metadata,
    )
    endpoint = f"{args.phase}_judge_audit_{args.role}_{size}"
    marker = _complete_endpoint_resumable(
        store,
        config.study_id,
        endpoint,
        config,
        (rows_path, metadata_path),
        parent_hashes={"audit_sample": sha256_file(sample_marker)},
    )
    marker_record = store.verify_endpoint(config.study_id, endpoint)
    if _parse_timestamp(marker_record["created_at"]) <= _parse_timestamp(
        sample_record["created_at"]
    ):
        raise ValueError("independent rater endpoint must postdate the sample marker")
    print(json.dumps({"role": args.role, "endpoint": endpoint, "marker": str(marker)}))
    return 0


def _seal_rlmf_judge_adjudication(
    args, config: RLMFConfig, store: RLMFArtifactStore
) -> int:
    size = args.size or _registered_audit_size(args.phase)
    pending_rows, sample_marker, sample_record = _pending_audit_rows(
        store, config.study_id, args.phase, size
    )
    sealed = {}
    for role in ("rater_a", "rater_b"):
        try:
            sealed[role] = _verified_sealed_source(
                store, config.study_id, args.phase, size, role
            )
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"verified {role} endpoint is required") from error
    _validate_rater_endpoints(sealed, sample_marker, sample_record, pending_rows)
    rater_completion_times = [
        _parse_timestamp(sealed[role][2]["created_at"])
        for role in ("rater_a", "rater_b")
    ]
    if datetime.now(timezone.utc) <= max(rater_completion_times):
        raise ValueError("adjudication endpoint must postdate both rater markers")
    ratings = {
        role: {row["audit_id"]: row["label"] for row in sealed[role][0]}
        for role in ("rater_a", "rater_b")
    }
    disagreements = {
        audit_id
        for audit_id in ratings["rater_a"]
        if ratings["rater_a"][audit_id] != ratings["rater_b"][audit_id]
    }
    source = _resolved_source_path(Path(args.path), "adjudication")
    rows = _read_rating_rows(source, allow_empty=True)
    mapping = {row["audit_id"]: row["label"] for row in rows}
    if len(mapping) != len(rows) or set(mapping) != disagreements:
        raise ValueError("adjudication must address exactly the rater disagreements")
    identity = _validated_identity(args.identity, "adjudication")
    if identity in {sealed[role][3]["identity"] for role in sealed}:
        raise ValueError("raters and adjudicator must have distinct identities")

    suffix = _audit_name_suffix(args.phase, size)
    rows_path = _write_jsonl_resumable(
        store,
        config.study_id,
        "audits",
        f"{args.phase}{suffix}_adjudication_sealed",
        rows,
    )
    metadata = {
        "schema_version": 1,
        "role": "adjudication",
        "identity": identity,
        "phase": args.phase,
        "size": size,
        "resolved_source_path": str(source),
        "input_sha256": sha256_file(source),
        "sample_marker_sha256": sha256_file(sample_marker),
        "rater_a_marker_sha256": sha256_file(sealed["rater_a"][1]),
        "rater_b_marker_sha256": sha256_file(sealed["rater_b"][1]),
    }
    metadata_path = _write_json_resumable(
        store,
        config.study_id,
        "audits",
        f"{args.phase}{suffix}_adjudication_source",
        metadata,
    )
    endpoint = f"{args.phase}_judge_audit_adjudication_{size}"
    marker = _complete_endpoint_resumable(
        store,
        config.study_id,
        endpoint,
        config,
        (rows_path, metadata_path),
        parent_hashes={
            "audit_sample": sha256_file(sample_marker),
            "rater_a": sha256_file(sealed["rater_a"][1]),
            "rater_b": sha256_file(sealed["rater_b"][1]),
        },
    )
    marker_record = store.verify_endpoint(config.study_id, endpoint)
    if _parse_timestamp(marker_record["created_at"]) <= max(
        _parse_timestamp(sealed[role][2]["created_at"])
        for role in ("rater_a", "rater_b")
    ):
        raise ValueError("adjudication endpoint must postdate both rater markers")
    print(json.dumps({"role": "adjudication", "endpoint": endpoint, "marker": str(marker)}))
    return 0


def _verified_endpoint_artifact(
    store: RLMFArtifactStore, study_id: str, endpoint: str, relative: str
) -> tuple[Path, Path, dict]:
    record = store.verify_endpoint(study_id, endpoint)
    if relative not in record["artifact_hashes"]:
        raise ValueError(f"endpoint {endpoint} does not bind {relative}")
    namespace = _study_namespace(store, study_id)
    path = namespace / relative
    if sha256_file(path) != record["artifact_hashes"][relative]:
        raise ValueError(f"endpoint {endpoint} artifact hash mismatch")
    marker = namespace / "endpoints" / f"{endpoint}.complete.json"
    return path, marker, record


def _proxy_freeze(
    *, parser_hash: str, aliases_path: Path, aliases_marker: Path, candidate_record: dict
) -> dict:
    return {
        "schema_version": 1,
        "parser_version": PARSER_VERSION,
        "parser_source_hash": parser_hash,
        "normalization_version": NORMALIZATION_VERSION,
        "alias_artifact_hash": sha256_file(aliases_path),
        "alias_endpoint_marker_hash": sha256_file(aliases_marker),
        "candidate_proxy_parent_hashes": {
            name: digest
            for name, digest in sorted(candidate_record["parent_hashes"].items())
            if name.startswith("proxy_")
        },
    }


def _verified_development_audit(
    store: RLMFArtifactStore, study_id: str
) -> tuple[Path, dict, dict]:
    try:
        decision_path, marker, record = _verified_endpoint_artifact(
            store,
            study_id,
            "development_judge_audit",
            "audits/development_200_decision.json",
        )
        metadata_path, sample_marker, sample_record = _verified_endpoint_artifact(
            store,
            study_id,
            "development_judge_audit_sample_200",
            "audits/development_200_metadata.json",
        )
    except (FileNotFoundError, ValueError) as error:
        raise ValueError("verified development_judge_audit is required") from error
    decision = _read_json(decision_path)
    if decision.get("phase") != "development" or decision.get("status") != "development_review":
        raise ValueError("development_judge_audit is not the final development review")
    if record["parent_hashes"].get("audit_sample") != sha256_file(sample_marker):
        raise ValueError("development_judge_audit does not bind its sealed sample")
    if _parse_timestamp(record["created_at"]) < _parse_timestamp(sample_record["created_at"]):
        raise ValueError("development_judge_audit predates its sealed sample")
    metadata = _read_json(metadata_path)
    if not isinstance(metadata.get("proxy_freeze"), dict):
        raise ValueError("development_judge_audit is missing its proxy freeze")
    return marker, record, metadata


def _verified_locked_audit(
    store: RLMFArtifactStore, study_id: str
) -> tuple[Path, dict]:
    try:
        decision_path, marker, record = _verified_endpoint_artifact(
            store,
            study_id,
            "locked_judge_audit",
            "audits/locked_400_decision.json",
        )
    except (FileNotFoundError, ValueError) as error:
        raise ValueError("verified locked_judge_audit is required") from error
    decision = _read_json(decision_path)
    if decision.get("phase") != "locked" or decision.get("passed") is not True:
        raise ValueError("locked_judge_audit did not pass")
    development_marker, development_record, _metadata = _verified_development_audit(
        store, study_id
    )
    if record["parent_hashes"].get("development_judge_audit") != sha256_file(
        development_marker
    ):
        raise ValueError("locked_judge_audit does not bind development_judge_audit")
    if _parse_timestamp(record["created_at"]) <= _parse_timestamp(
        development_record["created_at"]
    ):
        raise ValueError("locked_judge_audit must postdate development_judge_audit")
    return marker, record


def _load_frozen_aliases(path: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for row in _read_jsonl(path):
        if not {"example_id", "aliases"} <= set(row):
            raise ValueError("frozen alias artifact has an invalid schema")
        example_id = row.get("example_id")
        aliases = row.get("aliases")
        if not isinstance(example_id, str) or not example_id or example_id in result:
            raise ValueError("frozen alias example IDs must be unique strings")
        if (
            isinstance(aliases, (str, bytes))
            or not isinstance(aliases, list)
            or not aliases
            or any(not isinstance(alias, str) or not alias for alias in aliases)
        ):
            raise ValueError("frozen aliases must be non-empty string lists")
        result[example_id] = tuple(aliases)
    return result


def _bind_candidate_aliases(
    candidates: tuple[dict, ...], aliases_by_example: dict[str, tuple[str, ...]]
) -> tuple[dict, ...]:
    result = []
    for candidate in candidates:
        example_id = candidate.get("example_id")
        if example_id not in aliases_by_example:
            raise ValueError("audit candidate is not present in the frozen alias artifact")
        aliases = aliases_by_example[example_id]
        supplied = candidate.get("gold_aliases", candidate.get("aliases"))
        if supplied is not None and tuple(supplied) != aliases:
            raise ValueError("audit candidate aliases do not match the frozen alias artifact")
        result.append({**candidate, "gold_aliases": list(aliases)})
    return tuple(result)


def _same_audit_identity(left: AuditRow, right: AuditRow) -> bool:
    left_record = left.to_ledger_record()
    right_record = right.to_ledger_record()
    for field in ("rater_a", "rater_b", "adjudicated_label"):
        left_record[field] = None
        right_record[field] = None
    return left_record == right_record


def _pending_audit_rows(
    store: RLMFArtifactStore, study_id: str, phase: str, size: int
) -> tuple[tuple[AuditRow, ...], Path, dict]:
    suffix = _audit_name_suffix(phase, size)
    ledger_path, sample_marker, sample_record = _verified_endpoint_artifact(
        store,
        study_id,
        f"{phase}_judge_audit_sample_{size}",
        f"audits/{phase}{suffix}_ledger.jsonl",
    )
    rows = tuple(AuditRow.from_ledger_record(row) for row in _read_jsonl(ledger_path))
    pending = tuple(row for row in rows if row.rater_a is None and row.rater_b is None)
    if not pending:
        raise ValueError("audit ledger contains no pending independent ratings")
    return pending, sample_marker, sample_record


def _resolved_source_path(path: Path, role: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} source file is missing or a symlink")
    return path.resolve()


def _validated_identity(value: object, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} identity must be a non-empty string")
    return value.strip()


def _require_exact_rating_ids(
    rows: tuple[dict, ...], pending_rows: tuple[AuditRow, ...], role: str
) -> None:
    audit_ids = [row["audit_id"] for row in rows]
    expected_ids = {row.audit_id for row in pending_rows}
    if len(set(audit_ids)) != len(audit_ids) or set(audit_ids) != expected_ids:
        raise ValueError(f"{role} must contain exactly the pending audit IDs")


def _verified_sealed_source(
    store: RLMFArtifactStore, study_id: str, phase: str, size: int, role: str
) -> tuple[tuple[dict, ...], Path, dict, dict]:
    suffix = _audit_name_suffix(phase, size)
    endpoint = f"{phase}_judge_audit_{role}_{size}"
    rows_relative = f"audits/{phase}{suffix}_{role}_sealed.jsonl"
    metadata_relative = f"audits/{phase}{suffix}_{role}_source.json"
    rows_path, marker, record = _verified_endpoint_artifact(
        store, study_id, endpoint, rows_relative
    )
    metadata_path, metadata_marker, metadata_record = _verified_endpoint_artifact(
        store, study_id, endpoint, metadata_relative
    )
    if marker != metadata_marker or record != metadata_record:
        raise ValueError(f"{role} endpoint artifacts do not share one marker")
    metadata = _read_json(metadata_path)
    expected = {
        "schema_version",
        "role",
        "identity",
        "phase",
        "size",
        "resolved_source_path",
        "input_sha256",
        "sample_marker_sha256",
    }
    if role == "adjudication":
        expected |= {"rater_a_marker_sha256", "rater_b_marker_sha256"}
    if set(metadata) != expected or metadata.get("schema_version") != 1:
        raise ValueError(f"{role} endpoint metadata has an invalid schema")
    if (
        metadata.get("role") != role
        or metadata.get("phase") != phase
        or metadata.get("size") != size
    ):
        raise ValueError(f"{role} endpoint metadata does not match the audit")
    _validated_identity(metadata.get("identity"), role)
    if not isinstance(metadata.get("resolved_source_path"), str):
        raise ValueError(f"{role} endpoint is missing resolved source provenance")
    if not isinstance(metadata.get("input_sha256"), str):
        raise ValueError(f"{role} endpoint is missing source hash provenance")
    rows = _read_rating_rows(rows_path, allow_empty=role == "adjudication")
    return rows, marker, record, metadata


def _validate_rater_endpoints(
    sealed: dict,
    sample_marker: Path,
    sample_record: dict,
    pending_rows: tuple[AuditRow, ...],
) -> None:
    for role in ("rater_a", "rater_b"):
        rows, marker, record, metadata = sealed[role]
        _require_exact_rating_ids(rows, pending_rows, role)
        if (
            record["parent_hashes"].get("audit_sample") != sha256_file(sample_marker)
            or metadata["sample_marker_sha256"] != sha256_file(sample_marker)
        ):
            raise ValueError(f"{role} endpoint does not bind the audit sample")
        if _parse_timestamp(record["created_at"]) <= _parse_timestamp(
            sample_record["created_at"]
        ):
            raise ValueError(f"{role} endpoint must postdate the sample marker")
    metadata_a = sealed["rater_a"][3]
    metadata_b = sealed["rater_b"][3]
    if metadata_a["identity"] == metadata_b["identity"]:
        raise ValueError("independent raters must have distinct identities")
    if metadata_a["resolved_source_path"] == metadata_b["resolved_source_path"]:
        raise ValueError("independent raters must use distinct resolved source paths")
    if metadata_a["input_sha256"] == metadata_b["input_sha256"]:
        raise ValueError("independent raters must use distinct source hashes")
    if sealed["rater_a"][1] == sealed["rater_b"][1] or sha256_file(
        sealed["rater_a"][1]
    ) == sha256_file(sealed["rater_b"][1]):
        raise ValueError("independent raters must have distinct endpoint markers and hashes")


def _load_rating_manifest(
    path: Path,
    pending_rows: tuple[AuditRow, ...],
    *,
    store: RLMFArtifactStore,
    study_id: str,
    phase: str,
    size: int,
    sample_marker: Path,
    sample_record: dict,
) -> tuple[dict[str, dict[str, str]], dict[str, dict]]:
    manifest = _read_json(path)
    if set(manifest) != {"schema_version", "rater_a", "rater_b", "adjudication"}:
        raise ValueError("rating manifest has an invalid schema")
    if manifest["schema_version"] != 2:
        raise ValueError("rating manifest schema_version must be 2")
    sealed = {}
    for role in ("rater_a", "rater_b", "adjudication"):
        entry = manifest[role]
        expected_endpoint = f"{phase}_judge_audit_{role}_{size}"
        if not isinstance(entry, dict) or set(entry) != {"endpoint", "marker_sha256"}:
            raise ValueError(f"{role} manifest entry has an invalid endpoint schema")
        if entry["endpoint"] != expected_endpoint:
            raise ValueError(f"{role} manifest references the wrong endpoint")
        sealed[role] = _verified_sealed_source(store, study_id, phase, size, role)
        if entry["marker_sha256"] != sha256_file(sealed[role][1]):
            raise ValueError(f"{role} endpoint marker hash mismatch")
    _validate_rater_endpoints(sealed, sample_marker, sample_record, pending_rows)
    identities = [sealed[role][3]["identity"] for role in sealed]
    if len(set(identities)) != 3:
        raise ValueError("raters and adjudicator must have distinct identities")
    rater_hashes = {
        role: sha256_file(sealed[role][1]) for role in ("rater_a", "rater_b")
    }
    adjudication_rows, adjudication_marker, adjudication_record, adjudication_metadata = (
        sealed["adjudication"]
    )
    if (
        adjudication_record["parent_hashes"].get("rater_a")
        != rater_hashes["rater_a"]
        or adjudication_record["parent_hashes"].get("rater_b")
        != rater_hashes["rater_b"]
    ):
        raise ValueError("adjudication endpoint does not bind both rater endpoints")
    if (
        adjudication_metadata["rater_a_marker_sha256"] != rater_hashes["rater_a"]
        or adjudication_metadata["rater_b_marker_sha256"] != rater_hashes["rater_b"]
    ):
        raise ValueError("adjudication source metadata does not bind both raters")
    rater_times = [
        _parse_timestamp(sealed[role][2]["created_at"])
        for role in ("rater_a", "rater_b")
    ]
    if _parse_timestamp(adjudication_record["created_at"]) <= max(rater_times):
        raise ValueError("adjudication endpoint must be sealed after both rater markers")

    source_rows = {name: sealed[name][0] for name in sealed}
    expected_ids = {row.audit_id for row in pending_rows}
    ratings: dict[str, dict[str, str]] = {}
    for name in ("rater_a", "rater_b"):
        mapping = {row["audit_id"]: row["label"] for row in source_rows[name]}
        if len(mapping) != len(source_rows[name]) or set(mapping) != expected_ids:
            raise ValueError(f"{name} must contain exactly the pending audit IDs")
        ratings[name] = mapping
    disagreements = {
        audit_id
        for audit_id in expected_ids
        if ratings["rater_a"][audit_id] != ratings["rater_b"][audit_id]
    }
    adjudication = {
        row["audit_id"]: row["label"] for row in source_rows["adjudication"]
    }
    if len(adjudication) != len(source_rows["adjudication"]):
        raise ValueError("adjudication audit IDs must be unique")
    if set(adjudication) != disagreements:
        raise ValueError("adjudication must address exactly the rater disagreements")
    ratings["adjudication"] = adjudication
    metadata = {}
    for name in sealed:
        metadata[name] = {
            "endpoint": manifest[name]["endpoint"],
            "marker_sha256": sha256_file(sealed[name][1]),
            "identity": sealed[name][3]["identity"],
            "completion_marker_created_at": sealed[name][2]["created_at"],
            "input_sha256": sealed[name][3]["input_sha256"],
            "audit_ids": [row["audit_id"] for row in source_rows[name]],
        }
    return ratings, metadata


def _read_rating_rows(path: Path, *, allow_empty: bool) -> tuple[dict, ...]:
    try:
        rows = tuple(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid rating JSONL input: {path}") from error
    if not rows and not allow_empty:
        raise ValueError("independent rating source must not be empty")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"audit_id", "label"}:
            raise ValueError("rating source contains hidden metadata or an invalid schema")
        if not isinstance(row["audit_id"], str) or row["label"] not in {
            "correct",
            "incorrect",
            "ambiguous",
        }:
            raise ValueError("rating source contains an invalid audit ID or label")
    return rows


def _write_jsonl_resumable(
    store: RLMFArtifactStore,
    study_id: str,
    section: str,
    name: str,
    rows,
) -> Path:
    frozen_rows = tuple(rows)
    expected = b"".join(
        json.dumps(row, sort_keys=True).encode("utf-8") + b"\n" for row in frozen_rows
    )
    path = _rlmf_audit_path(store.root, study_id, section, name)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != expected:
            raise ValueError(f"immutable audit artifact collision: {path}")
        return path
    return store.write_jsonl(study_id, section, name, frozen_rows)


def _write_json_resumable(
    store: RLMFArtifactStore,
    study_id: str,
    section: str,
    name: str,
    value,
) -> Path:
    expected = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
    path = _rlmf_audit_path(store.root, study_id, section, name).with_suffix(".json")
    if path.exists():
        if path.is_symlink() or path.read_bytes() != expected:
            raise ValueError(f"immutable audit artifact collision: {path}")
        return path
    return store.write_json(study_id, section, name, value)


def _complete_endpoint_resumable(
    store: RLMFArtifactStore,
    study_id: str,
    endpoint: str,
    config: RLMFConfig,
    paths,
    *,
    parent_hashes: dict[str, str],
) -> Path:
    frozen_paths = tuple(paths)
    marker = _study_namespace(store, study_id) / "endpoints" / f"{endpoint}.complete.json"
    if not marker.exists():
        return store.complete_endpoint(
            study_id,
            endpoint,
            config,
            frozen_paths,
            parent_hashes=parent_hashes,
        )
    record = store.verify_endpoint(study_id, endpoint)
    expected_parents = {"config": config.config_hash, **parent_hashes}
    expected_artifacts = {
        path.resolve().relative_to(_study_namespace(store, study_id)).as_posix(): sha256_file(path)
        for path in frozen_paths
    }
    if record["parent_hashes"] != dict(sorted(expected_parents.items())):
        raise ValueError(f"immutable endpoint parent collision: {endpoint}")
    if record["artifact_hashes"] != dict(sorted(expected_artifacts.items())):
        raise ValueError(f"immutable endpoint artifact collision: {endpoint}")
    return marker


def _study_namespace(store: RLMFArtifactStore, study_id: str) -> Path:
    return Path(store.root) / "runs" / "rlmf" / study_id


def _read_json(path: str | Path) -> dict:
    source = Path(path)
    try:
        value = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {source}")
    return value


def _parse_timestamp(value: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("audit timestamps must be ISO-8601 strings") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("audit timestamps must include a timezone")
    return parsed


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
