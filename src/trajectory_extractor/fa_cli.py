from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import stat
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import trajectory_extractor.fa_entities as fa_entities
import trajectory_extractor.fa_confirmatory_source as fa_confirmatory_source
import trajectory_extractor.fa_confirmatory_synthetics as fa_confirmatory_synthetics
from trajectory_extractor.fa_activations import (
    HFSelectedPositionRunner,
    load_activation_records,
    resume_activation_shard,
    write_activation_shard,
)
from trajectory_extractor.fa_artifacts import FAArtifactStore
from trajectory_extractor.fa_config import FAConfig, SMOKE_CHAT_TEMPLATE_SHA256
from trajectory_extractor.fa_confirmatory_source import (
    SOURCE_REVISION as CONFIRMATORY_SOURCE_REVISION,
)
from trajectory_extractor.fa_data import (
    CONFIRMATORY_POWER_SEED,
    CONFIRMATORY_POWER_SIMULATIONS,
    REGISTERED_POWER_GRID,
    REGISTERED_ENTITY_DOMAINS,
    FAExample,
    PowerAudit,
    PowerCell,
    audit_dataset,
    build_factorial_examples,
    build_manifest,
    build_same_string_examples,
    simulate_interaction_power,
)
from trajectory_extractor.fa_entities import (
    CandidateEntity,
    EntityMatch,
    NaturalnessAudit,
    NaturalnessRating,
    ScreeningQuestion,
    SyntheticCandidate,
    audit_naturalness_manifest,
    match_synthetic_entities,
    order_screening_questions,
    score_screening,
)
from trajectory_extractor.fa_runtime import (
    HFModelRunner,
    load_pinned_tokenizer,
    run_generation_shard,
    validate_runner_binding,
)
from trajectory_extractor.fa_scoring import (
    SameStringSealEvidence,
    behavioral_gate,
    crossed_bootstrap,
    estimate_behavior,
    score_response,
)
from trajectory_extractor.fa_report import (
    build_report,
    load_closed_f1_evidence,
    load_closed_f2a_evidence,
)
from trajectory_extractor.fa_features import (
    HFTeacherForcedScorer,
    OutputEvidence,
    UnsupportedAnswerOutcome,
    materialize_probe_rows,
)
from trajectory_extractor.fa_probes import (
    TASKS,
    NullSelectionResult,
    ProbeRow,
    ProbeSourceIdentity,
    ProbeTestAuthorization,
    SelectionManifest,
    evaluate_probe_bundle_once,
    f2a_selection_bundle_hash,
    fit_selection,
    run_full_selection_nulls,
)
from trajectory_extractor.fa_pilot_analysis import (
    PILOT_PERMUTATION_SEEDS,
    analyze_pilot_rows,
    build_pilot_analysis_rows,
)
from trajectory_extractor.fa_naturalness import (
    compile_adjudication_response_from_issuance,
    compile_initial_responses_from_issuance,
    issuance_pair_stimulus_sha256s,
    packet_issuance_record,
    prepare_adjudication_packet,
    prepare_initial_rating_packets,
    naturalness_matches_sha256,
    rating_record,
    submission_record,
    verify_submission_record,
)


FA_COMMANDS = (
    "fa-run-screening", "fa-screen-entities", "fa-assemble-screened-matches", "fa-prepare-naturalness-ratings", "fa-compile-naturalness-ratings", "fa-finalize-naturalness-adjudication", "fa-build-pilot", "fa-build-confirmatory", "fa-audit-manifest",
    "fa-run-generation", "fa-score-behavior",
    "fa-extract-activations", "fa-analyze-pilot-activations", "fa-materialize-probe-rows", "fa-fit-probes", "fa-seal-behavior-test", "fa-seal-selection", "fa-unlock-endpoint",
    "fa-evaluate-behavior-test", "fa-evaluate-probe-test", "fa-evaluate-intervention-test",
    "fa-run-interventions", "fa-select-circuit-cases", "fa-audit-circuit-fidelity", "fa-build-report",
)
_IMPLEMENTED = frozenset(
    (
        "fa-run-screening",
        "fa-screen-entities",
        "fa-assemble-screened-matches",
        "fa-prepare-naturalness-ratings",
        "fa-compile-naturalness-ratings",
        "fa-finalize-naturalness-adjudication",
        "fa-build-pilot",
        "fa-build-confirmatory",
        "fa-audit-manifest",
        "fa-run-generation",
        "fa-score-behavior",
        "fa-extract-activations",
        "fa-analyze-pilot-activations",
        "fa-materialize-probe-rows",
        "fa-fit-probes",
        "fa-seal-behavior-test",
        "fa-seal-selection",
        "fa-unlock-endpoint",
        "fa-evaluate-behavior-test",
        "fa-evaluate-probe-test",
        "fa-build-report",
    )
)
_GENERATION_NAMESPACES = ("pilot", "mechanism_train", "locked_validation", "circuit_dev", "behavior_test", "probe_test", "intervention_test")
_PROTECTED = frozenset({"behavior_test", "probe_test", "intervention_test"})
_CONFIRMATORY_RESERVE_PER_DOMAIN = {
    "mechanism_train": 4,
    "locked_validation": 2,
    "behavior_test": 3,
    "probe_test": 2,
    "intervention_test": 2,
}
_SMOKE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "familiarity_answerability_qwen17b_smoke.json"
)
_PREREGISTRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "familiarity_answerability_preregistration.md"
)
_NATURALNESS_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "fa_naturalness_rating_protocol.md"
)
_PILOT_ANALYSIS_SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "amendments"
    / "2026-07-23-fa-pilot-analysis-v13.json"
)
_PILOT_ANALYSIS_AMENDMENTS = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "amendments"
    / "2026-07-23-fa-pilot-mechanistic-v11.md",
    Path(__file__).resolve().parents[2]
    / "docs"
    / "amendments"
    / "2026-07-23-fa-pilot-anchor-adapter-v12.md",
    Path(__file__).resolve().parents[2]
    / "docs"
    / "amendments"
    / "2026-07-23-fa-pilot-analysis-v13.md",
)
_TOKENIZER_LOADER = None
_POWER_EXECUTOR = simulate_interaction_power
_SMOKE_CHAT_TEMPLATE_SHA256 = SMOKE_CHAT_TEMPLATE_SHA256
_ACTIVATION_RUNNER_FACTORY = HFSelectedPositionRunner
_ACTIVATION_SHARD_WRITER = write_activation_shard
_BEHAVIOR_BOOTSTRAP = crossed_bootstrap
_BEHAVIOR_GATE = behavioral_gate
_PROBE_SELECTOR = fit_selection
_PROBE_NULL_SELECTOR = run_full_selection_nulls
_PROBE_BUNDLE_EVALUATOR = evaluate_probe_bundle_once
_PROBE_SCORER_FACTORY = HFTeacherForcedScorer.from_config
_PROBE_ROW_MATERIALIZER = materialize_probe_rows


@dataclass(frozen=True)
class VerifiedPromptManifest:
    config_hash: str
    manifest_sha256: str
    full_manifest_sha256: str
    chat_template_sha256: str
    namespace: str
    examples: tuple[FAExample, ...]
    model_sha256: str
    tokenizer_sha256: str
    tokenizer_pin_manifest_path: Path
    tokenizer_pin_sha256: str
    naturalness_audit_manifest_path: Path | None
    naturalness_audit_sha256: str | None
    generation: dict[str, Any]
    shard_manifest_path: Path
    shard_sha256: str


@dataclass(frozen=True)
class VerifiedF2ASelectionBundle:
    selections: Mapping[str, SelectionManifest]
    null_selections: Mapping[str, tuple[NullSelectionResult, ...]]
    selection_bundle_hash: str
    probe_test_prompt_sha256: str
    probe_test_task_identities_sha256: str
    shard_manifest_path: Path
    shard_sha256: str


class _RecordedOutputScorer:
    """Replay immutable exact-sequence evidence without loading the model again."""

    def __init__(self, evidence: Mapping[str, OutputEvidence]) -> None:
        self._evidence = dict(evidence)

    def score(self, example: FAExample) -> OutputEvidence:
        try:
            return self._evidence[example.example_id]
        except KeyError as error:
            raise ValueError("recorded output evidence is missing an example") from error


def register_fa_subcommands(subparsers: argparse._SubParsersAction) -> None:
    parsers: dict[str, argparse.ArgumentParser] = {}
    for command in FA_COMMANDS:
        parser = subparsers.add_parser(command)
        parser.error = lambda message, command=command: _argument_error(command, message)
        parser.add_argument("--config", required=True)
        parser.add_argument("--root", default=".")
        parsers[command] = parser
    screening = parsers["fa-run-screening"]
    screening.add_argument("--candidates-manifest", required=True)
    screening.add_argument("--questions-manifest", required=True)
    screening.add_argument("--shard-id", required=True)
    screening.add_argument("--namespace", choices=_GENERATION_NAMESPACES, required=True)
    screening.add_argument("--source-integrity-manifest")
    screen_entities = parsers["fa-screen-entities"]
    screen_entities.add_argument("--candidates-manifest", required=True)
    screen_entities.add_argument("--questions-manifest", required=True)
    screen_entities.add_argument("--screening-manifest", required=True)
    screen_entities.add_argument("--synthetic-manifest", required=True)
    screen_entities.add_argument("--source-integrity-manifest")
    assemble = parsers["fa-assemble-screened-matches"]
    assemble.add_argument(
        "--screened-matches-manifest",
        action="append",
        required=True,
    )
    assemble.add_argument("--shard-id", required=True)
    for command in (
        "fa-prepare-naturalness-ratings",
        "fa-compile-naturalness-ratings",
        "fa-finalize-naturalness-adjudication",
    ):
        source = parsers[command].add_mutually_exclusive_group(required=True)
        source.add_argument("--screened-matches-manifest")
        source.add_argument("--matches-manifest")
    prepare_ratings = parsers["fa-prepare-naturalness-ratings"]
    prepare_ratings.add_argument("--output-dir", required=True)
    prepare_ratings.add_argument("--rater-id", action="append", required=True)
    prepare_ratings.add_argument("--shard-id", required=True)
    compile_ratings = parsers["fa-compile-naturalness-ratings"]
    compile_ratings.add_argument("--issuance-manifest", required=True)
    compile_ratings.add_argument("--response", action="append", required=True)
    compile_ratings.add_argument("--shard-id", required=True)
    compile_ratings.add_argument("--adjudicator-id")
    compile_ratings.add_argument("--adjudication-output-dir")
    finalize_ratings = parsers["fa-finalize-naturalness-adjudication"]
    finalize_ratings.add_argument("--initial-submission-manifest", required=True)
    finalize_ratings.add_argument("--adjudication-issuance-manifest", required=True)
    finalize_ratings.add_argument("--adjudication-response", required=True)
    finalize_ratings.add_argument("--shard-id", required=True)
    parsers["fa-build-pilot"].add_argument("--matches-manifest", required=True)
    parsers["fa-build-confirmatory"].add_argument("--matches-manifest", required=True)
    parsers["fa-build-confirmatory"].add_argument("--pilot-gate-manifest", required=True)
    parsers["fa-build-confirmatory"].add_argument(
        "--naturalness-ratings-manifest", required=True
    )
    power = parsers["fa-build-confirmatory"].add_mutually_exclusive_group(required=True)
    power.add_argument("--power-audit-manifest")
    power.add_argument("--run-registered-power-audit", action="store_true")
    parsers["fa-audit-manifest"].add_argument("--manifest", required=True)
    generation = parsers["fa-run-generation"]
    generation.add_argument("--manifest", required=True)
    generation.add_argument("--shard-id", required=True)
    generation.add_argument("--namespace", choices=_GENERATION_NAMESPACES, required=True)
    generation.add_argument("--resume", action="store_true")
    activations = parsers["fa-extract-activations"]
    activations.add_argument("--manifest", required=True)
    activations.add_argument("--shard-id", required=True)
    activations.add_argument("--namespace", choices=_GENERATION_NAMESPACES, required=True)
    activations.add_argument("--layers")
    activations.add_argument("--resume", action="store_true")
    pilot_analysis = parsers["fa-analyze-pilot-activations"]
    pilot_analysis.add_argument("--manifest", required=True)
    pilot_analysis.add_argument("--activation-manifest", required=True)
    pilot_analysis.add_argument("--pilot-gate-manifest", required=True)
    pilot_analysis.add_argument("--shard-id", required=True)
    materialize = parsers["fa-materialize-probe-rows"]
    materialize.add_argument("--manifest", required=True)
    materialize.add_argument("--metadata-manifest", required=True)
    materialize.add_argument("--shard-id", required=True)
    materialize.add_argument(
        "--namespace",
        choices=("mechanism_train", "locked_validation"),
        required=True,
    )
    materialize.add_argument("--resume", action="store_true")
    fit_probes = parsers["fa-fit-probes"]
    fit_probes.add_argument("--train-rows-manifest", required=True)
    fit_probes.add_argument("--validation-rows-manifest", required=True)
    fit_probes.add_argument("--probe-test-manifest", required=True)
    fit_probes.add_argument("--shard-id", required=True)
    behavior_seal = parsers["fa-seal-behavior-test"]
    behavior_seal.add_argument("--behavior-test-manifest", required=True)
    seal_selection = parsers["fa-seal-selection"]
    seal_selection.add_argument("--selection-manifest", required=True)
    seal_selection.add_argument("--probe-test-manifest", required=True)
    behavior_test = parsers["fa-evaluate-behavior-test"]
    behavior_test.add_argument("--manifest", required=True)
    behavior_test.add_argument("--shard-id", required=True)
    probe_test = parsers["fa-evaluate-probe-test"]
    probe_test.add_argument("--selection-manifest", required=True)
    probe_test.add_argument("--probe-test-manifest", required=True)
    probe_test.add_argument("--metadata-manifest", required=True)
    probe_test.add_argument("--shard-id", required=True)
    report = parsers["fa-build-report"]
    report.add_argument("--behavior-test-manifest")
    report.add_argument("--probe-test-manifest")
    report.add_argument("--selection-manifest")
    report.add_argument("--output", required=True)
    score = parsers["fa-score-behavior"]
    score.add_argument("--manifest", required=True)
    score.add_argument("--generation-manifest", required=True)


def dispatch_fa(args: argparse.Namespace) -> int | None:
    command = getattr(args, "command", None)
    if command not in FA_COMMANDS:
        return None
    if command not in _IMPLEMENTED:
        print(
            json.dumps(
                {
                    "command": command,
                    "error": {
                        "message": "FA command is not implemented",
                        "type": "NotImplementedError",
                    },
                    "status": "not_implemented",
                },
                sort_keys=True,
            )
        )
        return 2
    try:
        config = FAConfig.from_json(args.config)
        root = Path(args.root)
        if command == "fa-run-screening":
            payload = _run_screening(config, root, args)
        elif command == "fa-screen-entities":
            payload = _screen_entities(config, root, args)
        elif command == "fa-assemble-screened-matches":
            payload = _assemble_screened_matches(config, root, args)
        elif command == "fa-prepare-naturalness-ratings":
            payload = _prepare_naturalness_ratings(config, root, args)
        elif command == "fa-compile-naturalness-ratings":
            payload = _compile_naturalness_ratings(config, root, args)
        elif command == "fa-finalize-naturalness-adjudication":
            payload = _finalize_naturalness_adjudication(config, root, args)
        elif command in {"fa-build-pilot", "fa-build-confirmatory"}:
            payload = _build_manifest(config, root, args, confirmatory=command == "fa-build-confirmatory")
        elif command == "fa-audit-manifest":
            payload = _audit_manifest(config, root, args)
        elif command == "fa-run-generation":
            payload = _run_generation(config, root, args)
        elif command == "fa-extract-activations":
            payload = _extract_activations(config, root, args)
        elif command == "fa-analyze-pilot-activations":
            payload = _analyze_pilot_activations(config, root, args)
        elif command == "fa-materialize-probe-rows":
            payload = _materialize_probe_rows(config, root, args)
        elif command == "fa-fit-probes":
            payload = _fit_probes(config, root, args)
        elif command == "fa-seal-behavior-test":
            payload = _seal_behavior_test(config, root, args)
        elif command == "fa-seal-selection":
            payload = _seal_probe_selection(config, root, args)
        elif command == "fa-evaluate-behavior-test":
            payload = _evaluate_behavior_test(config, root, args)
        elif command == "fa-evaluate-probe-test":
            payload = _evaluate_probe_test(config, root, args)
        elif command == "fa-build-report":
            payload = _build_evidence_report(config, root, args)
        elif command == "fa-unlock-endpoint":
            payload = _reject_standalone_unlock()
        else:
            payload = _score_behavior(config, root, args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": command,
                    "error": {"message": str(error), "type": type(error).__name__},
                    "status": "error",
                },
                sort_keys=True,
            )
        )
        return 3 if isinstance(error, (ImportError, OSError, RuntimeError)) else 2
    print(json.dumps(_json_safe({"command": command, **payload}), sort_keys=True, allow_nan=False))
    return 3 if payload.get("status") == "infrastructure_failure" else 0


def _reject_standalone_unlock() -> dict[str, Any]:
    raise ValueError(
        "standalone endpoint unlock is disabled; a dedicated protected evaluation "
        "command must acquire and close its lease atomically"
    )


def _prepare_naturalness_ratings(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    store = FAArtifactStore(root)
    matches = _load_naturalness_matches(config, root, args)
    if len(args.rater_id) != 2:
        raise ValueError("exactly two --rater-id values are required")
    protocol_sha256 = hashlib.sha256(_PREREGISTRATION_PATH.read_bytes()).hexdigest()
    rating_protocol_sha256 = hashlib.sha256(
        _NATURALNESS_PROTOCOL_PATH.read_bytes()
    ).hexdigest()
    prepared = prepare_initial_rating_packets(
        matches,
        config_sha256=config.config_hash,
        protocol_sha256=protocol_sha256,
        rating_protocol_sha256=rating_protocol_sha256,
        output_dir=args.output_dir,
        rater_ids=args.rater_id,
    )
    row, lineage = packet_issuance_record(prepared["private_key"])
    issuance = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        args.shard_id,
        (row,),
        lineage,
        record_kind="naturalness_packet_issuance",
    )
    _restrict_private_naturalness_shard(issuance)
    return {
        **prepared,
        "issuance_manifest": str(issuance.manifest_path),
        "issuance_sha256": issuance.sha256,
    }


def _compile_naturalness_ratings(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    store = FAArtifactStore(root)
    matches = _load_naturalness_matches(config, root, args)
    protocol_sha256 = hashlib.sha256(_PREREGISTRATION_PATH.read_bytes()).hexdigest()
    rating_protocol_sha256 = hashlib.sha256(
        _NATURALNESS_PROTOCOL_PATH.read_bytes()
    ).hexdigest()
    issuance_row, issuance_shard = _load_naturalness_packet_issuance(
        store, args.issuance_manifest, config, expected_purpose="initial"
    )
    ratings, assignments, disagreements, responses = (
        compile_initial_responses_from_issuance(
        matches,
        issuance=issuance_row,
        response_paths=args.response,
        config_sha256=config.config_hash,
        protocol_sha256=protocol_sha256,
        rating_protocol_sha256=rating_protocol_sha256,
        )
    )
    if disagreements:
        if not args.adjudicator_id or not args.adjudication_output_dir:
            raise ValueError(
                "rater disagreement requires --adjudicator-id and "
                "--adjudication-output-dir"
            )
        initial_raters = {value.rater_id for value in ratings}
        if args.adjudicator_id in initial_raters:
            raise ValueError("third rater must be independent")
    submission_row, submission_lineage = submission_record(
        ratings,
        assignments,
        responses,
        config_sha256=config.config_hash,
        issuance_manifest=str(issuance_shard.manifest_path.relative_to(store.root)),
        issuance_sha256=issuance_shard.sha256,
        disagreement_pair_ids=disagreements,
    )
    submission = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        f"{args.shard_id}-initial",
        (submission_row,),
        submission_lineage,
        record_kind="naturalness_submission",
    )
    if disagreements:
        packet = prepare_adjudication_packet(
            matches,
            pair_ids=disagreements,
            config_sha256=config.config_hash,
            protocol_sha256=protocol_sha256,
            rating_protocol_sha256=rating_protocol_sha256,
            output_dir=args.adjudication_output_dir,
            adjudicator_id=args.adjudicator_id,
        )
        adjudication_row, adjudication_lineage = packet_issuance_record(
            packet["private_key"]
        )
        adjudication = store.write_completed_shard(
            config.run_id,
            "mechanism_train",
            f"{args.shard_id}-adjudication-issuance",
            (adjudication_row,),
            {
                **adjudication_lineage,
                "initial_submission_sha256": submission.sha256,
            },
            record_kind="naturalness_packet_issuance",
        )
        _restrict_private_naturalness_shard(adjudication)
        return {
            "status": "needs_adjudication",
            "disagreement_pair_count": len(disagreements),
            "initial_submission_manifest": str(submission.manifest_path),
            "initial_submission_sha256": submission.sha256,
            "adjudication": packet,
            "adjudication_issuance_manifest": str(adjudication.manifest_path),
            "adjudication_issuance_sha256": adjudication.sha256,
        }

    audit_naturalness_manifest(matches, ratings)
    row, lineage = rating_record(
        ratings,
        assignments,
        config_sha256=config.config_hash,
        protocol_sha256=protocol_sha256,
        additional_lineage={
            "rating_protocol_sha256": rating_protocol_sha256,
            "matches_sha256": naturalness_matches_sha256(matches),
            "initial_submission_manifest": str(
                submission.manifest_path.relative_to(store.root)
            ),
            "initial_submission_sha256": submission.sha256,
        },
    )
    shard = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        args.shard_id,
        (row,),
        lineage,
        record_kind="naturalness_ratings",
    )
    return {
        "status": "compiled",
        "pair_count": len(matches),
        "rating_count": len(ratings),
        "initial_submission_manifest": str(submission.manifest_path),
        "ratings_manifest": str(shard.manifest_path),
        "sha256": shard.sha256,
    }


def _finalize_naturalness_adjudication(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    store = FAArtifactStore(root)
    matches = _load_naturalness_matches(config, root, args)
    protocol_sha256 = hashlib.sha256(_PREREGISTRATION_PATH.read_bytes()).hexdigest()
    rating_protocol_sha256 = hashlib.sha256(
        _NATURALNESS_PROTOCOL_PATH.read_bytes()
    ).hexdigest()
    initial_row, initial_shard = _load_naturalness_submission(
        store, args.initial_submission_manifest, config
    )
    disagreements = tuple(initial_row["disagreement_pair_ids"])
    if not disagreements:
        raise ValueError("initial submission has no registered disagreements")
    issuance_row, issuance_shard = _load_naturalness_packet_issuance(
        store,
        args.adjudication_issuance_manifest,
        config,
        expected_purpose="adjudication",
    )
    issuance_lineage = _read_json_object(issuance_shard.manifest_path)["lineage"]
    if issuance_lineage.get("initial_submission_sha256") != initial_shard.sha256:
        raise ValueError(
            "adjudication issuance does not bind the initial disagreement artifact"
        )
    third_ratings, third_assignments, third_responses = (
        compile_adjudication_response_from_issuance(
            matches,
            issuance=issuance_row,
            response_path=args.adjudication_response,
            expected_pair_ids=disagreements,
            config_sha256=config.config_hash,
            protocol_sha256=protocol_sha256,
            rating_protocol_sha256=rating_protocol_sha256,
        )
    )
    initial_ratings = tuple(
        NaturalnessRating(**value) for value in initial_row["ratings"]
    )
    initial_assignments = tuple(initial_row["assignments"])
    initial_raters = {value.rater_id for value in initial_ratings}
    if any(value.rater_id in initial_raters for value in third_ratings):
        raise ValueError("third rater must be independent")
    ratings = (*initial_ratings, *third_ratings)
    assignments = (*initial_assignments, *third_assignments)
    audit_naturalness_manifest(matches, ratings)

    adjudication_row, adjudication_lineage = submission_record(
        third_ratings,
        third_assignments,
        third_responses,
        config_sha256=config.config_hash,
        issuance_manifest=str(issuance_shard.manifest_path.relative_to(store.root)),
        issuance_sha256=issuance_shard.sha256,
        disagreement_pair_ids=(),
    )
    adjudication_submission = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        f"{args.shard_id}-adjudication",
        (adjudication_row,),
        {
            **adjudication_lineage,
            "initial_submission_sha256": initial_shard.sha256,
        },
        record_kind="naturalness_submission",
    )
    row, lineage = rating_record(
        ratings,
        assignments,
        config_sha256=config.config_hash,
        protocol_sha256=protocol_sha256,
        additional_lineage={
            "rating_protocol_sha256": rating_protocol_sha256,
            "matches_sha256": naturalness_matches_sha256(matches),
            "initial_submission_manifest": str(
                initial_shard.manifest_path.relative_to(store.root)
            ),
            "initial_submission_sha256": initial_shard.sha256,
            "adjudication_submission_manifest": str(
                adjudication_submission.manifest_path.relative_to(store.root)
            ),
            "adjudication_submission_sha256": adjudication_submission.sha256,
        },
    )
    shard = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        args.shard_id,
        (row,),
        lineage,
        record_kind="naturalness_ratings",
    )
    return {
        "status": "compiled",
        "pair_count": len(matches),
        "rating_count": len(ratings),
        "initial_submission_manifest": str(initial_shard.manifest_path),
        "adjudication_submission_manifest": str(
            adjudication_submission.manifest_path
        ),
        "ratings_manifest": str(shard.manifest_path),
        "sha256": shard.sha256,
    }


def _load_naturalness_matches(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> tuple[EntityMatch, ...]:
    screened = getattr(args, "screened_matches_manifest", None)
    raw = getattr(args, "matches_manifest", None)
    if screened:
        store = FAArtifactStore(root)
        shard = store.verify_shard(screened)
        if shard.record_kind == "screened_match_collection":
            return _load_verified_screened_match_collection(
                store, screened, config
            )
        if config.profile == "confirmatory":
            raise ValueError(
                "confirmatory naturalness requires a verified screened-match collection"
            )
        return _load_verified_screened_matches(store, screened, config)
    if config.profile == "confirmatory":
        raise ValueError(
            "confirmatory naturalness requires a verified screened-match collection"
        )
    rows = _read_json_rows(raw)
    try:
        matches = tuple(EntityMatch(**_without_schema(row)) for row in rows)
    except (TypeError, ValueError) as error:
        raise ValueError("naturalness matches manifest is invalid") from error
    if not matches or len({value.pair_id for value in matches}) != len(matches):
        raise ValueError("naturalness matches must contain unique pairs")
    return matches


def _assemble_screened_matches(
    config: FAConfig,
    root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Bind all split-specific screening shards into one immutable collection."""
    if config.profile != "confirmatory":
        raise ValueError("screened-match assembly requires the confirmatory config")
    store = FAArtifactStore(root)
    manifest_paths = tuple(Path(value) for value in args.screened_matches_manifest)
    if len(manifest_paths) != len(config.split_counts):
        raise ValueError("screened-match assembly requires exactly one shard per split")
    children = []
    matches = []
    seen_namespaces = set()
    for path in manifest_paths:
        shard = store.verify_shard(path)
        if shard.namespace in seen_namespaces:
            raise ValueError("screened-match assembly contains a duplicate split shard")
        split_matches = _load_verified_screened_matches(store, path, config)
        seen_namespaces.add(shard.namespace)
        matches.extend(split_matches)
        children.append(
            {
                "namespace": shard.namespace,
                "manifest_path": str(shard.manifest_path.relative_to(store.root)),
                "sha256": shard.sha256,
            }
        )
    if seen_namespaces != set(config.split_counts):
        raise ValueError("screened-match assembly does not cover every confirmatory split")
    _audit_confirmatory_match_pool(config, matches)
    ordered = tuple(sorted(matches, key=lambda row: (row.split, row.pair_id)))
    lineage = {
        "config_sha256": config.config_hash,
        "matching_policy_sha256": _matching_policy_sha256(),
        "children": sorted(children, key=lambda row: row["namespace"]),
        "matches_sha256": naturalness_matches_sha256(ordered),
    }
    shard = _write_or_verify_screening_shard(
        store,
        config.run_id,
        "mechanism_train",
        _safe_cli_id(args.shard_id, "screened-match collection shard-id"),
        [
            {"kind": "screened_match_collection", **asdict(row)}
            for row in ordered
        ],
        lineage,
        record_kind="screened_match_collection",
    )
    return {
        "status": "assembled",
        "manifest": str(shard.manifest_path),
        "count": len(ordered),
        "matches_sha256": lineage["matches_sha256"],
    }


def _load_naturalness_packet_issuance(
    store: FAArtifactStore,
    path: str | Path,
    config: FAConfig,
    *,
    expected_purpose: str,
) -> tuple[dict[str, Any], Any]:
    shard = _require_verified_shard_kind(
        store,
        path,
        "naturalness_packet_issuance",
        "verified naturalness packet issuance manifest",
    )
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    if shard.namespace != "mechanism_train":
        raise ValueError("naturalness packet issuance must use mechanism_train")
    rows = _read_json_rows(shard.data_path)
    if len(rows) != 1:
        raise ValueError("naturalness packet issuance must contain one record")
    row = rows[0]
    required = {
        "kind",
        "schema_version",
        "purpose",
        "config_sha256",
        "protocol_sha256",
        "rating_protocol_sha256",
        "matches_sha256",
        "private_key",
        "packets",
    }
    protocol_sha256 = hashlib.sha256(_PREREGISTRATION_PATH.read_bytes()).hexdigest()
    rating_protocol_sha256 = hashlib.sha256(
        _NATURALNESS_PROTOCOL_PATH.read_bytes()
    ).hexdigest()
    if (
        set(row) != required
        or row.get("kind") != "naturalness_packet_issuance"
        or row.get("schema_version") != 1
        or row.get("purpose") != expected_purpose
        or row.get("config_sha256") != config.config_hash
        or row.get("protocol_sha256") != protocol_sha256
        or row.get("rating_protocol_sha256") != rating_protocol_sha256
    ):
        raise ValueError("naturalness packet issuance has an invalid schema")
    _require_private_naturalness_shard(shard)
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config.config_hash,
            "protocol_sha256": protocol_sha256,
            "rating_protocol_sha256": rating_protocol_sha256,
            "matches_sha256": row["matches_sha256"],
            "issuance_sha256": _sha256_json(row),
        },
    )
    return row, shard


def _restrict_private_naturalness_shard(shard: Any) -> None:
    for path in (shard.data_path, shard.manifest_path):
        path.chmod(0o600)


def _require_private_naturalness_shard(shard: Any) -> None:
    for path in (shard.data_path, shard.manifest_path):
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError as error:
            raise ValueError("naturalness packet issuance permissions are unreadable") from error
        if mode & 0o077:
            raise ValueError(
                "naturalness packet issuance must not be group- or world-readable"
            )


def _load_naturalness_submission(
    store: FAArtifactStore,
    path: str | Path,
    config: FAConfig,
) -> tuple[dict[str, Any], Any]:
    shard = _require_verified_shard_kind(
        store,
        path,
        "naturalness_submission",
        "verified naturalness submission manifest",
    )
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    if shard.namespace != "mechanism_train":
        raise ValueError("naturalness submission must use mechanism_train")
    rows = _read_json_rows(shard.data_path)
    required = {
        "kind",
        "schema_version",
        "config_sha256",
        "issuance_manifest",
        "issuance_sha256",
        "ratings",
        "assignments",
        "responses",
        "disagreement_pair_ids",
    }
    if (
        len(rows) != 1
        or set(rows[0]) != required
        or rows[0].get("kind") != "naturalness_submission"
        or rows[0].get("schema_version") != 1
        or rows[0].get("config_sha256") != config.config_hash
    ):
        raise ValueError("naturalness submission has an invalid schema")
    row = rows[0]
    issuance_path = _artifact_path_from_record(
        store, row["issuance_manifest"], "naturalness packet issuance manifest"
    )
    issuance = _require_verified_shard_kind(
        store,
        issuance_path,
        "naturalness_packet_issuance",
        "verified naturalness packet issuance manifest",
    )
    if issuance.sha256 != row["issuance_sha256"]:
        raise ValueError("naturalness submission issuance hash does not verify")
    issuance_rows = _read_json_rows(issuance.data_path)
    if (
        len(issuance_rows) != 1
        or issuance_rows[0].get("purpose") not in {"initial", "adjudication"}
    ):
        raise ValueError("naturalness submission issuance is invalid")
    issuance_row, verified_issuance = _load_naturalness_packet_issuance(
        store,
        issuance.manifest_path,
        config,
        expected_purpose=issuance_rows[0]["purpose"],
    )
    if verified_issuance != issuance:
        raise ValueError("naturalness submission issuance identity changed")
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config.config_hash,
            "issuance_sha256": issuance.sha256,
            "submission_sha256": _sha256_json(row),
        },
    )
    try:
        ratings = tuple(NaturalnessRating(**value) for value in row["ratings"])
    except (TypeError, ValueError) as error:
        raise ValueError("naturalness submission ratings are invalid") from error
    if len({(value.pair_id, value.rater_id) for value in ratings}) != len(ratings):
        raise ValueError("naturalness submission ratings must be unique")
    verify_submission_record(issuance_row, row)
    return row, shard


def _argument_error(command: str, message: str) -> None:
    print(
        json.dumps(
            {
                "command": command,
                "error": {"message": message, "type": "ArgumentError"},
                "status": "error",
            },
            sort_keys=True,
        )
    )
    raise SystemExit(2)


def _run_screening(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    candidates = tuple(
        CandidateEntity(**_without_schema(row))
        for row in _read_json_rows(args.candidates_manifest)
    )
    questions = tuple(
        ScreeningQuestion(**_without_schema(row))
        for row in _read_json_rows(args.questions_manifest)
    )
    _screening_required_count(candidates, config)
    source_integrity_sha256 = _verify_confirmatory_source_inputs(
        config,
        root,
        args.namespace,
        args.candidates_manifest,
        args.questions_manifest,
        getattr(args, "source_integrity_manifest", None),
    )
    joined = order_screening_questions(candidates, questions)
    if any(candidate.split != args.namespace for candidate, _ in joined):
        raise ValueError("screening candidates must all belong to the requested namespace")

    runner = HFModelRunner(config)
    template_hash = config.chat_template_sha256 or _SMOKE_CHAT_TEMPLATE_SHA256
    validate_runner_binding(
        runner,
        config,
        expected_chat_template_sha256=template_hash,
    )
    ordered = tuple(
        (candidate, index, question)
        for candidate, candidate_questions in joined
        for index, question in enumerate(candidate_questions)
    )
    rendered_prompts = tuple(
        _render_screening_prompt(runner, question.prompt)
        for _, _, question in ordered
    )
    try:
        outputs = tuple(runner.generate(rendered_prompts, dict(config.generation)))
        if len(outputs) != len(ordered) or any(
            not isinstance(output, str) for output in outputs
        ):
            raise RuntimeError("model runner returned invalid screening completions")
        status = "completed"
        exception_class = None
    except Exception as error:
        outputs = (None,) * len(ordered)
        status = "infrastructure_failure"
        exception_class = type(error).__name__

    candidate_hash = _candidate_manifest_sha256(candidates)
    question_hash = _screening_question_manifest_sha256(questions)
    model_hash, tokenizer_hash = _config_runtime_hashes(config)
    records = []
    for (candidate, index, question), rendered, output in zip(
        ordered, rendered_prompts, outputs, strict=True
    ):
        records.append(
            {
                "kind": "screening_completion",
                "schema_version": 1,
                "entity_id": candidate.entity_id,
                "qid": candidate.qid,
                "question_id": question.question_id,
                "question_index": index,
                "prompt": question.prompt,
                "accepted_aliases": list(question.accepted_aliases),
                "source_provenance": question.source_provenance,
                "rendered_prompt_sha256": hashlib.sha256(
                    rendered.encode("utf-8")
                ).hexdigest(),
                "raw_output": output,
                "answer_text": (
                    None if output is None else _screening_answer_text(output)
                ),
                "generation": dict(config.generation),
                "config_sha256": config.config_hash,
                "model_sha256": model_hash,
                "tokenizer_sha256": tokenizer_hash,
                "chat_template_sha256": template_hash,
                "status": status,
                "exception_class": exception_class,
            }
        )
    shard = FAArtifactStore(root).write_completed_shard(
        config.run_id,
        args.namespace,
        args.shard_id,
        records,
        {
            "config_sha256": config.config_hash,
            "candidate_manifest_sha256": candidate_hash,
            "questions_manifest_sha256": question_hash,
            "model_sha256": model_hash,
            "tokenizer_sha256": tokenizer_hash,
            "chat_template_sha256": template_hash,
            "source_integrity_sha256": source_integrity_sha256,
        },
        record_kind="screening_completion",
    )
    if status != "completed":
        return {
            "status": "infrastructure_failure",
            "error": {
                "message": "screening generation failed; preserve the artifact and retry with a new shard id",
                "type": exception_class or "InfrastructureFailure",
            },
            "shard_manifest": str(shard.manifest_path),
            "sha256": shard.sha256,
        }
    return {
        "status": "generated",
        "shard_manifest": str(shard.manifest_path),
        "sha256": shard.sha256,
        "count": len(records),
    }


def _screen_entities(config: FAConfig, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    store = FAArtifactStore(root)
    candidates = tuple(CandidateEntity(**_without_schema(row)) for row in _read_json_rows(args.candidates_manifest))
    questions = tuple(
        ScreeningQuestion(**_without_schema(row))
        for row in _read_json_rows(args.questions_manifest)
    )
    synthetic = tuple(SyntheticCandidate(**_without_schema(row)) for row in _read_json_rows(args.synthetic_manifest))
    order_screening_questions(candidates, questions)
    required_count = _screening_required_count(candidates, config)
    source_integrity_sha256 = _verify_confirmatory_source_inputs(
        config,
        root,
        candidates[0].split if candidates else "",
        args.candidates_manifest,
        args.questions_manifest,
        getattr(args, "source_integrity_manifest", None),
        synthetic_manifest=args.synthetic_manifest,
    )
    reserve_per_domain = _screening_reserve_per_domain(candidates, config)
    screening_shard = _require_verified_shard_kind(
        store,
        args.screening_manifest,
        "screening_completion",
        "verified screening completion manifest",
    )
    prepared = load_pinned_tokenizer(config, tokenizer_loader=_TOKENIZER_LOADER)
    completions = _load_verified_screening_completions(
        store,
        args.screening_manifest,
        candidates,
        questions,
        config,
        tokenizer=prepared.tokenizer,
        source_integrity_sha256=source_integrity_sha256,
    )
    results = tuple(
        score_screening(candidate, completions.get(candidate.entity_id, ()))
        for candidate in candidates
    )
    qualified = tuple(
        candidate
        for candidate, result in zip(candidates, results, strict=True)
        if result.qualifies
    )
    matchable_qualified = _matchable_screening_candidates(
        qualified,
        synthetic,
        prepared.tokenizer,
        required_variants=3 if config.profile == "confirmatory" else 1,
        generator_revision=(
            fa_confirmatory_synthetics.GENERATOR_REVISION
            if config.profile == "confirmatory"
            else None
        ),
    )
    audit_lineage = {
        "config_sha256": config.config_hash,
        "candidate_manifest_sha256": _candidate_manifest_sha256(candidates),
        "questions_manifest_sha256": _screening_question_manifest_sha256(questions),
        "synthetic_manifest_sha256": _synthetic_manifest_sha256(synthetic),
        "matchable_qualified_sha256": _sha256_json(
            [candidate.entity_id for candidate in matchable_qualified]
        ),
        "screening_completion_sha256": screening_shard.sha256,
        "screening_parser_sha256": _screening_parser_sha256(),
        "source_integrity_sha256": source_integrity_sha256,
    }
    try:
        selected = _select_domain_balanced_candidates(
            matchable_qualified,
            required_count=required_count,
            reserve_per_domain=reserve_per_domain,
        )
    except ValueError as error:
        audit = _write_screening_audit(
            store,
            config,
            screening_shard,
            candidates,
            results,
            (),
            required_count,
            reserve_per_domain,
            audit_lineage,
            decision="stopped",
            stop_reason=str(error),
        )
        raise ValueError(
            f"{error}; screening audit: {audit.manifest_path}"
        ) from error
    matches = match_synthetic_entities(selected, synthetic, prepared.tokenizer)
    matching_policy_hash = _matching_policy_sha256()
    audit = _write_screening_audit(
        store,
        config,
        screening_shard,
        candidates,
        results,
        selected,
        required_count,
        reserve_per_domain,
        audit_lineage,
        decision="passed",
        stop_reason=None,
    )
    model_hash, tokenizer_hash = _config_runtime_hashes(config)
    template_hash = config.chat_template_sha256 or _SMOKE_CHAT_TEMPLATE_SHA256
    match_shard = _write_or_verify_screening_shard(
        store,
        config.run_id,
        screening_shard.namespace,
        f"screened-matches-{screening_shard.shard_id}-{matching_policy_hash[:12]}",
        [{"kind": "screened_match", **asdict(row)} for row in matches],
        {
            **audit_lineage,
            "matching_policy_sha256": matching_policy_hash,
            "screening_audit_sha256": audit.sha256,
            "model_sha256": model_hash,
            "tokenizer_sha256": tokenizer_hash,
            "chat_template_sha256": template_hash,
            "screening_completion_manifest": str(
                screening_shard.manifest_path.relative_to(store.root)
            ),
            "screening_audit_manifest": str(
                audit.manifest_path.relative_to(store.root)
            ),
        },
        record_kind="screened_match",
    )
    return {
        "status": "screened",
        "manifest": str(match_shard.manifest_path),
        "audit_manifest": str(audit.manifest_path),
        "count": len(matches),
    }


def _write_screening_audit(
    store: FAArtifactStore,
    config: FAConfig,
    screening_shard: Any,
    candidates: Sequence[CandidateEntity],
    results: Sequence[Any],
    selected: Sequence[CandidateEntity],
    required_count: int,
    reserve_per_domain: int,
    lineage: Mapping[str, Any],
    *,
    decision: str,
    stop_reason: str | None,
):
    domains = tuple(REGISTERED_ENTITY_DOMAINS)
    quota = required_count // len(domains)
    selected_quota = quota + reserve_per_domain
    qualified_ids = {
        domain: [
            candidate.entity_id
            for candidate, result in zip(candidates, results, strict=True)
            if candidate.coarse_type == domain and result.qualifies
        ]
        for domain in domains
    }
    row = {
        "kind": "screening_audit",
        "schema_version": 1,
        "decision": decision,
        "stop_reason": stop_reason,
        "required_count": required_count,
        "quota_per_domain": quota,
        "reserve_per_domain": reserve_per_domain,
        "selected_quota_per_domain": selected_quota,
        "selected_count": len(selected),
        "candidate_order": [candidate.entity_id for candidate in candidates],
        "candidate_scores": [asdict(result) for result in results],
        "qualified_entity_ids_by_domain": qualified_ids,
        "selected_entity_ids": [candidate.entity_id for candidate in selected],
        "config_sha256": config.config_hash,
        "screening_completion_sha256": screening_shard.sha256,
        "screening_parser_sha256": _screening_parser_sha256(),
    }
    return _write_or_verify_screening_shard(
        store,
        config.run_id,
        screening_shard.namespace,
        (
            f"screening-audit-{screening_shard.shard_id}-"
            f"{_sha256_json(lineage)[:12]}"
        ),
        [row],
        lineage,
        record_kind="screening_audit",
    )


def _write_or_verify_screening_shard(
    store: FAArtifactStore,
    run_id: str,
    namespace: str,
    shard_id: str,
    rows: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
    *,
    record_kind: str,
):
    expected_rows = tuple(_json_safe(dict(row)) for row in rows)
    expected_lineage = _json_safe(dict(lineage))
    try:
        return store.write_completed_shard(
            run_id,
            namespace,
            shard_id,
            expected_rows,
            expected_lineage,
            record_kind=record_kind,
        )
    except FileExistsError:
        matches = tuple(
            shard
            for shard in store.resume_verified_shards(run_id, namespace)
            if shard.shard_id == shard_id
        )
        if len(matches) != 1:
            raise ValueError(
                "existing screening artifact is incomplete or ambiguous"
            ) from None
        shard = matches[0]
        if shard.record_kind != record_kind:
            raise ValueError("existing screening artifact has the wrong kind")
        if _read_json_rows(shard.data_path) != expected_rows:
            raise ValueError("existing screening artifact rows do not match")
        if _read_json_object(shard.manifest_path).get("lineage") != expected_lineage:
            raise ValueError("existing screening artifact lineage does not match")
        return shard


def _screening_required_count(
    candidates: Sequence[CandidateEntity],
    config: FAConfig,
) -> int:
    candidate_rows = tuple(candidates)
    if len({candidate.entity_id for candidate in candidate_rows}) != len(candidate_rows):
        raise ValueError("screening candidates contain duplicate entity IDs")
    if len({candidate.qid for candidate in candidate_rows}) != len(candidate_rows):
        raise ValueError("screening candidates contain duplicate QIDs")
    candidate_splits = {candidate.split for candidate in candidate_rows}
    if len(candidate_splits) != 1:
        raise ValueError("screening candidates must belong to exactly one split")
    candidate_split = next(iter(candidate_splits))
    try:
        required_count = config.split_counts[candidate_split]
    except KeyError as error:
        raise ValueError("screening candidate split is not registered in the config") from error
    if config.profile == "confirmatory":
        expected_pool = required_count * 2
        if len(candidate_rows) != expected_pool:
            raise ValueError(
                "confirmatory screening requires the exact registered 2x source pool"
            )
        expected_per_domain = expected_pool // len(REGISTERED_ENTITY_DOMAINS)
        observed = {
            domain: sum(
                candidate.coarse_type == domain for candidate in candidate_rows
            )
            for domain in REGISTERED_ENTITY_DOMAINS
        }
        if observed != {
            domain: expected_per_domain for domain in REGISTERED_ENTITY_DOMAINS
        }:
            raise ValueError(
                "confirmatory screening requires exact registered domain balance"
            )
    return required_count


def _verify_confirmatory_source_inputs(
    config: FAConfig,
    root: Path,
    split: str,
    candidate_manifest: str | Path,
    question_manifest: str | Path,
    integrity_manifest: str | Path | None,
    *,
    synthetic_manifest: str | Path | None = None,
) -> str | None:
    if config.profile != "confirmatory":
        if integrity_manifest is not None:
            raise ValueError("smoke screening cannot claim confirmatory source integrity")
        return None
    if integrity_manifest is None:
        raise ValueError(
            "confirmatory screening requires the frozen source-integrity manifest"
        )
    root = Path(root).resolve()

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    integrity_path = resolve(integrity_manifest)
    integrity = _read_json_object(integrity_path)
    if (
        integrity.get("schema_version") != 1
        or integrity.get("source_revision") != CONFIRMATORY_SOURCE_REVISION
    ):
        raise ValueError("confirmatory source integrity has the wrong revision")
    if (
        integrity.get("source_matching_policy_sha256")
        != fa_confirmatory_source.source_matching_policy_sha256()
    ):
        raise ValueError(
            "confirmatory source matching policy hash does not verify"
        )
    files = integrity.get("materialized_files")
    registered = files.get(split) if isinstance(files, Mapping) else None
    if not isinstance(registered, Mapping):
        raise ValueError("confirmatory source integrity does not register this split")
    candidate_path = resolve(candidate_manifest)
    question_path = resolve(question_manifest)
    if (
        resolve(str(registered.get("candidate_manifest", ""))) != candidate_path
        or resolve(str(registered.get("question_manifest", ""))) != question_path
    ):
        raise ValueError("confirmatory source input paths do not match the frozen snapshot")
    candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    question_sha256 = hashlib.sha256(question_path.read_bytes()).hexdigest()
    if (
        registered.get("candidate_sha256") != candidate_sha256
        or registered.get("question_sha256") != question_sha256
    ):
        raise ValueError("confirmatory source input hashes do not match the frozen snapshot")
    snapshot_path = resolve(str(integrity.get("source_snapshot", "")))
    if (
        not snapshot_path.is_file()
        or integrity.get("source_snapshot_sha256")
        != hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    ):
        raise ValueError("confirmatory source snapshot hash does not verify")
    synthetic_snapshot_path = resolve(
        str(integrity.get("synthetic_snapshot", ""))
    )
    if (
        not synthetic_snapshot_path.is_file()
        or integrity.get("synthetic_snapshot_sha256")
        != hashlib.sha256(synthetic_snapshot_path.read_bytes()).hexdigest()
    ):
        raise ValueError("confirmatory synthetic snapshot hash does not verify")
    synthetic_files = integrity.get("synthetic_files")
    if not isinstance(synthetic_files, Mapping):
        raise ValueError("confirmatory source integrity omits synthetic files")
    for synthetic_split, synthetic_record in synthetic_files.items():
        if (
            not isinstance(synthetic_split, str)
            or not isinstance(synthetic_record, Mapping)
        ):
            raise ValueError("confirmatory synthetic integrity is malformed")
        synthetic_path = resolve(str(synthetic_record.get("path", "")))
        if (
            not synthetic_path.is_file()
            or synthetic_record.get("sha256")
            != hashlib.sha256(synthetic_path.read_bytes()).hexdigest()
        ):
            raise ValueError("confirmatory synthetic file hash does not verify")
    if synthetic_manifest is not None:
        registered_synthetic = synthetic_files.get(split)
        if not isinstance(registered_synthetic, Mapping):
            raise ValueError(
                "confirmatory source integrity does not register split synthetics"
            )
        synthetic_path = resolve(synthetic_manifest)
        if resolve(str(registered_synthetic.get("path", ""))) != synthetic_path:
            raise ValueError(
                "confirmatory synthetic input path does not match the frozen snapshot"
            )
    return hashlib.sha256(integrity_path.read_bytes()).hexdigest()


def _select_domain_balanced_candidates(
    qualified: Sequence[CandidateEntity],
    *,
    required_count: int,
    reserve_per_domain: int = 0,
) -> tuple[CandidateEntity, ...]:
    domains = tuple(REGISTERED_ENTITY_DOMAINS)
    if type(required_count) is not int or required_count <= 0:
        raise ValueError("required screening count must be a positive integer")
    if required_count % len(domains) != 0:
        raise ValueError("required screening count must divide evenly across domains")
    if type(reserve_per_domain) is not int or reserve_per_domain < 0:
        raise ValueError("reserve_per_domain must be a nonnegative integer")
    quota = required_count // len(domains) + reserve_per_domain
    selected_counts = {domain: 0 for domain in domains}
    selected: list[CandidateEntity] = []
    for candidate in qualified:
        if candidate.coarse_type not in selected_counts:
            raise ValueError("qualified candidate uses an unregistered entity domain")
        if selected_counts[candidate.coarse_type] >= quota:
            continue
        selected.append(candidate)
        selected_counts[candidate.coarse_type] += 1
    shortages = {
        domain: quota - selected_counts[domain]
        for domain in domains
        if selected_counts[domain] < quota
    }
    if shortages:
        details = ", ".join(
            f"{domain}:{missing}" for domain, missing in sorted(shortages.items())
        )
        raise ValueError(f"qualified candidates do not meet exact domain balance ({details})")
    return tuple(selected)


def _screening_reserve_per_domain(
    candidates: Sequence[CandidateEntity],
    config: FAConfig,
) -> int:
    """Return the pre-outcome reserve quota registered for one screening split."""
    if config.profile != "confirmatory":
        return 0
    splits = {candidate.split for candidate in candidates}
    if len(splits) != 1:
        raise ValueError("screening candidates must belong to exactly one split")
    split = next(iter(splits))
    try:
        return _CONFIRMATORY_RESERVE_PER_DOMAIN[split]
    except KeyError as error:
        raise ValueError("confirmatory screening split has no registered reserve") from error


def _load_verified_screening_completions(
    store: FAArtifactStore,
    manifest_path: str | Path,
    candidates: Sequence[CandidateEntity],
    questions: Sequence[ScreeningQuestion],
    config: FAConfig,
    *,
    tokenizer: Any | None = None,
    source_integrity_sha256: str | None = None,
) -> dict[str, tuple[str, str, str]]:
    shard = _require_verified_shard_kind(
        store,
        manifest_path,
        "screening_completion",
        "verified screening completion manifest",
    )
    candidate_splits = {candidate.split for candidate in candidates}
    if len(candidate_splits) != 1 or shard.namespace != next(iter(candidate_splits)):
        raise ValueError("screening completion namespace does not match candidate split")
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    model_hash, tokenizer_hash = _config_runtime_hashes(config)
    template_hash = config.chat_template_sha256 or _SMOKE_CHAT_TEMPLATE_SHA256
    expected_lineage = {
            "config_sha256": config.config_hash,
            "candidate_manifest_sha256": _candidate_manifest_sha256(candidates),
            "questions_manifest_sha256": _screening_question_manifest_sha256(
                questions
            ),
            "model_sha256": model_hash,
            "tokenizer_sha256": tokenizer_hash,
            "chat_template_sha256": template_hash,
    }
    if config.profile == "confirmatory":
        expected_lineage["source_integrity_sha256"] = source_integrity_sha256
    _verify_shard_lineage(shard, expected_lineage)

    by_entity = {candidate.entity_id: candidate for candidate in candidates}
    expected_questions = {
        (candidate.entity_id, index): question
        for candidate, candidate_questions in order_screening_questions(
            candidates, questions
        )
        for index, question in enumerate(candidate_questions)
    }
    if tokenizer is None:
        tokenizer = load_pinned_tokenizer(
            config, tokenizer_loader=_TOKENIZER_LOADER
        ).tokenizer
    grouped: dict[str, dict[int, str]] = {}
    rows = _read_json_rows(shard.data_path)
    for row in rows:
        entity_id = row.get("entity_id")
        candidate = by_entity.get(entity_id)
        index = row.get("question_index")
        question = expected_questions.get((entity_id, index))
        if (
            candidate is None
            or question is None
            or row.get("kind") != "screening_completion"
            or row.get("qid") != candidate.qid
            or row.get("question_id") != question.question_id
            or row.get("prompt") != question.prompt
            or row.get("source_provenance") != question.source_provenance
            or row.get("rendered_prompt_sha256")
            != _screening_rendered_prompt_sha256(
                config, tokenizer, question.prompt
            )
            or type(index) is not int
            or index not in {0, 1, 2}
            or row.get("accepted_aliases") != list(question.accepted_aliases)
            or row.get("generation") != dict(config.generation)
            or row.get("config_sha256") != config.config_hash
            or row.get("model_sha256") != model_hash
            or row.get("tokenizer_sha256") != tokenizer_hash
            or row.get("chat_template_sha256") != template_hash
            or row.get("status") != "completed"
            or row.get("exception_class") is not None
            or not isinstance(row.get("raw_output"), str)
            or not isinstance(row.get("answer_text"), str)
            or row.get("answer_text")
            != _screening_answer_text(row.get("raw_output"))
        ):
            raise ValueError("screening completion row has invalid data or provenance")
        entity_rows = grouped.setdefault(entity_id, {})
        if index in entity_rows:
            raise ValueError("screening completion contains a duplicate question index")
        entity_rows[index] = row["answer_text"]

    if set(grouped) != set(by_entity) or any(
        set(entity_rows) != {0, 1, 2} for entity_rows in grouped.values()
    ):
        raise ValueError(
            "screening completion must contain exactly three rows per candidate"
        )
    return {
        entity_id: (
            entity_rows[0],
            entity_rows[1],
            entity_rows[2],
        )
        for entity_id, entity_rows in grouped.items()
    }


def _candidate_manifest_sha256(candidates: Sequence[CandidateEntity]) -> str:
    return _sha256_json([asdict(candidate) for candidate in candidates])


def _screening_question_manifest_sha256(
    questions: Sequence[ScreeningQuestion],
) -> str:
    return _sha256_json(
        [
            asdict(question)
            for question in sorted(questions, key=lambda value: value.question_id)
        ]
    )


def _synthetic_manifest_sha256(
    synthetic: Sequence[SyntheticCandidate],
) -> str:
    return _sha256_json([asdict(candidate) for candidate in synthetic])


def _screening_parser_sha256() -> str:
    return _sha256_json(
        {
            "revision": "fa-screening-answer-v1",
            "implementation": inspect.getsource(_screening_answer_text),
            "rules": (
                "strip",
                "suffix-after-final-think-close",
                "last-nonempty-line",
                "suffix-after-final-colon",
                "single-matching-quote-pair",
            ),
        }
    )


def _matching_policy_sha256() -> str:
    return _sha256_json(
        {
            "revision": "fa-entity-matching-v5",
            "source_matching_policy_sha256": (
                fa_confirmatory_source.source_matching_policy_sha256()
            ),
            "character_tolerance": fa_entities.CHARACTER_TOLERANCE,
            "sentence_frame": fa_entities.TOKENIZER_SENTENCE_FRAME,
            "same_string_facts": fa_entities.SAME_STRING_EXPOSURE_FACTS,
            "confirmatory_reserve_per_domain": (
                _CONFIRMATORY_RESERVE_PER_DOMAIN
            ),
            "implementations": {
                "source_matchability_filter": inspect.getsource(
                    fa_confirmatory_source.filter_matchable_source_records
                ),
                "pseudonym_generator": inspect.getsource(
                    fa_confirmatory_synthetics.generate_synthetic_candidates
                ),
                "pseudonym_proposal": inspect.getsource(
                    fa_confirmatory_synthetics._pseudonym
                ),
                "matchability_filter": inspect.getsource(
                    _matchable_screening_candidates
                ),
                "selection": inspect.getsource(
                    _select_domain_balanced_candidates
                ),
                "match": inspect.getsource(
                    fa_entities.match_synthetic_entities
                ),
                "assignment": inspect.getsource(
                    fa_entities._deterministic_assignment
                ),
                "make_match": inspect.getsource(fa_entities._make_match),
                "surface": inspect.getsource(fa_entities._surface_compatible),
                "token_count": inspect.getsource(fa_entities._token_count),
                "same_string_prefix": inspect.getsource(
                    fa_entities.render_same_string_exposure_prefix
                ),
                "same_string_token_count": inspect.getsource(
                    fa_entities._same_string_token_count
                ),
            },
        }
    )


def _matchable_screening_candidates(
    candidates: Sequence[CandidateEntity],
    synthetic: Sequence[SyntheticCandidate],
    tokenizer: Any,
    *,
    required_variants: int = 1,
    generator_revision: str | None = None,
) -> tuple[CandidateEntity, ...]:
    """Keep candidates with the complete registered exact-surface reserve."""
    if type(required_variants) is not int or required_variants < 1:
        raise ValueError("required_variants must be a positive integer")
    matchable = []
    for candidate in candidates:
        compatible = [
            synthetic_candidate
            for synthetic_candidate in synthetic
            if (
                candidate.split == synthetic_candidate.split
                and (
                    generator_revision is None
                    or synthetic_candidate.generator_revision
                    == generator_revision
                )
                and (
                    required_variants == 1
                    or synthetic_candidate.candidate_id.startswith(
                        f"syn-{candidate.entity_id}-v"
                    )
                )
                and fa_entities._surface_compatible(
                    candidate,
                    synthetic_candidate,
                    tokenizer,
                )
            )
        ]
        if required_variants == 1 and compatible:
            matchable.append(candidate)
            continue
        if len(compatible) != required_variants:
            continue
        if len({row.candidate_id for row in compatible}) != required_variants:
            continue
        if (
            len({fa_entities._normal_form(row.name) for row in compatible})
            != required_variants
        ):
            continue
        matchable.append(candidate)
    return tuple(matchable)


def _render_screening_prompt(runner: Any, prompt: str) -> str:
    render = getattr(runner, "render_prompt", None)
    if not callable(render):
        return prompt
    value = render(prompt)
    if not isinstance(value, str):
        raise ValueError("screening runner rendered prompt must be text")
    return value


def _screening_rendered_prompt_sha256(
    config: FAConfig,
    tokenizer: Any,
    prompt: str,
) -> str:
    options = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if config.model_id.startswith("Qwen/Qwen3-"):
        options["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        **options,
    )
    if not isinstance(rendered, str):
        raise ValueError("screening tokenizer rendered prompt must be text")
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _screening_answer_text(raw_output: str) -> str:
    value = raw_output.strip()
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[1].strip()
    lines = tuple(line.strip() for line in value.splitlines() if line.strip())
    value = lines[-1] if lines else ""
    if ":" in value:
        value = value.rsplit(":", 1)[1].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def _build_manifest(config: FAConfig, root: Path, args: argparse.Namespace, *, confirmatory: bool) -> dict[str, Any]:
    store = FAArtifactStore(root)
    gate = None
    naturalness_shard = None
    if confirmatory:
        if config.profile != "confirmatory":
            raise ValueError("confirmatory construction requires the confirmatory config")
        smoke_config = FAConfig.from_json(_SMOKE_CONFIG_PATH)
        gate = _load_verified_pilot_gate(
            store, args.pilot_gate_manifest, smoke_config
        )
        if gate.get("status") != "passed":
            raise ValueError("confirmatory construction requires a verified passed pilot gate")
    elif config.profile != "smoke":
        raise ValueError("pilot construction requires the smoke config")
    if confirmatory:
        matches = _load_verified_screened_match_collection(
            store,
            args.matches_manifest,
            config,
        )
    else:
        matches = _load_verified_screened_matches(
            store, args.matches_manifest, config
        )
    if confirmatory:
        ratings, ratings_shard = _load_verified_naturalness_ratings(
            store,
            args.naturalness_ratings_manifest,
            config,
        )
        audit = audit_naturalness_manifest(matches, ratings)
        if not audit.accepted_pair_ids:
            raise ValueError("confirmatory construction requires accepted naturalness pairs")
        audited_matches = matches
        matches = _select_confirmatory_matches(config, matches, audit)
        naturalness_shard = _write_naturalness_audit(
            store, config, audited_matches, audit, ratings, ratings_shard
        )
    else:
        pilot_count = config.split_counts["pilot"]
        if len(matches) != pilot_count or any(
            match.split != "pilot" for match in matches
        ):
            raise ValueError(
                f"pilot construction requires exactly {pilot_count} pilot matches"
            )
    prepared = load_pinned_tokenizer(config, tokenizer_loader=_TOKENIZER_LOADER)
    factorial_rows = build_factorial_examples(config, matches, tokenizer=prepared.tokenizer)
    rows = factorial_rows + build_same_string_examples(
        config, matches, tokenizer=prepared.tokenizer
    )
    power_shard = None
    power_audit = None
    if confirmatory:
        power_audit, power_shard = _prepare_power_audit(
            store,
            config,
            factorial_rows,
            args.power_audit_manifest,
            run_registered=args.run_registered_power_audit,
        )
    manifest = build_manifest(config, rows, power_audit=power_audit)
    tokenizer_pin = _write_tokenizer_pin(store, config, prepared, manifest.manifest_sha256)
    probe_metadata = _write_probe_metadata(
        store,
        config,
        manifest.manifest_sha256,
        matches,
        rows,
    )
    capabilities = {}
    for namespace in sorted({row.split for row in rows}):
        namespace_rows = tuple(row for row in rows if row.split == namespace)
        capabilities[namespace] = _write_prompt_capability(
            store,
            config,
            manifest.manifest_sha256,
            namespace,
            namespace_rows,
            prepared.chat_template_sha256,
            tokenizer_pin,
            naturalness_shard,
        )

    if not confirmatory:
        pilot = capabilities.get("pilot")
        if pilot is None or len(capabilities) != 1:
            raise ValueError("pilot construction must emit exactly one pilot capability")
        return {
            "status": "built",
            "manifest": str(pilot.manifest_path),
            "count": len(rows),
            "manifest_sha256": manifest.manifest_sha256,
            "tokenizer_pin_manifest": str(tokenizer_pin.manifest_path),
            "probe_metadata_manifest": str(probe_metadata.manifest_path),
        }

    assert power_shard is not None and gate is not None
    missing_protected = _PROTECTED - capabilities.keys()
    if missing_protected:
        raise ValueError(
            "confirmatory manifest is missing protected endpoint capabilities: "
            + ", ".join(sorted(missing_protected))
        )
    index_row = _confirmatory_index_record(
        store,
        config,
        manifest.manifest_sha256,
        rows,
        capabilities,
        power_shard,
        tokenizer_pin,
        naturalness_shard,
    )
    index = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        f"confirmatory-index-{manifest.manifest_sha256[:16]}",
        [index_row],
        {
            "config_sha256": config.config_hash,
            "source_manifest_sha256": manifest.manifest_sha256,
            "power_audit_sha256": power_shard.sha256,
        },
        record_kind="confirmatory_index",
    )
    return {
        "status": "built",
        "manifest": str(index.manifest_path),
        "count": len(rows),
        "manifest_sha256": manifest.manifest_sha256,
        "power_audit_manifest": str(power_shard.manifest_path),
        "namespace_manifests": {
            namespace: str(shard.manifest_path)
            for namespace, shard in sorted(capabilities.items())
        },
        "tokenizer_pin_manifest": str(tokenizer_pin.manifest_path),
        "naturalness_audit_manifest": str(naturalness_shard.manifest_path),
        "naturalness_audit_sha256": naturalness_shard.sha256,
        "probe_metadata_manifest": str(probe_metadata.manifest_path),
    }


def _confirmatory_index_record(
    store,
    config,
    full_manifest_sha256,
    rows,
    capabilities,
    power_shard,
    tokenizer_pin,
    naturalness_audit=None,
):
    return {
        "kind": "confirmatory_index",
        "config_sha256": config.config_hash,
        "full_manifest_sha256": full_manifest_sha256,
        "power_audit_manifest": str(power_shard.manifest_path.relative_to(store.root)),
        "power_audit_sha256": power_shard.sha256,
        "tokenizer_pin_manifest": str(tokenizer_pin.manifest_path.relative_to(store.root)),
        "tokenizer_pin_sha256": tokenizer_pin.sha256,
        "naturalness_audit_manifest": (
            None
            if naturalness_audit is None
            else str(naturalness_audit.manifest_path.relative_to(store.root))
        ),
        "naturalness_audit_sha256": (
            None if naturalness_audit is None else naturalness_audit.sha256
        ),
        "capabilities": {
            namespace: {
                "manifest_path": str(shard.manifest_path.relative_to(store.root)),
                "sha256": shard.sha256,
                "example_ids": [row.example_id for row in rows if row.split == namespace],
                "canonical_payload_sha256s": [
                    row.canonical_payload_sha256 for row in rows if row.split == namespace
                ],
            }
            for namespace, shard in sorted(capabilities.items())
        },
    }


def _audit_manifest(config: FAConfig, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(FAArtifactStore(root), args.manifest, config)
    factorial = tuple(row for row in manifest.examples if row.block == "factorial")
    same_string = tuple(row for row in manifest.examples if row.block == "same_string")
    prepared = load_pinned_tokenizer(config, tokenizer_loader=_TOKENIZER_LOADER)
    audit = audit_dataset(factorial, same_string, tokenizer=prepared.tokenizer)
    return {"status": "passed" if audit.passed else "failed", "checks": dict(audit.checks), "violations": list(audit.violations)}


def _select_confirmatory_matches(
    config: FAConfig,
    matches: Sequence[EntityMatch],
    audit: NaturalnessAudit,
) -> tuple[EntityMatch, ...]:
    """Fill every registered split/domain quota from accepted pairs in hash order."""

    if config.profile != "confirmatory":
        raise ValueError("reserve selection requires the confirmatory config")
    accepted = frozenset(audit.accepted_pair_ids)
    if accepted & frozenset(audit.excluded_pair_ids):
        raise ValueError("naturalness audit has overlapping accepted and excluded pairs")
    by_pair = {match.pair_id: match for match in matches}
    if len(by_pair) != len(tuple(matches)) or not accepted.issubset(by_pair):
        raise ValueError("naturalness audit does not match the sealed pair manifest")

    selected: list[EntityMatch] = []
    domain_count = len(REGISTERED_ENTITY_DOMAINS)
    for split, split_count in sorted(config.split_counts.items()):
        if split_count % domain_count:
            raise ValueError("confirmatory split counts must be divisible by four domains")
        quota = split_count // domain_count
        for domain in REGISTERED_ENTITY_DOMAINS:
            candidates = sorted(
                (
                    match
                    for match in matches
                    if match.pair_id in accepted
                    and match.split == split
                    and match.coarse_type == domain
                ),
                key=lambda match: (
                    hashlib.sha256(match.pair_id.encode("utf-8")).hexdigest(),
                    match.pair_id,
                ),
            )
            if len(candidates) < quota:
                raise ValueError(
                    f"naturalness exclusions leave fewer than {quota} accepted "
                    f"pairs for {split}/{domain}"
                )
            selected.extend(candidates[:quota])
    return tuple(sorted(selected, key=lambda match: (match.split, match.pair_id)))


def _run_generation(config: FAConfig, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.namespace in _PROTECTED:
        raise ValueError(
            "fa-run-generation is generic-only and cannot evaluate protected test namespaces"
        )
    store = FAArtifactStore(root)
    manifest = _load_manifest(store, args.manifest, config)
    if any(row.split != args.namespace for row in manifest.examples):
        raise ValueError("explicit generation manifest contains another namespace")
    runner = HFModelRunner(config)
    validate_runner_binding(
        runner,
        config,
        expected_chat_template_sha256=manifest.chat_template_sha256,
    )
    shard = run_generation_shard(
        runner,
        manifest,
        store,
        args.shard_id,
        config=config,
        namespace=args.namespace,
        resume_partial=args.resume,
    )
    records = _load_verified_generation_sidecar(
        store, shard.manifest_path, manifest, config
    )
    if not all(record["status"] == "completed" for record in records):
        exception_classes = sorted(
            {
                str(record.get("exception_class"))
                for record in records
                if record["status"] == "infrastructure_failure"
            }
        )
        return {
            "status": "infrastructure_failure",
            "error": {
                "message": "generation infrastructure failure; retry the existing artifact transaction",
                "type": ",".join(exception_classes) or "InfrastructureFailure",
            },
            "shard_manifest": str(shard.manifest_path),
            "sha256": shard.sha256,
        }
    return {
        "status": "generated",
        "shard_manifest": str(shard.manifest_path),
        "sha256": shard.sha256,
    }


def _extract_activations(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    if args.namespace in _PROTECTED:
        raise ValueError(
            "fa-extract-activations is generic-only and cannot evaluate protected test namespaces"
        )
    store = FAArtifactStore(root)
    manifest = _load_manifest(store, args.manifest, config)
    if manifest.namespace != args.namespace:
        raise ValueError("explicit activation manifest contains another namespace")
    layers = _registered_extraction_layers(config, args.layers)
    shard_id = _safe_cli_id(args.shard_id, "activation shard-id")
    model_runner = HFModelRunner(config)
    validate_runner_binding(
        model_runner,
        config,
        expected_chat_template_sha256=manifest.chat_template_sha256,
    )
    runner = _ACTIVATION_RUNNER_FACTORY(
        model_runner.model,
        model_runner.tokenizer,
        model_id=config.model_id,
        model_revision=config.model_revision,
        tokenizer_revision=config.tokenizer_revision,
    )
    destination = (
        root.absolute()
        / "runs"
        / "familiarity_answerability"
        / config.run_id
        / "activations"
        / args.namespace
        / f"{shard_id}.npz"
    )
    shard = _ACTIVATION_SHARD_WRITER(
        runner,
        manifest.examples,
        layers,
        destination=destination,
    )
    return {
        "status": "extracted",
        "manifest": str(shard.manifest_path),
        "request_sha256": shard.request_sha256,
        "row_count": shard.row_count,
    }


def _analyze_pilot_activations(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Analyze only the frozen development pilot and publish immutable OOF evidence."""

    if config.profile != "smoke":
        raise ValueError("pilot activation analysis requires a smoke config")
    store = FAArtifactStore(root)
    manifest = _load_manifest(store, args.manifest, config)
    if manifest.namespace != "pilot":
        raise ValueError("pilot activation analysis requires the pilot namespace")
    gate = _load_verified_pilot_gate(store, args.pilot_gate_manifest, config)
    if gate["status"] != "passed":
        raise ValueError("pilot activation analysis requires a verified passed gate")
    gate_shard = _require_verified_shard_kind(
        store,
        args.pilot_gate_manifest,
        "pilot_gate",
        "verified pilot gate sidecar manifest",
    )
    gate_rows = _read_json_rows(gate_shard.data_path)
    if gate_rows[0].get("prompt_manifest_sha256") != manifest.shard_sha256:
        raise ValueError("pilot gate does not bind the explicit prompt manifest")

    activation_manifest = Path(args.activation_manifest).absolute()
    if not activation_manifest.is_relative_to(store.root):
        raise ValueError("activation manifest must stay under the FA artifact root")
    activation_sidecar = _read_json_object(activation_manifest)
    npz_name = activation_sidecar.get("npz_file")
    if not isinstance(npz_name, str) or Path(npz_name).name != npz_name:
        raise ValueError("activation manifest has an invalid NPZ path")
    activation_shard = resume_activation_shard(
        activation_manifest.parent / npz_name
    )
    if activation_shard.manifest_path != activation_manifest:
        raise ValueError("activation manifest path does not match its shard")
    activations = load_activation_records(activation_manifest)
    rows = build_pilot_analysis_rows(manifest.examples, activations)
    result = analyze_pilot_rows(
        rows,
        permutation_seeds=PILOT_PERMUTATION_SEEDS,
    )

    if not _PILOT_ANALYSIS_SPEC_PATH.is_file() or any(
        not path.is_file() for path in _PILOT_ANALYSIS_AMENDMENTS
    ):
        raise ValueError("pilot analysis registration files are missing")
    spec_sha256 = _file_sha256(_PILOT_ANALYSIS_SPEC_PATH)
    amendment_sha256s = {
        path.name: _file_sha256(path) for path in _PILOT_ANALYSIS_AMENDMENTS
    }
    implementation_sha256 = _file_sha256(
        Path(inspect.getfile(analyze_pilot_rows))
    )
    lineage = {
        "config_sha256": config.config_hash,
        "prompt_manifest": str(
            manifest.shard_manifest_path.relative_to(store.root)
        ),
        "prompt_manifest_sha256": manifest.shard_sha256,
        "pilot_gate_manifest": str(gate_shard.manifest_path.relative_to(store.root)),
        "pilot_gate_sha256": gate_shard.sha256,
        "pilot_gate_evidence_sha256": gate["evidence_sha256"],
        "activation_manifest": str(activation_manifest.relative_to(store.root)),
        "activation_manifest_sha256": _file_sha256(activation_manifest),
        "activation_request_sha256": activation_shard.request_sha256,
        "activation_npz_sha256": activation_shard.npz_sha256,
        "activation_index_sha256": activation_shard.index_sha256,
        "analysis_spec": str(_PILOT_ANALYSIS_SPEC_PATH.relative_to(store.root)),
        "analysis_spec_sha256": spec_sha256,
        "analysis_amendment_sha256s": amendment_sha256s,
        "analysis_implementation_sha256": implementation_sha256,
        "analysis_sha256": result.analysis_sha256,
        "permutation_seed_sha256": _sha256_json(PILOT_PERMUTATION_SEEDS),
    }
    shard_id = _safe_cli_id(args.shard_id, "pilot analysis shard-id")
    predictions = _write_or_resume_records(
        store,
        run_id=config.run_id,
        namespace="pilot",
        shard_id=f"{shard_id}-predictions",
        rows=result.prediction_records,
        lineage=lineage,
        record_kind="pilot_predictions",
        allow_resume=True,
    )
    metrics = _write_or_resume_records(
        store,
        run_id=config.run_id,
        namespace="pilot",
        shard_id=f"{shard_id}-metrics",
        rows=result.metric_records,
        lineage={
            **lineage,
            "predictions_manifest": str(
                predictions.manifest_path.relative_to(store.root)
            ),
            "predictions_sha256": predictions.sha256,
        },
        record_kind="pilot_metrics",
        allow_resume=True,
    )
    return {
        "status": "analyzed",
        "claim_scope": "development_only_model_specific_decodability",
        "example_count": result.example_count,
        "group_count": result.group_count,
        "metric_count": len(result.metric_records),
        "prediction_count": len(result.prediction_records),
        "analysis_sha256": result.analysis_sha256,
        "predictions_manifest": str(predictions.manifest_path),
        "metrics_manifest": str(metrics.manifest_path),
    }


def _materialize_probe_rows(
    config: FAConfig,
    root: Path,
    args: argparse.Namespace,
    *,
    authorization: ProbeTestAuthorization | None = None,
) -> dict[str, Any]:
    """Generate, extract, and seal compact evidence for all registered F2A tasks."""

    store = FAArtifactStore(root)
    manifest = _load_manifest(store, args.manifest, config)
    if manifest.namespace != args.namespace:
        raise ValueError("probe materialization manifest uses another namespace")
    if args.namespace == "probe_test":
        if not isinstance(authorization, ProbeTestAuthorization):
            raise ValueError(
                "probe_test materialization requires an active one-use authorization"
            )
        if store.endpoint_state("probe_test", manifest.shard_manifest_path) != "unlocked_once":
            raise ValueError("probe_test must be unlocked before materialization")
    elif authorization is not None:
        raise ValueError("probe authorization is valid only for probe_test materialization")
    metadata, metadata_shard = _load_probe_metadata_manifest(
        store,
        args.metadata_manifest,
        config,
        expected_full_manifest_sha256=manifest.full_manifest_sha256,
        expected_example_ids=frozenset(row.example_id for row in manifest.examples),
    )
    shard_id = _safe_cli_id(args.shard_id, "probe materialization shard-id")
    if args.resume:
        resumed = _resume_probe_materialization(
            store,
            config,
            manifest,
            metadata_shard,
            shard_id=shard_id,
            namespace=args.namespace,
            authorization=authorization,
        )
        if resumed is not None:
            return resumed
    model_runner = HFModelRunner(config)
    validate_runner_binding(
        model_runner,
        config,
        expected_chat_template_sha256=manifest.chat_template_sha256,
    )

    generation_shard = run_generation_shard(
        model_runner,
        manifest,
        store,
        f"{shard_id}-generation",
        config=config,
        namespace=args.namespace,
    )
    generation_records = _load_verified_generation_sidecar(
        store, generation_shard.manifest_path, manifest, config
    )
    if not all(record["status"] == "completed" for record in generation_records):
        exception_classes = sorted(
            {
                str(record.get("exception_class"))
                for record in generation_records
                if record["status"] == "infrastructure_failure"
            }
        )
        return {
            "status": "infrastructure_failure",
            "error": {
                "message": "probe generation failed; retry the same materialization transaction",
                "type": ",".join(exception_classes) or "InfrastructureFailure",
            },
            "generation_manifest": str(generation_shard.manifest_path),
        }

    activation_runner = _ACTIVATION_RUNNER_FACTORY(
        model_runner.model,
        model_runner.tokenizer,
        model_id=config.model_id,
        model_revision=config.model_revision,
        tokenizer_revision=config.tokenizer_revision,
    )
    activation_destination = (
        root.absolute()
        / "runs"
        / "familiarity_answerability"
        / config.run_id
        / "activations"
        / args.namespace
        / f"{shard_id}.npz"
    )
    activation_shard = _ACTIVATION_SHARD_WRITER(
        activation_runner,
        manifest.examples,
        tuple(range(26)),
        destination=activation_destination,
    )
    activations = load_activation_records(activation_shard.manifest_path)
    scored = _score_rows(generation_records, manifest)
    unsupported_outcomes = _unsupported_outcomes(scored, manifest.examples)
    scorer = _PROBE_SCORER_FACTORY(
        model_runner.model, model_runner.tokenizer, config
    )
    rows, output_evidence = _PROBE_ROW_MATERIALIZER(
        manifest.examples,
        activations,
        scorer,
        metadata,
        unsupported_outcomes=unsupported_outcomes,
    )

    activation_by_id = {row.example_id: row for row in activations}
    output_by_id = {row.example_id: row for row in output_evidence}
    compact_records = tuple(
        {
            "kind": "probe_rows",
            "schema_version": 2,
            "example_id": example.example_id,
            "source_sha256": example.canonical_payload_sha256,
            "activation_sha256": activation_by_id[example.example_id].activation_sha256,
            "output_evidence_sha256": output_by_id[example.example_id].sha256,
            "output_evidence": dict(output_by_id[example.example_id].canonical_payload),
            "unsupported_outcome": (
                None
                if example.example_id not in unsupported_outcomes
                else asdict(unsupported_outcomes[example.example_id])
            ),
        }
        for example in sorted(manifest.examples, key=lambda row: row.example_id)
    )
    task_identities = _task_source_identity_records(manifest.examples)
    lineage = {
        "config_sha256": config.config_hash,
        "prompt_manifest": str(manifest.shard_manifest_path.relative_to(store.root)),
        "prompt_manifest_sha256": manifest.shard_sha256,
        "generation_sidecar_manifest": str(
            generation_shard.manifest_path.relative_to(store.root)
        ),
        "generation_sidecar_sha256": generation_shard.sha256,
        "activation_manifest": str(
            activation_shard.manifest_path.relative_to(store.root)
        ),
        "activation_manifest_sha256": _file_sha256(
            activation_shard.manifest_path
        ),
        "activation_request_sha256": activation_shard.request_sha256,
        "activation_npz_sha256": activation_shard.npz_sha256,
        "activation_index_sha256": activation_shard.index_sha256,
        "metadata_manifest": str(metadata_shard.manifest_path.relative_to(store.root)),
        "metadata_manifest_sha256": metadata_shard.sha256,
        "task_source_identities_sha256": _sha256_json(task_identities),
        "materialization_schema_sha256": _probe_materialization_schema_sha256(),
    }
    if authorization is not None:
        lineage["probe_authorization_sha256"] = authorization.sha256
    probe_shard = _write_or_resume_records(
        store,
        run_id=config.run_id,
        namespace=args.namespace,
        shard_id=shard_id,
        rows=compact_records,
        lineage=lineage,
        record_kind="probe_rows",
        allow_resume=bool(args.resume),
    )
    return {
        "status": "materialized",
        "probe_rows_manifest": str(probe_shard.manifest_path),
        "generation_manifest": str(generation_shard.manifest_path),
        "activation_manifest": str(activation_shard.manifest_path),
        "example_count": len(manifest.examples),
        "probe_row_count": len(rows),
        "compact_evidence_count": len(compact_records),
    }


def _resume_probe_materialization(
    store: FAArtifactStore,
    config: FAConfig,
    prompt: VerifiedPromptManifest,
    metadata_shard: Any,
    *,
    shard_id: str,
    namespace: str,
    authorization: ProbeTestAuthorization | None,
) -> dict[str, Any] | None:
    candidate = (
        store.root
        / "runs"
        / "familiarity_answerability"
        / config.run_id
        / "shards"
        / namespace
        / f"{shard_id}.jsonl.manifest.json"
    )
    if not candidate.exists():
        return None
    shard = _require_verified_shard_kind(
        store, candidate, "probe_rows", "resumed probe rows manifest"
    )
    if shard.namespace != namespace:
        raise ValueError("resumed probe rows use another namespace")
    expected_lineage = {
        "config_sha256": config.config_hash,
        "prompt_manifest_sha256": prompt.shard_sha256,
        "metadata_manifest_sha256": metadata_shard.sha256,
        "materialization_schema_sha256": _probe_materialization_schema_sha256(),
    }
    if authorization is not None:
        expected_lineage["probe_authorization_sha256"] = authorization.sha256
    _verify_shard_lineage(shard, expected_lineage)
    records = _read_json_rows(shard.data_path)
    rows = _reconstruct_compact_probe_rows(
        store,
        shard,
        records,
        config,
        expected_namespace=namespace,
        authorization=authorization,
    )
    return {
        "status": "recovered",
        "probe_rows_manifest": str(shard.manifest_path),
        "example_count": len(prompt.examples),
        "probe_row_count": len(rows),
        "compact_evidence_count": len(records),
    }


def _registered_extraction_layers(config: FAConfig, raw: str | None) -> tuple[int, ...]:
    registered = tuple(range(26))
    if config.profile == "confirmatory":
        if raw is not None and _parse_layer_ids(raw) != registered:
            raise ValueError("confirmatory extraction must use all 26 registered layers")
        return registered
    if raw is None:
        raise ValueError("smoke activation extraction requires explicit --layers")
    return _parse_layer_ids(raw)


def _parse_layer_ids(raw: str) -> tuple[int, ...]:
    if not isinstance(raw, str) or not raw:
        raise ValueError("activation layers must be a comma-separated integer sequence")
    try:
        layers = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise ValueError("activation layers must be a comma-separated integer sequence") from error
    if not layers or any(value < 0 for value in layers) or layers != tuple(sorted(set(layers))):
        raise ValueError("activation layers must be unique increasing nonnegative integers")
    return layers


def _safe_cli_id(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError(f"{field} is invalid")
    if value in {".", ".."} or ".." in value:
        raise ValueError(f"{field} is invalid")
    return value

def _unsupported_outcomes(
    scored: Sequence[Any], examples: Sequence[FAExample]
) -> dict[str, UnsupportedAnswerOutcome]:
    scored_by_id = {row.example_id: row for row in scored}
    if len(scored_by_id) != len(tuple(scored)):
        raise ValueError("scored outputs contain duplicate example IDs")
    expected_ids = {row.example_id for row in examples}
    if set(scored_by_id) != expected_ids:
        raise ValueError("scored outputs do not match the exact example IDs")
    result: dict[str, UnsupportedAnswerOutcome] = {}
    for example in examples:
        if example.answerability not in {"distractor_bound", "code_absent"}:
            continue
        scored_row = scored_by_id[example.example_id]
        status = (
            "missing"
            if not scored_row.completed
            else "valid" if scored_row.valid_format else "invalid"
        )
        result[example.example_id] = UnsupportedAnswerOutcome(
            example.example_id,
            scored_row.answer_attempt,
            status,
        )
    return result


def _probe_materialization_schema_sha256() -> str:
    return _sha256_json(
        {
            "schema_version": 2,
            "record_fields": [
                "activation_sha256",
                "example_id",
                "kind",
                "output_evidence",
                "output_evidence_sha256",
                "schema_version",
                "source_sha256",
                "unsupported_outcome",
            ],
            "reconstruction": "prompt+activation+metadata+generation+exact_output_evidence",
        }
    )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_probes(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    store = FAArtifactStore(root)
    train_rows, train_shard = _load_probe_rows_manifest(
        store,
        args.train_rows_manifest,
        config,
        expected_namespace="mechanism_train",
    )
    validation_rows, validation_shard = _load_probe_rows_manifest(
        store,
        args.validation_rows_manifest,
        config,
        expected_namespace="locked_validation",
    )
    task_identities, prompt_shard = _load_prompt_task_source_identities(
        store,
        args.probe_test_manifest,
        config,
        expected_namespace="probe_test",
    )
    train_by_task = _probe_rows_by_task(train_rows)
    validation_by_task = _probe_rows_by_task(validation_rows)
    protected_ids = {
        identity.example_id
        for identities in task_identities.values()
        for identity in identities
    }
    selections = {
        task: _PROBE_SELECTOR(
            train_by_task[task],
            validation_by_task[task],
            protected_test_ids=protected_ids,
        )
        for task in TASKS
    }
    nulls: dict[str, tuple[NullSelectionResult, ...]] = {}
    for task in TASKS:
        kwargs: dict[str, Any] = {
            "protected_test_ids": protected_ids,
            "probe_test_source_identities": task_identities[task],
        }
        if config.profile == "smoke":
            kwargs.update(
                seeds=(2026072201,),
                _allow_test_seed_override=True,
            )
        nulls[task] = tuple(
            _PROBE_NULL_SELECTOR(
                train_by_task[task], validation_by_task[task], **kwargs
            )
        )
    selection_bundle_hash = f2a_selection_bundle_hash(selections)
    task_identities_sha256 = _task_source_identities_sha256(task_identities)
    mode = "registered_confirmatory" if config.profile == "confirmatory" else "smoke_rehearsal"
    row = {
        "kind": "selection_manifest",
        "schema_version": 1,
        "config_sha256": config.config_hash,
        "selection_bundle_hash": selection_bundle_hash,
        "probe_test_prompt_sha256": prompt_shard.sha256,
        "probe_test_task_identities_sha256": task_identities_sha256,
        "selection_mode": mode,
        "selections": {
            task: selections[task].to_record() for task in TASKS
        },
        "null_selections": {
            task: [result.to_record() for result in nulls[task]]
            for task in TASKS
        },
    }
    shard = store.write_completed_shard(
        config.run_id,
        "locked_validation",
        _safe_cli_id(args.shard_id, "selection shard-id"),
        (row,),
        {
            "config_sha256": config.config_hash,
            "train_probe_rows_sha256": train_shard.sha256,
            "validation_probe_rows_sha256": validation_shard.sha256,
            "probe_test_prompt_sha256": prompt_shard.sha256,
            "probe_test_task_identities_sha256": task_identities_sha256,
            "selection_bundle_hash": selection_bundle_hash,
            "selection_mode": mode,
        },
        record_kind="selection_manifest",
    )
    return {
        "status": "selected",
        "selection_manifest": str(shard.manifest_path),
        "selection_bundle_hash": selection_bundle_hash,
        "selection_mode": mode,
        "null_count": sum(len(values) for values in nulls.values()),
    }


def _seal_probe_selection(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    store = FAArtifactStore(root)
    selection = _load_f2a_selection_bundle(
        store, args.selection_manifest, config
    )
    task_identities, prompt_shard = _load_prompt_task_source_identities(
        store,
        args.probe_test_manifest,
        config,
        expected_namespace="probe_test",
    )
    if selection.probe_test_prompt_sha256 != prompt_shard.sha256:
        raise ValueError("selection bundle does not bind the probe_test prompt capability")
    if selection.probe_test_task_identities_sha256 != _task_source_identities_sha256(
        task_identities
    ):
        raise ValueError("selection bundle does not bind probe_test task identities")
    sealed_path = store.seal_endpoint(
        "probe_test",
        (prompt_shard,),
        {
            "preregistration": hashlib.sha256(
                _PREREGISTRATION_PATH.read_bytes()
            ).hexdigest(),
            "selection_manifest": selection.selection_bundle_hash,
        },
    )
    return {
        "status": "sealed",
        "endpoint": "probe_test",
        "endpoint_state": "sealed",
        "sealed_state": str(sealed_path),
        "selection_bundle_hash": selection.selection_bundle_hash,
    }


def _seal_behavior_test(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    store = FAArtifactStore(root)
    manifest = _load_manifest(
        store, args.behavior_test_manifest, config
    )
    if manifest.namespace != "behavior_test":
        raise ValueError("behavior sealing requires a behavior_test prompt manifest")
    try:
        store.verify_endpoint_artifact(
            "behavior_test", manifest.shard_manifest_path
        )
    except ValueError as error:
        if str(error) != "behavior_test is not sealed":
            raise
    else:
        raise ValueError("behavior_test is already sealed")

    preregistration_sha256 = hashlib.sha256(
        _PREREGISTRATION_PATH.read_bytes()
    ).hexdigest()
    selection_record = {
        "kind": "behavior_selection_manifest",
        "schema_version": 1,
        "config_sha256": config.config_hash,
        "prompt_manifest_sha256": manifest.shard_sha256,
        "full_manifest_sha256": manifest.full_manifest_sha256,
        "preregistration_sha256": preregistration_sha256,
        "bootstrap_seed": config.bootstrap_seed,
        "bootstrap_replicates": config.bootstrap_replicates,
        "thresholds": dict(config.thresholds),
        "selection_frozen_before_endpoint_open": True,
    }
    selection_sha256 = _sha256_json(selection_record)
    selection_shard = _write_or_resume_single_record(
        store,
        run_id=config.run_id,
        namespace="locked_validation",
        shard_id=f"behavior-selection-{manifest.shard_sha256[:16]}",
        row=selection_record,
        lineage={
            "config_sha256": config.config_hash,
            "prompt_manifest_sha256": manifest.shard_sha256,
            "preregistration_sha256": preregistration_sha256,
            "selection_sha256": selection_sha256,
        },
        record_kind="behavior_selection_manifest",
    )
    sealed_path = store.seal_endpoint(
        "behavior_test",
        (store.verify_shard(manifest.shard_manifest_path),),
        {
            "preregistration": preregistration_sha256,
            "selection_manifest": selection_sha256,
        },
    )
    return {
        "status": "sealed",
        "endpoint": "behavior_test",
        "endpoint_state": "sealed",
        "sealed_state": str(sealed_path),
        "selection_manifest": str(selection_shard.manifest_path),
        "selection_sha256": selection_sha256,
    }


def _evaluate_probe_test(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    store = FAArtifactStore(root)
    selection = _load_f2a_selection_bundle(
        store, args.selection_manifest, config
    )
    task_identities, prompt_shard = _load_prompt_task_source_identities(
        store,
        args.probe_test_manifest,
        config,
        expected_namespace="probe_test",
    )
    if selection.probe_test_prompt_sha256 != prompt_shard.sha256:
        raise ValueError("selection bundle does not bind the probe_test prompt capability")
    if selection.probe_test_task_identities_sha256 != _task_source_identities_sha256(
        task_identities
    ):
        raise ValueError("selection bundle does not bind probe_test task identities")
    store.verify_endpoint_artifact("probe_test", prompt_shard.manifest_path)
    state = store.endpoint_state("probe_test", prompt_shard.manifest_path)
    if state == "closed":
        closed = store.read_closed_metrics(
            "probe_test", prompt_shard.manifest_path
        )
        return {
            "status": "recovered",
            "endpoint_state": "closed",
            "metrics_manifest": str(closed.metrics_artifact.manifest_path),
            "selection_bundle_hash": selection.selection_bundle_hash,
        }
    receipt = store.unlock_or_resume_endpoint(
        "probe_test", prompt_shard.manifest_path
    )
    authorization = ProbeTestAuthorization.from_unlock_receipt(receipt)
    materialized = _materialize_probe_rows(
        config,
        root,
        argparse.Namespace(
            manifest=str(prompt_shard.manifest_path),
            metadata_manifest=args.metadata_manifest,
            namespace="probe_test",
            shard_id=args.shard_id,
            resume=True,
        ),
        authorization=authorization,
    )
    if materialized["status"] == "infrastructure_failure":
        return {
            **materialized,
            "endpoint_state": "unlocked_once",
            "selection_bundle_hash": selection.selection_bundle_hash,
        }
    probe_rows_manifest = materialized["probe_rows_manifest"]
    rows, _ = _load_probe_rows_manifest(
        store,
        probe_rows_manifest,
        config,
        expected_namespace="probe_test",
        authorization=authorization,
        expected_prompt_sha256=prompt_shard.sha256,
        expected_task_identities=task_identities,
    )
    rows_by_task = _probe_rows_by_task(rows)
    bundle = _PROBE_BUNDLE_EVALUATOR(
        selection.selections,
        authorization,
        rows_by_task,
        null_selections_by_task=selection.null_selections,
        store=store,
        endpoint_manifest_path=prompt_shard.manifest_path,
    )
    closed = store.read_closed_metrics("probe_test", prompt_shard.manifest_path)
    return {
        "status": "evaluated",
        "endpoint_state": "closed",
        "metrics_manifest": str(closed.metrics_artifact.manifest_path),
        "selection_bundle_hash": selection.selection_bundle_hash,
        "bundle_sha256": bundle.sha256,
        "gate_status": bundle.gates.status,
    }


def _build_evidence_report(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    behavior_manifest = getattr(args, "behavior_test_manifest", None)
    probe_manifest = getattr(args, "probe_test_manifest", None)
    selection_manifest = getattr(args, "selection_manifest", None)
    if behavior_manifest is None and probe_manifest is None:
        raise ValueError(
            "fa-build-report requires at least one closed behavior_test or probe_test manifest"
        )
    if (probe_manifest is None) != (selection_manifest is None):
        raise ValueError(
            "probe_test reporting requires both --probe-test-manifest and --selection-manifest"
        )

    store = FAArtifactStore(root)
    if behavior_manifest is not None:
        _verify_artifact_run_id(Path(behavior_manifest), config.run_id)
    if probe_manifest is not None:
        _verify_artifact_run_id(Path(probe_manifest), config.run_id)
    behavior = (
        None
        if behavior_manifest is None
        else load_closed_f1_evidence(store, behavior_manifest)
    )
    if behavior is not None and behavior.gate.config_hash != config.config_hash:
        raise ValueError("closed F1 evidence belongs to another config")

    f2a = None
    if probe_manifest is not None:
        selection = _load_f2a_selection_bundle(store, selection_manifest, config)
        f2a = load_closed_f2a_evidence(
            store,
            probe_manifest,
            selections=selection.selections,
        )

    output = _root_scoped_output(root, args.output)
    report = build_report(
        behavior=behavior,
        f2a=f2a,
        f2b=None,
        circuit=None,
        output=output,
    )
    return {
        "status": "reported",
        "report": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "phases": {
            "F1": "evaluated" if behavior is not None else "unavailable",
            "F2A": "evaluated" if f2a is not None else "unavailable",
            "F2B": "skipped",
            "F3": "skipped",
        },
    }


def _root_scoped_output(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("report output must be a nonempty path")
    base = root.absolute().resolve()
    candidate = Path(value)
    destination = (
        candidate.absolute().resolve()
        if candidate.is_absolute()
        else (base / candidate).resolve()
    )
    try:
        destination.relative_to(base)
    except ValueError as error:
        raise ValueError("report output must remain inside --root") from error
    return destination


def _load_probe_metadata_manifest(
    store: FAArtifactStore,
    path: str | Path,
    config: FAConfig,
    *,
    expected_full_manifest_sha256: str,
    expected_example_ids: frozenset[str],
) -> tuple[dict[str, Any], Any]:
    shard = _require_verified_shard_kind(
        store, path, "probe_metadata", "verified probe metadata manifest"
    )
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    rows = _read_json_rows(shard.data_path)
    if len(rows) != 1:
        raise ValueError("probe metadata manifest must contain one record")
    record = rows[0]
    expected = {
        "kind",
        "schema_version",
        "config_sha256",
        "full_manifest_sha256",
        "manifest_revision",
        "rows",
    }
    if (
        set(record) != expected
        or record.get("kind") != "probe_metadata"
        or record.get("schema_version") != 1
        or record.get("config_sha256") != config.config_hash
        or record.get("full_manifest_sha256") != expected_full_manifest_sha256
    ):
        raise ValueError("probe metadata manifest identity is invalid")
    metadata_rows = record.get("rows")
    if not isinstance(metadata_rows, list) or not metadata_rows:
        raise ValueError("probe metadata rows are missing")
    expected_row_schema = {
        "example_id",
        "entity_id",
        "template_id",
        "relation_id",
        "domain",
        "condition",
    }
    if any(set(row) != expected_row_schema for row in metadata_rows):
        raise ValueError("probe metadata row schema is invalid")
    observed_ids = [row.get("example_id") for row in metadata_rows]
    if len(set(observed_ids)) != len(observed_ids) or not expected_example_ids.issubset(
        observed_ids
    ):
        raise ValueError("probe metadata does not cover the exact prompt examples")
    metadata = {
        "manifest_revision": record["manifest_revision"],
        "rows": [
            row for row in metadata_rows if row["example_id"] in expected_example_ids
        ],
    }
    if {row["example_id"] for row in metadata["rows"]} != expected_example_ids:
        raise ValueError("probe metadata subset does not match prompt examples")
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config.config_hash,
            "source_manifest_sha256": expected_full_manifest_sha256,
            "metadata_sha256": _sha256_json(record),
        },
    )
    return metadata, shard


def _load_probe_rows_manifest(
    store: FAArtifactStore,
    path: str | Path,
    config: FAConfig,
    *,
    expected_namespace: str,
    authorization: ProbeTestAuthorization | None = None,
    expected_prompt_sha256: str | None = None,
    expected_task_identities: Mapping[
        str, Sequence[ProbeSourceIdentity]
    ] | None = None,
) -> tuple[tuple[ProbeRow, ...], Any]:
    if expected_namespace == "probe_test" and not isinstance(
        authorization, ProbeTestAuthorization
    ):
        raise ValueError("protected probe rows require a probe_test authorization")
    shard = _require_verified_shard_kind(
        store, path, "probe_rows", "verified probe rows manifest"
    )
    if shard.namespace != expected_namespace:
        raise ValueError("probe rows artifact uses the wrong namespace")
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    expected_lineage: dict[str, Any] = {"config_sha256": config.config_hash}
    if expected_prompt_sha256 is not None:
        expected_lineage["prompt_manifest_sha256"] = expected_prompt_sha256
    if expected_task_identities is not None:
        expected_lineage["task_source_identities_sha256"] = (
            _task_source_identities_sha256(expected_task_identities)
        )
    _verify_shard_lineage(shard, expected_lineage)
    raw_rows = _read_json_rows(shard.data_path)
    compact_schema = {
        "kind",
        "schema_version",
        "example_id",
        "source_sha256",
        "activation_sha256",
        "output_evidence_sha256",
        "output_evidence",
        "unsupported_outcome",
    }
    if raw_rows and all(
        set(record) == compact_schema
        and record.get("kind") == "probe_rows"
        and record.get("schema_version") == 2
        for record in raw_rows
    ):
        rows = _reconstruct_compact_probe_rows(
            store,
            shard,
            raw_rows,
            config,
            expected_namespace=expected_namespace,
            authorization=authorization,
        )
    else:
        if config.profile != "smoke" or any(
            set(record) != {"kind", "row"} or record.get("kind") != "probe_rows"
            for record in raw_rows
        ):
            raise ValueError("probe rows artifact has an invalid wrapper schema")
        rows = tuple(ProbeRow.from_record(record["row"]) for record in raw_rows)
    by_task = _probe_rows_by_task(rows)
    if expected_task_identities is not None:
        for task in TASKS:
            observed = tuple(
                sorted(
                    (ProbeSourceIdentity.from_row(row) for row in by_task[task]),
                    key=lambda item: (
                        item.example_id,
                        item.canonical_payload_sha256,
                    ),
                )
            )
            if observed != tuple(expected_task_identities[task]):
                raise ValueError(
                    f"{task} probe rows do not match the sealed task identities"
                )
    return rows, shard


def _reconstruct_compact_probe_rows(
    store: FAArtifactStore,
    shard: Any,
    records: Sequence[Mapping[str, Any]],
    config: FAConfig,
    *,
    expected_namespace: str,
    authorization: ProbeTestAuthorization | None = None,
) -> tuple[ProbeRow, ...]:
    sidecar = _read_json_object(shard.manifest_path)
    lineage = sidecar.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("compact probe evidence lineage is invalid")
    expected_schema = _probe_materialization_schema_sha256()
    if lineage.get("materialization_schema_sha256") != expected_schema:
        raise ValueError("compact probe evidence schema hash does not verify")
    if expected_namespace == "probe_test" and (
        not isinstance(authorization, ProbeTestAuthorization)
        or lineage.get("probe_authorization_sha256") != authorization.sha256
    ):
        raise ValueError("compact probe evidence is not bound to this probe authorization")

    prompt_path = _artifact_path_from_record(
        store, lineage.get("prompt_manifest"), "probe prompt manifest"
    )
    prompt = _load_manifest(store, prompt_path, config)
    if prompt.namespace != expected_namespace or prompt.shard_sha256 != lineage.get(
        "prompt_manifest_sha256"
    ):
        raise ValueError("compact probe evidence does not bind its prompt capability")
    metadata_path = _artifact_path_from_record(
        store, lineage.get("metadata_manifest"), "probe metadata manifest"
    )
    metadata, metadata_shard = _load_probe_metadata_manifest(
        store,
        metadata_path,
        config,
        expected_full_manifest_sha256=prompt.full_manifest_sha256,
        expected_example_ids=frozenset(row.example_id for row in prompt.examples),
    )
    if metadata_shard.sha256 != lineage.get("metadata_manifest_sha256"):
        raise ValueError("compact probe evidence metadata hash does not verify")

    activation_path = _artifact_path_from_record(
        store, lineage.get("activation_manifest"), "probe activation manifest"
    )
    if _file_sha256(activation_path) != lineage.get("activation_manifest_sha256"):
        raise ValueError("compact probe activation manifest hash does not verify")
    activations = load_activation_records(activation_path)
    activation_by_id = {row.example_id: row for row in activations}
    if (
        len(activation_by_id) != len(activations)
        or set(activation_by_id) != {row.example_id for row in prompt.examples}
    ):
        raise ValueError("compact probe activations do not match prompt examples")
    activation_manifest = _read_json_object(activation_path)
    for name, key in (
        ("request_sha256", "activation_request_sha256"),
        ("npz_sha256", "activation_npz_sha256"),
        ("index_sha256", "activation_index_sha256"),
    ):
        if activation_manifest.get(name) != lineage.get(key):
            raise ValueError(f"compact probe evidence does not bind {key}")

    generation_path = _artifact_path_from_record(
        store,
        lineage.get("generation_sidecar_manifest"),
        "probe generation sidecar",
    )
    generation_shard = _require_generation_sidecar_manifest(store, generation_path)
    if generation_shard.sha256 != lineage.get("generation_sidecar_sha256"):
        raise ValueError("compact probe generation hash does not verify")
    scored = _score_rows(
        _load_verified_generation_sidecar(store, generation_path, prompt, config),
        prompt,
    )
    outcomes = _unsupported_outcomes(scored, prompt.examples)

    expected_ids = tuple(sorted(row.example_id for row in prompt.examples))
    observed_ids = tuple(sorted(str(record.get("example_id")) for record in records))
    if observed_ids != expected_ids or len(set(observed_ids)) != len(observed_ids):
        raise ValueError("compact probe evidence does not match exact prompt IDs")
    outputs: dict[str, OutputEvidence] = {}
    for record in records:
        example_id = record["example_id"]
        output = OutputEvidence.from_record(record["output_evidence"])
        activation = activation_by_id[example_id]
        outcome = outcomes.get(example_id)
        stored_outcome = record.get("unsupported_outcome")
        expected_outcome = None if outcome is None else asdict(outcome)
        if (
            record["source_sha256"] != output.source_sha256
            or record["activation_sha256"] != activation.activation_sha256
            or record["output_evidence_sha256"] != output.sha256
            or stored_outcome != expected_outcome
        ):
            raise ValueError("compact probe evidence record does not verify")
        outputs[example_id] = output

    rows, rebuilt_outputs = _PROBE_ROW_MATERIALIZER(
        prompt.examples,
        activations,
        _RecordedOutputScorer(outputs),
        metadata,
        unsupported_outcomes=outcomes,
    )
    if {row.sha256 for row in rebuilt_outputs} != {
        row.sha256 for row in outputs.values()
    }:
        raise ValueError("compact output evidence reconstruction is incomplete")
    return rows


def _probe_rows_by_task(
    rows: Sequence[ProbeRow],
) -> Mapping[str, tuple[ProbeRow, ...]]:
    values = tuple(rows)
    if not values:
        raise ValueError("probe rows artifact is empty")
    grouped = {
        task: tuple(
            sorted(
                (row for row in values if row.task == task),
                key=lambda row: (row.example_id, row.sha256),
            )
        )
        for task in TASKS
    }
    if any(not grouped[task] for task in TASKS):
        raise ValueError("probe rows artifact must contain every registered task")
    if len({(row.task, row.example_id) for row in values}) != len(values):
        raise ValueError("probe rows artifact contains duplicate task/example IDs")
    familiarity = {
        row.example_id: row.source_sha256 for row in grouped["familiarity"]
    }
    answerability = {
        row.example_id: row.source_sha256 for row in grouped["answerability"]
    }
    unsupported = {
        row.example_id: row.source_sha256
        for row in grouped["unsupported_answer"]
    }
    if familiarity != answerability or not set(unsupported.items()).issubset(
        familiarity.items()
    ):
        raise ValueError("probe task rows do not share a coherent source identity set")
    return grouped


def _task_source_identities_sha256(
    values: Mapping[str, Sequence[ProbeSourceIdentity]],
) -> str:
    if set(values) != set(TASKS):
        raise ValueError("task source identities require every registered task")
    return _sha256_json(
        {
            task: [identity.to_record() for identity in values[task]]
            for task in TASKS
        }
    )


def _load_f2a_selection_bundle(
    store: FAArtifactStore,
    path: str | Path,
    config: FAConfig,
) -> VerifiedF2ASelectionBundle:
    shard = _require_verified_shard_kind(
        store, path, "selection_manifest", "verified F2A selection manifest"
    )
    if shard.namespace != "locked_validation":
        raise ValueError("F2A selection manifest must use locked_validation")
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    records = _read_json_rows(shard.data_path)
    expected = {
        "kind",
        "schema_version",
        "config_sha256",
        "selection_bundle_hash",
        "probe_test_prompt_sha256",
        "probe_test_task_identities_sha256",
        "selection_mode",
        "selections",
        "null_selections",
    }
    if len(records) != 1 or set(records[0]) != expected:
        raise ValueError("F2A selection manifest has an invalid schema")
    record = records[0]
    if (
        record["kind"] != "selection_manifest"
        or record["schema_version"] != 1
        or record["config_sha256"] != config.config_hash
    ):
        raise ValueError("F2A selection manifest identity is invalid")
    if not isinstance(record["selections"], dict) or set(
        record["selections"]
    ) != set(TASKS):
        raise ValueError("F2A selection manifest requires every registered task")
    if not isinstance(record["null_selections"], dict) or set(
        record["null_selections"]
    ) != set(TASKS):
        raise ValueError("F2A null selections require every registered task")
    selections = {
        task: SelectionManifest.from_record(record["selections"][task])
        for task in TASKS
    }
    nulls: dict[str, tuple[NullSelectionResult, ...]] = {}
    for task in TASKS:
        raw = record["null_selections"][task]
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"F2A {task} null selections are missing")
        loaded = tuple(NullSelectionResult.from_record(item) for item in raw)
        if any(item.selection.task != task or item.test_metrics is not None for item in loaded):
            raise ValueError("F2A null selections are task-mismatched or already scored")
        nulls[task] = loaded
    bundle_hash = f2a_selection_bundle_hash(selections)
    if record["selection_bundle_hash"] != bundle_hash:
        raise ValueError("F2A selection bundle hash does not verify")
    mode = record["selection_mode"]
    expected_count = 396 if mode == "registered_confirmatory" else 4
    if mode not in {"registered_confirmatory", "smoke_rehearsal"} or any(
        len(nulls[task]) != expected_count for task in TASKS
    ):
        raise ValueError("F2A null selection count does not match its mode")
    prompt_sha256 = record["probe_test_prompt_sha256"]
    _required_sha256(prompt_sha256, "probe_test_prompt_sha256")
    task_identities_sha256 = record["probe_test_task_identities_sha256"]
    _required_sha256(
        task_identities_sha256, "probe_test_task_identities_sha256"
    )
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config.config_hash,
            "probe_test_prompt_sha256": prompt_sha256,
            "probe_test_task_identities_sha256": task_identities_sha256,
            "selection_bundle_hash": bundle_hash,
            "selection_mode": mode,
        },
    )
    return VerifiedF2ASelectionBundle(
        selections=selections,
        null_selections=nulls,
        selection_bundle_hash=bundle_hash,
        probe_test_prompt_sha256=prompt_sha256,
        probe_test_task_identities_sha256=task_identities_sha256,
        shard_manifest_path=shard.manifest_path,
        shard_sha256=shard.sha256,
    )


def _evaluate_behavior_test(
    config: FAConfig, root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    store = FAArtifactStore(root)
    manifest = _load_manifest(store, args.manifest, config)
    if manifest.namespace != "behavior_test":
        raise ValueError("behavior evaluation requires the protected behavior_test manifest")
    state = store.endpoint_state("behavior_test", manifest.shard_manifest_path)
    if state == "closed":
        raise ValueError("behavior_test is already closed")
    if state == "evaluated":
        store.close_endpoint("behavior_test")
        return {"status": "recovered", "endpoint_state": "closed"}
    receipt = store.unlock_or_resume_endpoint(
        "behavior_test", manifest.shard_manifest_path
    )
    runner = HFModelRunner(config)
    validate_runner_binding(
        runner,
        config,
        expected_chat_template_sha256=manifest.chat_template_sha256,
    )
    generation = run_generation_shard(
        runner,
        manifest,
        store,
        _safe_cli_id(args.shard_id, "behavior shard-id"),
        config=config,
        namespace="behavior_test",
    )
    records = _load_verified_generation_sidecar(
        store, generation.manifest_path, manifest, config
    )
    if not all(record["status"] == "completed" for record in records):
        return {
            "status": "infrastructure_failure",
            "endpoint_state": "unlocked_once",
            "generation_manifest": str(generation.manifest_path),
        }
    scored = _score_rows(records, manifest)
    metrics = estimate_behavior(scored)
    bootstrap = _BEHAVIOR_BOOTSTRAP(
        scored,
        replicates=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )
    gate = _BEHAVIOR_GATE(
        metrics,
        bootstrap,
        thresholds=config.thresholds,
        same_string_sealed=any(
            row.block == "same_string" for row in manifest.examples
        ),
        config_hash=config.config_hash,
        manifest_hash=manifest.full_manifest_sha256,
        same_string_seal=(
            SameStringSealEvidence.from_registered_block(
                source_manifest_sha256=manifest.full_manifest_sha256,
                example_ids=tuple(
                    row.example_id
                    for row in manifest.examples
                    if row.block == "same_string"
                ),
            )
            if any(row.block == "same_string" for row in manifest.examples)
            else None
        ),
    )
    evidence = {
        "metrics": _json_safe(metrics.to_record()),
        "bootstrap": _json_safe(bootstrap.to_record()),
        "gate": _json_safe(gate.to_record()),
        "scored_rows": [_json_safe(row.to_record()) for row in scored],
    }
    row = {
        "kind": "metrics",
        "phase": "F1",
        **evidence,
        "evidence_sha256": _sha256_json(evidence),
    }
    lineage = {
        "config_sha256": config.config_hash,
        "preregistration_sha256": receipt.preregistration_hash,
        "selection_sha256": receipt.selection_manifest_hash,
        "prompt_manifest_sha256": manifest.shard_sha256,
        "generation_manifest_sha256": generation.sha256,
        "evidence_sha256": row["evidence_sha256"],
    }
    metrics_shard = _write_or_resume_metrics(
        store,
        config.run_id,
        "behavior_test",
        f"behavior-metrics-{receipt.lease_id}",
        row,
        lineage,
    )
    store.mark_evaluated(receipt, metrics_shard.data_path)
    store.close_endpoint("behavior_test")
    return {
        "status": "evaluated",
        "endpoint_state": "closed",
        "metrics_manifest": str(metrics_shard.manifest_path),
        "evidence_sha256": row["evidence_sha256"],
    }


def _write_or_resume_metrics(
    store: FAArtifactStore,
    run_id: str,
    namespace: str,
    shard_id: str,
    row: dict[str, Any],
    lineage: dict[str, Any],
):
    try:
        return store.write_completed_shard(
            run_id,
            namespace,
            shard_id,
            (row,),
            lineage,
            record_kind="metrics",
        )
    except FileExistsError:
        candidate = (
            store.root
            / "runs"
            / "familiarity_answerability"
            / run_id
            / "shards"
            / namespace
            / f"{shard_id}.jsonl.manifest.json"
        )
        existing = store.verify_shard(candidate)
        expected = json.dumps(
            row, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        if (
            existing.record_kind != "metrics"
            or existing.row_count != 1
            or existing.sha256 != hashlib.sha256(expected).hexdigest()
        ):
            raise ValueError("existing metrics artifact does not match this evaluation")
        sidecar = _read_json_object(existing.manifest_path)
        if sidecar.get("lineage") != lineage:
            raise ValueError("existing metrics lineage does not match this evaluation")
        return existing


def _write_or_resume_single_record(
    store: FAArtifactStore,
    *,
    run_id: str,
    namespace: str,
    shard_id: str,
    row: Mapping[str, Any],
    lineage: Mapping[str, Any],
    record_kind: str,
):
    try:
        return store.write_completed_shard(
            run_id,
            namespace,
            shard_id,
            (dict(row),),
            dict(lineage),
            record_kind=record_kind,
        )
    except FileExistsError:
        candidate = (
            store.root
            / "runs"
            / "familiarity_answerability"
            / run_id
            / "shards"
            / namespace
            / f"{shard_id}.jsonl.manifest.json"
        )
        existing = store.verify_shard(candidate)
        expected = json.dumps(
            dict(row), allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        sidecar = _read_json_object(existing.manifest_path)
        if (
            existing.record_kind != record_kind
            or existing.row_count != 1
            or existing.sha256 != hashlib.sha256(expected).hexdigest()
            or sidecar.get("lineage") != dict(lineage)
        ):
            raise ValueError(
                "existing single-record artifact does not match this transaction"
            )
        return existing


def _write_or_resume_records(
    store: FAArtifactStore,
    *,
    run_id: str,
    namespace: str,
    shard_id: str,
    rows: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
    record_kind: str,
    allow_resume: bool,
):
    canonical_rows = tuple(dict(row) for row in rows)
    try:
        return store.write_completed_shard(
            run_id,
            namespace,
            shard_id,
            canonical_rows,
            dict(lineage),
            record_kind=record_kind,
        )
    except FileExistsError:
        if not allow_resume:
            raise
        candidate = (
            store.root
            / "runs"
            / "familiarity_answerability"
            / run_id
            / "shards"
            / namespace
            / f"{shard_id}.jsonl.manifest.json"
        )
        existing = store.verify_shard(candidate)
        expected = b"".join(
            json.dumps(
                row, allow_nan=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            + b"\n"
            for row in canonical_rows
        )
        sidecar = _read_json_object(existing.manifest_path)
        if (
            existing.record_kind != record_kind
            or existing.row_count != len(canonical_rows)
            or existing.sha256 != hashlib.sha256(expected).hexdigest()
            or sidecar.get("lineage") != dict(lineage)
        ):
            raise ValueError(
                "existing multi-record artifact does not match this transaction"
            )
        return existing


def _score_behavior(config: FAConfig, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    store = FAArtifactStore(root)
    manifest = _load_manifest(store, args.manifest, config)
    generation_shard = _require_generation_sidecar_manifest(
        store, args.generation_manifest
    )
    records = _load_verified_generation_sidecar(
        store, generation_shard.manifest_path, manifest, config
    )
    rows = _score_rows(records, manifest)
    metrics = estimate_behavior(rows)
    gate = _pilot_gate(rows, metrics)
    metrics_record = _json_safe(metrics.to_record())
    evidence_sha256 = _sha256_json(
        {"metrics": metrics_record, "pilot_gate": gate}
    )
    gate_shard = store.write_completed_shard(
        config.run_id,
        generation_shard.namespace,
        f"pilot-gate-{generation_shard.sha256[:16]}",
        [
            {
                "kind": "pilot_gate",
                "config_sha256": config.config_hash,
                "source_manifest_sha256": manifest.manifest_sha256,
                "prompt_manifest": str(
                    manifest.shard_manifest_path.relative_to(store.root)
                ),
                "prompt_manifest_sha256": manifest.shard_sha256,
                "tokenizer_pin_manifest": str(
                    manifest.tokenizer_pin_manifest_path.relative_to(store.root)
                ),
                "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
                "chat_template_sha256": manifest.chat_template_sha256,
                "generation_sidecar_manifest": str(
                    generation_shard.manifest_path.relative_to(store.root)
                ),
                "generation_sidecar_sha256": generation_shard.sha256,
                "pilot_gate": gate,
                "metrics": metrics_record,
                "evidence_sha256": evidence_sha256,
            }
        ],
        {
            "config_sha256": config.config_hash,
            "source_manifest_sha256": manifest.manifest_sha256,
            "generation_sidecar_sha256": generation_shard.sha256,
            "prompt_manifest_sha256": manifest.shard_sha256,
            "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
            "chat_template_sha256": manifest.chat_template_sha256,
        },
        record_kind="pilot_gate",
    )
    return {
        "status": "scored",
        "metrics": metrics_record,
        "pilot_gate": gate,
        "pilot_gate_manifest": str(gate_shard.manifest_path),
    }


def _score_rows(records: tuple[dict[str, Any], ...], manifest: Any):
    examples = {row.example_id: row for row in manifest.examples}
    codes = frozenset(row.registry_code for row in manifest.examples)
    scored = []
    for record in records:
        example_record = record.get("example")
        example_id = record.get("example_id") or (example_record or {}).get("example_id")
        example = examples.get(example_id)
        if example is None:
            raise ValueError("generation record does not bind a manifest example")
        status = record.get("status")
        scored.append(score_response(
            example, record.get("raw_output"), registered_codes=codes or {example.registry_code},
            infrastructure_marked=status != "completed",
        ))
    if not scored:
        raise ValueError("generation manifest contains no records")
    return tuple(scored)


def _pilot_gate(rows: tuple[Any, ...], metrics: Any) -> dict[str, Any]:
    target = [row for row in rows if row.answerability == "target_bound"]
    accuracy = sum(row.outcome.value == "exact_target_code" for row in target) / len(target) if target else 0.0
    reasons = []
    if accuracy < 0.70:
        reasons.append("target_bound_accuracy_below_70_percent")
    for cell, denominator in metrics.denominators.items():
        if denominator and metrics.invalid_format_counts[cell] / denominator > 0.05:
            reasons.append(f"invalid_format_above_5_percent:{':'.join(cell)}")
    absent = [row for row in rows if row.answerability in {"distractor_bound", "code_absent"}]
    if absent:
        rate = sum(row.answer_attempt for row in absent) / len(absent)
        if rate <= 0.05 or rate >= 0.95:
            reasons.append("absent_answer_attempt_at_registered_floor_or_ceiling")
    else:
        rate = None
        reasons.append("absent_answer_attempt_at_registered_floor_or_ceiling")
    return {"status": "passed" if not reasons else "blocked", "reasons": reasons, "target_bound_accuracy": accuracy, "absent_answer_attempt_rate": rate}


def _load_manifest(
    store: FAArtifactStore, path: str | Path, config: FAConfig | None = None
) -> VerifiedPromptManifest:
    shard = _require_verified_shard_kind(
        store, path, "prompt_manifest", "verified prompt manifest"
    )
    rows = _read_json_rows(shard.data_path)
    if len(rows) != 1:
        raise ValueError("prompt manifest must contain exactly one capability record")
    value = rows[0]
    required = {
        "kind",
        "config_hash",
        "full_manifest_sha256",
        "subset_manifest_sha256",
        "chat_template_sha256",
        "namespace",
        "model_sha256",
        "tokenizer_sha256",
        "tokenizer_pin_manifest",
        "tokenizer_pin_sha256",
        "naturalness_audit_manifest",
        "naturalness_audit_sha256",
        "generation",
        "examples",
    }
    if set(value) != required or value.get("kind") != "prompt_manifest":
        raise ValueError("prompt manifest has an invalid schema")
    config_hash = _required_sha256(value.get("config_hash"), "prompt config_hash")
    if config is not None and config_hash != config.config_hash:
        raise ValueError("explicit manifest config hash does not match config")
    namespace = value.get("namespace")
    if namespace != shard.namespace or namespace not in _GENERATION_NAMESPACES:
        raise ValueError("prompt manifest namespace does not match its immutable shard")
    raw_examples = value.get("examples")
    if not isinstance(raw_examples, list) or not raw_examples:
        raise ValueError("prompt manifest must contain canonical examples")
    try:
        examples = tuple(FAExample(**row) for row in raw_examples)
    except (TypeError, ValueError) as error:
        raise ValueError(f"prompt manifest contains an invalid FAExample: {error}") from error
    if any(_record_value(example) != raw for example, raw in zip(examples, raw_examples, strict=True)):
        raise ValueError("prompt manifest FAExample payload is not canonical")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("prompt manifest contains duplicate example IDs")
    if any(example.split != namespace for example in examples):
        raise ValueError("prompt manifest contains another namespace")
    template_hash = _required_sha256(
        value.get("chat_template_sha256"), "prepared chat_template_sha256"
    )
    if config is not None and config.chat_template_sha256 and template_hash != config.chat_template_sha256:
        raise ValueError("manifest chat template pin does not match config")
    full_hash = _required_sha256(
        value.get("full_manifest_sha256"), "full_manifest_sha256"
    )
    if config is None:
        raise ValueError("prompt verification requires an immutable FA config")
    tokenizer_pin_path = _artifact_path_from_record(
        store, value.get("tokenizer_pin_manifest"), "tokenizer pin manifest"
    )
    tokenizer_pin_sha256 = _required_sha256(
        value.get("tokenizer_pin_sha256"), "tokenizer_pin_sha256"
    )
    _load_verified_tokenizer_pin(
        store,
        tokenizer_pin_path,
        config,
        expected_sha256=tokenizer_pin_sha256,
        expected_source_manifest_sha256=full_hash,
        expected_chat_template_sha256=template_hash,
    )
    naturalness_audit_path = None
    naturalness_audit_sha256 = None
    if config.profile == "confirmatory":
        naturalness_audit_path = _artifact_path_from_record(
            store,
            value.get("naturalness_audit_manifest"),
            "naturalness audit manifest",
        )
        naturalness_audit_sha256 = _required_sha256(
            value.get("naturalness_audit_sha256"),
            "naturalness_audit_sha256",
        )
        _load_verified_naturalness_audit(
            store,
            naturalness_audit_path,
            config,
            expected_sha256=naturalness_audit_sha256,
            expected_pair_ids=frozenset(
                example.entity_unit_id for example in examples
            ),
        )
    elif (
        value.get("naturalness_audit_manifest") is not None
        or value.get("naturalness_audit_sha256") is not None
    ):
        raise ValueError("smoke prompt manifests cannot claim a naturalness audit")
    subset_hash = _prompt_subset_sha256(
        config_hash,
        full_hash,
        namespace,
        template_hash,
        tokenizer_pin_sha256,
        naturalness_audit_sha256,
        examples,
    )
    if value.get("subset_manifest_sha256") != subset_hash:
        raise ValueError("prompt subset manifest identity does not verify")
    model_hash = _required_sha256(value.get("model_sha256"), "prompt model_sha256")
    tokenizer_hash = _required_sha256(
        value.get("tokenizer_sha256"), "prompt tokenizer_sha256"
    )
    generation = value.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("prompt generation config is invalid")
    if config is not None:
        expected_model, expected_tokenizer = _config_runtime_hashes(config)
        if (model_hash, tokenizer_hash, generation) != (
            expected_model,
            expected_tokenizer,
            dict(config.generation),
        ):
            raise ValueError("prompt runtime pins do not match config")
    task_source_identities = _task_source_identity_records(examples)
    source_identities = task_source_identities["familiarity"]
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config_hash,
            "source_manifest_sha256": full_hash,
            "subset_manifest_sha256": subset_hash,
            "chat_template_sha256": template_hash,
            "tokenizer_pin_sha256": tokenizer_pin_sha256,
            "naturalness_audit_sha256": naturalness_audit_sha256,
            "source_identities": source_identities,
            "source_identities_sha256": _sha256_json(source_identities),
            "task_source_identities": task_source_identities,
            "task_source_identities_sha256": _sha256_json(
                task_source_identities
            ),
        },
    )
    return VerifiedPromptManifest(
        config_hash=config_hash,
        manifest_sha256=subset_hash,
        full_manifest_sha256=full_hash,
        chat_template_sha256=template_hash,
        namespace=namespace,
        examples=examples,
        model_sha256=model_hash,
        tokenizer_sha256=tokenizer_hash,
        tokenizer_pin_manifest_path=tokenizer_pin_path,
        tokenizer_pin_sha256=tokenizer_pin_sha256,
        naturalness_audit_manifest_path=naturalness_audit_path,
        naturalness_audit_sha256=naturalness_audit_sha256,
        generation=dict(generation),
        shard_manifest_path=shard.manifest_path,
        shard_sha256=shard.sha256,
    )


def _load_prompt_source_identities(
    store: FAArtifactStore,
    path: str | Path,
    config: FAConfig,
    *,
    expected_namespace: str,
) -> tuple[tuple[ProbeSourceIdentity, ...], Any]:
    """Load only pre-outcome IDs and hashes from a prompt capability sidecar."""

    task_identities, shard = _load_prompt_task_source_identities(
        store, path, config, expected_namespace=expected_namespace
    )
    return task_identities["familiarity"], shard


def _load_prompt_task_source_identities(
    store: FAArtifactStore,
    path: str | Path,
    config: FAConfig,
    *,
    expected_namespace: str,
) -> tuple[Mapping[str, tuple[ProbeSourceIdentity, ...]], Any]:
    """Load task-scoped identities without parsing protected prompt contents."""

    shard = _require_verified_shard_kind(
        store, path, "prompt_manifest", "verified prompt identity capability"
    )
    if shard.namespace != expected_namespace:
        raise ValueError("prompt identity capability uses the wrong namespace")
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    sidecar = _read_json_object(shard.manifest_path)
    lineage = sidecar.get("lineage")
    if not isinstance(lineage, dict) or lineage.get("config_sha256") != config.config_hash:
        raise ValueError("prompt identity capability does not bind the config")
    raw = lineage.get("task_source_identities")
    if not isinstance(raw, dict) or set(raw) != set(TASKS):
        raise ValueError("prompt identity capability has no task-scoped identities")
    task_identities: dict[str, tuple[ProbeSourceIdentity, ...]] = {}
    for task in TASKS:
        records = raw.get(task)
        if not isinstance(records, list) or not records:
            raise ValueError(f"prompt identity capability has no {task} identities")
        identities = tuple(ProbeSourceIdentity.from_record(item) for item in records)
        canonical = tuple(
            sorted(
                identities,
                key=lambda item: (item.example_id, item.canonical_payload_sha256),
            )
        )
        if identities != canonical or len(set(identities)) != len(identities):
            raise ValueError(
                "prompt source identities must be unique and canonically ordered"
            )
        task_identities[task] = identities
    canonical_record = {
        task: [item.to_record() for item in task_identities[task]] for task in TASKS
    }
    if lineage.get("task_source_identities_sha256") != _sha256_json(
        canonical_record
    ):
        raise ValueError("prompt task source identity hash does not verify")
    if task_identities["familiarity"] != task_identities["answerability"]:
        raise ValueError("familiarity and answerability identities must match")
    if not set(task_identities["unsupported_answer"]).issubset(
        task_identities["familiarity"]
    ):
        raise ValueError("unsupported-answer identities must be a strict task subset")
    return task_identities, shard


def _task_source_identity_records(
    examples: Sequence[Any],
) -> dict[str, list[dict[str, str]]]:
    ordered = tuple(sorted(examples, key=lambda example: example.example_id))
    source = [
        {
            "example_id": example.example_id,
            "canonical_payload_sha256": example.canonical_payload_sha256,
        }
        for example in ordered
    ]
    return {
        "familiarity": source,
        "answerability": source,
        "unsupported_answer": [
            identity
            for identity, example in zip(source, ordered, strict=True)
            if example.answerability in {"distractor_bound", "code_absent"}
        ],
    }


def _write_prompt_capability(
    store: FAArtifactStore,
    config: FAConfig,
    full_manifest_sha256: str,
    namespace: str,
    examples: tuple[FAExample, ...],
    chat_template_sha256: str,
    tokenizer_pin,
    naturalness_audit=None,
):
    if config.profile == "confirmatory" and naturalness_audit is None:
        raise ValueError("confirmatory prompt capability requires a naturalness audit")
    if config.profile == "smoke" and naturalness_audit is not None:
        raise ValueError("smoke prompt capability cannot claim a naturalness audit")
    ordered = tuple(sorted(examples, key=lambda row: row.example_id))
    template_hash = _required_sha256(
        chat_template_sha256, "prepared chat_template_sha256"
    )
    subset_hash = _prompt_subset_sha256(
        config.config_hash,
        full_manifest_sha256,
        namespace,
        template_hash,
        tokenizer_pin.sha256,
        None if naturalness_audit is None else naturalness_audit.sha256,
        ordered,
    )
    model_hash, tokenizer_hash = _config_runtime_hashes(config)
    task_source_identities = _task_source_identity_records(ordered)
    source_identities = task_source_identities["familiarity"]
    row = {
        "kind": "prompt_manifest",
        "config_hash": config.config_hash,
        "full_manifest_sha256": full_manifest_sha256,
        "subset_manifest_sha256": subset_hash,
        "chat_template_sha256": template_hash,
        "namespace": namespace,
        "model_sha256": model_hash,
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_pin_manifest": str(
            tokenizer_pin.manifest_path.relative_to(store.root)
        ),
        "tokenizer_pin_sha256": tokenizer_pin.sha256,
        "naturalness_audit_manifest": (
            None
            if naturalness_audit is None
            else str(naturalness_audit.manifest_path.relative_to(store.root))
        ),
        "naturalness_audit_sha256": (
            None if naturalness_audit is None else naturalness_audit.sha256
        ),
        "generation": dict(config.generation),
        "examples": [_record_value(example) for example in ordered],
    }
    return _write_or_resume_single_record(
        store,
        run_id=config.run_id,
        namespace=namespace,
        shard_id=f"prompt-capability-{subset_hash[:16]}",
        row=row,
        lineage={
            "config_sha256": config.config_hash,
            "source_manifest_sha256": full_manifest_sha256,
            "subset_manifest_sha256": subset_hash,
            "chat_template_sha256": template_hash,
            "tokenizer_pin_sha256": tokenizer_pin.sha256,
            "naturalness_audit_sha256": (
                None if naturalness_audit is None else naturalness_audit.sha256
            ),
            "source_identities": source_identities,
            "source_identities_sha256": _sha256_json(source_identities),
            "task_source_identities": task_source_identities,
            "task_source_identities_sha256": _sha256_json(
                task_source_identities
            ),
        },
        record_kind="prompt_manifest",
    )


def _prompt_subset_sha256(
    config_hash,
    full_hash,
    namespace,
    template_hash,
    tokenizer_pin_sha256,
    naturalness_audit_sha256,
    examples,
):
    return _sha256_json(
        {
            "config_hash": config_hash,
            "full_manifest_sha256": full_hash,
            "namespace": namespace,
            "chat_template_sha256": template_hash,
            "tokenizer_pin_sha256": tokenizer_pin_sha256,
            "naturalness_audit_sha256": naturalness_audit_sha256,
            "examples": [_record_value(row) for row in examples],
        }
    )


def _write_naturalness_audit(
    store, config, matches, audit, ratings, ratings_shard
):
    row = {
        "kind": "naturalness_audit",
        "config_sha256": config.config_hash,
        "ratings_manifest": str(ratings_shard.manifest_path.relative_to(store.root)),
        "ratings_sha256": ratings_shard.sha256,
        "matches": [_record_value(value) for value in sorted(matches, key=lambda value: value.pair_id)],
        "ratings": [
            _record_value(value)
            for value in sorted(
                ratings,
                key=lambda value: (value.pair_id, value.round, value.rater_id),
            )
        ],
        "accepted_pair_ids": list(audit.accepted_pair_ids),
        "excluded_pair_ids": list(audit.excluded_pair_ids),
        "third_rater_pair_ids": list(audit.third_rater_pair_ids),
        "decisions": dict(audit.decisions),
    }
    audit_hash = _sha256_json(row)
    return store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        f"naturalness-audit-{audit_hash[:16]}",
        [row],
        {
            "config_sha256": config.config_hash,
            "audit_sha256": audit_hash,
            "ratings_sha256": ratings_shard.sha256,
        },
        record_kind="naturalness_audit",
    )


def _write_probe_metadata(store, config, full_manifest_sha256, matches, examples):
    domain_by_pair = {match.pair_id: match.coarse_type for match in matches}
    if len(domain_by_pair) != len(tuple(matches)):
        raise ValueError("probe metadata requires unique matched entity pairs")
    if any(domain not in REGISTERED_ENTITY_DOMAINS for domain in domain_by_pair.values()):
        raise ValueError("probe metadata contains an unregistered entity domain")
    metadata_rows = [
        {
            "example_id": example.example_id,
            "entity_id": example.entity_unit_id,
            "template_id": example.template_family,
            "relation_id": "archive_code",
            "domain": domain_by_pair[example.entity_unit_id],
            "condition": example.block,
        }
        for example in sorted(examples, key=lambda row: row.example_id)
    ]
    if len({row["example_id"] for row in metadata_rows}) != len(metadata_rows):
        raise ValueError("probe metadata contains duplicate example IDs")
    row = {
        "kind": "probe_metadata",
        "schema_version": 1,
        "config_sha256": config.config_hash,
        "full_manifest_sha256": full_manifest_sha256,
        "manifest_revision": "2026-07-22",
        "rows": metadata_rows,
    }
    metadata_sha256 = _sha256_json(row)
    return _write_or_resume_single_record(
        store,
        run_id=config.run_id,
        namespace=(
            "mechanism_train" if config.profile == "confirmatory" else "pilot"
        ),
        shard_id=f"probe-metadata-{metadata_sha256[:16]}",
        row=row,
        lineage={
            "config_sha256": config.config_hash,
            "source_manifest_sha256": full_manifest_sha256,
            "metadata_sha256": metadata_sha256,
        },
        record_kind="probe_metadata",
    )


def _load_verified_naturalness_audit(
    store,
    path,
    config,
    *,
    expected_sha256,
    expected_pair_ids,
):
    shard = _require_verified_shard_kind(
        store, path, "naturalness_audit", "verified naturalness audit manifest"
    )
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    if shard.namespace != "mechanism_train" or shard.sha256 != expected_sha256:
        raise ValueError("naturalness audit identity does not verify")
    rows = _read_json_rows(shard.data_path)
    if len(rows) != 1 or rows[0].get("kind") != "naturalness_audit":
        raise ValueError("naturalness audit has an invalid schema")
    row = rows[0]
    required = {
        "kind",
        "config_sha256",
        "ratings_manifest",
        "ratings_sha256",
        "matches",
        "ratings",
        "accepted_pair_ids",
        "excluded_pair_ids",
        "third_rater_pair_ids",
        "decisions",
    }
    if set(row) != required or row.get("config_sha256") != config.config_hash:
        raise ValueError("naturalness audit has an invalid schema or config")
    ratings_path = _artifact_path_from_record(
        store, row.get("ratings_manifest"), "naturalness ratings manifest"
    )
    expected_ratings_sha256 = _required_sha256(
        row.get("ratings_sha256"), "naturalness ratings sha256"
    )
    verified_ratings, ratings_shard = _load_verified_naturalness_ratings(
        store, ratings_path, config
    )
    if ratings_shard.sha256 != expected_ratings_sha256:
        raise ValueError("naturalness ratings identity does not verify")
    try:
        matches = tuple(EntityMatch(**value) for value in row["matches"])
        ratings = tuple(NaturalnessRating(**value) for value in row["ratings"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("naturalness audit evidence is invalid") from error
    recomputed = audit_naturalness_manifest(matches, ratings)
    if ratings != verified_ratings:
        raise ValueError("naturalness audit ratings differ from their source artifact")
    ratings_lineage = _read_json_object(ratings_shard.manifest_path).get("lineage")
    if (
        not isinstance(ratings_lineage, dict)
        or ratings_lineage.get("matches_sha256")
        != naturalness_matches_sha256(matches)
    ):
        raise ValueError("naturalness ratings do not bind the audited entity pairs")
    recorded = {
        "accepted_pair_ids": list(recomputed.accepted_pair_ids),
        "excluded_pair_ids": list(recomputed.excluded_pair_ids),
        "third_rater_pair_ids": list(recomputed.third_rater_pair_ids),
        "decisions": dict(recomputed.decisions),
    }
    if any(row[name] != value for name, value in recorded.items()):
        raise ValueError("naturalness audit does not match deterministic recomputation")
    accepted = frozenset(recomputed.accepted_pair_ids)
    if not expected_pair_ids <= accepted:
        raise ValueError("naturalness audit does not accept every prompt entity unit")
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config.config_hash,
            "audit_sha256": _sha256_json(row),
            "ratings_sha256": expected_ratings_sha256,
        },
    )
    return shard


def _load_verified_naturalness_ratings(store, path, config):
    shard = _require_verified_shard_kind(
        store,
        path,
        "naturalness_ratings",
        "verified naturalness ratings manifest",
    )
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    if shard.namespace != "mechanism_train":
        raise ValueError("naturalness ratings must use mechanism_train namespace")
    rows = _read_json_rows(shard.data_path)
    required = {
        "kind",
        "schema_version",
        "config_sha256",
        "protocol_sha256",
        "blinding_manifest_sha256",
        "assignments",
        "ratings",
    }
    if len(rows) != 1 or set(rows[0]) != required:
        raise ValueError("naturalness ratings have an invalid schema")
    row = rows[0]
    if row.get("kind") != "naturalness_ratings" or row.get("schema_version") != 1:
        raise ValueError("naturalness ratings schema_version must equal one")
    if row.get("config_sha256") != config.config_hash:
        raise ValueError("naturalness ratings config does not match")
    protocol_sha256 = hashlib.sha256(_PREREGISTRATION_PATH.read_bytes()).hexdigest()
    if row.get("protocol_sha256") != protocol_sha256:
        raise ValueError("naturalness ratings protocol hash does not match preregistration")
    rating_protocol_sha256 = hashlib.sha256(
        _NATURALNESS_PROTOCOL_PATH.read_bytes()
    ).hexdigest()
    assignments = row.get("assignments")
    raw_ratings = row.get("ratings")
    if not isinstance(assignments, list) or not isinstance(raw_ratings, list):
        raise ValueError("naturalness ratings evidence must be lists")
    if row.get("blinding_manifest_sha256") != _sha256_json(assignments):
        raise ValueError("naturalness ratings blinding manifest does not verify")
    assignment_keys = set()
    submissions = set()
    slots_by_pair = {}
    for assignment in assignments:
        if not isinstance(assignment, dict) or set(assignment) != {
            "pair_id",
            "rater_id",
            "blind_slot",
            "submission_sha256",
        }:
            raise ValueError("naturalness rating assignment has an invalid schema")
        key = (assignment["pair_id"], assignment["rater_id"])
        if key in assignment_keys:
            raise ValueError("naturalness rating assignments must be unique")
        assignment_keys.add(key)
        submission = _required_sha256(
            assignment.get("submission_sha256"), "rating submission sha256"
        )
        if submission in submissions:
            raise ValueError("rating submissions must be unique")
        submissions.add(submission)
        slot = assignment.get("blind_slot")
        if slot not in {"slot-a", "slot-b", "adjudicator"}:
            raise ValueError("naturalness rating blind slot is invalid")
        slots_by_pair.setdefault(assignment["pair_id"], set()).add(slot)
    try:
        ratings = tuple(NaturalnessRating(**value) for value in raw_ratings)
    except (TypeError, ValueError) as error:
        raise ValueError(f"naturalness rating schema_version is invalid: {error}") from error
    rating_keys = {(rating.pair_id, rating.rater_id) for rating in ratings}
    if len(rating_keys) != len(ratings) or rating_keys != assignment_keys:
        raise ValueError("naturalness ratings must match blinded assignments exactly")
    for pair_id, pair_ratings in _group_ratings(ratings).items():
        initial = tuple(value for value in pair_ratings if value.round == 1)
        adjudicators = tuple(value for value in pair_ratings if value.round == 2)
        expected_slots = {"slot-a", "slot-b"} | (
            {"adjudicator"} if adjudicators else set()
        )
        if len(initial) != 2 or len(adjudicators) > 1 or slots_by_pair[pair_id] != expected_slots:
            raise ValueError("naturalness ratings do not prove the registered blind assignment")
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config.config_hash,
            "protocol_sha256": protocol_sha256,
            "rating_protocol_sha256": rating_protocol_sha256,
            "blinding_manifest_sha256": row["blinding_manifest_sha256"],
        },
    )
    lineage = _read_json_object(shard.manifest_path)["lineage"]
    matches_sha256 = _required_sha256(
        lineage.get("matches_sha256"), "naturalness matches sha256"
    )
    initial_path = _artifact_path_from_record(
        store,
        lineage.get("initial_submission_manifest"),
        "initial naturalness submission manifest",
    )
    initial_row, initial_shard = _load_naturalness_submission(
        store, initial_path, config
    )
    if initial_shard.sha256 != lineage.get("initial_submission_sha256"):
        raise ValueError("initial naturalness submission identity does not verify")
    initial_issuance_path = _artifact_path_from_record(
        store,
        initial_row["issuance_manifest"],
        "initial naturalness packet issuance manifest",
    )
    initial_issuance_row, _ = _load_naturalness_packet_issuance(
        store,
        initial_issuance_path,
        config,
        expected_purpose="initial",
    )
    if initial_issuance_row["matches_sha256"] != matches_sha256:
        raise ValueError(
            "naturalness ratings entity pairs do not match the initial issuance"
        )
    evidence_ratings = list(initial_row["ratings"])
    evidence_assignments = list(initial_row["assignments"])
    has_round_two = any(value.round == 2 for value in ratings)
    registered_disagreements = frozenset(initial_row["disagreement_pair_ids"])
    if has_round_two != bool(registered_disagreements):
        raise ValueError(
            "naturalness adjudication must match the registered disagreement state"
        )
    adjudication_fields = {
        "adjudication_submission_manifest",
        "adjudication_submission_sha256",
    }
    present_adjudication_fields = adjudication_fields & set(lineage)
    if has_round_two:
        if present_adjudication_fields != adjudication_fields:
            raise ValueError(
                "round-two naturalness ratings require adjudication lineage"
            )
        adjudication_path = _artifact_path_from_record(
            store,
            lineage["adjudication_submission_manifest"],
            "adjudication naturalness submission manifest",
        )
        adjudication_row, adjudication_shard = _load_naturalness_submission(
            store, adjudication_path, config
        )
        if (
            adjudication_shard.sha256
            != lineage["adjudication_submission_sha256"]
        ):
            raise ValueError(
                "adjudication naturalness submission identity does not verify"
            )
        adjudication_issuance_path = _artifact_path_from_record(
            store,
            adjudication_row["issuance_manifest"],
            "adjudication naturalness packet issuance manifest",
        )
        adjudication_issuance_row, adjudication_issuance_shard = (
            _load_naturalness_packet_issuance(
                store,
                adjudication_issuance_path,
                config,
                expected_purpose="adjudication",
            )
        )
        adjudication_issuance_lineage = _read_json_object(
            adjudication_issuance_shard.manifest_path
        )["lineage"]
        adjudication_submission_lineage = _read_json_object(
            adjudication_shard.manifest_path
        )["lineage"]
        if (
            adjudication_issuance_lineage.get("initial_submission_sha256")
            != initial_shard.sha256
            or adjudication_submission_lineage.get("initial_submission_sha256")
            != initial_shard.sha256
        ):
            raise ValueError(
                "naturalness adjudication does not bind the initial submission"
            )
        adjudication_pair_ids = {
            value["pair_id"] for value in adjudication_row["ratings"]
        }
        issued_pair_ids = {
            value["pair_id"]
            for value in adjudication_issuance_row["private_key"]["items"]
        }
        initial_stimuli = issuance_pair_stimulus_sha256s(initial_issuance_row)
        adjudication_stimuli = issuance_pair_stimulus_sha256s(
            adjudication_issuance_row
        )
        if not registered_disagreements <= set(initial_stimuli):
            raise ValueError(
                "naturalness disagreements do not match the initial stimuli"
            )
        expected_adjudication_stimuli = {
            pair_id: initial_stimuli[pair_id]
            for pair_id in registered_disagreements
        }
        if (
            adjudication_pair_ids != registered_disagreements
            or issued_pair_ids != registered_disagreements
            or adjudication_stimuli != expected_adjudication_stimuli
        ):
            raise ValueError(
                "naturalness adjudication does not match registered stimuli"
            )
        evidence_ratings.extend(adjudication_row["ratings"])
        evidence_assignments.extend(adjudication_row["assignments"])
    elif present_adjudication_fields:
        raise ValueError("adjudication lineage is invalid without round-two ratings")
    evidence_ratings.sort(
        key=lambda value: (value["pair_id"], value["round"], value["rater_id"])
    )
    evidence_assignments.sort(
        key=lambda value: (
            value["pair_id"],
            value["blind_slot"],
            value["rater_id"],
        )
    )
    if row["ratings"] != evidence_ratings or row["assignments"] != evidence_assignments:
        raise ValueError(
            "naturalness ratings differ from immutable human submissions"
        )
    return tuple(sorted(ratings, key=lambda value: (value.pair_id, value.round, value.rater_id))), shard


def _group_ratings(ratings):
    grouped = {}
    for rating in ratings:
        grouped.setdefault(rating.pair_id, []).append(rating)
    return grouped


def _record_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        result = value
    elif hasattr(value, "__dataclass_fields__"):
        result = asdict(value)
    elif hasattr(value, "__dict__"):
        result = dict(value.__dict__)
    else:
        raise ValueError("manifest example must be a serializable record")
    return json.loads(json.dumps(result, sort_keys=True, default=list))


def _write_tokenizer_pin(store, config, prepared, source_manifest_sha256):
    source_hash = _required_sha256(
        source_manifest_sha256, "source manifest sha256"
    )
    row = {
        "kind": "tokenizer_pin",
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "chat_template_sha256": prepared.chat_template_sha256,
        "chat_template_utf8_hex": prepared.chat_template_bytes.hex(),
    }
    return _write_or_resume_single_record(
        store,
        run_id=config.run_id,
        namespace=(
            "mechanism_train" if config.profile == "confirmatory" else "pilot"
        ),
        shard_id=(
            f"tokenizer-pin-{prepared.chat_template_sha256[:12]}-"
            f"{source_hash[:12]}"
        ),
        row=row,
        lineage={
            "config_sha256": config.config_hash,
            "source_manifest_sha256": source_hash,
            "chat_template_sha256": prepared.chat_template_sha256,
        },
        record_kind="tokenizer_pin",
    )


def _load_verified_tokenizer_pin(
    store: FAArtifactStore,
    path: str | Path,
    expected_config: FAConfig,
    *,
    expected_sha256: str,
    expected_source_manifest_sha256: str,
    expected_chat_template_sha256: str,
):
    shard = _require_verified_shard_kind(
        store, path, "tokenizer_pin", "verified tokenizer pin manifest"
    )
    _verify_artifact_run_id(shard.manifest_path, expected_config.run_id)
    if shard.sha256 != _required_sha256(expected_sha256, "tokenizer pin sha256"):
        raise ValueError("tokenizer pin shard hash does not verify")
    expected_namespace = (
        "mechanism_train" if expected_config.profile == "confirmatory" else "pilot"
    )
    if shard.namespace != expected_namespace:
        raise ValueError("tokenizer pin namespace does not match the registered config")
    rows = _read_json_rows(shard.data_path)
    required = {
        "kind",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "chat_template_sha256",
        "chat_template_utf8_hex",
    }
    if len(rows) != 1 or set(rows[0]) != required or rows[0].get("kind") != "tokenizer_pin":
        raise ValueError("tokenizer pin has an invalid schema")
    row = rows[0]
    if (
        row.get("model_id"),
        row.get("model_revision"),
        row.get("tokenizer_revision"),
    ) != (
        expected_config.model_id,
        expected_config.model_revision,
        expected_config.tokenizer_revision,
    ):
        raise ValueError("tokenizer pin model identity does not match the registered config")
    claimed_template_hash = _required_sha256(
        row.get("chat_template_sha256"), "tokenizer pin chat_template_sha256"
    )
    template_hex = row.get("chat_template_utf8_hex")
    if not isinstance(template_hex, str):
        raise ValueError("tokenizer pin chat template bytes are invalid")
    try:
        template_bytes = bytes.fromhex(template_hex)
    except ValueError as error:
        raise ValueError("tokenizer pin chat template bytes are invalid") from error
    if template_bytes.hex() != template_hex:
        raise ValueError("tokenizer pin chat template bytes are not canonical")
    if not template_bytes:
        raise ValueError("tokenizer pin chat template bytes must be nonempty")
    if hashlib.sha256(template_bytes).hexdigest() != claimed_template_hash:
        raise ValueError("tokenizer pin chat template bytes do not match the claimed hash")
    if claimed_template_hash != _required_sha256(
        expected_chat_template_sha256, "expected tokenizer pin chat template hash"
    ):
        raise ValueError("tokenizer pin chat template does not match the prompt capability")
    if (
        expected_config.profile == "smoke"
        and claimed_template_hash != _SMOKE_CHAT_TEMPLATE_SHA256
    ):
        raise ValueError(
            "tokenizer pin chat template does not match the registered smoke tokenizer revision"
        )
    if expected_config.chat_template_sha256 and (
        claimed_template_hash != expected_config.chat_template_sha256
    ):
        raise ValueError("tokenizer pin chat template does not match the registered config")
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": expected_config.config_hash,
            "source_manifest_sha256": _required_sha256(
                expected_source_manifest_sha256,
                "tokenizer pin source_manifest_sha256",
            ),
            "chat_template_sha256": claimed_template_hash,
        },
    )
    return shard


def _prepare_power_audit(
    store,
    config,
    factorial_rows,
    explicit_manifest,
    *,
    run_registered,
):
    design_hash = _design_sha256(factorial_rows)
    if explicit_manifest:
        shard = _require_verified_shard_kind(
            store, explicit_manifest, "power_audit", "verified power audit manifest"
        )
        rows = _read_json_rows(shard.data_path)
        if len(rows) != 1 or set(rows[0]) != {"kind", "audit"}:
            raise ValueError("power audit manifest has an invalid schema")
        audit = _power_audit_from_record(rows[0].get("audit"))
        _verify_shard_lineage(
            shard,
            {"config_sha256": config.config_hash, "design_sha256": design_hash},
        )
        _validate_power_audit(audit, design_hash)
        return audit, shard
    if not run_registered:
        raise ValueError("confirmatory build requires an explicit power audit preparation mode")
    audit = _POWER_EXECUTOR(
        factorial_rows,
        REGISTERED_POWER_GRID.interactions,
        {
            "entity_icc": REGISTERED_POWER_GRID.entity_iccs,
            "template_icc": REGISTERED_POWER_GRID.template_iccs,
            "invalid_format_rate": REGISTERED_POWER_GRID.invalid_format_rates,
        },
        CONFIRMATORY_POWER_SEED,
        simulations=CONFIRMATORY_POWER_SIMULATIONS,
    )
    if not isinstance(audit, PowerAudit):
        raise ValueError("registered power executor must return a PowerAudit")
    _validate_power_audit(audit, design_hash)
    audit_hash = _sha256_json(asdict(audit))
    shard = store.write_completed_shard(
        config.run_id,
        "mechanism_train",
        f"power-audit-{audit_hash[:16]}",
        [{"kind": "power_audit", "audit": asdict(audit)}],
        {
            "config_sha256": config.config_hash,
            "design_sha256": design_hash,
        },
        record_kind="power_audit",
    )
    return audit, shard


def _load_verified_pilot_gate(
    store: FAArtifactStore,
    path: str | Path,
    expected_config: FAConfig,
) -> dict[str, Any]:
    if expected_config.profile != "smoke":
        raise ValueError("pilot gate identity must be a smoke config")
    shard = _require_verified_shard_kind(
        store, path, "pilot_gate", "verified pilot gate sidecar manifest"
    )
    _verify_artifact_run_id(shard.manifest_path, expected_config.run_id)
    rows = _read_json_rows(shard.data_path)
    required = {
        "kind",
        "config_sha256",
        "source_manifest_sha256",
        "prompt_manifest",
        "prompt_manifest_sha256",
        "tokenizer_pin_manifest",
        "tokenizer_pin_sha256",
        "chat_template_sha256",
        "generation_sidecar_manifest",
        "generation_sidecar_sha256",
        "pilot_gate",
        "metrics",
        "evidence_sha256",
    }
    if len(rows) != 1 or set(rows[0]) != required or rows[0].get("kind") != "pilot_gate":
        raise ValueError("pilot gate tokenizer pin chain has an invalid schema")
    row = rows[0]
    prompt_path = _artifact_path_from_record(
        store, row.get("prompt_manifest"), "pilot prompt manifest"
    )
    try:
        manifest = _load_manifest(store, prompt_path, expected_config)
    except ValueError as error:
        raise ValueError(
            f"pilot gate does not match the exact registered smoke config: {error}"
        ) from error
    _verify_artifact_run_id(manifest.shard_manifest_path, expected_config.run_id)
    if manifest.namespace != "pilot" or manifest.shard_sha256 != row.get("prompt_manifest_sha256"):
        raise ValueError("pilot gate prompt manifest does not verify")
    if not manifest.chat_template_sha256:
        raise ValueError("pilot gate must bind an observed nonempty chat template pin")
    gate_pin_path = _artifact_path_from_record(
        store, row.get("tokenizer_pin_manifest"), "pilot gate tokenizer pin manifest"
    )
    _load_verified_tokenizer_pin(
        store,
        gate_pin_path,
        expected_config,
        expected_sha256=row.get("tokenizer_pin_sha256"),
        expected_source_manifest_sha256=manifest.full_manifest_sha256,
        expected_chat_template_sha256=row.get("chat_template_sha256"),
    )
    if (
        gate_pin_path != manifest.tokenizer_pin_manifest_path
        or row.get("tokenizer_pin_sha256") != manifest.tokenizer_pin_sha256
        or row.get("chat_template_sha256") != manifest.chat_template_sha256
    ):
        raise ValueError("pilot gate does not bind the prompt tokenizer pin and template")
    if (
        row.get("config_sha256") != expected_config.config_hash
        or row.get("source_manifest_sha256") != manifest.manifest_sha256
    ):
        raise ValueError("pilot gate does not match the exact registered smoke config")
    generation_path = _artifact_path_from_record(
        store, row.get("generation_sidecar_manifest"), "generation sidecar manifest"
    )
    generation = _require_generation_sidecar_manifest(store, generation_path)
    _verify_artifact_run_id(generation.manifest_path, expected_config.run_id)
    if generation.sha256 != row.get("generation_sidecar_sha256"):
        raise ValueError("pilot gate generation sidecar hash does not verify")
    if shard.namespace != "pilot" or generation.namespace != "pilot":
        raise ValueError("confirmatory construction requires pilot-namespace gate evidence")
    records = _load_verified_generation_sidecar(
        store, generation.manifest_path, manifest, expected_config
    )
    scored = _score_rows(records, manifest)
    metrics = _json_safe(estimate_behavior(scored).to_record())
    gate = _pilot_gate(scored, estimate_behavior(scored))
    evidence_hash = _sha256_json({"metrics": metrics, "pilot_gate": gate})
    if row.get("metrics") != metrics or row.get("pilot_gate") != gate:
        raise ValueError("stored pilot gate evidence does not match deterministic recomputation")
    if row.get("evidence_sha256") != evidence_hash:
        raise ValueError("pilot gate evidence hash does not verify")
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": expected_config.config_hash,
            "source_manifest_sha256": manifest.manifest_sha256,
            "generation_sidecar_sha256": generation.sha256,
            "prompt_manifest_sha256": manifest.shard_sha256,
            "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
            "chat_template_sha256": manifest.chat_template_sha256,
        },
    )
    return {"status": gate["status"], "evidence_sha256": evidence_hash}


def _load_verified_screened_matches(
    store: FAArtifactStore,
    manifest_path: str | Path,
    config: FAConfig,
) -> tuple[EntityMatch, ...]:
    shard = _require_verified_shard_kind(
        store,
        manifest_path,
        "screened_match",
        "verified screened-match manifest",
    )
    allowed_namespaces = set(config.split_counts)
    if shard.namespace not in allowed_namespaces:
        raise ValueError("screened-match manifest uses an unregistered split namespace")
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    model_hash, tokenizer_hash = _config_runtime_hashes(config)
    template_hash = config.chat_template_sha256 or _SMOKE_CHAT_TEMPLATE_SHA256
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config.config_hash,
            "model_sha256": model_hash,
            "tokenizer_sha256": tokenizer_hash,
            "chat_template_sha256": template_hash,
        },
    )
    lineage = _read_json_object(shard.manifest_path)["lineage"]
    for field in (
        "candidate_manifest_sha256",
        "questions_manifest_sha256",
        "synthetic_manifest_sha256",
        "screening_completion_sha256",
        "screening_audit_sha256",
        "screening_parser_sha256",
        "matching_policy_sha256",
    ):
        _required_sha256(lineage.get(field), field)
    if lineage["matching_policy_sha256"] != _matching_policy_sha256():
        raise ValueError("screened-match matching policy does not match current code")
    completion = _require_verified_shard_kind(
        store,
        _artifact_path_from_record(
            store,
            lineage.get("screening_completion_manifest"),
            "screening completion parent manifest",
        ),
        "screening_completion",
        "verified screening completion parent manifest",
    )
    audit = _require_verified_shard_kind(
        store,
        _artifact_path_from_record(
            store,
            lineage.get("screening_audit_manifest"),
            "screening audit parent manifest",
        ),
        "screening_audit",
        "verified screening audit parent manifest",
    )
    if completion.sha256 != lineage["screening_completion_sha256"]:
        raise ValueError("screened-match completion parent hash does not verify")
    if audit.sha256 != lineage["screening_audit_sha256"]:
        raise ValueError("screened-match audit parent hash does not verify")
    parent_lineage = {
        name: lineage[name]
        for name in (
            "config_sha256",
            "candidate_manifest_sha256",
            "questions_manifest_sha256",
            "synthetic_manifest_sha256",
            "screening_completion_sha256",
            "screening_parser_sha256",
        )
    }
    _verify_shard_lineage(audit, parent_lineage)
    _verify_shard_lineage(
        completion,
        {
            "config_sha256": config.config_hash,
            "candidate_manifest_sha256": lineage["candidate_manifest_sha256"],
            "questions_manifest_sha256": lineage["questions_manifest_sha256"],
            "model_sha256": model_hash,
            "tokenizer_sha256": tokenizer_hash,
            "chat_template_sha256": template_hash,
        },
    )

    matches = []
    for row in _read_json_rows(shard.data_path):
        if row.get("kind") != "screened_match":
            raise ValueError("screened-match row has an invalid record kind")
        values = {
            key: value
            for key, value in row.items()
            if key not in {"kind", "schema_version"}
        }
        matches.append(EntityMatch(**values))
    unique_fields = (
        ("pair IDs", (match.pair_id for match in matches)),
        ("real entity IDs", (match.real_entity_id for match in matches)),
        ("real QIDs", (match.real_qid for match in matches)),
        (
            "synthetic candidate IDs",
            (match.synthetic_candidate_id for match in matches),
        ),
    )
    for label, values in unique_fields:
        items = tuple(values)
        if len(set(items)) != len(items):
            raise ValueError(f"screened-match manifest contains duplicate {label}")
    if config.profile == "smoke":
        if shard.namespace != "pilot":
            raise ValueError("smoke screened-match manifest must use the pilot namespace")
        reserve_per_domain = 0
    elif config.profile == "confirmatory":
        reserve_per_domain = _CONFIRMATORY_RESERVE_PER_DOMAIN.get(shard.namespace)
        if reserve_per_domain is None:
            raise ValueError("confirmatory screened-match split has no registered reserve")
    else:
        raise ValueError("screened-match config profile is not registered")
    expected_count = (
        config.split_counts[shard.namespace]
        + reserve_per_domain * len(REGISTERED_ENTITY_DOMAINS)
    )
    if len(matches) != expected_count:
        raise ValueError(
            f"screened-match manifest must contain exactly {expected_count} pairs"
        )
    domains = tuple(REGISTERED_ENTITY_DOMAINS)
    if expected_count % len(domains) != 0:
        raise ValueError("pilot count cannot be balanced across registered domains")
    quota = (
        config.split_counts[shard.namespace] // len(domains)
        + reserve_per_domain
    )
    domain_counts = Counter(match.coarse_type for match in matches)
    if domain_counts != Counter({domain: quota for domain in domains}):
        raise ValueError("screened-match manifest is not exactly domain balanced")
    audit_rows = _read_json_rows(audit.data_path)
    if len(audit_rows) != 1 or audit_rows[0].get("decision") != "passed":
        raise ValueError("screened-match manifest requires a passed screening audit")
    if audit_rows[0].get("screening_completion_sha256") != completion.sha256:
        raise ValueError("screening audit does not bind its completion parent")
    audited_entity_ids = audit_rows[0].get("selected_entity_ids")
    matched_entity_ids = [match.real_entity_id for match in matches]
    if (
        not isinstance(audited_entity_ids, list)
        or len(audited_entity_ids) != len(matched_entity_ids)
        or len(set(audited_entity_ids)) != len(audited_entity_ids)
        or set(audited_entity_ids) != set(matched_entity_ids)
    ):
        raise ValueError("screened-match rows do not match the audited selection")
    if any(match.split != shard.namespace for match in matches):
        raise ValueError("screened-match rows do not match their split namespace")
    if (
        audit_rows[0].get("required_count")
        != config.split_counts[shard.namespace]
        or audit_rows[0].get("reserve_per_domain") != reserve_per_domain
        or audit_rows[0].get("selected_count") != expected_count
    ):
        raise ValueError("screening audit does not bind the registered reserve policy")
    return tuple(matches)


def _audit_confirmatory_match_pool(
    config: FAConfig,
    matches: Sequence[EntityMatch],
) -> None:
    rows = tuple(matches)
    expected_total = sum(
        count
        + _CONFIRMATORY_RESERVE_PER_DOMAIN[split]
        * len(REGISTERED_ENTITY_DOMAINS)
        for split, count in config.split_counts.items()
    )
    if len(rows) != expected_total:
        raise ValueError(
            f"confirmatory screened-match pool must contain exactly {expected_total} pairs"
        )
    for label, values in (
        ("pair IDs", (row.pair_id for row in rows)),
        ("real entity IDs", (row.real_entity_id for row in rows)),
        ("real QIDs", (row.real_qid for row in rows)),
        (
            "synthetic candidate IDs",
            (row.synthetic_candidate_id for row in rows),
        ),
        ("real names", (row.real_name.casefold() for row in rows)),
        ("synthetic names", (row.synthetic_name.casefold() for row in rows)),
    ):
        items = tuple(values)
        if len(items) != len(set(items)):
            raise ValueError(f"confirmatory screened-match pool has duplicate {label}")
    for split, split_count in config.split_counts.items():
        reserve = _CONFIRMATORY_RESERVE_PER_DOMAIN[split]
        quota = split_count // len(REGISTERED_ENTITY_DOMAINS) + reserve
        counts = Counter(
            row.coarse_type for row in rows if row.split == split
        )
        if counts != Counter(
            {domain: quota for domain in REGISTERED_ENTITY_DOMAINS}
        ):
            raise ValueError(
                f"confirmatory screened-match pool is not balanced for {split}"
            )


def _load_verified_screened_match_collection(
    store: FAArtifactStore,
    manifest_path: str | Path,
    config: FAConfig,
) -> tuple[EntityMatch, ...]:
    if config.profile != "confirmatory":
        raise ValueError("screened-match collection requires the confirmatory config")
    shard = _require_verified_shard_kind(
        store,
        manifest_path,
        "screened_match_collection",
        "verified screened-match collection manifest",
    )
    if shard.namespace != "mechanism_train":
        raise ValueError("screened-match collection must use mechanism_train")
    _verify_artifact_run_id(shard.manifest_path, config.run_id)
    lineage = _read_json_object(shard.manifest_path).get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("screened-match collection lineage is missing")
    _verify_shard_lineage(
        shard,
        {
            "config_sha256": config.config_hash,
            "matching_policy_sha256": _matching_policy_sha256(),
        },
    )
    children = lineage.get("children")
    if not isinstance(children, list) or len(children) != len(config.split_counts):
        raise ValueError("screened-match collection has invalid child lineage")
    child_matches = []
    child_namespaces = set()
    for child in children:
        if (
            not isinstance(child, Mapping)
            or set(child) != {"namespace", "manifest_path", "sha256"}
        ):
            raise ValueError("screened-match collection child has invalid schema")
        namespace = child.get("namespace")
        if namespace in child_namespaces:
            raise ValueError("screened-match collection repeats a child namespace")
        child_path = _artifact_path_from_record(
            store,
            child.get("manifest_path"),
            "screened-match child manifest",
        )
        child_shard = store.verify_shard(child_path)
        if (
            child_shard.namespace != namespace
            or child_shard.sha256 != child.get("sha256")
        ):
            raise ValueError("screened-match collection child identity changed")
        child_matches.extend(
            _load_verified_screened_matches(store, child_path, config)
        )
        child_namespaces.add(namespace)
    if child_namespaces != set(config.split_counts):
        raise ValueError("screened-match collection does not cover every split")
    expected = tuple(
        sorted(child_matches, key=lambda row: (row.split, row.pair_id))
    )
    rows = _read_json_rows(shard.data_path)
    try:
        observed = tuple(
            EntityMatch(
                **{
                    key: value
                    for key, value in _without_schema(row).items()
                    if key != "kind"
                }
            )
            for row in rows
            if row.get("kind") == "screened_match_collection"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("screened-match collection rows are invalid") from error
    if len(observed) != len(rows) or observed != expected:
        raise ValueError("screened-match collection rows differ from child shards")
    _audit_confirmatory_match_pool(config, observed)
    if lineage.get("matches_sha256") != naturalness_matches_sha256(observed):
        raise ValueError("screened-match collection hash does not verify")
    return observed


def _require_generation_sidecar_manifest(
    store: FAArtifactStore, path: str | Path, *, label: str = "verified generation sidecar manifest"
):
    return _require_verified_shard_kind(store, path, "generation", label)


def _load_verified_generation_sidecar(store, manifest_path, manifest, config=None):
    shard = _require_generation_sidecar_manifest(store, manifest_path)
    expected_model = manifest.model_sha256
    expected_tokenizer = manifest.tokenizer_sha256
    if config is not None and (expected_model, expected_tokenizer) != _config_runtime_hashes(config):
        raise ValueError("generation runtime pins do not match config")
    expected_lineage = {
        "config_sha256": manifest.config_hash,
        "source_manifest_sha256": manifest.manifest_sha256,
        "model_sha256": expected_model,
        "tokenizer_sha256": expected_tokenizer,
        "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
        "chat_template_sha256": manifest.chat_template_sha256,
    }
    _verify_shard_lineage(shard, expected_lineage)
    examples = {row.example_id: row for row in manifest.examples}
    records = _read_json_rows(shard.data_path)
    expected_multiset = Counter(row.example_id for row in manifest.examples)
    observed_multiset = Counter(record.get("example_id") for record in records)
    if observed_multiset != expected_multiset:
        raise ValueError(
            "verified generation sidecar must contain every expected example exactly once"
        )
    for record in records:
        example = examples.get(record.get("example_id"))
        if example is None:
            raise ValueError("verified generation sidecar contains an unregistered example")
        expected = {
            "config_sha256": manifest.config_hash,
            "example_sha256": example.canonical_payload_sha256,
            "model_sha256": expected_model,
            "tokenizer_sha256": expected_tokenizer,
            "tokenizer_pin_sha256": manifest.tokenizer_pin_sha256,
            "chat_template_sha256": manifest.chat_template_sha256,
        }
        if any(record.get(name) != value for name, value in expected.items()):
            raise ValueError("verified generation sidecar provenance does not match its manifest")
        if record.get("kind") != "generation":
            raise ValueError("verified generation sidecar has an invalid record kind")
        if record.get("generation") != manifest.generation:
            raise ValueError("verified generation sidecar generation config does not match")
        if record.get("example") != _record_value(example):
            raise ValueError("verified generation sidecar canonical example does not match")
        if record.get("status") not in {"completed", "infrastructure_failure"}:
            raise ValueError("verified generation sidecar has an invalid completion status")
    return records


def _require_verified_shard_kind(store, path, record_kind, label):
    source = Path(path)
    if not source.name.endswith(".jsonl.manifest.json"):
        raise ValueError(f"{label} must name an immutable shard manifest")
    try:
        shard = store.verify_shard(source)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"{label} does not verify: {error}") from error
    if shard.record_kind != record_kind:
        raise ValueError(f"{label} has record kind {shard.record_kind}, expected {record_kind}")
    return shard


def _config_runtime_hashes(config: FAConfig) -> tuple[str, str]:
    return (
        _sha256_json({"model_id": config.model_id, "revision": config.model_revision}),
        _sha256_json(
            {"model_id": config.model_id, "revision": config.tokenizer_revision}
        ),
    )


def _design_sha256(rows) -> str:
    return hashlib.sha256(
        "\n".join(sorted(row.example_id for row in rows)).encode("utf-8")
    ).hexdigest()


def _power_audit_from_record(value: Any) -> PowerAudit:
    if not isinstance(value, dict) or set(value) != {
        "design_sha256",
        "seed",
        "simulations",
        "cells",
        "registered_grid",
    }:
        raise ValueError("power audit record has an invalid schema")
    cells = value.get("cells")
    if not isinstance(cells, list):
        raise ValueError("power audit cells must be a list")
    try:
        return PowerAudit(
            design_sha256=value["design_sha256"],
            seed=value["seed"],
            simulations=value["simulations"],
            cells=tuple(PowerCell(**cell) for cell in cells),
            registered_grid=value["registered_grid"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"power audit record is invalid: {error}") from error


def _validate_power_audit(audit: PowerAudit, design_hash: str) -> None:
    expected = {
        (absent, entity, template, invalid, interaction)
        for absent in REGISTERED_POWER_GRID.absent_attempt_rates
        for entity in REGISTERED_POWER_GRID.entity_iccs
        for template in REGISTERED_POWER_GRID.template_iccs
        for invalid in REGISTERED_POWER_GRID.invalid_format_rates
        for interaction in REGISTERED_POWER_GRID.interactions
    }
    if (
        not audit.registered_grid
        or audit.design_sha256 != design_hash
        or audit.seed != CONFIRMATORY_POWER_SEED
        or audit.simulations != CONFIRMATORY_POWER_SIMULATIONS
        or len(audit.cells) != 180
    ):
        raise ValueError("power audit does not match the exact registered design")
    observed = set()
    for cell in audit.cells:
        values = (
            cell.absent_attempt_rate,
            cell.entity_icc,
            cell.template_icc,
            cell.invalid_format_rate,
            cell.interaction,
            cell.estimated_power,
            cell.monte_carlo_standard_error,
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError("power audit contains non-finite values")
        key = values[:5]
        if key not in expected or key in observed:
            raise ValueError("power audit must contain all 180 unique registered cells")
        observed.add(key)
        expected_mcse = math.sqrt(
            cell.estimated_power * (1.0 - cell.estimated_power) / cell.simulations
        ) if 0.0 <= cell.estimated_power <= 1.0 and cell.simulations > 0 else math.nan
        if (
            cell.simulations != CONFIRMATORY_POWER_SIMULATIONS
            or not 0.0 <= cell.estimated_power <= 1.0
            or not math.isclose(
                cell.monte_carlo_standard_error,
                expected_mcse,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("power audit cell power or MCSE is invalid")
    if observed != expected:
        raise ValueError("power audit must contain all 180 unique registered cells")


def _verify_shard_lineage(shard, expected):
    sidecar = _read_json_object(shard.manifest_path)
    lineage = sidecar.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("verified shard lineage is invalid")
    for name, value in expected.items():
        if lineage.get(name) != value:
            raise ValueError(f"verified shard lineage does not bind {name}")


def _verify_artifact_run_id(manifest_path: Path, expected_run_id: str) -> None:
    sidecar = _read_json_object(manifest_path)
    if sidecar.get("run_id") != expected_run_id:
        raise ValueError("pilot gate artifact does not belong to the registered smoke run")


def _artifact_path_from_record(store, value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{label} path is invalid")
    path = store.root / relative
    try:
        path.relative_to(store.root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes the artifact root") from error
    return path


def _read_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("explicit manifest must be a JSON object")
    return value


def _read_json_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    source = Path(path)
    if source.suffix == ".jsonl":
        values = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    else:
        value = json.loads(source.read_text(encoding="utf-8"))
        values = value.get("rows", value) if isinstance(value, dict) else value
    if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
        raise ValueError("explicit manifest must contain JSON object rows")
    return tuple(values)


def _without_schema(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "schema_version"}


def _write_manifest(root: Path, run_id: str, label: str, value: Any) -> Path:
    directory = root / "fa_manifests" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{label}-{hashlib.sha256(json.dumps(value, sort_keys=True, default=list).encode()).hexdigest()}.json"
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError("immutable manifest path collision")
        return path
    path.write_text(payload, encoding="utf-8")
    return path


def _required_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value
