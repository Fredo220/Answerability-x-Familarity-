"""Deterministic R11 relation selection and held-out instrument validation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trajectory_extractor.fa_confirmatory_source import REGISTERED_DOMAINS

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def select_relation_triplets(
    rows: Sequence[Mapping[str, Any]],
    *,
    relation_bank: Mapping[str, Sequence[str]],
    qualification_threshold: int = 2,
    expected_candidates_per_domain: int,
    expected_validation_candidates_per_domain: int | None = None,
    minimum_qualified_per_domain_validation: int,
    minimum_success_per_relation_validation: int = 1,
    config_sha256: str,
    source_manifest_sha256: str,
    git_commit: str,
    development_execution_identity_sha256: str,
    screening_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one three-relation instrument per domain on open data only."""
    if qualification_threshold != 2:
        raise ValueError("R11 fixes the qualification threshold at two of three")
    if (
        type(expected_candidates_per_domain) is not int
        or expected_candidates_per_domain <= 0
        or type(minimum_qualified_per_domain_validation) is not int
        or minimum_qualified_per_domain_validation <= 0
    ):
        raise ValueError("R11 candidate counts and validation gate must be positive")
    if expected_validation_candidates_per_domain is None:
        expected_validation_candidates_per_domain = expected_candidates_per_domain
    if (
        type(expected_validation_candidates_per_domain) is not int
        or expected_validation_candidates_per_domain <= 0
        or type(minimum_success_per_relation_validation) is not int
        or minimum_success_per_relation_validation < 0
    ):
        raise ValueError("R11 validation design is invalid")
    if (
        _SHA256.fullmatch(config_sha256) is None
        or _SHA256.fullmatch(source_manifest_sha256) is None
        or _SHA256.fullmatch(development_execution_identity_sha256) is None
        or _GIT_SHA.fullmatch(git_commit) is None
    ):
        raise ValueError("R11 selection provenance binding is invalid")
    required_screening_identity = {
        "config_sha256",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "chat_template_sha256",
        "parser_sha256",
        "semantic_audit_sha256",
    }
    if set(screening_identity) != required_screening_identity:
        raise ValueError("R11 screening identity binding is incomplete")
    bank = _validate_relation_bank(relation_bank)
    observations, split = _validate_rows(rows, bank)
    if split != "instrument_development":
        raise ValueError("relation selection requires instrument_development")
    _require_complete_matrix(
        observations,
        bank,
        expected_candidates_per_domain=expected_candidates_per_domain,
    )

    selected_relations: dict[str, list[str]] = {}
    selected_metrics: dict[str, dict[str, int]] = {}
    combination_audit: dict[str, list[dict[str, Any]]] = {}
    for domain in REGISTERED_DOMAINS:
        domain_rows = {
            key: value
            for key, value in observations.items()
            if key[0] == domain
        }
        candidates = []
        for relations in itertools.combinations(bank[domain], 3):
            metrics = _combination_metrics(
                domain_rows,
                relations,
                qualification_threshold,
            )
            candidates.append(
                {
                    "relations": list(relations),
                    **metrics,
                }
            )
        ranked = sorted(
            candidates,
            key=lambda row: (
                -row["qualified_count"],
                -row["minimum_relation_success_count"],
                -row["total_success_count"],
                tuple(row["relations"]),
            ),
        )
        winner = ranked[0]
        selected_relations[domain] = list(winner["relations"])
        selected_metrics[domain] = {
            key: int(winner[key])
            for key in (
                "complete_entity_count",
                "qualified_count",
                "minimum_relation_success_count",
                "total_success_count",
            )
        }
        combination_audit[domain] = ranked

    result = {
        "schema_version": 1,
        "kind": "fa_r11_relation_selection",
        "claim_scope": "open_instrument_development_only",
        "selection_rule": (
            "maximize qualified_count; then minimum relation successes; "
            "then total successes; then lexicographic relation IDs"
        ),
        "qualification_threshold": qualification_threshold,
        "config_sha256": config_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "git_commit": git_commit,
        "development_execution_identity_sha256": (
            development_execution_identity_sha256
        ),
        "screening_identity": dict(screening_identity),
        "development_candidates_per_domain": expected_candidates_per_domain,
        "validation_candidates_per_domain": expected_validation_candidates_per_domain,
        "minimum_qualified_per_domain_validation": (
            minimum_qualified_per_domain_validation
        ),
        "minimum_success_per_relation_validation": (
            minimum_success_per_relation_validation
        ),
        "relation_bank": {
            domain: list(bank[domain]) for domain in REGISTERED_DOMAINS
        },
        "relation_bank_sha256": _canonical_sha256(
            {domain: list(bank[domain]) for domain in REGISTERED_DOMAINS}
        ),
        "development_qids": sorted(
            {
                qid
                for row in rows
                if isinstance((qid := row.get("qid")), str) and qid
            }
        ),
        "development_items_sha256": _canonical_sha256(
            sorted(
                (dict(row) for row in rows),
                key=lambda row: (
                    str(row.get("domain")),
                    str(row.get("entity_id")),
                    str(row.get("relation_id")),
                ),
            )
        ),
        "selected_relations": selected_relations,
        "selected_metrics": selected_metrics,
        "combination_audit": combination_audit,
    }
    return {**result, "selection_sha256": _canonical_sha256(result)}


def assess_frozen_validation(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection: Mapping[str, Any],
    validation_execution_identity_sha256: str,
    validation_items_sha256: str,
) -> dict[str, Any]:
    """Evaluate a frozen R11 instrument once on entity-disjoint validation."""
    frozen = dict(selection)
    selection_sha256 = frozen.pop("selection_sha256", None)
    if selection_sha256 != _canonical_sha256(frozen):
        raise ValueError("R11 selection hash mismatch")
    if (
        _SHA256.fullmatch(validation_execution_identity_sha256) is None
        or _SHA256.fullmatch(validation_items_sha256) is None
    ):
        raise ValueError("R11 validation provenance binding is invalid")
    threshold = selection.get("qualification_threshold")
    if threshold != 2:
        raise ValueError("validation requires the frozen two-of-three threshold")
    selected = _validate_relation_bank(selection.get("selected_relations", {}))
    if any(len(relations) != 3 for relations in selected.values()):
        raise ValueError("validation requires exactly three frozen relations")
    selected_rows = [
        row
        for row in rows
        if str(row.get("domain", "")) in selected
        and str(row.get("relation_id", ""))
        in selected[str(row.get("domain", ""))]
    ]
    observations, split = _validate_rows(selected_rows, selected)
    if split != "construction_validation":
        raise ValueError("frozen validation requires construction_validation")
    expected_validation_count = selection.get("validation_candidates_per_domain")
    _require_complete_matrix(
        observations,
        selected,
        expected_candidates_per_domain=expected_validation_count,
    )
    development_qids = selection.get("development_qids")
    if not isinstance(development_qids, list) or any(
        not isinstance(qid, str) for qid in development_qids
    ):
        raise ValueError("R11 selection lacks frozen development QIDs")
    validation_qids = {
        str(row.get("qid"))
        for row in selected_rows
        if isinstance(row.get("qid"), str)
    }
    if validation_qids.intersection(development_qids):
        raise ValueError("development and validation must be entity-disjoint")
    minimum_qualified_per_domain = selection.get(
        "minimum_qualified_per_domain_validation"
    )
    minimum_success_per_relation = selection.get(
        "minimum_success_per_relation_validation"
    )
    if (
        type(minimum_qualified_per_domain) is not int
        or minimum_qualified_per_domain <= 0
        or type(minimum_success_per_relation) is not int
        or minimum_success_per_relation < 0
    ):
        raise ValueError("R11 selection has an invalid frozen validation gate")

    by_domain = {}
    failed_domains = []
    for domain in REGISTERED_DOMAINS:
        metrics = _combination_metrics(
            {
                key: value
                for key, value in observations.items()
                if key[0] == domain
            },
            tuple(selected[domain]),
            threshold,
        )
        relation_successes = _relation_successes(
            observations,
            domain,
            tuple(selected[domain]),
        )
        passed = (
            metrics["qualified_count"] >= minimum_qualified_per_domain
            and all(
                count >= minimum_success_per_relation
                for count in relation_successes.values()
            )
        )
        if not passed:
            failed_domains.append(domain)
        by_domain[domain] = {
            **metrics,
            "relation_success_count": relation_successes,
            "gate_passed": passed,
        }

    result = {
        "schema_version": 1,
        "kind": "fa_r11_frozen_validation",
        "claim_scope": "instrument_readiness_only",
        "gate_passed": not failed_domains,
        "failed_domains": failed_domains,
        "qualification_threshold": threshold,
        "minimum_qualified_per_domain": minimum_qualified_per_domain,
        "minimum_success_per_relation": minimum_success_per_relation,
        "selected_relations": {
            domain: list(selected[domain]) for domain in REGISTERED_DOMAINS
        },
        "selection_sha256": selection_sha256,
        "validation_execution_identity_sha256": (
            validation_execution_identity_sha256
        ),
        "validation_items_sha256": validation_items_sha256,
        "by_domain": by_domain,
    }
    return {**result, "validation_sha256": _canonical_sha256(result)}


def main(argv: Sequence[str] | None = None) -> int:
    """Select on open development or evaluate one frozen validation run."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--items", type=Path, required=True)
    select_parser.add_argument("--config", type=Path, required=True)
    select_parser.add_argument("--source-manifest", type=Path, required=True)
    select_parser.add_argument("--execution-identity", type=Path, required=True)
    select_parser.add_argument("--git-commit", required=True)
    select_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--items", type=Path, required=True)
    validate_parser.add_argument("--selection", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "select":
        config = _read_json(args.config)
        execution_identity = _read_json(args.execution_identity)
        source_manifest_sha256 = _sha256_file(args.source_manifest)
        if (
            execution_identity.get("split") != "instrument_development"
            or execution_identity.get("source_manifest_sha256")
            != source_manifest_sha256
            or execution_identity.get("git_commit") != args.git_commit
        ):
            raise ValueError("R11 development execution identity mismatch")
        screening_identity = {
            key: execution_identity[key]
            for key in (
                "config_sha256",
                "model_id",
                "model_revision",
                "tokenizer_revision",
                "chat_template_sha256",
                "parser_sha256",
                "semantic_audit_sha256",
            )
        }
        result = select_relation_triplets(
            _read_jsonl(args.items),
            relation_bank=config["relation_bank"],
            qualification_threshold=config["qualification_threshold"],
            expected_candidates_per_domain=config[
                "development_candidates_per_domain"
            ],
            expected_validation_candidates_per_domain=config[
                "validation_candidates_per_domain"
            ],
            minimum_qualified_per_domain_validation=config[
                "minimum_qualified_per_domain_validation"
            ],
            config_sha256=_sha256_file(args.config),
            source_manifest_sha256=source_manifest_sha256,
            git_commit=args.git_commit,
            development_execution_identity_sha256=_sha256_file(
                args.execution_identity
            ),
            screening_identity=screening_identity,
        )
    else:
        items_path = args.items
        execution_identity_path = items_path.parent / "execution_identity.json"
        execution_identity = _read_json(execution_identity_path)
        selection = _read_json(args.selection)
        if (
            execution_identity.get("split") != "construction_validation"
            or execution_identity.get("selection_sha256")
            != selection.get("selection_sha256")
            or execution_identity.get("source_manifest_sha256")
            != selection.get("source_manifest_sha256")
            or execution_identity.get("git_commit") != selection.get("git_commit")
        ):
            raise ValueError("R11 validation execution identity mismatch")
        result = assess_frozen_validation(
            _read_jsonl(items_path),
            selection=selection,
            validation_execution_identity_sha256=_sha256_file(
                execution_identity_path
            ),
            validation_items_sha256=_sha256_file(items_path),
        )
    _write_json_immutable(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _validate_relation_bank(
    relation_bank: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if not isinstance(relation_bank, Mapping) or set(relation_bank) != set(
        REGISTERED_DOMAINS
    ):
        raise ValueError("relation bank must cover exactly the registered domains")
    normalized = {}
    for domain in REGISTERED_DOMAINS:
        relations = tuple(sorted(str(value) for value in relation_bank[domain]))
        if len(relations) < 3 or len(set(relations)) != len(relations):
            raise ValueError("each domain requires at least three unique relations")
        if any(not relation for relation in relations):
            raise ValueError("relation IDs must be nonempty")
        normalized[domain] = relations
    return normalized


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    relation_bank: Mapping[str, Sequence[str]],
) -> tuple[dict[tuple[str, str, str], bool], str]:
    if not rows:
        raise ValueError("screening rows must be nonempty")
    observations: dict[tuple[str, str, str], bool] = {}
    entity_qids: dict[tuple[str, str], str] = {}
    splits = set()
    for row in rows:
        domain = str(row.get("domain", ""))
        entity_id = str(row.get("entity_id", ""))
        relation_id = str(row.get("relation_id", ""))
        qid = str(row.get("qid", ""))
        split = str(row.get("split", ""))
        is_correct = row.get("is_correct")
        if domain not in relation_bank:
            raise ValueError("screening row has an unregistered domain")
        if (
            not entity_id
            or not re.fullmatch(r"Q[1-9][0-9]*", qid)
            or relation_id not in relation_bank[domain]
        ):
            raise ValueError("screening row has an invalid entity or relation")
        if type(is_correct) is not bool:
            raise ValueError("screening correctness must be boolean")
        key = (domain, entity_id, relation_id)
        if key in observations:
            raise ValueError("screening rows contain a duplicate observation")
        entity_key = (domain, entity_id)
        if entity_key in entity_qids and entity_qids[entity_key] != qid:
            raise ValueError("screening entity maps to inconsistent QIDs")
        entity_qids[entity_key] = qid
        observations[key] = is_correct
        splits.add(split)
    if len(splits) != 1:
        raise ValueError("screening rows must use exactly one split")
    return observations, splits.pop()


def _require_complete_matrix(
    observations: Mapping[tuple[str, str, str], bool],
    relation_bank: Mapping[str, Sequence[str]],
    *,
    expected_candidates_per_domain: Any,
) -> None:
    if (
        type(expected_candidates_per_domain) is not int
        or expected_candidates_per_domain <= 0
    ):
        raise ValueError("expected candidate count must be positive")
    for domain in REGISTERED_DOMAINS:
        entities = sorted(
            {
                entity_id
                for row_domain, entity_id, _ in observations
                if row_domain == domain
            }
        )
        if len(entities) != expected_candidates_per_domain:
            raise ValueError("screening rows violate the frozen candidate count")
        expected = {
            (domain, entity_id, relation)
            for entity_id in entities
            for relation in relation_bank[domain]
        }
        observed = {
            key for key in observations if key[0] == domain
        }
        if observed != expected:
            raise ValueError("screening rows require a complete relation matrix")


def _combination_metrics(
    observations: Mapping[tuple[str, str, str], bool],
    relations: Sequence[str],
    threshold: int,
) -> dict[str, int]:
    entity_ids = sorted({entity_id for _, entity_id, _ in observations})
    domain = next(iter(observations))[0] if observations else ""
    complete = [
        entity_id
        for entity_id in entity_ids
        if all(
            (domain, entity_id, relation) in observations
            for relation in relations
        )
    ]
    qualified = sum(
        sum(observations[(domain, entity_id, relation)] for relation in relations)
        >= threshold
        for entity_id in complete
    )
    relation_successes = [
        sum(observations[(domain, entity_id, relation)] for entity_id in complete)
        for relation in relations
    ]
    return {
        "complete_entity_count": len(complete),
        "qualified_count": qualified,
        "minimum_relation_success_count": min(relation_successes, default=0),
        "total_success_count": sum(relation_successes),
    }


def _relation_successes(
    observations: Mapping[tuple[str, str, str], bool],
    domain: str,
    relations: tuple[str, ...],
) -> dict[str, int]:
    entity_ids = {
        entity_id
        for row_domain, entity_id, _ in observations
        if row_domain == domain
    }
    return {
        relation: sum(
            observations.get((domain, entity_id, relation), False)
            for entity_id in entity_ids
        )
        for relation in relations
    }


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact must be an object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL artifact must contain objects")
    return rows


def _write_json_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to replace immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
