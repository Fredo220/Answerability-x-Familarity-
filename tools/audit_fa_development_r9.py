#!/usr/bin/env python3
"""Independently audit the frozen Source-v6 R9 development corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

AUDITOR_ID = "independent-r9-structural-auditor"
SOURCE_REVISION = "fa-development-source-v6-r9"
SPLITS = ("instrument_development", "construction_validation")
DOMAINS = ("creative_work", "organization", "person", "place")
SELECTION_SEED = 20260725
CANDIDATES_PER_DOMAIN = 24
_RESERVED_OUTPUTS = frozenset({"unknown"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path, label: str, blockers: list[str]) -> str | None:
    try:
        return _sha256_file(path)
    except OSError as error:
        blockers.append(f"{label}: cannot hash file ({error})")
        return None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mapping(value: Any, label: str, blockers: list[str]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    blockers.append(f"{label}: expected a JSON object")
    return {}


def _array(value: Any, label: str, blockers: list[str]) -> list[Any]:
    if isinstance(value, list):
        return value
    blockers.append(f"{label}: expected a JSON array")
    return []


def _load_json(path: Path, label: str, blockers: list[str]) -> Any:
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        blockers.append(f"{label}: cannot read JSON ({error})")
        return None


def _load_jsonl(path: Path, label: str, blockers: list[str]) -> list[Any]:
    try:
        return _read_jsonl(path)
    except (OSError, json.JSONDecodeError) as error:
        blockers.append(f"{label}: cannot read JSONL ({error})")
        return []


def _safe_child(root: Path, name: Any, label: str, blockers: list[str]) -> Path | None:
    if not isinstance(name, str) or not name:
        blockers.append(f"{label}: missing relative path")
        return None
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        blockers.append(f"{label}: path escapes source root")
        return None
    return candidate


def _check(condition: bool, message: str, blockers: list[str]) -> None:
    if not condition:
        blockers.append(message)


def _normal_form(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _selection_key(domain: str, qid: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{SELECTION_SEED}:{domain}:{qid}".encode()
    ).hexdigest()
    return digest, qid


def _valid_correction_surfaces(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(surface, str)
            and surface.strip() == surface
            and bool(surface)
            and "\n" not in surface
            and _normal_form(surface) not in _RESERVED_OUTPUTS
            for surface in value
        )
        and len({_normal_form(surface) for surface in value}) == len(value)
    )


def _manifest_files(
    root: Path,
    registry: dict[str, Any],
    label: str,
    blockers: list[str],
) -> dict[str, tuple[list[Any], list[Any]]]:
    materialized = _mapping(registry.get("materialized_files"), label, blockers)
    rows: dict[str, tuple[list[Any], list[Any]]] = {}
    _check(
        set(materialized) == set(SPLITS),
        f"{label}: split registry mismatch",
        blockers,
    )
    for split in SPLITS:
        entry = _mapping(materialized.get(split), f"{label}.{split}", blockers)
        candidate_path = _safe_child(
            root,
            entry.get("candidate_manifest"),
            f"{label}.{split}.candidate_manifest",
            blockers,
        )
        question_path = _safe_child(
            root,
            entry.get("question_manifest"),
            f"{label}.{split}.question_manifest",
            blockers,
        )
        candidates = (
            _array(
                _load_json(candidate_path, f"{label}.{split}.candidates", blockers),
                f"{label}.{split}.candidates",
                blockers,
            )
            if candidate_path is not None
            else []
        )
        questions = (
            _array(
                _load_json(question_path, f"{label}.{split}.questions", blockers),
                f"{label}.{split}.questions",
                blockers,
            )
            if question_path is not None
            else []
        )
        if candidate_path is not None:
            candidate_sha256 = _file_sha256(
                candidate_path, f"{label}.{split}.candidates", blockers
            )
            _check(
                candidate_sha256 == entry.get("candidate_sha256"),
                f"{label}.{split}: candidate manifest hash mismatch",
                blockers,
            )
        if question_path is not None:
            question_sha256 = _file_sha256(
                question_path, f"{label}.{split}.questions", blockers
            )
            _check(
                question_sha256 == entry.get("question_sha256"),
                f"{label}.{split}: question manifest hash mismatch",
                blockers,
            )
        rows[split] = (candidates, questions)
    return rows


def _qid_set(rows: list[Any], label: str, blockers: list[str]) -> set[str]:
    qids = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("qid"), str):
            blockers.append(f"{label}: row has no string QID")
            continue
        qids.append(row["qid"])
    _check(len(qids) == len(set(qids)), f"{label}: duplicate QIDs", blockers)
    return set(qids)


def _verify_r8_inputs(
    r8_root: Path, derivation: dict[str, Any], blockers: list[str]
) -> tuple[set[str], set[str], str]:
    frame_path = r8_root / "frame" / "development_source_frame_v1.json"
    integrity_path = r8_root / "source_integrity_v1.json"
    audit_path = r8_root / "pre_model_semantic_audit_v1.json"
    audit_items_path = r8_root / "pre_model_semantic_audit_items_v1.jsonl"
    inputs = {
        "r8_frame_sha256": frame_path,
        "r8_integrity_sha256": integrity_path,
        "r8_audit_sha256": audit_path,
        "r8_audit_items_sha256": audit_items_path,
    }
    for field, path in inputs.items():
        try:
            observed = _sha256_file(path)
        except OSError as error:
            blockers.append(f"{field}: cannot hash input ({error})")
            continue
        _check(
            derivation.get(field) == observed,
            f"derivation {field} mismatch",
            blockers,
        )

    frame = _mapping(_load_json(frame_path, "R8 frame", blockers), "R8 frame", blockers)
    integrity = _mapping(
        _load_json(integrity_path, "R8 integrity", blockers), "R8 integrity", blockers
    )
    audit = _mapping(
        _load_json(audit_path, "R8 semantic audit", blockers),
        "R8 semantic audit",
        blockers,
    )
    audit_rows = _load_jsonl(audit_items_path, "R8 semantic audit items", blockers)
    manifests = _manifest_files(r8_root, integrity, "R8 integrity", blockers)
    r8_candidates = [row for candidates, _ in manifests.values() for row in candidates]
    r8_questions = [row for _, questions in manifests.values() for row in questions]
    r8_qids = _qid_set(r8_candidates, "R8 candidates", blockers)
    _check(len(r8_candidates) == 192, "R8 candidate count is not 192", blockers)
    _check(len(r8_qids) == 192, "R8 candidate QIDs are not unique", blockers)
    _check(len(r8_questions) == 576, "R8 question count is not 576", blockers)
    _check(len(audit_rows) == 576, "R8 semantic audit row count is not 576", blockers)

    audit_items_sha256 = ""
    try:
        audit_items_sha256 = _sha256_file(audit_items_path)
    except OSError:
        pass
    _check(
        audit.get("items_sha256") == audit_items_sha256,
        "R8 semantic audit does not bind its items file",
        blockers,
    )
    _check(
        derivation.get("r8_audit_items_sha256") == audit_items_sha256,
        "derivation does not bind R8 semantic audit items",
        blockers,
    )

    excluded_prior = frame.get("excluded_prior_qids")
    prior_qids = (
        set(excluded_prior)
        if isinstance(excluded_prior, list)
        and all(isinstance(qid, str) for qid in excluded_prior)
        else set()
    )
    _check(
        isinstance(excluded_prior, list)
        and all(isinstance(qid, str) for qid in excluded_prior)
        and len(prior_qids) == len(excluded_prior),
        "R8 frame prior exclusions are invalid or duplicate",
        blockers,
    )
    return r8_qids, prior_qids, audit_items_sha256


def _verify_corrections(
    r8_root: Path,
    r9_root: Path,
    derivation: dict[str, Any],
    audit_items_sha256: str,
    blockers: list[str],
) -> None:
    corrections_path = r9_root / "alias_corrections_v1.json"
    corrections = _mapping(
        _load_json(corrections_path, "R9 correction manifest", blockers),
        "R9 correction manifest",
        blockers,
    )
    try:
        correction_sha256 = _sha256_file(corrections_path)
    except OSError:
        correction_sha256 = ""
    _check(
        derivation.get("corrections_sha256") == correction_sha256,
        "derivation correction manifest hash mismatch",
        blockers,
    )
    _check(
        corrections.get("schema_version") == 1,
        "R9 correction manifest schema mismatch",
        blockers,
    )
    _check(
        corrections.get("kind") == "fa_source_v6_r9_alias_corrections",
        "R9 correction manifest kind mismatch",
        blockers,
    )
    _check(
        corrections.get("source_revision") == SOURCE_REVISION,
        "R9 correction manifest source revision mismatch",
        blockers,
    )
    _check(
        corrections.get("source_audit_items_sha256") == audit_items_sha256,
        "R9 corrections do not bind R8 audit items",
        blockers,
    )
    items = _array(corrections.get("items"), "R9 correction items", blockers)
    items_sha256 = _canonical_sha256(items)
    _check(
        corrections.get("items_sha256") == items_sha256,
        "R9 correction manifest items hash mismatch",
        blockers,
    )
    _check(
        derivation.get("correction_items_sha256") == items_sha256,
        "derivation correction items hash mismatch",
        blockers,
    )
    _check(
        corrections.get("item_count") == len(items),
        "R9 correction manifest item count mismatch",
        blockers,
    )

    audit_rows = _load_jsonl(
        r8_root / "pre_model_semantic_audit_items_v1.jsonl",
        "R8 semantic audit items for correction binding",
        blockers,
    )
    audit_by_question = {
        row["question_id"]
        : row
        for row in audit_rows
        if isinstance(row, dict) and isinstance(row.get("question_id"), str)
    }
    expected_questions = {
        row["question_id"]
        for row in audit_rows
        if isinstance(row, dict)
        and isinstance(row.get("question_id"), str)
        and isinstance(row.get("blocker_ids"), list)
        and "ordinary_surface_missing" in row["blocker_ids"]
    }
    correction_questions: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("question_id"), str):
            blockers.append("R9 correction item is malformed")
            continue
        question_id = item["question_id"]
        correction_questions.add(question_id)
        audit_row = audit_by_question.get(question_id)
        if not isinstance(audit_row, dict):
            blockers.append("R9 correction references a missing R8 audit item")
            continue
        _check(
            item.get("qid") == audit_row.get("qid"),
            f"R9 correction QID mismatch for {question_id}",
            blockers,
        )
        _check(
            item.get("blocker_id") == "ordinary_surface_missing"
            and item.get("blocker_id") in audit_row.get("blocker_ids", []),
            f"R9 correction blocker binding mismatch for {question_id}",
            blockers,
        )
        _check(
            _valid_correction_surfaces(item.get("accepted_surfaces_to_add")),
            f"R9 correction surfaces are invalid for {question_id}",
            blockers,
        )
    _check(
        len(correction_questions) == len(items),
        "R9 correction items have duplicate question IDs",
        blockers,
    )
    _check(
        correction_questions == expected_questions,
        "R9 corrections do not exactly cover ordinary-surface R8 audit items",
        blockers,
    )


def _replay_expected_decisions(
    r8_root: Path,
    r9_root: Path,
    blockers: list[str],
) -> dict[str, dict[str, Any]]:
    """Independently replay R9 eligibility, SHA selection, and split assignment."""
    r8_root = r8_root.resolve()
    r9_root = r9_root.resolve()
    integrity = _mapping(
        _load_json(r8_root / "source_integrity_v1.json", "R8 replay integrity", blockers),
        "R8 replay integrity",
        blockers,
    )
    manifests = _manifest_files(r8_root, integrity, "R8 replay integrity", blockers)
    candidates = [row for rows, _ in manifests.values() for row in rows]
    audit_rows = _load_jsonl(
        r8_root / "pre_model_semantic_audit_items_v1.jsonl",
        "R8 replay semantic audit items",
        blockers,
    )
    corrections = _mapping(
        _load_json(
            r9_root / "alias_corrections_v1.json",
            "R9 replay correction manifest",
            blockers,
        ),
        "R9 replay correction manifest",
        blockers,
    )
    correction_items = _array(
        corrections.get("items"),
        "R9 replay correction items",
        blockers,
    )
    correction_by_question = {
        row["question_id"]: row
        for row in correction_items
        if isinstance(row, dict) and isinstance(row.get("question_id"), str)
    }
    rows_by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in audit_rows:
        if isinstance(row, dict) and isinstance(row.get("qid"), str):
            rows_by_qid.setdefault(row["qid"], []).append(row)

    decisions: dict[str, dict[str, Any]] = {}
    eligible: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("qid"), str):
            blockers.append("R8 replay candidate is malformed")
            continue
        qid = candidate["qid"]
        domain = candidate.get("coarse_type")
        rows = rows_by_qid.get(qid, [])
        _check(len(rows) == 3, f"R9 replay {qid} does not have three audit rows", blockers)
        hard_blockers = sorted(
            {
                blocker
                for row in rows
                for blocker in row.get("blocker_ids", [])
                if blocker != "ordinary_surface_missing"
            }
        )
        correction_question_ids = sorted(
            row["question_id"]
            for row in rows
            if row.get("question_id") in correction_by_question
        )
        decisions[qid] = {
            "qid": qid,
            "domain": domain,
            "hard_blockers": hard_blockers,
            "correction_question_ids": correction_question_ids,
            "decision": "excluded_hard_blocker" if hard_blockers else "eligible",
        }
        if hard_blockers:
            continue

        aliases = [list(values) for values in candidate.get("screening_aliases", [])]
        _check(
            len(aliases) == 3 and all(isinstance(values, list) for values in aliases),
            f"R9 replay {qid} screening aliases are malformed",
            blockers,
        )
        if len(aliases) != 3:
            continue
        for row in rows:
            correction = correction_by_question.get(row.get("question_id"))
            if correction is None:
                continue
            try:
                index = int(row["question_id"].rsplit("-q", 1)[1]) - 1
            except (KeyError, ValueError, IndexError):
                blockers.append(f"R9 replay correction index is invalid for {qid}")
                continue
            if index not in range(3):
                blockers.append(f"R9 replay correction index is out of range for {qid}")
                continue
            aliases[index].extend(correction.get("accepted_surfaces_to_add", []))
            aliases[index] = list(dict.fromkeys(aliases[index]))
        eligible[qid] = {
            "domain": domain,
            "name": candidate.get("name"),
            "aliases": aliases,
            "corrected": bool(correction_question_ids),
        }

    candidate_owners: dict[str, set[str]] = {}
    answer_owners: dict[str, set[str]] = {}
    conflicts: set[str] = set()
    for qid, row in eligible.items():
        name = row.get("name")
        if not isinstance(name, str):
            blockers.append(f"R9 replay candidate {qid} has no string name")
            continue
        candidate_name = _normal_form(name)
        answer_aliases = {
            _normal_form(alias)
            for values in row["aliases"]
            for alias in values
            if isinstance(alias, str)
        }
        candidate_owners.setdefault(candidate_name, set()).add(qid)
        for alias in answer_aliases:
            answer_owners.setdefault(alias, set()).add(qid)
        if candidate_name in answer_aliases:
            conflicts.add(qid)
    for owners in candidate_owners.values():
        if len(owners) > 1:
            conflicts.update(owners)
    for surface in candidate_owners.keys() & answer_owners.keys():
        conflicts.update(candidate_owners[surface])
        conflicts.update(answer_owners[surface])

    safe_by_domain: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    for qid, row in eligible.items():
        aliases = row["aliases"]
        if qid in conflicts:
            decisions[qid]["decision"] = "excluded_surface_conflict"
        elif any(
            _normal_form(alias) in _RESERVED_OUTPUTS
            for values in aliases
            for alias in values
            if isinstance(alias, str)
        ):
            decisions[qid]["decision"] = "excluded_reserved_output"
        elif row["domain"] in safe_by_domain:
            safe_by_domain[row["domain"]].append(qid)
        else:
            blockers.append(f"R9 replay candidate {qid} has an invalid domain")

    selected_by_domain: dict[str, list[str]] = {}
    for domain in DOMAINS:
        safe = sorted(safe_by_domain[domain], key=lambda qid: _selection_key(domain, qid))
        _check(
            len(safe) >= CANDIDATES_PER_DOMAIN,
            f"R9 replay {domain} has fewer than 24 safe candidates",
            blockers,
        )
        selected_by_domain[domain] = safe[:CANDIDATES_PER_DOMAIN]

    selected_splits: dict[str, str] = {}
    for domain in DOMAINS:
        counts = {split: 0 for split in SPLITS}
        ordered = sorted(
            selected_by_domain[domain],
            key=lambda qid: (
                -int(eligible[qid]["corrected"]),
                _selection_key(domain, qid),
            ),
        )
        for qid in ordered:
            split = min(SPLITS, key=lambda value: (counts[value], value))
            selected_splits[qid] = split
            counts[split] += 1
        _check(
            set(counts.values()) == {12},
            f"R9 replay {domain} split assignment is not 12/12",
            blockers,
        )

    for qid, row in eligible.items():
        if qid in selected_splits:
            decisions[qid]["decision"] = "included"
            decisions[qid]["split"] = selected_splits[qid]
        elif decisions[qid]["decision"] == "eligible":
            decisions[qid]["decision"] = "eligible_reserve"
    return decisions


def _verify_r9_source(
    r8_qids: set[str],
    prior_qids: set[str],
    r8_root: Path,
    r9_root: Path,
    derivation: dict[str, Any],
    blockers: list[str],
) -> tuple[int, int]:
    integrity_path = r9_root / "source_integrity_v1.json"
    integrity = _mapping(
        _load_json(integrity_path, "R9 integrity", blockers), "R9 integrity", blockers
    )
    _check(
        integrity.get("schema_version") == 1,
        "R9 integrity schema mismatch",
        blockers,
    )
    _check(
        integrity.get("source_revision") == SOURCE_REVISION,
        "R9 integrity source revision mismatch",
        blockers,
    )
    snapshot_path = _safe_child(
        r9_root, integrity.get("source_snapshot"), "R9 source snapshot", blockers
    )
    snapshot = _mapping(
        _load_json(snapshot_path, "R9 source snapshot", blockers)
        if snapshot_path is not None
        else None,
        "R9 source snapshot",
        blockers,
    )
    if snapshot_path is not None:
        snapshot_sha256 = _file_sha256(
            snapshot_path, "R9 source snapshot", blockers
        )
        _check(
            snapshot_sha256 == integrity.get("source_snapshot_sha256"),
            "R9 source snapshot hash mismatch",
            blockers,
        )
    _check(snapshot.get("schema_version") == 1, "R9 snapshot schema mismatch", blockers)
    _check(
        snapshot.get("source_revision") == SOURCE_REVISION,
        "R9 snapshot source revision mismatch",
        blockers,
    )
    _check(
        snapshot.get("split_seed") == 20260725,
        "R9 snapshot split seed mismatch",
        blockers,
    )
    _check(
        snapshot.get("candidates_per_domain_per_split") == 12,
        "R9 snapshot candidates-per-domain-per-split mismatch",
        blockers,
    )
    _check(
        snapshot.get("splits") == list(SPLITS),
        "R9 snapshot split list mismatch",
        blockers,
    )
    _check(
        snapshot.get("source_frame_sha256") == derivation.get("r8_frame_sha256"),
        "R9 snapshot source frame hash does not match derivation",
        blockers,
    )
    _check(
        integrity.get("materialized_files") == snapshot.get("materialized_files"),
        "R9 integrity and snapshot manifest registries differ",
        blockers,
    )
    manifests = _manifest_files(r9_root, integrity, "R9 integrity", blockers)
    candidate_count = 0
    question_count = 0
    selected_splits: dict[str, str] = {}
    question_ids: set[str] = set()
    for split in SPLITS:
        candidates, questions = manifests[split]
        candidate_count += len(candidates)
        question_count += len(questions)
        qids = _qid_set(candidates, f"R9 {split} candidates", blockers)
        _check(len(candidates) == 48, f"R9 {split} candidate count is not 48", blockers)
        _check(len(questions) == 144, f"R9 {split} question count is not 144", blockers)
        domains = Counter(
            row.get("coarse_type") for row in candidates if isinstance(row, dict)
        )
        _check(
            domains == Counter({domain: 12 for domain in DOMAINS}),
            f"R9 {split} domain balance is not 12 per domain",
            blockers,
        )
        for row in candidates:
            if not isinstance(row, dict):
                blockers.append(f"R9 {split} candidate is malformed")
                continue
            qid = row.get("qid")
            _check(
                row.get("split") == split,
                f"R9 {split} candidate split mismatch",
                blockers,
            )
            if isinstance(qid, str):
                if qid in selected_splits:
                    blockers.append("R9 candidates reuse a QID across splits")
                selected_splits[qid] = split
        questions_by_qid: Counter[str] = Counter()
        for row in questions:
            if not isinstance(row, dict):
                blockers.append(f"R9 {split} question is malformed")
                continue
            question_id = row.get("question_id")
            qid = row.get("qid")
            if not isinstance(question_id, str) or question_id in question_ids:
                blockers.append("R9 questions have missing or duplicate question IDs")
            elif isinstance(question_id, str):
                question_ids.add(question_id)
            if isinstance(qid, str):
                questions_by_qid[qid] += 1
            else:
                blockers.append("R9 question has no string QID")
        _check(
            set(questions_by_qid) == qids and set(questions_by_qid.values()) == {3},
            f"R9 {split} does not have three questions for every QID",
            blockers,
        )
    _check(candidate_count == 96, "R9 candidate count is not 96", blockers)
    _check(question_count == 288, "R9 question count is not 288", blockers)
    _check(len(selected_splits) == 96, "R9 candidate QIDs are not unique", blockers)
    snapshot_audit = _mapping(snapshot.get("audit"), "R9 snapshot audit", blockers)
    _check(
        snapshot_audit.get("candidate_count") == candidate_count
        and snapshot_audit.get("question_count") == question_count,
        "R9 snapshot audit counts mismatch",
        blockers,
    )

    source_v7_path = _safe_child(
        r9_root,
        integrity.get("source_v7_exclusions"),
        "R9 source-v7 exclusions",
        blockers,
    )
    source_v7 = _mapping(
        _load_json(source_v7_path, "R9 source-v7 exclusions", blockers)
        if source_v7_path is not None
        else None,
        "R9 source-v7 exclusions",
        blockers,
    )
    if source_v7_path is not None:
        source_v7_hash = _file_sha256(
            source_v7_path, "R9 source-v7 exclusions", blockers
        )
        _check(
            source_v7_hash == integrity.get("source_v7_exclusions_sha256")
            and source_v7_hash == snapshot.get("source_v7_exclusions_sha256"),
            "R9 source-v7 exclusions hash mismatch",
            blockers,
        )
    _check(
        snapshot.get("source_v7_exclusions") == integrity.get("source_v7_exclusions"),
        "R9 integrity and snapshot exclusion paths differ",
        blockers,
    )
    source_v7_qids = _qid_set(
        [
            {"qid": value}
            for value in _array(
                source_v7.get("excluded_qids"),
                "R9 source-v7 excluded QIDs",
                blockers,
            )
        ],
        "R9 source-v7 exclusions",
        blockers,
    )
    _check(
        source_v7.get("schema_version") == 1
        and source_v7.get("kind") == "source_v7_exclusions"
        and source_v7.get("source_revision") == SOURCE_REVISION,
        "R9 source-v7 exclusions identity mismatch",
        blockers,
    )
    _check(
        source_v7_qids == r8_qids | prior_qids,
        "R9 source-v7 exclusions are not exactly R8 candidates plus prior exclusions",
        blockers,
    )
    if source_v7_path is not None:
        _check(
            derivation.get("source_v7_exclusions_file") == source_v7_path.name
            and derivation.get("source_v7_exclusions_sha256")
            == _sha256_file(source_v7_path),
            "R9 derivation does not bind source-v7 exclusions",
            blockers,
        )

    decisions = _array(derivation.get("decisions"), "R9 derivation decisions", blockers)
    _check(
        derivation.get("decisions_sha256") == _canonical_sha256(decisions),
        "R9 derivation decisions hash mismatch",
        blockers,
    )
    decision_by_qid: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict) or not isinstance(decision.get("qid"), str):
            blockers.append("R9 derivation decision is malformed")
            continue
        qid = decision["qid"]
        if qid in decision_by_qid:
            blockers.append("R9 derivation decisions have duplicate QIDs")
        decision_by_qid[qid] = decision
    _check(len(decisions) == 192, "R9 derivation decision count is not 192", blockers)
    _check(
        set(decision_by_qid) == r8_qids,
        "R9 derivation decisions do not cover exactly the R8 candidates",
        blockers,
    )
    included = {
        qid: decision.get("split")
        for qid, decision in decision_by_qid.items()
        if decision.get("decision") == "included"
    }
    _check(
        len(included) == 96,
        "R9 derivation does not include exactly 96 QIDs",
        blockers,
    )
    _check(
        included == selected_splits,
        "R9 included decision split assignments do not match materialized candidates",
        blockers,
    )
    expected_decisions = _replay_expected_decisions(r8_root, r9_root, blockers)
    _check(
        decision_by_qid == expected_decisions,
        "R9 decisions differ from independent eligibility/selection/split replay",
        blockers,
    )

    confirmatory_path = _safe_child(
        r9_root,
        derivation.get("confirmatory_exclusions_file"),
        "R9 confirmatory exclusions",
        blockers,
    )
    confirmatory_rows = (
        _array(
            _load_json(confirmatory_path, "R9 confirmatory exclusions", blockers),
            "R9 confirmatory exclusions",
            blockers,
        )
        if confirmatory_path is not None
        else []
    )
    if confirmatory_path is not None:
        confirmatory_sha256 = _file_sha256(
            confirmatory_path, "R9 confirmatory exclusions", blockers
        )
        _check(
            confirmatory_sha256 == derivation.get("confirmatory_exclusions_sha256"),
            "R9 confirmatory exclusions hash mismatch",
            blockers,
        )
    confirmatory_qids = _qid_set(
        confirmatory_rows, "R9 confirmatory exclusions", blockers
    )
    _check(
        source_v7_qids == confirmatory_qids,
        "R9 future exclusion artifacts contain different QID sets",
        blockers,
    )
    _check(
        len(confirmatory_rows) == derivation.get("future_excluded_qid_count"),
        "R9 confirmatory exclusion count does not match derivation",
        blockers,
    )
    _check(
        r8_qids.issubset(confirmatory_qids) and prior_qids.issubset(confirmatory_qids),
        "R9 future exclusions omit R8 candidates or prior exclusions",
        blockers,
    )
    return candidate_count, question_count


def audit(r8_root: Path, r9_root: Path) -> dict[str, Any]:
    blockers: list[str] = []
    r8_root = r8_root.resolve()
    r9_root = r9_root.resolve()
    if not r8_root.is_dir():
        blockers.append("R8 root is not a directory")
    if not r9_root.is_dir():
        blockers.append("R9 root is not a directory")

    derivation_path = r9_root / "r9_derivation_manifest_v1.json"
    derivation = _mapping(
        _load_json(derivation_path, "R9 derivation manifest", blockers),
        "R9 derivation manifest",
        blockers,
    )
    try:
        derivation_sha256 = _sha256_file(derivation_path)
    except OSError:
        derivation_sha256 = None
    integrity_path = r9_root / "source_integrity_v1.json"
    try:
        source_integrity_sha256 = _sha256_file(integrity_path)
    except OSError:
        source_integrity_sha256 = None

    _check(
        derivation.get("schema_version") == 1,
        "R9 derivation schema mismatch",
        blockers,
    )
    _check(
        derivation.get("kind") == "fa_source_v6_r9_derivation",
        "R9 derivation kind mismatch",
        blockers,
    )
    _check(
        derivation.get("source_revision") == SOURCE_REVISION,
        "R9 derivation source revision mismatch",
        blockers,
    )
    _check(
        derivation.get("source_integrity_sha256") == source_integrity_sha256,
        "R9 derivation source integrity hash mismatch",
        blockers,
    )

    r8_qids, prior_qids, audit_items_sha256 = _verify_r8_inputs(
        r8_root, derivation, blockers
    )
    _verify_corrections(r8_root, r9_root, derivation, audit_items_sha256, blockers)
    candidate_count, question_count = _verify_r9_source(
        r8_qids, prior_qids, r8_root, r9_root, derivation, blockers
    )
    blockers = sorted(set(blockers))
    return {
        "schema_version": 1,
        "kind": "fa_source_v6_r9_structural_provenance_audit",
        "source_revision": SOURCE_REVISION,
        "source_integrity_sha256": source_integrity_sha256,
        "derivation_sha256": derivation_sha256,
        "status": "passed" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "candidate_count": candidate_count,
        "question_count": question_count,
        "decision_count": len(
            _array(derivation.get("decisions"), "R9 derivation decisions", [])
        ),
        "independent_code_path": True,
        "auditor_id": AUDITOR_ID,
        "audit_code_sha256": _sha256_file(Path(__file__).resolve()),
    }


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise RuntimeError(
            f"refusing to overwrite immutable audit output: {path}"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r8-root", type=Path, required=True)
    parser.add_argument("--r9-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = audit(args.r8_root, args.r9_root)
        _write_immutable(args.output, payload)
    except (OSError, RuntimeError) as error:
        print(f"audit failed before output: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
