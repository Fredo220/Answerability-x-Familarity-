"""Audit-qualified R9 development corpus derived from frozen R8 evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_confirmatory_source import REGISTERED_DOMAINS, SourceRecord
from trajectory_extractor.fa_confirmatory_synthetics import (
    generate_synthetic_candidates,
)
from trajectory_extractor.fa_development_source import (
    DEVELOPMENT_SPLITS,
    DevelopmentSourceDesign,
    _normal_form,
    _surface_conflict_qids,
    materialize_development_manifests,
    write_development_source,
)
from trajectory_extractor.fa_entities import CandidateEntity

R8_CONSTRUCTION_COMMIT = "0aa6f0ef0798073a0a464b3639479d41083d77ac"
R8_FRAME_SHA256 = "2b790d4607a52cc0bd5ebe9d40d577bb11c833a1366e5c31985ae4538a4d5450"
R8_INTEGRITY_SHA256 = "71510bce1a103351226eaf6450f2226336bf1e99534a9a1f025b2c076a22ff5b"
R8_AUDIT_SHA256 = "e61d1ce59e13cc25f6b2b2b51764c5d05ae864321228eb6e0d65022087d8db22"
R8_AUDIT_ITEMS_SHA256 = (
    "fd1cf8f6099c3f224d2a8f2e124f00594703f9c56040cb95389f44a36f9325c8"
)
R9_CORRECTIONS_SHA256 = (
    "20feb70c4042301aa733472faf5b5863f5e6fe7f2098a01f3b77342ee9a47409"
)
R9_CORRECTION_ITEMS_SHA256 = (
    "29323754adfed2487dd47e8d1efe8fd5f054a604eeeffb5968b60ac9169fe891"
)
R9_AMENDMENT_PATH = Path(
    "docs/amendments/2026-07-25-fa-source-v6-r9-audit-qualified-development.md"
)
R9_SOURCE_REVISION = "fa-development-source-v6-r9"
R9_SELECTION_SEED = 20260725
R9_CANDIDATES_PER_DOMAIN_PER_SPLIT = 12
_RESERVED_OUTPUTS = frozenset({"unknown"})


def derive_r9_source(
    *,
    r8_root: Path,
    corrections_path: Path,
    output_dir: Path,
    tokenizer: Any,
    config: FAConfig,
) -> dict[str, Any]:
    """Derive and write the frozen R9 development source."""
    if config.profile != "confirmatory":
        raise ValueError("R9 derivation requires the pinned confirmatory config")
    construction_commit = _current_clean_commit()
    amendment_path = R9_AMENDMENT_PATH
    if not amendment_path.is_file():
        raise ValueError("R9 registered amendment is missing")
    inputs = _load_and_verify_inputs(r8_root, corrections_path)
    records_by_domain, decisions, correction_counts = _qualified_records(inputs)
    conflicts = _surface_conflict_qids(records_by_domain)
    safe_records = {}
    for domain in REGISTERED_DOMAINS:
        accepted = []
        for record in records_by_domain[domain]:
            if record.qid in conflicts:
                decisions[record.qid]["decision"] = "excluded_surface_conflict"
            elif _record_has_reserved_output(record):
                decisions[record.qid]["decision"] = "excluded_reserved_output"
            else:
                accepted.append(record)
        safe_records[domain] = tuple(accepted)
    for domain, records in safe_records.items():
        if len(records) < R9_CANDIDATES_PER_DOMAIN_PER_SPLIT * 2:
            raise ValueError(
                f"R9 {domain} has {len(records)} safe records but requires 24"
            )

    selected = {
        domain: tuple(
            sorted(records, key=lambda row: _selection_key(domain, row.qid))[:24]
        )
        for domain, records in safe_records.items()
    }
    assigned = _balanced_assignment(selected, correction_counts)
    _verify_pseudonym_feasibility(assigned, tokenizer)

    design = DevelopmentSourceDesign(
        revision=R9_SOURCE_REVISION,
        split_seed=R9_SELECTION_SEED,
        candidates_per_domain_per_split=R9_CANDIDATES_PER_DOMAIN_PER_SPLIT,
    )
    manifests = materialize_development_manifests(
        assigned,
        design=design,
        retrieval_date=inputs["frame"]["retrieval_date"],
        query_hashes=inputs["frame"]["query_sha256s"],
    )
    inspected_qids = {row["qid"] for row in inputs["candidates"]}
    future_excluded_qids = (
        set(inputs["frame"]["excluded_prior_qids"]) | inspected_qids
    )
    snapshot = write_development_source(
        output_dir,
        manifests,
        design=design,
        excluded_qids=set(inputs["frame"]["excluded_prior_qids"]),
        future_excluded_qids=future_excluded_qids,
        source_frame_sha256=R8_FRAME_SHA256,
    )
    confirmatory_exclusions_path = (
        output_dir / "confirmatory_excluded_candidates_v1.json"
    )
    _write_json(
        confirmatory_exclusions_path,
        [{"qid": qid} for qid in sorted(future_excluded_qids)],
    )

    selected_splits = {
        record.qid: split
        for split, records in assigned.items()
        for record in records
    }
    for qid, split in selected_splits.items():
        decisions[qid]["decision"] = "included"
        decisions[qid]["split"] = split
    for domain, records in safe_records.items():
        for record in records:
            if record.qid not in selected_splits:
                decisions[record.qid]["decision"] = "eligible_reserve"

    integrity_path = output_dir / "source_integrity_v1.json"
    decision_rows = [decisions[qid] for qid in sorted(decisions)]
    source_v7_exclusions_path = snapshot["source_v7_exclusions"]
    derivation = {
        "schema_version": 1,
        "kind": "fa_source_v6_r9_derivation",
        "source_revision": R9_SOURCE_REVISION,
        "claim_scope": "instrument_development_only",
        "r9_construction_commit": construction_commit,
        "r8_construction_commit": R8_CONSTRUCTION_COMMIT,
        "r8_frame_sha256": R8_FRAME_SHA256,
        "r8_integrity_sha256": R8_INTEGRITY_SHA256,
        "r8_audit_sha256": R8_AUDIT_SHA256,
        "r8_audit_items_sha256": R8_AUDIT_ITEMS_SHA256,
        "corrections_sha256": R9_CORRECTIONS_SHA256,
        "correction_items_sha256": R9_CORRECTION_ITEMS_SHA256,
        "amendment_sha256": _sha256_file(amendment_path),
        "implementation_sha256s": _implementation_sha256s(),
        "config_sha256": config.config_hash,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "chat_template_sha256": config.chat_template_sha256,
        "selection_seed": R9_SELECTION_SEED,
        "selection_formula": 'SHA256(seed + ":" + domain + ":" + qid), qid',
        "source_integrity_sha256": _sha256_file(integrity_path),
        "source_v7_exclusions_file": source_v7_exclusions_path.name,
        "source_v7_exclusions_sha256": _sha256_file(
            source_v7_exclusions_path
        ),
        "future_excluded_qid_count": len(future_excluded_qids),
        "confirmatory_exclusions_file": confirmatory_exclusions_path.name,
        "confirmatory_exclusions_sha256": _sha256_file(
            confirmatory_exclusions_path
        ),
        "decision_count": len(decisions),
        "decisions_sha256": _canonical_sha256(decision_rows),
        "decisions": decision_rows,
    }
    derivation_path = output_dir / "r9_derivation_manifest_v1.json"
    _write_json(derivation_path, derivation)
    return {
        "status": "materialized",
        "candidate_count": snapshot["audit"]["candidate_count"],
        "question_count": snapshot["audit"]["question_count"],
        "source_integrity_sha256": _sha256_file(integrity_path),
        "derivation_manifest": str(derivation_path),
        "derivation_manifest_sha256": _sha256_file(derivation_path),
        "confirmatory_exclusions_sha256": _sha256_file(
            confirmatory_exclusions_path
        ),
        "counts_by_domain": {
            domain: sum(
                record.domain == domain
                for split in DEVELOPMENT_SPLITS
                for record in assigned[split]
            )
            for domain in REGISTERED_DOMAINS
        },
    }


def _load_and_verify_inputs(r8_root: Path, corrections_path: Path) -> dict[str, Any]:
    paths = {
        "frame": r8_root / "frame/development_source_frame_v1.json",
        "integrity": r8_root / "source_integrity_v1.json",
        "audit": r8_root / "pre_model_semantic_audit_v1.json",
        "audit_items": r8_root / "pre_model_semantic_audit_items_v1.jsonl",
    }
    expected_hashes = {
        "frame": R8_FRAME_SHA256,
        "integrity": R8_INTEGRITY_SHA256,
        "audit": R8_AUDIT_SHA256,
        "audit_items": R8_AUDIT_ITEMS_SHA256,
    }
    for key, path in paths.items():
        if _sha256_file(path) != expected_hashes[key]:
            raise ValueError(f"R9 frozen {key} hash mismatch")
    if _sha256_file(corrections_path) != R9_CORRECTIONS_SHA256:
        raise ValueError("R9 correction manifest hash mismatch")

    integrity = _read_json(paths["integrity"])
    frame = _read_json(paths["frame"])
    audit = _read_json(paths["audit"])
    audit_rows = _read_jsonl(paths["audit_items"])
    corrections = _read_json(corrections_path)
    if (
        audit.get("status") != "blocked"
        or audit.get("items_sha256") != R8_AUDIT_ITEMS_SHA256
        or corrections.get("source_audit_items_sha256")
        != R8_AUDIT_ITEMS_SHA256
        or corrections.get("items_sha256") != R9_CORRECTION_ITEMS_SHA256
        or _canonical_sha256(corrections.get("items")) != R9_CORRECTION_ITEMS_SHA256
    ):
        raise ValueError("R9 audit or correction identity mismatch")

    candidates = []
    questions = []
    for split in DEVELOPMENT_SPLITS:
        files = integrity["materialized_files"][split]
        candidate_path = r8_root / files["candidate_manifest"]
        question_path = r8_root / files["question_manifest"]
        if (
            _sha256_file(candidate_path) != files["candidate_sha256"]
            or _sha256_file(question_path) != files["question_sha256"]
        ):
            raise ValueError("R9 input manifest hash mismatch")
        candidates.extend(_read_json(candidate_path))
        questions.extend(_read_json(question_path))
    if len(candidates) != 192 or len(questions) != 576 or len(audit_rows) != 576:
        raise ValueError("R9 requires complete R8 audit coverage")
    question_identity = {row["question_id"]: row["qid"] for row in questions}
    audit_identity = {row["question_id"]: row["qid"] for row in audit_rows}
    if audit_identity != question_identity:
        raise ValueError("R9 audit question coverage mismatch")
    return {
        "frame": frame,
        "candidates": candidates,
        "questions": questions,
        "audit_rows": audit_rows,
        "corrections": corrections,
    }


def _qualified_records(
    inputs: dict[str, Any],
) -> tuple[
    dict[str, tuple[SourceRecord, ...]],
    dict[str, dict[str, Any]],
    dict[str, int],
]:
    audit_by_question = {row["question_id"]: row for row in inputs["audit_rows"]}
    corrections = inputs["corrections"]["items"]
    correction_by_question = {row["question_id"]: row for row in corrections}
    ordinary_questions = {
        row["question_id"]
        for row in inputs["audit_rows"]
        if "ordinary_surface_missing" in row["blocker_ids"]
    }
    if (
        len(correction_by_question) != len(corrections)
        or set(correction_by_question) != ordinary_questions
    ):
        raise ValueError("R9 correction coverage mismatch")
    for question_id, correction in correction_by_question.items():
        if (
            correction.get("qid") != audit_by_question[question_id]["qid"]
            or correction.get("blocker_id") != "ordinary_surface_missing"
            or not _valid_correction_surfaces(
                correction.get("accepted_surfaces_to_add")
            )
        ):
            raise ValueError("R9 correction item is invalid")

    rows_by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in inputs["audit_rows"]:
        rows_by_qid.setdefault(row["qid"], []).append(row)
    frame_records = {
        row["qid"]: _source_record(row)
        for domain in REGISTERED_DOMAINS
        for row in inputs["frame"]["records_by_domain"][domain]
    }
    candidates = {row["qid"]: row for row in inputs["candidates"]}
    if set(frame_records) != set(candidates) or set(rows_by_qid) != set(candidates):
        raise ValueError("R9 candidate identity mismatch")

    records_by_domain: dict[str, list[SourceRecord]] = {
        domain: [] for domain in REGISTERED_DOMAINS
    }
    decisions = {}
    correction_counts = {}
    for qid, candidate in candidates.items():
        rows = rows_by_qid[qid]
        hard_blockers = sorted(
            {
                blocker
                for row in rows
                for blocker in row["blocker_ids"]
                if blocker != "ordinary_surface_missing"
            }
        )
        decisions[qid] = {
            "qid": qid,
            "domain": candidate["coarse_type"],
            "hard_blockers": hard_blockers,
            "correction_question_ids": sorted(
                row["question_id"]
                for row in rows
                if row["question_id"] in correction_by_question
            ),
            "decision": "excluded_hard_blocker" if hard_blockers else "eligible",
        }
        if hard_blockers:
            continue
        record = frame_records[qid]
        values = [(field, list(aliases)) for field, aliases in record.property_values]
        for row in rows:
            correction = correction_by_question.get(row["question_id"])
            if correction is None:
                continue
            index = int(row["question_id"].rsplit("-q", 1)[1]) - 1
            values[index][1].extend(correction["accepted_surfaces_to_add"])
        property_values = tuple(
            (field, tuple(dict.fromkeys(aliases))) for field, aliases in values
        )
        corrected = SourceRecord(
            qid=record.qid,
            label=record.label,
            domain=record.domain,
            sitelinks=record.sitelinks,
            source_rank=record.source_rank,
            property_values=property_values,
        )
        records_by_domain[record.domain].append(corrected)
        correction_counts[qid] = len(decisions[qid]["correction_question_ids"])
    return (
        {domain: tuple(rows) for domain, rows in records_by_domain.items()},
        decisions,
        correction_counts,
    )


def _balanced_assignment(
    selected: dict[str, tuple[SourceRecord, ...]],
    correction_counts: dict[str, int],
) -> dict[str, tuple[SourceRecord, ...]]:
    assigned: dict[str, list[SourceRecord]] = {
        split: [] for split in DEVELOPMENT_SPLITS
    }
    for domain in REGISTERED_DOMAINS:
        counts = {split: 0 for split in DEVELOPMENT_SPLITS}
        ordered = sorted(
            selected[domain],
            key=lambda row: (
                -int(correction_counts[row.qid] > 0),
                _selection_key(domain, row.qid),
            ),
        )
        for record in ordered:
            split = min(DEVELOPMENT_SPLITS, key=lambda value: (counts[value], value))
            assigned[split].append(record)
            counts[split] += 1
        if set(counts.values()) != {R9_CANDIDATES_PER_DOMAIN_PER_SPLIT}:
            raise ValueError("R9 split balance failed")
    return {
        split: tuple(sorted(rows, key=lambda row: (row.domain, row.qid)))
        for split, rows in assigned.items()
    }


def _verify_pseudonym_feasibility(
    assigned: dict[str, tuple[SourceRecord, ...]],
    tokenizer: Any,
) -> None:
    candidates = []
    for split, records in assigned.items():
        for record in records:
            candidates.append(
                CandidateEntity(
                    entity_id=f"development-v6-{split}-{record.domain}-{record.qid.lower()}",
                    qid=record.qid,
                    name=record.label,
                    coarse_type=record.domain,
                    split=split,
                    source_query=f"https://www.wikidata.org/wiki/{record.qid}",
                    source_provenance="R9 pre-materialization feasibility",
                    screening_aliases=tuple(
                        aliases for _, aliases in record.property_values
                    ),
                )
            )
    synthetic = generate_synthetic_candidates(candidates, tokenizer)
    if len(synthetic) != len(candidates) * 3:
        raise ValueError("R9 pseudonym feasibility failed")


def _source_record(row: dict[str, Any]) -> SourceRecord:
    return SourceRecord(
        qid=row["qid"],
        label=row["label"],
        domain=row["domain"],
        sitelinks=row["sitelinks"],
        source_rank=row["source_rank"],
        property_values=tuple(
            (property_id, tuple(aliases))
            for property_id, aliases in row["property_values"]
        ),
    )


def _record_has_reserved_output(record: SourceRecord) -> bool:
    return any(
        _normal_form(alias) in _RESERVED_OUTPUTS
        for _, aliases in record.property_values
        for alias in aliases
    )


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


def _selection_key(domain: str, qid: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{R9_SELECTION_SEED}:{domain}:{qid}".encode()
    ).hexdigest()
    return digest, qid


def _implementation_sha256s() -> dict[str, str]:
    module_dir = Path(__file__).parent
    paths = {
        "fa_development_r9.py": Path(__file__),
        "fa_development_source.py": module_dir / "fa_development_source.py",
        "fa_confirmatory_synthetics.py": (
            module_dir / "fa_confirmatory_synthetics.py"
        ),
        "fa_entities.py": module_dir / "fa_entities.py",
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _current_clean_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot verify the R9 construction checkout") from error
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("R9 construction commit is invalid")
    if dirty:
        raise ValueError("R9 derivation requires a clean git checkout")
    return commit


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read R9 input: {path}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read R9 input: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"R9 JSONL rows must be objects: {path}")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"immutable R9 artifact differs: {path}")
    path.write_text(payload, encoding="utf-8")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"cannot hash R9 input: {path}") from error
