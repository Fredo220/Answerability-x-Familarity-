"""Blinded human-rating packets for the Familiarity-vs-Answerability study."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from trajectory_extractor.fa_entities import (
    EntityMatch,
    NaturalnessRating,
    naturalness_rating_passes,
)


SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_RATING_FIELDS = (
    "packet_id",
    "rater_id",
    "item_id",
    "candidate_a_naturalness",
    "candidate_a_type_fit",
    "candidate_a_malformed",
    "candidate_b_naturalness",
    "candidate_b_type_fit",
    "candidate_b_malformed",
    "independence_attested",
)
_STIMULUS_FIELDS = (
    "coarse_type",
    "candidate_a",
    "candidate_b",
    "sentence_a",
    "sentence_b",
)
_RATING_QUESTION = (
    "Rate candidates A and B: naturalness 1-5, type fit 1-5, "
    "and malformed true/false."
)
_WORKSHEET_FIELDS = (
    "packet_id",
    "rater_id",
    "item_id",
    "rating_question",
    "coarse_type",
    "candidate_a",
    "sentence_a",
    "candidate_a_naturalness",
    "candidate_a_type_fit",
    "candidate_a_malformed",
    "candidate_b",
    "sentence_b",
    "candidate_b_naturalness",
    "candidate_b_type_fit",
    "candidate_b_malformed",
    "independence_attested",
)
_PRIVATE_ITEM_FIELDS = {
    "packet_id",
    "item_id",
    "pair_id",
    "rater_id",
    "blind_slot",
    "candidate_a_role",
    "candidate_b_role",
    "packet_item_sha256",
}
_PUBLIC_PACKET_FIELDS = {
    "schema_version",
    "purpose",
    "packet_id",
    "rater_id",
    "rating_protocol_sha256",
    "instructions",
    "scale",
    "items",
}
_PUBLIC_ITEM_FIELDS = {
    "item_id",
    "coarse_type",
    "candidate_a",
    "candidate_b",
    "sentence_a",
    "sentence_b",
}


def prepare_initial_rating_packets(
    matches: Sequence[EntityMatch],
    *,
    config_sha256: str,
    protocol_sha256: str,
    rating_protocol_sha256: str,
    output_dir: str | Path,
    rater_ids: Sequence[str],
) -> dict[str, Any]:
    """Write two deterministic packets and a separate private unblinding key."""

    raters = tuple(rater_ids)
    if len(raters) != 2 or len(set(raters)) != 2:
        raise ValueError("initial naturalness review requires two distinct raters")
    return _prepare_packets(
        matches,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        rating_protocol_sha256=rating_protocol_sha256,
        output_dir=output_dir,
        rater_slots=((raters[0], "slot-a"), (raters[1], "slot-b")),
        purpose="initial",
    )


def compile_initial_responses(
    matches: Sequence[EntityMatch],
    *,
    private_key_path: str | Path,
    response_paths: Sequence[str | Path],
    config_sha256: str,
    protocol_sha256: str,
    rating_protocol_sha256: str,
) -> tuple[
    tuple[NaturalnessRating, ...],
    tuple[dict[str, str], ...],
    tuple[str, ...],
    tuple[dict[str, str], ...],
]:
    """Compile two filled blinded packets and identify adjudication cases."""

    key = _load_key(
        private_key_path,
        matches,
        expected_purpose="initial",
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        rating_protocol_sha256=rating_protocol_sha256,
    )
    expected_raters = tuple(row["rater_id"] for row in key["raters"])
    if len(tuple(response_paths)) != 2:
        raise ValueError("exactly two initial response files are required")
    responses = _load_responses(response_paths)
    if set(responses) != set(expected_raters):
        raise ValueError("response rater IDs do not match the private key")

    ratings, assignments = _responses_to_ratings(key, responses, round_number=1)
    grouped: dict[str, list[NaturalnessRating]] = {}
    for rating in ratings:
        grouped.setdefault(rating.pair_id, []).append(rating)
    disagreements = tuple(
        sorted(
            pair_id
            for pair_id, pair_ratings in grouped.items()
            if naturalness_rating_passes(pair_ratings[0])
            != naturalness_rating_passes(pair_ratings[1])
        )
    )
    return ratings, assignments, disagreements, _response_evidence(responses)


def packet_issuance_record(
    private_key_path: str | Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Capture issued packets and their private mapping as one immutable record."""

    key_path = Path(private_key_path)
    try:
        key = json.loads(key_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("private unblinding key is unreadable") from error
    packets = _packet_documents_from_files(_packet_root(key_path), key)
    row = {
        "kind": "naturalness_packet_issuance",
        "schema_version": SCHEMA_VERSION,
        "purpose": key["purpose"],
        "config_sha256": key["config_sha256"],
        "protocol_sha256": key["protocol_sha256"],
        "rating_protocol_sha256": key["rating_protocol_sha256"],
        "matches_sha256": key["matches_sha256"],
        "private_key": key,
        "packets": packets,
    }
    lineage = {
        "config_sha256": key["config_sha256"],
        "protocol_sha256": key["protocol_sha256"],
        "rating_protocol_sha256": key["rating_protocol_sha256"],
        "matches_sha256": key["matches_sha256"],
        "issuance_sha256": _sha256_json(row),
    }
    return row, lineage


def compile_initial_responses_from_issuance(
    matches: Sequence[EntityMatch],
    *,
    issuance: Mapping[str, Any],
    response_paths: Sequence[str | Path],
    config_sha256: str,
    protocol_sha256: str,
    rating_protocol_sha256: str,
) -> tuple[
    tuple[NaturalnessRating, ...],
    tuple[dict[str, str], ...],
    tuple[str, ...],
    tuple[dict[str, str], ...],
]:
    """Compile initial responses from a verified immutable packet issuance."""

    key = _validate_issuance(
        issuance,
        matches,
        expected_purpose="initial",
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        rating_protocol_sha256=rating_protocol_sha256,
    )
    if len(tuple(response_paths)) != 2:
        raise ValueError("exactly two initial response files are required")
    responses = _load_responses(response_paths)
    expected_raters = {row["rater_id"] for row in key["raters"]}
    if set(responses) != expected_raters:
        raise ValueError("response rater IDs do not match the packet issuance")
    ratings, assignments = _responses_to_ratings(key, responses, round_number=1)
    grouped: dict[str, list[NaturalnessRating]] = {}
    for rating in ratings:
        grouped.setdefault(rating.pair_id, []).append(rating)
    disagreements = tuple(
        sorted(
            pair_id
            for pair_id, pair_ratings in grouped.items()
            if naturalness_rating_passes(pair_ratings[0])
            != naturalness_rating_passes(pair_ratings[1])
        )
    )
    return ratings, assignments, disagreements, _response_evidence(responses)


def prepare_adjudication_packet(
    matches: Sequence[EntityMatch],
    *,
    pair_ids: Iterable[str],
    config_sha256: str,
    protocol_sha256: str,
    rating_protocol_sha256: str,
    output_dir: str | Path,
    adjudicator_id: str,
) -> dict[str, Any]:
    """Write one blinded packet containing only registered disagreements."""

    selected_ids = frozenset(pair_ids)
    selected = tuple(match for match in matches if match.pair_id in selected_ids)
    if not selected_ids or {match.pair_id for match in selected} != selected_ids:
        raise ValueError("adjudication requires known registered disagreement pairs")
    return _prepare_packets(
        selected,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        rating_protocol_sha256=rating_protocol_sha256,
        output_dir=output_dir,
        rater_slots=((adjudicator_id, "adjudicator"),),
        purpose="adjudication",
    )


def compile_adjudication_response(
    matches: Sequence[EntityMatch],
    *,
    private_key_path: str | Path,
    response_path: str | Path,
    expected_pair_ids: Iterable[str],
    config_sha256: str,
    protocol_sha256: str,
    rating_protocol_sha256: str,
) -> tuple[
    tuple[NaturalnessRating, ...],
    tuple[dict[str, str], ...],
    tuple[dict[str, str], ...],
]:
    """Compile a third-rater packet for exactly the registered disagreements."""

    expected = frozenset(expected_pair_ids)
    key = _load_key(
        private_key_path,
        matches,
        expected_purpose="adjudication",
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        rating_protocol_sha256=rating_protocol_sha256,
    )
    if frozenset(row["pair_id"] for row in key["items"]) != expected:
        raise ValueError("adjudication key does not match registered disagreements")
    responses = _load_responses((response_path,))
    expected_rater = key["raters"][0]["rater_id"]
    if set(responses) != {expected_rater}:
        raise ValueError("adjudication response rater does not match private key")
    ratings, assignments = _responses_to_ratings(key, responses, round_number=2)
    return ratings, assignments, _response_evidence(responses)


def compile_adjudication_response_from_issuance(
    matches: Sequence[EntityMatch],
    *,
    issuance: Mapping[str, Any],
    response_path: str | Path,
    expected_pair_ids: Iterable[str],
    config_sha256: str,
    protocol_sha256: str,
    rating_protocol_sha256: str,
) -> tuple[
    tuple[NaturalnessRating, ...],
    tuple[dict[str, str], ...],
    tuple[dict[str, str], ...],
]:
    """Compile a third-rater response from an immutable issuance."""

    expected = frozenset(expected_pair_ids)
    key = _validate_issuance(
        issuance,
        matches,
        expected_purpose="adjudication",
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        rating_protocol_sha256=rating_protocol_sha256,
    )
    if frozenset(row["pair_id"] for row in key["items"]) != expected:
        raise ValueError("adjudication issuance does not match registered disagreements")
    responses = _load_responses((response_path,))
    expected_rater = key["raters"][0]["rater_id"]
    if set(responses) != {expected_rater}:
        raise ValueError("adjudication response rater does not match issuance")
    ratings, assignments = _responses_to_ratings(key, responses, round_number=2)
    return ratings, assignments, _response_evidence(responses)


def submission_record(
    ratings: Sequence[NaturalnessRating],
    assignments: Sequence[Mapping[str, str]],
    responses: Sequence[Mapping[str, str]],
    *,
    config_sha256: str,
    issuance_manifest: str,
    issuance_sha256: str,
    disagreement_pair_ids: Iterable[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Preserve one completed human-submission round as immutable evidence."""

    row = {
        "kind": "naturalness_submission",
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config_sha256,
        "issuance_manifest": issuance_manifest,
        "issuance_sha256": issuance_sha256,
        "ratings": [
            asdict(value)
            for value in sorted(
                ratings, key=lambda value: (value.pair_id, value.round, value.rater_id)
            )
        ],
        "assignments": sorted(
            (dict(value) for value in assignments),
            key=lambda value: (
                value["pair_id"],
                value["blind_slot"],
                value["rater_id"],
            ),
        ),
        "responses": sorted(
            (dict(value) for value in responses),
            key=lambda value: (value["rater_id"], value["item_id"]),
        ),
        "disagreement_pair_ids": sorted(frozenset(disagreement_pair_ids)),
    }
    lineage = {
        "config_sha256": config_sha256,
        "issuance_sha256": issuance_sha256,
        "submission_sha256": _sha256_json(row),
    }
    return row, lineage


def verify_submission_record(
    issuance: Mapping[str, Any], submission: Mapping[str, Any]
) -> None:
    """Recompute ratings and assignments from preserved packet and response rows."""

    purpose = issuance.get("purpose")
    if purpose not in {"initial", "adjudication"}:
        raise ValueError("naturalness submission issuance purpose is invalid")
    key = issuance.get("private_key")
    if not isinstance(key, dict):
        raise ValueError("naturalness submission issuance key is invalid")
    _verify_packet_documents(key, issuance.get("packets"))
    raw_responses = submission.get("responses")
    if not isinstance(raw_responses, list):
        raise ValueError("naturalness submission responses are invalid")
    responses: dict[str, dict[str, dict[str, str]]] = {}
    for row in raw_responses:
        if not isinstance(row, dict) or set(row) != set(_RATING_FIELDS):
            raise ValueError("naturalness submission response row is invalid")
        rater_id = row.get("rater_id")
        item_id = row.get("item_id")
        if (
            not isinstance(rater_id, str)
            or not isinstance(item_id, str)
            or item_id in responses.setdefault(rater_id, {})
        ):
            raise ValueError("naturalness submission response identities are invalid")
        responses[rater_id][item_id] = row
    expected_raters = {row["rater_id"] for row in key["raters"]}
    if set(responses) != expected_raters:
        raise ValueError(
            "naturalness submission response raters do not match the issuance"
        )
    round_number = 1 if purpose == "initial" else 2
    ratings, assignments = _responses_to_ratings(
        key, responses, round_number=round_number
    )
    expected_ratings = [
        asdict(value)
        for value in sorted(
            ratings, key=lambda value: (value.pair_id, value.round, value.rater_id)
        )
    ]
    expected_assignments = sorted(
        (dict(value) for value in assignments),
        key=lambda value: (
            value["pair_id"],
            value["blind_slot"],
            value["rater_id"],
        ),
    )
    if (
        submission.get("ratings") != expected_ratings
        or submission.get("assignments") != expected_assignments
    ):
        raise ValueError(
            "naturalness submission does not match preserved human responses"
        )
    disagreements = []
    if purpose == "initial":
        grouped: dict[str, list[NaturalnessRating]] = {}
        for rating in ratings:
            grouped.setdefault(rating.pair_id, []).append(rating)
        disagreements = sorted(
            pair_id
            for pair_id, pair_ratings in grouped.items()
            if naturalness_rating_passes(pair_ratings[0])
            != naturalness_rating_passes(pair_ratings[1])
        )
    if submission.get("disagreement_pair_ids") != disagreements:
        raise ValueError(
            "naturalness submission disagreement registration does not verify"
        )


def rating_record(
    ratings: Sequence[NaturalnessRating],
    assignments: Sequence[Mapping[str, str]],
    *,
    config_sha256: str,
    protocol_sha256: str,
    additional_lineage: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build the exact immutable record and required lineage."""

    ordered_assignments = sorted(
        (dict(value) for value in assignments),
        key=lambda value: (value["pair_id"], value["blind_slot"], value["rater_id"]),
    )
    blinding_sha256 = _sha256_json(ordered_assignments)
    row = {
        "kind": "naturalness_ratings",
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "blinding_manifest_sha256": blinding_sha256,
        "assignments": ordered_assignments,
        "ratings": [
            asdict(value)
            for value in sorted(
                ratings, key=lambda value: (value.pair_id, value.round, value.rater_id)
            )
        ],
    }
    extra = dict(additional_lineage or {})
    protected = {
        "config_sha256",
        "protocol_sha256",
        "blinding_manifest_sha256",
    }
    if protected & set(extra):
        raise ValueError("additional lineage cannot override registered hashes")
    lineage = {
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "blinding_manifest_sha256": blinding_sha256,
        **extra,
    }
    return row, lineage


def _prepare_packets(
    matches: Sequence[EntityMatch],
    *,
    config_sha256: str,
    protocol_sha256: str,
    rating_protocol_sha256: str,
    output_dir: str | Path,
    rater_slots: Sequence[tuple[str, str]],
    purpose: str,
) -> dict[str, Any]:
    match_rows = tuple(sorted(matches, key=lambda value: value.pair_id))
    if not match_rows or any(not isinstance(value, EntityMatch) for value in match_rows):
        raise ValueError("rating packets require EntityMatch records")
    if len({value.pair_id for value in match_rows}) != len(match_rows):
        raise ValueError("rating packets require unique pair IDs")
    for rater_id, slot in rater_slots:
        _safe_id(rater_id, "rater_id")
        if slot not in {"slot-a", "slot-b", "adjudicator"}:
            raise ValueError("invalid blind slot")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    public_dir = target / "public"
    private_dir = target / "private"
    public_dir.mkdir()
    private_dir.mkdir()
    matches_sha256 = _matches_sha256(match_rows)
    private_items: list[dict[str, str]] = []
    rater_records: list[dict[str, str]] = []

    for rater_id, blind_slot in rater_slots:
        packet_id = "packet-" + _sha256_text(
            f"{config_sha256}|{purpose}|{rater_id}|{matches_sha256}"
        )[:16]
        entries = []
        for match in match_rows:
            item_id = "item-" + _sha256_text(
                f"{config_sha256}|{purpose}|{rater_id}|{match.pair_id}"
            )[:16]
            orientation = int(
                _sha256_text(f"{protocol_sha256}|{match.pair_id}"), 16
            ) % 2 == 0
            real_first = (
                orientation
                if purpose == "adjudication" or blind_slot == "slot-a"
                else not orientation
            )
            if real_first:
                name_a, role_a = match.real_name, "real"
                name_b, role_b = match.synthetic_name, "synthetic"
            else:
                name_a, role_a = match.synthetic_name, "synthetic"
                name_b, role_b = match.real_name, "real"
            packet_item = {
                "item_id": item_id,
                "coarse_type": match.coarse_type,
                "candidate_a": name_a,
                "candidate_b": name_b,
                "sentence_a": _neutral_sentence(match.coarse_type, name_a),
                "sentence_b": _neutral_sentence(match.coarse_type, name_b),
            }
            entries.append(packet_item)
            private_items.append(
                {
                    "packet_id": packet_id,
                    "item_id": item_id,
                    "pair_id": match.pair_id,
                    "rater_id": rater_id,
                    "blind_slot": blind_slot,
                    "candidate_a_role": role_a,
                    "candidate_b_role": role_b,
                    "packet_item_sha256": _sha256_json(packet_item),
                }
            )

        entries.sort(key=lambda value: _sha256_text(f"{rater_id}|{value['item_id']}"))
        packet = {
            "schema_version": SCHEMA_VERSION,
            "purpose": purpose,
            "packet_id": packet_id,
            "rater_id": rater_id,
            "rating_protocol_sha256": rating_protocol_sha256,
            "instructions": (
                "Rate each candidate independently. Do not discuss ratings with "
                "another rater and do not use a language model or web search."
            ),
            "scale": {
                "naturalness": "1=clearly malformed, 3=plausible, 5=fully natural",
                "type_fit": "1=does not fit the stated type, 3=plausible, 5=strong fit",
                "malformed": "true only for an orthographic or linguistic defect",
            },
            "items": entries,
        }
        packet_path = public_dir / f"{rater_id}-packet.json"
        _write_json(packet_path, packet)
        response_path = public_dir / f"{rater_id}-response.csv"
        _write_response_template(response_path, packet_id, rater_id, entries)
        rater_records.append(
            {
                "rater_id": rater_id,
                "blind_slot": blind_slot,
                "packet_file": str(packet_path.relative_to(target)),
                "packet_sha256": _sha256_bytes(packet_path.read_bytes()),
                "response_file": str(response_path.relative_to(target)),
            }
        )

    key = {
        "schema_version": SCHEMA_VERSION,
        "purpose": purpose,
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "rating_protocol_sha256": rating_protocol_sha256,
        "matches_sha256": matches_sha256,
        "raters": sorted(rater_records, key=lambda value: value["blind_slot"]),
        "items": sorted(private_items, key=lambda value: value["item_id"]),
    }
    key_path = private_dir / "unblinding-key.json"
    _write_json(key_path, key)
    key_path.chmod(0o600)
    return {
        "status": "prepared",
        "purpose": purpose,
        "pair_count": len(match_rows),
        "output_dir": str(target),
        "private_key": str(key_path),
        "private_key_sha256": _sha256_bytes(key_path.read_bytes()),
        "packet_sha256s": {
            value["rater_id"]: value["packet_sha256"] for value in rater_records
        },
    }


def _responses_to_ratings(
    key: Mapping[str, Any],
    responses: Mapping[str, Mapping[str, Mapping[str, str]]],
    *,
    round_number: int,
) -> tuple[tuple[NaturalnessRating, ...], tuple[dict[str, str], ...]]:
    items_by_rater: dict[str, dict[str, Mapping[str, str]]] = {}
    for item in key["items"]:
        items_by_rater.setdefault(item["rater_id"], {})[item["item_id"]] = item

    ratings = []
    assignments = []
    for rater in key["raters"]:
        rater_id = rater["rater_id"]
        expected = items_by_rater[rater_id]
        observed = responses[rater_id]
        if set(observed) != set(expected):
            raise ValueError("response items do not match the private key")
        for item_id, mapping in expected.items():
            response = observed[item_id]
            if response["packet_id"] != mapping["packet_id"]:
                raise ValueError("response packet ID does not match the private key")
            if set(_STIMULUS_FIELDS).issubset(response):
                displayed = {
                    field: response[field]
                    for field in ("item_id", *_STIMULUS_FIELDS)
                }
                if (
                    response.get("rating_question") != _RATING_QUESTION
                    or _sha256_json(displayed) != mapping["packet_item_sha256"]
                ):
                    raise ValueError("human-facing rating stimulus was edited")
            if not _rating_bool(response["independence_attested"]):
                raise ValueError("rater independence must be explicitly attested")
            values = {
                "candidate_a_naturalness": _rating_int(
                    response["candidate_a_naturalness"]
                ),
                "candidate_a_type_fit": _rating_int(response["candidate_a_type_fit"]),
                "candidate_a_malformed": _rating_bool(
                    response["candidate_a_malformed"]
                ),
                "candidate_b_naturalness": _rating_int(
                    response["candidate_b_naturalness"]
                ),
                "candidate_b_type_fit": _rating_int(response["candidate_b_type_fit"]),
                "candidate_b_malformed": _rating_bool(
                    response["candidate_b_malformed"]
                ),
            }
            role_values = {
                mapping["candidate_a_role"]: (
                    values["candidate_a_naturalness"],
                    values["candidate_a_type_fit"],
                    values["candidate_a_malformed"],
                ),
                mapping["candidate_b_role"]: (
                    values["candidate_b_naturalness"],
                    values["candidate_b_type_fit"],
                    values["candidate_b_malformed"],
                ),
            }
            real = role_values["real"]
            synthetic = role_values["synthetic"]
            ratings.append(
                NaturalnessRating(
                    pair_id=mapping["pair_id"],
                    rater_id=rater_id,
                    real_naturalness=real[0],
                    synthetic_naturalness=synthetic[0],
                    real_type_fit=real[1],
                    synthetic_type_fit=synthetic[1],
                    synthetic_malformed=synthetic[2],
                    round=round_number,
                    disagreement_registered=round_number == 2,
                )
            )
            assignments.append(
                {
                    "pair_id": mapping["pair_id"],
                    "rater_id": rater_id,
                    "blind_slot": rater["blind_slot"],
                    "submission_sha256": _sha256_json(
                        {
                            "packet_item_sha256": mapping["packet_item_sha256"],
                            "private_mapping": dict(mapping),
                            "response": {
                                field: response[field] for field in _RATING_FIELDS
                            },
                        }
                    ),
                }
            )
    return (
        tuple(sorted(ratings, key=lambda value: (value.pair_id, value.rater_id))),
        tuple(
            sorted(
                assignments,
                key=lambda value: (
                    value["pair_id"],
                    value["blind_slot"],
                    value["rater_id"],
                ),
            )
        ),
    )


def _load_key(
    path: str | Path,
    matches: Sequence[EntityMatch],
    *,
    expected_purpose: str,
    config_sha256: str,
    protocol_sha256: str,
    rating_protocol_sha256: str,
) -> dict[str, Any]:
    key_path = Path(path)
    try:
        key = json.loads(key_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("private unblinding key is unreadable") from error
    required = {
        "schema_version",
        "purpose",
        "config_sha256",
        "protocol_sha256",
        "rating_protocol_sha256",
        "matches_sha256",
        "raters",
        "items",
    }
    if (
        not isinstance(key, dict)
        or set(key) != required
        or key["schema_version"] != SCHEMA_VERSION
        or key["purpose"] != expected_purpose
    ):
        raise ValueError("private unblinding key has an invalid schema")
    if (
        key["config_sha256"] != config_sha256
        or key["protocol_sha256"] != protocol_sha256
        or key["rating_protocol_sha256"] != rating_protocol_sha256
    ):
        raise ValueError("private unblinding key does not match the active protocol")
    if key["matches_sha256"] != _matches_sha256(matches):
        raise ValueError("private unblinding key does not match the entity pairs")
    _verify_packet_files(_packet_root(key_path), key)
    return key


def _packet_root(key_path: Path) -> Path:
    """Resolve packet paths relative to their shared public/private root."""

    return key_path.parent.parent if key_path.parent.name == "private" else key_path.parent


def _verify_packet_files(directory: Path, key: Mapping[str, Any]) -> None:
    _packet_documents_from_files(directory, key)


def _packet_documents_from_files(
    directory: Path, key: Mapping[str, Any]
) -> list[dict[str, Any]]:
    documents = []
    for rater in key.get("raters", ()):
        if not isinstance(rater, dict) or set(rater) != {
            "rater_id",
            "blind_slot",
            "packet_file",
            "packet_sha256",
            "response_file",
        }:
            raise ValueError("private unblinding key rater record is invalid")
        for file_field in ("packet_file", "response_file"):
            relative = Path(rater[file_field])
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ValueError("private unblinding key file name is invalid")
        packet_path = directory / Path(rater["packet_file"])
        try:
            packet_bytes = packet_path.read_bytes()
            packet = json.loads(packet_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("blinded rating packet is unreadable") from error
        if _sha256_bytes(packet_bytes) != rater["packet_sha256"]:
            raise ValueError("blinded rating packet hash does not verify")
        documents.append(
            {
                "rater_id": rater["rater_id"],
                "packet_sha256": rater["packet_sha256"],
                "packet": packet,
            }
        )
    _verify_packet_documents(key, documents)
    return sorted(documents, key=lambda value: value["rater_id"])


def _validate_issuance(
    issuance: Mapping[str, Any],
    matches: Sequence[EntityMatch],
    *,
    expected_purpose: str,
    config_sha256: str,
    protocol_sha256: str,
    rating_protocol_sha256: str,
) -> dict[str, Any]:
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
    if (
        not isinstance(issuance, Mapping)
        or set(issuance) != required
        or issuance.get("kind") != "naturalness_packet_issuance"
        or issuance.get("schema_version") != SCHEMA_VERSION
        or issuance.get("purpose") != expected_purpose
        or issuance.get("config_sha256") != config_sha256
        or issuance.get("protocol_sha256") != protocol_sha256
        or issuance.get("rating_protocol_sha256") != rating_protocol_sha256
        or issuance.get("matches_sha256") != _matches_sha256(matches)
    ):
        raise ValueError("naturalness packet issuance has an invalid identity")
    key = issuance.get("private_key")
    if not isinstance(key, dict):
        raise ValueError("naturalness packet issuance private key is invalid")
    for field in (
        "purpose",
        "config_sha256",
        "protocol_sha256",
        "rating_protocol_sha256",
        "matches_sha256",
    ):
        if key.get(field) != issuance.get(field):
            raise ValueError("naturalness packet issuance key does not verify")
    _verify_packet_documents(key, issuance.get("packets"))
    return key


def _verify_packet_documents(
    key: Mapping[str, Any], documents: object
) -> None:
    if not isinstance(key["items"], list) or not isinstance(key["raters"], list):
        raise ValueError("private unblinding key collections are invalid")
    if not isinstance(documents, list):
        raise ValueError("issued packet documents are invalid")
    document_by_rater = {}
    for document in documents:
        if not isinstance(document, dict) or set(document) != {
            "rater_id",
            "packet_sha256",
            "packet",
        }:
            raise ValueError("issued packet document has an invalid schema")
        if document["rater_id"] in document_by_rater:
            raise ValueError("issued packet document raters must be unique")
        document_by_rater[document["rater_id"]] = document
    private_by_rater: dict[str, dict[str, Mapping[str, str]]] = {}
    for item in key["items"]:
        if not isinstance(item, dict) or set(item) != _PRIVATE_ITEM_FIELDS:
            raise ValueError("private unblinding key items are invalid")
        if {item["candidate_a_role"], item["candidate_b_role"]} != {
            "real",
            "synthetic",
        }:
            raise ValueError("private unblinding key roles are invalid")
        by_item = private_by_rater.setdefault(item["rater_id"], {})
        if item["item_id"] in by_item:
            raise ValueError("private unblinding key item IDs must be unique")
        by_item[item["item_id"]] = item
    observed_raters = set()
    for rater in key["raters"]:
        if not isinstance(rater, dict) or set(rater) != {
            "rater_id",
            "blind_slot",
            "packet_file",
            "packet_sha256",
            "response_file",
        }:
            raise ValueError("private unblinding key rater record is invalid")
        if rater["rater_id"] in observed_raters:
            raise ValueError("private unblinding key raters must be unique")
        observed_raters.add(rater["rater_id"])
        document = document_by_rater.get(rater["rater_id"])
        if (
            document is None
            or document["packet_sha256"] != rater["packet_sha256"]
        ):
            raise ValueError("issued packet document does not match the private key")
        packet = document["packet"]
        packet_bytes = (
            json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8")
        if _sha256_bytes(packet_bytes) != rater["packet_sha256"]:
            raise ValueError("blinded rating packet hash does not verify")
        if (
            not isinstance(packet, dict)
            or set(packet) != _PUBLIC_PACKET_FIELDS
            or packet.get("rater_id") != rater["rater_id"]
            or packet.get("purpose") != key["purpose"]
            or packet.get("rating_protocol_sha256")
            != key["rating_protocol_sha256"]
        ):
            raise ValueError("blinded rating packet identity does not verify")
        public_items = packet.get("items")
        if not isinstance(public_items, list) or any(
            not isinstance(item, dict) or set(item) != _PUBLIC_ITEM_FIELDS
            for item in public_items
        ):
            raise ValueError("blinded rating packet items are invalid")
        private_items = private_by_rater.get(rater["rater_id"], {})
        if {item.get("item_id") for item in public_items} != set(private_items):
            raise ValueError("blinded rating packet items do not match the private key")
        for item in public_items:
            private = private_items[item["item_id"]]
            if (
                private.get("packet_id") != packet.get("packet_id")
                or private.get("packet_item_sha256") != _sha256_json(item)
            ):
                raise ValueError("blinded packet-to-key mapping does not verify")
    if observed_raters != set(private_by_rater):
        raise ValueError("private unblinding key rater coverage is invalid")
    if observed_raters != set(document_by_rater):
        raise ValueError("issued packet document coverage is invalid")


def _response_evidence(
    responses: Mapping[str, Mapping[str, Mapping[str, str]]]
) -> tuple[dict[str, str], ...]:
    return tuple(
        sorted(
            (
                {field: row[field] for field in _RATING_FIELDS}
                for rater_rows in responses.values()
                for row in rater_rows.values()
            ),
            key=lambda value: (value["rater_id"], value["item_id"]),
        )
    )


def _load_responses(
    paths: Sequence[str | Path],
) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = {}
    for path in paths:
        try:
            with Path(path).open(newline="", encoding="utf-8") as handle:
                rows = tuple(csv.DictReader(handle))
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            raise ValueError("naturalness response file is unreadable") from error
        if not rows or set(rows[0]) != set(_WORKSHEET_FIELDS):
            raise ValueError("naturalness response file has an invalid schema")
        rater_ids = {row.get("rater_id") for row in rows}
        if len(rater_ids) != 1:
            raise ValueError("one response file must contain exactly one rater ID")
        rater_id = rater_ids.pop()
        _safe_id(rater_id, "response rater_id")
        if rater_id in result:
            raise ValueError("naturalness response rater IDs must be unique")
        by_item = {row["item_id"]: row for row in rows}
        if len(by_item) != len(rows) or any(not item_id for item_id in by_item):
            raise ValueError("naturalness response item IDs must be unique")
        result[rater_id] = by_item
    return result


def naturalness_matches_sha256(matches: Sequence[EntityMatch]) -> str:
    return _sha256_json(
        [asdict(value) for value in sorted(matches, key=lambda value: value.pair_id)]
    )


def issuance_pair_stimulus_sha256s(
    issuance: Mapping[str, Any],
) -> dict[str, str]:
    """Hash the exact blinded stimuli shown for each registered entity pair."""

    key = issuance.get("private_key")
    if not isinstance(key, dict):
        raise ValueError("naturalness packet issuance private key is invalid")
    documents = issuance.get("packets")
    _verify_packet_documents(key, documents)
    public_items = {
        (document["rater_id"], item["item_id"]): item
        for document in documents
        for item in document["packet"]["items"]
    }
    result: dict[str, str] = {}
    for item in key["items"]:
        public = public_items[(item["rater_id"], item["item_id"])]
        by_role = {
            item["candidate_a_role"]: {
                "name": public["candidate_a"],
                "sentence": public["sentence_a"],
            },
            item["candidate_b_role"]: {
                "name": public["candidate_b"],
                "sentence": public["sentence_b"],
            },
        }
        stimulus = {
            "pair_id": item["pair_id"],
            "coarse_type": public["coarse_type"],
            "real": by_role["real"],
            "synthetic": by_role["synthetic"],
        }
        digest = _sha256_json(stimulus)
        previous = result.setdefault(item["pair_id"], digest)
        if previous != digest:
            raise ValueError(
                "naturalness issuance presents inconsistent stimuli for one pair"
            )
    return dict(sorted(result.items()))


_matches_sha256 = naturalness_matches_sha256


def _neutral_sentence(coarse_type: str, name: str) -> str:
    type_label = coarse_type.replace("_", " ")
    return f"The following entry has entity type '{type_label}' and name: {name}."


def _write_response_template(
    path: Path,
    packet_id: str,
    rater_id: str,
    entries: Sequence[Mapping[str, str]],
) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_WORKSHEET_FIELDS)
        writer.writeheader()
        for item in entries:
            writer.writerow(
                {
                    "packet_id": packet_id,
                    "rater_id": rater_id,
                    "item_id": item["item_id"],
                    "rating_question": _RATING_QUESTION,
                    **{field: item[field] for field in _STIMULUS_FIELDS},
                }
            )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _rating_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("naturalness and type-fit ratings must be integers") from error
    if not 1 <= parsed <= 5 or str(parsed) != str(value).strip():
        raise ValueError("naturalness and type-fit ratings must be integers from 1 to 5")
    return parsed


def _rating_bool(value: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError("malformed ratings must be true or false")


def _safe_id(value: object, label: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} must be a safe identifier")


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
