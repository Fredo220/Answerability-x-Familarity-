"""Deterministic corpus for the Same-String representation replication v3."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REP_V3_STUDY_ID = "same-string-representation-replication-v3"
REP_V3_SEED = 20260803
REP_V3_SPLIT_COUNTS: Mapping[str, int] = {
    "representation_train": 32,
    "representation_validation": 8,
    "entity_test": 20,
    "template_test": 20,
}
REP_V3_SEEN_TEMPLATES = (
    "registry_bullets",
    "ledger_bullets",
    "catalog_bullets",
    "record_bullets",
)
REP_V3_HELDOUT_TEMPLATES = ("dossier_bullets", "index_bullets")
REP_V3_EXPOSURES = ("low_exposure", "high_exposure")
REP_V3_ANSWERABILITY = ("target_unbound", "target_bound")

_TEMPLATE_WORDS = {
    "registry_bullets": ("Registry", "registry"),
    "ledger_bullets": ("Ledger", "ledger"),
    "catalog_bullets": ("Catalog", "catalog"),
    "record_bullets": ("Record", "record"),
    "dossier_bullets": ("Dossier", "dossier"),
    "index_bullets": ("Index", "index"),
}
_PROPERTIES = (
    "amber",
    "cobalt",
    "ivory",
    "silver",
    "violet",
    "umber",
    "scarlet",
    "teal",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class ReplicationPromptV3:
    example_id: str
    entity_unit_id: str
    split: str
    template_family: str
    exposure: str
    answerability: str
    target_text: str
    distractor_text: str
    registry_code: str
    neutral_property: str
    user_text: str
    target_intro_span: tuple[int, int]
    target_query_span: tuple[int, int]
    rendered_token_ids: tuple[int, ...]
    rendered_prompt_sha256: str
    block: str = "same_string"

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_intro_span", tuple(self.target_intro_span))
        object.__setattr__(self, "target_query_span", tuple(self.target_query_span))
        object.__setattr__(self, "rendered_token_ids", tuple(self.rendered_token_ids))
        if self.block != "same_string":
            raise ValueError("v3 prompts must use the Same-String block")
        if self.split not in REP_V3_SPLIT_COUNTS:
            raise ValueError("v3 prompt split is invalid")
        if self.exposure not in REP_V3_EXPOSURES:
            raise ValueError("v3 prompt exposure is invalid")
        if self.answerability not in REP_V3_ANSWERABILITY:
            raise ValueError("v3 prompt answerability is invalid")
        if self.template_family not in _TEMPLATE_WORDS:
            raise ValueError("v3 template family is invalid")
        for span in (self.target_intro_span, self.target_query_span):
            if len(span) != 2 or self.user_text[slice(*span)] != self.target_text:
                raise ValueError("v3 target spans must bind the target text")
        if self.target_intro_span[1] > self.target_query_span[0]:
            raise ValueError("v3 target spans must be ordered")
        expected = _example_id_payload(self)
        if self.example_id != _sha256(expected):
            raise ValueError("v3 example ID must derive from canonical content")


@dataclass(frozen=True)
class ReplicationAuditV3:
    checks: Mapping[str, bool]
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(self.checks.values()) and not self.violations


@dataclass(frozen=True)
class ReplicationCorpusV3:
    prompts: tuple[ReplicationPromptV3, ...]
    audit: ReplicationAuditV3
    manifest_sha256: str
    tokenizer_id: str


@dataclass(frozen=True)
class ReplicationCorpusPathsV3:
    prompts: Path
    manifest: Path


def _example_id_payload(row: ReplicationPromptV3) -> dict[str, Any]:
    return {
        key: value
        for key, value in asdict(row).items()
        if key not in {"example_id", "rendered_token_ids", "rendered_prompt_sha256"}
    }


def _render_user_text(
    *,
    family: str,
    target: str,
    distractor: str,
    code: str,
    property_name: str,
    exposure: str,
    answerability: str,
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    heading, noun = _TEMPLATE_WORDS[family]
    if exposure == "high_exposure":
        exposure_lines = (
            f"- {target} is widely documented in several {noun} reports.\n"
            f"- {distractor} is briefly listed in one {noun} note."
        )
    else:
        exposure_lines = (
            f"- {distractor} is widely documented in several {noun} reports.\n"
            f"- {target} is briefly listed in one {noun} note."
        )
    if answerability == "target_bound":
        task_lines = (
            f"- {target} has archive code {code}.\n"
            f"- {distractor} has marker {property_name}."
        )
    else:
        task_lines = (
            f"- {distractor} has archive code {code}.\n"
            f"- {target} has marker {property_name}."
        )
    user_text = (
        f"{heading} context:\n{exposure_lines} Task: The target is {target}. "
        f"Read these entries:\n{task_lines}\n"
        f"Question: What is the archive code for {target}?"
    )
    task_start = user_text.index(" Task: ") + len(" Task: ")
    intro_start = user_text.index(target, task_start)
    query_start = user_text.rindex(target)
    return (
        user_text,
        (intro_start, intro_start + len(target)),
        (query_start, query_start + len(target)),
    )


def _tokenize_prompt(tokenizer: Any, user_text: str) -> tuple[str, tuple[int, ...]]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, add_special_tokens=False)
    raw_ids = encoded["input_ids"]
    if raw_ids and isinstance(raw_ids[0], Sequence):
        raw_ids = raw_ids[0]
    return rendered, tuple(int(value) for value in raw_ids)


def _unit_allocation() -> tuple[tuple[str, int, str], ...]:
    allocation = []
    global_index = 0
    for split, count in REP_V3_SPLIT_COUNTS.items():
        families = (
            REP_V3_HELDOUT_TEMPLATES
            if split == "template_test"
            else REP_V3_SEEN_TEMPLATES
        )
        for local_index in range(count):
            allocation.append((split, global_index, families[local_index % len(families)]))
            global_index += 1
    return tuple(allocation)


def build_replication_corpus(tokenizer: Any) -> ReplicationCorpusV3:
    prompts = []
    for split, unit_index, family in _unit_allocation():
        unit = f"rep-v3-unit-{unit_index:03d}"
        target = f"Nomen{2 * unit_index:03d}"
        distractor = f"Nomen{2 * unit_index + 1:03d}"
        code = f"K{unit_index:03d}"
        property_name = _PROPERTIES[unit_index % len(_PROPERTIES)]
        for exposure in REP_V3_EXPOSURES:
            for answerability in REP_V3_ANSWERABILITY:
                user_text, intro_span, query_span = _render_user_text(
                    family=family,
                    target=target,
                    distractor=distractor,
                    code=code,
                    property_name=property_name,
                    exposure=exposure,
                    answerability=answerability,
                )
                rendered, token_ids = _tokenize_prompt(tokenizer, user_text)
                fields = {
                    "entity_unit_id": unit,
                    "split": split,
                    "template_family": family,
                    "exposure": exposure,
                    "answerability": answerability,
                    "target_text": target,
                    "distractor_text": distractor,
                    "registry_code": code,
                    "neutral_property": property_name,
                    "user_text": user_text,
                    "target_intro_span": intro_span,
                    "target_query_span": query_span,
                    "rendered_token_ids": token_ids,
                    "rendered_prompt_sha256": _sha256_bytes(rendered.encode("utf-8")),
                    "block": "same_string",
                }
                example_id = _sha256(
                    {
                        key: value
                        for key, value in fields.items()
                        if key not in {"rendered_token_ids", "rendered_prompt_sha256"}
                    }
                )
                prompts.append(ReplicationPromptV3(example_id=example_id, **fields))
    prepared = tuple(sorted(prompts, key=lambda row: row.example_id))
    audit = audit_replication_corpus(prepared, tokenizer)
    payload = [_prompt_record(row) for row in prepared]
    return ReplicationCorpusV3(
        prompts=prepared,
        audit=audit,
        manifest_sha256=_sha256(payload),
        tokenizer_id=str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)),
    )


def _prompt_record(row: ReplicationPromptV3) -> dict[str, Any]:
    value = asdict(row)
    value["target_intro_span"] = list(row.target_intro_span)
    value["target_query_span"] = list(row.target_query_span)
    value["rendered_token_ids"] = list(row.rendered_token_ids)
    return value


def audit_replication_corpus(
    prompts: Sequence[ReplicationPromptV3], tokenizer: Any
) -> ReplicationAuditV3:
    rows = tuple(prompts)
    by_unit: dict[str, list[ReplicationPromptV3]] = {}
    for row in rows:
        by_unit.setdefault(row.entity_unit_id, []).append(row)
    expected_cells = Counter(
        (exposure, answerability)
        for exposure in REP_V3_EXPOSURES
        for answerability in REP_V3_ANSWERABILITY
    )
    complete = all(
        Counter((row.exposure, row.answerability) for row in unit_rows) == expected_cells
        for unit_rows in by_unit.values()
    )
    split_counts = all(
        len({row.entity_unit_id for row in rows if row.split == split}) == count
        for split, count in REP_V3_SPLIT_COUNTS.items()
    )
    seen_test = {
        row.template_family for row in rows if row.split == "template_test"
    }
    other_templates = {
        row.template_family for row in rows if row.split != "template_test"
    }

    def multiset_equal(task: str) -> bool:
        for unit_rows in by_unit.values():
            other_values = REP_V3_EXPOSURES if task == "answerability" else REP_V3_ANSWERABILITY
            for other in other_values:
                selected = [
                    row
                    for row in unit_rows
                    if (row.exposure if task == "answerability" else row.answerability)
                    == other
                ]
                if len(selected) != 2 or Counter(selected[0].rendered_token_ids) != Counter(
                    selected[1].rendered_token_ids
                ):
                    return False
        return True

    bos_id = getattr(tokenizer, "bos_token_id", None)
    checks = {
        "row_count": len(rows) == 320 and len({row.example_id for row in rows}) == 320,
        "split_counts": split_counts,
        "complete_2x2_units": len(by_unit) == 80 and complete,
        "split_identity_disjointness": all(
            len({row.split for row in unit_rows}) == 1 for unit_rows in by_unit.values()
        ),
        "template_holdout": seen_test == set(REP_V3_HELDOUT_TEMPLATES)
        and not seen_test.intersection(other_templates)
        and other_templates == set(REP_V3_SEEN_TEMPLATES),
        "answerability_token_multisets": multiset_equal("answerability"),
        "exposure_token_multisets": multiset_equal("exposure"),
        "single_bos": bos_id is not None
        and all(row.rendered_token_ids.count(int(bos_id)) == 1 for row in rows),
        "unit_constants": all(
            len({row.target_text for row in unit_rows}) == 1
            and len({row.distractor_text for row in unit_rows}) == 1
            and len({row.registry_code for row in unit_rows}) == 1
            and len({row.neutral_property for row in unit_rows}) == 1
            and len({row.template_family for row in unit_rows}) == 1
            for unit_rows in by_unit.values()
        ),
        "code_not_in_exposure": all(
            row.registry_code not in row.user_text.split(" Task: ", 1)[0] for row in rows
        ),
    }
    violations = tuple(sorted(name for name, passed in checks.items() if not passed))
    return ReplicationAuditV3(checks=checks, violations=violations)


def write_replication_corpus(
    corpus: ReplicationCorpusV3, destination: str | Path
) -> ReplicationCorpusPathsV3:
    if not corpus.audit.passed:
        raise ValueError("cannot write a v3 corpus that failed its audit")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    prompts_path = root / "same_string_replication_v3_prompts.jsonl"
    manifest_path = root / "same_string_replication_v3_manifest.json"
    prompt_bytes = b"".join(_canonical_json(_prompt_record(row)) + b"\n" for row in corpus.prompts)
    prompts_path.write_bytes(prompt_bytes)
    manifest = {
        "schema_version": 1,
        "study_id": REP_V3_STUDY_ID,
        "row_count": len(corpus.prompts),
        "unit_count": len({row.entity_unit_id for row in corpus.prompts}),
        "split_counts": dict(REP_V3_SPLIT_COUNTS),
        "tokenizer_id": corpus.tokenizer_id,
        "prompts_file": prompts_path.name,
        "prompts_sha256": _sha256_bytes(prompt_bytes),
        "manifest_sha256": corpus.manifest_sha256,
        "audit": {"checks": dict(corpus.audit.checks), "violations": list(corpus.audit.violations)},
    }
    manifest_path.write_bytes(_canonical_json(manifest) + b"\n")
    return ReplicationCorpusPathsV3(prompts=prompts_path, manifest=manifest_path)


def verify_replication_corpus(
    manifest_path: str | Path, tokenizer: Any
) -> ReplicationCorpusV3:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    prompts_path = path.parent / manifest["prompts_file"]
    prompt_bytes = prompts_path.read_bytes()
    if _sha256_bytes(prompt_bytes) != manifest.get("prompts_sha256"):
        raise ValueError("v3 prompt file hash mismatch")
    raw_rows = [json.loads(line) for line in prompt_bytes.splitlines() if line]
    try:
        rows = tuple(ReplicationPromptV3(**row) for row in raw_rows)
    except (TypeError, ValueError) as error:
        raise ValueError("v3 prompt records are not canonical") from error
    reconstructed = build_replication_corpus(tokenizer)
    if tuple(_prompt_record(row) for row in rows) != tuple(
        _prompt_record(row) for row in reconstructed.prompts
    ):
        raise ValueError("v3 corpus does not reconstruct from the frozen design")
    if reconstructed.manifest_sha256 != manifest.get("manifest_sha256"):
        raise ValueError("v3 manifest hash mismatch")
    return reconstructed
