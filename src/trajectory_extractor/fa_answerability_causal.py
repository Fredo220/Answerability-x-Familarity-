"""Frozen construction contracts for the Same-String answerability causal pilot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from trajectory_extractor.fa_activations import (
    ANCHOR_NAMES,
    ActivationRecord,
    load_activation_records,
    resume_activation_shard,
)
from trajectory_extractor.fa_same_string_replication_v3 import (
    REP_V3_SPLIT_COUNTS,
    REP_V3_STUDY_ID,
    ReplicationPromptV3,
)


CAUSAL_STUDY_ID = "same-string-answerability-causal-pilot-v1"
CAUSAL_SPLIT_SEED = 20260804
CAUSAL_SPLIT_COUNTS: Mapping[str, int] = {
    "causal_validation": 12,
    "causal_entity_test": 18,
    "causal_template_test": 18,
}
CAUSAL_VALIDATION_LAYERS = (0, 6, 12, 18, 25)
CAUSAL_VALIDATION_MULTIPLIERS = (0.25, 0.5, 1.0)
CAUSAL_EXPOSURES = ("low_exposure", "high_exposure")
CAUSAL_ANSWERABILITY = ("target_unbound", "target_bound")
CAUSAL_DIRECTION_ANCHOR = "user_prompt_end"
_SEEN_TEMPLATES = (
    "registry_bullets",
    "ledger_bullets",
    "catalog_bullets",
    "record_bullets",
)
_FRESH_TEMPLATES = ("archive_cards", "filing_cards")
_TEMPLATE_WORDS = {
    "registry_bullets": ("Registry", "registry"),
    "ledger_bullets": ("Ledger", "ledger"),
    "catalog_bullets": ("Catalog", "catalog"),
    "record_bullets": ("Record", "record"),
    "archive_cards": ("Archive", "archive"),
    "filing_cards": ("Filing", "filing"),
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
_OUTPUT_CONTRACT = (
    "Reply with only the archive code. If the target's archive code is not stated, "
    "reply UNKNOWN."
)
_SHA256_LENGTH = 64


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise ValueError(f"{name} must be a SHA-256 hash")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hash") from error
    return value


@dataclass(frozen=True)
class CausalPrompt:
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
    output_contract: str = _OUTPUT_CONTRACT
    block: str = "same_string_causal"

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_intro_span", tuple(self.target_intro_span))
        object.__setattr__(self, "target_query_span", tuple(self.target_query_span))
        object.__setattr__(self, "rendered_token_ids", tuple(int(x) for x in self.rendered_token_ids))
        if self.block != "same_string_causal":
            raise ValueError("causal prompts must use the causal Same-String block")
        if self.split not in CAUSAL_SPLIT_COUNTS:
            raise ValueError("causal prompt split is invalid")
        if self.template_family not in _TEMPLATE_WORDS:
            raise ValueError("causal prompt template family is invalid")
        if self.exposure not in CAUSAL_EXPOSURES:
            raise ValueError("causal prompt exposure is invalid")
        if self.answerability not in CAUSAL_ANSWERABILITY:
            raise ValueError("causal prompt answerability is invalid")
        if self.output_contract != _OUTPUT_CONTRACT:
            raise ValueError("causal prompt output contract is not registered")
        for span in (self.target_intro_span, self.target_query_span):
            if len(span) != 2 or self.user_text[slice(*span)] != self.target_text:
                raise ValueError("causal target spans must bind the target text")
        if self.target_intro_span[1] > self.target_query_span[0]:
            raise ValueError("causal target spans must be ordered")
        _require_sha256(self.rendered_prompt_sha256, "causal rendered prompt hash")
        if self.example_id != _sha256(_prompt_identity_payload(self)):
            raise ValueError("causal example ID must derive from canonical content")


@dataclass(frozen=True)
class CausalAudit:
    checks: Mapping[str, bool]
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(self.checks.values()) and not self.violations


@dataclass(frozen=True)
class CausalCorpus:
    prompts: tuple[CausalPrompt, ...]
    audit: CausalAudit
    manifest_sha256: str
    tokenizer_id: str


@dataclass(frozen=True)
class CausalCorpusPaths:
    prompts: Path
    manifest: Path


@dataclass(frozen=True)
class AnswerabilityDirection:
    layer_id: int
    vector: np.ndarray
    natural_scale: float
    training_unit_count: int
    source_split: str
    direction_sha256: str

    def __post_init__(self) -> None:
        vector = np.array(self.vector, dtype=np.float64, copy=True, order="C")
        if self.layer_id not in CAUSAL_VALIDATION_LAYERS:
            raise ValueError("direction layer is not registered")
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("direction vector must be a nonempty finite vector")
        if not np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-9, atol=1e-9):
            raise ValueError("direction vector must have unit L2 norm")
        if not np.isfinite(self.natural_scale) or self.natural_scale <= 0.0:
            raise ValueError("direction natural scale must be positive and finite")
        if self.training_unit_count < 1 or self.source_split != "representation_train":
            raise ValueError("direction must be trained only on representation_train")
        vector.setflags(write=False)
        object.__setattr__(self, "vector", vector)
        expected = _sha256(_direction_record(self, include_hash=False))
        if self.direction_sha256 != expected:
            raise ValueError("direction hash does not match direction content")


@dataclass(frozen=True)
class DirectionBundle:
    directions: tuple[AnswerabilityDirection, ...]
    source: "V3DirectionProvenance"
    bundle_sha256: str

    def __post_init__(self) -> None:
        directions = tuple(self.directions)
        if tuple(direction.layer_id for direction in directions) != CAUSAL_VALIDATION_LAYERS:
            raise ValueError("direction bundle must contain every registered layer in order")
        if not isinstance(self.source, V3DirectionProvenance):
            raise ValueError("direction bundle requires verified v3 provenance")
        object.__setattr__(self, "directions", directions)
        expected = _sha256(_direction_bundle_record(self, include_hash=False))
        if self.bundle_sha256 != expected:
            raise ValueError("direction bundle hash does not match direction content")


@dataclass(frozen=True)
class V3DirectionProvenance:
    v3_prompts_sha256: str
    v3_manifest_sha256: str
    activation_manifest_sha256: str
    activation_npz_sha256: str
    activation_index_sha256: str
    activation_request_sha256: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    chat_template_sha256: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.v3_prompts_sha256, "v3 prompts hash"),
            (self.v3_manifest_sha256, "v3 manifest hash"),
            (self.activation_manifest_sha256, "activation manifest hash"),
            (self.activation_npz_sha256, "activation NPZ hash"),
            (self.activation_index_sha256, "activation index hash"),
            (self.activation_request_sha256, "activation request hash"),
            (self.chat_template_sha256, "chat template hash"),
        ):
            _require_sha256(value, name)
        for value, name in (
            (self.model_id, "v3 model ID"),
            (self.model_revision, "v3 model revision"),
            (self.tokenizer_id, "v3 tokenizer ID"),
            (self.tokenizer_revision, "v3 tokenizer revision"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be nonempty")


@dataclass(frozen=True)
class VerifiedV3TrainingInputs:
    prompts: tuple[ReplicationPromptV3, ...]
    records: tuple[ActivationRecord, ...]
    source: V3DirectionProvenance

    def __post_init__(self) -> None:
        prompts = tuple(self.prompts)
        records = tuple(self.records)
        if not prompts or not isinstance(self.source, V3DirectionProvenance):
            raise ValueError("verified v3 training inputs are incomplete")
        if any(row.split != "representation_train" for row in prompts):
            raise ValueError("verified inputs must contain representation_train rows only")
        if {row.example_id for row in prompts} != {row.example_id for row in records}:
            raise ValueError("verified v3 activation identities do not match prompts")
        object.__setattr__(self, "prompts", prompts)
        object.__setattr__(self, "records", records)


@dataclass(frozen=True)
class CausalExpectedProvenance:
    corpus_sha256: str
    direction_bundle_sha256: str
    direction_hashes: Mapping[int, str]
    model_sha256: str
    tokenizer_sha256: str

    def __post_init__(self) -> None:
        hashes = {int(layer): value for layer, value in self.direction_hashes.items()}
        if tuple(sorted(hashes)) != CAUSAL_VALIDATION_LAYERS:
            raise ValueError("expected provenance must bind every registered direction layer")
        for value, name in (
            (self.corpus_sha256, "expected corpus hash"),
            (self.direction_bundle_sha256, "expected direction bundle hash"),
            (self.model_sha256, "expected model hash"),
            (self.tokenizer_sha256, "expected tokenizer hash"),
            *[(value, "expected direction hash") for value in hashes.values()],
        ):
            _require_sha256(value, name)
        object.__setattr__(self, "direction_hashes", MappingProxyType(hashes))


@dataclass(frozen=True)
class ValidationCandidate:
    layer_id: int
    multiplier: float
    unit_effects: tuple[tuple[str, float], ...]
    invalid_output_rate: float
    bound_accuracy_drop: float
    corpus_sha256: str
    direction_bundle_sha256: str
    model_sha256: str
    tokenizer_sha256: str
    direction_sha256: str

    def __post_init__(self) -> None:
        effects = tuple((str(unit), float(effect)) for unit, effect in self.unit_effects)
        if self.layer_id not in CAUSAL_VALIDATION_LAYERS:
            raise ValueError("validation candidate layer is not registered")
        if float(self.multiplier) not in CAUSAL_VALIDATION_MULTIPLIERS:
            raise ValueError("validation candidate multiplier is not registered")
        if not effects or len({unit for unit, _ in effects}) != len(effects):
            raise ValueError("validation unit effects must have unique unit IDs")
        if not all(unit and np.isfinite(effect) for unit, effect in effects):
            raise ValueError("validation unit effects must be finite")
        if not np.isfinite(self.invalid_output_rate) or not np.isfinite(self.bound_accuracy_drop):
            raise ValueError("validation gate values must be finite")
        for value, name in (
            (self.corpus_sha256, "validation corpus hash"),
            (self.direction_bundle_sha256, "validation direction bundle hash"),
            (self.model_sha256, "validation model hash"),
            (self.tokenizer_sha256, "validation tokenizer hash"),
            (self.direction_sha256, "validation direction hash"),
        ):
            _require_sha256(value, name)
        object.__setattr__(self, "unit_effects", tuple(sorted(effects)))
        object.__setattr__(self, "multiplier", float(self.multiplier))
        object.__setattr__(self, "invalid_output_rate", float(self.invalid_output_rate))
        object.__setattr__(self, "bound_accuracy_drop", float(self.bound_accuracy_drop))

    @property
    def mean_bidirectional_effect(self) -> float:
        return float(np.mean([effect for _, effect in self.unit_effects]))


@dataclass(frozen=True)
class ValidationSelection:
    layer_id: int
    multiplier: float
    mean_bidirectional_effect: float
    direction_sha256: str
    corpus_sha256: str
    direction_bundle_sha256: str
    model_sha256: str
    tokenizer_sha256: str
    corpus_sha256: str
    selection_sha256: str

    def __post_init__(self) -> None:
        if self.layer_id not in CAUSAL_VALIDATION_LAYERS:
            raise ValueError("selected layer is not registered")
        if float(self.multiplier) not in CAUSAL_VALIDATION_MULTIPLIERS:
            raise ValueError("selected multiplier is not registered")
        if not np.isfinite(self.mean_bidirectional_effect):
            raise ValueError("selected effect must be finite")
        for value, name in (
            (self.direction_sha256, "selection direction hash"),
            (self.corpus_sha256, "selection corpus hash"),
            (self.direction_bundle_sha256, "selection direction bundle hash"),
            (self.model_sha256, "selection model hash"),
            (self.tokenizer_sha256, "selection tokenizer hash"),
        ):
            _require_sha256(value, name)
        expected = _sha256(_selection_record(self, include_hash=False))
        if self.selection_sha256 != expected:
            raise ValueError("selection hash does not match selection content")


def _prompt_identity_payload(row: CausalPrompt) -> dict[str, Any]:
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
        f"Question: What is the archive code for {target}?\n{_OUTPUT_CONTRACT}"
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
    rows = []
    global_index = 0
    for split, count in CAUSAL_SPLIT_COUNTS.items():
        families = _FRESH_TEMPLATES if split == "causal_template_test" else _SEEN_TEMPLATES
        for local_index in range(count):
            rows.append((split, global_index, families[local_index % len(families)]))
            global_index += 1
    return tuple(rows)


def build_causal_corpus(
    tokenizer: Any, *, v3_prompts: Sequence[ReplicationPromptV3] | None = None
) -> CausalCorpus:
    if v3_prompts is None:
        raise ValueError("causal corpus construction requires v3 exclusions")
    prompts = []
    for split, unit_index, family in _unit_allocation():
        target = f"Causa{2 * unit_index:04d}"
        distractor = f"Causa{2 * unit_index + 1:04d}"
        code = f"Z{unit_index:04d}"
        property_name = _PROPERTIES[unit_index % len(_PROPERTIES)]
        unit = f"causal-v1-unit-{unit_index:03d}"
        for exposure in CAUSAL_EXPOSURES:
            for answerability in CAUSAL_ANSWERABILITY:
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
                    "output_contract": _OUTPUT_CONTRACT,
                    "block": "same_string_causal",
                }
                example_id = _sha256(
                    {
                        key: value
                        for key, value in fields.items()
                        if key not in {"rendered_token_ids", "rendered_prompt_sha256"}
                    }
                )
                prompts.append(CausalPrompt(example_id=example_id, **fields))
    prepared = tuple(sorted(prompts, key=lambda row: row.example_id))
    audit = audit_causal_corpus(prepared, tokenizer, v3_prompts=v3_prompts)
    return CausalCorpus(
        prompts=prepared,
        audit=audit,
        manifest_sha256=_sha256([_prompt_record(row) for row in prepared]),
        tokenizer_id=str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)),
    )


def _prompt_record(row: CausalPrompt) -> dict[str, Any]:
    value = asdict(row)
    value["target_intro_span"] = list(row.target_intro_span)
    value["target_query_span"] = list(row.target_query_span)
    value["rendered_token_ids"] = list(row.rendered_token_ids)
    return value


def _identity_sets_from_v3(v3_prompts: Sequence[Any]) -> dict[str, frozenset[str]]:
    protected = [
        row
        for row in v3_prompts
        if getattr(row, "split", None) in {"entity_test", "template_test"}
    ]
    return {
        "example_ids": frozenset(str(row.example_id) for row in protected),
        "entity_unit_ids": frozenset(str(row.entity_unit_id) for row in protected),
        "names": frozenset(
            str(value)
            for row in protected
            for value in (row.target_text, row.distractor_text)
        ),
        "registry_codes": frozenset(str(row.registry_code) for row in protected),
    }


def _has_complete_v3_test_exclusions(v3_prompts: Sequence[ReplicationPromptV3]) -> bool:
    for split in ("entity_test", "template_test"):
        rows = [row for row in v3_prompts if row.split == split]
        if (
            len(rows) != 4 * REP_V3_SPLIT_COUNTS[split]
            or len({row.entity_unit_id for row in rows}) != REP_V3_SPLIT_COUNTS[split]
        ):
            return False
    return True


def audit_causal_corpus(
    prompts: Sequence[CausalPrompt],
    tokenizer: Any,
    *,
    v3_prompts: Sequence[ReplicationPromptV3] | None = None,
) -> CausalAudit:
    if v3_prompts is None:
        raise ValueError("causal corpus audit requires v3 exclusions")
    if not _has_complete_v3_test_exclusions(v3_prompts):
        raise ValueError("causal corpus audit requires complete v3 exclusions")
    rows = tuple(prompts)
    by_unit: dict[str, list[CausalPrompt]] = {}
    for row in rows:
        by_unit.setdefault(row.entity_unit_id, []).append(row)
    expected_cells = Counter(
        (exposure, answerability)
        for exposure in CAUSAL_EXPOSURES
        for answerability in CAUSAL_ANSWERABILITY
    )
    identity_sets = _identity_sets_from_v3(v3_prompts)
    if not all(identity_sets.values()):
        raise ValueError("causal corpus audit requires complete v3 exclusions")

    def paired_multisets(axis: str) -> bool:
        opposite = CAUSAL_EXPOSURES if axis == "answerability" else CAUSAL_ANSWERABILITY
        for unit_rows in by_unit.values():
            for value in opposite:
                selected = [
                    row
                    for row in unit_rows
                    if (row.exposure if axis == "answerability" else row.answerability)
                    == value
                ]
                if len(selected) != 2 or Counter(selected[0].rendered_token_ids) != Counter(
                    selected[1].rendered_token_ids
                ):
                    return False
        return True

    def tokenizer_replay() -> bool:
        try:
            for row in rows:
                rendered, token_ids = _tokenize_prompt(tokenizer, row.user_text)
                if token_ids != row.rendered_token_ids:
                    return False
                if _sha256_bytes(rendered.encode("utf-8")) != row.rendered_prompt_sha256:
                    return False
        except (KeyError, TypeError, ValueError):
            return False
        return True

    current_identities = {
        "example_ids": {row.example_id for row in rows},
        "entity_unit_ids": {row.entity_unit_id for row in rows},
        "names": {value for row in rows for value in (row.target_text, row.distractor_text)},
        "registry_codes": {row.registry_code for row in rows},
    }
    identity_isolated = all(
        not current_identities.get(name, set()).intersection(values)
        for name, values in identity_sets.items()
    )
    split_counts = all(
        len({row.entity_unit_id for row in rows if row.split == split}) == count
        for split, count in CAUSAL_SPLIT_COUNTS.items()
    )
    split_identities = {
        split: {
            "names": {
                value
                for row in rows
                if row.split == split
                for value in (row.target_text, row.distractor_text)
            },
            "registry_codes": {row.registry_code for row in rows if row.split == split},
        }
        for split in CAUSAL_SPLIT_COUNTS
    }
    causal_identity_disjointness = all(
        not split_identities[left][kind].intersection(split_identities[right][kind])
        for index, left in enumerate(CAUSAL_SPLIT_COUNTS)
        for right in tuple(CAUSAL_SPLIT_COUNTS)[index + 1 :]
        for kind in ("names", "registry_codes")
    )
    fresh_templates = {
        row.template_family for row in rows if row.split == "causal_template_test"
    }
    other_templates = {
        row.template_family for row in rows if row.split != "causal_template_test"
    }
    checks = {
        "row_count": len(rows) == 192 and len({row.example_id for row in rows}) == 192,
        "split_counts": split_counts,
        "complete_2x2_units": len(by_unit) == 48
        and all(
            Counter((row.exposure, row.answerability) for row in unit_rows)
            == expected_cells
            for unit_rows in by_unit.values()
        ),
        "split_identity_disjointness": all(
            len({row.split for row in unit_rows}) == 1 for unit_rows in by_unit.values()
        ),
        "causal_identity_disjointness": causal_identity_disjointness,
        "fresh_template_holdout": fresh_templates == set(_FRESH_TEMPLATES)
        and not fresh_templates.intersection(other_templates)
        and other_templates == set(_SEEN_TEMPLATES),
        "answerability_token_multisets": paired_multisets("answerability"),
        "exposure_token_multisets": paired_multisets("exposure"),
        "tokenizer_replay": tokenizer_replay(),
        "unit_constants": all(
            len({row.target_text for row in unit_rows}) == 1
            and len({row.distractor_text for row in unit_rows}) == 1
            and len({row.registry_code for row in unit_rows}) == 1
            and len({row.neutral_property for row in unit_rows}) == 1
            and len({row.template_family for row in unit_rows}) == 1
            for unit_rows in by_unit.values()
        ),
        "output_contract": all(row.output_contract == _OUTPUT_CONTRACT for row in rows),
        "v3_test_identity_isolation": identity_isolated,
    }
    violations = tuple(sorted(name for name, passed in checks.items() if not passed))
    return CausalAudit(checks=checks, violations=violations)


def write_causal_corpus(corpus: CausalCorpus, destination: str | Path) -> CausalCorpusPaths:
    if not corpus.audit.passed:
        raise ValueError("cannot write a causal corpus that failed its audit")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    prompts_path = root / "same_string_answerability_causal_prompts.jsonl"
    manifest_path = root / "same_string_answerability_causal_manifest.json"
    prompt_bytes = b"".join(_canonical_json(_prompt_record(row)) + b"\n" for row in corpus.prompts)
    prompts_path.write_bytes(prompt_bytes)
    manifest = {
        "schema_version": 1,
        "study_id": CAUSAL_STUDY_ID,
        "row_count": len(corpus.prompts),
        "unit_count": len({row.entity_unit_id for row in corpus.prompts}),
        "split_counts": dict(CAUSAL_SPLIT_COUNTS),
        "tokenizer_id": corpus.tokenizer_id,
        "prompts_file": prompts_path.name,
        "prompts_sha256": _sha256_bytes(prompt_bytes),
        "manifest_sha256": corpus.manifest_sha256,
        "audit": {"checks": dict(corpus.audit.checks), "violations": list(corpus.audit.violations)},
    }
    manifest_path.write_bytes(_canonical_json(manifest) + b"\n")
    return CausalCorpusPaths(prompts=prompts_path, manifest=manifest_path)


def verify_causal_corpus(
    manifest_path: str | Path,
    tokenizer: Any,
    *,
    v3_prompts: Sequence[ReplicationPromptV3] | None = None,
) -> CausalCorpus:
    if v3_prompts is None:
        raise ValueError("causal corpus verification requires v3 exclusions")
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    prompts_path = path.parent / manifest["prompts_file"]
    prompt_bytes = prompts_path.read_bytes()
    if _sha256_bytes(prompt_bytes) != manifest.get("prompts_sha256"):
        raise ValueError("causal prompt file hash mismatch")
    try:
        rows = tuple(CausalPrompt(**json.loads(line)) for line in prompt_bytes.splitlines() if line)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("causal prompt records are not canonical") from error
    reconstructed = build_causal_corpus(tokenizer, v3_prompts=v3_prompts)
    if tuple(_prompt_record(row) for row in rows) != tuple(
        _prompt_record(row) for row in reconstructed.prompts
    ):
        raise ValueError("causal corpus does not reconstruct from the frozen design")
    if reconstructed.manifest_sha256 != manifest.get("manifest_sha256"):
        raise ValueError("causal manifest hash mismatch")
    return reconstructed


def _activations_at_user_prompt_end(
    records: Sequence[ActivationRecord],
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    anchor_index = ANCHOR_NAMES.index(CAUSAL_DIRECTION_ANCHOR)
    for record in records:
        if record.layer_ids != CAUSAL_VALIDATION_LAYERS:
            raise ValueError("v3 activations must contain the registered causal layers")
        if record.anchor_names != ANCHOR_NAMES or record.example_id in values:
            raise ValueError("v3 activations have invalid causal provenance")
        values[record.example_id] = np.array(record.activations[anchor_index], copy=True)
    return values


def load_v3_training_direction_inputs(
    *,
    v3_manifest_path: str | Path,
    activation_manifest_path: str | Path,
    expected_model_id: str,
    expected_model_revision: str,
    expected_tokenizer_id: str,
    expected_tokenizer_revision: str,
    expected_chat_template_sha256: str,
) -> VerifiedV3TrainingInputs:
    """Load and bind v3 training prompts to their verified activation shard."""
    v3_manifest_path = Path(v3_manifest_path)
    try:
        v3_manifest_bytes = v3_manifest_path.read_bytes()
        v3_manifest = json.loads(v3_manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("v3 prompt manifest is unreadable") from error
    if not isinstance(v3_manifest, dict) or v3_manifest.get("study_id") != REP_V3_STUDY_ID:
        raise ValueError("v3 prompt manifest has an invalid study identity")
    prompt_name = v3_manifest.get("prompts_file")
    if not isinstance(prompt_name, str) or Path(prompt_name).name != prompt_name:
        raise ValueError("v3 prompt manifest has an invalid prompts path")
    try:
        prompt_bytes = (v3_manifest_path.parent / prompt_name).read_bytes()
        prompts = tuple(
            ReplicationPromptV3(**json.loads(line))
            for line in prompt_bytes.splitlines()
            if line
        )
    except (OSError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("v3 prompts are unreadable or noncanonical") from error
    if _sha256_bytes(prompt_bytes) != v3_manifest.get("prompts_sha256"):
        raise ValueError("v3 prompt file hash mismatch")
    if v3_manifest.get("tokenizer_id") != expected_tokenizer_id:
        raise ValueError("v3 tokenizer ID does not match the expected tokenizer ID")
    train = tuple(row for row in prompts if row.split == "representation_train")
    if len(train) != 128 or len({row.example_id for row in train}) != len(train):
        raise ValueError("v3 representation_train prompt identities are invalid")

    activation_manifest_path = Path(activation_manifest_path)
    try:
        activation_manifest_bytes = activation_manifest_path.read_bytes()
        activation_manifest = json.loads(activation_manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("v3 activation manifest is unreadable") from error
    if not isinstance(activation_manifest, dict):
        raise ValueError("v3 activation manifest has an invalid schema")
    npz_name = activation_manifest.get("npz_file")
    if not isinstance(npz_name, str):
        raise ValueError("v3 activation manifest has an invalid NPZ path")
    shard = resume_activation_shard(activation_manifest_path.parent / npz_name)
    records = load_activation_records(activation_manifest_path)
    if (
        shard.manifest_path != activation_manifest_path.absolute()
        or shard.row_count != len(train)
    ):
        raise ValueError("v3 activation manifest does not match representation_train")
    if {row.example_id for row in records} != {row.example_id for row in train}:
        raise ValueError("v3 activation identities do not match representation_train")

    expected_values = (
        expected_model_id,
        expected_model_revision,
        expected_tokenizer_id,
        expected_tokenizer_revision,
        expected_chat_template_sha256,
    )
    labels = (
        "model ID",
        "model revision",
        "tokenizer ID",
        "tokenizer revision",
        "chat template hash",
    )
    prompts_by_id = {row.example_id: row for row in train}
    for record in records:
        actual_values = (
            record.model_id,
            record.model_revision,
            record.anchors.tokenizer_id,
            record.anchors.tokenizer_revision,
            record.anchors.chat_template_sha256,
        )
        if actual_values != expected_values:
            mismatch = next(
                label
                for label, actual, expected in zip(labels, actual_values, expected_values)
                if actual != expected
            )
            raise ValueError(f"v3 activation {mismatch} does not match expected provenance")
        prompt = prompts_by_id[record.example_id]
        if (
            record.anchors.rendered_prompt_sha256 != prompt.rendered_prompt_sha256
            or record.anchors.input_ids != prompt.rendered_token_ids
        ):
            raise ValueError("v3 activation prompt identity does not match the v3 prompt")
    source = V3DirectionProvenance(
        v3_prompts_sha256=v3_manifest["prompts_sha256"],
        v3_manifest_sha256=_sha256_bytes(v3_manifest_bytes),
        activation_manifest_sha256=_sha256_bytes(activation_manifest_bytes),
        activation_npz_sha256=shard.npz_sha256,
        activation_index_sha256=shard.index_sha256,
        activation_request_sha256=shard.request_sha256,
        model_id=expected_model_id,
        model_revision=expected_model_revision,
        tokenizer_id=expected_tokenizer_id,
        tokenizer_revision=expected_tokenizer_revision,
        chat_template_sha256=expected_chat_template_sha256,
    )
    return VerifiedV3TrainingInputs(prompts=train, records=records, source=source)


def fit_train_only_directions(source: VerifiedV3TrainingInputs) -> DirectionBundle:
    """Fit directions only from verified v3 training inputs."""
    if not isinstance(source, VerifiedV3TrainingInputs):
        raise ValueError("direction fitting requires verified v3 training inputs")
    return _fit_train_only_directions_from_arrays(
        source.prompts,
        _activations_at_user_prompt_end(source.records),
        source.source,
    )


def _fit_train_only_directions_from_arrays(
    v3_prompts: Sequence[ReplicationPromptV3],
    activations: Mapping[str, np.ndarray],
    source: V3DirectionProvenance,
) -> DirectionBundle:
    train = tuple(row for row in v3_prompts if row.split == "representation_train")
    if not train:
        raise ValueError("directions require representation_train rows only")
    if any(row.split != "representation_train" for row in train):
        raise ValueError("directions require representation_train rows only")
    by_unit: dict[str, list[Any]] = {}
    for row in train:
        if row.answerability not in CAUSAL_ANSWERABILITY:
            raise ValueError("direction labels must use v3 answerability labels")
        if row.exposure not in CAUSAL_EXPOSURES:
            raise ValueError("direction labels must use v3 exposure labels")
        by_unit.setdefault(str(row.entity_unit_id), []).append(row)
    expected_cells = Counter(
        (exposure, answerability)
        for exposure in CAUSAL_EXPOSURES
        for answerability in CAUSAL_ANSWERABILITY
    )
    if not all(
        Counter((row.exposure, row.answerability) for row in rows) == expected_cells
        for rows in by_unit.values()
    ):
        raise ValueError("direction fitting requires complete 2x2 representation_train units")
    expected_ids = {str(row.example_id) for row in train}
    if set(activations) != expected_ids:
        raise ValueError("direction activations must match representation_train rows only")
    normalized: dict[str, np.ndarray] = {}
    hidden_size: int | None = None
    for example_id, value in activations.items():
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != len(CAUSAL_VALIDATION_LAYERS):
            raise ValueError("direction activations must be [layer, hidden]")
        if not np.isfinite(array).all():
            raise ValueError("direction activations must be finite")
        if hidden_size is None:
            hidden_size = int(array.shape[1])
        if array.shape[1] != hidden_size:
            raise ValueError("direction activations must share a hidden size")
        normalized[example_id] = array
    directions = []
    for layer_index, layer_id in enumerate(CAUSAL_VALIDATION_LAYERS):
        paired_deltas = []
        for unit_id in sorted(by_unit):
            rows = by_unit[unit_id]
            for exposure in CAUSAL_EXPOSURES:
                cells = {row.answerability: row for row in rows if row.exposure == exposure}
                paired_deltas.append(
                    normalized[cells["target_bound"].example_id][layer_index]
                    - normalized[cells["target_unbound"].example_id][layer_index]
                )
        raw = np.mean(np.stack(paired_deltas), axis=0)
        norm = float(np.linalg.norm(raw))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("direction mean must have a nonzero finite norm")
        vector = raw / norm
        natural_scale = float(np.median([float(delta @ vector) for delta in paired_deltas]))
        if natural_scale <= 0.0 or not np.isfinite(natural_scale):
            raise ValueError("direction natural scale must be positive")
        record = {
            "layer_id": layer_id,
            "vector": vector.tolist(),
            "natural_scale": natural_scale,
            "training_unit_count": len(by_unit),
            "source_split": "representation_train",
        }
        directions.append(
            AnswerabilityDirection(
                vector=vector,
                direction_sha256=_sha256(record),
                **{key: value for key, value in record.items() if key != "vector"},
            )
        )
    provisional = {
        "directions": [_direction_record(direction) for direction in directions],
        "source": _v3_source_record(source),
    }
    return DirectionBundle(
        directions=tuple(directions),
        source=source,
        bundle_sha256=_sha256(provisional),
    )


def _direction_record(
    direction: AnswerabilityDirection, *, include_hash: bool = True
) -> dict[str, Any]:
    record = {
        "layer_id": direction.layer_id,
        "vector": direction.vector.tolist(),
        "natural_scale": direction.natural_scale,
        "training_unit_count": direction.training_unit_count,
        "source_split": direction.source_split,
    }
    if include_hash:
        record["direction_sha256"] = direction.direction_sha256
    return record


def _direction_bundle_record(bundle: DirectionBundle, *, include_hash: bool = True) -> dict[str, Any]:
    record = {
        "directions": [_direction_record(direction) for direction in bundle.directions],
        "source": _v3_source_record(bundle.source),
    }
    if include_hash:
        record["bundle_sha256"] = bundle.bundle_sha256
    return record


def _v3_source_record(source: V3DirectionProvenance) -> dict[str, str]:
    return {
        "v3_prompts_sha256": source.v3_prompts_sha256,
        "v3_manifest_sha256": source.v3_manifest_sha256,
        "activation_manifest_sha256": source.activation_manifest_sha256,
        "activation_npz_sha256": source.activation_npz_sha256,
        "activation_index_sha256": source.activation_index_sha256,
        "activation_request_sha256": source.activation_request_sha256,
        "model_id": source.model_id,
        "model_revision": source.model_revision,
        "tokenizer_id": source.tokenizer_id,
        "tokenizer_revision": source.tokenizer_revision,
        "chat_template_sha256": source.chat_template_sha256,
    }


def write_direction_bundle(bundle: DirectionBundle, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(_direction_bundle_record(bundle)) + b"\n")
    return path


def verify_direction_bundle(path: str | Path) -> DirectionBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        directions = tuple(AnswerabilityDirection(**record) for record in payload["directions"])
        return DirectionBundle(
            directions=directions,
            source=V3DirectionProvenance(**payload["source"]),
            bundle_sha256=payload["bundle_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("direction bundle is not canonical") from error


def select_causal_intervention(
    candidates: Sequence[ValidationCandidate],
    corpus: CausalCorpus,
    expected_provenance: CausalExpectedProvenance,
) -> ValidationSelection:
    if not corpus.audit.passed:
        raise ValueError("selection requires an audited causal corpus")
    if not isinstance(expected_provenance, CausalExpectedProvenance):
        raise ValueError("selection requires typed expected provenance")
    if corpus.manifest_sha256 != expected_provenance.corpus_sha256:
        raise ValueError("selection corpus hash does not match expected provenance")
    prepared = tuple(candidates)
    if not prepared:
        raise ValueError("selection requires validation candidates")
    fixed_grid = {
        (layer_id, multiplier)
        for layer_id in CAUSAL_VALIDATION_LAYERS
        for multiplier in CAUSAL_VALIDATION_MULTIPLIERS
    }
    if {(row.layer_id, row.multiplier) for row in prepared} != fixed_grid:
        raise ValueError("selection candidates must contain the complete fixed grid")
    if len(prepared) != len(fixed_grid):
        raise ValueError("selection candidates must contain the complete fixed grid")
    expected_units = {
        row.entity_unit_id for row in corpus.prompts if row.split == "causal_validation"
    }
    for candidate in prepared:
        if {unit for unit, _ in candidate.unit_effects} != expected_units:
            raise ValueError("selection candidates must contain all causal validation units")
        if candidate.corpus_sha256 != expected_provenance.corpus_sha256:
            raise ValueError("selection candidate corpus hash does not match expected provenance")
        if candidate.direction_bundle_sha256 != expected_provenance.direction_bundle_sha256:
            raise ValueError("selection candidate direction bundle hash does not match expected provenance")
        if candidate.model_sha256 != expected_provenance.model_sha256:
            raise ValueError("selection candidate model hash does not match expected provenance")
        if candidate.tokenizer_sha256 != expected_provenance.tokenizer_sha256:
            raise ValueError("selection candidate tokenizer hash does not match expected provenance")
        if candidate.direction_sha256 != expected_provenance.direction_hashes[candidate.layer_id]:
            raise ValueError("selection candidate direction hash does not match expected provenance")
    eligible = [
        candidate
        for candidate in prepared
        if candidate.invalid_output_rate <= 0.05 and candidate.bound_accuracy_drop <= 0.05
    ]
    if not eligible:
        raise ValueError("no validation candidate passes registered gates")
    selected = min(
        eligible,
        key=lambda row: (-row.mean_bidirectional_effect, row.multiplier, row.layer_id),
    )
    record = {
        "layer_id": selected.layer_id,
        "multiplier": selected.multiplier,
        "mean_bidirectional_effect": selected.mean_bidirectional_effect,
        "direction_sha256": selected.direction_sha256,
        "corpus_sha256": selected.corpus_sha256,
        "direction_bundle_sha256": selected.direction_bundle_sha256,
        "model_sha256": selected.model_sha256,
        "tokenizer_sha256": selected.tokenizer_sha256,
    }
    return ValidationSelection(selection_sha256=_sha256(record), **record)


def _selection_record(selection: ValidationSelection, *, include_hash: bool = True) -> dict[str, Any]:
    record = {
        "layer_id": selection.layer_id,
        "multiplier": selection.multiplier,
        "mean_bidirectional_effect": selection.mean_bidirectional_effect,
        "direction_sha256": selection.direction_sha256,
        "corpus_sha256": selection.corpus_sha256,
        "direction_bundle_sha256": selection.direction_bundle_sha256,
        "model_sha256": selection.model_sha256,
        "tokenizer_sha256": selection.tokenizer_sha256,
        "corpus_sha256": selection.corpus_sha256,
    }
    if include_hash:
        record["selection_sha256"] = selection.selection_sha256
    return record


def write_selection_manifest(selection: ValidationSelection, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(_selection_record(selection)) + b"\n")
    return path


def verify_selection_manifest(path: str | Path) -> ValidationSelection:
    try:
        return ValidationSelection(**json.loads(Path(path).read_text(encoding="utf-8")))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("selection manifest is not canonical") from error
