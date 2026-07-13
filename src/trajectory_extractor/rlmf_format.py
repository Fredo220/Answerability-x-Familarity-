from __future__ import annotations

import hashlib
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from trajectory_extractor.rlmf_types import ParsedRLMFOutput


_BUCKET_TEXT = frozenset(f"{value / 10:.1f}" for value in range(11))
_ANSWER_SCHEMA = re.compile(
    r"\A\s*<sentence>(?P<answer>[^<]+)</sentence>\s*"
    r"<confidence>(?P<confidence>[^<]+)</confidence>\s*\Z",
    re.DOTALL,
)
_METASCORE_SCHEMA = re.compile(
    r"\A\s*<metascore>(?P<metascore>[^<]+)</metascore>\s*\Z", re.DOTALL
)
_ARMS = ("standard_grpo", "rlmf")
_JUDGMENT_TYPES = ("correctness", "equivalence")
_LABELS = frozenset({"correct", "incorrect", "ambiguous"})
_FINAL_LABELS = frozenset({"correct", "incorrect", "ambiguous"})
_PHASE_SIZES = {"development": (200,), "locked": (400,), "test": (1000, 1250, 1500, 1750, 2000)}


@dataclass(frozen=True)
class ParsedMetacognitiveOutput:
    metascore: float | None = None
    valid_format: bool = False

    def __post_init__(self) -> None:
        if self.metascore is not None and self.metascore not in {
            value / 10 for value in range(11)
        }:
            raise ValueError("metascore must use a registered confidence value")
        if self.valid_format != (self.metascore is not None):
            raise ValueError("valid_format must agree with metascore presence")


@dataclass(frozen=True)
class AuditRow:
    audit_id: str
    source_id: str
    phase: str
    arm: str
    judgment_type: str
    proxy_label: bool
    question: str
    answer: str
    comparison_answer: str
    reference_answer: str
    rater_a: str | None = None
    rater_b: str | None = None
    adjudicated_label: str | None = None

    def __post_init__(self) -> None:
        if not self.audit_id or not self.source_id:
            raise ValueError("audit identifiers must be non-empty")
        if self.phase not in _PHASE_SIZES:
            raise ValueError("audit phase is not registered")
        if self.arm not in _ARMS:
            raise ValueError("audit arm is not registered")
        if self.judgment_type not in _JUDGMENT_TYPES:
            raise ValueError("audit judgment_type is not registered")
        if type(self.proxy_label) is not bool:
            raise ValueError("proxy_label must be boolean")
        for field in ("question", "answer", "comparison_answer", "reference_answer"):
            if not isinstance(getattr(self, field), str):
                raise ValueError(f"{field} must be a string")
        for field in ("rater_a", "rater_b", "adjudicated_label"):
            value = getattr(self, field)
            if value is not None and value not in _LABELS:
                raise ValueError(f"{field} must be a registered audit label")

    def rater_payload(self) -> dict[str, str]:
        """The only record exposed to independent raters."""
        return {
            "audit_id": self.audit_id,
            "judgment_type": self.judgment_type,
            "question": self.question,
            "answer": self.answer,
            "comparison_answer": self.comparison_answer,
            "reference_answer": self.reference_answer,
        }

    def to_ledger_record(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "source_id": self.source_id,
            "phase": self.phase,
            "arm": self.arm,
            "judgment_type": self.judgment_type,
            "proxy_label": self.proxy_label,
            "question": self.question,
            "answer": self.answer,
            "comparison_answer": self.comparison_answer,
            "reference_answer": self.reference_answer,
            "rater_a": self.rater_a,
            "rater_b": self.rater_b,
            "adjudicated_label": self.adjudicated_label,
        }

    @classmethod
    def from_ledger_record(cls, value: Mapping[str, Any]) -> "AuditRow":
        if not isinstance(value, Mapping):
            raise ValueError("audit ledger row must be a mapping")
        return cls(**{field: value.get(field) for field in cls.__dataclass_fields__})

    @property
    def final_label(self) -> str:
        if self.rater_a is None or self.rater_b is None:
            raise ValueError("both independent rater labels are required")
        if self.rater_a != self.rater_b:
            if self.adjudicated_label is None:
                raise ValueError("rater disagreements must be adjudicated before scoring")
            return self.adjudicated_label
        if self.adjudicated_label is not None and self.adjudicated_label != self.rater_a:
            raise ValueError("adjudication must not override an agreement")
        return self.rater_a


@dataclass(frozen=True)
class Interval:
    lower: float
    estimate: float
    upper: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.lower, self.estimate, self.upper)):
            raise ValueError("interval values must be finite")
        if not self.lower <= self.upper:
            raise ValueError("interval bounds are invalid")


@dataclass(frozen=True)
class JudgeAuditDecision:
    phase: str
    status: str
    passed: bool | None
    kappa: float
    ambiguous_fraction: float
    sensitivity: Mapping[str, float]
    specificity: Mapping[str, float]
    proxy_revision_permitted: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "passed": self.passed,
            "kappa": self.kappa,
            "ambiguous_fraction": self.ambiguous_fraction,
            "sensitivity": dict(self.sensitivity),
            "specificity": dict(self.specificity),
            "proxy_revision_permitted": self.proxy_revision_permitted,
        }


def parse_rlmf_output(text: str) -> ParsedRLMFOutput:
    if not isinstance(text, str):
        return ParsedRLMFOutput(answer="")
    match = _ANSWER_SCHEMA.fullmatch(text)
    if match is None:
        return ParsedRLMFOutput(answer="")
    answer = match.group("answer").strip()
    confidence = _registered_bucket(match.group("confidence"))
    if not answer or confidence is None:
        return ParsedRLMFOutput(answer="")
    return ParsedRLMFOutput(answer=answer, confidence=confidence, valid_format=True)


def parse_metascore_output(text: str) -> ParsedMetacognitiveOutput:
    if not isinstance(text, str):
        return ParsedMetacognitiveOutput()
    match = _METASCORE_SCHEMA.fullmatch(text)
    if match is None:
        return ParsedMetacognitiveOutput()
    metascore = _registered_bucket(match.group("metascore"))
    if metascore is None:
        return ParsedMetacognitiveOutput()
    return ParsedMetacognitiveOutput(metascore=metascore, valid_format=True)


def normalized_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    value = unicodedata.normalize("NFKC", text).casefold()
    value = "".join(
        "" if unicodedata.category(character).startswith(("P", "S")) else character
        for character in value
    )
    value = " ".join(value.split())
    return re.sub(r"\A(?:a|an|the)\s+", "", value)


def alias_exact_match(answer: str, aliases: Sequence[str]) -> bool:
    normalized = normalized_answer(answer)
    if not normalized:
        return False
    return any(normalized == normalized_answer(alias) for alias in aliases)


def completion_equivalent(left: str, right: str, gold_aliases: Sequence[str]) -> bool:
    left_normalized = normalized_answer(left)
    right_normalized = normalized_answer(right)
    if not left_normalized or not right_normalized:
        return False
    return left_normalized == right_normalized or (
        alias_exact_match(left, gold_aliases) and alias_exact_match(right, gold_aliases)
    )


def build_judge_audit_sample(
    completions: Sequence[Any], *, phase: str, size: int, seed: int
) -> tuple[AuditRow, ...]:
    quotas = _stratum_quotas(phase, size)
    if type(seed) is not int:
        raise ValueError("audit seed must be an integer")
    candidates = [_candidate_row(value, phase, index) for index, value in enumerate(completions)]
    selected: list[AuditRow] = []
    for stratum, quota in quotas.items():
        available = [row for row in candidates if _stratum(row) == stratum]
        if len(available) < quota:
            raise ValueError(f"insufficient candidates for audit stratum {stratum}")
        selected.extend(sorted(available, key=lambda row: _rank(seed, row.source_id))[:quota])
    selected.sort(key=lambda row: _rank(seed, f"sample:{row.source_id}"))
    return tuple(
        AuditRow(
            audit_id=f"audit-{index:04d}",
            source_id=row.source_id,
            phase=row.phase,
            arm=row.arm,
            judgment_type=row.judgment_type,
            proxy_label=row.proxy_label,
            question=row.question,
            answer=row.answer,
            comparison_answer=row.comparison_answer,
            reference_answer=row.reference_answer,
        )
        for index, row in enumerate(selected, start=1)
    )


def score_blinded_judge_audit(rows: Sequence[AuditRow]) -> JudgeAuditDecision:
    rows = tuple(rows)
    if not rows:
        raise ValueError("audit must contain rows")
    phase = rows[0].phase
    if any(row.phase != phase for row in rows):
        raise ValueError("audit rows must use one phase")
    expected = _stratum_quotas(phase, len(rows))
    actual = {stratum: sum(_stratum(row) == stratum for row in rows) for stratum in expected}
    if actual != expected:
        raise ValueError("audit rows do not match the registered stratum balance")
    if len({row.audit_id for row in rows}) != len(rows):
        raise ValueError("audit IDs must be unique")
    final_labels = [row.final_label for row in rows]
    ambiguous = sum(label == "ambiguous" for label in final_labels) / len(rows)
    kappa = _cohen_kappa(rows)
    sensitivity, specificity = _classification_metrics(rows)
    if phase == "development":
        return JudgeAuditDecision(
            phase=phase,
            status="development_review",
            passed=None,
            kappa=kappa,
            ambiguous_fraction=ambiguous,
            sensitivity=sensitivity,
            specificity=specificity,
            proxy_revision_permitted=True,
        )
    if phase == "test":
        return JudgeAuditDecision(
            phase=phase,
            status="measurement_bias_only",
            passed=None,
            kappa=kappa,
            ambiguous_fraction=ambiguous,
            sensitivity=sensitivity,
            specificity=specificity,
            proxy_revision_permitted=False,
        )
    passed = (
        kappa >= 0.80
        and ambiguous <= 0.05
        and all(value >= 0.90 for value in sensitivity.values())
        and all(value >= 0.90 for value in specificity.values())
    )
    return JudgeAuditDecision(
        phase=phase,
        status="passed" if passed else "failed",
        passed=passed,
        kappa=kappa,
        ambiguous_fraction=ambiguous,
        sensitivity=sensitivity,
        specificity=specificity,
        proxy_revision_permitted=False,
    )


def bound_differential_judge_bias(rows: Sequence[AuditRow], *, replicates: int) -> Interval:
    rows = tuple(rows)
    if not rows or any(row.phase != "test" for row in rows):
        raise ValueError("differential judge bias requires a completed test audit")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be positive")
    _ = score_blinded_judge_audit(rows)
    usable = tuple(row for row in rows if row.final_label != "ambiguous")
    if not usable:
        raise ValueError("test audit contains no non-ambiguous labels")
    estimate = _maximum_arm_differential_bias(usable)
    rng = random.Random(_audit_rng_seed(usable))
    samples = sorted(
        _maximum_arm_differential_bias(_stratified_resample(usable, rng))
        for _ in range(replicates)
    )
    upper_index = max(0, math.ceil(0.95 * len(samples)) - 1)
    return Interval(lower=0.0, estimate=estimate, upper=samples[upper_index])


def _registered_bucket(value: str) -> float | None:
    candidate = value.strip()
    if candidate not in _BUCKET_TEXT:
        return None
    parsed = float(candidate)
    return parsed if math.isfinite(parsed) else None


def _candidate_row(value: Any, phase: str, index: int) -> AuditRow:
    def get(name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    source_id = get("candidate_id", get("source_id", f"candidate-{index}"))
    judgment_type = get("judgment_type")
    answer = get("answer", "")
    comparison_answer = get("comparison_answer", "")
    aliases = get("gold_aliases", ())
    proxy_label = get("proxy_label")
    if proxy_label is None:
        if judgment_type == "correctness":
            proxy_label = alias_exact_match(answer, aliases)
        elif judgment_type == "equivalence":
            proxy_label = completion_equivalent(answer, comparison_answer, aliases)
    if phase == "development" and get("split") not in {"pre_sft", "rl_train"}:
        raise ValueError("development audit requires pre_sft or rl_train completions")
    return AuditRow(
        audit_id=f"candidate-{index}",
        source_id=str(source_id),
        phase=phase,
        arm=get("arm"),
        judgment_type=judgment_type,
        proxy_label=proxy_label,
        question=get("question", ""),
        answer=answer,
        comparison_answer=comparison_answer,
        reference_answer=get("reference_answer", ""),
    )


def _stratum_quotas(phase: str, size: int) -> dict[tuple[str, str, bool], int]:
    if phase not in _PHASE_SIZES or size not in _PHASE_SIZES[phase]:
        raise ValueError("audit phase and size are not registered")
    strata = [(arm, judgment_type, label) for arm in _ARMS for judgment_type in _JUDGMENT_TYPES for label in (False, True)]
    base, remainder = divmod(size, len(strata))
    return {stratum: base + (index < remainder) for index, stratum in enumerate(strata)}


def _stratum(row: AuditRow) -> tuple[str, str, bool]:
    return row.arm, row.judgment_type, row.proxy_label


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _cohen_kappa(rows: Sequence[AuditRow]) -> float:
    labels_a = [row.rater_a for row in rows]
    labels_b = [row.rater_b for row in rows]
    if None in labels_a or None in labels_b:
        raise ValueError("both independent rater labels are required")
    observed = sum(left == right for left, right in zip(labels_a, labels_b)) / len(rows)
    expected = sum(
        labels_a.count(label) / len(rows) * labels_b.count(label) / len(rows)
        for label in _LABELS
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _classification_metrics(rows: Sequence[AuditRow]) -> tuple[dict[str, float], dict[str, float]]:
    sensitivity: dict[str, float] = {}
    specificity: dict[str, float] = {}
    for arm in _ARMS:
        for judgment_type in _JUDGMENT_TYPES:
            relevant = [
                row
                for row in rows
                if row.arm == arm and row.judgment_type == judgment_type and row.final_label != "ambiguous"
            ]
            human_positive = [row for row in relevant if row.final_label == "correct"]
            human_negative = [row for row in relevant if row.final_label == "incorrect"]
            if not human_positive or not human_negative:
                raise ValueError("each audit arm and judgment type needs both adjudicated classes")
            key = f"{arm}:{judgment_type}"
            sensitivity[key] = sum(row.proxy_label for row in human_positive) / len(human_positive)
            specificity[key] = sum(not row.proxy_label for row in human_negative) / len(human_negative)
    return sensitivity, specificity


def _maximum_arm_differential_bias(rows: Sequence[AuditRow]) -> float:
    differences = []
    for judgment_type in _JUDGMENT_TYPES:
        biases = {}
        for arm in _ARMS:
            relevant = [
                row for row in rows if row.arm == arm and row.judgment_type == judgment_type
            ]
            if not relevant:
                raise ValueError("test audit must include every arm and judgment type")
            biases[arm] = fmean(int(row.proxy_label) - int(row.final_label == "correct") for row in relevant)
        differences.append(abs(biases["rlmf"] - biases["standard_grpo"]))
    return max(differences)


def _stratified_resample(rows: Sequence[AuditRow], rng: random.Random) -> tuple[AuditRow, ...]:
    result: list[AuditRow] = []
    for arm in _ARMS:
        for judgment_type in _JUDGMENT_TYPES:
            for proxy_label in (False, True):
                stratum = [
                    row
                    for row in rows
                    if (row.arm, row.judgment_type, row.proxy_label)
                    == (arm, judgment_type, proxy_label)
                ]
                if not stratum:
                    raise ValueError("test audit must retain every registered stratum")
                result.extend(rng.choice(stratum) for _ in stratum)
    return tuple(result)


def _audit_rng_seed(rows: Sequence[AuditRow]) -> int:
    payload = "\n".join(
        f"{row.audit_id}:{row.arm}:{row.judgment_type}:{row.proxy_label}:{row.final_label}"
        for row in sorted(rows, key=lambda row: row.audit_id)
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
