from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trajectory_extractor.rlmf_types import ParsedRLMFOutput


PARSER_VERSION = "rlmf-output-parser-v1"
NORMALIZATION_VERSION = "rlmf-answer-normalization-v1"
REGISTERED_AUDIT_SEED = 20260713

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
_DEVELOPMENT_SPLITS = ("pre_sft", "rl_train")
_JUDGMENT_TYPES = ("correctness", "equivalence")
_LABELS = frozenset({"correct", "incorrect", "ambiguous"})
_PHASE_SIZES = {
    "development": (200,),
    "locked": (400,),
    "test": (1000, 1250, 1500, 1750, 2000),
}
_ARTICLES = frozenset({"a", "an", "the"})


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
    example_id: str
    phase: str
    split: str
    arm: str | None
    judgment_type: str
    proxy_label: bool
    question: str
    answer: str
    comparison_answer: str
    reference_answer: str
    aliases: tuple[str, ...]
    rater_a: str | None = None
    rater_b: str | None = None
    adjudicated_label: str | None = None

    def __post_init__(self) -> None:
        for field in ("audit_id", "source_id", "example_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("audit identifiers must be non-empty strings")
        object.__setattr__(self, "aliases", _validated_aliases(self.aliases))
        if self.phase not in _PHASE_SIZES:
            raise ValueError("audit phase is not registered")
        _validate_phase_assignment(self.phase, self.split, self.arm)
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
        """Return the complete and only payload exposed to independent raters."""
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
            "example_id": self.example_id,
            "phase": self.phase,
            "split": self.split,
            "arm": self.arm,
            "judgment_type": self.judgment_type,
            "proxy_label": self.proxy_label,
            "question": self.question,
            "answer": self.answer,
            "comparison_answer": self.comparison_answer,
            "reference_answer": self.reference_answer,
            "aliases": list(self.aliases),
            "rater_a": self.rater_a,
            "rater_b": self.rater_b,
            "adjudicated_label": self.adjudicated_label,
        }

    @classmethod
    def from_ledger_record(cls, value: Mapping[str, Any]) -> "AuditRow":
        if not isinstance(value, Mapping):
            raise ValueError("audit ledger row must be a mapping")
        fields = cls.__dataclass_fields__
        if set(value) != set(fields):
            raise ValueError("audit ledger row has an invalid schema")
        return cls(**{field: value[field] for field in fields})

    @property
    def final_label(self) -> str:
        if self.rater_a is None or self.rater_b is None:
            raise ValueError("both independent rater labels are required")
        if self.rater_a != self.rater_b:
            if self.adjudicated_label is None:
                raise ValueError("rater disagreements must be adjudicated before scoring")
            return self.adjudicated_label
        if self.adjudicated_label is not None:
            raise ValueError("adjudication may only address rater disagreements")
        return self.rater_a


@dataclass(frozen=True)
class Interval:
    lower: float
    estimate: float
    upper: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) for value in (self.lower, self.estimate, self.upper)
        ):
            raise ValueError("interval values must be finite")
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("interval bounds are invalid")

    def to_record(self) -> dict[str, float]:
        return {
            "lower": self.lower,
            "estimate": self.estimate,
            "upper": self.upper,
        }


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
        "" if unicodedata.category(character).startswith("P") else character
        for character in value
    )
    return " ".join(token for token in value.split() if token not in _ARTICLES)


def alias_exact_match(answer: str, aliases: Sequence[str]) -> bool:
    frozen_aliases = _validated_aliases(aliases)
    normalized = normalized_answer(answer)
    if not normalized:
        return False
    return any(normalized == normalized_answer(alias) for alias in frozen_aliases)


def completion_equivalent(
    left: str, right: str, gold_aliases: Sequence[str]
) -> bool:
    frozen_aliases = _validated_aliases(gold_aliases)
    left_normalized = normalized_answer(left)
    right_normalized = normalized_answer(right)
    if not left_normalized or not right_normalized:
        return False
    return left_normalized == right_normalized or (
        alias_exact_match(left, frozen_aliases)
        and alias_exact_match(right, frozen_aliases)
    )


def build_judge_audit_sample(
    completions: Sequence[Any], *, phase: str, size: int, seed: int
) -> tuple[AuditRow, ...]:
    quotas = _stratum_quotas(phase, size)
    if type(seed) is not int:
        raise ValueError("audit seed must be an integer")
    candidates = tuple(_candidate_row(value, phase) for value in completions)
    source_ids = [row.source_id for row in candidates]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("candidate source IDs must be unique")

    selected: list[AuditRow] = []
    for stratum, quota in quotas.items():
        available = [row for row in candidates if _stratum(row) == stratum]
        if len(available) < quota:
            raise ValueError(f"insufficient candidates for audit stratum {stratum}")
        selected.extend(
            sorted(available, key=lambda row: (_rank(seed, row.source_id), row.source_id))[
                :quota
            ]
        )
    selected.sort(key=lambda row: (_rank(seed, f"sample:{row.source_id}"), row.source_id))
    result = tuple(
        AuditRow(
            audit_id="audit-"
            + hashlib.sha256(f"{phase}:{row.source_id}".encode("utf-8")).hexdigest(),
            source_id=row.source_id,
            example_id=row.example_id,
            phase=row.phase,
            split=row.split,
            arm=row.arm,
            judgment_type=row.judgment_type,
            proxy_label=row.proxy_label,
            question=row.question,
            answer=row.answer,
            comparison_answer=row.comparison_answer,
            reference_answer=row.reference_answer,
            aliases=row.aliases,
        )
        for row in selected
    )
    if len({row.audit_id for row in result}) != len(result):
        raise ValueError("stable audit ID collision")
    return result


def score_blinded_judge_audit(rows: Sequence[AuditRow]) -> JudgeAuditDecision:
    rows = tuple(rows)
    if not rows:
        raise ValueError("audit must contain rows")
    phase = rows[0].phase
    if any(row.phase != phase for row in rows):
        raise ValueError("audit rows must use one phase")
    expected = _stratum_quotas(phase, len(rows))
    actual = {
        stratum: sum(_stratum(row) == stratum for row in rows) for stratum in expected
    }
    if actual != expected:
        raise ValueError("audit rows do not match the registered stratum balance")
    if len({row.audit_id for row in rows}) != len(rows):
        raise ValueError("audit IDs must be unique")
    if len({row.source_id for row in rows}) != len(rows):
        raise ValueError("audit source IDs must be unique")

    final_labels = [row.final_label for row in rows]
    ambiguous = sum(label == "ambiguous" for label in final_labels) / len(rows)
    kappa = _cohen_kappa(rows)
    sensitivity, specificity = _classification_metrics(rows, phase)
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


def estimate_arm_confusion_uncertainty(
    rows: Sequence[AuditRow],
) -> dict[str, dict[str, dict[str, float]]]:
    rows = tuple(rows)
    if not rows or any(row.phase != "test" for row in rows):
        raise ValueError("arm confusion uncertainty requires a completed test audit")
    decision = score_blinded_judge_audit(rows)
    if decision.passed is not True:
        raise ValueError("test audit reliability gates must pass before uncertainty estimation")
    result: dict[str, dict[str, dict[str, float]]] = {}
    for arm in _ARMS:
        for judgment_type in _JUDGMENT_TYPES:
            relevant = [
                row
                for row in rows
                if row.arm == arm
                and row.judgment_type == judgment_type
                and row.final_label != "ambiguous"
            ]
            positives = [row for row in relevant if row.final_label == "correct"]
            negatives = [row for row in relevant if row.final_label == "incorrect"]
            result[f"{arm}:{judgment_type}"] = {
                "sensitivity": _wilson_interval(
                    sum(row.proxy_label for row in positives), len(positives)
                ).to_record(),
                "specificity": _wilson_interval(
                    sum(not row.proxy_label for row in negatives), len(negatives)
                ).to_record(),
            }
    return result


def bound_differential_judge_bias(
    rows: Sequence[AuditRow], *, replicates: int
) -> Interval:
    del rows, replicates
    raise ValueError(
        "delta_cMFG_star judge-bias propagation requires behavioral records and is "
        "deferred to Task 5/10"
    )


def _registered_bucket(value: str) -> float | None:
    candidate = value.strip()
    if candidate not in _BUCKET_TEXT:
        return None
    parsed = float(candidate)
    return parsed if math.isfinite(parsed) else None


def _validated_aliases(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("aliases must be a non-string sequence")
    aliases = tuple(value)
    if not aliases or any(not isinstance(alias, str) for alias in aliases):
        raise ValueError("aliases must contain non-empty strings")
    if any(not normalized_answer(alias) for alias in aliases):
        raise ValueError("aliases must contain non-empty strings")
    return aliases


def _candidate_row(value: Any, phase: str) -> AuditRow:
    def get(name: str, default: Any = None) -> Any:
        return (
            value.get(name, default)
            if isinstance(value, Mapping)
            else getattr(value, name, default)
        )

    source_id = get("candidate_id", get("source_id"))
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("candidate source ID must be a non-empty string")
    example_id = get("example_id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError("candidate example ID must be a non-empty string")
    split = get("split")
    arm = get("arm")
    _validate_phase_assignment(phase, split, arm)
    judgment_type = get("judgment_type")
    answer = get("answer", "")
    comparison_answer = get("comparison_answer", "")
    aliases = _validated_aliases(get("gold_aliases", get("aliases")))
    if judgment_type == "correctness":
        recomputed_proxy = alias_exact_match(answer, aliases)
    elif judgment_type == "equivalence":
        recomputed_proxy = completion_equivalent(answer, comparison_answer, aliases)
    else:
        raise ValueError("audit judgment_type is not registered")
    supplied_proxy = get("proxy_label")
    if supplied_proxy is not None:
        if type(supplied_proxy) is not bool or supplied_proxy != recomputed_proxy:
            raise ValueError("supplied proxy_label does not match the frozen proxy judge")
    return AuditRow(
        audit_id="candidate-" + hashlib.sha256(source_id.encode("utf-8")).hexdigest(),
        source_id=source_id.strip(),
        example_id=example_id.strip(),
        phase=phase,
        split=split,
        arm=arm,
        judgment_type=judgment_type,
        proxy_label=recomputed_proxy,
        question=get("question", ""),
        answer=answer,
        comparison_answer=comparison_answer,
        reference_answer=get("reference_answer", ""),
        aliases=aliases,
    )


def _validate_phase_assignment(phase: str, split: Any, arm: Any) -> None:
    if phase == "development":
        if split not in _DEVELOPMENT_SPLITS:
            raise ValueError("development audit split must be pre_sft or rl_train")
        if arm is not None:
            raise ValueError("development audit uses shared pre-treatment material without arms")
        return
    expected_split = "validation" if phase == "locked" else "test" if phase == "test" else None
    if expected_split is None:
        raise ValueError("audit phase is not registered")
    if split != expected_split:
        raise ValueError(f"{phase} audit split must be {expected_split}")
    if arm not in _ARMS:
        raise ValueError("audit arm is not registered")


def _stratum_quotas(phase: str, size: int) -> dict[tuple[Any, str, bool], int]:
    if phase not in _PHASE_SIZES or size not in _PHASE_SIZES[phase]:
        raise ValueError("audit phase and size are not registered")
    groups: Sequence[str] = _DEVELOPMENT_SPLITS if phase == "development" else _ARMS
    strata = [
        (group, judgment_type, label)
        for group in groups
        for judgment_type in _JUDGMENT_TYPES
        for label in (False, True)
    ]
    base, remainder = divmod(size, len(strata))
    return {
        stratum: base + (index < remainder) for index, stratum in enumerate(strata)
    }


def _stratum(row: AuditRow) -> tuple[Any, str, bool]:
    group = row.split if row.phase == "development" else row.arm
    return group, row.judgment_type, row.proxy_label


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


def _classification_metrics(
    rows: Sequence[AuditRow], phase: str
) -> tuple[dict[str, float], dict[str, float]]:
    sensitivity: dict[str, float] = {}
    specificity: dict[str, float] = {}
    groups: Sequence[str] = _DEVELOPMENT_SPLITS if phase == "development" else _ARMS
    for group in groups:
        for judgment_type in _JUDGMENT_TYPES:
            relevant = [
                row
                for row in rows
                if (row.split if phase == "development" else row.arm) == group
                and row.judgment_type == judgment_type
                and row.final_label != "ambiguous"
            ]
            human_positive = [row for row in relevant if row.final_label == "correct"]
            human_negative = [row for row in relevant if row.final_label == "incorrect"]
            key = f"{group}:{judgment_type}"
            sensitivity[key] = (
                sum(row.proxy_label for row in human_positive) / len(human_positive)
                if human_positive
                else 0.0
            )
            specificity[key] = (
                sum(not row.proxy_label for row in human_negative) / len(human_negative)
                if human_negative
                else 0.0
            )
    return sensitivity, specificity


def _wilson_interval(successes: int, total: int) -> Interval:
    if total < 1:
        return Interval(lower=0.0, estimate=0.0, upper=1.0)
    estimate = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return Interval(
        lower=min(estimate, max(0.0, center - radius)),
        estimate=estimate,
        upper=max(estimate, min(1.0, center + radius)),
    )
