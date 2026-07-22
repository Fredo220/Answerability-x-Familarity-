"""Provenance-bound F2A feature construction.

This bridge deliberately has no generation path.  Candidate likelihoods are
teacher-forced over the complete rendered prompt plus an exact registered suffix,
then bound to the activation extraction record before a ``ProbeRow`` is built.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from trajectory_extractor.fa_activations import ActivationRecord
from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_data import FAExample
from trajectory_extractor.fa_probes import OUTPUT_CONTROL_SCHEMA_SHA256, ProbeRow


UNKNOWN_SUFFIX = "UNKNOWN"
OUTPUT_FEATURE_DIM = 11
OUTPUT_FEATURE_NAMES = (
    "target_sequence_logp",
    "unknown_sequence_logp",
    "target_minus_unknown_logp",
    "maximum_sequence_logp",
    "candidate_logsumexp",
    "normalized_target_probability",
    "normalized_unknown_probability",
    "binary_candidate_entropy",
    "absolute_normalized_probability_margin",
    "signed_probability_margin",
    "maximum_candidate_confidence",
)

TARGET_SEQUENCE_LOGP = 0
UNKNOWN_SEQUENCE_LOGP = 1
TARGET_MINUS_UNKNOWN_LOGP = 2
MAXIMUM_SEQUENCE_LOGP = 3
CANDIDATE_LOGSUMEXP = 4
NORMALIZED_TARGET_PROBABILITY = 5
NORMALIZED_UNKNOWN_PROBABILITY = 6
BINARY_CANDIDATE_ENTROPY = 7
ABSOLUTE_NORMALIZED_PROBABILITY_MARGIN = 8
SIGNED_PROBABILITY_MARGIN = 9
MAXIMUM_CANDIDATE_CONFIDENCE = 10

_NUMERIC_SURFACE_FEATURE_NAMES = (
    "target_character_count",
    "distractor_character_count",
    "prompt_character_count",
    "target_word_count",
    "distractor_word_count",
    "prompt_word_count",
    "target_uppercase_character_count",
    "distractor_uppercase_character_count",
    "prompt_uppercase_character_count",
    "rendered_prompt_token_count",
    "target_is_first",
    "target_is_second",
    "code_is_first",
    "code_is_second",
    "code_is_absent",
)
REGISTERED_ENTITY_DOMAINS = (
    "person",
    "place",
    "organization",
    "creative_work",
)
REGISTERED_PROMPT_TEMPLATES = (
    "train_registry_direct",
    "train_registry_possessive",
    "train_registry_query",
    "validation_archive_direct",
    "validation_archive_possessive",
    "validation_archive_query",
    "behavior_catalog_direct",
    "behavior_catalog_inverse",
    "behavior_ledger_direct",
    "behavior_ledger_query",
    "probe_index_direct",
    "probe_index_inverse",
    "probe_file_direct",
    "probe_file_query",
    "intervention_register_direct",
    "intervention_register_inverse",
    "intervention_dossier_direct",
    "intervention_dossier_query",
)
SURFACE_FEATURE_NAMES = (
    *_NUMERIC_SURFACE_FEATURE_NAMES,
    *(f"entity_domain_{domain}" for domain in REGISTERED_ENTITY_DOMAINS),
    *(f"prompt_template_{template}" for template in REGISTERED_PROMPT_TEMPLATES),
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_field(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _revision(value: object, name: str) -> str:
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ValueError(f"{name} must be an immutable revision")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _readonly_vector(values: Sequence[float], name: str, *, size: int | None = None) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1 or (size is not None and array.size != size):
        expected = "a one-dimensional vector" if size is None else f"exactly {size} values"
        raise ValueError(f"{name} must contain {expected}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64)


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("canonical payload must be a mapping")
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return tuple(_freeze(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class OutputEvidence:
    """Exact two-candidate teacher-forced scores plus prompt/model provenance."""

    example_id: str
    source_sha256: str
    target_code: str
    unknown_suffix: str
    target_logp: float
    unknown_logp: float
    prompt_bytes: bytes
    rendered_prompt_sha256: str
    prompt_input_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    unknown_token_ids: tuple[int, ...]
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_config_sha256: str
    chat_template_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_bytes", bytes(self.prompt_bytes))
        for name in ("prompt_input_ids", "target_token_ids", "unknown_token_ids"):
            values = tuple(getattr(self, name))
            if not values or any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"{name} must be nonempty nonnegative token IDs")
            object.__setattr__(self, name, values)
        for name in ("example_id", "target_code", "model_id", "tokenizer_id"):
            _text(getattr(self, name), name)
        _sha_field(self.source_sha256, "source_sha256")
        _revision(self.model_revision, "model_revision")
        _revision(self.tokenizer_revision, "tokenizer_revision")
        for name in (
            "rendered_prompt_sha256",
            "tokenizer_config_sha256",
            "chat_template_sha256",
            "config_sha256",
        ):
            _sha_field(getattr(self, name), name)
        if hashlib.sha256(self.prompt_bytes).hexdigest() != self.rendered_prompt_sha256:
            raise ValueError("rendered_prompt_sha256 does not match prompt_bytes")
        if self.unknown_suffix != UNKNOWN_SUFFIX:
            raise ValueError("unknown_suffix must be the exact registered UNKNOWN suffix")
        if self.target_code == self.unknown_suffix:
            raise ValueError("target code must differ from the UNKNOWN suffix")
        if not math.isfinite(float(self.target_logp)) or not math.isfinite(float(self.unknown_logp)):
            raise ValueError("sequence log probabilities must be finite")
        object.__setattr__(self, "target_logp", float(self.target_logp))
        object.__setattr__(self, "unknown_logp", float(self.unknown_logp))

    @property
    def canonical_payload(self) -> Mapping[str, Any]:
        return _frozen_mapping(
            {
                "example_id": self.example_id,
                "source_sha256": self.source_sha256,
                "target_code": self.target_code,
                "unknown_suffix": self.unknown_suffix,
                "target_logp": self.target_logp,
                "unknown_logp": self.unknown_logp,
                "prompt_utf8_hex": self.prompt_bytes.hex(),
                "rendered_prompt_sha256": self.rendered_prompt_sha256,
                "prompt_input_ids": list(self.prompt_input_ids),
                "target_token_ids": list(self.target_token_ids),
                "unknown_token_ids": list(self.unknown_token_ids),
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "tokenizer_id": self.tokenizer_id,
                "tokenizer_revision": self.tokenizer_revision,
                "tokenizer_config_sha256": self.tokenizer_config_sha256,
                "chat_template_sha256": self.chat_template_sha256,
                "config_sha256": self.config_sha256,
            }
        )

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(_thaw(self.canonical_payload))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True)
class FeatureEvidence:
    """The fixed surface/output baselines bound to one extracted activation record."""

    example_id: str
    source_sha256: str
    activation_sha256: str
    output_evidence: OutputEvidence
    output_evidence_sha256: str
    config_sha256: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_config_sha256: str
    chat_template_sha256: str
    rendered_prompt_sha256: str
    surface_features: tuple[float, ...]
    output_features: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.output_evidence, OutputEvidence):
            raise ValueError("output_evidence must be an OutputEvidence record")
        for name in ("example_id", "model_id", "tokenizer_id"):
            _text(getattr(self, name), name)
        for name in (
            "source_sha256",
            "activation_sha256",
            "output_evidence_sha256",
            "config_sha256",
            "tokenizer_config_sha256",
            "chat_template_sha256",
            "rendered_prompt_sha256",
        ):
            _sha_field(getattr(self, name), name)
        _revision(self.model_revision, "model_revision")
        _revision(self.tokenizer_revision, "tokenizer_revision")
        surface = tuple(float(value) for value in self.surface_features)
        output = tuple(float(value) for value in self.output_features)
        if not all(math.isfinite(value) for value in surface):
            raise ValueError("surface_features must be finite")
        if len(output) != OUTPUT_FEATURE_DIM or not all(math.isfinite(value) for value in output):
            raise ValueError("output_features must be exactly 11 finite values")
        object.__setattr__(self, "surface_features", surface)
        object.__setattr__(self, "output_features", output)
        if self.output_evidence.sha256 != self.output_evidence_sha256:
            raise ValueError("output_evidence_sha256 does not match output_evidence")
        expected_output = tuple(output_feature_vector(self.output_evidence))
        if output != expected_output:
            raise ValueError("output_features must be derived from exact OutputEvidence scores")
        evidence = self.output_evidence
        checks = (
            evidence.example_id == self.example_id,
            evidence.source_sha256 == self.source_sha256,
            evidence.config_sha256 == self.config_sha256,
            evidence.model_id == self.model_id,
            evidence.model_revision == self.model_revision,
            evidence.tokenizer_id == self.tokenizer_id,
            evidence.tokenizer_revision == self.tokenizer_revision,
            evidence.tokenizer_config_sha256 == self.tokenizer_config_sha256,
            evidence.chat_template_sha256 == self.chat_template_sha256,
            evidence.rendered_prompt_sha256 == self.rendered_prompt_sha256,
        )
        if not all(checks):
            raise ValueError("FeatureEvidence provenance does not match OutputEvidence")

    @classmethod
    def from_records(
        cls,
        example: FAExample,
        activation: ActivationRecord,
        output: OutputEvidence,
        config_sha256: str,
    ) -> "FeatureEvidence":
        _validate_provenance(example, activation, output, config_sha256)
        return cls(
            example_id=example.example_id,
            source_sha256=example.canonical_payload_sha256,
            activation_sha256=activation.activation_sha256,
            output_evidence=output,
            output_evidence_sha256=output.sha256,
            config_sha256=config_sha256,
            model_id=output.model_id,
            model_revision=output.model_revision,
            tokenizer_id=output.tokenizer_id,
            tokenizer_revision=output.tokenizer_revision,
            tokenizer_config_sha256=output.tokenizer_config_sha256,
            chat_template_sha256=output.chat_template_sha256,
            rendered_prompt_sha256=output.rendered_prompt_sha256,
            surface_features=surface_feature_vector(example),
            output_features=output_feature_vector(output),
        )

    @property
    def canonical_payload(self) -> Mapping[str, Any]:
        return _frozen_mapping(
            {
                "example_id": self.example_id,
                "source_sha256": self.source_sha256,
                "activation_sha256": self.activation_sha256,
                "output_evidence": _thaw(self.output_evidence.canonical_payload),
                "output_evidence_sha256": self.output_evidence_sha256,
                "config_sha256": self.config_sha256,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "tokenizer_id": self.tokenizer_id,
                "tokenizer_revision": self.tokenizer_revision,
                "tokenizer_config_sha256": self.tokenizer_config_sha256,
                "chat_template_sha256": self.chat_template_sha256,
                "rendered_prompt_sha256": self.rendered_prompt_sha256,
                "surface_features": list(self.surface_features),
                "output_features": list(self.output_features),
            }
        )

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(_thaw(self.canonical_payload))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@runtime_checkable
class ExactSequenceScorer(Protocol):
    """Scores only registered candidate sequences without generating a completion."""

    def score(self, example: FAExample) -> OutputEvidence: ...


class HFTeacherForcedScorer:
    """Hugging Face causal-LM adapter for exact full-string teacher forcing."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str,
        model_revision: str,
        config_sha256: str,
    ) -> None:
        _text(model_id, "model_id")
        _revision(model_revision, "model_revision")
        _sha_field(config_sha256, "config_sha256")
        if not callable(model):
            raise ValueError("model must be callable as a causal language model")
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._model_revision = model_revision
        self._config_sha256 = config_sha256
        self._tokenizer_id, self._tokenizer_revision, self._tokenizer_config_sha256 = _tokenizer_provenance(tokenizer)
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str):
            raise ValueError("tokenizer must expose exact chat_template bytes")
        self._chat_template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

    @classmethod
    def from_config(cls, model: Any, tokenizer: Any, config: FAConfig) -> "HFTeacherForcedScorer":
        if not isinstance(config, FAConfig):
            raise ValueError("config must be an FAConfig")
        return cls(
            model,
            tokenizer,
            model_id=config.model_id,
            model_revision=config.model_revision,
            config_sha256=config.config_hash,
        )

    def score(self, example: FAExample) -> OutputEvidence:
        if not isinstance(example, FAExample):
            raise ValueError("exact sequence scoring requires an FAExample")
        self._require_eval_mode()
        prompt = _render_prompt(self._tokenizer, example.user_text)
        prompt_ids = _token_ids(self._tokenizer, prompt)
        target_ids, target_logp = self._score_suffix(prompt, prompt_ids, example.registry_code)
        unknown_ids, unknown_logp = self._score_suffix(prompt, prompt_ids, UNKNOWN_SUFFIX)
        return OutputEvidence(
            example_id=example.example_id,
            source_sha256=example.canonical_payload_sha256,
            target_code=example.registry_code,
            unknown_suffix=UNKNOWN_SUFFIX,
            target_logp=target_logp,
            unknown_logp=unknown_logp,
            prompt_bytes=prompt.encode("utf-8"),
            rendered_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            prompt_input_ids=prompt_ids,
            target_token_ids=target_ids,
            unknown_token_ids=unknown_ids,
            model_id=self._model_id,
            model_revision=self._model_revision,
            tokenizer_id=self._tokenizer_id,
            tokenizer_revision=self._tokenizer_revision,
            tokenizer_config_sha256=self._tokenizer_config_sha256,
            chat_template_sha256=self._chat_template_sha256,
            config_sha256=self._config_sha256,
        )

    def _require_eval_mode(self) -> None:
        eval_mode = getattr(self._model, "eval", None)
        if not callable(eval_mode):
            raise ValueError("teacher-forced scoring requires model.eval() support")
        eval_mode()
        if getattr(self._model, "training", None) is not False:
            raise ValueError("teacher-forced scoring requires eval mode; model remains in training mode")

    def _score_suffix(self, prompt: str, prompt_ids: tuple[int, ...], suffix: str) -> tuple[tuple[int, ...], float]:
        full_ids = _token_ids(self._tokenizer, prompt + suffix)
        if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("candidate suffix is not an exact compatible token continuation")
        suffix_ids = full_ids[len(prompt_ids) :]
        try:
            import torch
        except ImportError as error:  # pragma: no cover - locked F2A runtime includes torch
            raise RuntimeError("HF teacher-forced scoring requires torch") from error
        device = _model_device(self._model, torch)
        ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        # The final input token has no next-token target, so score logits[:-1] vs ids[1:].
        with torch.inference_mode():
            result = self._model(input_ids=ids[:, :-1])
            logits = getattr(result, "logits", None)
            if logits is None:
                raise ValueError("causal model result must expose logits")
            if tuple(logits.shape[:2]) != (1, len(full_ids) - 1):
                raise ValueError("causal model logits do not align with exact teacher forcing")
            log_probs = torch.log_softmax(logits[0], dim=-1)
            targets = ids[0, 1:]
            token_logps = log_probs.gather(1, targets[:, None]).squeeze(1)
            score = token_logps[len(prompt_ids) - 1 :].sum()
        value = float(score.detach().cpu().item())
        if not math.isfinite(value):
            raise ValueError("teacher-forced sequence score must be finite")
        return suffix_ids, value


@dataclass(frozen=True)
class VerifiedDomainRelation:
    """Explicit human-verified grouping metadata; never inferred from names."""

    example_id: str
    entity_id: str
    template_id: str
    relation_id: str
    domain: str
    condition: str
    metadata_manifest: Mapping[str, Any] = field(repr=False, compare=False)
    metadata_row: Mapping[str, Any] = field(repr=False, compare=False)
    metadata_manifest_sha256: str = field(init=False)
    metadata_row_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("example_id", "entity_id", "template_id", "relation_id", "domain", "condition"):
            _text(getattr(self, name), f"verified {name}")
        manifest = _freeze_metadata_manifest(self.metadata_manifest)
        row = _freeze_metadata_row(self.metadata_row)
        manifest_rows = manifest["rows"]
        if row not in manifest_rows:
            raise ValueError("verified metadata row must be an exact membership match in the metadata manifest")
        row_checks = (
            row["example_id"] == self.example_id,
            row["entity_id"] == self.entity_id,
            row["template_id"] == self.template_id,
            row["relation_id"] == self.relation_id,
            row["domain"] == self.domain,
            row["condition"] == self.condition,
        )
        if not all(row_checks):
            raise ValueError("verified metadata row does not match the requested metadata fields")
        object.__setattr__(self, "metadata_manifest", manifest)
        object.__setattr__(self, "metadata_row", row)
        object.__setattr__(self, "metadata_manifest_sha256", _sha(_thaw(manifest)))
        object.__setattr__(self, "metadata_row_sha256", _sha(_thaw(row)))

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        example_id: str,
        entity_id: str,
        template_id: str,
    ) -> "VerifiedDomainRelation":
        frozen_manifest = _freeze_metadata_manifest(manifest)
        matches = tuple(
            row
            for row in frozen_manifest["rows"]
            if (
                row["example_id"] == example_id
                and row["entity_id"] == entity_id
                and row["template_id"] == template_id
            )
        )
        if not matches:
            raise ValueError("verified metadata lookup failed membership checks against the metadata manifest")
        if len(matches) != 1:
            raise ValueError("verified metadata lookup is ambiguous for the exact example/entity/template key")
        row = matches[0]
        return cls(
            example_id=example_id,
            entity_id=entity_id,
            template_id=template_id,
            relation_id=row["relation_id"],
            domain=row["domain"],
            condition=row["condition"],
            metadata_manifest=frozen_manifest,
            metadata_row=row,
        )

    @property
    def canonical_payload(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "example_id": self.example_id,
                "entity_id": self.entity_id,
                "template_id": self.template_id,
                "relation_id": self.relation_id,
                "domain": self.domain,
                "condition": self.condition,
            }
        )

    @property
    def sha256(self) -> str:
        return _sha(dict(self.canonical_payload))


@dataclass(frozen=True)
class UnsupportedAnswerOutcome:
    """Outcome target and status kept outside the feature evidence payload."""

    example_id: str
    answer_attempt: int
    outcome_status: str

    def __post_init__(self) -> None:
        _text(self.example_id, "example_id")
        if type(self.answer_attempt) is not int or self.answer_attempt not in {0, 1}:
            raise ValueError("answer_attempt must be 0 or 1")
        if self.outcome_status not in {"valid", "missing", "invalid"}:
            raise ValueError("outcome_status is not registered")


def output_feature_vector(evidence: OutputEvidence) -> np.ndarray:
    """The registered 11D output-aligned control, derived only from two exact scores."""
    if not isinstance(evidence, OutputEvidence):
        raise ValueError("output features require OutputEvidence")
    target = evidence.target_logp
    unknown = evidence.unknown_logp
    maximum = max(target, unknown)
    logsumexp = maximum + math.log(math.exp(target - maximum) + math.exp(unknown - maximum))
    target_probability = math.exp(target - logsumexp)
    unknown_probability = math.exp(unknown - logsumexp)
    entropy = -sum(
        probability * math.log(probability)
        for probability in (target_probability, unknown_probability)
        if probability > 0.0
    )
    return _readonly_vector(
        (
            target,
            unknown,
            target - unknown,
            maximum,
            logsumexp,
            target_probability,
            unknown_probability,
            entropy,
            abs(target_probability - unknown_probability),
            target_probability - unknown_probability,
            max(target_probability, unknown_probability),
        ),
        "output feature vector",
        size=OUTPUT_FEATURE_DIM,
    )


def surface_feature_vector(example: FAExample) -> tuple[float, ...]:
    """Pre-metadata numeric controls; experimental factors and completions are excluded."""
    if not isinstance(example, FAExample):
        raise ValueError("surface features require an FAExample")
    return (
        float(len(example.target_text)),
        float(len(example.distractor_text)),
        float(len(example.user_text)),
        float(len(example.target_text.split())),
        float(len(example.distractor_text.split())),
        float(len(example.user_text.split())),
        float(sum(character.isupper() for character in example.target_text)),
        float(sum(character.isupper() for character in example.distractor_text)),
        float(sum(character.isupper() for character in example.user_text)),
        float(example.rendered_token_count),
        float(example.entity_order == "target_first"),
        float(example.entity_order == "target_second"),
        float(example.code_position == "first"),
        float(example.code_position == "second"),
        float(example.code_position == "absent"),
    )


def _probe_surface_feature_vector(
    example: FAExample,
    metadata: VerifiedDomainRelation,
) -> tuple[float, ...]:
    """Add fixed one-hot controls only after metadata membership has been verified."""
    if metadata.domain not in REGISTERED_ENTITY_DOMAINS:
        raise ValueError("verified metadata must use a registered entity domain")
    if metadata.template_id not in REGISTERED_PROMPT_TEMPLATES:
        raise ValueError("verified metadata must use a registered prompt template")
    return (
        *surface_feature_vector(example),
        *(float(metadata.domain == domain) for domain in REGISTERED_ENTITY_DOMAINS),
        *(float(metadata.template_id == template) for template in REGISTERED_PROMPT_TEMPLATES),
    )


def build_probe_row(
    example: FAExample,
    activation: ActivationRecord,
    features: FeatureEvidence,
    metadata: VerifiedDomainRelation,
    *,
    task: str,
    outcome: UnsupportedAnswerOutcome | None = None,
) -> ProbeRow:
    """Build a validated ProbeRow after all feature values have already been fixed."""
    _validate_feature_binding(example, activation, features, metadata)
    surface_features = _probe_surface_feature_vector(example, metadata)
    if task == "familiarity":
        label: int | str = int(example.target_familiarity == "screened_real")
        outcome_status = "valid"
    elif task == "answerability":
        label = example.answerability
        outcome_status = "valid"
    elif task == "unsupported_answer":
        if example.answerability not in {"distractor_bound", "code_absent"}:
            raise ValueError("unsupported-answer rows require an evidence-absent example")
        if not isinstance(outcome, UnsupportedAnswerOutcome) or outcome.example_id != example.example_id:
            raise ValueError("unsupported-answer rows require the matching explicit outcome")
        label = outcome.answer_attempt
        outcome_status = outcome.outcome_status
    else:
        raise ValueError("task is not registered")
    return ProbeRow(
        example_id=example.example_id,
        split=example.split,
        task=task,
        label=label,
        entity_id=metadata.entity_id,
        template_id=metadata.template_id,
        relation_id=metadata.relation_id,
        domain=metadata.domain,
        condition=metadata.condition,
        answerability_condition=example.answerability,
        target_familiarity_condition=example.target_familiarity,
        distractor_familiarity_condition=example.distractor_familiarity,
        surface_features=surface_features,
        output_margin_features=features.output_features,
        residual_features=activation.activations,
        sae_features=None,
        outcome_status=outcome_status,
        source_sha256=features.source_sha256,
        activation_sha256=features.activation_sha256,
        metadata_manifest_sha256=metadata.metadata_manifest_sha256,
        metadata_row_sha256=metadata.metadata_row_sha256,
        output_control_schema_sha256=OUTPUT_CONTROL_SCHEMA_SHA256,
        output_evidence_sha256=features.output_evidence_sha256,
    )


def build_probe_rows(
    records: Sequence[tuple[Any, ...]],
    *,
    task: str,
    expected_example_ids: Sequence[str],
) -> tuple[ProbeRow, ...]:
    """Convert an explicitly supplied set of records, binding its exact ID multiset."""
    rows: list[ProbeRow] = []
    received_ids: list[str] = []
    for record in records:
        values = tuple(record)
        if len(values) not in {4, 5}:
            raise ValueError("each batch record must contain example, activation, features, metadata, and optional outcome")
        example, activation, features, metadata = values[:4]
        outcome = values[4] if len(values) == 5 else None
        row = build_probe_row(
            example,
            activation,
            features,
            metadata,
            task=task,
            outcome=outcome,
        )
        rows.append(row)
        received_ids.append(row.example_id)
    if len(set(received_ids)) != len(received_ids):
        raise ValueError("batch records do not match the expected exact ID multiset: duplicate example IDs")
    expected = tuple(expected_example_ids)
    if not expected or any(not isinstance(example_id, str) or not example_id for example_id in expected):
        raise ValueError("expected_example_ids must be a nonempty explicit ID multiset")
    if Counter(received_ids) != Counter(expected):
        raise ValueError("batch records do not match the expected exact ID multiset")
    return tuple(sorted(rows, key=lambda row: (row.example_id, row.sha256)))


def _validate_provenance(
    example: FAExample,
    activation: ActivationRecord,
    output: OutputEvidence,
    config_sha256: str,
) -> None:
    if not isinstance(example, FAExample) or not isinstance(activation, ActivationRecord) or not isinstance(output, OutputEvidence):
        raise ValueError("feature provenance requires FAExample, ActivationRecord, and OutputEvidence")
    _sha_field(config_sha256, "config_sha256")
    anchors = activation.anchors
    checks = (
        output.example_id == example.example_id == activation.example_id,
        output.source_sha256 == example.canonical_payload_sha256,
        output.target_code == example.registry_code,
        output.config_sha256 == config_sha256,
        output.model_id == activation.model_id,
        output.model_revision == activation.model_revision,
        output.tokenizer_id == anchors.tokenizer_id,
        output.tokenizer_revision == anchors.tokenizer_revision,
        output.tokenizer_config_sha256 == anchors.tokenizer_config_sha256,
        output.chat_template_sha256 == anchors.chat_template_sha256,
        output.rendered_prompt_sha256 == anchors.rendered_prompt_sha256,
        output.prompt_bytes == anchors.rendered_bytes,
        output.prompt_input_ids == anchors.input_ids,
        tuple(example.rendered_token_ids) == anchors.input_ids,
    )
    if not all(checks):
        raise ValueError("output and activation provenance do not bind to the exact example")


def _validate_feature_binding(
    example: FAExample,
    activation: ActivationRecord,
    features: FeatureEvidence,
    metadata: VerifiedDomainRelation,
) -> None:
    if not isinstance(features, FeatureEvidence) or not isinstance(metadata, VerifiedDomainRelation):
        raise ValueError("probe row requires FeatureEvidence and verified domain/relation metadata")
    _validate_provenance(example, activation, features.output_evidence, features.config_sha256)
    if metadata.entity_id != example.entity_unit_id or metadata.template_id != example.template_family:
        raise ValueError("verified metadata does not bind to the exact example metadata")
    checks = (
        features.example_id == example.example_id == activation.example_id == metadata.example_id,
        features.source_sha256 == example.canonical_payload_sha256,
        features.activation_sha256 == activation.activation_sha256,
        features.model_id == activation.model_id,
        features.model_revision == activation.model_revision,
        features.tokenizer_id == activation.anchors.tokenizer_id,
        features.tokenizer_revision == activation.anchors.tokenizer_revision,
        features.tokenizer_config_sha256 == activation.anchors.tokenizer_config_sha256,
        features.chat_template_sha256 == activation.anchors.chat_template_sha256,
        features.rendered_prompt_sha256 == activation.anchors.rendered_prompt_sha256,
        activation.layer_ids == tuple(range(26)),
        activation.activations.shape[:2] == (3, 26),
        features.surface_features == surface_feature_vector(example),
        features.output_features == tuple(output_feature_vector(features.output_evidence)),
    )
    if not all(checks):
        raise ValueError("feature provenance does not bind to the exact example/activation metadata")


def _render_prompt(tokenizer: Any, user_text: str) -> str:
    render = getattr(tokenizer, "apply_chat_template", None)
    if not callable(render):
        raise ValueError("tokenizer must implement apply_chat_template")
    prompt = render(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("chat template must render a nonempty prompt")
    return prompt


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, Mapping):
        values = encoded.get("input_ids")
    else:
        values = getattr(encoded, "input_ids", None)
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, Sequence) and values and isinstance(values[0], Sequence):
        if len(values) != 1:
            raise ValueError("exact candidate scoring requires unbatched tokenization")
        values = values[0]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("tokenizer must return input_ids")
    result = tuple(values)
    if not result or any(type(value) is not int or value < 0 for value in result):
        raise ValueError("tokenizer input_ids must be nonempty nonnegative integers")
    return result


def _tokenizer_provenance(tokenizer: Any) -> tuple[str, str, str]:
    identifier = getattr(tokenizer, "name_or_path", None)
    _text(identifier, "tokenizer ID")
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    revision = init_kwargs.get("revision") if isinstance(init_kwargs, Mapping) else None
    _revision(revision, "tokenizer revision")
    config = {
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "name_or_path": identifier,
        "revision": revision,
        "init_kwargs": _json_safe(dict(init_kwargs)),
    }
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return identifier, revision, digest


def _freeze_metadata_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("metadata manifest must be a mapping")
    rows = manifest.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("metadata manifest must contain a nonempty rows sequence for membership verification")
    frozen = _frozen_mapping(manifest)
    frozen_rows = frozen.get("rows")
    if not isinstance(frozen_rows, tuple) or not frozen_rows:
        raise ValueError("metadata manifest rows must be immutable records")
    for row in frozen_rows:
        _freeze_metadata_row(row)
    return frozen


def _freeze_metadata_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("metadata row must be a mapping")
    frozen_row = _frozen_mapping(row)
    for key in ("example_id", "entity_id", "template_id", "relation_id", "domain", "condition"):
        _text(frozen_row.get(key), f"metadata row {key}")
    return frozen_row


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("tokenizer init_kwargs must be canonical JSON values")


def _model_device(model: Any, torch: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")
