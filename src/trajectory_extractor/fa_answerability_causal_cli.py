"""CLI orchestration for the Same-String answerability causal pilot."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from trajectory_extractor.fa_answerability_causal import (
    CAUSAL_ANSWERABILITY,
    CAUSAL_DIRECTION_ANCHOR,
    CAUSAL_EXPOSURES,
    CAUSAL_REPLICATION_STUDY_ID,
    CAUSAL_SPLIT_COUNTS,
    CAUSAL_STUDY_ID,
    CAUSAL_VALIDATION_LAYERS,
    CAUSAL_VALIDATION_MULTIPLIERS,
    CausalExpectedProvenance,
    LabelShuffledDirectionArtifact,
    ValidationCandidate,
    ValidationSelection,
    V3DirectionProvenance,
    build_label_shuffled_direction,
    build_causal_corpus,
    fit_train_only_directions,
    load_v3_training_direction_inputs,
    lock_causal_intervention,
    select_causal_intervention,
    verify_causal_corpus,
    verify_direction_bundle,
    write_causal_corpus,
    write_direction_bundle,
)
from trajectory_extractor.fa_answerability_causal_analysis import (
    CAUSAL_CONTROLS,
    BaselineScore,
    CausalAnalysisStore,
    CausalEvidence,
    CausalEvaluationSeal,
    ControlScore,
    ExecutionAuditHashes,
    GenerationClass,
    GenerationResult,
    ManipulationCheck,
    PreservationResult,
    PrimaryScore,
    analyze_causal_study,
    deterministic_farthest_layer,
    seal_causal_evaluation,
)
from trajectory_extractor.fa_answerability_causal_runtime import (
    generate_causal_completion,
    resolve_causal_anchor,
    score_answerability_candidates,
    vector_audit_hashes,
)


CAUSAL_COMMANDS = (
    "fa-causal-prepare",
    "fa-causal-run-validation",
    "fa-causal-run-shard",
    "fa-causal-evaluate",
)
_UNSELECTED_SHA256 = "0" * 64
_EXPECTED_CONFIG = {
    "schema_version": 1,
    "profile": "confirmatory",
    "study_id": CAUSAL_STUDY_ID,
    "run_id": CAUSAL_STUDY_ID,
    "model_id": "google/gemma-2-2b-it",
    "model_revision": "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
    "tokenizer_revision": "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
    "chat_template_sha256": "ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6",
    "split_seed": 20260804,
    "split_counts": dict(CAUSAL_SPLIT_COUNTS),
    "factorial_cells": {
        "exposure": list(CAUSAL_EXPOSURES),
        "answerability": list(CAUSAL_ANSWERABILITY),
    },
    "direction": {
        "source_split": "representation_train",
        "anchor": CAUSAL_DIRECTION_ANCHOR,
        "layers": list(CAUSAL_VALIDATION_LAYERS),
        "scale": "median_positive_paired_projection_gap",
    },
    "validation_selection": {
        "split": "causal_validation",
        "multipliers": list(CAUSAL_VALIDATION_MULTIPLIERS),
        "invalid_output_rate_max": 0.05,
        "bound_accuracy_drop_max": 0.05,
        "tie_break": ["smaller_multiplier", "earlier_layer"],
    },
    "statistics": {
        "bootstrap_draws": 10000,
        "bootstrap_seed": 20260804,
        "sign_flip_draws": 9999,
        "sign_flip_seed": 20260804,
    },
    "generation": {"do_sample": False, "max_new_tokens": 16, "temperature": 0.0},
}
_EXPECTED_REPLICATION_CONFIG = {
    **_EXPECTED_CONFIG,
    "study_id": CAUSAL_REPLICATION_STUDY_ID,
    "run_id": CAUSAL_REPLICATION_STUDY_ID,
    "split_seed": 20260805,
    "validation_selection": {
        **_EXPECTED_CONFIG["validation_selection"],
        "mode": "locked_from_v1",
        "locked_layer": 18,
        "locked_multiplier": 1.0,
    },
    "statistics": dict(_EXPECTED_CONFIG["statistics"]),
    "corpus": {
        "unit_offset": 100,
        "unit_prefix": "causal-v2-unit",
        "fresh_templates": ["briefing_panels", "dispatch_panels"],
        "excluded_causal_manifest": (
            "release/familiarity_answerability/answerability_causal_pilot_v1/"
            "prepared/corpus/same_string_answerability_causal_manifest.json"
        ),
        "excluded_causal_manifest_sha256": (
            "c91aa923d358c0b7ba2bbb3f0f11314efb1d45526cda4fa8ae63e19a6449d4ec"
        ),
    },
    "preregistration": {
        "path": "docs/familiarity_answerability_causal_replication_v2_preregistration.md",
        "sha256": "6a159773bf6def8dbd028c07b6f3e730b6ca1a33e9d363d7e9653460c4b0c395",
    },
}
_EXPECTED_CONFIGS = {
    value["study_id"]: value
    for value in (_EXPECTED_CONFIG, _EXPECTED_REPLICATION_CONFIG)
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _registered_repository_file(value: Mapping[str, Any], *, field: str) -> Path:
    relative = value.get("path")
    expected = _require_sha256(value.get("sha256"), f"{field} hash")
    if not isinstance(relative, str):
        raise ValueError(f"{field} path is invalid")
    candidate = (_REPOSITORY_ROOT / relative).resolve()
    if _REPOSITORY_ROOT.resolve() not in candidate.parents:
        raise ValueError(f"{field} path escapes the repository")
    if _sha256_file(candidate) != expected:
        raise ValueError(f"{field} hash does not match the registered file")
    return candidate


def _registered_exclusion_manifest(config: CausalCLIConfig) -> Path | None:
    if config.study_id != CAUSAL_REPLICATION_STUDY_ID:
        return None
    relative = config.corpus.get("excluded_causal_manifest")
    if not isinstance(relative, str):
        raise ValueError("causal replication exclusion path is invalid")
    candidate = (_REPOSITORY_ROOT / relative).resolve()
    if _REPOSITORY_ROOT.resolve() not in candidate.parents:
        raise ValueError("causal replication exclusion path escapes the repository")
    if not candidate.is_file():
        raise ValueError("causal replication exclusion artifact is missing")
    expected = _require_sha256(
        config.corpus.get("excluded_causal_manifest_sha256"),
        "causal replication exclusion artifact hash",
    )
    if _sha256_file(candidate) != expected:
        raise ValueError("causal replication exclusion artifact hash mismatch")
    return candidate


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate causal config key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class CausalCLIConfig:
    value: Mapping[str, Any]
    config_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", MappingProxyType(dict(self.value)))
        _require_sha256(self.config_sha256, "causal config hash")

    def __getattr__(self, name: str) -> Any:
        try:
            return self.value[name]
        except KeyError as error:
            raise AttributeError(name) from error


def load_causal_config(path: str | Path) -> CausalCLIConfig:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("causal config is unreadable") from error
    if not isinstance(value, dict) or value != _EXPECTED_CONFIGS.get(value.get("study_id")):
        raise ValueError("config does not match the registered causal config")
    return CausalCLIConfig(value=value, config_sha256=_sha256_json(value))


@dataclass(frozen=True)
class CausalTokenizerBinding:
    tokenizer: Any
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    chat_template_sha256: str


@dataclass(frozen=True)
class InterventionRequest:
    purpose: str
    control: str
    layer_id: int
    multiplier: float
    sign: int
    anchor: str
    vector: tuple[float, ...]
    identity_sha256: str
    request_sha256: str = ""

    def __post_init__(self) -> None:
        if self.layer_id not in CAUSAL_VALIDATION_LAYERS:
            raise ValueError("intervention request layer is not registered")
        if not np.isfinite(self.multiplier) or self.multiplier < 0.0:
            raise ValueError("intervention request multiplier is invalid")
        if self.sign not in {-1, 0, 1}:
            raise ValueError("intervention request sign is invalid")
        vector = tuple(float(value) for value in self.vector)
        if self.sign == 0 and vector:
            raise ValueError("no-op intervention request cannot carry a vector")
        if self.sign != 0 and (not vector or not np.isfinite(vector).all()):
            raise ValueError("active intervention request requires a finite vector")
        _require_sha256(self.identity_sha256, "intervention identity hash")
        payload = {
            "purpose": self.purpose,
            "control": self.control,
            "layer_id": self.layer_id,
            "multiplier": float(self.multiplier),
            "sign": self.sign,
            "anchor": self.anchor,
            "vector": vector,
            "identity_sha256": self.identity_sha256,
        }
        expected = _sha256_json(payload)
        if self.request_sha256 and self.request_sha256 != expected:
            raise ValueError("intervention request hash does not match content")
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "multiplier", float(self.multiplier))
        object.__setattr__(self, "request_sha256", expected)


@dataclass(frozen=True)
class RuntimeReceipt:
    loader_mode: str
    device: str
    dtype: str
    peak_memory_bytes: int
    memory_limit_bytes: int
    quantization: str
    batch_size: int
    gradients_enabled: bool
    smoke_request_sha256: str

    def __post_init__(self) -> None:
        if self.loader_mode not in {"cuda_4bit", "registered_full_precision"}:
            raise ValueError("runtime loader mode is not registered")
        if not self.device or not self.dtype or not self.quantization:
            raise ValueError("runtime device, dtype, and quantization must be recorded")
        if (
            type(self.peak_memory_bytes) is not int
            or type(self.memory_limit_bytes) is not int
            or self.peak_memory_bytes < 0
            or self.memory_limit_bytes <= 0
            or self.peak_memory_bytes > self.memory_limit_bytes
        ):
            raise ValueError("runtime smoke memory boundary is invalid")
        if self.batch_size != 1 or self.gradients_enabled:
            raise ValueError("causal runtime requires batch 1 with gradients disabled")
        _require_sha256(self.smoke_request_sha256, "smoke request hash")
        if self.loader_mode == "cuda_4bit" and self.quantization != "bitsandbytes_nf4":
            raise ValueError("CUDA causal runtime must use the registered 4-bit loader")
        if self.loader_mode == "registered_full_precision" and self.quantization != "none":
            raise ValueError("full-precision fallback cannot claim quantization")

    @property
    def runtime_sha256(self) -> str:
        return _sha256_json(
            {
                "loader_mode": self.loader_mode,
                "device": self.device,
                "dtype": self.dtype,
                "memory_limit_bytes": self.memory_limit_bytes,
                "quantization": self.quantization,
                "batch_size": self.batch_size,
                "gradients_enabled": self.gradients_enabled,
            }
        )


@dataclass(frozen=True)
class RunObservation:
    raw_margin: float
    length_normalized_margin: float
    generated_text: str
    primary_projection_delta: float
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not all(
            np.isfinite(value)
            for value in (
                self.raw_margin,
                self.length_normalized_margin,
                self.primary_projection_delta,
            )
        ):
            raise ValueError("causal observation values must be finite")
        if not isinstance(self.generated_text, str):
            raise ValueError("causal observation generation must be text")
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


class CausalRunner(Protocol):
    def smoke(self, prompt: Any, request: InterventionRequest) -> RuntimeReceipt: ...

    def observe(
        self, prompt: Any, request: InterventionRequest
    ) -> RunObservation: ...

    def runtime_identity(self, request_sha256: str) -> RuntimeReceipt: ...


def _load_pinned_causal_tokenizer(config: CausalCLIConfig) -> CausalTokenizerBinding:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment-specific
        raise ImportError("Install transformers to load the pinned causal tokenizer") from error
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.tokenizer_revision,
    )
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("pinned causal tokenizer has no chat template")
    template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if template_hash != config.chat_template_sha256:
        raise ValueError("causal tokenizer chat template hash does not match config")
    return CausalTokenizerBinding(
        tokenizer=tokenizer,
        model_id=config.model_id,
        model_revision=config.model_revision,
        tokenizer_id=str(getattr(tokenizer, "name_or_path", config.model_id)),
        tokenizer_revision=config.tokenizer_revision,
        chat_template_sha256=template_hash,
    )


def _load_hf_causal_runner(config: CausalCLIConfig) -> CausalRunner:
    return HFCausalRunner.from_pretrained(config)


@dataclass(frozen=True)
class CausalDependencies:
    tokenizer_loader: Callable[[CausalCLIConfig], CausalTokenizerBinding] = (
        _load_pinned_causal_tokenizer
    )
    runner_factory: Callable[[CausalCLIConfig], CausalRunner] = _load_hf_causal_runner


class HFCausalRunner:
    """Small adapter from the audited runtime primitives to the CLI protocol."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        loader_mode: str,
        quantization: str,
        memory_limit_bytes: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.loader_mode = loader_mode
        self.quantization = quantization
        self.memory_limit_bytes = int(memory_limit_bytes)

    @classmethod
    def from_pretrained(cls, config: CausalCLIConfig) -> "HFCausalRunner":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:  # pragma: no cover - environment-specific
            raise ImportError(
                "Install the causal Colab requirements before loading Gemma"
            ) from error

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.tokenizer_revision,
        )
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or hashlib.sha256(
            template.encode("utf-8")
        ).hexdigest() != config.chat_template_sha256:
            raise ValueError("causal runner tokenizer does not match the registered template")

        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                revision=config.model_revision,
                quantization_config=quantization_config,
                device_map={"": 0},
                low_cpu_mem_usage=True,
            )
            torch.cuda.reset_peak_memory_stats()
            return cls(
                model=model,
                tokenizer=tokenizer,
                loader_mode="cuda_4bit",
                quantization="bitsandbytes_nf4",
                memory_limit_bytes=torch.cuda.get_device_properties(0).total_memory,
            )

        if os.environ.get("FA_CAUSAL_ALLOW_CPU") != "1":
            raise RuntimeError(
                "The registered live run requires a CUDA Colab runtime; set "
                "FA_CAUSAL_ALLOW_CPU=1 only for an explicit full-precision smoke"
            )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to("cpu")
        return cls(
            model=model,
            tokenizer=tokenizer,
            loader_mode="registered_full_precision",
            quantization="none",
            memory_limit_bytes=8_000_000_000,
        )

    def _render(self, prompt: Any) -> str:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise ValueError("causal chat template did not return text")
        return rendered

    def observe(self, prompt: Any, request: InterventionRequest) -> RunObservation:
        rendered = self._render(prompt)
        active = request.sign != 0
        layer_id = request.layer_id if active else None
        vector = np.asarray(request.vector, dtype=np.float64) if active else None
        scores = score_answerability_candidates(
            self.model,
            self.tokenizer,
            prompt,
            rendered,
            layer_id=layer_id,
            vector=vector,
            anchor_name=request.anchor,
        )
        generation = generate_causal_completion(
            self.model,
            self.tokenizer,
            prompt,
            rendered,
            layer_id=layer_id,
            vector=vector,
            anchor_name=request.anchor,
            max_new_tokens=16,
        )
        runtime_audit = generation.audit
        margin_audits = (scores.correct.audit, scores.unknown.audit)
        if active and (runtime_audit is None or any(item is None for item in margin_audits)):
            raise ValueError("active causal observation is missing a runtime audit")
        if not active and (runtime_audit is not None or any(item is not None for item in margin_audits)):
            raise ValueError("no-intervention observation unexpectedly modified a residual site")
        if active:
            expected_runtime = (
                runtime_audit.source_vector_sha256,
                runtime_audit.applied_vector_sha256,
                runtime_audit.represented_dtype,
                runtime_audit.represented_device,
                runtime_audit.layer_id,
                runtime_audit.position,
                runtime_audit.prompt_token_ids_sha256,
                runtime_audit.model_prefix_token_ids_sha256,
            )
            if any(
                (
                    item.source_vector_sha256,
                    item.applied_vector_sha256,
                    item.represented_dtype,
                    item.represented_device,
                    item.layer_id,
                    item.position,
                    item.prompt_token_ids_sha256,
                    item.model_prefix_token_ids_sha256,
                )
                != expected_runtime
                for item in margin_audits
            ):
                raise ValueError("margin and generation forwards used different interventions")
        audit = {
            "request_sha256": request.request_sha256,
            "example_id": prompt.example_id,
            "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
            "hook_call_count": 1 if active else 0,
            "modified_site_count": 1 if active else 0,
            "hook_cleanup_verified": True,
            "represented_device": (
                runtime_audit.represented_device if runtime_audit is not None else "none"
            ),
            "represented_dtype": (
                runtime_audit.represented_dtype if runtime_audit is not None else "torch.float64"
            ),
            "source_vector_sha256": (
                runtime_audit.source_vector_sha256 if runtime_audit is not None else "0" * 64
            ),
            "applied_vector_sha256": (
                runtime_audit.applied_vector_sha256 if runtime_audit is not None else "0" * 64
            ),
            "layer_id": runtime_audit.layer_id if runtime_audit is not None else request.layer_id,
            "position": runtime_audit.position if runtime_audit is not None else None,
            "prompt_token_ids_sha256": (
                runtime_audit.prompt_token_ids_sha256 if runtime_audit is not None else None
            ),
            "model_prefix_token_ids_sha256": (
                runtime_audit.model_prefix_token_ids_sha256 if runtime_audit is not None else None
            ),
            "margin_forward_audits": (
                [asdict(item) for item in margin_audits] if active else []
            ),
        }
        projection_delta = (
            request.sign * float(np.linalg.norm(vector)) if active else 0.0
        )
        return RunObservation(
            raw_margin=scores.raw_margin,
            length_normalized_margin=scores.length_normalized_margin,
            generated_text=generation.generated_text,
            primary_projection_delta=projection_delta,
            audit=audit,
        )

    def runtime_identity(self, request_sha256: str) -> RuntimeReceipt:
        peak = 0
        device = "cpu"
        dtype = str(next(self.model.parameters()).dtype)
        try:
            import torch

            if torch.cuda.is_available():
                peak = int(torch.cuda.max_memory_allocated())
                device = "cuda:0"
        except (ImportError, StopIteration):  # pragma: no cover - environment-specific
            pass
        return RuntimeReceipt(
            loader_mode=self.loader_mode,
            device=device,
            dtype=dtype,
            peak_memory_bytes=peak,
            memory_limit_bytes=self.memory_limit_bytes,
            quantization=self.quantization,
            batch_size=1,
            gradients_enabled=False,
            smoke_request_sha256=request_sha256,
        )

    def smoke(self, prompt: Any, request: InterventionRequest) -> RuntimeReceipt:
        self.observe(prompt, request)
        return self.runtime_identity(request.request_sha256)

    def unrelated_preservation(self, request: InterventionRequest) -> Mapping[str, Any]:
        rows = []
        for index in range(4):
            target = f"Neutral{index:04d}"
            code = f"U{index:04d}"
            user_text = (
                f"The archive code for {target} is {code}. "
                f"What is the archive code for {target}? Reply with only the archive code."
            )
            first = user_text.index(target)
            second = user_text.index(target, first + len(target))
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            tokenized = self.tokenizer(rendered, add_special_tokens=False)
            prompt = type("PreservationPrompt", (), {})()
            prompt.example_id = f"unrelated-code-lookup-{index}"
            prompt.user_text = user_text
            prompt.target_text = target
            prompt.target_intro_span = (first, first + len(target))
            prompt.target_query_span = (second, second + len(target))
            prompt.registry_code = code
            prompt.rendered_token_ids = tuple(int(value) for value in tokenized["input_ids"])
            prompt.rendered_prompt_sha256 = hashlib.sha256(
                rendered.encode("utf-8")
            ).hexdigest()
            baseline = generate_causal_completion(
                self.model, self.tokenizer, prompt, rendered, max_new_tokens=16
            ).generated_text.strip()
            steered = generate_causal_completion(
                self.model,
                self.tokenizer,
                prompt,
                rendered,
                layer_id=request.layer_id,
                vector=np.asarray(request.vector, dtype=np.float64),
                anchor_name=request.anchor,
                max_new_tokens=16,
            ).generated_text.strip()
            rows.append(
                {
                    "prompt_id": prompt.example_id,
                    "expected": code,
                    "baseline": baseline,
                    "generated": steered,
                }
            )
        return {
            "passed": all(
                row["baseline"] == row["expected"]
                and row["generated"] == row["expected"]
                for row in rows
            ),
            "rows": rows,
        }


@dataclass(frozen=True)
class AtomicJSONReceipt:
    path: Path
    payload: Mapping[str, Any]
    receipt_sha256: str
    resumed: bool


class AtomicJSONReceiptStore:
    """Canonical JSON receipts with process-crash-safe same-request resume."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).absolute()

    def write_or_resume(
        self,
        relative_path: str | Path,
        *,
        request_sha256: str,
        payload: Mapping[str, Any],
        resume: bool,
    ) -> AtomicJSONReceipt:
        request_sha256 = _require_sha256(request_sha256, "request hash")
        relative = Path(relative_path)
        if relative.is_absolute() or not relative.parts or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("receipt path must be a safe relative path")
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                existing, receipt_hash = self._read(path)
                if existing.get("request_sha256") != request_sha256:
                    raise ValueError("resume request hash does not match existing receipt")
                if not resume:
                    raise ValueError("receipt already exists; pass --resume for same-hash resume")
                return AtomicJSONReceipt(
                    path=path,
                    payload=MappingProxyType(existing),
                    receipt_sha256=receipt_hash,
                    resumed=True,
                )
            record = {**dict(payload), "request_sha256": request_sha256}
            receipt_hash = _sha256_json(record)
            record["receipt_sha256"] = receipt_hash
            self._atomic_write(path, _canonical_json(record) + b"\n")
            return AtomicJSONReceipt(
                path=path,
                payload=MappingProxyType(record),
                receipt_sha256=receipt_hash,
                resumed=False,
            )

    @staticmethod
    def _read(path: Path) -> tuple[dict[str, Any], str]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("atomic receipt is unreadable") from error
        if not isinstance(value, dict):
            raise ValueError("atomic receipt must be a JSON object")
        stored_hash = _require_sha256(value.get("receipt_sha256"), "receipt hash")
        content = {key: item for key, item in value.items() if key != "receipt_sha256"}
        if _sha256_json(content) != stored_hash:
            raise ValueError("atomic receipt hash does not verify")
        return value, stored_hash

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _validate_tokenizer_binding(
    binding: CausalTokenizerBinding,
    config: CausalCLIConfig,
) -> None:
    expected = (
        config.model_id,
        config.model_revision,
        config.model_id,
        config.tokenizer_revision,
        config.chat_template_sha256,
    )
    observed = (
        binding.model_id,
        binding.model_revision,
        binding.tokenizer_id,
        binding.tokenizer_revision,
        binding.chat_template_sha256,
    )
    if observed != expected:
        raise ValueError("causal tokenizer binding does not match config")


def _implementation_sha256() -> str:
    package = Path(__file__).resolve().parent
    paths = (
        package / "fa_answerability_causal.py",
        package / "fa_answerability_causal_runtime.py",
        package / "fa_answerability_causal_analysis.py",
        package / "fa_answerability_causal_cli.py",
        package / "fa_cli.py",
    )
    return _sha256_json(
        [{"path": path.name, "sha256": _sha256_file(path)} for path in paths]
    )


def _runtime_policy_sha256() -> str:
    return _sha256_json(
        {
            "batch_size": 1,
            "cuda_loader": "bitsandbytes_nf4_4bit",
            "fallback_loader": "registered_full_precision",
            "gradients": False,
            "intervention": "prefill_only_addition",
        }
    )


def _model_sha256(config: CausalCLIConfig) -> str:
    return _sha256_json(
        {"model_id": config.model_id, "revision": config.model_revision}
    )


def _tokenizer_sha256(config: CausalCLIConfig) -> str:
    return _sha256_json(
        {"model_id": config.model_id, "revision": config.tokenizer_revision}
    )


def _write_hashed_manifest(path: Path, payload: Mapping[str, Any]) -> Path:
    record = dict(payload)
    record["manifest_sha256"] = _sha256_json(record)
    content = _canonical_json(record) + b"\n"
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("immutable manifest path already contains different bytes")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    AtomicJSONReceiptStore._atomic_write(path, content)
    return path


def prepare_causal(
    args: argparse.Namespace,
    *,
    dependencies: CausalDependencies | None = None,
) -> dict[str, Any]:
    dependencies = dependencies or CausalDependencies()
    config = load_causal_config(args.config)
    root = Path(args.root).absolute()
    output = Path(args.output_dir)
    if output.is_absolute():
        output_root = output
    else:
        output_root = root / output
    output_root.mkdir(parents=True, exist_ok=True)
    binding = dependencies.tokenizer_loader(config)
    _validate_tokenizer_binding(binding, config)

    preregistration_path = None
    preregistration_sha256 = None
    if config.study_id == CAUSAL_REPLICATION_STUDY_ID:
        preregistration_path = _registered_repository_file(
            config.preregistration,
            field="causal replication preregistration",
        )
        preregistration_sha256 = _sha256_file(preregistration_path)
    excluded_causal_manifest = _registered_exclusion_manifest(config)

    v3_manifest = Path(args.v3_corpus_manifest).absolute()
    activation_manifest = Path(args.v3_training_activation_manifest).absolute()
    corpus = build_causal_corpus(
        binding.tokenizer,
        v3_manifest_path=v3_manifest,
        study_id=config.study_id,
        excluded_causal_manifest_path=excluded_causal_manifest,
    )
    corpus_paths = write_causal_corpus(corpus, output_root / "corpus")
    verified_training = load_v3_training_direction_inputs(
        v3_manifest_path=v3_manifest,
        activation_manifest_path=activation_manifest,
        expected_model_id=config.model_id,
        expected_model_revision=config.model_revision,
        expected_tokenizer_id=config.model_id,
        expected_tokenizer_revision=config.tokenizer_revision,
        expected_chat_template_sha256=config.chat_template_sha256,
    )
    bundle = fit_train_only_directions(verified_training)
    direction_path = write_direction_bundle(
        bundle,
        output_root / "direction" / "train_only_direction_bundle.json",
    )
    implementation_hash = _implementation_sha256()
    model_hash = _model_sha256(config)
    tokenizer_hash = _tokenizer_sha256(config)
    split_hash = _sha256_json(
        {
            "split_counts": dict(CAUSAL_SPLIT_COUNTS),
            "unit_ids": {
                split: sorted(
                    {
                        row.entity_unit_id
                        for row in corpus.prompts
                        if row.split == split
                    }
                )
                for split in CAUSAL_SPLIT_COUNTS
            },
        }
    )
    control_hash = _sha256_json(
        {"controls": ["baseline", "primary", *CAUSAL_CONTROLS], "random_members": 5}
    )
    request_hash = _sha256_json(
        {
            "command": "fa-causal-prepare",
            "config_sha256": config.config_sha256,
            "implementation_sha256": implementation_hash,
            "v3_manifest_sha256": _sha256_file(v3_manifest),
            "v3_activation_manifest_sha256": _sha256_file(activation_manifest),
            "preregistration_sha256": preregistration_sha256,
            "excluded_causal_manifest_sha256": (
                _sha256_file(excluded_causal_manifest)
                if excluded_causal_manifest is not None
                else None
            ),
        }
    )
    payload = {
        "schema_version": 1,
        "kind": "causal_pre_outcome_identity_seal",
        "study_id": config.study_id,
        "config_sha256": config.config_sha256,
        "implementation_sha256": implementation_hash,
        "model_sha256": model_hash,
        "tokenizer_sha256": tokenizer_hash,
        "corpus_sha256": corpus.manifest_sha256,
        "direction_bundle_sha256": bundle.bundle_sha256,
        "selection_sha256": _UNSELECTED_SHA256,
        "runtime_sha256": _runtime_policy_sha256(),
        "split_sha256": split_hash,
        "control_sha256": control_hash,
        "request_sha256": request_hash,
        "v3_corpus_manifest": str(v3_manifest),
        "v3_training_activation_manifest": str(activation_manifest),
        "preregistration": (
            str(preregistration_path) if preregistration_path is not None else None
        ),
        "preregistration_sha256": preregistration_sha256,
        "excluded_causal_manifest": (
            str(excluded_causal_manifest)
            if excluded_causal_manifest is not None
            else None
        ),
        "excluded_causal_manifest_sha256": (
            _sha256_file(excluded_causal_manifest)
            if excluded_causal_manifest is not None
            else None
        ),
        "causal_corpus_manifest": str(corpus_paths.manifest.absolute()),
        "direction_bundle": str(direction_path.absolute()),
        "audit": {
            "passed": corpus.audit.passed,
            "checks": dict(corpus.audit.checks),
            "violations": list(corpus.audit.violations),
        },
    }
    manifest_path = _write_hashed_manifest(
        output_root / "pre_outcome_identity_seal.json",
        payload,
    )
    return {
        "status": "prepared",
        "prepare_manifest": str(manifest_path.absolute()),
        "request_sha256": request_hash,
    }


def _read_hashed_manifest(path: str | Path, *, kind: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{kind} manifest is unreadable") from error
    if not isinstance(value, dict) or value.get("kind") != kind:
        raise ValueError(f"{kind} manifest has an invalid identity")
    stored = _require_sha256(value.get("manifest_sha256"), f"{kind} manifest hash")
    content = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if _sha256_json(content) != stored:
        raise ValueError(f"{kind} manifest hash does not verify")
    return value


def _expected_provenance(config: CausalCLIConfig, prepare: Mapping[str, Any], bundle: Any):
    return CausalExpectedProvenance(
        corpus_sha256=prepare["corpus_sha256"],
        direction_bundle_sha256=prepare["direction_bundle_sha256"],
        direction_hashes={
            direction.layer_id: direction.direction_sha256
            for direction in bundle.directions
        },
        model_sha256=_model_sha256(config),
        tokenizer_sha256=_tokenizer_sha256(config),
    )


def _load_prepared(
    args: argparse.Namespace,
    dependencies: CausalDependencies,
) -> tuple[CausalCLIConfig, dict[str, Any], Any, Any, CausalTokenizerBinding]:
    config = load_causal_config(args.config)
    prepare = _read_hashed_manifest(
        args.prepare_manifest,
        kind="causal_pre_outcome_identity_seal",
    )
    expected = {
        "study_id": config.study_id,
        "config_sha256": config.config_sha256,
        "implementation_sha256": _implementation_sha256(),
        "model_sha256": _model_sha256(config),
        "tokenizer_sha256": _tokenizer_sha256(config),
        "selection_sha256": _UNSELECTED_SHA256,
        "runtime_sha256": _runtime_policy_sha256(),
    }
    if any(prepare.get(field) != value for field, value in expected.items()):
        raise ValueError("prepare manifest does not match current causal identity")
    excluded_causal_manifest = _registered_exclusion_manifest(config)
    if config.study_id == CAUSAL_REPLICATION_STUDY_ID:
        preregistration_path = _registered_repository_file(
            config.preregistration,
            field="causal replication preregistration",
        )
        if (
            prepare.get("preregistration") != str(preregistration_path)
            or prepare.get("preregistration_sha256") != _sha256_file(preregistration_path)
            or prepare.get("excluded_causal_manifest")
            != str(excluded_causal_manifest)
            or prepare.get("excluded_causal_manifest_sha256")
            != _sha256_file(excluded_causal_manifest)
        ):
            raise ValueError("prepare manifest does not match registered v2 exclusions")
    binding = dependencies.tokenizer_loader(config)
    _validate_tokenizer_binding(binding, config)
    corpus = verify_causal_corpus(
        prepare["causal_corpus_manifest"],
        binding.tokenizer,
        v3_manifest_path=prepare["v3_corpus_manifest"],
        expected_study_id=config.study_id,
        excluded_causal_manifest_path=excluded_causal_manifest,
    )
    bundle = verify_direction_bundle(prepare["direction_bundle"])
    if (
        corpus.manifest_sha256 != prepare["corpus_sha256"]
        or bundle.bundle_sha256 != prepare["direction_bundle_sha256"]
    ):
        raise ValueError("prepared corpus or direction hash does not verify")
    return config, prepare, corpus, bundle, binding


def _request(
    *,
    purpose: str,
    control: str,
    layer_id: int,
    multiplier: float,
    sign: int,
    anchor: str,
    vector: Any,
    identity_sha256: str,
) -> InterventionRequest:
    values: tuple[float, ...] = ()
    if sign:
        values = tuple(float(item) for item in np.asarray(vector, dtype=np.float64))
    return InterventionRequest(
        purpose=purpose,
        control=control,
        layer_id=layer_id,
        multiplier=multiplier,
        sign=sign,
        anchor=anchor,
        vector=values,
        identity_sha256=identity_sha256,
    )


def _validate_observation(
    prompt: Any,
    request: InterventionRequest,
    observation: RunObservation,
) -> None:
    if not isinstance(observation, RunObservation):
        raise ValueError("causal runner returned an untyped observation")
    audit = observation.audit
    expected = {
        "request_sha256": request.request_sha256,
        "example_id": prompt.example_id,
        "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
        "hook_call_count": 1 if request.sign else 0,
        "modified_site_count": 1 if request.sign else 0,
        "hook_cleanup_verified": True,
    }
    if any(audit.get(field) != value for field, value in expected.items()):
        raise ValueError("causal runner observation audit does not match request")
    if request.sign and (
        not isinstance(audit.get("represented_device"), str)
        or not isinstance(audit.get("represented_dtype"), str)
    ):
        raise ValueError("causal runner audit must record represented device and dtype")
    if request.sign:
        hashes = vector_audit_hashes(
            np.asarray(request.vector, dtype=np.float64),
            represented_dtype=audit["represented_dtype"],
        )
        if (
            audit.get("source_vector_sha256") != hashes.source_vector_sha256
            or audit.get("applied_vector_sha256") != hashes.applied_vector_sha256
            or audit.get("layer_id") != request.layer_id
        ):
            raise ValueError("causal runner vector audit does not match request")
        margin_audits = audit.get("margin_forward_audits")
        if not isinstance(margin_audits, list) or len(margin_audits) != 2:
            raise ValueError("causal margin forwards require two runtime audits")
        for margin_audit in margin_audits:
            if not isinstance(margin_audit, Mapping) or (
                margin_audit.get("source_vector_sha256")
                != hashes.source_vector_sha256
                or margin_audit.get("applied_vector_sha256")
                != hashes.applied_vector_sha256
                or margin_audit.get("rendered_prompt_sha256")
                != prompt.rendered_prompt_sha256
                or margin_audit.get("example_id") != prompt.example_id
                or margin_audit.get("layer_id") != request.layer_id
                or margin_audit.get("position") != audit.get("position")
                or margin_audit.get("prompt_token_ids_sha256")
                != audit.get("prompt_token_ids_sha256")
                or margin_audit.get("model_prefix_token_ids_sha256")
                != audit.get("model_prefix_token_ids_sha256")
                or margin_audit.get("hook_call_count") != 1
                or margin_audit.get("modified_site_count") != 1
                or margin_audit.get("hook_cleanup_verified") is not True
            ):
                raise ValueError("causal margin runtime audit does not match request")


def _observe(runner: CausalRunner, prompt: Any, request: InterventionRequest) -> RunObservation:
    observation = runner.observe(prompt, request)
    _validate_observation(prompt, request, observation)
    return observation


def _valid_generation(text: str) -> bool:
    stripped = text.strip()
    return stripped == "UNKNOWN" or bool(re.fullmatch(r"Z[0-9]{4}", stripped))


def _validation_candidate(
    *,
    runner: CausalRunner,
    prompts: tuple[Any, ...],
    direction: Any,
    multiplier: float,
    identity_sha256: str,
    provenance: CausalExpectedProvenance,
) -> tuple[ValidationCandidate, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_unit: dict[str, list[tuple[Any, RunObservation, RunObservation]]] = {}
    invalid = 0
    baseline_bound_correct = 0
    preserved_bound_correct = 0
    bound_count = 0
    scaled = direction.natural_scale * multiplier * direction.vector
    for prompt in prompts:
        baseline_request = _request(
            purpose="validation_baseline",
            control="no_intervention",
            layer_id=direction.layer_id,
            multiplier=multiplier,
            sign=0,
            anchor=CAUSAL_DIRECTION_ANCHOR,
            vector=(),
            identity_sha256=identity_sha256,
        )
        baseline = _observe(runner, prompt, baseline_request)
        sign = 1 if prompt.answerability == "target_unbound" else -1
        intervention_request = _request(
            purpose="validation_grid",
            control="primary",
            layer_id=direction.layer_id,
            multiplier=multiplier,
            sign=sign,
            anchor=CAUSAL_DIRECTION_ANCHOR,
            vector=sign * scaled,
            identity_sha256=identity_sha256,
        )
        intervention = _observe(runner, prompt, intervention_request)
        invalid += int(not _valid_generation(intervention.generated_text))
        preservation = None
        if prompt.answerability == "target_bound":
            bound_count += 1
            baseline_bound_correct += int(
                baseline.generated_text.strip() == prompt.registry_code
            )
            preservation_request = _request(
                purpose="validation_preservation",
                control="primary",
                layer_id=direction.layer_id,
                multiplier=multiplier,
                sign=1,
                anchor=CAUSAL_DIRECTION_ANCHOR,
                vector=scaled,
                identity_sha256=identity_sha256,
            )
            preservation = _observe(runner, prompt, preservation_request)
            preserved_bound_correct += int(
                preservation.generated_text.strip() == prompt.registry_code
            )
        by_unit.setdefault(prompt.entity_unit_id, []).append(
            (prompt, baseline, intervention)
        )
        rows.append(
            {
                "example_id": prompt.example_id,
                "unit_id": prompt.entity_unit_id,
                "exposure": prompt.exposure,
                "answerability": prompt.answerability,
                "baseline": _observation_record(baseline),
                "intervention": _observation_record(intervention),
                "preservation": (
                    _observation_record(preservation) if preservation is not None else None
                ),
            }
        )
    unit_effects = []
    for unit_id, unit_rows in sorted(by_unit.items()):
        unbound = [
            intervention.raw_margin - baseline.raw_margin
            for prompt, baseline, intervention in unit_rows
            if prompt.answerability == "target_unbound"
        ]
        bound = [
            baseline.raw_margin - intervention.raw_margin
            for prompt, baseline, intervention in unit_rows
            if prompt.answerability == "target_bound"
        ]
        if len(unbound) != 2 or len(bound) != 2:
            raise ValueError("validation candidate has an incomplete factorial unit")
        unit_effects.append((unit_id, 0.5 * (float(np.mean(unbound)) + float(np.mean(bound)))))
    candidate = ValidationCandidate(
        layer_id=direction.layer_id,
        multiplier=multiplier,
        unit_effects=tuple(unit_effects),
        invalid_output_rate=invalid / len(prompts),
        bound_accuracy_drop=(baseline_bound_correct - preserved_bound_correct) / bound_count,
        corpus_sha256=provenance.corpus_sha256,
        direction_bundle_sha256=provenance.direction_bundle_sha256,
        model_sha256=provenance.model_sha256,
        tokenizer_sha256=provenance.tokenizer_sha256,
        direction_sha256=direction.direction_sha256,
    )
    return candidate, rows


def _observation_record(observation: RunObservation) -> dict[str, Any]:
    return {
        "raw_margin": observation.raw_margin,
        "length_normalized_margin": observation.length_normalized_margin,
        "generated_text": observation.generated_text,
        "primary_projection_delta": observation.primary_projection_delta,
        "audit": dict(observation.audit),
    }


def _candidate_record(candidate: ValidationCandidate) -> dict[str, Any]:
    value = asdict(candidate)
    value["unit_effects"] = [list(item) for item in candidate.unit_effects]
    return value


def _candidate_from_record(value: Mapping[str, Any]) -> ValidationCandidate:
    return ValidationCandidate(
        **{
            **dict(value),
            "unit_effects": tuple(tuple(item) for item in value["unit_effects"]),
        }
    )


def _selection_record(
    selection: ValidationSelection,
    *,
    prepare: Mapping[str, Any],
    runtime_sha256: str,
    request_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "causal_validation_selection",
        "status": "selected",
        "config_sha256": prepare["config_sha256"],
        "implementation_sha256": prepare["implementation_sha256"],
        "model_sha256": selection.model_sha256,
        "tokenizer_sha256": selection.tokenizer_sha256,
        "corpus_sha256": selection.corpus_sha256,
        "direction_bundle_sha256": selection.direction_bundle_sha256,
        "selection_sha256": selection.selection_sha256,
        "runtime_sha256": runtime_sha256,
        "split_sha256": prepare["split_sha256"],
        "control_sha256": prepare["control_sha256"],
        "request_sha256": request_sha256,
        "layer_id": selection.layer_id,
        "multiplier": selection.multiplier,
        "mean_bidirectional_effect": selection.mean_bidirectional_effect,
        "direction_sha256": selection.direction_sha256,
    }


_RANDOM_SEEDS = (2026080401, 2026080402, 2026080403, 2026080404, 2026080405)


def _random_control_vectors(primary: np.ndarray) -> tuple[tuple[float, ...], ...]:
    norm = float(np.linalg.norm(primary))
    values = []
    for seed in _RANDOM_SEEDS:
        generator = np.random.default_rng(seed)
        candidate = generator.normal(size=primary.shape)
        candidate = candidate - float(candidate @ primary) / float(primary @ primary) * primary
        candidate_norm = float(np.linalg.norm(candidate))
        if not np.isfinite(candidate_norm) or candidate_norm <= 1e-12:
            raise ValueError("random control vector could not be orthogonalized")
        values.append(tuple((candidate * (norm / candidate_norm)).tolist()))
    return tuple(values)


def _label_shuffle_record(artifact: LabelShuffledDirectionArtifact) -> dict[str, Any]:
    return {
        "v3_manifest_path": str(artifact.v3_manifest_path),
        "activation_manifest_path": str(artifact.activation_manifest_path),
        "expected_model_id": artifact.expected_model_id,
        "expected_model_revision": artifact.expected_model_revision,
        "expected_tokenizer_id": artifact.expected_tokenizer_id,
        "expected_tokenizer_revision": artifact.expected_tokenizer_revision,
        "expected_chat_template_sha256": artifact.expected_chat_template_sha256,
        "source": asdict(artifact.source),
        "layer_id": artifact.layer_id,
        "unit_permutation": list(artifact.unit_permutation),
        "vector": artifact.vector.tolist(),
        "artifact_sha256": artifact.artifact_sha256,
    }


def _seal_record(
    seal: CausalEvaluationSeal,
    *,
    prepare: Mapping[str, Any],
    request_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "causal_evaluation_seal",
        "status": "selected_and_sealed",
        "config_sha256": prepare["config_sha256"],
        "implementation_sha256": prepare["implementation_sha256"],
        "model_sha256": seal.model_sha256,
        "tokenizer_sha256": seal.tokenizer_sha256,
        "corpus_sha256": seal.corpus_sha256,
        "direction_bundle_sha256": seal.direction_bundle_sha256,
        "selection_sha256": seal.selection_sha256,
        "runtime_sha256": seal.runtime_sha256,
        "output_contract_sha256": seal.output_contract_sha256,
        "split_sha256": prepare["split_sha256"],
        "control_sha256": prepare["control_sha256"],
        "request_sha256": request_sha256,
        "direction_sha256": seal.direction_sha256,
        "layer_id": seal.layer_id,
        "multiplier": seal.multiplier,
        "anchor": seal.anchor,
        "random_seeds": list(seal.random_seeds),
        "expected_unit_ids_by_split": {
            split: list(unit_ids)
            for split, unit_ids in seal.expected_unit_ids_by_split.items()
        },
        "primary_vector": list(seal.primary_vector),
        "label_shuffle_artifact": _label_shuffle_record(seal.label_shuffle_artifact),
        "random_vectors": [list(vector) for vector in seal.random_vectors],
        "control_vector_artifact_hashes": dict(seal.control_vector_artifact_hashes),
        "seal_sha256": seal.seal_sha256,
    }


def run_causal_validation(
    args: argparse.Namespace,
    *,
    dependencies: CausalDependencies | None = None,
) -> dict[str, Any]:
    dependencies = dependencies or CausalDependencies()
    config, prepare, corpus, bundle, _binding = _load_prepared(args, dependencies)
    runner = dependencies.runner_factory(config)
    validation_prompts = tuple(
        row for row in corpus.prompts if row.split == "causal_validation"
    )
    first_direction = bundle.directions[0]
    smoke_request = _request(
        purpose="unprotected_smoke",
        control="primary",
        layer_id=first_direction.layer_id,
        multiplier=CAUSAL_VALIDATION_MULTIPLIERS[0],
        sign=1,
        anchor=CAUSAL_DIRECTION_ANCHOR,
        vector=(
            CAUSAL_VALIDATION_MULTIPLIERS[0]
            * first_direction.natural_scale
            * first_direction.vector
        ),
        identity_sha256=prepare["request_sha256"],
    )
    runtime = runner.smoke(validation_prompts[0], smoke_request)
    if not isinstance(runtime, RuntimeReceipt):
        raise ValueError("causal runner returned an untyped smoke receipt")
    if runtime.smoke_request_sha256 != smoke_request.request_sha256:
        raise ValueError("runtime smoke receipt does not match smoke request")
    runtime_hash = runtime.runtime_sha256
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = Path(args.root).absolute() / output_root
    identity_hash = _sha256_json(
        {
            "prepare_request_sha256": prepare["request_sha256"],
            "runtime_sha256": runtime_hash,
            "split": "causal_validation",
        }
    )
    smoke_payload = {
        "kind": "causal_runtime_smoke",
        **{field: prepare[field] for field in (
            "config_sha256",
            "implementation_sha256",
            "model_sha256",
            "tokenizer_sha256",
            "corpus_sha256",
            "direction_bundle_sha256",
            "selection_sha256",
            "split_sha256",
            "control_sha256",
        )},
        "runtime_sha256": runtime_hash,
        "runtime": asdict(runtime),
    }
    AtomicJSONReceiptStore(output_root).write_or_resume(
        "raw/runtime-smoke.json",
        request_sha256=smoke_request.request_sha256,
        payload=smoke_payload,
        resume=args.resume,
    )
    provenance = _expected_provenance(config, prepare, bundle)
    candidates = []
    raw_store = AtomicJSONReceiptStore(output_root)
    locked_mode = config.validation_selection.get("mode") == "locked_from_v1"
    if locked_mode:
        validation_directions = tuple(
            direction
            for direction in bundle.directions
            if direction.layer_id == config.validation_selection["locked_layer"]
        )
        validation_multipliers = (
            float(config.validation_selection["locked_multiplier"]),
        )
    else:
        validation_directions = bundle.directions
        validation_multipliers = CAUSAL_VALIDATION_MULTIPLIERS
    for direction in validation_directions:
        for multiplier in validation_multipliers:
            candidate_request_hash = _sha256_json(
                {
                    "command": "fa-causal-run-validation",
                    "identity_sha256": identity_hash,
                    "layer_id": direction.layer_id,
                    "multiplier": multiplier,
                }
            )
            relative = f"raw/candidate-layer-{direction.layer_id}-multiplier-{multiplier:g}.json"
            path = output_root / relative
            if path.exists():
                receipt = raw_store.write_or_resume(
                    relative,
                    request_sha256=candidate_request_hash,
                    payload={},
                    resume=args.resume,
                )
                candidate = _candidate_from_record(receipt.payload["candidate"])
            else:
                candidate, rows = _validation_candidate(
                    runner=runner,
                    prompts=validation_prompts,
                    direction=direction,
                    multiplier=multiplier,
                    identity_sha256=identity_hash,
                    provenance=provenance,
                )
                receipt = raw_store.write_or_resume(
                    relative,
                    request_sha256=candidate_request_hash,
                    payload={
                        "kind": "causal_validation_candidate_evidence",
                        "config_sha256": prepare["config_sha256"],
                        "implementation_sha256": prepare["implementation_sha256"],
                        "model_sha256": prepare["model_sha256"],
                        "tokenizer_sha256": prepare["tokenizer_sha256"],
                        "corpus_sha256": prepare["corpus_sha256"],
                        "direction_bundle_sha256": prepare["direction_bundle_sha256"],
                        "selection_sha256": _UNSELECTED_SHA256,
                        "runtime_sha256": runtime_hash,
                        "split_sha256": prepare["split_sha256"],
                        "control_sha256": prepare["control_sha256"],
                        "split": "causal_validation",
                        "candidate": _candidate_record(candidate),
                        "rows": rows,
                    },
                    resume=args.resume,
                )
            candidates.append(candidate)
    if locked_mode:
        selection = lock_causal_intervention(
            candidates[0],
            corpus,
            provenance,
            layer_id=int(config.validation_selection["locked_layer"]),
            multiplier=float(config.validation_selection["locked_multiplier"]),
        )
    else:
        selection = select_causal_intervention(candidates, corpus, provenance)
    selection_request_hash = _sha256_json(
        {
            "identity_sha256": identity_hash,
            "candidate_request_sha256s": sorted(
                _sha256_json(_candidate_record(candidate)) for candidate in candidates
            ),
            "selection_sha256": selection.selection_sha256,
        }
    )
    selection_path = _write_hashed_manifest(
        output_root / "causal_validation_selection.json",
        _selection_record(
            selection,
            prepare=prepare,
            runtime_sha256=runtime_hash,
            request_sha256=selection_request_hash,
        ),
    )
    training = load_v3_training_direction_inputs(
        v3_manifest_path=prepare["v3_corpus_manifest"],
        activation_manifest_path=prepare["v3_training_activation_manifest"],
        expected_model_id=config.model_id,
        expected_model_revision=config.model_revision,
        expected_tokenizer_id=config.model_id,
        expected_tokenizer_revision=config.tokenizer_revision,
        expected_chat_template_sha256=config.chat_template_sha256,
    )
    unit_ids = sorted({row.entity_unit_id for row in training.prompts})
    shuffled = list(unit_ids)
    np.random.default_rng(config.split_seed).shuffle(shuffled)
    label_shuffle = build_label_shuffled_direction(
        training,
        layer_id=selection.layer_id,
        unit_permutation=shuffled,
    )
    selected_direction = next(
        direction for direction in bundle.directions if direction.layer_id == selection.layer_id
    )
    random_vectors = _random_control_vectors(selected_direction.vector)
    expected_test_units = {
        split: sorted(
            {
                row.entity_unit_id
                for row in corpus.prompts
                if row.split == split
            }
        )
        for split in ("causal_entity_test", "causal_template_test")
    }
    seal = seal_causal_evaluation(
        selection=selection,
        expected_provenance=provenance,
        runtime_sha256=runtime_hash,
        output_contract_sha256=_sha256_json(
            {"output_contract": validation_prompts[0].output_contract}
        ),
        random_seeds=_RANDOM_SEEDS,
        expected_unit_ids_by_split=expected_test_units,
        primary_vector=selected_direction.vector,
        label_shuffle_artifact=label_shuffle,
        random_vectors=random_vectors,
    )
    seal_request_hash = _sha256_json(
        {
            "selection_request_sha256": selection_request_hash,
            "selection_sha256": selection.selection_sha256,
            "runtime_sha256": runtime_hash,
            "label_shuffle_artifact_sha256": label_shuffle.artifact_sha256,
            "random_seeds": _RANDOM_SEEDS,
        }
    )
    seal_path = _write_hashed_manifest(
        output_root / "causal_evaluation_seal.json",
        _seal_record(seal, prepare=prepare, request_sha256=seal_request_hash),
    )
    return {
        "status": "selected_and_sealed",
        "selection_manifest": str(selection_path.absolute()),
        "seal_manifest": str(seal_path.absolute()),
        "selection_sha256": selection.selection_sha256,
        "seal_sha256": seal.seal_sha256,
    }


def _label_shuffle_from_record(value: Mapping[str, Any]) -> LabelShuffledDirectionArtifact:
    return LabelShuffledDirectionArtifact(
        v3_manifest_path=value["v3_manifest_path"],
        activation_manifest_path=value["activation_manifest_path"],
        expected_model_id=value["expected_model_id"],
        expected_model_revision=value["expected_model_revision"],
        expected_tokenizer_id=value["expected_tokenizer_id"],
        expected_tokenizer_revision=value["expected_tokenizer_revision"],
        expected_chat_template_sha256=value["expected_chat_template_sha256"],
        source=V3DirectionProvenance(**value["source"]),
        layer_id=value["layer_id"],
        unit_permutation=tuple(value["unit_permutation"]),
        vector=np.asarray(value["vector"], dtype=np.float64),
        artifact_sha256=value["artifact_sha256"],
    )


def _load_evaluation_seal(
    path: str | Path,
    *,
    prepare: Mapping[str, Any],
) -> tuple[CausalEvaluationSeal, dict[str, Any]]:
    value = _read_hashed_manifest(path, kind="causal_evaluation_seal")
    expected = {
        field: prepare[field]
        for field in (
            "config_sha256",
            "implementation_sha256",
            "model_sha256",
            "tokenizer_sha256",
            "corpus_sha256",
            "direction_bundle_sha256",
            "split_sha256",
            "control_sha256",
        )
    }
    if any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise ValueError("evaluation seal does not match prepared identity")
    seal = CausalEvaluationSeal(
        corpus_sha256=value["corpus_sha256"],
        selection_sha256=value["selection_sha256"],
        direction_bundle_sha256=value["direction_bundle_sha256"],
        direction_sha256=value["direction_sha256"],
        model_sha256=value["model_sha256"],
        tokenizer_sha256=value["tokenizer_sha256"],
        runtime_sha256=value["runtime_sha256"],
        output_contract_sha256=value["output_contract_sha256"],
        layer_id=value["layer_id"],
        multiplier=value["multiplier"],
        anchor=value["anchor"],
        random_seeds=tuple(value["random_seeds"]),
        expected_unit_ids_by_split={
            split: tuple(unit_ids)
            for split, unit_ids in value["expected_unit_ids_by_split"].items()
        },
        primary_vector=tuple(value["primary_vector"]),
        label_shuffle_artifact=_label_shuffle_from_record(value["label_shuffle_artifact"]),
        random_vectors=tuple(tuple(vector) for vector in value["random_vectors"]),
        control_vector_artifact_hashes=value["control_vector_artifact_hashes"],
        seal_sha256=value["seal_sha256"],
    )
    return seal, value


def _shard_relative_path(
    split: str,
    control: str,
    unit_id: str,
    member: int | None,
) -> str:
    member_name = "" if member is None else f"-member-{member}"
    return f"{split}/{control}{member_name}/{unit_id}.json"


def _control_execution(
    seal: CausalEvaluationSeal,
    *,
    control: str,
    answerability: str,
    member: int | None,
) -> tuple[int, int, str, np.ndarray]:
    primary_sign = 1 if answerability == "target_unbound" else -1
    if control in {"baseline", "no_intervention"}:
        return 0, seal.layer_id, seal.anchor, np.asarray((), dtype=np.float64)
    sign = -primary_sign if control == "sign_reversed" else primary_sign
    layer = seal.layer_id
    anchor = seal.anchor
    vector = np.asarray(seal.primary_vector, dtype=np.float64)
    if control == "label_shuffled_direction":
        vector = np.asarray(seal.label_shuffle_artifact.vector, dtype=np.float64)
    elif control == "norm_matched_random":
        if member is None:
            raise ValueError("norm_matched_random requires a member")
        vector = np.asarray(seal.random_vectors[member], dtype=np.float64)
    elif control == "wrong_anchor":
        anchor = "target_intro_end"
    elif control == "wrong_layer":
        layer = deterministic_farthest_layer(seal.layer_id)
    return sign, layer, anchor, vector


def _replay_runtime_smoke(
    runner: CausalRunner,
    *,
    prepare: Mapping[str, Any],
    corpus: Any,
    bundle: Any,
) -> RuntimeReceipt:
    prompt = next(row for row in corpus.prompts if row.split == "causal_validation")
    direction = bundle.directions[0]
    request = _request(
        purpose="unprotected_smoke",
        control="primary",
        layer_id=direction.layer_id,
        multiplier=CAUSAL_VALIDATION_MULTIPLIERS[0],
        sign=1,
        anchor=CAUSAL_DIRECTION_ANCHOR,
        vector=(
            CAUSAL_VALIDATION_MULTIPLIERS[0]
            * direction.natural_scale
            * direction.vector
        ),
        identity_sha256=prepare["request_sha256"],
    )
    receipt = runner.smoke(prompt, request)
    if not isinstance(receipt, RuntimeReceipt) or receipt.smoke_request_sha256 != request.request_sha256:
        raise ValueError("runtime smoke receipt does not match the sealed smoke request")
    return receipt


def _shard_request_sha256(
    prepare: Mapping[str, Any],
    seal: CausalEvaluationSeal,
    *,
    split: str,
    control: str,
    unit_id: str,
    member: int | None,
) -> str:
    return _sha256_json(
        {
            "command": "fa-causal-run-shard",
            "config_sha256": prepare["config_sha256"],
            "implementation_sha256": prepare["implementation_sha256"],
            "corpus_sha256": seal.corpus_sha256,
            "direction_bundle_sha256": seal.direction_bundle_sha256,
            "selection_sha256": seal.selection_sha256,
            "runtime_sha256": seal.runtime_sha256,
            "seal_sha256": seal.seal_sha256,
            "split": split,
            "control": control,
            "unit_id": unit_id,
            "member": member,
        }
    )


def run_causal_shard(
    args: argparse.Namespace,
    *,
    dependencies: CausalDependencies | None = None,
) -> dict[str, Any]:
    dependencies = dependencies or CausalDependencies()
    config, prepare, corpus, bundle, _binding = _load_prepared(args, dependencies)
    seal, seal_record = _load_evaluation_seal(args.seal_manifest, prepare=prepare)
    if args.split not in seal.expected_unit_ids_by_split:
        raise ValueError("shard split is not registered in the seal")
    if args.unit_id not in seal.expected_unit_ids_by_split[args.split]:
        raise ValueError("shard unit is not registered in the sealed split")
    if args.control == "norm_matched_random":
        if args.member not in range(5):
            raise ValueError("norm_matched_random requires one registered member")
    elif args.member is not None:
        raise ValueError("only norm_matched_random may specify a member")
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = Path(args.root).absolute() / output_root
    relative = _shard_relative_path(args.split, args.control, args.unit_id, args.member)
    request_hash = _shard_request_sha256(
        prepare,
        seal,
        split=args.split,
        control=args.control,
        unit_id=args.unit_id,
        member=args.member,
    )
    store = AtomicJSONReceiptStore(output_root)
    receipt_path = output_root / relative
    if receipt_path.exists():
        receipt = store.write_or_resume(
            relative,
            request_sha256=request_hash,
            payload={},
            resume=args.resume,
        )
        return {
            "status": "resumed",
            "shard_receipt": str(receipt.path),
            "request_sha256": request_hash,
        }
    runner = dependencies.runner_factory(config)
    runtime_identity = getattr(runner, "runtime_identity", None)
    runtime = (
        runtime_identity(request_hash)
        if callable(runtime_identity)
        else _replay_runtime_smoke(
            runner,
            prepare=prepare,
            corpus=corpus,
            bundle=bundle,
        )
    )
    if runtime.runtime_sha256 != seal.runtime_sha256:
        raise ValueError("current runtime smoke does not match the sealed runtime")
    selected_direction = next(
        direction for direction in bundle.directions if direction.layer_id == seal.layer_id
    )
    prompts = tuple(
        row
        for row in corpus.prompts
        if row.split == args.split and row.entity_unit_id == args.unit_id
    )
    if len(prompts) != 4:
        raise ValueError("shard requires one complete sealed 2x2 unit")
    rows = []
    for prompt in prompts:
        sign, layer, anchor, control_vector = _control_execution(
            seal,
            control=args.control,
            answerability=prompt.answerability,
            member=args.member,
        )
        applied = sign * seal.multiplier * selected_direction.natural_scale * control_vector
        request = _request(
            purpose="causal_test_shard",
            control=args.control,
            layer_id=layer,
            multiplier=seal.multiplier,
            sign=sign,
            anchor=anchor,
            vector=applied,
            identity_sha256=request_hash,
        )
        observation = _observe(runner, prompt, request)
        preservation = None
        if args.control == "primary" and prompt.answerability == "target_bound":
            preservation_request = _request(
                purpose="causal_test_bound_preservation",
                control="primary",
                layer_id=seal.layer_id,
                multiplier=seal.multiplier,
                sign=1,
                anchor=seal.anchor,
                vector=(
                    seal.multiplier
                    * selected_direction.natural_scale
                    * np.asarray(seal.primary_vector, dtype=np.float64)
                ),
                identity_sha256=request_hash,
            )
            preservation = _observe(runner, prompt, preservation_request)
        rows.append(
            {
                "example_id": prompt.example_id,
                "unit_id": prompt.entity_unit_id,
                "split": prompt.split,
                "exposure": prompt.exposure,
                "answerability": prompt.answerability,
                "registry_code": prompt.registry_code,
                "observation": _observation_record(observation),
                "bound_preservation": (
                    _observation_record(preservation) if preservation is not None else None
                ),
            }
        )
    unrelated = None
    if args.control == "primary":
        preserve = getattr(runner, "unrelated_preservation", None)
        if not callable(preserve):
            raise ValueError("causal runner must execute the unrelated preservation set")
        preservation_request = _request(
            purpose="causal_test_unrelated_preservation",
            control="primary",
            layer_id=seal.layer_id,
            multiplier=seal.multiplier,
            sign=1,
            anchor=seal.anchor,
            vector=(
                seal.multiplier
                * selected_direction.natural_scale
                * np.asarray(seal.primary_vector, dtype=np.float64)
            ),
            identity_sha256=request_hash,
        )
        unrelated = preserve(preservation_request)
        if (
            not isinstance(unrelated, Mapping)
            or type(unrelated.get("passed")) is not bool
            or not isinstance(unrelated.get("rows"), list)
        ):
            raise ValueError("unrelated preservation result is invalid")
    payload = {
        "kind": "causal_test_unit_shard",
        "status": "completed",
        "config_sha256": prepare["config_sha256"],
        "implementation_sha256": prepare["implementation_sha256"],
        "model_sha256": seal.model_sha256,
        "tokenizer_sha256": seal.tokenizer_sha256,
        "corpus_sha256": seal.corpus_sha256,
        "direction_bundle_sha256": seal.direction_bundle_sha256,
        "selection_sha256": seal.selection_sha256,
        "runtime_sha256": seal.runtime_sha256,
        "split_sha256": prepare["split_sha256"],
        "control_sha256": prepare["control_sha256"],
        "seal_sha256": seal.seal_sha256,
        "split": args.split,
        "control": args.control,
        "unit_id": args.unit_id,
        "member": args.member,
        "rows": rows,
        "unrelated_preservation": unrelated,
        "seal_manifest_sha256": seal_record["manifest_sha256"],
    }
    receipt = store.write_or_resume(
        relative,
        request_sha256=request_hash,
        payload=payload,
        resume=args.resume,
    )
    return {
        "status": "resumed" if receipt.resumed else "completed",
        "shard_receipt": str(receipt.path),
        "request_sha256": request_hash,
    }


def expected_causal_shards(
    seal: Mapping[str, Any] | CausalEvaluationSeal,
) -> tuple[dict[str, Any], ...]:
    expected_units = (
        seal.expected_unit_ids_by_split
        if isinstance(seal, CausalEvaluationSeal)
        else seal["expected_unit_ids_by_split"]
    )
    schedule = (
        ("baseline", None),
        ("primary", None),
        *((control, None) for control in CAUSAL_CONTROLS if control != "norm_matched_random"),
        *(("norm_matched_random", member) for member in range(5)),
    )
    rows = []
    for split in ("causal_entity_test", "causal_template_test"):
        for unit_id in expected_units[split]:
            for control, member in schedule:
                rows.append(
                    {
                        "split": split,
                        "unit_id": unit_id,
                        "control": control,
                        "member": member,
                        "relative_path": _shard_relative_path(
                            split, control, unit_id, member
                        ),
                    }
                )
    return tuple(rows)


def _generation_result(
    generated_text: str,
    *,
    expected_code: str,
    all_codes: frozenset[str],
) -> GenerationResult:
    value = generated_text.strip()
    if value == expected_code:
        response_class = GenerationClass.CORRECT_CODE
    elif value == "UNKNOWN":
        response_class = GenerationClass.UNKNOWN
    elif re.fullmatch(r"Z[0-9]{4}", value):
        response_class = GenerationClass.OTHER_CODE
    else:
        response_class = GenerationClass.INVALID
    return GenerationResult(
        response_class=response_class,
        format_valid=response_class != GenerationClass.INVALID,
        copied_from_other_unit=(value in all_codes and value != expected_code),
    )


def _score_audit(
    seal: CausalEvaluationSeal,
    control: str,
    observation: Mapping[str, Any],
    *,
    member: int | None = None,
) -> ExecutionAuditHashes:
    represented_dtype = observation["audit"].get("represented_dtype", "torch.float64")
    registered_control = "no_intervention" if control == "baseline" else control
    return ExecutionAuditHashes.for_control(
        seal,
        registered_control,
        member=member,
        represented_dtype=represented_dtype,
    )


def _observation_from_record(value: Mapping[str, Any]) -> RunObservation:
    return RunObservation(
        raw_margin=value["raw_margin"],
        length_normalized_margin=value["length_normalized_margin"],
        generated_text=value["generated_text"],
        primary_projection_delta=value["primary_projection_delta"],
        audit=value["audit"],
    )


def _verify_shard_runtime_evidence(
    payload: Mapping[str, Any],
    *,
    prepare: Mapping[str, Any],
    seal: CausalEvaluationSeal,
    corpus: Any,
    bundle: Any,
    tokenizer: Any,
) -> None:
    split = payload["split"]
    control = payload["control"]
    unit_id = payload["unit_id"]
    member = payload["member"]
    request_hash = _shard_request_sha256(
        prepare,
        seal,
        split=split,
        control=control,
        unit_id=unit_id,
        member=member,
    )
    if payload.get("request_sha256") != request_hash:
        raise ValueError("causal shard request hash does not match the sealed schedule")

    expected_prompts = {
        prompt.example_id: prompt
        for prompt in corpus.prompts
        if prompt.split == split and prompt.entity_unit_id == unit_id
    }
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("all registered shards must contain one complete 2x2 unit")
    if {row.get("example_id") for row in rows} != set(expected_prompts):
        raise ValueError("causal shard examples do not match the sealed corpus")
    selected_direction = next(
        direction for direction in bundle.directions if direction.layer_id == seal.layer_id
    )
    for row in rows:
        prompt = expected_prompts[row["example_id"]]
        expected_fields = {
            "unit_id": prompt.entity_unit_id,
            "split": prompt.split,
            "exposure": prompt.exposure,
            "answerability": prompt.answerability,
            "registry_code": prompt.registry_code,
        }
        if any(row.get(field) != value for field, value in expected_fields.items()):
            raise ValueError("causal shard row does not match the sealed prompt")
        sign, layer, anchor, control_vector = _control_execution(
            seal,
            control=control,
            answerability=prompt.answerability,
            member=member,
        )
        applied = sign * seal.multiplier * selected_direction.natural_scale * control_vector
        request = _request(
            purpose="causal_test_shard",
            control=control,
            layer_id=layer,
            multiplier=seal.multiplier,
            sign=sign,
            anchor=anchor,
            vector=applied,
            identity_sha256=request_hash,
        )
        observation = _observation_from_record(row["observation"])
        _validate_observation(
            prompt,
            request,
            observation,
        )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        resolved = resolve_causal_anchor(
            prompt,
            rendered,
            tokenizer,
            anchor_name=request.anchor,
        )
        if request.sign and (
            observation.audit.get("position") != resolved.position
            or observation.audit.get("prompt_token_ids_sha256")
            != resolved.prompt_token_ids_sha256
            or observation.audit.get("model_prefix_token_ids_sha256")
            != resolved.prompt_token_ids_sha256
        ):
            raise ValueError("causal runtime site does not match the sealed anchor")
        preservation = row.get("bound_preservation")
        needs_preservation = control == "primary" and prompt.answerability == "target_bound"
        if needs_preservation != (preservation is not None):
            raise ValueError("causal bound-preservation schedule is incomplete")
        if needs_preservation:
            preservation_request = _request(
                purpose="causal_test_bound_preservation",
                control="primary",
                layer_id=seal.layer_id,
                multiplier=seal.multiplier,
                sign=1,
                anchor=seal.anchor,
                vector=(
                    seal.multiplier
                    * selected_direction.natural_scale
                    * np.asarray(seal.primary_vector, dtype=np.float64)
                ),
                identity_sha256=request_hash,
            )
            _validate_observation(
                prompt,
                preservation_request,
                _observation_from_record(preservation),
            )
            preservation_observation = _observation_from_record(preservation)
            preservation_resolved = resolve_causal_anchor(
                prompt,
                rendered,
                tokenizer,
                anchor_name=preservation_request.anchor,
            )
            if (
                preservation_observation.audit.get("position")
                != preservation_resolved.position
                or preservation_observation.audit.get("prompt_token_ids_sha256")
                != preservation_resolved.prompt_token_ids_sha256
                or preservation_observation.audit.get("model_prefix_token_ids_sha256")
                != preservation_resolved.prompt_token_ids_sha256
            ):
                raise ValueError(
                    "causal preservation site does not match the sealed anchor"
                )

    unrelated = payload.get("unrelated_preservation")
    if control == "primary":
        if not isinstance(unrelated, Mapping) or not isinstance(unrelated.get("rows"), list):
            raise ValueError("causal unrelated-preservation evidence is missing")
        expected_ids = {f"unrelated-code-lookup-{index}" for index in range(4)}
        if {row.get("prompt_id") for row in unrelated["rows"]} != expected_ids:
            raise ValueError("causal unrelated-preservation schedule is incomplete")
        recomputed_passed = True
        for row in unrelated["rows"]:
            index = int(row["prompt_id"].rsplit("-", 1)[1])
            expected = f"U{index:04d}"
            if (
                row.get("expected") != expected
                or not isinstance(row.get("baseline"), str)
                or not isinstance(row.get("generated"), str)
            ):
                raise ValueError("causal unrelated-preservation target is invalid")
            recomputed_passed = recomputed_passed and (
                row["baseline"] == expected
                and row.get("generated") == expected
            )
        if unrelated.get("passed") is not recomputed_passed:
            raise ValueError("causal unrelated-preservation gate does not match rows")
    elif unrelated is not None:
        raise ValueError("unregistered control carried unrelated-preservation evidence")


def _evidence_from_receipts(
    split: str,
    seal: CausalEvaluationSeal,
    receipts: Mapping[tuple[str, str, str, int | None], Mapping[str, Any]],
) -> CausalEvidence:
    all_codes = frozenset(
        row["registry_code"]
        for (row_split, _control, _unit, _member), payload in receipts.items()
        if row_split == split
        for row in payload["rows"]
    )
    baselines: list[BaselineScore] = []
    primary: list[PrimaryScore] = []
    controls: list[ControlScore] = []
    manipulation: list[ManipulationCheck] = []
    baseline_bound = []
    preserved_bound = []
    unrelated = []

    for unit_id in seal.expected_unit_ids_by_split[split]:
        baseline_payload = receipts[(split, "baseline", unit_id, None)]
        primary_payload = receipts[(split, "primary", unit_id, None)]
        primary_deltas = []
        for row in baseline_payload["rows"]:
            observation = row["observation"]
            generation = _generation_result(
                observation["generated_text"],
                expected_code=row["registry_code"],
                all_codes=all_codes,
            )
            baselines.append(
                BaselineScore(
                    unit_id=unit_id,
                    split=split,
                    exposure=row["exposure"],
                    answerability=row["answerability"],
                    raw_margin=observation["raw_margin"],
                    length_normalized_margin=observation["length_normalized_margin"],
                    generation=generation,
                    audit=_score_audit(seal, "baseline", observation),
                )
            )
            if row["answerability"] == "target_bound":
                baseline_bound.append(
                    generation.response_class == GenerationClass.CORRECT_CODE
                )

        for row in primary_payload["rows"]:
            observation = row["observation"]
            generation = _generation_result(
                observation["generated_text"],
                expected_code=row["registry_code"],
                all_codes=all_codes,
            )
            sign = 1 if row["answerability"] == "target_unbound" else -1
            primary.append(
                PrimaryScore(
                    unit_id=unit_id,
                    split=split,
                    exposure=row["exposure"],
                    answerability=row["answerability"],
                    raw_margin=observation["raw_margin"],
                    length_normalized_margin=observation["length_normalized_margin"],
                    generation=generation,
                    audit=_score_audit(seal, "primary", observation),
                    sign=sign,
                )
            )
            primary_deltas.append(sign * observation["primary_projection_delta"])
            if row["answerability"] == "target_bound":
                preservation = row.get("bound_preservation")
                preserved_bound.append(
                    preservation is not None
                    and preservation["generated_text"].strip() == row["registry_code"]
                )
        manipulation.append(
            ManipulationCheck(
                unit_id=unit_id,
                split=split,
                primary_projection_delta=float(np.mean(primary_deltas)),
            )
        )
        unrelated_result = primary_payload.get("unrelated_preservation")
        unrelated.append(
            isinstance(unrelated_result, Mapping)
            and unrelated_result.get("passed") is True
        )

        for control in CAUSAL_CONTROLS:
            members = range(5) if control == "norm_matched_random" else (None,)
            for member in members:
                payload = receipts[(split, control, unit_id, member)]
                for row in payload["rows"]:
                    observation = row["observation"]
                    sign = 0 if control == "no_intervention" else (
                        1 if row["answerability"] == "target_unbound" else -1
                    )
                    if control == "sign_reversed":
                        sign *= -1
                    controls.append(
                        ControlScore(
                            unit_id=unit_id,
                            split=split,
                            exposure=row["exposure"],
                            answerability=row["answerability"],
                            raw_margin=observation["raw_margin"],
                            length_normalized_margin=observation[
                                "length_normalized_margin"
                            ],
                            generation=_generation_result(
                                observation["generated_text"],
                                expected_code=row["registry_code"],
                                all_codes=all_codes,
                            ),
                            audit=_score_audit(
                                seal, control, observation, member=member
                            ),
                            sign=sign,
                            control_member=member,
                            control=control,
                        )
                    )

    bound_drop = float(np.mean(baseline_bound) - np.mean(preserved_bound))
    return CausalEvidence(
        split=split,
        seal=seal,
        baselines=tuple(baselines),
        primary_scores=tuple(primary),
        control_scores=tuple(controls),
        manipulation_checks=tuple(manipulation),
        preservation=PreservationResult(
            split=split,
            bound_accuracy_drop=bound_drop,
            unrelated_task_preserved=all(unrelated),
        ),
    )


def _decision_record(decision: Any) -> dict[str, Any]:
    return {
        "status": decision.status,
        "reasons": list(decision.reasons),
        "mean_effect": decision.mean_effect,
        "unbound_effect": decision.unbound_effect,
        "bound_effect": decision.bound_effect,
        "bootstrap": asdict(decision.bootstrap) if decision.bootstrap else None,
        "sign_flip": asdict(decision.sign_flip) if decision.sign_flip else None,
        "control_effects": {
            name: asdict(value) for name, value in decision.control_effects.items()
        },
        "strongest_control": decision.strongest_control,
        "contrast_bootstrap": (
            asdict(decision.contrast_bootstrap)
            if decision.contrast_bootstrap
            else None
        ),
        "length_normalized_sensitivity": (
            asdict(decision.length_normalized_sensitivity)
            if decision.length_normalized_sensitivity
            else None
        ),
    }


def evaluate_causal(
    args: argparse.Namespace,
    *,
    dependencies: CausalDependencies | None = None,
) -> dict[str, Any]:
    dependencies = dependencies or CausalDependencies()
    config, prepare, corpus, bundle, binding = _load_prepared(args, dependencies)
    seal, seal_record = _load_evaluation_seal(args.seal_manifest, prepare=prepare)
    evidence_root = Path(args.evidence_dir)
    if not evidence_root.is_absolute():
        evidence_root = Path(args.root).absolute() / evidence_root

    schedule = expected_causal_shards(seal)
    missing = [row["relative_path"] for row in schedule if not (evidence_root / row["relative_path"]).is_file()]
    if missing:
        raise ValueError(
            f"all registered shards are required before evaluation ({len(missing)} missing)"
        )

    receipts: dict[tuple[str, str, str, int | None], Mapping[str, Any]] = {}
    for item in schedule:
        path = evidence_root / item["relative_path"]
        payload, _receipt_sha256 = AtomicJSONReceiptStore._read(path)
        expected_identity = {
            "kind": "causal_test_unit_shard",
            "status": "completed",
            "config_sha256": prepare["config_sha256"],
            "implementation_sha256": prepare["implementation_sha256"],
            "corpus_sha256": seal.corpus_sha256,
            "direction_bundle_sha256": seal.direction_bundle_sha256,
            "selection_sha256": seal.selection_sha256,
            "runtime_sha256": seal.runtime_sha256,
            "seal_sha256": seal.seal_sha256,
            "split": item["split"],
            "control": item["control"],
            "unit_id": item["unit_id"],
            "member": item["member"],
            "seal_manifest_sha256": seal_record["manifest_sha256"],
            "request_sha256": _shard_request_sha256(
                prepare,
                seal,
                split=item["split"],
                control=item["control"],
                unit_id=item["unit_id"],
                member=item["member"],
            ),
        }
        if any(payload.get(field) != value for field, value in expected_identity.items()):
            raise ValueError("causal shard identity does not match the sealed schedule")
        _verify_shard_runtime_evidence(
            payload,
            prepare=prepare,
            seal=seal,
            corpus=corpus,
            bundle=bundle,
            tokenizer=binding.tokenizer,
        )
        receipts[(item["split"], item["control"], item["unit_id"], item["member"])] = payload

    evidence = {
        split: _evidence_from_receipts(split, seal, receipts)
        for split in ("causal_entity_test", "causal_template_test")
    }
    study = analyze_causal_study(evidence)
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = Path(args.root).absolute() / output_root
    if (output_root / "result.json").exists():
        raise ValueError("completed causal endpoint cannot be reopened")

    analysis_store = CausalAnalysisStore(output_root / "endpoint_state")
    split_decisions = {
        split: analysis_store.complete(value) for split, value in evidence.items()
    }
    if any(
        split_decisions[split].status != study.split_decisions[split].status
        for split in split_decisions
    ):
        raise RuntimeError("causal study and split analyses disagree")
    result = {
        "schema_version": 1,
        "kind": "same_string_answerability_causal_result",
        "study_id": config.study_id,
        "status": study.status,
        "reasons": list(study.reasons),
        "seal_sha256": seal.seal_sha256,
        "split_decisions": {
            split: _decision_record(decision)
            for split, decision in split_decisions.items()
        },
        "claim_boundary": (
            "Task-specific local causal influence only; no claim of general "
            "metacognition, hallucination prevention, or frontier-model transfer."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = _write_hashed_manifest(output_root / "result.json", result)
    report_path = output_root / "result.md"
    lines = [
        "# Same-String Answerability Causal Study",
        "",
        f"**Decision:** `{study.status}`",
        "",
        "Both registered test splits were evaluated separately against the frozen controls.",
        "",
    ]
    for split, decision in split_decisions.items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- status: `{decision.status}`",
                f"- mean bidirectional effect: `{decision.mean_effect}`",
                f"- reasons: `{', '.join(decision.reasons) if decision.reasons else 'none'}`",
                "",
            ]
        )
    lines.extend(["## Claim boundary", "", result["claim_boundary"], ""])
    AtomicJSONReceiptStore._atomic_write(
        report_path, "\n".join(lines).encode("utf-8")
    )
    return {
        "status": study.status,
        "result": str(result_path.absolute()),
        "report": str(report_path.absolute()),
    }


def register_causal_subcommands(
    subparsers: argparse._SubParsersAction,
) -> None:
    parsers: dict[str, argparse.ArgumentParser] = {}
    for command in CAUSAL_COMMANDS:
        parser = subparsers.add_parser(command)
        parser.add_argument("--config", required=True)
        parser.add_argument("--root", default=".")
        parsers[command] = parser

    prepare = parsers["fa-causal-prepare"]
    prepare.add_argument("--v3-corpus-manifest", required=True)
    prepare.add_argument("--v3-training-activation-manifest", required=True)
    prepare.add_argument("--output-dir", required=True)

    validation = parsers["fa-causal-run-validation"]
    validation.add_argument("--prepare-manifest", required=True)
    validation.add_argument("--output-dir", required=True)
    validation.add_argument("--resume", action="store_true")

    shard = parsers["fa-causal-run-shard"]
    shard.add_argument("--prepare-manifest", required=True)
    shard.add_argument("--seal-manifest", required=True)
    shard.add_argument(
        "--split",
        choices=("causal_entity_test", "causal_template_test"),
        required=True,
    )
    shard.add_argument(
        "--control",
        choices=("baseline", "primary", *CAUSAL_CONTROLS),
        required=True,
    )
    shard.add_argument("--unit-id", required=True)
    shard.add_argument("--member", type=int, choices=range(5))
    shard.add_argument("--output-dir", required=True)
    shard.add_argument("--resume", action="store_true")

    evaluate = parsers["fa-causal-evaluate"]
    evaluate.add_argument("--prepare-manifest", required=True)
    evaluate.add_argument("--seal-manifest", required=True)
    evaluate.add_argument("--evidence-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)


def dispatch_causal(args: argparse.Namespace) -> int:
    handlers = {
        "fa-causal-prepare": prepare_causal,
        "fa-causal-run-validation": run_causal_validation,
        "fa-causal-run-shard": run_causal_shard,
        "fa-causal-evaluate": evaluate_causal,
    }
    handler = handlers.get(getattr(args, "command", None))
    if handler is None:
        return 2
    try:
        result = handler(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "status": "failed",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0
