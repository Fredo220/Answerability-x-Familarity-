"""Registered, position-only activation extraction for Familiarity-vs-Answerability."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


ANCHOR_NAMES = (
    "target_intro_end",
    "user_prompt_end",
    "assistant_prefix_end",
)
POSITION_SEMANTICS = (
    "last token overlapping the target introduction in user-authored text",
    "last token overlapping user-authored text",
    "last token of the rendered assistant generation prefix",
)


@dataclass(frozen=True)
class AnchorRecord:
    """Byte-exact prompt provenance and the three registered token positions."""

    example_id: str
    rendered_bytes: bytes
    rendered_prompt_sha256: str
    input_ids: tuple[int, ...]
    special_tokens_mask: tuple[int, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    target_intro_end: int
    user_prompt_end: int
    assistant_prefix_end: int
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_config_sha256: str
    chat_template_sha256: str
    anchor_names: tuple[str, ...] = ANCHOR_NAMES
    position_semantics: tuple[str, ...] = POSITION_SEMANTICS

    def __post_init__(self) -> None:
        object.__setattr__(self, "rendered_bytes", bytes(self.rendered_bytes))
        object.__setattr__(self, "input_ids", tuple(int(value) for value in self.input_ids))
        object.__setattr__(
            self,
            "special_tokens_mask",
            tuple(int(value) for value in self.special_tokens_mask),
        )
        object.__setattr__(
            self,
            "offset_mapping",
            tuple((int(start), int(end)) for start, end in self.offset_mapping),
        )
        object.__setattr__(self, "anchor_names", tuple(self.anchor_names))
        object.__setattr__(self, "position_semantics", tuple(self.position_semantics))
        if hashlib.sha256(self.rendered_bytes).hexdigest() != self.rendered_prompt_sha256:
            raise ValueError("rendered prompt hash does not match rendered bytes")
        if not self.input_ids or len(self.input_ids) != len(self.special_tokens_mask):
            raise ValueError("token IDs and special-token mask must be nonempty and aligned")
        if len(self.offset_mapping) != len(self.input_ids):
            raise ValueError("offset mapping must align with token IDs")
        if self.anchor_names != ANCHOR_NAMES or self.position_semantics != POSITION_SEMANTICS:
            raise ValueError("anchor names and position semantics must be registered")
        if not (0 <= self.target_intro_end < self.user_prompt_end < self.assistant_prefix_end):
            raise ValueError("registered anchors must be strictly ordered")
        if self.assistant_prefix_end != len(self.input_ids) - 1:
            raise ValueError("assistant_prefix_end must be the final rendered prompt token")
        _required_provenance(self.tokenizer_id, "tokenizer ID")
        _required_revision(self.tokenizer_revision, "tokenizer revision")

    @property
    def anchor_indices(self) -> tuple[int, int, int]:
        return self.target_intro_end, self.user_prompt_end, self.assistant_prefix_end


@dataclass(frozen=True)
class ActivationRecord:
    """The three registered positions for selected layers of one prompt only."""

    example_id: str
    anchors: AnchorRecord
    layer_ids: tuple[int, ...]
    activations: np.ndarray
    dtype: str
    shape: tuple[int, int, int]
    activation_sha256: str
    model_id: str
    model_revision: str
    anchor_names: tuple[str, ...] = ANCHOR_NAMES

    def __post_init__(self) -> None:
        layers = tuple(int(value) for value in self.layer_ids)
        array = np.array(self.activations, copy=True, order="C")
        array.setflags(write=False)
        object.__setattr__(self, "layer_ids", layers)
        object.__setattr__(self, "activations", array)
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(self, "anchor_names", tuple(self.anchor_names))
        if self.example_id != self.anchors.example_id:
            raise ValueError("activation example ID must match anchor provenance")
        if self.anchor_names != self.anchors.anchor_names:
            raise ValueError("activation anchors must use the registered order")
        if array.ndim != 3 or array.shape != self.shape:
            raise ValueError("activation shape metadata does not match the array")
        if array.shape[:2] != (len(ANCHOR_NAMES), len(layers)):
            raise ValueError("activations must have shape [anchor, layer, hidden]")
        if array.dtype.name != self.dtype:
            raise ValueError("activation dtype metadata does not match the array")
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError("activations must contain finite floating-point values")
        _required_provenance(self.model_id, "model ID")
        _required_revision(self.model_revision, "model revision")
        if hashlib.sha256(self.activation_hash_payload).hexdigest() != self.activation_sha256:
            raise ValueError("activation hash does not match activation data")

    @property
    def activation_hash_payload(self) -> bytes:
        return _activation_hash_payload(
            self.example_id,
            self.layer_ids,
            self.anchor_names,
            self.activations,
        )


@dataclass(frozen=True)
class ActivationShard:
    """Paths and hashes for one verified immutable activation shard."""

    npz_path: Path
    index_path: Path
    manifest_path: Path
    npz_sha256: str
    index_sha256: str
    row_count: int
    request_sha256: str


class HFSelectedPositionRunner:
    """Model adapter that captures residual outputs only at requested positions."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str = "unrecorded-model",
        model_revision: str = "unrecorded-revision",
        tokenizer_revision: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_revision = model_revision
        tokenizer_id, resolved_revision, _config_hash = _tokenizer_provenance(
            tokenizer,
            tokenizer_revision=tokenizer_revision,
        )
        self.tokenizer_id = tokenizer_id
        self.tokenizer_revision = resolved_revision
        self.hidden_size = _model_hidden_size(model)
        self._capture: _HFSelectedCapture | None = None

    def selected_position_hooks(
        self, *, layer_ids: Sequence[int], positions: Sequence[int]
    ) -> "_HFSelectedCapture":
        return _HFSelectedCapture(self, tuple(layer_ids), tuple(positions))

    def run_selected(self, input_ids: Sequence[int]) -> None:
        if self._capture is None:
            raise RuntimeError("selected-position hooks must be active before model execution")
        try:
            import torch
        except ImportError as error:  # pragma: no cover - production dependency check
            raise ImportError("PyTorch is required for selected-position extraction") from error
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        encoded = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
        with torch.no_grad():
            self.model(input_ids=encoded, use_cache=False)


class _HFSelectedCapture(AbstractContextManager):
    def __init__(
        self,
        runner: HFSelectedPositionRunner,
        layer_ids: tuple[int, ...],
        positions: tuple[int, ...],
    ):
        self.runner = runner
        self.layer_ids = layer_ids
        self.positions = positions
        self.handles: list[Any] = []
        self.selected: dict[int, Any] = {}

    def __enter__(self) -> "_HFSelectedCapture":
        if self.runner._capture is not None:
            raise RuntimeError("selected-position capture is already active")
        layers = _transformer_layers(self.runner.model)
        if self.layer_ids[-1] >= len(layers):
            raise ValueError("registered layer ID exceeds the model layer count")
        self.runner._capture = self
        try:
            for layer_id in self.layer_ids:
                self.handles.append(
                    layers[layer_id].register_forward_hook(self._hook_for(layer_id))
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _hook_for(self, layer_id: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            try:
                import torch
            except ImportError as error:  # pragma: no cover
                raise ImportError("PyTorch is required for selected-position extraction") from error
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("transformer layer output must have shape [1, token, hidden]")
            if self.positions[-1] >= hidden.shape[1]:
                raise ValueError("registered position exceeds transformer sequence length")
            indices = torch.tensor(self.positions, dtype=torch.long, device=hidden.device)
            self.selected[layer_id] = hidden[0].index_select(0, indices).detach().cpu()

        return capture

    @property
    def activations(self) -> Any:
        if set(self.selected) != set(self.layer_ids):
            return None
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise ImportError("PyTorch is required for selected-position extraction") from error
        return torch.stack([self.selected[layer_id] for layer_id in self.layer_ids], dim=0)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.selected.clear()
        self.runner._capture = None


def resolve_registered_anchors(
    example: Any,
    tokenizer: Any,
    *,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
) -> AnchorRecord:
    """Resolve registered anchors against the exact rendered chat-template bytes.

    ``target_intro_end`` is the final token overlapping the target's introduction
    in the task body, not its later occurrence in the query. ``user_prompt_end``
    is the final token overlapping user-authored text. ``assistant_prefix_end`` is
    the final token emitted by the template's generation prefix.
    """

    user_text = _required_text(getattr(example, "user_text", None), "example.user_text")
    target_text = _required_text(getattr(example, "target_text", None), "example.target_text")
    example_id = _required_text(getattr(example, "example_id", None), "example.example_id")
    render = getattr(tokenizer, "apply_chat_template", None)
    if not callable(render):
        raise ValueError("anchor resolution requires tokenizer.apply_chat_template")
    intro_span_in_user, _query_span_in_user = _target_role_spans(
        example, user_text, target_text
    )
    messages = [{"role": "user", "content": user_text}]
    rendered_without_generation = render(
        messages, tokenize=False, add_generation_prompt=False
    )
    rendered = render(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered_without_generation, str) or not isinstance(rendered, str):
        raise ValueError("chat template must render text when tokenize=False")
    if (
        not rendered.startswith(rendered_without_generation)
        or len(rendered) == len(rendered_without_generation)
    ):
        raise ValueError("chat template must add a nonempty assistant generation suffix")
    user_occurrences = _occurrences(rendered, user_text)
    if len(user_occurrences) != 1:
        raise ValueError("rendered prompt contains an ambiguous user-text occurrence")
    user_start = user_occurrences[0]
    target_span = tuple(user_start + value for value in intro_span_in_user)
    user_span = (user_start, user_start + len(user_text))

    input_ids, special_mask, offsets = _tokenize_rendered(tokenizer, rendered)
    without_generation_ids, _without_mask, _without_offsets = _tokenize_rendered(
        tokenizer, rendered_without_generation
    )
    template_ids = _normalize_ids(
        render(messages, tokenize=True, add_generation_prompt=True)
    )
    template_ids_without_generation = _normalize_ids(
        render(messages, tokenize=True, add_generation_prompt=False)
    )
    if tuple(input_ids) != tuple(template_ids):
        raise ValueError("rendered text token IDs do not match chat-template token IDs")
    if tuple(without_generation_ids) != tuple(template_ids_without_generation):
        raise ValueError("non-generation rendering token IDs do not match chat-template token IDs")
    if (
        len(input_ids) <= len(without_generation_ids)
        or input_ids[: len(without_generation_ids)] != without_generation_ids
    ):
        raise ValueError("assistant generation suffix must extend a compatible token prefix")
    target_end = _last_overlapping_token(offsets, target_span, "target introduction")
    user_end = _last_overlapping_token(offsets, user_span, "user prompt")
    assistant_end = len(input_ids) - 1
    tokenizer_id, tokenizer_revision, tokenizer_config_sha256 = _tokenizer_provenance(
        tokenizer,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        raise ValueError("anchor resolution requires tokenizer.chat_template bytes")
    rendered_bytes = rendered.encode("utf-8")
    return AnchorRecord(
        example_id=example_id,
        rendered_bytes=rendered_bytes,
        rendered_prompt_sha256=hashlib.sha256(rendered_bytes).hexdigest(),
        input_ids=tuple(input_ids),
        special_tokens_mask=tuple(special_mask),
        offset_mapping=tuple(offsets),
        target_intro_end=target_end,
        user_prompt_end=user_end,
        assistant_prefix_end=assistant_end,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        tokenizer_config_sha256=tokenizer_config_sha256,
        chat_template_sha256=hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
    )


def extract_registered_anchors(
    runner: Any,
    example: Any,
    registered_layers: Sequence[int],
    *,
    tokenizer: Any | None = None,
) -> ActivationRecord:
    """Run one prompt while capturing only registered positions and layers.

    The runner interface exposes ``selected_position_hooks`` as a
    context manager plus ``run_selected(input_ids)``. The context is exited
    before any validation error is raised, ensuring prompt-local hooks cannot
    leak into a later prompt or decoding call.
    """

    layer_ids = _registered_layer_ids(registered_layers)
    resolved_tokenizer = tokenizer if tokenizer is not None else getattr(runner, "tokenizer", None)
    if resolved_tokenizer is None:
        raise ValueError("activation runner must expose a tokenizer")
    model_id, model_revision, tokenizer_id, tokenizer_revision, _config_hash = (
        _runner_provenance(runner, resolved_tokenizer)
    )
    anchors = resolve_registered_anchors(
        example,
        resolved_tokenizer,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )
    return _extract_with_anchors(
        runner,
        anchors,
        layer_ids,
        model_id=model_id,
        model_revision=model_revision,
    )


def _extract_with_anchors(
    runner: Any,
    anchors: AnchorRecord,
    layer_ids: tuple[int, ...],
    *,
    model_id: str,
    model_revision: str,
) -> ActivationRecord:
    raw = _run_selected(runner, anchors.input_ids, layer_ids, anchors.anchor_indices)
    array = _to_numpy(raw)
    expected_prefix = (len(layer_ids), len(ANCHOR_NAMES))
    if array.ndim != 3 or array.shape[:2] != expected_prefix:
        raise ValueError(
            "runner activation shape must be [layer, registered_position, hidden]"
        )
    hidden_size = getattr(runner, "hidden_size", None)
    if type(hidden_size) is int and array.shape[2] != hidden_size:
        raise ValueError("runner activation shape does not match runner hidden_size")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("runner activations must be finite floating-point values")
    selected = np.ascontiguousarray(array.transpose(1, 0, 2))
    digest = hashlib.sha256(
        _activation_hash_payload(anchors.example_id, layer_ids, ANCHOR_NAMES, selected)
    ).hexdigest()
    return ActivationRecord(
        example_id=anchors.example_id,
        anchors=anchors,
        layer_ids=layer_ids,
        activations=selected,
        dtype=selected.dtype.name,
        shape=selected.shape,
        activation_sha256=digest,
        model_id=model_id,
        model_revision=model_revision,
    )


def write_activation_shard(
    runner: Any,
    examples: Sequence[Any],
    registered_layers: Sequence[int],
    *,
    destination: str | Path,
    tokenizer: Any | None = None,
) -> ActivationShard:
    """Extract and publish a deterministic NPZ plus a verified JSONL index.

    Each prompt's selected-position tensor is written to a temporary ``.npy``
    before the next prompt runs. The final NPZ is assembled one member at a
    time, so neither extraction nor serialization retains a dataset-wide tensor.
    """

    destination = Path(destination).absolute()
    if destination.suffix != ".npz":
        raise ValueError("activation shard destination must end in .npz")
    prepared = tuple(examples)
    if not prepared:
        raise ValueError("activation shard requires at least one example")
    example_ids = tuple(
        _required_text(getattr(item, "example_id", None), "example.example_id")
        for item in prepared
    )
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("activation shard example IDs must be unique")
    layer_ids = _registered_layer_ids(registered_layers)
    resolved_tokenizer = tokenizer if tokenizer is not None else getattr(runner, "tokenizer", None)
    if resolved_tokenizer is None:
        raise ValueError("activation runner must expose a tokenizer")
    request_sha256, resolved_anchors, extraction_provenance = _prepare_extraction_request(
        runner, prepared, layer_ids, resolved_tokenizer
    )
    index_path, manifest_path = _shard_companions(destination)
    components = (destination, index_path, manifest_path)
    _reject_symlinked_components(components)
    existing = tuple(os.path.lexists(path) for path in components)
    if existing == (True, True, True):
        resumed = resume_activation_shard(destination)
        if resumed.request_sha256 != request_sha256:
            raise FileExistsError("activation shard belongs to a different extraction request")
        return resumed
    if any(existing):
        raise FileExistsError("partial activation shard exists; refusing to clobber it")

    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{destination.stem}.", dir=destination.parent))
    temporary_npz = work / destination.name
    temporary_index = work / index_path.name
    temporary_manifest = work / manifest_path.name
    members: list[tuple[str, Path]] = []
    published: list[Path] = []
    try:
        with temporary_index.open("xb") as index_file:
            model_id, model_revision, _tokenizer_id, _tokenizer_revision, _config_hash = (
                extraction_provenance
            )
            for row_number, anchors in enumerate(resolved_anchors):
                record = _extract_with_anchors(
                    runner,
                    anchors,
                    layer_ids,
                    model_id=model_id,
                    model_revision=model_revision,
                )
                member = f"{row_number:06d}-{hashlib.sha256(record.example_id.encode('utf-8')).hexdigest()[:16]}.npy"
                array_path = work / member
                np.save(array_path, record.activations, allow_pickle=False)
                members.append((member, array_path))
                index_file.write(_canonical_json_line(_activation_index_row(record, member)))
                del record
                # CPython normally releases immediately; explicit collection also
                # bounds alternate runtimes before the next model prompt.
                gc.collect()
            index_file.flush()
            os.fsync(index_file.fileno())
        _write_deterministic_npz(temporary_npz, members)
        npz_sha256 = _sha256_file(temporary_npz)
        index_sha256 = _sha256_file(temporary_index)
        manifest = {
            "schema_version": 1,
            "npz_file": destination.name,
            "index_file": index_path.name,
            "npz_sha256": npz_sha256,
            "index_sha256": index_sha256,
            "row_count": len(prepared),
            "request_sha256": request_sha256,
        }
        temporary_manifest.write_bytes(_canonical_json(manifest) + b"\n")
        for source, target in (
            (temporary_npz, destination),
            (temporary_index, index_path),
            (temporary_manifest, manifest_path),
        ):
            _reject_symlinked_components((target,))
            os.link(source, target)
            published.append(target)
        return resume_activation_shard(destination)
    except BaseException:
        for path in reversed(published):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


def resume_activation_shard(destination: str | Path) -> ActivationShard:
    """Verify both shard files, every indexed array, and their activation hashes."""

    destination = Path(destination).absolute()
    index_path, manifest_path = _shard_companions(destination)
    _reject_symlinked_components((destination, index_path, manifest_path))
    if not manifest_path.is_file():
        raise FileNotFoundError("activation shard manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("activation shard manifest is unreadable") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("activation shard manifest has an invalid schema")
    if manifest.get("npz_file") != destination.name or manifest.get("index_file") != index_path.name:
        raise ValueError("activation shard manifest paths do not match the destination")
    npz_sha256 = _required_sha256(manifest.get("npz_sha256"), "NPZ hash")
    index_sha256 = _required_sha256(manifest.get("index_sha256"), "index hash")
    request_sha256 = _required_sha256(manifest.get("request_sha256"), "request hash")
    row_count = manifest.get("row_count")
    if type(row_count) is not int or row_count < 1:
        raise ValueError("activation shard row count is invalid")
    if not destination.is_file() or _sha256_file(destination) != npz_sha256:
        raise ValueError("activation shard NPZ hash mismatch")
    if not index_path.is_file() or _sha256_file(index_path) != index_sha256:
        raise ValueError("activation shard index hash mismatch")
    rows = _read_index(index_path)
    if len(rows) != row_count:
        raise ValueError("activation shard index row count mismatch")
    expected_members = []
    with zipfile.ZipFile(destination, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("activation shard NPZ has duplicate members")
        for row in rows:
            member = row.get("npz_member")
            if not isinstance(member, str) or Path(member).name != member or not member.endswith(".npy"):
                raise ValueError("activation shard index has an invalid NPZ member")
            expected_members.append(member)
            try:
                with archive.open(member) as source:
                    array = np.load(source, allow_pickle=False)
            except (KeyError, ValueError) as error:
                raise ValueError("activation shard NPZ member is missing or invalid") from error
            _verify_indexed_activation(row, array)
            del array
        if names != expected_members:
            raise ValueError("activation shard NPZ members do not match the index")
    return ActivationShard(
        npz_path=destination,
        index_path=index_path,
        manifest_path=manifest_path,
        npz_sha256=npz_sha256,
        index_sha256=index_sha256,
        row_count=row_count,
        request_sha256=request_sha256,
    )


def load_activation_records(manifest_path: str | Path) -> tuple[ActivationRecord, ...]:
    """Verify an immutable activation shard and reconstruct its typed rows.

    The manifest is the capability passed between Colab extraction and local
    analysis. Accepting the NPZ path directly would bypass that explicit
    boundary, so callers must provide the exact sidecar manifest.
    """

    manifest_path = Path(manifest_path).absolute()
    if not manifest_path.name.endswith(".manifest.json"):
        raise ValueError("activation loader requires the exact sidecar manifest path")
    _reject_symlinked_components((manifest_path,))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("activation shard manifest is unreadable") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("activation shard manifest has an invalid schema")
    npz_name = manifest.get("npz_file")
    if (
        not isinstance(npz_name, str)
        or Path(npz_name).name != npz_name
        or not npz_name.endswith(".npz")
    ):
        raise ValueError("activation shard manifest has an invalid NPZ path")
    shard = resume_activation_shard(manifest_path.parent / npz_name)
    if shard.manifest_path != manifest_path:
        raise ValueError("activation loader requires the exact sidecar manifest path")

    rows = _read_index(shard.index_path)
    records: list[ActivationRecord] = []
    with zipfile.ZipFile(shard.npz_path, "r") as archive:
        for row in rows:
            member = row["npz_member"]
            with archive.open(member) as source:
                array = np.array(np.load(source, allow_pickle=False), copy=True, order="C")
            anchors = _anchor_record_from_index(row)
            records.append(
                ActivationRecord(
                    example_id=row["example_id"],
                    anchors=anchors,
                    layer_ids=tuple(row["layer_ids"]),
                    activations=array,
                    dtype=row["dtype"],
                    shape=tuple(row["shape"]),
                    activation_sha256=row["activation_sha256"],
                    model_id=row["model_id"],
                    model_revision=row["model_revision"],
                    anchor_names=tuple(row["anchor_names"]),
                )
            )
    return tuple(records)


def _anchor_record_from_index(row: Mapping[str, Any]) -> AnchorRecord:
    payload = row.get("anchors")
    if not isinstance(payload, Mapping):
        raise ValueError("activation shard anchor provenance is invalid")
    try:
        rendered = bytes.fromhex(payload["rendered_utf8_hex"])
        positions = tuple(payload["anchor_indices"])
        return AnchorRecord(
            example_id=row["example_id"],
            rendered_bytes=rendered,
            rendered_prompt_sha256=payload["rendered_prompt_sha256"],
            input_ids=tuple(payload["input_ids"]),
            special_tokens_mask=tuple(payload["special_tokens_mask"]),
            offset_mapping=tuple(tuple(pair) for pair in payload["offset_mapping"]),
            target_intro_end=positions[0],
            user_prompt_end=positions[1],
            assistant_prefix_end=positions[2],
            tokenizer_id=payload["tokenizer_id"],
            tokenizer_revision=payload["tokenizer_revision"],
            tokenizer_config_sha256=payload["tokenizer_config_sha256"],
            chat_template_sha256=payload["chat_template_sha256"],
            position_semantics=tuple(payload["position_semantics"]),
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError("activation shard anchor provenance is invalid") from error


def _run_selected(
    runner: Any,
    input_ids: tuple[int, ...],
    layer_ids: tuple[int, ...],
    positions: tuple[int, ...],
) -> Any:
    hooks = getattr(runner, "selected_position_hooks", None)
    run = getattr(runner, "run_selected", None)
    if not callable(hooks) or not callable(run):
        raise ValueError(
            "runner must implement selected_position_hooks and run_selected"
        )
    with hooks(layer_ids=layer_ids, positions=positions) as capture:
        run(input_ids)
        raw = getattr(capture, "activations", None)
    if raw is None:
        raise ValueError("runner did not capture selected activations")
    return raw


def _registered_layer_ids(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("registered layer IDs must be an increasing integer sequence")
    layers = tuple(values)
    if (
        not layers
        or any(type(value) is not int or value < 0 for value in layers)
        or layers != tuple(sorted(set(layers)))
    ):
        raise ValueError("registered layer IDs must be a unique increasing nonnegative sequence")
    return layers


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    try:
        import torch
    except ImportError:  # pragma: no cover - NumPy-backed runners do not require torch
        torch = None
    if torch is not None and isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
        value = value.to(torch.float32)
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _activation_hash_payload(
    example_id: str,
    layer_ids: Sequence[int],
    anchor_names: Sequence[str],
    array: np.ndarray,
) -> bytes:
    header = json.dumps(
        {
            "anchor_names": list(anchor_names),
            "dtype": array.dtype.name,
            "example_id": example_id,
            "layer_ids": list(layer_ids),
            "shape": list(array.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return header + b"\0" + np.ascontiguousarray(array).tobytes(order="C")


def _activation_index_row(record: ActivationRecord, member: str) -> dict[str, Any]:
    anchors = record.anchors
    return {
        "schema_version": 1,
        "example_id": record.example_id,
        "npz_member": member,
        "anchor_names": list(record.anchor_names),
        "layer_ids": list(record.layer_ids),
        "dtype": record.dtype,
        "shape": list(record.shape),
        "activation_sha256": record.activation_sha256,
        "model_id": record.model_id,
        "model_revision": record.model_revision,
        "anchors": _anchor_provenance_payload(anchors),
    }


def _verify_indexed_activation(row: Any, array: np.ndarray) -> None:
    if not isinstance(row, dict) or row.get("schema_version") != 1:
        raise ValueError("activation shard index row has an invalid schema")
    example_id = _required_text(row.get("example_id"), "index example_id")
    anchor_names = row.get("anchor_names")
    layer_ids = row.get("layer_ids")
    shape = row.get("shape")
    dtype = row.get("dtype")
    digest = _required_sha256(row.get("activation_sha256"), "activation hash")
    _required_provenance(row.get("model_id"), "model ID")
    _required_revision(row.get("model_revision"), "model revision")
    if anchor_names != list(ANCHOR_NAMES):
        raise ValueError("activation shard index has unregistered anchors")
    try:
        layers = _registered_layer_ids(layer_ids)
    except (TypeError, ValueError) as error:
        raise ValueError("activation shard index has invalid layer IDs") from error
    if shape != list(array.shape) or dtype != array.dtype.name:
        raise ValueError("activation shard array metadata mismatch")
    if array.ndim != 3 or array.shape[:2] != (len(ANCHOR_NAMES), len(layers)):
        raise ValueError("activation shard array has an invalid selected-position shape")
    actual = hashlib.sha256(
        _activation_hash_payload(example_id, layers, ANCHOR_NAMES, array)
    ).hexdigest()
    if actual != digest:
        raise ValueError("activation shard activation hash mismatch")
    anchors = row.get("anchors")
    if not isinstance(anchors, dict):
        raise ValueError("activation shard anchor provenance is invalid")
    try:
        rendered_bytes = bytes.fromhex(anchors.get("rendered_utf8_hex", ""))
    except (TypeError, ValueError) as error:
        raise ValueError("activation shard rendered bytes are invalid") from error
    if hashlib.sha256(rendered_bytes).hexdigest() != anchors.get("rendered_prompt_sha256"):
        raise ValueError("activation shard rendered prompt hash mismatch")
    input_ids = anchors.get("input_ids")
    special_mask = anchors.get("special_tokens_mask")
    anchor_indices = anchors.get("anchor_indices")
    semantics = anchors.get("position_semantics")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or not isinstance(special_mask, list)
        or len(input_ids) != len(special_mask)
        or semantics != list(POSITION_SEMANTICS)
        or not isinstance(anchor_indices, list)
        or len(anchor_indices) != len(ANCHOR_NAMES)
        or not (0 <= anchor_indices[0] < anchor_indices[1] < anchor_indices[2] == len(input_ids) - 1)
    ):
        raise ValueError("activation shard anchor positions are invalid")
    offsets = anchors.get("offset_mapping")
    if not isinstance(offsets, list) or len(offsets) != len(input_ids):
        raise ValueError("activation shard offset mapping is invalid")
    _required_provenance(anchors.get("tokenizer_id"), "tokenizer ID")
    _required_revision(anchors.get("tokenizer_revision"), "tokenizer revision")
    _required_sha256(anchors.get("tokenizer_config_sha256"), "tokenizer config hash")
    _required_sha256(anchors.get("chat_template_sha256"), "chat template hash")


def _prepare_extraction_request(
    runner: Any,
    examples: Sequence[Any],
    layer_ids: tuple[int, ...],
    tokenizer: Any,
) -> tuple[str, tuple[AnchorRecord, ...], tuple[str, str, str, str, str]]:
    provenance = _runner_provenance(runner, tokenizer)
    model_id, model_revision, tokenizer_id, tokenizer_revision, tokenizer_config_sha256 = (
        provenance
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        raise ValueError("activation extraction requires tokenizer.chat_template bytes")
    anchors = tuple(
        resolve_registered_anchors(
            item,
            tokenizer,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
        )
        for item in examples
    )
    rows = [
        {
            "canonical_example_content": _canonical_example_content(item),
            "resolved_anchor_provenance": _anchor_provenance_payload(anchor),
        }
        for item, anchor in zip(examples, anchors, strict=True)
    ]
    request = {
        "schema_version": 2,
        "examples": rows,
        "layer_ids": list(layer_ids),
        "anchor_names": list(ANCHOR_NAMES),
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(_canonical_json(request)).hexdigest(), anchors, provenance


def _anchor_provenance_payload(anchors: AnchorRecord) -> dict[str, Any]:
    return {
        "rendered_utf8_hex": anchors.rendered_bytes.hex(),
        "rendered_prompt_sha256": anchors.rendered_prompt_sha256,
        "input_ids": list(anchors.input_ids),
        "special_tokens_mask": list(anchors.special_tokens_mask),
        "offset_mapping": [list(pair) for pair in anchors.offset_mapping],
        "anchor_indices": list(anchors.anchor_indices),
        "position_semantics": list(anchors.position_semantics),
        "tokenizer_id": anchors.tokenizer_id,
        "tokenizer_revision": anchors.tokenizer_revision,
        "tokenizer_config_sha256": anchors.tokenizer_config_sha256,
        "chat_template_sha256": anchors.chat_template_sha256,
    }


def _canonical_example_content(example: Any) -> dict[str, Any]:
    if is_dataclass(example) and not isinstance(example, type):
        payload = {field.name: getattr(example, field.name) for field in fields(example)}
    else:
        try:
            payload = {
                name: value
                for name, value in vars(example).items()
                if not name.startswith("_")
            }
        except TypeError as error:
            raise ValueError("example must expose canonical content fields") from error
    if not payload:
        raise ValueError("example must expose canonical content fields")
    return _json_safe(payload)


def _write_deterministic_npz(path: Path, members: Sequence[tuple[str, Path]]) -> None:
    with zipfile.ZipFile(path, mode="x", compression=zipfile.ZIP_STORED) as archive:
        for member, source in members:
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            with source.open("rb") as source_file, archive.open(info, "w") as target:
                while chunk := source_file.read(1024 * 1024):
                    target.write(chunk)


def _read_index(path: Path) -> list[Any]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.endswith("\n"):
                    raise ValueError("activation shard index must end every row with a newline")
                rows.append(json.loads(line))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("activation shard index is unreadable") from error
    return rows


def _shard_companions(destination: Path) -> tuple[Path, Path]:
    if destination.suffix != ".npz":
        raise ValueError("activation shard destination must end in .npz")
    return destination.with_suffix(".jsonl"), destination.with_suffix(".manifest.json")


def _reject_symlinked_components(paths: Sequence[Path]) -> None:
    if any(path.is_symlink() for path in paths):
        raise ValueError("activation shard components must not be symlinks")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _canonical_json_line(value: Any) -> bytes:
    return _canonical_json(value) + b"\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _transformer_layers(model: Any) -> Sequence[Any]:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(
            getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None),
            "layers",
            None,
        ),
    )
    for candidate in candidates:
        if candidate is not None:
            try:
                if len(candidate) > 0:
                    return candidate
            except TypeError:
                pass
    raise ValueError("could not locate transformer layers on activation model")


def _model_hidden_size(model: Any) -> int:
    candidates = (
        getattr(model, "hidden_size", None),
        getattr(getattr(model, "config", None), "hidden_size", None),
    )
    for candidate in candidates:
        if type(candidate) is int and candidate > 0:
            return candidate
    raise ValueError("activation model must expose a positive hidden_size")


def _target_role_spans(
    example: Any, user_text: str, target_text: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    spans = []
    for name in ("target_intro_span", "target_query_span"):
        value = getattr(example, name, None)
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
            or any(type(item) is not int for item in value)
        ):
            raise ValueError("structured target spans must be recorded as integer pairs")
        span = tuple(value)
        start, end = span
        if start < 0 or end <= start or end > len(user_text) or user_text[start:end] != target_text:
            raise ValueError("structured target spans must identify the target text")
        spans.append(span)
    intro_span, query_span = spans
    if intro_span[1] > query_span[0]:
        raise ValueError("structured target spans must preserve introduction/query order")
    block = getattr(example, "block", None)
    if block not in {"factorial", "same_string"}:
        raise ValueError("example.block must identify a registered prompt block")
    if block == "same_string":
        markers = _occurrences(user_text, " Task: ")
        if len(markers) != 1 or intro_span[0] < markers[0] + len(" Task: "):
            raise ValueError("structured target spans violate same-string task semantics")
    return intro_span, query_span


def _tokenize_rendered(
    tokenizer: Any, rendered: str
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]]:
    kwargs = {
        "add_special_tokens": False,
        "return_special_tokens_mask": True,
        "return_offsets_mapping": True,
    }
    try:
        encoded = tokenizer(rendered, **kwargs)
    except (NotImplementedError, TypeError) as error:
        raise ValueError(
            "anchor resolution requires an exact offset mapping for the full rendered prompt"
        ) from error
    if not isinstance(encoded, Mapping):
        raise ValueError("tokenizer must return a mapping for rendered text")
    input_ids = _normalize_ids(encoded.get("input_ids"))
    raw_mask = encoded.get("special_tokens_mask")
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    if raw_mask is None:
        special_mask = tuple(int(token in special_ids) for token in input_ids)
    else:
        added_mask = _normalize_sequence(raw_mask, "special_tokens_mask")
        special_mask = tuple(
            int(bool(masked) or token in special_ids)
            for token, masked in zip(input_ids, added_mask, strict=True)
        )
    raw_offsets = encoded.get("offset_mapping")
    if raw_offsets is None:
        raise ValueError(
            "anchor resolution requires an exact offset mapping for the full rendered prompt"
        )
    offsets = tuple(
        (int(pair[0]), int(pair[1]))
        for pair in _normalize_sequence(raw_offsets, "offset_mapping")
    )
    if len(offsets) != len(input_ids) or any(
        start < 0 or end < start or end > len(rendered) for start, end in offsets
    ):
        raise ValueError("rendered prompt offset mapping is invalid")
    nonempty_offsets = [(start, end) for start, end in offsets if end > start]
    if any(
        current[0] < previous[0]
        for previous, current in zip(nonempty_offsets, nonempty_offsets[1:])
    ):
        raise ValueError("rendered prompt offset mapping is not monotonic")
    return tuple(input_ids), special_mask, offsets


def _last_overlapping_token(
    offsets: Sequence[tuple[int, int]], span: tuple[int, int], label: str
) -> int:
    overlapping = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < span[1] and end > span[0]
    ]
    if not overlapping:
        raise ValueError(f"{label} does not overlap any rendered token")
    return overlapping[-1]


def _tokenizer_provenance(
    tokenizer: Any,
    *,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
) -> tuple[str, str, str]:
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(init_kwargs, Mapping):
        init_kwargs = {}
    metadata_id = getattr(tokenizer, "name_or_path", None)
    metadata_revision = getattr(tokenizer, "revision", None) or init_kwargs.get("revision")
    resolved_id = tokenizer_id if tokenizer_id is not None else metadata_id
    resolved_revision = (
        tokenizer_revision if tokenizer_revision is not None else metadata_revision
    )
    resolved_id = _required_provenance(resolved_id, "tokenizer ID")
    resolved_revision = _required_revision(resolved_revision, "tokenizer revision")
    if metadata_id is not None and str(metadata_id) != resolved_id:
        raise ValueError("runner tokenizer ID does not match tokenizer metadata")
    if metadata_revision is not None and str(metadata_revision) != resolved_revision:
        raise ValueError("runner tokenizer revision does not match tokenizer metadata")
    config = {
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "name_or_path": resolved_id,
        "revision": resolved_revision,
        "init_kwargs": _json_safe(init_kwargs),
    }
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return resolved_id, resolved_revision, digest


def _runner_provenance(
    runner: Any, tokenizer: Any
) -> tuple[str, str, str, str, str]:
    model_id = _required_provenance(getattr(runner, "model_id", None), "model ID")
    model_revision = _required_revision(
        getattr(runner, "model_revision", None), "model revision"
    )
    tokenizer_id = _required_provenance(
        getattr(runner, "tokenizer_id", None), "tokenizer ID"
    )
    tokenizer_revision = _required_revision(
        getattr(runner, "tokenizer_revision", None), "tokenizer revision"
    )
    tokenizer_id, tokenizer_revision, config_hash = _tokenizer_provenance(
        tokenizer,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )
    return model_id, model_revision, tokenizer_id, tokenizer_revision, config_hash


def _required_provenance(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().casefold().startswith(
        ("unrecorded", "unknown")
    ):
        raise ValueError(f"{label} must be explicitly pinned and recorded")
    return value.strip()


def _required_revision(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be an immutable lowercase 40-hex commit hash")
    return value


def _normalize_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    values = _normalize_sequence(value, "input_ids")
    if len(values) == 1 and isinstance(values[0], Sequence) and not isinstance(values[0], (str, bytes)):
        values = tuple(values[0])
    if not values or any(type(item) is not int for item in values):
        raise ValueError("token IDs must be a nonempty integer sequence")
    return tuple(values)


def _normalize_sequence(value: Any, label: str) -> tuple[Any, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return tuple(value)


def _occurrences(text: str, needle: str) -> list[int]:
    found = []
    start = 0
    while True:
        offset = text.find(needle, start)
        if offset < 0:
            return found
        found.append(offset)
        start = offset + 1


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
