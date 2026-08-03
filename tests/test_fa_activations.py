from __future__ import annotations

import hashlib
import gc
import json
import weakref
from contextlib import AbstractContextManager
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trajectory_extractor.fa_activations import (
    HFSelectedPositionRunner,
    extract_registered_anchors,
    load_activation_records,
    resolve_registered_anchors,
    resume_activation_shard,
    write_activation_shard,
)


MODEL_REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40


class FakeTokenizer:
    chat_template = "fake-chat-template-v1"
    name_or_path = "fake/tokenizer"
    all_special_ids = (1, 2, 3)
    init_kwargs = {"revision": TOKENIZER_REVISION, "use_fast": True}

    _specials = {"<bos>": 1, "<user>": 2, "<assistant>": 3}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        rendered = f"<bos><user>{messages[0]['content']}"
        if add_generation_prompt:
            rendered += "<assistant>"
        if not tokenize:
            return rendered
        return self(rendered, add_special_tokens=False)["input_ids"]

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
    ):
        assert add_special_tokens is False
        input_ids = []
        offsets = []
        index = 0
        while index < len(text):
            special = next((item for item in self._specials if text.startswith(item, index)), None)
            if special is not None:
                input_ids.append(self._specials[special])
                offsets.append((index, index + len(special)))
                index += len(special)
            else:
                input_ids.append(1000 + ord(text[index]))
                offsets.append((index, index + 1))
                index += 1
        result = {"input_ids": input_ids}
        if return_special_tokens_mask:
            result["special_tokens_mask"] = [int(token in self.all_special_ids) for token in input_ids]
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class NoOffsetTokenizer(FakeTokenizer):
    def __call__(self, text, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            raise NotImplementedError("slow tokenizer")
        return super().__call__(text, **kwargs)


class ContextualNoOffsetTokenizer(NoOffsetTokenizer):
    """A slow tokenizer whose final token changes when more context is appended."""

    def __call__(self, text, **kwargs):
        result = super().__call__(text, **kwargs)
        if text and not kwargs.get("return_offsets_mapping"):
            result["input_ids"][-1] = 9000 + ord(text[-1]) + len(text)
        return result


class ContextualBoundaryTokenizer(FakeTokenizer):
    def __call__(self, text, **kwargs):
        result = super().__call__(text, **kwargs)
        if text:
            result["input_ids"][-1] += len(text)
        return result


class AddedSpecialMaskTokenizer(FakeTokenizer):
    def __call__(self, text, **kwargs):
        result = super().__call__(text, **kwargs)
        if kwargs.get("return_special_tokens_mask"):
            result["special_tokens_mask"] = [0] * len(result["input_ids"])
        return result


class MappingChatTemplateTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        rendered = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if not tokenize:
            return rendered
        return {
            "input_ids": self(rendered, add_special_tokens=False)["input_ids"]
        }


def example(
    user_text="Ada has archive code K7M2Q. What is Ada's archive code?",
    *,
    example_id="example-1",
    target_text="Ada",
    block="factorial",
    target_intro_span=None,
    target_query_span=None,
):
    if target_intro_span is None:
        start = user_text.index(target_text)
        target_intro_span = (start, start + len(target_text))
    if target_query_span is None:
        start = user_text.rindex(target_text)
        target_query_span = (start, start + len(target_text))
    return SimpleNamespace(
        example_id=example_id,
        canonical_payload_sha256=hashlib.sha256(example_id.encode("utf-8")).hexdigest(),
        user_text=user_text,
        target_text=target_text,
        block=block,
        target_intro_span=target_intro_span,
        target_query_span=target_query_span,
    )


def test_anchor_resolution_records_exact_rendering_provenance_and_position_semantics():
    tokenizer = FakeTokenizer()

    record = resolve_registered_anchors(example(), tokenizer)

    assert record.rendered_bytes.decode("utf-8") == tokenizer.apply_chat_template(
        [{"role": "user", "content": example().user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert record.rendered_prompt_sha256 == hashlib.sha256(record.rendered_bytes).hexdigest()
    assert record.target_intro_end < record.user_prompt_end < record.assistant_prefix_end
    assert record.assistant_prefix_end == len(record.input_ids) - 1
    assert len(record.input_ids) == len(record.special_tokens_mask) == len(record.offset_mapping)
    assert record.special_tokens_mask[0] == 1
    assert record.anchor_names == (
        "target_intro_end",
        "user_prompt_end",
        "assistant_prefix_end",
    )
    assert record.anchor_indices == (
        record.target_intro_end,
        record.user_prompt_end,
        record.assistant_prefix_end,
    )
    assert record.position_semantics == (
        "last token overlapping the target introduction in user-authored text",
        "last token overlapping user-authored text",
        "last token of the rendered assistant generation prefix",
    )
    assert record.tokenizer_id == "fake/tokenizer"
    assert record.tokenizer_revision == TOKENIZER_REVISION
    assert record.tokenizer_config_sha256
    assert record.chat_template_sha256 == hashlib.sha256(
        tokenizer.chat_template.encode("utf-8")
    ).hexdigest()
    with pytest.raises(FrozenInstanceError):
        record.target_intro_end = 0


@pytest.mark.parametrize("tokenizer", [NoOffsetTokenizer(), ContextualNoOffsetTokenizer()])
def test_anchor_resolution_without_exact_full_render_offsets_fails_closed(tokenizer):
    with pytest.raises(ValueError, match="exact offset mapping"):
        resolve_registered_anchors(example(), tokenizer)


def test_anchor_resolution_uses_structured_spans_instead_of_first_occurrence():
    user_text = "Ada met Ada yesterday. What is Ada's archive code?"
    second = user_text.index("Ada", 1)
    query = user_text.rindex("Ada")
    structured = example(
        user_text,
        target_intro_span=(second, second + 3),
        target_query_span=(query, query + 3),
    )

    record = resolve_registered_anchors(structured, FakeTokenizer())

    rendered_user_start = record.rendered_bytes.decode("utf-8").index(user_text)
    assert record.offset_mapping[record.target_intro_end] == (
        rendered_user_start + second + 2,
        rendered_user_start + second + 3,
    )


def test_anchor_resolution_rejects_invalid_structured_role_spans():
    malformed = example(target_intro_span=(39, 42), target_query_span=(0, 3))

    with pytest.raises(ValueError, match="structured target spans"):
        resolve_registered_anchors(malformed, FakeTokenizer())


class IgnoredGenerationFlagTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=False,
        )


def test_anchor_resolution_proves_nonempty_compatible_assistant_generation_suffix():
    with pytest.raises(ValueError, match="assistant generation suffix"):
        resolve_registered_anchors(example(), IgnoredGenerationFlagTokenizer())


def test_anchor_resolution_rejects_contextual_bpe_boundary_without_token_prefix():
    with pytest.raises(ValueError, match="compatible token prefix"):
        resolve_registered_anchors(example(), ContextualBoundaryTokenizer())


class MissingRevisionTokenizer(FakeTokenizer):
    init_kwargs = {"use_fast": True}


def test_direct_anchor_resolution_rejects_tokenizer_without_pinned_revision():
    with pytest.raises(ValueError, match="tokenizer revision"):
        resolve_registered_anchors(example(), MissingRevisionTokenizer())


@pytest.mark.parametrize(
    "revision",
    [
        "main",
        "latest",
        "refs/tags/v1.0",
        "A" * 40,
        "a" * 39,
        "a" * 41,
        "unrecorded",
    ],
)
def test_anchor_resolution_rejects_mutable_or_malformed_tokenizer_revisions(revision):
    tokenizer = FakeTokenizer()
    tokenizer.init_kwargs = {**tokenizer.init_kwargs, "revision": revision}

    with pytest.raises(ValueError, match="tokenizer revision"):
        resolve_registered_anchors(example(), tokenizer)


def test_special_token_mask_marks_template_ids_even_when_added_token_mask_is_zero():
    record = resolve_registered_anchors(example(), AddedSpecialMaskTokenizer())

    assert record.special_tokens_mask[0] == 1
    assert record.special_tokens_mask[-1] == 1


def test_anchor_resolution_accepts_mapping_from_tokenized_chat_template():
    record = resolve_registered_anchors(
        example(), MappingChatTemplateTokenizer()
    )

    assert record.input_ids


def test_same_string_anchor_ignores_repeated_exposure_mentions_before_task():
    target = "Ada"
    repeated = SimpleNamespace(
        example_id="same-string-1",
        canonical_payload_sha256="b" * 64,
        target_text=target,
        block="same_string",
        user_text=(
            "Ada visits Cedar Park. Ada keeps a notebook. Ada prefers tea. Ada collects cards. "
            "Task: Bea has shape oval. Ada has code K7M2Q. What is Ada's archive code?"
        ),
    )
    intro = repeated.user_text.index("Ada has code")
    query = repeated.user_text.rindex("Ada")
    repeated.target_intro_span = (intro, intro + len(target))
    repeated.target_query_span = (query, query + len(target))

    record = resolve_registered_anchors(repeated, FakeTokenizer())

    task_intro_end = repeated.user_text.index("Ada has code") + len(target) - 1
    rendered_user_start = record.rendered_bytes.decode("utf-8").index(repeated.user_text)
    assert record.offset_mapping[record.target_intro_end] == (
        rendered_user_start + task_intro_end,
        rendered_user_start + task_intro_end + 1,
    )


class _Capture(AbstractContextManager):
    def __init__(self, runner, layer_ids, positions):
        self.runner = runner
        self.layer_ids = layer_ids
        self.positions = positions
        self.activations = None

    def __enter__(self):
        assert self.runner.active_capture is None
        self.runner.active_capture = self
        self.runner.active_handles += len(self.layer_ids)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.runner.active_capture = None
        self.runner.active_handles = 0


class FakeRunner:
    hidden_size = 5
    model_id = "fake/model"
    model_revision = MODEL_REVISION
    tokenizer_id = "fake/tokenizer"
    tokenizer_revision = TOKENIZER_REVISION

    def __init__(self, *, malformed=False):
        self.tokenizer = FakeTokenizer()
        self.malformed = malformed
        self.calls = []
        self.active_capture = None
        self.active_handles = 0

    def selected_position_hooks(self, *, layer_ids, positions):
        self.calls.append((tuple(layer_ids), tuple(positions)))
        return _Capture(self, tuple(layer_ids), tuple(positions))

    def run_selected(self, input_ids):
        assert self.active_capture is not None
        assert self.active_handles == len(self.active_capture.layer_ids)
        layers = len(self.active_capture.layer_ids)
        positions = len(self.active_capture.positions)
        shape = (layers, positions, self.hidden_size)
        if self.malformed:
            shape = (layers, positions + 1, self.hidden_size)
        self.active_capture.activations = np.arange(np.prod(shape), dtype=np.float16).reshape(shape)


class _BatchCapture(AbstractContextManager):
    def __init__(self, runner, layer_ids, positions):
        self.runner = runner
        self.layer_ids = tuple(layer_ids)
        self.positions = tuple(tuple(row) for row in positions)
        self.activations = None

    def __enter__(self):
        self.runner.batch_capture = self
        return self

    def __exit__(self, *_args):
        self.runner.batch_capture = None


class FakeBatchRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.batch_capture = None
        self.batch_calls = 0

    def selected_batch_hooks(self, *, layer_ids, positions):
        return _BatchCapture(self, layer_ids, positions)

    def run_selected_batch(self, input_ids):
        self.batch_calls += 1
        assert self.batch_capture is not None
        self.batch_capture.activations = np.ones(
            (
                len(input_ids),
                len(self.batch_capture.layer_ids),
                3,
                self.hidden_size,
            ),
            dtype=np.float32,
        )


def test_activation_shard_batches_equal_length_prompts(tmp_path):
    runner = FakeBatchRunner()
    examples = tuple(example(example_id=f"batch-{index}") for index in range(8))

    shard = write_activation_shard(
        runner,
        examples,
        (0, 1),
        destination=tmp_path / "batched.npz",
    )

    assert shard.row_count == 8
    assert runner.batch_calls == 2
    assert len(load_activation_records(shard.manifest_path)) == 8


def test_activation_extraction_stores_only_registered_positions_and_selected_layers():
    runner = FakeRunner()
    anchors = resolve_registered_anchors(example(), runner.tokenizer)

    result = extract_registered_anchors(runner, example(), registered_layers=(4, 12, 20))

    assert runner.calls == [((4, 12, 20), anchors.anchor_indices)]
    assert runner.active_handles == 0
    assert result.example_id == "example-1"
    assert result.layer_ids == (4, 12, 20)
    assert result.anchor_names == anchors.anchor_names
    assert result.activations.shape == (3, 3, runner.hidden_size)
    assert result.shape == (3, 3, runner.hidden_size)
    assert result.dtype == "float16"
    assert result.model_id == runner.model_id
    assert result.model_revision == runner.model_revision
    assert result.anchors.tokenizer_id == runner.tokenizer_id
    assert result.anchors.tokenizer_revision == runner.tokenizer_revision
    assert result.activation_sha256 == hashlib.sha256(
        result.activation_hash_payload
    ).hexdigest()
    assert result.activations.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        result.activations[0, 0, 0] = 0
    with pytest.raises(FrozenInstanceError):
        result.dtype = "float32"


def test_malformed_runner_output_fails_closed_and_releases_selected_position_hooks():
    runner = FakeRunner(malformed=True)

    with pytest.raises(ValueError, match="runner activation shape"):
        extract_registered_anchors(runner, example(), registered_layers=(4, 12, 20))

    assert runner.active_capture is None
    assert runner.active_handles == 0


@pytest.mark.parametrize("layers", [(), (4, 4), (4, -1), (4, 2)])
def test_activation_extraction_rejects_nonregistered_layer_sequences(layers):
    with pytest.raises(ValueError, match="layer IDs"):
        extract_registered_anchors(FakeRunner(), example(), registered_layers=layers)


class TrackingRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.previous_arrays = []

    def run_selected(self, input_ids):
        gc.collect()
        assert all(reference() is None for reference in self.previous_arrays)
        super().run_selected(input_ids)
        self.previous_arrays = [weakref.ref(self.active_capture.activations)]


class NeverRunRunner(FakeRunner):
    def selected_position_hooks(self, *, layer_ids, positions):
        raise AssertionError("verified resume must not invoke the runner")


def test_streaming_writer_is_memory_bounded_verified_and_records_activation_metadata(tmp_path):
    runner = TrackingRunner()
    examples = (example(example_id="example-1"), example(example_id="example-2"))
    destination = tmp_path / "activations.npz"

    shard = write_activation_shard(
        runner,
        examples,
        registered_layers=(4, 12, 20),
        destination=destination,
    )

    assert runner.active_handles == 0
    assert len(runner.calls) == 2
    assert shard == resume_activation_shard(destination)
    assert shard.row_count == 2
    assert shard.npz_sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert shard.index_sha256 == hashlib.sha256(shard.index_path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in shard.index_path.read_text(encoding="utf-8").splitlines()]
    assert [row["example_id"] for row in rows] == ["example-1", "example-2"]
    assert all(row["layer_ids"] == [4, 12, 20] for row in rows)
    assert all(row["shape"] == [3, 3, runner.hidden_size] for row in rows)
    assert all(row["dtype"] == "float16" for row in rows)
    assert all(len(row["activation_sha256"]) == 64 for row in rows)
    assert all(row["anchor_names"] == list(resolve_registered_anchors(examples[0], runner.tokenizer).anchor_names) for row in rows)
    assert all("rendered_utf8_hex" in row["anchors"] for row in rows)
    assert all("hidden_states" not in row for row in rows)


def test_verified_loader_reconstructs_typed_activation_records_in_index_order(tmp_path):
    runner = FakeRunner()
    examples = (example(example_id="example-1"), example(example_id="example-2"))
    shard = write_activation_shard(
        runner,
        examples,
        registered_layers=(4, 12, 20),
        destination=tmp_path / "typed.npz",
    )

    records = load_activation_records(shard.manifest_path)

    assert tuple(record.example_id for record in records) == ("example-1", "example-2")
    assert all(record.layer_ids == (4, 12, 20) for record in records)
    assert all(record.shape == (3, 3, runner.hidden_size) for record in records)
    assert all(record.activations.flags.writeable is False for record in records)
    assert all(
        record.activation_sha256
        == hashlib.sha256(record.activation_hash_payload).hexdigest()
        for record in records
    )


def test_verified_loader_requires_the_exact_sidecar_manifest_path(tmp_path):
    shard = write_activation_shard(
        FakeRunner(),
        (example(example_id="example-1"),),
        registered_layers=(4,),
        destination=tmp_path / "typed.npz",
    )

    with pytest.raises(ValueError, match="manifest path"):
        load_activation_records(shard.npz_path)


def test_activation_shards_are_deterministic_resume_without_running_and_no_clobber(tmp_path):
    examples = (example(example_id="example-1"), example(example_id="example-2"))
    first = write_activation_shard(
        FakeRunner(), examples, registered_layers=(4, 12, 20), destination=tmp_path / "a" / "same.npz"
    )
    second = write_activation_shard(
        FakeRunner(), examples, registered_layers=(4, 12, 20), destination=tmp_path / "b" / "same.npz"
    )

    assert first.npz_path.read_bytes() == second.npz_path.read_bytes()
    assert first.index_path.read_bytes() == second.index_path.read_bytes()
    resumed = write_activation_shard(
        NeverRunRunner(),
        examples,
        registered_layers=(4, 12, 20),
        destination=first.npz_path,
    )
    assert resumed == first
    with pytest.raises(FileExistsError, match="different extraction request"):
        write_activation_shard(
            NeverRunRunner(),
            examples,
            registered_layers=(4, 12),
            destination=first.npz_path,
        )


@pytest.mark.parametrize("mutation", ["target", "block", "anchor"])
def test_resume_request_hash_binds_content_and_resolved_anchor_provenance(tmp_path, mutation):
    user_text = "Ada met Ada. Task: Ada has code K7M2Q. What is Ada's archive code?"
    task_intro = user_text.index("Ada", user_text.index("Task:"))
    query = user_text.rindex("Ada")
    original = example(
        user_text,
        target_intro_span=(task_intro, task_intro + 3),
        target_query_span=(query, query + 3),
    )
    altered = original
    if mutation == "target":
        intro = user_text.index("code")
        target_query = user_text.rindex("code")
        altered = example(
            user_text,
            target_text="code",
            target_intro_span=(intro, intro + 4),
            target_query_span=(target_query, target_query + 4),
        )
    elif mutation == "block":
        altered = example(
            user_text,
            block="same_string",
            target_intro_span=(task_intro, task_intro + 3),
            target_query_span=(query, query + 3),
        )
    else:
        earlier_intro = user_text.index("Ada")
        altered = example(
            user_text,
            target_intro_span=(earlier_intro, earlier_intro + 3),
            target_query_span=(query, query + 3),
        )

    destination = tmp_path / f"{mutation}.npz"
    write_activation_shard(
        FakeRunner(), (original,), registered_layers=(4,), destination=destination
    )

    with pytest.raises(FileExistsError, match="different extraction request"):
        write_activation_shard(
            NeverRunRunner(),
            (altered,),
            registered_layers=(4,),
            destination=destination,
        )


@pytest.mark.parametrize(
    "field,label,value",
    [
        ("model_id", "model ID", None),
        ("model_revision", "model revision", "unrecorded-revision"),
        ("tokenizer_id", "tokenizer ID", "unknown"),
        ("tokenizer_revision", "tokenizer revision", "unrecorded"),
    ],
)
def test_extraction_rejects_unrecorded_runner_provenance_before_installing_hooks(
    field, label, value
):
    runner = FakeRunner()
    setattr(runner, field, value)

    with pytest.raises(ValueError, match=label):
        extract_registered_anchors(runner, example(), registered_layers=(4,))

    assert runner.calls == []


@pytest.mark.parametrize(
    "revision",
    ["main", "latest", "refs/tags/v1.0", "A" * 40, "a" * 39, "a" * 41],
)
def test_extraction_rejects_mutable_or_malformed_model_revisions_before_hooks(revision):
    runner = FakeRunner()
    runner.model_revision = revision

    with pytest.raises(ValueError, match="model revision"):
        extract_registered_anchors(runner, example(), registered_layers=(4,))

    assert runner.calls == []


class ExtractOnlyRunner:
    hidden_size = 5
    model_id = "fake/model"
    model_revision = MODEL_REVISION
    tokenizer_id = "fake/tokenizer"
    tokenizer_revision = TOKENIZER_REVISION

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.extract_calls = 0

    def extract_selected(self, input_ids, *, layer_ids, positions):
        self.extract_calls += 1
        return np.zeros((len(layer_ids), len(positions), self.hidden_size), dtype=np.float32)


def test_extraction_rejects_extract_selected_only_runner_without_installing_hooks():
    runner = ExtractOnlyRunner()

    with pytest.raises(ValueError, match="selected_position_hooks.*run_selected"):
        extract_registered_anchors(runner, example(), registered_layers=(4,))

    assert runner.extract_calls == 0


@pytest.mark.parametrize("component", ["npz", "index", "manifest"])
def test_resume_rejects_symlinked_shard_components(tmp_path, component):
    shard = write_activation_shard(
        FakeRunner(), (example(),), registered_layers=(4,), destination=tmp_path / "shard.npz"
    )
    path = {
        "npz": shard.npz_path,
        "index": shard.index_path,
        "manifest": shard.manifest_path,
    }[component]
    real_path = path.with_name(f"real-{path.name}")
    path.rename(real_path)
    path.symlink_to(real_path.name)

    with pytest.raises(ValueError, match="symlink"):
        resume_activation_shard(shard.npz_path)


@pytest.mark.parametrize("component", ["npz", "index", "manifest"])
def test_publication_rejects_symlink_alias_components(tmp_path, component):
    destination = tmp_path / "publish.npz"
    paths = {
        "npz": destination,
        "index": destination.with_suffix(".jsonl"),
        "manifest": destination.with_suffix(".manifest.json"),
    }
    paths[component].symlink_to(tmp_path / "missing-target")

    with pytest.raises(ValueError, match="symlink"):
        write_activation_shard(
            FakeRunner(), (example(),), registered_layers=(4,), destination=destination
        )


def test_activation_resume_fails_closed_on_npz_or_index_tampering(tmp_path):
    examples = (example(example_id="example-1"),)
    npz_shard = write_activation_shard(
        FakeRunner(), examples, registered_layers=(4, 12), destination=tmp_path / "npz.npz"
    )
    npz_shard.npz_path.write_bytes(npz_shard.npz_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="NPZ hash mismatch"):
        resume_activation_shard(npz_shard.npz_path)

    index_shard = write_activation_shard(
        FakeRunner(), examples, registered_layers=(4, 12), destination=tmp_path / "index.npz"
    )
    index_shard.index_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="index hash mismatch"):
        resume_activation_shard(index_shard.npz_path)


def test_writer_rejects_partial_destination_instead_of_clobbering(tmp_path):
    destination = tmp_path / "partial.npz"
    destination.write_bytes(b"partial")

    with pytest.raises(FileExistsError, match="partial activation shard"):
        write_activation_shard(
            FakeRunner(), (example(),), registered_layers=(4,), destination=destination
        )

    assert destination.read_bytes() == b"partial"


class RecordingLayer(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.hook_counts_during_forward = []

    def forward(self, hidden):
        self.hook_counts_during_forward.append(len(self._forward_hooks))
        return (hidden + self.layer_id + 1,)


class TinyModel(torch.nn.Module):
    def __init__(self, *, fail=False, activation_dtype=torch.float32):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(RecordingLayer(index) for index in range(4))
        self.hidden_size = 5
        self.fail = fail
        self.activation_dtype = activation_dtype
        self.forward_kwargs = None

    def forward(self, *, input_ids, **kwargs):
        self.forward_kwargs = kwargs
        hidden = torch.arange(
            input_ids.shape[1] * self.hidden_size,
            dtype=torch.float32,
            device=input_ids.device,
        ).to(self.activation_dtype).reshape(1, input_ids.shape[1], self.hidden_size)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
            if self.fail and layer.layer_id == 1:
                raise RuntimeError("injected forward failure")
        return SimpleNamespace(last_hidden_state=hidden)


def test_hf_selected_position_runner_hooks_only_selected_layers_and_releases_them():
    model = TinyModel()
    runner = HFSelectedPositionRunner(
        model,
        FakeTokenizer(),
        model_id="fake/model",
        model_revision=MODEL_REVISION,
    )

    result = extract_registered_anchors(runner, example(), registered_layers=(1, 3))

    assert result.activations.shape == (3, 2, model.hidden_size)
    assert result.activations.dtype == np.float32
    assert model.forward_kwargs == {"use_cache": False}
    assert [layer.hook_counts_during_forward for layer in model.model.layers] == [
        [0],
        [1],
        [0],
        [1],
    ]
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_hf_selected_position_runner_converts_bfloat16_to_honest_float32_storage():
    model = TinyModel(activation_dtype=torch.bfloat16)
    runner = HFSelectedPositionRunner(
        model,
        FakeTokenizer(),
        model_id="fake/model",
        model_revision=MODEL_REVISION,
    )

    result = extract_registered_anchors(runner, example(), registered_layers=(1, 3))

    assert result.activations.dtype == np.float32
    assert result.dtype == "float32"
    assert runner._capture is None
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_hf_selected_position_runner_releases_hooks_after_model_failure():
    model = TinyModel(fail=True)
    runner = HFSelectedPositionRunner(
        model,
        FakeTokenizer(),
        model_id="fake/model",
        model_revision=MODEL_REVISION,
    )

    with pytest.raises(RuntimeError, match="injected forward failure"):
        extract_registered_anchors(runner, example(), registered_layers=(1, 3))

    assert all(not layer._forward_hooks for layer in model.model.layers)
