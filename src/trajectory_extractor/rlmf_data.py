from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets import load_dataset

from trajectory_extractor.rlmf_artifacts import RLMFArtifactStore
from trajectory_extractor.rlmf_types import PopQAExample, RLMFConfig


_SPLIT_ORDER = ("pre_sft", "rl_train", "validation", "test")
_MINIMUM_CONFIRMATORY_COMPONENTS = 896


def normalize_popqa_row(row: Mapping[str, Any]) -> PopQAExample:
    """Return the stable, alias-normalized record used by the RLMF snapshot."""
    if not isinstance(row, Mapping):
        raise ValueError("PopQA row must be a mapping")
    subject = _subject_id(row)
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("PopQA row must contain a non-empty question")
    aliases = row.get("possible_answers")
    if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Sequence):
        raise ValueError("PopQA row must contain possible_answers")
    normalized = tuple(sorted({_normalize_alias(alias) for alias in aliases if _normalize_alias(alias)}))
    if not normalized:
        raise ValueError("PopQA row must contain at least one non-empty possible answer")
    return PopQAExample(
        example_id="popqa-" + hashlib.sha256(subject.encode("utf-8")).hexdigest()[:16],
        subject=subject,
        question=" ".join(question.split()),
        answers=normalized,
    )


def select_subject_and_answer_disjoint_splits(
    examples: Sequence[PopQAExample], counts: Mapping[str, int], split_seed: int
) -> dict[str, tuple[PopQAExample, ...]]:
    indexes, _component_ids, _discarded, _eligible = _select_indexes(examples, counts, split_seed)
    return {
        split: tuple(examples[index] for index in indexes[split]) for split in _SPLIT_ORDER
    }


def write_popqa_snapshot(config: RLMFConfig, store: RLMFArtifactStore) -> dict[str, Path]:
    """Fetch and seal the pinned PopQA test split without relaxing disjointness."""
    if not isinstance(config, RLMFConfig):
        raise ValueError("config must be an RLMFConfig")
    if not isinstance(store, RLMFArtifactStore):
        raise ValueError("store must be an RLMFArtifactStore")
    rows = list(
        load_dataset(config.dataset_id, revision=config.dataset_revision, split="test")
    )
    examples = [normalize_popqa_row(row) for row in rows]
    indexes, component_ids, discarded, eligible = _select_indexes(
        examples, config.split_counts, config.split_seed
    )
    if eligible < _MINIMUM_CONFIRMATORY_COMPONENTS:
        raise ValueError(
            f"PopQA has only {eligible} eligible components; require at least "
            f"{_MINIMUM_CONFIRMATORY_COMPONENTS} without relaxing disjointness"
        )

    split_for_index = {
        index: split for split, selected in indexes.items() for index in selected
    }
    normalized_rows = [
        {
            "example_id": examples[index].example_id,
            "subject_id": examples[index].subject,
            "question": examples[index].question,
            "answers": list(examples[index].answers),
            "split": split_for_index[index],
            "relation": _relation(rows[index]),
            "popularity": _popularity(rows[index]),
            "alias_component_id": component_ids[index],
        }
        for index in sorted(split_for_index)
    ]
    aliases = [
        {
            "example_id": normalized["example_id"],
            "subject_id": normalized["subject_id"],
            "alias_component_id": normalized["alias_component_id"],
            "aliases": normalized["answers"],
        }
        for normalized in normalized_rows
    ]
    source_rows = [
        {
            "source_index": index,
            "row": dict(row),
            "example_id": examples[index].example_id,
            "subject_id": examples[index].subject,
            "aliases": list(examples[index].answers),
            "alias_component_id": component_ids[index],
        }
        for index, row in enumerate(rows)
    ]
    discarded_rows = [
        {
            "source_index": index,
            "example_id": examples[index].example_id,
            "subject_id": examples[index].subject,
            "aliases": list(examples[index].answers),
            "alias_component_id": component_ids[index],
            "reason": reason,
        }
        for index, reason in sorted(discarded.items())
    ]
    manifest = {
        "study_id": config.study_id,
        "dataset": {
            "id": config.dataset_id,
            "revision": config.dataset_revision,
            "split": "test",
        },
        "split_seed": config.split_seed,
        "counts": {split: len(indexes[split]) for split in _SPLIT_ORDER},
        "split_ids": {
            split: [examples[index].example_id for index in indexes[split]]
            for split in _SPLIT_ORDER
        },
        "relation_counts": _relation_counts(normalized_rows),
        "popularity": _popularity_summary(normalized_rows),
        "eligible_components": eligible,
        "discarded_rows": len(discarded_rows),
    }
    paths = {
        "source_rows": store.write_jsonl(config.study_id, "data", "popqa_source_rows", source_rows),
        "normalized_rows": store.write_jsonl(
            config.study_id, "data", "popqa_snapshot", normalized_rows
        ),
        "aliases": store.write_jsonl(config.study_id, "data", "aliases", aliases),
        "discarded_rows": store.write_jsonl(
            config.study_id, "data", "discarded_rows", discarded_rows
        ),
        "split_manifest": store.write_json(
            config.study_id, "data", "split_manifest", manifest
        ),
    }
    paths["completion"] = store.complete_endpoint(
        config.study_id, "prepare-data", config, paths
    )
    return paths


def _select_indexes(
    examples: Sequence[PopQAExample], counts: Mapping[str, int], split_seed: int
) -> tuple[dict[str, tuple[int, ...]], dict[int, str], dict[int, str], int]:
    if set(counts) != set(_SPLIT_ORDER) or any(
        type(count) is not int or count < 1 for count in counts.values()
    ):
        raise ValueError("counts must contain positive values for every registered split")
    if type(split_seed) is not int or split_seed < 1:
        raise ValueError("split_seed must be a positive integer")

    by_subject: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        if not isinstance(example, PopQAExample):
            raise ValueError("examples must contain PopQAExample records")
        by_subject[example.subject].append(index)

    # All source aliases participate in connectivity, including aliases that
    # belong to rows later discarded as duplicate subjects.
    parents = list(range(len(examples)))
    aliases_to_index: dict[str, int] = {}
    for subject_rows in by_subject.values():
        first_index = subject_rows[0]
        for index in subject_rows[1:]:
            _union(parents, first_index, index)
    for index, example in enumerate(examples):
        for alias in example.answers:
            previous = aliases_to_index.setdefault(alias, index)
            _union(parents, index, previous)

    subject_indexes: list[int] = []
    discarded: dict[int, str] = {}
    for subject, subject_rows in by_subject.items():
        chosen = min(subject_rows, key=lambda index: _example_tie_key(examples[index]))
        subject_indexes.append(chosen)
        for index in subject_rows:
            if index != chosen:
                discarded[index] = "duplicate_subject"

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(examples)):
        components[_find(parents, index)].append(index)
    component_ids: dict[int, str] = {}
    retained: list[int] = []
    chosen_subject_indexes = set(subject_indexes)
    for members in components.values():
        aliases = sorted(
            {alias for index in members for alias in examples[index].answers}
        )
        component_id = "component-" + hashlib.sha256(
            "\x00".join(aliases).encode("utf-8")
        ).hexdigest()[:16]
        for index in members:
            component_ids[index] = component_id
        candidates = [index for index in members if index in chosen_subject_indexes]
        selected = min(
            candidates,
            key=lambda index: (_selection_hash(split_seed, examples[index].subject), _example_tie_key(examples[index])),
        )
        retained.append(selected)
        for index in candidates:
            if index != selected:
                discarded[index] = "answer_component_overlap"

    retained.sort(
        key=lambda index: (_selection_hash(split_seed, examples[index].subject), _example_tie_key(examples[index]))
    )
    required = sum(counts.values())
    if len(retained) < required:
        raise ValueError(
            f"PopQA has {len(retained)} eligible components, but {required} are required"
        )
    splits: dict[str, tuple[int, ...]] = {}
    cursor = 0
    for split in _SPLIT_ORDER:
        next_cursor = cursor + counts[split]
        splits[split] = tuple(retained[cursor:next_cursor])
        cursor = next_cursor
    return splits, component_ids, discarded, len(retained)


def _subject_id(row: Mapping[str, Any]) -> str:
    for field in ("s_uri", "subject_uri", "subj_id", "subject_id", "subject", "subj"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("PopQA row must contain a subject URI or ID")


def _normalize_alias(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _selection_hash(split_seed: int, subject_id: str) -> str:
    return hashlib.sha256(f"{split_seed}:{subject_id}".encode("utf-8")).hexdigest()


def _example_tie_key(example: PopQAExample) -> str:
    return json.dumps(
        {"subject": example.subject, "question": example.question, "answers": example.answers},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _find(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _union(parents: list[int], left: int, right: int) -> None:
    left_root = _find(parents, left)
    right_root = _find(parents, right)
    if left_root != right_root:
        parents[max(left_root, right_root)] = min(left_root, right_root)


def _relation(row: Mapping[str, Any]) -> str | None:
    value = row.get("prop", row.get("relation"))
    return value.strip() if isinstance(value, str) and value.strip() else None


def _popularity(row: Mapping[str, Any]) -> float | None:
    value = row.get("s_pop", row.get("popularity"))
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        return None
    return float(value)


def _relation_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        relation = row["relation"]
        if relation is not None:
            counts[str(relation)] += 1
    return dict(sorted(counts.items()))


def _popularity_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    values = [float(row["popularity"]) for row in rows if row["popularity"] is not None]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }
