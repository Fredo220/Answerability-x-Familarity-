import re
from collections import Counter
import json

from trajectory_extractor.datasets.concept_mixing import (
    generate_concept_mixing_examples,
    write_examples_jsonl,
)


def test_concept_dataset_has_preregistered_split_sizes_and_no_group_leakage():
    examples = generate_concept_mixing_examples(total=1200, seed=17)

    counts = {split: sum(item.split == split for item in examples) for split in ("train", "val", "test")}
    assert counts == {"train": 720, "val": 240, "test": 240}

    family_sets = {
        split: {item.entity_family for item in examples if item.split == split}
        for split in ("train", "val", "test")
    }
    template_sets = {
        split: {item.template_group for item in examples if item.split == split}
        for split in ("train", "val", "test")
    }
    assert family_sets["train"].isdisjoint(family_sets["val"] | family_sets["test"])
    assert family_sets["val"].isdisjoint(family_sets["test"])
    assert template_sets["train"].isdisjoint(template_sets["val"] | template_sets["test"])
    assert all(item.answer in item.context for item in examples)
    assert len({item.prompt for item in examples}) == 1200
    assert all(item.distractor_count in {2, 4, 6} for item in examples)
    assert all(item.distractor_answers for item in examples)
    assert all(len(item.distractor_answers) == item.distractor_count for item in examples)
    assert all(len(item.context.splitlines()) == item.distractor_count + 1 for item in examples)

    for split in ("train", "val", "test"):
        split_examples = [item for item in examples if item.split == split]
        for field in ("relation", "distractor_count", "name_similarity", "entity_rarity"):
            counts_by_value = Counter(getattr(item, field) for item in split_examples)
            assert max(counts_by_value.values()) - min(counts_by_value.values()) <= 1
        for distractor_count in (2, 4, 6):
            positions = {
                item.answer_position
                for item in split_examples
                if item.distractor_count == distractor_count
            }
            assert positions == set(range(distractor_count + 1))

    train_pairs = Counter(
        (item.template_group, item.entity_rarity)
        for item in examples
        if item.split == "train"
    )
    assert len(train_pairs) == 4
    assert min(train_pairs.values()) > 0

    entity_sets = {}
    fact_sets = {}
    for split in ("train", "val", "test"):
        split_examples = [item for item in examples if item.split == split]
        entity_sets[split] = {
            value
            for item in split_examples
            for value in re.findall(r"(?:Dr\.|Prof\.) [A-Za-z]+ [A-Za-z]+", item.context)
        }
        fact_sets[split] = {
            (item.target_entity, item.relation, item.answer) for item in split_examples
        }
    assert entity_sets["train"].isdisjoint(entity_sets["val"] | entity_sets["test"])
    assert entity_sets["val"].isdisjoint(entity_sets["test"])
    assert fact_sets["train"].isdisjoint(fact_sets["val"] | fact_sets["test"])


def test_concept_dataset_is_deterministic():
    first = generate_concept_mixing_examples(total=20, seed=4)
    second = generate_concept_mixing_examples(total=20, seed=4)

    assert first == second


def test_concept_dataset_writer_records_hash_and_design(tmp_path):
    examples = generate_concept_mixing_examples(total=20, seed=4)
    path = tmp_path / "concept.jsonl"
    write_examples_jsonl(examples, path, seed=4)
    manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["seed"] == 4
    assert manifest["count"] == 20
    assert manifest["schema_version"] == 2
    assert len(manifest["sha256"]) == 64
