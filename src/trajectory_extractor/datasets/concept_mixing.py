from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import random


@dataclass(frozen=True)
class ConceptMixingExample:
    example_id: str
    split: str
    entity_family: str
    template_group: str
    context: str
    question: str
    prompt: str
    answer: str
    relation: str
    distractor_count: int
    name_similarity: str
    answer_position: int
    target_entity: str
    distractor_answers: tuple[str, ...]
    entity_rarity: str


SPLIT_COUNTS = {"train": 720, "val": 240, "test": 240}
SPLIT_TEMPLATE_GROUPS = {
    "train": ("train_direct", "train_table"),
    "val": ("val_query",),
    "test": ("test_audit",),
}
RELATIONS = ("invented", "discovered", "mapped", "designed")
OBJECTS = (
    "Zephyroscope",
    "Valtirium",
    "Navor Chains",
    "Orilon Engine",
    "Ceryx Field",
    "Talven Lens",
    "Meridian Coil",
    "Aster Prism",
    "Boreal Array",
    "Kelvin Loom",
    "Solace Rotor",
    "Vesper Gauge",
)


def generate_concept_mixing_examples(total: int = 1200, seed: int = 42) -> list[ConceptMixingExample]:
    if total < 1:
        raise ValueError("total must be positive")
    if total == 1200:
        counts = SPLIT_COUNTS
    else:
        train = int(total * 0.6)
        val = int(total * 0.2)
        counts = {"train": train, "val": val, "test": total - train - val}
    examples: list[ConceptMixingExample] = []
    family_offset = 0
    for split_index, (split, count) in enumerate(counts.items()):
        rng = random.Random(seed + split_index * 10_000)
        templates = SPLIT_TEMPLATE_GROUPS[split]
        distractor_counts = _balanced_values((2, 4, 6), count, rng)
        similarities = _balanced_values((True, False), count, rng)
        rarities = _balanced_values((True, False), count, rng)
        target_indices = _balanced_values(tuple(range(7)), count, rng)
        relations = _balanced_values(RELATIONS, count, rng)
        answers = _balanced_values(OBJECTS, count, rng)
        template_groups = _balanced_values(templates, count, rng)
        answer_positions = [0] * count
        for distractor_count in (2, 4, 6):
            indices = [
                index
                for index, value in enumerate(distractor_counts)
                if value == distractor_count
            ]
            positions = _balanced_values(tuple(range(distractor_count + 1)), len(indices), rng)
            for index, position in zip(indices, positions, strict=True):
                answer_positions[index] = position
        for local_index in range(count):
            family_number = family_offset + local_index // 20
            family = f"{split}-family-{family_number:03d}"
            rare = bool(rarities[local_index])
            root = _root_name(family_number * 2 + int(rare), rare=rare)
            similar = bool(similarities[local_index])
            names = _entity_names(root, similar, count=7)
            target_index = int(target_indices[local_index])
            relation = str(relations[local_index])
            answer = str(answers[local_index])
            distractor_count = int(distractor_counts[local_index])
            target_fact = f"{names[target_index]} {relation} the {answer}."
            distractor_facts = []
            distractor_answers = []
            available_objects = [value for value in OBJECTS if value != answer]
            object_start = (seed + local_index * 5) % len(available_objects)
            available_objects = (
                available_objects[object_start:] + available_objects[:object_start]
            )
            for distractor in range(distractor_count):
                entity_index = (target_index + distractor + 1) % len(names)
                # The first distractor shares the queried relation. This makes
                # the task test entity-relation binding rather than keyword lookup.
                distractor_relation = (
                    relation
                    if distractor == 0
                    else RELATIONS[(RELATIONS.index(relation) + distractor) % len(RELATIONS)]
                )
                distractor_object = available_objects[distractor]
                distractor_answers.append(distractor_object)
                distractor_facts.append(
                    f"{names[entity_index]} {distractor_relation} the {distractor_object}."
                )
            answer_position = answer_positions[local_index]
            facts = distractor_facts.copy()
            facts.insert(answer_position, target_fact)
            context = "\n".join(facts)
            template_group = str(template_groups[local_index])
            question = _question(template_group, names[target_index], relation)
            prompt = (
                "Use only the facts below. Answer with the object name only.\n\n"
                f"{context}\n\nQuestion: {question}\nAnswer:"
            )
            examples.append(
                ConceptMixingExample(
                    example_id=f"{split}-{local_index:04d}",
                    split=split,
                    entity_family=family,
                    template_group=template_group,
                    context=context,
                    question=question,
                    prompt=prompt,
                    answer=answer,
                    relation=relation,
                    distractor_count=distractor_count,
                    name_similarity="high" if similar else "low",
                    answer_position=answer_position,
                    target_entity=names[target_index],
                    distractor_answers=tuple(sorted(set(distractor_answers))),
                    entity_rarity="rare_synthetic" if rare else "common_form",
                )
            )
        family_offset += (count + 19) // 20 + 100
    return examples


def write_examples_jsonl(
    examples: list[ConceptMixingExample], path: str | Path, *, seed: int | None = None
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), sort_keys=True) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 2,
        "generator": "trajectory_extractor.datasets.concept_mixing",
        "seed": seed,
        "count": len(examples),
        "split_counts": {
            split: sum(example.split == split for example in examples)
            for split in ("train", "val", "test")
        },
        "sha256": digest,
        "balanced_factors": [
            "relation",
            "distractor_count",
            "name_similarity",
            "entity_rarity",
            "answer_position_within_distractor_count",
        ],
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _root_name(index: int, *, rare: bool) -> str:
    # Encode the full family index so names never repeat across split groups.
    syllables = ("ba", "ce", "di", "fo", "gu", "ha", "jo", "ki", "lu", "me")
    if rare:
        first = "A" + syllables[index % 10] + syllables[(index // 10) % 10]
    else:
        common = ("Anna", "David", "Emma", "James", "Maria", "Robert", "Sarah", "Thomas")
        first = common[index % len(common)]
    last = "R" + "".join(
        syllables[(index // divisor) % 10] for divisor in (1, 10, 100, 1000)
    )
    return f"{first.title()} {last.title()}"


def _entity_names(root: str, similar: bool, *, count: int) -> tuple[str, ...]:
    first, last = root.split()
    if similar:
        suffixes = ("", "a", "en", "is", "on", "ar", "el")
        return tuple(f"Dr. {first}{suffix} {last}" for suffix in suffixes[:count])
    first_names = (first, "Sora", "Tavin", "Liora", "Marek", "Nadia", "Elian")
    titles = ("Dr.", "Prof.", "Dr.", "Prof.", "Dr.", "Prof.", "Dr.")
    return tuple(
        f"{title} {given} {last}" for title, given in zip(titles[:count], first_names[:count], strict=True)
    )


def _balanced_values(values: tuple, count: int, rng: random.Random) -> list:
    if not values:
        raise ValueError("balanced factor requires at least one value")
    repeated = [values[index % len(values)] for index in range(count)]
    rng.shuffle(repeated)
    return repeated


def _question(template: str, entity: str, relation: str) -> str:
    base = {"invented": "invent", "discovered": "discover", "mapped": "map", "designed": "design"}[
        relation
    ]
    if template.endswith("table"):
        return f"Identify the object that {entity} {relation}."
    if template.endswith("query"):
        return f"Which object was {relation} by {entity}?"
    if template.endswith("audit"):
        return f"According to the supplied records, which object did {entity} {base}?"
    return f"What object did {entity} {base}?"
