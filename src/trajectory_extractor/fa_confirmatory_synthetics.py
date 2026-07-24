"""Pinned-tokenizer pseudonym construction for the confirmatory FA corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trajectory_extractor import fa_entities
from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_entities import CandidateEntity, SyntheticCandidate
from trajectory_extractor.fa_runtime import load_pinned_tokenizer

GENERATOR_REVISION = "fa-confirmatory-pseudonyms-v2"
DEFAULT_VARIANTS_PER_ENTITY = 3
MAX_ATTEMPTS_PER_ENTITY = 5000

_DOMAIN_WORDS = {
    "person": (
        "Adrian", "Alina", "Amelia", "Caleb", "Celia", "Clara", "Dorian",
        "Elena", "Elias", "Emilia", "Felix", "Helena", "Jonas", "Julian",
        "Laura", "Leona", "Lucas", "Marian", "Mira", "Nadia", "Nolan",
        "Oscar", "Selena", "Silas", "Thea", "Victor", "Alden", "Bennett",
        "Carver", "Dalton", "Ellison", "Foster", "Harlan", "Lennox",
        "Mercer", "Norton", "Parker", "Rowan", "Sawyer", "Sutton", "Vernon",
    ),
    "place": (
        "Aster", "Belmont", "Cedar", "Dunwich", "Eldora", "Fairview",
        "Glenhaven", "Harbor", "Lakeside", "Linden", "Marlowe", "North",
        "Oakridge", "Port", "Riverton", "Rosewood", "Silver", "South",
        "Stonehaven", "Valley", "West", "Willow", "Windermere",
    ),
    "organization": (
        "Alliance", "Agency", "Association", "Center", "Collective", "Council",
        "Federation", "Forum", "Foundation", "Global", "Group", "Institute",
        "International", "Laboratory", "Network", "Research", "Society",
        "Trust", "Union", "United", "World",
    ),
    "creative_work": (
        "Amber", "Astral", "Beyond", "Broken", "Crystal", "Dawn", "Distant",
        "Echo", "Elarion", "Ember", "Falling", "Golden", "Harbor", "Hidden",
        "Journey", "Last", "Light", "Midnight", "River", "Selora", "Shadow",
        "Silent", "Sky", "Song", "Stone", "Storm", "Velora", "Winter",
    ),
}
_LOWER_WORDS = (
    "a", "an", "as", "at", "by", "for", "from", "in", "of", "on", "the",
    "to", "under", "with", "within",
)
_CONSONANTS = "bcdfghjklmnprstvwyz"
_VOWELS = "aeiou"


def generate_synthetic_candidates(
    candidates: Sequence[CandidateEntity],
    tokenizer: Any,
    *,
    variants_per_entity: int = DEFAULT_VARIANTS_PER_ENTITY,
    allow_incomplete: bool = False,
) -> tuple[SyntheticCandidate, ...]:
    """Generate multiple unique compatible pseudonyms for every source entity."""
    rows = tuple(candidates)
    if not rows:
        raise ValueError("synthetic construction requires candidate entities")
    if type(variants_per_entity) is not int or variants_per_entity < 1:
        raise ValueError("variants_per_entity must be a positive integer")
    forbidden = {
        _normal_form(value)
        for candidate in rows
        for value in (
            candidate.name,
            *(alias for aliases in candidate.screening_aliases for alias in aliases),
        )
    }
    output = []
    used = set(forbidden)
    for candidate in rows:
        candidate_start = len(output)
        accepted = 0
        for attempt in range(MAX_ATTEMPTS_PER_ENTITY):
            name = _pseudonym(candidate, attempt)
            normalized = _normal_form(name)
            if normalized in used:
                continue
            synthetic = SyntheticCandidate(
                candidate_id=(
                    f"syn-{candidate.entity_id}-v{accepted + 1:02d}"
                ),
                name=name,
                coarse_type=candidate.coarse_type,
                split=candidate.split,
                generator_revision=GENERATOR_REVISION,
            )
            if not fa_entities._surface_compatible(
                candidate, synthetic, tokenizer
            ):
                continue
            output.append(synthetic)
            used.add(normalized)
            accepted += 1
            if accepted == variants_per_entity:
                break
        if accepted != variants_per_entity:
            if allow_incomplete:
                for synthetic in output[candidate_start:]:
                    used.remove(_normal_form(synthetic.name))
                del output[candidate_start:]
                continue
            raise ValueError(
                f"no complete pinned-tokenizer pseudonym reserve for {candidate.entity_id}"
            )
    return tuple(output)


def generate_synthetic_manifests(
    candidate_paths: Sequence[Path],
    *,
    output_dir: Path,
    config: FAConfig,
    variants_per_entity: int = DEFAULT_VARIANTS_PER_ENTITY,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Load all split manifests, generate candidates, and verify global matching."""
    if config.profile != "confirmatory":
        raise ValueError("confirmatory pseudonyms require the confirmatory config")
    prepared = load_pinned_tokenizer(config)
    candidates = []
    for path in candidate_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"candidate manifest must contain a list: {path}")
        candidates.extend(
            CandidateEntity(
                **{
                    key: value
                    for key, value in row.items()
                    if key != "schema_version"
                }
            )
            for row in payload
        )
    if len({candidate.qid for candidate in candidates}) != len(candidates):
        raise ValueError("confirmatory candidate manifests contain duplicate QIDs")
    synthetic = generate_synthetic_candidates(
        candidates,
        prepared.tokenizer,
        variants_per_entity=variants_per_entity,
        allow_incomplete=True,
    )
    matchable_candidates = tuple(
        candidate
        for candidate in candidates
        if any(
            synthetic_candidate.candidate_id.startswith(
                f"syn-{candidate.entity_id}-"
            )
            for synthetic_candidate in synthetic
        )
    )
    matched = fa_entities.match_synthetic_entities(
        matchable_candidates, synthetic, prepared.tokenizer
    )
    if len(matched) != len(matchable_candidates):
        raise ValueError("pseudonym pool does not admit complete global matching")
    if require_complete and len(matchable_candidates) != len(candidates):
        raise ValueError(
            "selected source pool lost complete pseudonym reserves"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in sorted({candidate.split for candidate in candidates}):
        rows = [
            {"schema_version": 1, **asdict(candidate)}
            for candidate in synthetic
            if candidate.split == split
        ]
        path = output_dir / f"synthetic_candidates_{split}_v1.json"
        _write_json(path, rows)
        files[split] = {
            "path": str(path),
            "count": len(rows),
            "sha256": _sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "generator_revision": GENERATOR_REVISION,
        "model_id": config.model_id,
        "tokenizer_revision": config.tokenizer_revision,
        "chat_template_sha256": prepared.chat_template_sha256,
        "variants_per_entity": variants_per_entity,
        "candidate_count": len(candidates),
        "synthetic_count": len(synthetic),
        "matchable_candidate_count": len(matchable_candidates),
        "unmatchable_candidate_ids": sorted(
            candidate.entity_id
            for candidate in candidates
            if candidate not in matchable_candidates
        ),
        "complete_matching_count": len(matched),
        "files": files,
    }
    manifest_path = output_dir / "synthetic_source_snapshot_v1.json"
    _write_json(manifest_path, manifest)
    manifest["source_snapshot"] = str(manifest_path)
    manifest["source_snapshot_sha256"] = _sha256_file(manifest_path)
    return manifest


def _pseudonym(candidate: CandidateEntity, attempt: int) -> str:
    seed = int.from_bytes(
        hashlib.sha256(
            (
                f"{GENERATOR_REVISION}:{candidate.qid}:"
                f"{candidate.coarse_type}:{candidate.name}:{attempt}"
            ).encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = random.Random(seed)
    words = []
    domain_bank = _DOMAIN_WORDS[candidate.coarse_type]
    for index, source_word in enumerate(candidate.name.split()):
        letters = "".join(character for character in source_word if character.isalpha())
        length = len(source_word)
        pattern = _capitalization(source_word)
        if letters and len(letters) != length:
            words.append(
                _punctuation_shaped_word(
                    source_word,
                    domain_bank=domain_bank,
                    rng=rng,
                    offset=index + attempt,
                )
            )
            continue
        bank = _LOWER_WORDS if pattern == "lower" else domain_bank
        exact = [word for word in bank if len(word) == length]
        if exact:
            word = exact[rng.randrange(len(exact))]
            word = _apply_case(word, pattern, source_word)
        else:
            word = _generated_word(
                length,
                rng=rng,
                offset=index + attempt,
            )
            word = _apply_case(word, pattern, source_word)
        if not letters:
            word = source_word
        words.append(word)
    return " ".join(words)


def _punctuation_shaped_word(
    source_word: str,
    *,
    domain_bank: Sequence[str],
    rng: random.Random,
    offset: int,
) -> str:
    output = []
    start = 0
    segment_index = 0
    while start < len(source_word):
        if not source_word[start].isalpha():
            output.append(source_word[start])
            start += 1
            continue
        end = start + 1
        while end < len(source_word) and source_word[end].isalpha():
            end += 1
        source_segment = source_word[start:end]
        pattern = _capitalization(source_segment)
        bank = _LOWER_WORDS if pattern == "lower" else domain_bank
        exact = [word for word in bank if len(word) == len(source_segment)]
        if exact:
            replacement = exact[rng.randrange(len(exact))]
        else:
            replacement = _generated_word(
                len(source_segment),
                rng=rng,
                offset=offset + segment_index,
            )
        output.append(_apply_case(replacement, pattern, source_segment))
        segment_index += 1
        start = end
    return "".join(output)


def _generated_word(length: int, *, rng: random.Random, offset: int) -> str:
    if length <= 0:
        return ""
    start_with_vowel = (rng.randrange(2) + offset) % 2 == 0
    characters = []
    for index in range(length):
        use_vowel = (index % 2 == 0) == start_with_vowel
        alphabet = _VOWELS if use_vowel else _CONSONANTS
        characters.append(alphabet[rng.randrange(len(alphabet))])
    return "".join(characters)


def _capitalization(word: str) -> str:
    letters = "".join(character for character in word if character.isalpha())
    if not letters:
        return "none"
    if letters.isupper():
        return "upper"
    if letters.islower():
        return "lower"
    if letters[0].isupper() and letters[1:].islower():
        return "title"
    return "mixed"


def _apply_case(word: str, pattern: str, source_word: str) -> str:
    if pattern == "upper":
        return word.upper()
    if pattern == "lower":
        return word.lower()
    if pattern == "title":
        return word.title()
    if pattern == "mixed":
        return "".join(
            character.upper() if source.isupper() else character.lower()
            for character, source in zip(word, source_word, strict=True)
        )
    return word


def _normal_form(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _write_json(path: Path, payload: Any) -> None:
    normalized = json.loads(
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == normalized:
            return
        raise FileExistsError(
            f"refusing to overwrite a non-identical confirmatory synthetic file: {path}"
        )
    serialized = json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"stale confirmatory synthetic temporary file requires audit: {temporary}"
        )
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate pinned-tokenizer confirmatory pseudonym reserves."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants-per-entity",
        type=int,
        default=DEFAULT_VARIANTS_PER_ENTITY,
    )
    args = parser.parse_args(argv)
    result = generate_synthetic_manifests(
        args.candidate_manifest,
        output_dir=args.output_dir,
        config=FAConfig.from_json(args.config),
        variants_per_entity=args.variants_per_entity,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
