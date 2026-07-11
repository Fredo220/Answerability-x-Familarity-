from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JailbreakExample:
    example_id: str
    behavior: str
    category: str
    benign: bool
    source: str
    artifact: str | None = None
    pair_id: str | None = None
    behavior_name: str | None = None
    goal: str | None = None
    artifact_source: str | None = None


def load_official_jailbreakbench(path: str | Path) -> list[JailbreakExample]:
    """Load a frozen official export; this function never creates attacks."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    rows = _read_rows(source)
    examples: list[JailbreakExample] = []
    for index, row in enumerate(rows):
        behavior = _first(row, "prompt", "goal", "behavior")
        if not behavior:
            raise ValueError(f"Missing behavior text in row {index}")
        benign_value = str(row.get("benign", row.get("is_benign", "false"))).lower()
        examples.append(
            JailbreakExample(
                example_id=str(row.get("id", row.get("behavior_id", f"jbb-{index:03d}"))),
                behavior=behavior,
                category=str(row.get("category", "unknown")),
                benign=benign_value in {"1", "true", "yes"},
                source=str(row.get("source", "JailbreakBench")),
                artifact=row.get("artifact") or row.get("attack"),
                pair_id=str(row.get("pair_id")) if row.get("pair_id") is not None else None,
                behavior_name=(
                    str(row.get("behavior_name")) if row.get("behavior_name") is not None else None
                ),
                goal=str(row.get("goal")) if row.get("goal") is not None else None,
                artifact_source=(
                    str(row.get("artifact_source"))
                    if row.get("artifact_source") is not None
                    else None
                ),
            )
        )
    return examples


def build_jailbreak_study_file(
    harmful_path: str | Path,
    benign_path: str | Path,
    artifact_path: str | Path,
    output_path: str | Path,
    *,
    artifact_commit: str,
) -> dict[str, Any]:
    """Freeze the official matched JBB study without generating attack text."""
    harmful_source = Path(harmful_path)
    benign_source = Path(benign_path)
    artifact_source = Path(artifact_path)
    output = Path(output_path)
    harmful = _indexed_official_rows(harmful_source)
    benign = _indexed_official_rows(benign_source)
    artifact_rows = _indexed_artifact_rows(artifact_source)

    expected = set(range(100))
    if set(harmful) != expected or set(benign) != expected or set(artifact_rows) != expected:
        raise ValueError("Official harmful, benign, and artifact inputs must each contain indices 0..99")

    rows: list[dict[str, Any]] = []
    provenance = (
        "JailbreakBench/JBB-Behaviors + "
        f"JailbreakBench/artifacts@{artifact_commit}"
    )
    for index in sorted(expected):
        harmful_row = harmful[index]
        benign_row = benign[index]
        artifact_row = artifact_rows[index]
        for key in ("Behavior", "Category"):
            if _normalized_metadata(harmful_row[key]) != _normalized_metadata(benign_row[key]):
                raise ValueError(f"Harmful/benign mismatch at index {index}: {key}")
        expected_values = {
            "goal": harmful_row["Goal"],
            "behavior": harmful_row["Behavior"],
            "category": harmful_row["Category"],
        }
        for key, expected_value in expected_values.items():
            if str(artifact_row.get(key, "")) != expected_value:
                raise ValueError(f"Artifact mismatch at index {index}: {key}")
        frozen_prompt = artifact_row.get("prompt")
        if not isinstance(frozen_prompt, str) or not frozen_prompt.strip():
            raise ValueError(f"Artifact prompt is missing at index {index}")

        pair_id = f"jbb-pair-{index:03d}"
        shared = {
            "category": harmful_row["Category"],
            "source": provenance,
            "pair_id": pair_id,
            "behavior_name": harmful_row["Behavior"],
        }
        rows.append(
            {
                **shared,
                "id": f"jbb-harmful-{index:03d}",
                "prompt": frozen_prompt,
                "goal": harmful_row["Goal"],
                "benign": False,
                "artifact": frozen_prompt,
                "artifact_source": str(artifact_source),
            }
        )
        rows.append(
            {
                **shared,
                "id": f"jbb-benign-{index:03d}",
                "prompt": benign_row["Goal"],
                "goal": benign_row["Goal"],
                "benign": True,
                "artifact": None,
                "artifact_source": None,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    examples = load_official_jailbreakbench(output)
    counts = validate_jailbreak_study_set(examples)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_commit": artifact_commit,
        "artifact_file": str(artifact_source),
        "inputs": {
            "harmful": {"path": str(harmful_source), "sha256": _sha256(harmful_source)},
            "benign": {"path": str(benign_source), "sha256": _sha256(benign_source)},
            "artifact": {"path": str(artifact_source), "sha256": _sha256(artifact_source)},
        },
        "output": {"path": str(output), "sha256": _sha256(output)},
        "counts": counts,
        "pairing": "official index, Behavior, and Category equality",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_jailbreak_study_set(
    examples: list[JailbreakExample],
    *,
    expected_harmful: int = 100,
    expected_benign: int = 100,
) -> dict[str, int]:
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("JailbreakBench example IDs must be unique")
    harmful = [example for example in examples if not example.benign]
    benign = [example for example in examples if example.benign]
    if len(harmful) != expected_harmful or len(benign) != expected_benign:
        raise ValueError(
            f"Expected {expected_harmful} harmful and {expected_benign} benign examples; "
            f"found {len(harmful)} and {len(benign)}"
        )
    missing_artifacts = [example.example_id for example in harmful if not example.artifact]
    if missing_artifacts:
        raise ValueError(
            "Harmful behaviors require frozen published artifacts; missing for "
            + ", ".join(missing_artifacts[:5])
        )
    if any("jailbreakbench" not in example.source.casefold() for example in examples):
        raise ValueError("Every row must identify JailbreakBench as its source")
    if any(example.pair_id is None for example in examples):
        raise ValueError("Every harmful/benign row must have an explicit matched pair_id")
    pairs: dict[str, list[JailbreakExample]] = {}
    for example in examples:
        pairs.setdefault(str(example.pair_id), []).append(example)
    invalid_pairs = [
        pair_id
        for pair_id, rows in pairs.items()
        if len(rows) != 2
        or {row.benign for row in rows} != {False, True}
        or len({row.category for row in rows}) != 1
    ]
    if invalid_pairs:
        raise ValueError("Each pair_id must contain one harmful and one benign row")
    categories = {example.category for example in examples}
    if "unknown" in categories or len(categories) < 3:
        raise ValueError("Official study data requires at least three named behavior categories")
    return {
        "total": len(examples),
        "harmful": len(harmful),
        "benign": len(benign),
        "pairs": len(pairs),
    }


def grouped_folds(examples: list[JailbreakExample]) -> list[tuple[list[int], list[int]]]:
    """Leave-one-category-out folds prevent category leakage."""
    categories = sorted({example.category for example in examples})
    folds = []
    for category in categories:
        test = [index for index, example in enumerate(examples) if example.category == category]
        train = [index for index, example in enumerate(examples) if example.category != category]
        if train and test:
            folds.append((train, test))
    return folds


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        return value if isinstance(value, list) else value["behaviors"]
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("Expected .csv, .json, or .jsonl official export")


def _indexed_official_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = _read_rows(path)
    required = {"Index", "Goal", "Behavior", "Category", "Source"}
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Official JBB row is missing fields: {sorted(missing)}")
        index = int(row["Index"])
        if index in indexed:
            raise ValueError(f"Duplicate official JBB index: {index}")
        indexed[index] = row
    return indexed


def _indexed_artifact_rows(path: Path) -> dict[int, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("jailbreaks") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("Jailbreak artifact must be a list or contain a jailbreaks list")
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "index" not in row:
            raise ValueError("Every artifact row must be an object with an index")
        index = int(row["index"])
        if index in indexed:
            raise ValueError(f"Duplicate artifact index: {index}")
        indexed[index] = row
    return indexed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_metadata(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _first(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if row.get(key):
            return str(row[key])
    return ""
