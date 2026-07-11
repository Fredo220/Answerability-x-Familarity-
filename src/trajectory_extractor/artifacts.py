from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from trajectory_extractor.types import ActivationRun, ResponseRun, TrajectoryBatch


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write(self, run: ActivationRun) -> Path:
        run.validate()
        example_id = _safe_id(run.example_id)
        directory = self.root / _safe_id(run.run_id) / "examples"
        directory.mkdir(parents=True, exist_ok=True)
        array_path = directory / f"{example_id}.npz"
        metadata_path = directory / f"{example_id}.json"
        np.savez_compressed(
            array_path,
            response_token_ids=run.response_token_ids.astype(np.int64),
            hidden_states=run.hidden_states.astype(np.float16),
            token_logprobs=run.token_logprobs.astype(np.float32),
            token_entropies=run.token_entropies.astype(np.float32),
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "example_id": run.example_id,
                    "track": run.track,
                    "split": run.split,
                    "prompt": run.prompt,
                    "response": run.response,
                    "label": int(run.label),
                    "input_token_count": int(run.input_token_count),
                    "provenance": run.provenance,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return array_path

    def write_manifest(self, run_id: str, manifest: dict) -> Path:
        directory = self.root / _safe_id(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        for child in ("labels", "metrics", "bootstrap", "figures"):
            (directory / child).mkdir(exist_ok=True)
        return path

    def write_response(self, run: ResponseRun) -> Path:
        directory = self.root / _safe_id(run.run_id) / "responses"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe_id(run.example_id)}.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "example_id": run.example_id,
                    "track": run.track,
                    "split": run.split,
                    "prompt": run.prompt,
                    "response": run.response,
                    "label": int(run.label),
                    "provenance": run.provenance,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return path

    def write_json(self, run_id: str, section: str, name: str, value) -> Path:
        if section not in {"labels", "metrics", "bootstrap"}:
            raise ValueError("section must be labels, metrics, or bootstrap")
        directory = self.root / _safe_id(run_id) / section
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe_id(name)}.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True))
        return path

    def read(self, run_id: str, example_id: str) -> ActivationRun:
        directory = self.root / _safe_id(run_id) / "examples"
        safe_id = _safe_id(example_id)
        metadata = json.loads((directory / f"{safe_id}.json").read_text())
        arrays = np.load(directory / f"{safe_id}.npz")
        run = ActivationRun(
            **metadata,
            response_token_ids=arrays["response_token_ids"],
            hidden_states=arrays["hidden_states"],
            token_logprobs=arrays["token_logprobs"],
            token_entropies=arrays["token_entropies"],
        )
        run.validate()
        return run

    def read_response(self, run_id: str, example_id: str) -> ResponseRun:
        path = self.root / _safe_id(run_id) / "responses" / f"{_safe_id(example_id)}.json"
        return ResponseRun(**json.loads(path.read_text()))

    def example_ids(self, run_id: str) -> list[str]:
        directory = self.root / _safe_id(run_id) / "examples"
        return [json.loads(path.read_text())["example_id"] for path in sorted(directory.glob("*.json"))]

    def response_ids(self, run_id: str) -> list[str]:
        directory = self.root / _safe_id(run_id) / "responses"
        return [json.loads(path.read_text())["example_id"] for path in sorted(directory.glob("*.json"))]

    def judgable_ids(self, run_id: str) -> list[str]:
        examples = self.example_ids(run_id)
        responses = self.response_ids(run_id)
        if examples and responses:
            raise ValueError("A judged run cannot mix activation and response-only records")
        return examples or responses

    def read_judgable(self, run_id: str, example_id: str) -> ActivationRun | ResponseRun:
        if self.response_ids(run_id):
            return self.read_response(run_id, example_id)
        return self.read(run_id, example_id)

    def write_judgable(self, run: ActivationRun | ResponseRun) -> Path:
        if isinstance(run, ResponseRun):
            return self.write_response(run)
        return self.write(run)

    def has_example(self, run_id: str, example_id: str) -> bool:
        directory = self.root / _safe_id(run_id) / "examples"
        safe_id = _safe_id(example_id)
        return (directory / f"{safe_id}.json").exists() and (directory / f"{safe_id}.npz").exists()

    def has_response(self, run_id: str, example_id: str) -> bool:
        return (
            self.root / _safe_id(run_id) / "responses" / f"{_safe_id(example_id)}.json"
        ).exists()

    def load_batch(self, run_id: str, *, label_key: str | None = None) -> TrajectoryBatch:
        directory = self.root / _safe_id(run_id) / "examples"
        metadata_paths = sorted(directory.glob("*.json"))
        if not metadata_paths:
            raise FileNotFoundError(f"No examples found for run {run_id!r}")
        metadata = [json.loads(path.read_text()) for path in metadata_paths]
        if label_key not in (None, "exact_error"):
            missing = [
                str(item["example_id"])
                for item in metadata
                if label_key not in item.get("provenance", {})
            ]
            if missing:
                raise ValueError(
                    f"Label key {label_key!r} is missing for {len(missing)} examples"
                )
        shapes = []
        for path in metadata_paths:
            with np.load(directory / f"{path.stem}.npz") as arrays:
                shapes.append(arrays["hidden_states"].shape)
        max_tokens = max(shape[0] for shape in shapes)
        n_layers, hidden_dim = shapes[0][1:]
        hidden = np.zeros((len(metadata_paths), max_tokens, n_layers, hidden_dim), dtype=np.float16)
        mask = np.zeros((len(metadata_paths), max_tokens), dtype=bool)
        logprobs = np.zeros((len(metadata_paths), max_tokens), dtype=np.float32)
        entropies = np.zeros((len(metadata_paths), max_tokens), dtype=np.float32)
        for index, path in enumerate(metadata_paths):
            run = self.read(run_id, path.stem)
            token_count = run.hidden_states.shape[0]
            if run.hidden_states.shape[1:] != (n_layers, hidden_dim):
                raise ValueError("All runs must have the same layer and hidden dimensions")
            hidden[index, :token_count] = run.hidden_states
            mask[index, :token_count] = True
            logprobs[index, :token_count] = run.token_logprobs
            entropies[index, :token_count] = run.token_entropies
        return TrajectoryBatch(
            example_ids=tuple(str(item["example_id"]) for item in metadata),
            labels=np.asarray(
                [
                    item["label"]
                    if label_key is None or label_key == "exact_error"
                    else item["provenance"][label_key]
                    for item in metadata
                ],
                dtype=np.int64,
            ),
            splits=np.asarray([item["split"] for item in metadata]),
            hidden_states=hidden,
            token_mask=mask,
            token_logprobs=logprobs,
            token_entropies=entropies,
            provenance=tuple(dict(item.get("provenance", {})) for item in metadata),
        )


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError("Identifier must contain at least one safe character")
    return safe
