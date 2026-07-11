from __future__ import annotations

import difflib
import hashlib
import json
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi


@dataclass(frozen=True)
class FactualTriple:
    example_id: str
    subject: str
    relation: str
    object: str
    source_url: str
    distractors: tuple[str, ...]
    distractor_answers: tuple[str, ...]
    split: str

    @property
    def prompt(self) -> str:
        facts = [f"{self.subject} | {self.relation} | {self.object}"]
        facts.extend(self.distractors)
        return (
            "Use only the source-documented fact table below. "
            "Answer with the object name only.\n"
            "Subject | Relation | Object\n"
            + "\n".join(facts)
            + f"\nQuestion: For {self.subject}, what is the object for '{self.relation}'?\nAnswer:"
        )


def load_documented_triples(path: str | Path, *, limit: int = 200) -> list[FactualTriple]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) < limit:
        raise ValueError(f"Expected at least {limit} source-documented triples, found {len(rows)}")
    triples = []
    for line_number, row in enumerate(rows[:limit], start=1):
        required = {"id", "subject", "relation", "object", "source_url"}
        if not required.issubset(row):
            raise ValueError(f"Line {line_number} lacks required source-documented fields")
        if not str(row["source_url"]).startswith(("https://", "http://")):
            raise ValueError(f"Line {line_number} has an invalid source_url")
        split = "train" if line_number <= int(limit * 0.6) else "val" if line_number <= int(limit * 0.8) else "test"
        triples.append(
            FactualTriple(
                example_id=str(row["id"]),
                subject=str(row["subject"]),
                relation=str(row["relation"]),
                object=str(row["object"]),
                source_url=str(row["source_url"]),
                distractors=tuple(row.get("distractors", ())),
                distractor_answers=tuple(row.get("distractor_answers", ())),
                split=split,
            )
        )
    return triples


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_BIRTHPLACE_QUERY = """
SELECT DISTINCT ?subject ?subjectLabel ?object ?objectLabel WHERE {
  ?subject wdt:P19 ?object;
           rdfs:label ?subjectLabel.
  ?object rdfs:label ?objectLabel.
  FILTER(isIRI(?object))
  FILTER(LANG(?subjectLabel) = \"en\")
  FILTER(LANG(?objectLabel) = \"en\")
}
LIMIT 600
""".strip()


def fetch_wikidata_transfer_file(
    output_path: str | Path,
    *,
    limit: int = 200,
    endpoint: str = WIKIDATA_ENDPOINT,
) -> dict[str, Any]:
    """Fetch a deterministic, source-linked binding transfer set from Wikidata."""
    if limit < 1:
        raise ValueError("limit must be positive")
    request = Request(
        endpoint + "?" + urlencode({"query": WIKIDATA_BIRTHPLACE_QUERY, "format": "json"}),
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": "feature-dynamics-study/0.1 (public research prototype)",
        },
    )
    tls_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(  # noqa: S310 - fixed public endpoint with verified TLS
        request, timeout=90, context=tls_context
    ) as response:
        payload = json.load(response)
    rows = build_wikidata_transfer_rows(payload["results"]["bindings"], limit=limit)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "query": WIKIDATA_BIRTHPLACE_QUERY,
        "relation": "place of birth",
        "selection": "unique labels with four nearest-name, different-answer distractors",
        "count": len(rows),
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_wikidata_transfer_rows(bindings: list[dict[str, Any]], *, limit: int) -> list[dict]:
    candidates: list[dict[str, str]] = []
    seen_subjects: set[str] = set()
    for binding in bindings:
        subject_uri = str(binding.get("subject", {}).get("value", ""))
        subject = str(binding.get("subjectLabel", {}).get("value", "")).strip()
        object_name = str(binding.get("objectLabel", {}).get("value", "")).strip()
        if not subject_uri.startswith("http://www.wikidata.org/entity/Q"):
            continue
        if not subject or not object_name or subject.startswith("Q") or object_name.startswith("Q"):
            continue
        normalized_subject = " ".join(subject.casefold().split())
        if normalized_subject in seen_subjects:
            continue
        seen_subjects.add(normalized_subject)
        candidates.append(
            {
                "subject": subject,
                "object": object_name,
                "source_url": subject_uri.replace("http://", "https://"),
            }
        )

    candidates.sort(key=lambda candidate: candidate["source_url"])

    rows: list[dict] = []
    for target in candidates:
        alternatives = [candidate for candidate in candidates if candidate["object"] != target["object"]]
        alternatives.sort(
            key=lambda candidate: (
                -difflib.SequenceMatcher(
                    None, target["subject"].casefold(), candidate["subject"].casefold()
                ).ratio(),
                candidate["subject"],
            )
        )
        distractors = alternatives[:4]
        if len(distractors) < 4:
            continue
        rows.append(
            {
                "id": f"wikidata-birthplace-{len(rows):03d}",
                "subject": target["subject"],
                "relation": "place of birth",
                "object": target["object"],
                "source_url": target["source_url"],
                "distractors": [
                    f'{candidate["subject"]} | place of birth | {candidate["object"]}'
                    for candidate in distractors
                ],
                "distractor_answers": [candidate["object"] for candidate in distractors],
            }
        )
        if len(rows) == limit:
            return rows
    raise ValueError(f"Wikidata response yielded only {len(rows)} valid rows; expected {limit}")
