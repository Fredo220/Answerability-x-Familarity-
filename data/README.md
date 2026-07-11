# Data Layout

- `processed/concept_mixing.jsonl`: deterministic controlled dataset generated with seed 42.
- `processed/concept_mixing.jsonl.manifest.json`: generator schema, factor design, split counts, and checksum.
- `external/jailbreakbench/study.jsonl`: exactly 100 official harmful rows with frozen published artifacts and 100 matched benign rows; every pair shares an explicit `pair_id`.
- `external/real_transfer.jsonl`: 200 source-documented factual triples with explicit distractor answers.
- `external/`: ignored because upstream licenses and provenance remain authoritative.

Regenerate the controlled dataset with:

```bash
feature-dynamics generate-concept-data --total 1200 --seed 42
```
