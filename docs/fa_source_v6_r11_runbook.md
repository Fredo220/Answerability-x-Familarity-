# R11 Surplus Instrument Runbook

R11 is the shortest defensible path from the failed R10 instrument to a fresh
validation. It reuses the existing Gemma and Colab stack.

## 1. Before the run

1. Add the canonical R10 archive to the handoff and verify all member hashes.
2. Keep the checkout clean and record `git rev-parse HEAD`.
3. Build 32 development and 16 validation candidates per domain from unseen
   QIDs:

```bash
PYTHONPATH=src python tools/build_fa_r11_source.py \
  --config configs/fa_source_v6_r11_development.json \
  --output-dir /content/fa-r11-source \
  --query-limit 1000 \
  --retrieval-date 2026-07-28 \
  --exclusion-manifest /content/fa-r11-predecessor-exclusions.json
```

The exclusion manifest must enumerate every Source-v5 and R8-R10 QID and hash
each parent candidate manifest:

```json
{
  "schema_version": 1,
  "kind": "fa_r11_predecessor_exclusions",
  "parents": [
    {"name": "source-v5", "sha256": "<64 lowercase hex>"}
  ],
  "qids": ["Q1", "Q2"],
  "qids_sha256": "<canonical sorted-QID-list SHA-256>"
}
```

The builder rejects an absent or malformed predecessor manifest and fails
unless every domain has the registered surplus after exclusions.

4. The builder materializes one row per available registered relation with
these fields:

```json
{
  "split": "instrument_development",
  "domain": "person",
  "entity_id": "r11-instrument_development-person-q42",
  "qid": "Q42",
  "relation_id": "P569",
  "prompt": "In which year was Douglas Adams born? Answer with only the year.",
  "accepted_aliases": ["1952", "1952."],
  "is_correct": true
}
```

For R11, `organization` is the registered business-enterprise subpopulation
and `place` is the registered country subpopulation. Answer keys use primary
English labels and prefer current, non-ended Wikidata statements. These
restrictions were frozen after source-only feasibility and semantic audits,
before any R11 model completion was generated.

`is_correct` is added only by the unchanged exact-answer scorer after Gemma
generation. Before generation, independently audit every candidate fact,
accepted alias set, and prompt. Store the signed result as:

```json
{
  "schema_version": 1,
  "kind": "fa_r11_pre_model_semantic_audit",
  "status": "passed",
  "blocker_count": 0,
  "rows_sha256": "<screening_prompts_v1.jsonl SHA-256>",
  "items_file": "pre_model_semantic_audit_items_v1.jsonl",
  "items_sha256": "<audit-items SHA-256>",
  "item_count": "<exact prompt count>"
}
```

The item file must contain one row per entity-relation prompt, separate
nonempty structural and semantic auditor IDs, and passing structural and
semantic decisions. The two IDs must differ. The screening command rejects
missing coverage, failed rows, shared auditor IDs, or hash mismatches.

## 2. Screen and select on open development

```bash
COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"

PYTHONPATH=src python tools/run_fa_r11_screening.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --source-root /content/fa-r11-source \
  --split instrument_development \
  --output-root /content/fa-r11-artifacts \
  --pre-model-semantic-audit \
    /content/fa-r11-source/pre_model_semantic_audit_v1.json \
  --batch-size 16 \
  --git-commit "$COMMIT"
```

Re-run this exact command after interruption; completed immutable batches are
reused.

```bash
PYTHONPATH=src python tools/select_fa_r11_instrument.py select \
  --items /content/fa-r11-artifacts/instrument_development/<execution-identity-sha256>/screening_items.jsonl \
  --config configs/fa_source_v6_r11_development.json \
  --source-manifest /content/fa-r11-source/source_manifest_v1.json \
  --execution-identity \
    /content/fa-r11-artifacts/instrument_development/<execution-identity-sha256>/execution_identity.json \
  --git-commit "$COMMIT" \
  --output /content/fa-r11-freeze/relation_selection_v1.json
```

Retain the complete combination audit. Do not inspect validation before this
artifact is sealed.

## 3. Human audit and freeze

Run the existing two-rater packet and independent adjudication process on the
development output. Freeze the selected relations, prompts, aliases, parser,
model/config hashes, development item hash, and selection hash.

## 4. Screen and validate once

```bash
PYTHONPATH=src python tools/run_fa_r11_screening.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --source-root /content/fa-r11-source \
  --split construction_validation \
  --output-root /content/fa-r11-artifacts \
  --pre-model-semantic-audit \
    /content/fa-r11-source/pre_model_semantic_audit_v1.json \
  --selection /content/fa-r11-freeze/relation_selection_v1.json \
  --batch-size 16 \
  --git-commit "$COMMIT"
```

Then apply the frozen gate:

```bash
PYTHONPATH=src python tools/select_fa_r11_instrument.py validate \
  --items /content/fa-r11-artifacts/construction_validation/<execution-identity-sha256>/screening_items.jsonl \
  --selection /content/fa-r11-freeze/relation_selection_v1.json \
  --output /content/fa-r11-freeze/construction_validation_gate_v1.json
```

If the gate fails, stop. If it passes, construct the new confirmatory corpus
from entirely new QIDs and continue with the existing human audit, F1, and F2A
pipeline.

## 5. Local verification

```bash
.venv/bin/python -m pytest -q tests/test_fa_r11_*.py
.venv/bin/python -m pytest -q
```
