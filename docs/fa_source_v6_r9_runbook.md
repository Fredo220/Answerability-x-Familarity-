# Source-v6 R9 Development Runbook

R9 is an audit-qualified development corpus. It can establish instrument
readiness only. It cannot confirm Familiarity-by-Answerability, and every R8
and R9 QID remains excluded from future confirmatory corpora.

## 1. Git-bind derivation rules

Commit the R9 amendment, structured correction manifest, selection code,
tests, success criteria, and this runbook before deriving any R9 manifest.
Record the exact commit.

## 2. Derive once

From that clean commit:

```bash
PYTHONPATH=src python tools/build_fa_development_r9.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --r8-root data/fa/development_source_v6_r8 \
  --corrections data/fa/development_source_v6_r9/alias_corrections_v1.json \
  --output-dir data/fa/development_source_v6_r9
```

Any count, collision, pseudonym, lineage, or hash failure blocks R9. Changing
the correction table, seed, selection, split logic, or thresholds requires
R10.

The derivation also writes
`confirmatory_excluded_candidates_v1.json`, a list-shaped manifest accepted by
the confirmatory source builder. It includes every R8 candidate QID plus all
earlier registered exclusions, not only the 96 selected R9 candidates.

Future confirmatory construction must pass both
`source_v7_exclusions_v1.json` and `r9_derivation_manifest_v1.json` through
`--development-exclusions` and `--development-derivation`.

## 3. Audit before model execution

An independent code path must replay R9 lineage, selection decisions, source
hashes, balance, and manifest hashes. A different semantic auditor must inspect
all 96 candidates and all 288 question/ground-truth/accepted-surface triples.
The semantic manifest must hash-bind one JSONL row per question and complete
candidate and question coverage.

## 4. Git-bind the passing corpus

Commit and push the R9 manifests, derivation manifest, structural audit, and
zero-blocker semantic audit. Run Gemma only from a clean detached checkout of
that exact commit.

## 5. Colab screening

```bash
COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"

PYTHONPATH=src /content/fa-venv/bin/python \
  tools/run_fa_development_screening.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --source-root data/fa/development_source_v6_r9 \
  --split instrument_development \
  --output-root /content/fa-r9-artifacts \
  --checkpoint-root /content/drive/MyDrive/fa-r9-checkpoints \
  --batch-size 16 \
  --success-criteria configs/fa_source_v6_r9_success_criteria.json \
  --pre-model-structural-audit \
    data/fa/development_source_v6_r9/structural_provenance_audit_v1.json \
  --pre-model-semantic-audit \
    data/fa/development_source_v6_r9/pre_model_semantic_audit_v1.json \
  --git-commit "$COMMIT"
```

If the preregistered gate fails, report R9 as `not_evaluable`. If it passes,
complete the two-human-rater error audit and freeze the instrument before
opening `construction_validation` once.
