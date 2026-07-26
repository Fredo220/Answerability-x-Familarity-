# Source-v6 R10 Colab Runbook

R10 is a one-shot, open instrument follow-up. It reuses the unchanged,
previously unopened R9 `construction_validation` split. It cannot test the
Familiarity-by-Answerability hypothesis.

## Prepare the Colab checkout

For a fresh runtime:

```bash
!git clone --branch main \
  https://github.com/Fredo220/Answerability-x-Familarity-.git \
  /content/Answerability-x-Familarity-
```

For an existing checkout:

```bash
%cd /content/Answerability-x-Familarity-
!git fetch origin
!git checkout main
!git pull --ff-only origin main
```

Do not re-clone the repository or recreate the model runtime when this checkout
already exists. Record the exact clean commit checked out for the run.

## Execute once

```bash
COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"

PYTHONPATH=src /content/fa-venv/bin/python \
  tools/run_fa_development_screening.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --source-root data/fa/development_source_v6_r9 \
  --split construction_validation \
  --output-root /content/fa-r10-artifacts \
  --checkpoint-root /content/fa-r10-checkpoints \
  --batch-size 16 \
  --success-criteria configs/fa_source_v6_r10_success_criteria.json \
  --pre-model-structural-audit \
    data/fa/development_source_v6_r9/structural_provenance_audit_v1.json \
  --pre-model-semantic-audit \
    data/fa/development_source_v6_r9/pre_model_semantic_audit_v1.json \
  --followup-amendment \
    docs/amendments/2026-07-26-fa-source-v6-r10-heldout-instrument-validation.md \
  --prior-gate \
    docs/results/source_v6_r9_instrument_development_readiness_gate.json \
  --git-commit "$COMMIT"
```

The local checkpoint root is acceptable for this bounded development run.
Copy the final canonical artifacts out of the runtime immediately after
completion and record the runtime deviation if PyTorch differs from the core
lock.

## Decision

- Failed gate: report R10 as another instrument failure and stop.
- Passed gate: generate the registered audit packet for two independent human
  raters. Do not open protected endpoints before the human audit passes.
