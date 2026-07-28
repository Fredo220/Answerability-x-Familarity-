# Familiarity vs. Answerability in Language Models

Does a language model answer more readily when it recognizes an entity, even
when the information needed to answer is missing?

This project tests that question with a preregistered behavioral experiment
and a small mechanistic-interpretability follow-up on
`google/gemma-2-2b-it`.

## The Idea

The experiment separates two factors:

- **Familiarity:** Does the model recognize the target entity?
- **Answerability:** Does the prompt contain the evidence required to answer?

For example, the model may see either a familiar name such as an established
organization or a matched synthetic name. The prompt then either provides a
fictional archive code for that entity or leaves the code unavailable.

The primary question is:

> Does familiarity selectively increase answer attempts when the required
> evidence is absent?

This distinction matters because confidently completing a pattern is not the
same as having evidence for the requested answer.

## Experiment

The planned Fellowship artifact has two gated stages:

1. **F1: behavioral interaction**
   - Cross familiar and synthetic entities with answerable and unanswerable
     prompts.
   - Measure whether familiarity changes answer attempts specifically when
     evidence is absent.
2. **F2A: mechanistic pilot**
   - Extract internal activations before the answer is produced.
   - Test whether they add held-out predictive information beyond surface
     features and output confidence.

F2A runs only after a valid F1 corpus exists. Activation interchange and
attribution graphs are optional follow-ups; they cannot rescue a failed
behavioral result.

## Current Status

**Status as of 2026-07-28: instrument development, not hypothesis evidence.**

- The original Source-v5 corpus was `not_evaluable` because too few entities
  passed its frozen familiarity screen.
- R9 and R10 also failed their registered instrument-readiness gates. Their
  negative results remain preserved.
- R11 is the current repair. It uses a larger candidate pool and five
  preregistered relation questions per domain.
- Development data select three relations deterministically. Those relations
  are then evaluated once on fresh, entity-disjoint validation data.
- The familiarity rule remains unchanged: an entity must be answered
  correctly on at least two of three selected questions.

The committed R11 source contains:

| Split | Entities per domain | Relations per entity |
|---|---:|---:|
| Open instrument development | 32 | 5 |
| Construction validation | 16 | 5 |

The four registered domains are creative works, business enterprises, people,
and countries. Development and validation share no entity IDs.

The source has passed an AI-assisted development audit. An independent human
audit is still required before confirmatory reporting.

## What R11 Can Show

A passing R11 gate shows only that the familiarity-screening instrument is
feasible and stable on fresh entities.

It does **not** show that:

- familiarity causes hallucinations;
- the main Familiarity-by-Answerability hypothesis is true;
- Gemma represents human-like belief, knowledge, or intuition;
- the result generalizes to other tasks or models.

Only the later frozen F1 analysis can provide evidence for the behavioral
interaction. Null and `not_evaluable` outcomes are valid reportable results.

## Reproduce Locally

Use Python 3.12:

```bash
git clone https://github.com/Fredo220/Answerability-x-Familarity-.git
cd Answerability-x-Familarity-

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/fa-core.lock
python -m pip install --no-deps -e .
python -m pytest -q tests/test_fa_r11_*.py
```

The 8 GB local machine is suitable for tests, audits, analysis, and reporting.
The pinned Gemma generation requires a Colab GPU in the current setup.

## Run the Current R11 Screen

Add `HF_TOKEN` through Colab Secrets, check out a clean repository revision,
and run:

```bash
COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"

PYTHONPATH=src python tools/run_fa_r11_screening.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --source-root data/fa/development_source_v6_r11 \
  --split instrument_development \
  --output-root /content/fa-r11-artifacts \
  --pre-model-semantic-audit \
    data/fa/development_source_v6_r11/pre_model_semantic_audit_v1.json \
  --batch-size 16 \
  --git-commit "$COMMIT"
```

The command is resumable and verifies source, audit, model, tokenizer, parser,
and commit identities before reusing completed batches. Continue with the
[R11 runbook](docs/fa_source_v6_r11_runbook.md) to freeze the selected
relations and perform the one-shot validation.

## Research Integrity

- The Gemma model, tokenizer, chat template, parser, seeds, and thresholds are
  pinned and hash-bound.
- Development, validation, and confirmatory entities are separated.
- Protected endpoints open once and fail closed on incomplete evidence.
- Instrument changes are documented before new model outputs are inspected.
- Negative runs remain part of the public record.
- No access token should ever be pasted into code or committed.

## Key Documents

Start here:

- [R11 runbook](docs/fa_source_v6_r11_runbook.md)
- [R11 instrument amendment](docs/amendments/2026-07-28-fa-source-v6-r11-surplus-instrument-development.md)
- [Preregistration](docs/familiarity_answerability_preregistration.md)
- [Claim boundaries](docs/familiarity_answerability_claims.md)

Supporting protocols:

- [Human naturalness protocol](docs/fa_naturalness_rating_protocol.md)
- [Main execution runbook](docs/familiarity_answerability_runbook.md)
- [Implementation plan](docs/superpowers/plans/2026-07-22-familiarity-answerability-implementation.md)

Historical corrections and failed instrument attempts are retained under
[`docs/amendments`](docs/amendments) and [`docs/results`](docs/results).
