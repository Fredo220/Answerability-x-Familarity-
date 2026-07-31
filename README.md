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

## Why This Matters

A model may confuse **recognizing an entity** with **having enough evidence to
answer a question about it**. This distinction matters for hallucination
detection, uncertainty calibration, and reliable abstention.

This study is deliberately scoped so it can be completed reproducibly with
limited compute. It serves as a concrete demonstration of controlled
hypothesis testing, scientific integrity, and research execution. I plan to
complete it independently; with Fellowship mentorship and compute, I would
pursue a more ambitious research question aligned with Anthropic's priorities.

## Research Approach

1. **Controlled behavioral test — (Our contribution**)
   We use a 2×2 design that independently varies entity familiarity and
   answerability. Matched prompts and synthetic facts test whether familiarity
   increases answer attempts when the required evidence is absent.

2. **Population-level validation — (Our contribution**)
   We evaluate the interaction across multiple entities and domains rather
   than drawing conclusions from individual prompts.

3. **Internal activation analysis — (Combined approach**)
   We test whether model activations predict the behavioral effect before the
   answer is generated.

4. **Circuit tracing — (Anthropic methodology**)
   For preregistered representative cases, we use Anthropic-style Replacement
   Models, Cross-Layer Transcoders, and Attribution Graphs to generate
   hypotheses about the internal mechanism.

5. **Causal validation — (Combined contribution**)
   We validate graph hypotheses through fidelity checks, controlled
   interventions, and appropriate null controls in the original model.

Our primary contribution is the controlled Familiarity × Answerability
experiment and its systematic dataset. Anthropic's tools provide the
mechanistic framework used to investigate why the measured behavior occurs.

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

**Status as of 2026-08-01: R11 instrument audit failed; the hypothesis remains
untested.**

- The original Source-v5 corpus was `not_evaluable` because too few entities
  passed its frozen familiarity screen.
- R9 and R10 also failed their registered instrument-readiness gates. Their
  negative results remain preserved.
- R11 used a larger candidate pool and five preregistered relation questions
  per domain. Development data selected three relations deterministically.
- The strict-score yield was sufficient, but an independent human audit found
  14 disallowed ambiguity, granularity, or alias defects in its 24-item packet.
- R11 is therefore `not_evaluable`. Construction validation and confirmatory
  endpoints were not opened, so this outcome is not evidence for or against
  the main hypothesis.
- The familiarity rule remains unchanged: an entity must be answered
  correctly on at least two of three selected questions.

The committed R11 source contains:

| Split | Entities per domain | Relations per entity |
|---|---:|---:|
| Open instrument development | 32 | 5 |
| Construction validation | 16 | 5 |

The four registered domains are creative works, business enterprises, people,
and countries. Development and validation share no entity IDs.

The full R11 outcome and auditable artifacts are published in
[the R11 outcome report](docs/results/source_v6_r11_instrument_development_outcome.md).

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
