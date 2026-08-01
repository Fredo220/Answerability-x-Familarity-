# Familiarity vs. Answerability in Language Models

Does a language model answer more readily after receiving unrelated contextual
exposure to a target, even when the information needed to answer is missing?

This repository contains a controlled behavioral experiment and a gated small
mechanistic-interpretability follow-up on `google/gemma-2-2b-it`. The active
study is the **Same-String Primary Study**. It tests contextual familiarization
without depending on the failed real-entity R11 screening instrument.

## The Idea

The active experiment separates two factors while keeping the target string
identical within every four-prompt unit:

- **Contextual exposure:** Has the prompt introduced several unrelated facts
  about the target string?
- **Answerability:** Does the prompt contain the evidence required to answer?

The prompt either supplies a fictional archive code for the target or leaves
that code unavailable. High-exposure prompts first provide unrelated facts
about the same target; low-exposure controls assign matched facts elsewhere.

The primary question is:

> Does controlled contextual familiarization selectively increase answer
> attempts when the required evidence is absent?

This distinction matters because confidently completing a pattern is not the
same as having evidence for the requested answer.

## Why This Matters

A model may confuse **having seen more context about a target** with **having
enough evidence to answer a question about it**. This distinction matters for
hallucination detection, uncertainty calibration, and reliable abstention.

This study is deliberately scoped so it can be completed reproducibly with
limited compute. It serves as a concrete demonstration of controlled
hypothesis testing, scientific integrity, and research execution. I plan to
complete it independently; with Fellowship mentorship and compute, I would
pursue a more ambitious research question aligned with Anthropic's priorities.

## Research Approach

1. **Controlled behavioral test (our contribution)**
   We use a 2×2 Same-String design that independently varies contextual
   exposure and answerability. The target string and task remain fixed.

2. **Population-level validation (our contribution)**
   We evaluate the interaction across multiple entities and domains rather
   than drawing conclusions from individual prompts.

3. **Internal activation analysis (gated follow-up)**
   We test whether held-out activations decode exposure and answerability and
   predict unsupported answer attempts beyond registered controls.

4. **Local causal validation (gated follow-up)**
   If the behavioral and probe gates pass, we test matched activation
   replacement against reverse, shuffled, orthogonal, and norm-matched
   controls in the original model.

Our primary contribution is the controlled Same-String exposure ×
answerability experiment. Attribution graphs are an unregistered future
possibility, not part of the active confirmatory study.

## Active Experiment

The planned Fellowship artifact has two gated stages:

1. **Primary behavior study**
   - Cross high and low contextual exposure with present and absent evidence.
   - Estimate the registered difference-in-differences in answer attempts.
2. **Gated mechanistic pilot**
   - Extract internal activations before the answer is produced.
   - Test whether they add held-out predictive information beyond surface
     features and output confidence.

The mechanistic pilot runs only after the behavioral endpoint is complete.
Activation interchange and attribution graphs are optional follow-ups; they
cannot rescue a failed behavioral result.

## Current Status

**Status as of 2026-08-01: the Same-String implementation is pre-outcome. It is
not yet an empirical result.**

- The Same-String design, amendment, direct matching path, sealed-manifest
  construction, estimator, bootstrap, and behavior gate are implemented.
- The model-independent preflight passed all ten prompt audits. Two sealed,
  blinded naturalness packets now await independent human ratings; the runtime
  smoke and one-shot protected Gemma behavior run remain closed.
- R11 remains immutable and `not_evaluable`; it is preserved as a negative
  instrument-development result and is not repaired by this study.

### Preserved R11 record

- The original Source-v5 corpus was `not_evaluable` because too few entities
  passed its frozen familiarity screen.
- R9 and R10 also failed their registered instrument-readiness gates. These
  feasibility outcomes remain preserved.
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

The original Same-String v1 corpus is also `not_evaluable`: its ratings
compiled, but only 73 of 192 pairs passed the frozen naturalness rule and the
registered split/domain quotas could not be filled. The separately registered
balanced feasibility pilot v2 can provide limited evidence for the
contextual-exposure interaction. It cannot reinterpret R11 or v1. Null and
`not_evaluable` outcomes remain valid reportable results.

## Reproduce Locally

Use Python 3.12:

```bash
git clone https://github.com/Fredo220/Answerability-x-Familarity-.git
cd Answerability-x-Familarity-

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/fa-core.lock
python -m pip install --no-deps -e .
python -m pytest -q \
  tests/test_fa_data.py \
  tests/test_fa_scoring.py \
  tests/test_fa_cli.py \
  tests/test_fa_notebooks.py \
  -k "same_string or notebook"
```

The 8 GB local machine is suitable for tests, audits, analysis, and reporting.
The pinned Gemma generation requires a Colab GPU in the current setup.

## Run the Active Same-String Pilot

Read the frozen
[v2 amendment](docs/amendments/2026-08-01-fa-same-string-balanced-pilot-v2.md)
and open the thin
[v2 Colab launcher](notebooks/fa_same_string_feasibility_v2_colab.ipynb). The
notebook first reproduces the immutable v1 ratings, then deterministically
allocates 52 accepted pairs before any model compute. It uses only repository
CLI commands, a pinned commit, Colab Secrets, and a persistent Drive
checkpoint.

## Research Integrity

- The Gemma model, tokenizer, chat template, parser, seeds, and thresholds are
  pinned and hash-bound.
- Development, validation, and confirmatory entities are separated.
- Protected endpoints open once and fail closed on incomplete evidence.
- Instrument changes are documented before new model outputs are inspected.
- Negative runs remain part of the public record.
- No access token should ever be pasted into code or committed.

## Key Documents

Active pilot:

- [Balanced feasibility v2 amendment](docs/amendments/2026-08-01-fa-same-string-balanced-pilot-v2.md)
- [Balanced feasibility v2 Colab](notebooks/fa_same_string_feasibility_v2_colab.ipynb)

Preserved Same-String v1 record:

- [Same-String primary runbook](docs/fa_same_string_primary_runbook.md)
- [Same-String design](docs/superpowers/specs/2026-08-01-same-string-primary-hybrid-design.md)
- [Same-String amendment](docs/amendments/2026-08-01-fa-same-string-primary.md)
- [Same-String implementation plan](docs/superpowers/plans/2026-08-01-same-string-primary-hybrid-implementation.md)
- [Same-String preflight result](docs/results/same_string_primary_preflight.md)

Preserved R11 record:

- [R11 runbook](docs/fa_source_v6_r11_runbook.md)
- [R11 instrument amendment](docs/amendments/2026-07-28-fa-source-v6-r11-surplus-instrument-development.md)
- [R11 negative result](docs/results/source_v6_r11_instrument_development_outcome.md)
- [Preregistration](docs/familiarity_answerability_preregistration.md)
- [Claim boundaries](docs/familiarity_answerability_claims.md)

Supporting protocols:

- [Human naturalness protocol](docs/fa_naturalness_rating_protocol.md)
- [Main execution runbook](docs/familiarity_answerability_runbook.md)
- [Implementation plan](docs/superpowers/plans/2026-07-22-familiarity-answerability-implementation.md)

Historical corrections and failed instrument attempts are retained under
[`docs/amendments`](docs/amendments) and [`docs/results`](docs/results).
