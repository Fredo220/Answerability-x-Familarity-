# Familiarity vs. Answerability in Language Models

Does a language model answer more readily after receiving unrelated contextual
exposure to a target, even when the information needed to answer is missing?

This project investigates whether language models internally distinguish
between familiarity with a target and having enough evidence to answer a
question about it. It combines a controlled behavioral experiment on Gemma 2
2B with an exploratory representation-level analysis of the model's internal
activations.

## The Idea

The completed experiment separated two factors while keeping the target string
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

This study was deliberately scoped for reproducible execution with limited
compute. It demonstrates controlled hypothesis testing, scientific integrity,
and research execution. The completed pilot also exposes concrete design
limitations for a larger follow-up.

## Research Approach

1. **Controlled behavioral test (our contribution)**
   We use a 2×2 Same-String design that independently varies contextual
   exposure and answerability. The target string and task remain fixed.

2. **Population-level validation (our contribution)**
   We evaluate the interaction across multiple entities and domains rather
   than drawing conclusions from individual prompts.

3. **Internal activation analysis (exploratory follow-up complete)**
   Its central hypothesis is that the model may represent exposure and
   answerability differently internally, even though that separation did not
   appear as the hypothesized failure behavior in the small behavioral pilot.
   This is a hypothesis to be tested, not a result of the completed study.

4. **Local causal validation (future work outside v2)**
   A future registered study could test matched activation replacement against
   reverse, shuffled, orthogonal, and norm-matched controls. Study v2 contained
   no intervention split and supports no causal claim.

Our primary contribution is the controlled Same-String exposure ×
answerability experiment. Attribution graphs are an unregistered future
possibility, not part of the completed confirmatory pilot.

## Experiment

The registered plan had two gated stages:

1. **Primary behavior study**
   - Cross high and low contextual exposure with present and absent evidence.
   - Estimate the registered difference-in-differences in answer attempts.
2. **Gated mechanistic pilot**
   - Extract internal activations before the answer is produced.
   - Test whether they add held-out predictive information beyond surface
     features and output confidence.

The confirmatory mechanistic claim was permitted only if the behavioral gate
passed. The v2 amendment separately permits an exploratory representation-only
analysis, but no activation result can rescue the failed behavioral gate.

## Current Status

**Status as of 2026-08-02: the protected behavioral endpoint for the
Same-String Balanced Pilot v2 is complete, evaluable, and `not_supported`.**

- All 128 generations completed with 100% format validity.
- The registered exposure-by-answerability interaction was `-0.09375`, with a
  crossed-bootstrap 95% interval of `[-0.4000, 0.1818]`.
- The answerability manipulation worked behaviorally: target-bound attempt
  rates were `0.9063` under high exposure and `0.7500` under low exposure;
  code-absent attempt rates were `0.0625` and `0.0000`, respectively.
- The interaction and capability-preservation gates failed. The registered
  gated mechanistic pilot was therefore not run. A separately frozen
  exploratory representation-only pilot was completed later.
- A required pre-outcome power/MDE audit is absent from the verified snapshot.
  This disclosed protocol deviation limits v2 to an imprecise pilot result.
- The endpoint was opened once and is permanently closed. The downloaded
  content-addressed snapshot was restored and verified locally.
- R11 and Same-String v1 remain immutable `not_evaluable` instrument records.

See the [behavioral result report](docs/results/same_string_feasibility_v2_behavior_result.md)
and its [machine-readable record](docs/results/same_string_feasibility_v2_behavior_result.json).
The separate [representation-only result](docs/results/same_string_representation_pilot_v2.md)
reports the exploratory activation analysis without changing that decision.

### Representation-Only Pilot

The exploratory pilot found a simple position-dependent pattern:

- Contextual exposure was detectable internally after the target introduction.
- Answerability was at chance there and became detectable only after the model
  received the evidence-bearing question.
- Prompt surface features already predicted answerability perfectly. The pilot
  therefore does not show that internal activations add unique answerability
  information beyond the prompt itself.

This result used only four held-out units. It is exploratory, non-causal, and
does not establish metacognition or general hallucination detection.

## How to Interpret the Result

In this task, Gemma behaved more selectively than the original risk hypothesis
predicted. Answerability dominated the manipulation: the model usually returned
the archive code when the prompt supplied it and usually answered `UNKNOWN`
when the code was absent. Unrelated exposure did not produce the registered
positive increase in unsupported answer attempts.

This is a limited but directionally encouraging result for the tested model and
task. It does not prove that LLMs generally understand answerability, reason
reliably, or avoid hallucinations. The interval is wide, the absent-evidence
cells were near the abstention floor, and only one small checkpoint was tested.

Had the hypothesis been supported, the interpretation would have been a
reliability concern rather than evidence of a desirable capability: it would
suggest that merely making a target feel more familiar can make a model answer
without the information required to do so.

## Compute-Constrained Scope

This was intentionally built as a low-compute portfolio study. Development,
auditing, and analysis ran on a computer with 8 GB RAM; protected model
generation used only free-tier Google Colab capacity. Those constraints made a
small, reproducible behavioral pilot more responsible than an under-resourced
claim about frontier-model cognition.

The project is therefore meant to demonstrate research practice: isolate a
confound, preregister a decision rule, preserve failed instruments, open a
protected endpoint once, publish a negative result, and make the evidence
auditable. It is not presented as the largest experiment the question merits.

With greater research-compute access, a substantially more ambitious study
could test several model families and scales, increase the number of unseen
units and templates, run multiple registered seeds, and add properly powered
activation and causal-intervention studies.

## What a Stronger Follow-Up Should Test

A new study should preserve v2 unchanged and register a new endpoint before
generating outputs:

1. Calibrate prompts on open development data to avoid near-zero unsupported
   attempt rates and near-one answerable attempt rates.
2. Publish a power and minimum-detectable-effect analysis before freezing the
   sample size.
3. Add an independent manipulation check showing that high exposure actually
   changed the intended familiarity proxy.
4. Replicate the same 2x2 estimand on at least one larger model and one
   independent model family.
5. Use held-out activation probes to test whether familiarity and
   answerability are internally separable before the answer is produced.
6. If the behavioral and probe gates pass, use controlled activation patching
   with shuffled, reversed, orthogonal, and norm-matched controls to test local
   causal influence.

The decisive next question is not whether v2 can be made positive. It is
whether a better-powered, multi-model design can distinguish a genuine absence
of the effect from floor behavior, limited manipulation strength, and
model-specific behavior.

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

## Reproduce the Same-String Pilot

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
- [Balanced feasibility v2 behavioral result](docs/results/same_string_feasibility_v2_behavior_result.md)
- [Balanced feasibility v2 machine-readable result](docs/results/same_string_feasibility_v2_behavior_result.json)
- [Content-addressed v2 evidence snapshot](release/familiarity_answerability/README.md)

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
