# Familiarity vs. Answerability in Language Models

Can a language model tell the difference between **knowing more about a target**
and **having the evidence needed to answer a specific question**?

This repository contains two completed experiments on the pinned
`google/gemma-2-2b-it` checkpoint:

1. a behavioral test of whether unrelated exposure makes the model answer
   without evidence; and
2. a representation-level test of whether internal activations encode
   answerability on the same controlled task.

## Results at a Glance

| Experiment | Question | Result |
|---|---|---|
| Behavioral pilot | Does unrelated exposure selectively increase unsupported answer attempts? | **Not supported** |
| Representation replication | Do fixed internal activations predict answerability beyond the registered bag-of-ngrams baseline? | **Supported on this controlled task** |

These results are compatible. The model did **not** display the predicted
failure behavior, while its internal activations still contained decodable
information about whether the prompt supplied the answer.

![Summary of the behavioral pilot and representation replication](docs/assets/familiarity_answerability_summary.png)

*Figure 1. Left: descriptive answer-attempt rates from the behavioral pilot;
the registered interaction was not supported. Right: held-out AUROC
improvement of fixed activation probes over the registered TF-IDF baseline.
Inferential details and limitations are reported below.*

## The Idea in Plain English

The experiment separates two factors that are easy to confuse:

- **Familiarity / exposure:** the prompt provides several unrelated facts about
  a fictional target.
- **Answerability:** the prompt either provides the target's archive code or
  states that the code is unavailable.

Every experimental unit contains the same target string in all four
conditions:

| | Evidence present | Evidence absent |
|---|---|---|
| **High exposure** | Unrelated target facts, then its archive code | Unrelated target facts, but no archive code |
| **Low exposure** | Matched context elsewhere, then the target's archive code | Matched context elsewhere, but no archive code |

This 2x2 design asks whether exposure changes behavior differently when the
answer is available versus unavailable. Keeping the target string fixed helps
remove a major confound: a familiar-looking name should not be easier merely
because of its spelling or tokenization.

The main methodological contribution is this Same-String control: exposure
and answerability vary while the target and task remain fixed. Effects are
estimated across complete experimental units rather than inferred from a few
selected prompts.

## What We Found

### 1. Behavioral pilot: the predicted failure did not appear

The preregistered behavioral study evaluated 32 complete units across four
registered domains, or 128 model responses. All responses were complete and
validly formatted.

When the archive code was absent, the model almost always abstained. Unrelated
exposure did not produce the predicted selective increase in unsupported
answer attempts.

| Exposure | Evidence | Answer-attempt rate |
|---|---|---:|
| High | Absent | 6.25% |
| Low | Absent | 0.00% |
| High | Present | 90.63% |
| Low | Present | 75.00% |

The `6.25%` versus `0.00%` difference in the absent-evidence cells corresponds
to only two answer attempts out of 32 versus none out of 32. It is too sparse
to support a reliable exposure effect by itself. The registered analysis uses
the full exposure-by-answerability interaction, whose confidence interval
crosses zero.

The preregistered exposure-by-answerability interaction was `-0.09375`, with a
95% bootstrap interval of `[-0.4000, 0.1818]`. The registered decision was
therefore **`not_supported`**.

This is a limited negative result, not proof that familiarity never affects
LLM behavior. The study used one small model, a synthetic archive-code task,
and a modest sample. The absent-evidence condition was also close to an
abstention floor.

[Read the behavioral report](docs/results/same_string_feasibility_v2_behavior_result.md)
or inspect the
[machine-readable result](docs/results/same_string_feasibility_v2_behavior_result.json).

### 2. Representation replication: answerability was decodable internally

An initial four-unit activation pilot was too small for a reliable
generalization claim. A separately frozen replication therefore used 80
complete 2x2 units, or 320 prompts:

- 32 training units;
- 8 validation units;
- 20 test units with unseen entities; and
- 20 test units from two unseen template families.

Within-factor prompt pairs had identical rendered Gemma-token multisets. The
registered character/token TF-IDF baseline stayed at chance for answerability
on both test splits (`AUROC = 0.50`). Fixed residual-stream activations improved
held-out prediction. In plain language, the activation-based probe could
distinguish whether the prompt contained the required answer much more
reliably than the registered word-statistics baseline, including for unseen
entities and unseen prompt templates.

| Test split | Units | Paired log-loss improvement | 95% CI | AUROC improvement | Permutation p |
|---|---:|---:|---:|---:|---:|
| Unseen entities | 20 | 0.4717 | [0.3593, 0.5474] | 0.4050 | 0.001 |
| Unseen templates | 20 | 0.3029 | [0.0884, 0.4749] | 0.3831 | 0.001 |

A pre-evidence control remained at chance at every fixed layer. This supports
the narrow conclusion that, on this controlled Gemma 2 2B task, internal
activations contain held-out answerability information that the registered
bag-of-ngrams baseline does not recover.

Prompt-final activations containing prompt information is not surprising by
itself. The informative part is the controlled transfer result: the fixed
activation probes generalized beyond the registered bag-of-ngrams comparator
to unseen entities and two unseen template families, while the pre-evidence
control remained at chance. This narrows the plausible explanation beyond
simple token-frequency or template memorization, although it does not rule out
a stronger symbolic or sequence-aware prompt baseline.

The answerability information is still explicitly present in the prompt. A
symbolic parser could solve the task. The result therefore concerns
**internal representation and decodability**, not hidden knowledge or
information absent from the input.

[Read the v3 representation report](release/familiarity_answerability/representation_replication_v3/analysis/result.md)
or inspect its
[machine-readable result](release/familiarity_answerability/representation_replication_v3/analysis/result.json).

## What the Results Do and Do Not Mean

The completed evidence supports two scoped statements:

1. In the small behavioral pilot, unrelated contextual exposure did not cause
   the preregistered increase in unsupported answer attempts.
2. In the larger representation study, fixed activations predicted
   answerability beyond the registered bag-of-ngrams baseline across unseen
   entities and unseen prompt templates.

The project does **not** establish:

- a general mechanism of familiarity or hallucination;
- metacognition, human-like knowledge, or intuition;
- causal influence of the decoded activation patterns;
- information not already present in the prompt;
- generalization to larger models or natural factual questions; or
- a hallucination detector or prevention method.

## Why This Matters for AI Safety

For systems expected to abstain when required evidence is missing, confusing
"this target feels familiar" with "the required evidence is available" would
be a reliability failure. The behavioral pilot did not find that failure in
this setting.

Unsupported but confident answers can undermine human oversight and become a
safety concern in high-stakes settings. Understanding whether models track
evidence availability internally may eventually support better uncertainty
monitoring, but this study does not yet provide such a detector.

The representation result asks a different question: whether answerability is
tracked internally even when the tested failure behavior is absent. That is a
useful first step toward studying how models represent evidence availability,
but causal interventions and broader replications are required before making
stronger claims.

## Research Standards

The project was designed to make negative and positive results equally
auditable:

- hypotheses and decision rules were frozen before protected outcomes opened;
- model revision, prompts, parser, seeds, and thresholds are hash-bound;
- training, validation, and test units are separated;
- protected endpoints open once and fail closed on incomplete evidence;
- failed instruments and negative results remain in the public record; and
- the representation result cannot retroactively rescue the behavioral result.

The originally registered mechanistic pilot was gated on a positive behavioral
result and was therefore not run. The later representation pilot and v3
replication were frozen as separate analyses. They answer a different question
and cannot change the closed behavioral decision.

One protocol deviation is disclosed: the behavioral v2 snapshot lacks its
required pre-outcome power/minimum-detectable-effect artifact. This does not
change the registered endpoint decision, but it limits v2 to an imprecise
pilot rather than a definitive null result.

## Limitations and Next Steps

The most useful follow-up would:

1. repeat the design on larger and independent model families;
2. increase the number of unseen entities and prompt templates;
3. publish a power analysis before opening model outcomes;
4. compare activations with stronger symbolic and sequence-aware baselines;
5. test whether the representation result survives multiple seeds; and
6. use controlled activation replacement with shuffled, reversed, orthogonal,
   and norm-matched controls to test local causal influence.

The current study was intentionally scoped for an 8 GB local machine and
free-tier Colab compute. Local hardware handled tests, audits, analysis, and
reporting; Colab handled pinned Gemma generation and activation extraction.

## Reproduce

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

The local workflow covers tests, audits, analysis, and reporting. Regenerating
Gemma outputs and activations requires a suitable Colab GPU and a Hugging Face
token stored through Colab Secrets. Never paste or commit access tokens.

Regenerate the README figure from the published result JSON files with:

```bash
python tools/plot_readme_summary.py
```

## Key Evidence

| Artifact | Purpose |
|---|---|
| [Behavioral amendment](docs/amendments/2026-08-01-fa-same-string-balanced-pilot-v2.md) | Frozen v2 design and decision rules |
| [Behavioral Colab launcher](notebooks/fa_same_string_feasibility_v2_colab.ipynb) | Thin launcher for the pinned, resumable model run |
| [Behavioral result](docs/results/same_string_feasibility_v2_behavior_result.md) | Human-readable behavioral outcome |
| [Behavioral result JSON](docs/results/same_string_feasibility_v2_behavior_result.json) | Machine-readable behavioral outcome |
| [Behavioral evidence snapshot](release/familiarity_answerability/README.md) | Content-addressed protected evidence record |
| [Representation pilot](docs/results/same_string_representation_pilot_v2.md) | Exploratory four-unit activation study |
| [Representation v3 result](release/familiarity_answerability/representation_replication_v3/analysis/result.md) | Larger held-out representation replication |
| [Representation v3 JSON](release/familiarity_answerability/representation_replication_v3/analysis/result.json) | Machine-readable replication outcome |
| [Claim boundaries](docs/familiarity_answerability_claims.md) | Registered limits on interpretation |

<details>
<summary><strong>Technical history and preserved failed instruments</strong></summary>

### Earlier real-entity instrument

The original Source-v5 corpus and later R9/R10 runs did not satisfy their
registered readiness gates. R11 produced enough strictly scored candidates,
but an independent audit found 14 disallowed ambiguity, granularity, or alias
defects in its 24-item packet. R11 is therefore `not_evaluable`.

These runs tested whether the measurement instrument was ready. Their failure
is not evidence for or against the familiarity-by-answerability hypothesis.
Their records remain public so that corpus changes and stopping decisions are
auditable:

- [R11 outcome report](docs/results/source_v6_r11_instrument_development_outcome.md)
- [R11 runbook](docs/fa_source_v6_r11_runbook.md)
- [R11 amendment](docs/amendments/2026-07-28-fa-source-v6-r11-surplus-instrument-development.md)

### Same-String v1

The first Same-String corpus was also `not_evaluable`: only 73 of 192 pairs
passed the frozen naturalness rule, so registered split and domain quotas could
not be filled. It was preserved rather than silently repaired after the fact.

- [Same-String v1 runbook](docs/fa_same_string_primary_runbook.md)
- [Same-String v1 design](docs/superpowers/specs/2026-08-01-same-string-primary-hybrid-design.md)
- [Same-String v1 amendment](docs/amendments/2026-08-01-fa-same-string-primary.md)

### Full protocol record

- [Original preregistration](docs/familiarity_answerability_preregistration.md)
- [Main execution runbook](docs/familiarity_answerability_runbook.md)
- [Human naturalness protocol](docs/fa_naturalness_rating_protocol.md)
- [Historical amendments](docs/amendments)
- [Historical results](docs/results)

</details>
