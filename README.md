# Familiarity vs. Answerability in Language Models

Can a language model tell the difference between **knowing more about a target**
and **having the evidence needed to answer a specific question**?

This repository contains four completed stages on the pinned
`google/gemma-2-2b-it` checkpoint:

1. a behavioral test of whether unrelated exposure makes the model answer
   without evidence; and
2. a representation-level test of whether internal activations encode
   answerability on the same controlled task; and
3. a registered intervention study followed by a post-run control audit; and
4. a fresh causal replication with corrected controls and new test units.

## Results at a Glance

| Experiment | Question | Result |
|---|---|---|
| Behavioral pilot | Does unrelated exposure selectively increase unsupported answer attempts? | **Not supported** |
| Representation replication | Do fixed internal activations predict answerability beyond the registered bag-of-ngrams baseline? | **Supported on this controlled task** |
| Causal follow-up | Does the training-only answerability direction locally change the code-versus-`UNKNOWN` margin more than the registered controls? | **Live run completed; confirmatory causal test not evaluable after control audit** |
| Causal replication v2 | Does the corrected intervention beat every control on fresh entities and templates? | **Mixed split evidence; overall not supported** |

These results are compatible. The model did **not** display the predicted
failure behavior, while its internal activations still contained decodable
information about whether the prompt supplied the answer.

The evidence chain is therefore: **behavioral null -> robust held-out
decodability -> mixed causal evidence**. The intervention reliably shifted an
internal response margin, but its layer specificity did not generalize across
both registered test splits.

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

### 3. Causal follow-up: completed, but not confirmatory

The intervention study added the v3 training-only answerability direction on
fresh unanswerable prompts and subtracted it on fresh answerable prompts. It
then measured the change in the model's sequence log-probability margin
between the correct archive code and `UNKNOWN`.

The workflow uses 12 validation units to select one fixed layer and strength,
then evaluates 18 unseen-entity and 18 unseen-template units separately. Its
432 atomic receipts include the primary intervention, baseline, reversed,
label-shuffled, wrong-layer, wrong-anchor, no-intervention, and five
norm-matched random controls. A protected result is produced only when every
registered receipt verifies against the frozen corpus, runtime, prompt,
intervention site, and request hashes.

The free-Colab T4 run completed all 432 registered primary and control
receipts. The primary mean bidirectional effect was `0.1812` on unseen
entities and `0.1659` on unseen templates. The frozen evaluator returned
`not_supported` because neither split beat its strongest control. The
intervention shifted log-probability margins but changed none of the 144
primary-versus-baseline greedy outputs.

A required post-run audit then found a stricter problem: the sealed
label-shuffled vector was bit-for-bit identical to the primary vector. The
original shuffle permuted complete sets before averaging, an operation that
algebraically preserved the same mean direction. This mandatory control was
therefore not an independent null. The machine result remains unchanged, but
the scientific status is **not evaluable as a confirmatory causal test**. The
entity split also had a larger wrong-layer effect (`0.4388`) than the primary
effect, so the run does not support a layer-specific answerability mechanism.

The defect is fixed for future studies with balanced within-unit label swaps
and a fail-closed equality check. The opened test units cannot be reused to
claim confirmation; a new preregistered replication would require fresh
units. This correction does not change or rescue the completed result.

[Open the Colab notebook](notebooks/fa_answerability_causal_pilot_colab.ipynb),
[read the post-run audit](release/familiarity_answerability/answerability_causal_pilot_v1/POST_RUN_AUDIT.md),
or inspect the
[frozen machine result](release/familiarity_answerability/answerability_causal_pilot_v1/results/result.md).

### 4. Causal replication v2: the direction steered margins, but not robustly at one layer

The fresh v2 replication corrected the invalid shuffled-label control before
opening any outcome. It used new names, archive codes, and test templates while
keeping the intervention fixed at layer 18. All 432 registered primary and
control receipts completed and passed the independent hash and provenance
audit.

On unseen entities, the primary intervention had a clear positive margin
effect (`0.1997`, 95% CI `[0.1927, 0.2067]`). However, the same direction at
the registered wrong layer had an even larger effect (`0.3448`). This split
therefore failed the required layer-specific control contrast.

On unseen templates, the primary effect was also positive (`0.1705`, 95% CI
`[0.1608, 0.1804]`) and exceeded every registered control. That split was
`causally_supported`. Because the preregistration required both unpooled
splits to pass, the overall result is **`not_supported`**.

![Primary and control effects in the causal v2 replication](docs/assets/familiarity_answerability_causal_v2.png)

*Figure 2. Descriptive mean bidirectional margin effects. Positive primary
values indicate steering in the predicted direction. The frozen decision used
unit-level bootstrap intervals and required the primary effect to beat every
control on both splits. The wrong-layer result on unseen entities prevented
the overall layer-specific claim.*

The intervention changed only one of 144 primary-versus-baseline greedy
outputs. The result therefore shows local control of a probability margin, not
a reliable behavioral correction method. It also does not show that layer 18
is a unique answerability mechanism.

[Read the v2 post-run audit](release/familiarity_answerability/answerability_causal_replication_v2/POST_RUN_AUDIT.md),
inspect the
[machine-readable result](release/familiarity_answerability/answerability_causal_replication_v2/results/result.json),
or open the
[pinned Colab notebook](notebooks/fa_answerability_causal_replication_v2_colab.ipynb).

## What the Results Do and Do Not Mean

The completed evidence supports two scoped statements:

1. In the small behavioral pilot, unrelated contextual exposure did not cause
   the preregistered increase in unsupported answer attempts.
2. In the larger representation study, fixed activations predicted
   answerability beyond the registered bag-of-ngrams baseline across unseen
   entities and unseen prompt templates.

The v1 causal follow-up does not add a third supported statement. Its frozen
machine gate failed, and the post-run audit found that one mandatory control
was invalid.

The fresh v2 replication adds a narrower observation: the training-derived
direction causally shifted the registered output margin on both test splits,
but the effect was not robustly layer-specific. One split passed every control
and one did not, so the preregistered overall claim remains unsupported.

The project does **not** establish:

- a general mechanism of familiarity or hallucination;
- metacognition, human-like knowledge, or intuition;
- a robust, layer-specific causal mechanism for the decoded activation patterns;
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
but the mixed v2 intervention result and the single-model scope still require
broader replications before making stronger claims.

## Research Standards

The project was designed to make negative and positive results equally
auditable:

- hypotheses and decision rules were frozen before protected outcomes opened;
- model revision, prompts, parser, seeds, and thresholds are hash-bound;
- training, validation, and test units are separated;
- protected endpoints open once and fail closed on incomplete evidence;
- failed instruments and negative results remain in the public record; and
- the representation result cannot retroactively rescue the behavioral result.

The corrected v2 causal replication was registered as a separate study. It
used fresh test identities and could not reopen or rescue either the behavioral
decision or the invalidated v1 causal pilot.

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
6. map when and why the same direction transfers differently across layers and
   prompt families.

For a Fellowship-scale continuation, the highest-value extensions would be:

- compare the activation probe with an explicit binding parser, treated as a
  transparent task-solving baseline rather than an internal-state model;
- replicate on a second, larger model family; and
- replace archive-code prompts with natural factual and evidence-grounded
  questions while retaining matched answerability controls.

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
python tools/plot_causal_replication_v2.py
```

## Acknowledgments

**Sameh Aburadi** supported the independent execution and quality-control work
for this project. His contributions included assistance with the pinned
Gemma/Colab runs and artifact review, candidate and ground-truth quality checks,
human-rating and audit materials, and identifying ambiguity, granularity, and
alias defects in the evaluated corpus. These checks helped prevent instrument
problems from being misreported as evidence about the research hypothesis.

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
| [Causal machine result](release/familiarity_answerability/answerability_causal_pilot_v1/results/result.md) | Frozen one-use evaluator output |
| [Causal post-run audit](release/familiarity_answerability/answerability_causal_pilot_v1/POST_RUN_AUDIT.md) | Control defect, final interpretation, and claim boundary |
| [Causal v2 preregistration](docs/familiarity_answerability_causal_replication_v2_preregistration.md) | Frozen corrected replication design and decision rule |
| [Causal v2 result](release/familiarity_answerability/answerability_causal_replication_v2/results/result.md) | Fresh-unit one-use evaluator output |
| [Causal v2 post-run audit](release/familiarity_answerability/answerability_causal_replication_v2/POST_RUN_AUDIT.md) | Independent receipt, control, preservation, and result audit |
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
