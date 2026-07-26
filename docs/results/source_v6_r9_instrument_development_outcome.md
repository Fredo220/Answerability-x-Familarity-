# Source-v6 R9 Instrument-Development Outcome

## Status

**Not evaluable for confirmatory Familiarity-by-Answerability claims.**

The bounded `instrument_development` run completed on a Colab Tesla T4, but
the preregistered instrument-readiness gate failed. No protected endpoint was
opened, and this result is not evidence for or against the study hypothesis.

## Execution identity

- Git commit: `1b3bb3185e02b0adacb11da32ead4accd29922b7`
- Model: `google/gemma-2-2b-it`
- Model and tokenizer revision:
  `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`
- Source revision: `fa-development-source-v6-r9`
- Split: `instrument_development`
- Candidates: 48
- Prompts: 144
- Batch size: 16
- Structural and semantic pre-model audits: passed
- Runtime: Tesla T4, PyTorch `2.11.0+cu128`, Transformers `4.57.1`,
  Accelerate `1.12.0`

The PyTorch version differed from the core lock. This is recorded as a
development-runtime deviation. It does not change the gate result or expand
the allowed claim scope.

## Frozen gate result

The gate failed both registered criterion families:

| Domain | Qualified | Required |
| --- | ---: | ---: |
| creative_work | 6 | 8 |
| organization | 9 | 8 |
| person | 1 | 8 |
| place | 12 | 8 |

Registered relation-level minima also failed:

- `creative_work`: director `3/8`, release year `6/8`, country `9/8`
- `organization`: country `11/8`, headquarters `10/8`, inception `3/8`
- `person`: citizenship `5/8`, occupation `2/8`, birthplace `1/8`
- `place`: country `12/9`, continent `12/9`, direct administrative parent
  `1/8`

The immutable gate artifact reports:

```json
{
  "failed_criteria": [
    "qualified_by_domain",
    "success_by_domain_relation"
  ],
  "gate_passed": false
}
```

## Artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| execution identity | `888bf5a41ff411147e4e1a47a0f195df566587e21a3ef00fc7e971a8c290099b` |
| screening items | `78988d156bfc2a1d635c1e8bebe61fbe7b8061f772b549d84e912218f4489f00` |
| screening yield | `7a868f1f2e4f0981d613bb18c94abe2281a1a33f8043891275c794e0f71e28aa` |
| readiness gate | `68eb8fb85ead2b805521e057b6aff36fbd80e227a13ad495f8a81ad6284e5b5d` |
| compressed transfer bundle | `1e0356eb1dbbd350c8b735307cb572c16101057d74f96f35ef0cfa41a7e1131e` |

## Development-only diagnosis

The failures contain at least two distinct modes:

1. **Genuine factual misses.** Examples include wrong directors, release
   years, birthplaces, and direct administrative parents.
2. **Potential answer-surface or ontology misses.** Examples include
   `Professional footballer` versus `footballer`, `Pharaoh` versus
   `ruler`/`monarch`, and historical or alternate place/polity names.

The second category is being assessed by two independent exploratory AI
auditors. Those ratings are diagnostic only and cannot replace the two real
human raters required by the registered protocol.

Both exploratory auditors classified all 69 strict-score failures. They agreed
that 49 were incorrect and 11 were semantically equivalent; the remaining
nine had at least one ambiguous rating. A conservative sensitivity analysis
credited only those 11 consensus-equivalent answers:

| Domain | Strict qualified | Consensus-adjusted | Required |
| --- | ---: | ---: | ---: |
| creative_work | 6 | 6 | 8 |
| organization | 9 | 9 | 8 |
| person | 1 | 5 | 8 |
| place | 12 | 12 | 8 |

This does not rescue the gate. It shows that answer-surface coverage explains
part of the person-domain collapse, but a scorer correction alone is
insufficient. Director, inception, birthplace, and direct-administrative-parent
relations remain below their frozen minima.

## Fastest scientifically defensible follow-up

R9 remains immutable. A narrow R10 follow-up should reuse the current
pipeline, tests, source provenance, model runner, parser, and gate machinery.
It should change only the parts shown by this open development run to be
non-viable:

1. Freeze a source-grounded semantic-equivalence policy before any fresh model
   output. It must distinguish aliases and valid answer-type subcategories
   from merely related answers.
2. Replace relations that are poor natural answerability instruments, such as
   exact Wikidata `P131` and fragile inception dates, using a model-blind,
   documented relation rule.
3. Draw fresh development candidates with a fixed popularity and provenance
   rule, while permanently excluding every R8/R9 entity and question.
4. Run one bounded R10 development preflight. Do not tune thresholds after
   that run.
5. Only after a passing preflight, obtain two independent human ratings,
   freeze the instrument, and execute fresh protected splits once.

This is a corpus-and-scoring repair, not a rewrite of the research pipeline.
