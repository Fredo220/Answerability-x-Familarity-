# Source-v6 R11 Instrument-Development Outcome

## Status

**Not evaluable for confirmatory Familiarity-by-Answerability claims.**

The complete R11 open-development screen and frozen relation selection were
successfully reproduced. The subsequently registered independent human audit
failed its instrument-quality gate. No construction-validation or confirmatory
endpoint was opened. This result is therefore not evidence for or against the
Familiarity-by-Answerability hypothesis.

## Execution identity

- Git commit: `932c7fadded4f93be5b9495ce44634afc4064da3`
- Model: `google/gemma-2-2b-it`
- Model and tokenizer revision:
  `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`
- Split: `instrument_development`
- Entities: 32 per domain
- Candidate relations: 5 per entity
- Prompts: 640
- Decoding: deterministic greedy, `max_new_tokens=16`

The artifact contained all 40 expected batches and 640 unique item rows. A
deterministic replay reproduced the frozen relation selection byte-for-byte.

## Frozen development selection

The registered selection rule chose three relations per domain. Development
yield was numerically sufficient for a later validation attempt:

| Domain | Selected relations | Qualified entities |
| --- | --- | ---: |
| creative_work | `P495`, `P57`, `P577` | 24/32 |
| organization | `P159`, `P17`, `P571` | 27/32 |
| person | `P27`, `P569`, `P570` | 30/32 |
| place | `P30`, `P36`, `P37` | 31/32 |

Selection SHA-256:
`e6e65b362e725d6a8890a1a64a4c914b56164f848d05923446f227dce1046c29`.

## Independent human audit

Two independent initial raters assessed a deterministic blinded packet of 24
items: four strict-score failures and two strict-score successes per domain.
They agreed on 20 items. A third independent adjudicator resolved the four
disagreements.

Final decisions:

| Label | Count |
| --- | ---: |
| `no_error` | 8 |
| `relation_unknown` | 2 |
| `ambiguous_ground_truth` | 7 |
| `wrong_granularity` | 6 |
| `incomplete_alias_set` | 1 |

The registered gate allowed zero instrument/source/scoring defects. Fourteen
items received a disallowed instrument label, while the final human decisions
had zero disagreement with the strict scorer's pass/fail classification:

```json
{
  "adjudicated_count": 4,
  "disallowed_count": 14,
  "maximum_disallowed_count": 0,
  "scoring_disagreement_count": 0,
  "maximum_scoring_disagreement_count": 0,
  "gate_passed": false
}
```

The principal failure modes were ambiguous entity or event identity, answer
granularity mismatches, and incomplete alias coverage. Consequently, high
strict-score yield alone was insufficient to establish a valid familiarity
instrument.

## Claim boundary

R11 establishes the following development result:

> The selected three-question instruments produced sufficient strict-score
> yield on Gemma 2 2B, but failed independent human review for unambiguous
> ground truth and answer granularity.

R11 does **not** establish that familiarity and answerability interact, that
Gemma has or lacks a fact, or that the main hypothesis is false. The registered
stop rule prohibits opening construction-validation data or modifying R11
against protected outcomes.

The fastest defensible continuation is a new development-only instrument
revision that reuses the existing model runner, source provenance, batching,
selection code, audit compiler, tests, and protected-data boundary. It should
repair only the failure classes revealed here and must be frozen before any
fresh validation output is inspected.

## Public artifacts

| Artifact | SHA-256 |
| --- | --- |
| execution identity | `2c6900685f3d01d5ff011f04fd5453d4f1a60ddade9c3fba349fbb52f0584928` |
| screening items | `559419db06fa46d234aa6f1a2ffb2d6c1d4a7f8d8a09b6d18c0467939d59e3b7` |
| screening yield | `a7988440f01d4c3742b2d539bba4a7566db8f16a71e3ff9daf280641662950bb` |
| relation selection | `390cda7cedf94bc35cc4c950ff8a0cd1d0977e261bd6c9d91187b5d37eb7a438` |
| human scoring audit | `cac8c423cd677e2c3f4933f1d7ebe2bcc92668103838ed4cb08875d12fae6ad1` |
| partner transfer archive | `12d9286a21196252f785f9c9768c41280195def853d369a7493580d3590dd99c` |

