# Confirmatory Corpus Reserve Amendment

**Date:** 2026-07-24
**Scope:** Familiarity versus Answerability only
**Status:** Frozen before confirmatory screening completions, human ratings, or protected model outcomes are opened

## Reason

The preregistered protocol requires deterministic replacement from sealed reserve
pairs when human naturalness exclusions reduce a split/domain stratum below its
target. The original execution code retained only the target number of screened
pairs, which made that registered replacement rule impossible to execute.

This amendment operationalizes the existing reserve rule. It does not alter the
research question, hypotheses, protected split sizes, endpoints, model pin,
screening threshold, human-rating rule, or analysis thresholds.

## Frozen screening pool

The source pool contains two candidate real entities for every final entity slot.
The 192 final slots therefore begin with 384 split-isolated screening candidates:

| Split | Final units per domain | Screening candidates per domain |
|---|---:|---:|
| `mechanism_train` | 16 | 32 |
| `locked_validation` | 8 | 16 |
| `behavior_test` | 12 | 24 |
| `probe_test` | 6 | 12 |
| `intervention_test` | 6 | 12 |

Candidates are selected from a checked-in Wikidata-derived snapshot, excluding
all pilot QIDs. Within each domain, the highest-ranked eligible source pool is
assigned to splits by SHA-256 of the registered split seed, domain, and QID.
Candidate names and QIDs remain split-isolated.

This amendment originally activated `fa-confirmatory-wikidata-v2`. The later
pre-outcome source-v3 amendment supersedes v2 only to make year-valued answers
consistent with "founded" and "first released" wording. v3 retains the same
QIDs, split assignment, quotas, queries, retrieval date, and reserve policy.
It excludes QIDs that satisfy more than one registered coarse type and retains
aliases for every non-deprecated entity-valued screening property. The earlier
v1 snapshot used one sampled property value and is superseded before any Gemma
screening output, human rating, or protected outcome was opened.

## Frozen naturalness reserve

After pinned Gemma screening, the deterministic per-domain human-audit pool is:

| Split | Final units per domain | Human-audit reserve per domain | Audited pairs per domain |
|---|---:|---:|---:|
| `mechanism_train` | 16 | 4 | 20 |
| `locked_validation` | 8 | 2 | 10 |
| `behavior_test` | 12 | 3 | 15 |
| `probe_test` | 6 | 2 | 8 |
| `intervention_test` | 6 | 2 | 8 |

Qualified entities are selected in checked-in candidate-manifest order until
each audited-pair quota is filled. If screening does not fill every quota, the
confirmatory build stops. No quota or ordering may be changed after screening
outputs are inspected.

Two independent raters score every audited pair. A third independent adjudicator
rates only registered disagreements. Final confirmatory pairs are selected from
accepted audited pairs in the preregistered SHA-256 pair order. If exclusions
leave a final split/domain quota short, the confirmatory build stops and the
endpoint is `not_evaluable`; no post-outcome replacement is allowed.

## Source fields

Screening uses three type-specific Wikidata claims that are unrelated to the
synthetic archive task:

- people: country of citizenship, occupation, place of birth;
- places: country, containing administrative region, time zone;
- organizations: country, headquarters location, inception year;
- creative works: film director, first publication/release year, country of origin.

Wikidata structured data are CC0. Every candidate and question records its QID,
EntityData URL, retrieval date, source-builder revision, and domain-query hash.

## Claim boundary

This amendment creates no empirical evidence. Green software tests establish
only deterministic corpus construction and gate behavior. Familiarity is defined
only by the pinned Gemma two-of-three screening result, and confirmatory findings
exist only after genuine human ratings and one-time protected evaluations.
