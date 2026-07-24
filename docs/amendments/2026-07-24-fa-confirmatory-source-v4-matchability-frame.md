# Confirmatory Source v4 Matchability Frame

**Date:** 2026-07-24
**Scope:** Familiarity versus Answerability only
**Status:** Frozen before Gemma screening completions, human ratings, task
generation, activation extraction, or protected endpoint outcomes are opened

## Trigger

The v3 tokenizer-only audit produced complete three-pseudonym reserves for 167
of 384 source candidates. Matchability was strongly domain-dependent:
creative works 50/96, organizations 57/96, people 50/96, and places 10/96.
Across every eligible record already present in the v3 rank-180 cache, only 18
of 150 places were matchable. The registered confirmatory design requires 96
pre-screening candidates per domain, so v3 cannot fill the corpus even under
the impossible best case in which every remaining cached place matched.

No Gemma factual-screening completion, task output, activation, human rating,
or protected endpoint result was inspected before this amendment.

## Frozen v4 Sampling Rule

1. The source revision is `fa-confirmatory-wikidata-v4`.
2. Each registered Wikidata domain query has a fixed maximum rank of 1,200.
   This limit will not be increased after retrieval, regardless of yield.
3. Queries remain ordered by descending sitelink count and then QID.
4. Queries collect every distinct Wikidata direct/truthy value for each
   registered field. Year-valued fields retain the earliest collected positive
   year; entity-valued fields accept the English label and aliases of every
   collected QID.
5. Only English labels and aliases are fetched from the Wikidata entity API.
   This is a storage optimization and does not use model outcomes.
6. Pilot QIDs, cross-domain QIDs, duplicate normalized labels, ineligible
   labels, and records lacking all three registered factual values are removed
   before matchability testing.
7. Pseudonym generator revision `fa-confirmatory-pseudonyms-v2` is seeded by
   generator revision, QID, domain, source name, and attempt. It is therefore
   invariant to later split assignment.
8. For every eligible source record, the frozen generator attempts at most
   5,000 proposals and requires three unique pseudonyms satisfying every
   registered exact surface control. A failure is reported as
   `no_match_under_frozen_generator`; it is not a proof that no match exists.
9. Within each domain, the first 96 complete-matchable records in frozen
   source-rank order enter the screening pool.
10. Only after this selection does the existing SHA-256 split assignment run.
    It produces the unchanged per-domain pools `32/16/24/12/12` for
    mechanism-train, locked-validation, behavior-test, probe-test, and
    intervention-test.
11. The final 384-candidate pseudonym files are generated once, require exactly
    three unique candidate-linked reserves per source, and are hashed into the
    source-integrity artifact before screening can run.
12. Factual screening, reserve quotas, two-rater naturalness review,
    adjudication, endpoint thresholds, model revision, chat template, and all
    hypotheses remain unchanged.

If any domain has fewer than 96 complete matches by source rank 1,200, the
confirmatory study is `not_evaluable`. The source limit, generator, matching
predicate, ordering, or quotas must not be revised after factual-screening
outputs are inspected.

## Frozen Lineage

- Preregistration SHA-256:
  `1bc81440b507cc30ac899962c8ca3121718870e22a62fc0988fa9f7b8a8ccdf7`
- Matching-policy v5 SHA-256:
  `f08becd4442debc9cb73247597ca81027f9b5a70667dba7c88020bd1c44bbdf9`
- Source matchability-policy SHA-256:
  `dc6bbc175e91c673ab97e31f9db3075b1d15ca9e4e167f7f84d92835d5f47dbf`
- Gemma/tokenizer revision:
  `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`
- Chat-template SHA-256:
  `ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6`
- Confirmatory Gemma screening completions inspected: `false`
- Human ratings issued: `false`
- Confirmatory task outputs or activations inspected: `false`
- Protected endpoints opened: `false`

## Required Reporting

The public report must include:

- eligible, matchable, and selected counts by domain;
- included-versus-excluded summaries for source rank, word count, character
  count, punctuation, and tokenizer length;
- every source, tokenizer, generator, split, and materialized-file hash;
- all source-construction failures;
- the narrower population claim below.

## Claim Boundary

The confirmatory estimand applies to **Gemma-2 tokenizer-matchable entity
names sampled from the frozen Wikidata frame**, not to all familiar entities.
Matchability can correlate with token compactness and pretraining frequency.
The filter is a surface-control requirement, not evidence for Familiarity or
Answerability.
