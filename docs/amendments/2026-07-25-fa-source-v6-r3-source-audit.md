# Source-v6 R3 Pre-Model Source-Audit Amendment

**Date:** 2026-07-25
**Scope:** Open instrument development only
**Status:** Failed source-frame feasibility before any R3 model output

## Lineage

Source-v6 R1 failed its open-development model-yield gate. Source-v6 R2 was
never screened by Gemma: a pre-model source audit found obvious construct
errors in its Place items. Both revisions remain development evidence and
cannot support Familiarity-by-Answerability claims.

R3 preserves the R2 model, parser, prompt templates, relation set,
two-of-three qualification threshold, split seed, split sizes, and numerical
readiness gate. It changes only deterministic source eligibility.

## Additional Place eligibility rules

A Place is eligible only when:

1. its English label is not shared by another Wikidata city;
2. its unique current P131 value differs from its unique current P17 value;
3. no registered P17, P131, or P30 alias equals the queried entity label.

These rules exclude homonymous prompts such as an unqualified `La Paz`, cases
where a country is registered as the administrative region, and copy-the-name
answers such as city and district both being called `Neu-Ulm`.

The filters are applied to the full ranked open pool before split assignment.
They do not use model outputs. Every Source-v5, R1, and R2 QID is excluded from
R3. The R2 entity cache may seed labels and aliases; its hash is bound into R3
provenance and missing QIDs are fetched from Wikidata.

## R3 readiness gate

R3 may advance to `construction_validation` only if:

- all 96 entities and 288 prompts complete with immutable provenance;
- every domain qualifies at least 16/24 entities;
- every registered relation is answered correctly at least 16/24 times;
- Place P17 and P30 are each correct at least 18/24 times;
- a deterministic error packet is produced for independent human review.

The gate is evaluated by code. A freeze manifest must bind the R3 development
execution identity, items hash, summary hash, parser hash, source integrity,
config hash, and source-code commit. A failed or absent Development run cannot
open `construction_validation`.

## Outcome

The R3 source-frame build found 44 complete, tokenizer-compatible Place records
after applying the frozen eligibility rules, below the required 48. The build
therefore stopped fail-closed before split materialization and before any Gemma
screening. No model output or Familiarity-by-Answerability endpoint was opened.

R3 is retained as a source-feasibility result. It is not rescued by changing
its query limit. The separately registered R4 revision keeps the R3 filters and
expands only the open Place query window.

## Claim boundary

R3 is still instrument development. Passing the gate would show only that the
screening instrument has adequate open-development yield. It would not confirm
Familiarity, Answerability, hallucination, intuition, or a mechanism.
