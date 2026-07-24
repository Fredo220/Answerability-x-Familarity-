# Confirmatory Execution Status

**Study:** Familiarity versus Answerability
**Date:** 2026-07-24
**Claim status:** Confirmatory corpus `not_evaluable` under the frozen gate

## Completed

- Preserved the frozen `fa-confirmatory-wikidata-v2` source snapshot after an
  audit found that multi-release works could accept a later year for a
  first-release question.
- Materialized the corrected, superseding
  `fa-confirmatory-wikidata-v3` source snapshot without overwriting v2.
- Materialized 384 split-isolated candidate entities and 1,152 registered
  screening questions.
- Excluded every pilot QID and six coarse-type-ambiguous QIDs.
- Preserved all non-deprecated values for multivalued entity properties and
  retained only the earliest non-deprecated positive year for year-valued
  screening questions.
- Verified exact split/domain quotas, global QID/name/ID uniqueness, and exactly
  three source-bound questions per entity.
- Sealed every materialized file and raw QLever/Wikidata cache file in
  `data/fa/confirmatory_source_v3/source_integrity_v1.json`.
- Source-integrity artifact SHA-256:
  `98d140b0f39a6f8bd2db8a1b861f5bb33ac91e3a88eac786f5ec828158f43964`.
- Completed the tokenizer-only v3 feasibility audit. Only 167/384 candidates
  admitted complete three-pseudonym reserves, including 10/96 places in the
  assigned pool and 18/150 places across the complete cached source frame.
- Froze the v4 source amendment before any Gemma screening completion was
  opened. It raises the fixed maximum source rank to 1,200, applies exact
  matchability before split assignment, and narrows the estimand to
  Gemma-tokenizer-matchable entity names.
- Built and audited source v4, then rejected it before screening when the
  separate idempotence replay found 96 order-dependent pseudonym rows.
- Froze source v5 and pseudonym-generator v3 to canonicalize candidate order
  before global collision handling. All scientific endpoints, quotas, source
  ranks, model pins, and claim boundaries remain unchanged.
- Materialized and independently audited source v5:
  384 globally unique entities, 1,152 source-bound screening questions, and
  1,152 globally unique pseudonym reserves.
- Verified exact split and domain quotas, three questions and three reserves
  per entity, every registered source-integrity hash, and absence of real-name
  versus pseudonym collisions.
- Replayed pseudonym construction with two different manifest orders. All five
  split files were byte-identical and the normalized snapshot semantics were
  identical.
- Source-v5 integrity artifact SHA-256:
  `ac5c7c812482d38af397c1ba5a21355c82ca665d652159eb71c856fa87595bf5`.
- Complete matchability counts before selecting the first 96 per domain were:
  creative works 448, organizations 537, people 484, and places 108.
- Added fail-closed CLI checks requiring the exact 2x source pool and the
  active frozen integrity artifact before confirmatory screening.
- Ran the first pinned-Gemma `mechanism_train` source-qualification
  transaction at commit `39832250653b431ee93e0b12c97089de91e9e554`.
  It stopped before human packets or protected endpoints because the exact
  domain quota was not filled.
- Preserved the checksum-verified failed checkpoint in Google Drive. The
  stopped audit reported qualified counts of 31 creative works, 24
  organizations, 19 people, and 4 places.
- Invalidated that execution as model evidence after reproducing a runtime
  defect: already-rendered Gemma prompts received a second BOS token during
  tokenization.
- Froze the double-BOS implementation-correction amendment before rerunning.
  Sources, prompts, parser, aliases, thresholds, quotas, and model pins remain
  unchanged.
- Verified the corrected runtime and confirmatory execution paths with
  `135 passed`.
- Executed the corrected pinned-Gemma screening transaction in a fully reset
  Tesla T4 runtime at commit
  `945f6b58bb3b7ded7f4ab77ece11374bc683fe37`.
- Verified the launch bundle SHA-256:
  `f1c2f262d423c647a1d0884163db58a279e6579258e40e9b6da80c7f3a8437d4`.
- The corrected `mechanism_train` audit qualified 31 creative works, 25
  organizations, 19 people, and 4 places. The frozen selection required 20 per
  domain, so the transaction stopped with deficits of 1 person and 16 places.
- The corrected screening-completion data SHA-256 is
  `867f656a3d57bd18a6d0a8cac42bf2c03e0e482ddbc6862a34979b0b0f476f2d`.
- The corrected stopped-audit data SHA-256 is
  `03eafa841803e52afe6b6a3fd875aba2c396c3e61120b9b76a085cfe6b37f192`.
- The corrected Drive checkpoint metadata is
  `screening-mechanism_train-completion-ac8e17628e01c6d3-31f5a8c17cdc664e.checkpoint.json`,
  metadata SHA-256
  `31f5a8c17cdc664e14118b46e9207646bf600a110e8591ff32cf530399faf8dd`,
  and archive SHA-256
  `ac8e17628e01c6d3745ce3dde8154f2417576713e0fc215d8e0251f18ecd427a`.
- Applied the preregistered stop rule. No human packets were issued and no
  protected endpoint was opened.
- Added a regression guard that rejects commit-mismatched or unbound local
  screening shards before resume.
- Rebuilt the pinned Graphify architecture graph and report.

## Tokenizer Access

Hugging Face authentication, repository access, the exact pinned revision, and
the protected Gemma tokenizer were verified successfully.

## Current Construction Gate

The v3 corpus cannot satisfy the registered domain quotas under the frozen
exact-token controls. Source v4 met the quotas but is superseded because its
pseudonym output was order-dependent. Source v5 has passed the fixed-rank,
domain-balance, source-integrity, and order-invariance construction gates. The
first Gemma screening transaction is preserved but invalid because the runtime
double-inserted the BOS token. The corrected new-commit transaction reproduced
the frozen source-qualification shortfall after that implementation defect was
removed. Per the amendment stop rule, Source v5 is `not_evaluable`; it may not
be rescued by changing prompts, parsers, aliases, thresholds, quotas, or
candidate order. Human packet issuance is permanently closed for this frozen
corpus. The model and tokenizer revision remain:

```text
google/gemma-2-2b-it
299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
```

No alternate model or mutable revision may replace this pin.

## Human Gate

Human packets were not issued because pinned Gemma screening did not produce
the verified 244-pair collection. The following preregistered requirements
therefore remain unexecuted:

- Two distinct real people must independently rate every issued pair.
- A third distinct person rates only disagreements emitted by the compiler.
- AI agents, the researcher, copied identities, and empty templates are not
  human evidence.

## Protected Endpoints

No `behavior_test`, `probe_test`, or `intervention_test` endpoint was opened.
They remain closed for Source v5 because the construction gate failed.

## Invalidated Run Provenance

The double-BOS run remains an implementation artifact, not model evidence:

- Commit:
  `39832250653b431ee93e0b12c97089de91e9e554`
- Checkpoint metadata:
  `screening-mechanism_train-completion-906601635adca3e7-a1113eac36e218bc.checkpoint.json`
- Metadata SHA-256:
  `a1113eac36e218bcd366bde2bea04b0db22f04840771441367970865a15c40e6`
- Archive SHA-256:
  `906601635adca3e7fcc5da375a487df4ee7546be11f99f2f2eb68e39f62fdd6b`
- Screening-completion data SHA-256:
  `63654d41e240e5a5827d68f8aacf38bd925974c1f13f14dac9a3fea96a60e26b`
- Stopped-audit data SHA-256:
  `66e4bd4858741ef3da6652b6f4db392a39585d8beedc6dab64f0ec0d4721419d`

## Verification Boundary

The post-review focused confirmatory suite completed with `135 passed`. A
broader `test_fa_*` run was intentionally stopped after 26 minutes at
`437 passed`; it was not a completed full-suite run and is not reported as one.
A prior repository-wide run covering 1,040 tests from several independent
research tracks was stopped after 19 minutes at `423 passed`; it also was not a
completed full-suite run and is not reported as one.
