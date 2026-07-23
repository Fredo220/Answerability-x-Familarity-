# Familiarity vs. Answerability Pilot Screening Amendment v2

**Dated:** 2026-07-23
**Scope:** Development-only Qwen smoke pilot
**Confirmatory impact:** None

## Observed v1 outcome

The first immutable screening run used `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca` and produced:

```text
runs/familiarity_answerability/smoke-v1/shards/pilot/
entity-screening-20260723-0001.jsonl.manifest.json
```

Its data SHA-256 is:

```text
04483e1f215871abaca686f025214908333a372f0dedc239ab9503ce4bb394fa
```

Only `United Nations` passed the frozen two-of-three exact-or-alias recall
criterion. The other candidates failed because the 0.6B smoke model returned
an incorrect low-salience fact or ignored the requested short-answer format.
Qwen's default chat rendering also emitted a thinking envelope despite the
registered 16-token generation budget.

This is a pilot feasibility failure, not evidence about the study hypotheses.
The artifact remains immutable and publishable as a negative development
result.

## Changes frozen before v2 outcomes

1. Qwen3 smoke rendering sets `enable_thinking=False`. The confirmatory Gemma
   renderer is unchanged.
2. The eight entity identities, four domains, synthetic names, tokenizer
   matching rules, and two-of-three qualification threshold remain unchanged.
3. v2 uses short, high-salience factual relations and a uniform completion
   frame: the model must emit only the answer after a colon.
4. v1 candidate and question manifests remain checked in as
   `candidates_v1.json` and `screening_questions_v1.json`.
5. v2 remains development-only. Its questions, completions, and entity choices
   cannot select confirmatory layers, thresholds, effects, or claims.

## v2 gate

The pilot build still requires exactly eight qualified entities. If fewer than
eight pass, execution stops and an expanded, independently sourced candidate
pool is screened under a new dated amendment. No accepted alias may be added
after inspecting a v2 completion.
