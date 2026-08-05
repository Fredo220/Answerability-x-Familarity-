# Familiarity-vs-Answerability Release Artifacts

The content-addressed archive below contains the complete restored artifact
store for the Same-String balanced pilot v2, including manifests, protected
Gemma generations, metrics, endpoint lifecycle records, and the preceding
naturalness audit.

| Artifact | SHA-256 |
|---|---|
| [`fa-58f1f069cb6a1906ff17a0282805f859675ae80b0f707fc0f768fc7a956178e3.zip`](fa-58f1f069cb6a1906ff17a0282805f859675ae80b0f707fc0f768fc7a956178e3.zip) | `58f1f069cb6a1906ff17a0282805f859675ae80b0f707fc0f768fc7a956178e3` |

Verify it locally with:

```bash
shasum -a 256 \
  release/familiarity_answerability/fa-58f1f069cb6a1906ff17a0282805f859675ae80b0f707fc0f768fc7a956178e3.zip
```

The endpoint lifecycle inside the archive is `sealed -> unlocked_once ->
evaluated -> closed`. Rater identifiers are pseudonymous (`rater-a`,
`rater-b`, and `rater-c`); the archive contains no API token or personal rater
identifier.

## Causal Replication v2

The fresh causal replication is published as an expanded, directly inspectable
artifact tree:

- [release directory](answerability_causal_replication_v2/)
- [frozen result](answerability_causal_replication_v2/results/result.md)
- [machine-readable result](answerability_causal_replication_v2/results/result.json)
- [independent post-run audit](answerability_causal_replication_v2/POST_RUN_AUDIT.md)

The release contains all 432 registered JSON receipts and their 432 atomic
receipt locks, plus the prepared corpus, train-only direction bundle,
validation seal, endpoint state, and frozen result. The source ZIP had SHA-256
`c2e711527d64a3a9d7aa4fcfde14a55b0638e1cb88a311fc16920e9e3dacaff0`.
