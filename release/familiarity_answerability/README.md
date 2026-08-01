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
