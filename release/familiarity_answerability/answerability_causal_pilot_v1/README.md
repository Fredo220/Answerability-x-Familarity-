# Answerability Causal Pilot v1 Artifacts

This directory contains the complete live artifact set for
`same-string-answerability-causal-pilot-v1`.

- Environment: free Google Colab T4
- Selected intervention: layer 18, multiplier 1.0
- Registered test units: 18 unseen-entity and 18 unseen-template units
- Evidence receipts: 432 of 432
- Frozen machine decision: `not_supported`
- Final scientific interpretation after audit:
  `not_evaluable_as_confirmatory_causal_test`

The machine-generated files under `results/` are preserved unchanged. Read
[POST_RUN_AUDIT.md](POST_RUN_AUDIT.md) before interpreting them: a mandatory
label-shuffled control was found to be exactly identical to the primary
direction. The run therefore does not establish a feature-specific or
layer-specific causal mechanism.

The training activations used to reconstruct the direction are published in
the sibling `representation_replication_v3` release. Absolute `/content/...`
paths in the live manifests record the original Colab environment; the bound
content hashes are the portable provenance identifiers.

The downloaded archive was `fa-causal-pilot-v1.zip`, with SHA-256 recorded in
`ARCHIVE_SHA256.txt`. All 456 JSON files parsed successfully during import.
