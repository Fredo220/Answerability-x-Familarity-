# Causal Pilot Post-Run Amendment

This post-run amendment records an implementation defect discovered only
after the protected causal endpoints had opened. It does not alter the frozen
design, evidence, machine decision, thresholds, or test identities.

The mandatory label-shuffled control in
`same-string-answerability-causal-pilot-v1` was exactly equal to the primary
direction because the implementation permuted a complete set before taking
its mean. The frozen evaluator returned `not_supported`; the final scientific
interpretation is stricter:
`not_evaluable_as_confirmatory_causal_test`.

The corrected implementation uses balanced within-unit label swaps and
rejects exact primary/control equality before sealing. It may be used only in
a separately preregistered replication with fresh test units. Full evidence
and diagnostics are published in the
[artifact audit](../release/familiarity_answerability/answerability_causal_pilot_v1/POST_RUN_AUDIT.md).
