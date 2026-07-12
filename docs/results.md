# Frozen Concept Results

## Provenance

- Implementation SHA used for this run:
  `04568b9f1c1629ac7f08323b1c0602843fe91f48`
- Primary artifact SHA-256:
  `bf9c33a17f86d5a3c0851bedb302876aec2436bea662bc61099b0150f853bcd8`
- Dataset and evaluation: 1,200 exact artifact pairs from the frozen synthetic
  concept-binding study.

These results are a frozen record. They were not rerun or reinterpreted for this
report.

## Secondary Artifact Boundary

The live frozen secondary artifacts were created by implementation commit
`04568b9f1c1629ac7f08323b1c0602843fe91f48`, before the modern provenance and
completion sealing protocol existed. They contain no `analysis_id`, no
`analysis_provenance`, and no completion marker. They are therefore historical
evidence for the frozen result, not artifacts that satisfy the current sealed
secondary-analysis schema.

They are deliberately not retrofitted: doing so would create a new analysis
object after the fact. They are protected now by the legacy metrics guard, which
refuses a rerun when endpoint metrics already exist, together with the tracked
artifact hash and this tracked result report. The guard, hashes, and report
preserve the historical record; they do not turn the legacy files into modern
provenance-sealed artifacts.

## Registered Exact-Error Findings

The primary `exact_error` comparison is `not_supported`. The registered AUROC
delta is -0.03610154356423012, with 95% cluster-bootstrap CI
[-0.05645103380084038, -0.006502990610897541].

The secondary `exact_error` comparison is `not_supported`. Its AUROC delta is
0.006761066462559029, with 95% cluster-bootstrap CI
[-0.014911692589521408, 0.0319538488984108], raw paired entity-family
permutation p-value 0.6331834082958521, and BH-adjusted p-value 1.0. The
candidate AUROC is 0.9272866437045542 and the baseline AUROC is
0.9205255772419951.

The mechanistic `binding_error` endpoint is `not_evaluable`: it has 13 positive
examples across 8 clusters, below the frozen minimums of 20 positives and 10
clusters.

## Prefix And Crossing Boundary

The selected registered prefix is token 4/layer 16. It is the latest pre-token
prefix and has observed all but the final response token. This result therefore
does not demonstrate early warning or artificial intuition.

The validation probability surface contains independently fitted prefix
classifiers. A threshold selected for one classifier is not transferable to a
different prefix cell. Consequently, the primary and secondary crossing
diagnostics are `not_interpretable`. Numeric crossing diagnostics in both legacy
primary and secondary artifacts are invalid evidence and are ignored; they are
not used to make a timing claim.

The exploratory full monitor AUROC of 0.9535655058043118 is descriptive and
non-confirmatory. It does not alter the registered negative finding.

## Scope And Remaining Controls

This is a synthetic-only result for the resolved local concept task. It does not
establish transfer to documented external facts, other models, unsafe jailbreak
responses, or a reliable intervention trigger. It does not support a claim about
consciousness, introspection, ground-truth access, or safety.

The following controls remain pending:

- Concept falsification: full trajectory versus last token; PCA dimensions 16,
  32, and 64; raw dynamics versus operator residual versus combined; shuffled
  layers and random projection controls; and prompt-length, rarity-proxy, and
  distractor-count subgroups.
- Transfer: evaluate all 200 source-documented Wikidata triples with components
  fitted and prefixes selected only on synthetic train/validation data.
- Intervention: validation-only selection followed by held-out comparisons with
  no steering, norm-matched random, shuffled-label, ITI-style always-on, and
  operator-residual-triggered steering; require at least 20% relative error
  reduction, no more than five percentage points of matched-control loss, and a
  paired 95% bootstrap advantage over norm-matched random steering.
- Safety: complete the separate 200-example JailbreakBench extraction, Llama
  Guard labeling, frozen stratified 20% human audit, leave-one-category-out
  detection with category-clustered bootstrap, and held-out intervention study.
- Circuit follow-up: select held-out TP/FP/FN/TN cases deterministically, inspect
  base-checkpoint attribution graphs and feature interventions against
  norm-matched random features, and require a separate base-to-Instruct fidelity
  gate before treating those graphs as relevant to the target checkpoint.
- Replication: evaluate external transfer and multi-model behavior before making
  any broader reliability or safety claim.
