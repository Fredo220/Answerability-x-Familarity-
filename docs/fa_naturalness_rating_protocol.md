# Familiarity vs. Answerability Naturalness Rating Protocol

## Purpose

The human audit tests whether matched pseudonyms are comparably natural and
type-appropriate to screened real entity names. It is a pre-outcome dataset
quality gate, not a model evaluation and not evidence for the study hypotheses.

## Independence and Blinding

- Two initial raters work independently.
- Raters must not discuss items, use a language model, or search the web.
- Each rater sees opaque item IDs, a coarse entity type, and two names in
  neutral sentences.
- Pair IDs and real-versus-synthetic labels are absent from rater packets.
- Candidate order is reversed across the two initial raters for every pair.
- Public rater files are written under `public/`; the researcher keeps the
  permission-restricted `private/unblinding-key.json` and the immutable
  issuance data and manifest inaccessible to raters.
- Model generations, activation analyses, and endpoint results are not shown
  during rating.
- Each submitted row must explicitly attest that the rating was completed
  independently under these conditions.

This is source-label blinding, not complete perceptual blinding: a rater may
recognize a familiar real-world name. That unavoidable limitation is reported.
Independence is attested by each rater; the workflow does not independently
verify real-world identity behind a rater ID.

## Rating Fields

Each rater receives a separate JSON reference packet and a separate CSV
worksheet. Every CSV row already shows the entity type, candidate A and B, and
one neutral sentence for each candidate. The rater works only in their own CSV
and does not need to copy values from the JSON file.

For each candidate, fill in:

- `naturalness`: integer 1 to 5, where 1 is clearly malformed, 3 is plausible,
  and 5 is fully natural;
- `type_fit`: integer 1 to 5, where 1 does not fit the stated entity type, 3 is
  plausible, and 5 is a strong fit;
- `malformed`: `true` only for an orthographic or linguistic defect, otherwise
  `false`.

No field may be left blank.

Set `independence_attested` to `true` on every row. Do not edit `packet_id`,
`rater_id`, `item_id`, the question, entity type, names, or sentences. The
compiler verifies those displayed stimuli against the sealed JSON packet and
rejects edited worksheets.

## Registered Decision Rule

For each rater, a pair passes when the synthetic candidate is not marked
malformed and the absolute real-versus-synthetic naturalness difference is at
most one point.

If the two initial pass/fail verdicts disagree, the compiler emits a new
blinded packet containing only those disagreements. A third independent rater
must complete it. Third-rater assignments are explicitly registered and may
not be created for pairs where the initial raters agree.

The deterministic audit then excludes a pair if any used rater marks the
synthetic candidate malformed or if the median real-versus-synthetic
naturalness gap exceeds one point. Type-fit scores and agreement statistics are
preserved for descriptive summaries but do not silently alter this rule.

## Commands

Prepare two initial packets:

```bash
feature-dynamics fa-prepare-naturalness-ratings \
  --config <config.json> \
  --root <artifact-root> \
  --screened-matches-manifest <screened-match-collection.manifest.json> \
  --output-dir <new-rating-directory> \
  --rater-id <rater-a> \
  --rater-id <rater-b> \
  --shard-id <packet-issuance-id>
```

The command seals the exact public packets and private A/B mapping as an
immutable, permission-restricted `naturalness_packet_issuance` artifact before
distribution. Only the files under `public/` may be sent to raters. After both
response CSVs are complete, compile them against that issuance:

```bash
feature-dynamics fa-compile-naturalness-ratings \
  --config <config.json> \
  --root <artifact-root> \
  --screened-matches-manifest <screened-match-collection.manifest.json> \
  --issuance-manifest <packet-issuance.manifest.json> \
  --response <rater-a-response.csv> \
  --response <rater-b-response.csv> \
  --shard-id <new-shard-id> \
  --adjudicator-id <rater-c> \
  --adjudication-output-dir <new-adjudication-directory>
```

If the initial raters disagree, this command returns
`status=needs_adjudication`. It seals the initial submissions and disagreement
set, issues a separately sealed third-rater packet, and does not write a final
ratings artifact. After the third response is complete, finalize with:

```bash
feature-dynamics fa-finalize-naturalness-adjudication \
  --config <config.json> \
  --root <artifact-root> \
  --screened-matches-manifest <screened-match-collection.manifest.json> \
  --initial-submission-manifest <initial-submission.manifest.json> \
  --adjudication-issuance-manifest <adjudication-issuance.manifest.json> \
  --adjudication-response <rater-c-response.csv> \
  --shard-id <final-ratings-id>
```

## Evidence Boundary

The current `qwen17b_pilot_v6` packets are a workflow smoke test for eight pilot
pairs. Empty templates are not human evidence. Confirmatory Gemma construction
remains blocked until the full preregistered pair corpus exists and two
independent raters have completed this protocol.
