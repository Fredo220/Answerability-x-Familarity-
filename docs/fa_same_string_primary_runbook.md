# Same-String Primary Study Runbook

**Study:** `familiarity-answerability-same-string-gemma2-2b-v1`  
**Run:** `same-string-primary-v1`  
**Model:** `google/gemma-2-2b-it` at the revision pinned in the config  
**Status:** pre-outcome; this procedure is not empirical evidence

This runbook is the ordered execution path for the Same-String Primary Study.
The notebook is a thin Colab launcher for these repository commands:
[`fa_same_string_primary_colab.ipynb`](../notebooks/fa_same_string_primary_colab.ipynb).
R11 remains immutable and `not_evaluable`; none of its screened entities or
protected endpoints are reused here.

## What the study tests

Every unit uses the exact same synthetic target string in four prompts. The
study independently varies whether the target has received unrelated contextual
exposure and whether the requested archive code is actually supplied. The
primary endpoint is the difference-in-differences in answer attempts. It tests
contextual familiarization, not pretrained familiarity or general
hallucination propensity.

## Frozen sizes

`fa-prepare-same-string-matches` must produce exactly these complete units:

| Split | Units | Generated rows |
|---|---:|---:|
| `mechanism_train` | 64 | 256 |
| `locked_validation` | 32 | 128 |
| `behavior_test` | 48 | 192 |
| `probe_test` | 24 | 96 |
| `intervention_test` | 24 | 96 |

Each unit has four rows. Any different count, duplicate ID, split overlap,
target-string change, code leakage, relation leakage, or incomplete four-cell
unit is a hard stop.

## Environments

- Use the local machine for tests, deterministic construction, manifest audits,
  and reporting.
- Use Colab for model generation. Write active artifacts to local POSIX storage
  under `/content/fa-same-string-artifacts`; Google Drive is not treated as an
  atomic filesystem.
- Mirror only complete, verified transaction files to content-addressed archives under
  `/content/drive/MyDrive/fa-same-string-primary-v1/checkpoints`. During long
  model commands the launcher creates a new hash-verified archive every 60
  seconds and retains earlier valid archives. Interrupted or orphaned shard
  files are excluded rather than restored as valid state.
- Read `HF_TOKEN` from the process environment or Colab Secrets. Never place it
  in the notebook, shell history, logs, or repository.
- Check out the exact commit printed in the notebook before installing the
  package. Do not edit code inside the protected run.

## Gate order

The order below is mandatory. A later command may run only when every earlier
gate has passed.

### 1. Prepare direct Same-String matches

Run `fa-prepare-same-string-matches` with all five checked-in candidate
manifests and all five matched synthetic manifests under
`data/fa/confirmatory_source_v5/`.

Verify:

- unit counts are `64/32/48/24/24`;
- all manifests and checksum sidecars verify;
- the output belongs to `same-string-primary-v1`;
- no R11 screening artifact is a dependency.

Expected artifact: a verified Same-String match collection manifest.

### 2. Conduct the blind naturalness audit

Run `fa-prepare-naturalness-ratings` with exactly two independent raters. Give
each rater only the blinded public packet. Do not share the private unblinding
key or the other rater's responses.

Each rater receives their own JSON reference packet and their own CSV
worksheet. The CSV already displays the question, entity type, both candidates,
and neutral example sentences. Raters fill only the six rating columns and set
`independence_attested=true`; they must not edit the displayed stimuli.

After both response files are returned, run
`fa-compile-naturalness-ratings`. If it reports disagreements, issue the
already-generated blinded packet to one independent third adjudicator and run
`fa-finalize-naturalness-adjudication`. Do not resolve disagreements by editing
responses or source rows.

Expected artifacts:

- rating issuance manifest;
- two independent response submissions;
- optional adjudication issuance and submission manifests;
- final compiled naturalness-ratings manifest.

### 3. Pass the unprotected runtime smoke

Use the registered `Qwen/Qwen3-1.7B` smoke configuration. On a new artifact
root, the Colab launcher first creates its screened match input from the
checked-in `candidates_v4.json`, `screening_questions_v4.json`, and
`synthetic_candidates_v5.json`, then runs:

1. `fa-build-pilot`
2. `fa-run-generation --namespace pilot --resume`
3. `fa-score-behavior`

The screening steps are `fa-run-screening` followed by `fa-screen-entities`.
If screening records an infrastructure failure, preserve it and retry with a
new explicit smoke screening shard ID. Later generation uses `--resume` with
the same shard ID after restoring the latest valid Drive checkpoint.

This verifies formatting, execution, checkpointing, and scoring only. A passed
smoke is not empirical evidence for the Same-String hypothesis and cannot
select prompts, thresholds, or claims.

Expected artifact: a verified `pilot_gate` manifest with status `passed`.

### 4. Build and audit the confirmatory index

Run `fa-build-same-string-confirmatory` with the match collection, passed pilot
gate, and compiled naturalness ratings. The command must produce a
`same_string_confirmatory_index` and typed capabilities for every registered
split.

Run `fa-audit-manifest` on the returned index. Confirm that the audit passes and
that the behavior-test capability contains 48 units and 192 prompt rows.

Expected artifacts:

- namespace capability manifests;
- immutable Same-String seal;
- `same_string_confirmatory_index`;
- passed local audit output.

### 5. Open the protected behavior endpoint once

Protected generation begins only after:

- the two-rater naturalness audit has compiled;
- any required adjudication has completed;
- the unprotected pilot gate has passed;
- the local confirmatory manifest audit has passed;
- commit, model, tokenizer, template, config, and artifact hashes are recorded.

Then, exactly once:

1. run `fa-seal-behavior-test` on the audited confirmatory index;
2. run `fa-evaluate-behavior-test` with shard ID
   `same-string-primary-behavior-v1`;
3. checkpoint complete, verified transactions and notebook inputs to Drive.

Do not inspect protected prompt outcomes between sealing and evaluation. Do not
create a replacement endpoint because the result is null, negative, invalid,
or inconvenient.

## Resume after interruption

Restore the latest content-addressed Drive archive to the local artifact root
and use the same repository commit, config, confirmatory index, and shard ID.
The archive hash and every indexed member are checked before restoration, and
the CLI then re-verifies
artifact checksums and lineage before reusing any shard. Notebook state records
only paths and does not replace those checks.

- If generation stopped before endpoint unlock, resume the same command.
- If the endpoint is `unlocked_once`, rerun `fa-evaluate-behavior-test` with
  the same shard ID.
- If it is `evaluated`, rerunning verifies the sealed selection and closes the
  transaction without loading the model.
- Never delete the endpoint state or start a substitute behavior-test run.

## Required public result

Publish the point estimate, crossed-bootstrap interval, complete-unit count,
all four cell rates, output-validity rates, capability-preservation result,
gate decisions, hashes, and any failure state. The typed decision is
`supported`, `not_supported`, or `not_evaluable`; a null or wrong-direction
effect is `not_supported`. Runtime failures are reported separately as
`infrastructure_failure`. The mechanistic pilot remains gated and cannot
rescue the behavioral result.

## Entry points

- [Design](superpowers/specs/2026-08-01-same-string-primary-hybrid-design.md)
- [Amendment](amendments/2026-08-01-fa-same-string-primary.md)
- [Implementation plan](superpowers/plans/2026-08-01-same-string-primary-hybrid-implementation.md)
- [R11 outcome](results/source_v6_r11_instrument_development_outcome.md)
