# Same-String Answerability Causal Replication v2 Runbook

## Purpose

This runbook executes the fresh-unit v2 causal replication on a free Colab GPU.
It uses `configs/familiarity_answerability_causal_replication_v2.json`, keeps all
artifacts under the separate `fa-causal-replication-v2` namespace, and never
modifies or pools the completed v1 artifacts.

The intervention site is locked from v1: layer 18, multiplier 1.0, at
`user_prompt_end`. Validation checks runtime, output format, and preservation;
it cannot select another layer or strength.

## Frozen execution pin

The notebook is pinned to reviewed commit
`26188c9b9105d96446c0ea276fc84be5e444bd0e`. Do not edit this pin during
execution.

The pinned commit contains the reviewed source, v2 config, tests, and
hash-bound preregistration. This runbook and notebook are published in the
following execution-surface commit, but they clone and run only the frozen
research commit. Do not point Colab at an uncommitted working tree or paste
replacement source code into notebook cells.

## Free-Colab procedure

1. Open `notebooks/fa_answerability_causal_replication_v2_colab.ipynb` in a
   fresh GPU runtime.
2. Confirm the displayed pin is
   `26188c9b9105d96446c0ea276fc84be5e444bd0e`.
3. Add `HF_TOKEN` through Colab Secrets. Never paste a token into a cell.
4. Keep Drive checkpoints enabled when possible. The durable root is
   `/content/drive/MyDrive/fa-causal-replication-v2`; the runtime-local fallback
   is `/content/fa-causal-replication-v2`.
5. Run setup and preparation. Preparation must pass the fresh-corpus and
   provenance audits before Gemma is loaded.
6. Load Gemma once and run locked validation. Confirm that the selection
   artifact reports layer 18 and multiplier 1.0.
7. Run the sealed 432-shard schedule. Every receipt is atomic and
   `resume=True`; after interruption, reconnect Drive and rerun the cells in
   order to reuse matching completed receipts.
8. Run final evaluation only after all 432 registered receipts exist.
9. Run the export cell to create and download the complete ZIP archive.

Do not delete partial receipts, change the artifact root, alter the config, or
start a replacement endpoint after protected execution begins. A resumed run
must use the same commit, config, prepare manifest, seal, and artifact root.

## Gates and interpretation

The endpoint evaluates the two registered test splits separately. Missing
receipts, identity/hash mismatches, failed control geometry, or runtime
provenance failures are `not_evaluable`. A complete valid run that misses any
support criterion is `not_supported`. Only a complete run satisfying every
registered criterion on both splits is `causally_supported` within this
controlled Same-String task.

Even `causally_supported` does not establish general metacognition,
hallucination prevention, hidden knowledge, or transfer to larger models. The
study tests a task-local intervention on code-versus-`UNKNOWN` margins.

## Required archive contents

The downloaded ZIP must contain:

- the pre-outcome identity seal and audited v2 corpus;
- the training-only direction bundle;
- runtime smoke, locked validation, and evaluation seal;
- all 432 primary and control receipts, including recorded failures;
- endpoint state, machine-readable result, and Markdown report; and
- hashes binding config, implementation, model, tokenizer, corpus, directions,
  controls, runtime, and requests.
