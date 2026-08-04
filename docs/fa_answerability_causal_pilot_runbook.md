# Same-String Answerability Causal Pilot Runbook

## Status

This workflow is software-verified but has not yet produced a causal result.
The completed behavioral and representation studies remain closed and
unchanged. A result exists only after all registered shards from both fresh
test splits and every control have passed the sealed one-use evaluation.

## What the pilot tests

The pilot learns an answerability direction from the existing v3
`representation_train` activations. On fresh Same-String prompts it tests
whether adding that direction to an unanswerable prompt raises the model's
`code` versus `UNKNOWN` margin, and whether subtracting it from an answerable
prompt lowers the same margin.

This is a task-local causal test. It is not evidence of general metacognition,
hallucination prevention, familiarity effects, or transfer to larger models.

## Registered schedule

- 12 validation units select one layer and strength from the complete 15-cell
  grid.
- 18 unseen-entity units and 18 unseen-template units are tested separately.
- Every test unit runs baseline, primary, sign-reversed, label-shuffled,
  no-intervention, wrong-anchor, wrong-layer, and five norm-matched random
  controls: 432 atomic unit receipts in total.
- Test results are not pooled to rescue a failed split.

## Recommended free-Colab path

1. Open `notebooks/fa_answerability_causal_pilot_colab.ipynb` in a fresh GPU
   runtime.
2. Add `HF_TOKEN` through Colab Secrets. Never paste it into a cell.
3. Optionally enable Google Drive. The artifact directory itself is the
   checkpoint; every unit receipt is written atomically.
4. Run the setup and preparation cells.
5. Run validation once. It seals the selected layer, strength, runtime,
   directions, controls, and fresh test identities.
6. Run the shard cell. After interruption, rerun the notebook with the same
   artifact directory; existing same-hash receipts are reused without model
   work.
7. Run evaluation once only after all 432 receipts exist.

The notebook keeps one 4-bit Gemma instance in memory across all shards. Do
not replace the loop with one shell process per shard, because that would load
the model hundreds of times.

## CLI surfaces

The same operations are available individually:

```bash
feature-dynamics fa-causal-prepare --help
feature-dynamics fa-causal-run-validation --help
feature-dynamics fa-causal-run-shard --help
feature-dynamics fa-causal-evaluate --help
```

`fa-causal-run-shard --resume` accepts only a receipt with the same request
hash. `fa-causal-evaluate` checks the complete sealed schedule before
creating endpoint state. Partial evidence cannot be evaluated.

## Required published artifacts

- pre-outcome identity seal;
- frozen causal corpus and training-only direction bundle;
- validation candidate receipts, selection, and evaluation seal;
- every primary and control shard, including failures;
- runtime and memory receipt;
- one-use result JSON and Markdown report; and
- an explicit claim boundary and resource accounting.

If the live run cannot be completed, publish only the execution-ready package
and label it `software_verified_live_run_pending`.
