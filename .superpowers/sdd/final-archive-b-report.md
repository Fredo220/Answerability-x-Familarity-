# Final Archive B

Date: 2026-07-12

## Scope

Documentation-only archive audit. This work did not modify `src/`, `tests/`, or
the parent `runs/concept-main/secondary` artifact directory.

## Findings

1. Fresh transfer commands derive a dataset filename from the replication ID:
   `TRANSFER_DATA="data/external/${RUN_ID}-real-transfer.jsonl"`. Preparation
   writes to that path, extraction consumes that path, and transfer and
   intervention retain their separate derived run IDs.
2. `docs/legacy_secondary_sha256.txt` tracks SHA-256 values for all twelve
   frozen legacy secondary files: contrastive vectors, vector dynamics,
   predictions, metrics, method-comparison figures, and validation risk-gap
   figures for both `exact_error` and `binding_error`.
3. `docs/results.md` links the manifest and distinguishes audit attribution from
   embedded provenance: commit `04568b9f1c1629ac7f08323b1c0602843fe91f48` comes
   from the execution audit, while the legacy files contain no modern provenance
   or completion seal. The manifest is post-hoc archival identification only.
4. The registered negative finding and late-prefix boundary are unchanged.

## Verification

- From `/Users/friedrichreichelt/Documents/Machanistic Interpretability/runs/concept-main/secondary`:
  `shasum -a 256 -c "/Users/friedrichreichelt/Documents/Machanistic Interpretability/.worktrees/metacognitive-feature-flow/docs/legacy_secondary_sha256.txt"`
  returned `OK` for all 12 artifacts.
- `git diff --check` completed successfully.
- `.venv/bin/python -m pytest -q` completed successfully: `140 passed in
  13.40s`.

## Files Owned By This Audit

- `README.md`
- `docs/results.md`
- `docs/execution_plan.md`
- `docs/legacy_secondary_sha256.txt`
- `.superpowers/sdd/final-archive-b-report.md`
