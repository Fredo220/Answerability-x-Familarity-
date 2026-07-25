# Source-v6 Development Runbook

Source-v6 is open instrument development. It cannot test H1-H8 and cannot
rescue Source-v5.

## 1. Environment

Use Python 3.11 or 3.12:

```bash
python -m pip install -e ".[test,quality]"
```

For tokenizer/model access, set `HF_TOKEN` in the runtime environment. Never
write the token into this repository, a notebook cell, an artifact, or a log.

## 2. Build the Open Source Frame

Construct a QLever/Wikidata frame using the pinned source-frame command:

```bash
PYTHONPATH=src python tools/build_fa_development_frame.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --output-dir data/fa/development_source_v6_r6/frame \
  --query-limit 4000 \
  --place-query-limit 6000 \
  --required-per-domain 48 \
  --retrieval-date 2026-07-25 \
  --seed-entity-cache data/fa/development_source_v6_r5/frame/source_cache/entity_records_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_mechanism_train_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_locked_validation_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_behavior_test_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_probe_test_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_intervention_test_v1.json \
  --exclude-candidates data/fa/development_source_v6/candidate_entities_instrument_development_v1.json \
  --exclude-candidates data/fa/development_source_v6/candidate_entities_construction_validation_v1.json \
  --exclude-candidates data/fa/development_source_v6_r2/candidate_entities_instrument_development_v1.json \
  --exclude-candidates data/fa/development_source_v6_r2/candidate_entities_construction_validation_v1.json \
  --exclude-candidates data/fa/development_source_v6_r4/candidate_entities_instrument_development_v1.json \
  --exclude-candidates data/fa/development_source_v6_r4/candidate_entities_construction_validation_v1.json \
  --exclude-candidates data/fa/development_source_v6_r4/pre_model_semantic_exclusions_v1.json \
  --exclude-candidates data/fa/development_source_v6_r5/candidate_entities_instrument_development_v1.json \
  --exclude-candidates data/fa/development_source_v6_r5/candidate_entities_construction_validation_v1.json
```

The command fails closed if any domain has fewer than 48 complete
tokenizer-compatible pseudonym reserves. Changing the query limit, relations,
aliases, or wording creates a new instrument revision.

R6 keeps the R5 query limits and corrects only the pre-model semantic defects
registered in the R6 amendment. The R5 entity cache is a hash-bound fetch cache;
all R6 records are recomputed under the R6 code. All inspected R5 entities are
excluded.

## 3. Materialize Development Splits

```bash
PYTHONPATH=src python tools/build_fa_development_source.py materialize \
  --source-frame data/fa/development_source_v6_r6/frame/development_source_frame_v1.json \
  --output-dir data/fa/development_source_v6_r6 \
  --candidates-per-domain-per-split 24 \
  --split-seed 20260725 \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_mechanism_train_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_locked_validation_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_behavior_test_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_probe_test_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_intervention_test_v1.json \
  --exclude-candidates data/fa/development_source_v6/candidate_entities_instrument_development_v1.json \
  --exclude-candidates data/fa/development_source_v6/candidate_entities_construction_validation_v1.json \
  --exclude-candidates data/fa/development_source_v6_r2/candidate_entities_instrument_development_v1.json \
  --exclude-candidates data/fa/development_source_v6_r2/candidate_entities_construction_validation_v1.json \
  --exclude-candidates data/fa/development_source_v6_r4/candidate_entities_instrument_development_v1.json \
  --exclude-candidates data/fa/development_source_v6_r4/candidate_entities_construction_validation_v1.json \
  --exclude-candidates data/fa/development_source_v6_r4/pre_model_semantic_exclusions_v1.json \
  --exclude-candidates data/fa/development_source_v6_r5/candidate_entities_instrument_development_v1.json \
  --exclude-candidates data/fa/development_source_v6_r5/candidate_entities_construction_validation_v1.json
```

Do not rename either split to a protected confirmatory namespace.

## 4. Screen `instrument_development`

First require the machine-readable independent audit:

```text
data/fa/development_source_v6_r6/pre_model_semantic_audit_v1.json
```

It must bind the exact source-integrity SHA, report zero blockers, and be
committed with the corpus before model execution. Run Gemma from a clean,
detached checkout of the exact committed bundle. Keep the live output under
`/content`; only the content-addressed mirror belongs on Drive:

```bash
COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"

PYTHONPATH=src /content/fa-venv/bin/python \
  tools/run_fa_development_screening.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --source-root data/fa/development_source_v6_r6 \
  --split instrument_development \
  --output-root /content/fa-r6-artifacts \
  --checkpoint-root /content/drive/MyDrive/fa-r6-checkpoints \
  --batch-size 16 \
  --success-criteria configs/fa_source_v6_r6_success_criteria.json \
  --pre-model-semantic-audit \
    data/fa/development_source_v6_r6/pre_model_semantic_audit_v1.json \
  --git-commit "$COMMIT"
```

The checkpoint mirror snapshots each completed batch and final result to Drive
without hardlinks. A resume verifies the Git commit, criteria hash, semantic
audit hash, configuration hash, source-integrity hash, model and tokenizer
revisions, chat-template hash, parser hash, archive hash, and every member hash.

Inspect only the development split. Record every instrument revision and retain
failed revisions.

## 5. Manual Error Audit

Create the deterministic domain-balanced audit packet from
`instrument_development`. It contains at most four model-scored errors and two
model-scored successes per domain. The model score and sampling stratum are
hidden from raters. Two independent humans label every row using the registered
taxonomy, including `no_error`; a third independent human adjudicates only
disagreements. AI ratings are not human evidence. A completed independent audit
is required before the instrument can be frozen or `construction_validation`
opened.

R6 fixes the packet seed at `20260725`; if a domain has fewer errors, every
error in that domain is audited. Human and model correctness must agree on every
sampled row.
Any final `ambiguous_ground_truth`, `incomplete_alias_set`,
`wrong_granularity`, `parser_failure`, `source_error`, or `other` label fails
the instrument. These rules are hash-bound in the development execution
identity and cannot be replaced at freeze time.

## 6. Freeze the Instrument

Before opening `construction_validation`, freeze:

- relations and question templates;
- aliases and normalization;
- query/rank frame and raw pool sizes;
- matchability and pseudonym policy;
- the two-of-three qualification threshold;
- validation sample and success criteria.

The protocol and criteria are committed before development screening. The
generated freeze manifest is an immutable run artifact, not a new repository
commit: committing it would change `HEAD` and invalidate its binding to the
screening commit. Generate and retain it outside the clean checkout, record its
SHA-256 with the run artifacts, and use the same clean screening commit for
construction validation. Do not choose criteria after seeing validation
results.

Write the criteria as a JSON object, then seal the instrument:

```bash
PYTHONPATH=src python tools/freeze_fa_development_instrument.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --source-root data/fa/development_source_v6_r6 \
  --development-run-dir /content/fa-r6-artifacts/instrument_development/<execution-identity-sha256> \
  --success-criteria configs/fa_source_v6_r6_success_criteria.json \
  --manual-audit-manifest /content/fa-r6-audit/compiled_audit_v1.json \
  --output /content/fa-r6-freeze/instrument_freeze_v1.json \
  --git-commit <screening-code-commit>
```

## 7. Open Construction Validation Once

```bash
PYTHONPATH=src python tools/run_fa_development_screening.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --source-root data/fa/development_source_v6_r6 \
  --split construction_validation \
  --output-root /content/fa-r6-artifacts \
  --freeze-manifest /content/fa-r6-freeze/instrument_freeze_v1.json \
  --batch-size 16
```

If the frozen success criteria fail, report the revision as failed. Do not tune
against the same validation split.

## 8. Source-v7

Only after Source-v6 validation, instantiate and freeze the Source-v7
preregistration from `docs/source_v7_preregistration_template.md`. Source-v7
must exclude every Source-v5 and Source-v6 QID before screening.

## Verification

```bash
ruff check src/trajectory_extractor/fa_development_*.py tests/test_fa_development_*.py
pyright
pytest -q \
  tests/test_fa_development_source.py \
  tests/test_fa_development_frame.py \
  tests/test_fa_development_screening.py \
  tests/test_fa_entities.py \
  tests/test_fa_confirmatory_source.py \
  tests/test_fa_confirmatory_synthetics.py
tools/build_fa_graph.sh --force
```
