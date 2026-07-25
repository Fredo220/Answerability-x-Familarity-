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
  --output-dir data/fa/development_source_v6/frame \
  --query-limit 4000 \
  --required-per-domain 48 \
  --retrieval-date 2026-07-25 \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_mechanism_train_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_locked_validation_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_behavior_test_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_probe_test_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_intervention_test_v1.json
```

The command fails closed if any domain has fewer than 48 complete
tokenizer-compatible pseudonym reserves. Changing the query limit, relations,
aliases, or wording creates a new instrument revision.

## 3. Materialize Development Splits

```bash
PYTHONPATH=src python tools/build_fa_development_source.py materialize \
  --source-frame data/fa/development_source_v6/frame/development_source_frame_v1.json \
  --output-dir data/fa/development_source_v6 \
  --candidates-per-domain-per-split 24 \
  --split-seed 20260725 \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_mechanism_train_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_locked_validation_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_behavior_test_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_probe_test_v1.json \
  --exclude-candidates data/fa/confirmatory_source_v5/candidate_entities_intervention_test_v1.json
```

Do not rename either split to a protected confirmatory namespace.

## 4. Screen `instrument_development`

Run Gemma on Colab using the development-only screening command:

```bash
PYTHONPATH=src python tools/run_fa_development_screening.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --source-root data/fa/development_source_v6 \
  --split instrument_development \
  --output-root artifacts/fa/development_source_v6 \
  --batch-size 16
```

Copy immutable checkpoints to Drive after each completed batch. A resume must
match the Git commit, configuration hash, source-integrity hash, model revision,
tokenizer revision, and chat-template hash.

Inspect only the development split. Record every instrument revision and retain
failed revisions.

## 5. Freeze the Instrument

Before opening `construction_validation`, freeze:

- relations and question templates;
- aliases and normalization;
- query/rank frame and raw pool sizes;
- matchability and pseudonym policy;
- the two-of-three qualification threshold;
- validation sample and success criteria.

Commit the frozen protocol and hashes. Do not choose criteria after seeing the
validation results.

Write the criteria as a JSON object, then seal the instrument:

```bash
PYTHONPATH=src python tools/freeze_fa_development_instrument.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --source-root data/fa/development_source_v6 \
  --success-criteria configs/fa_source_v6_success_criteria.json \
  --output data/fa/development_source_v6/instrument_freeze_v1.json
```

## 6. Open Construction Validation Once

```bash
PYTHONPATH=src python tools/run_fa_development_screening.py \
  --config configs/familiarity_answerability_gemma2_2b.json \
  --source-root data/fa/development_source_v6 \
  --split construction_validation \
  --output-root artifacts/fa/development_source_v6 \
  --freeze-manifest data/fa/development_source_v6/instrument_freeze_v1.json \
  --batch-size 16
```

If the frozen success criteria fail, report the revision as failed. Do not tune
against the same validation split.

## 7. Manual Error Audit

Create the deterministic domain-balanced error packet. Two independent humans
label every row using the registered taxonomy. A third independent human
adjudicates only disagreements. AI ratings are not human evidence.

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
