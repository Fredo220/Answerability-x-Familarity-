# Task 3 Fix Report

## Implementation

- Commit: `abae1d7` (`fix: include duplicate aliases in PopQA components`)
- Components now use every normalized source row: rows are joined by subject and
  by alias before deterministic duplicate-subject and component selection.
- Source-row and discarded-row audit records include normalized aliases and a
  non-null deterministic `alias_component_id`.

## Evidence

### Red

Before the implementation change:

```text
.venv/bin/python -m pytest \
  tests/test_rlmf_data.py::test_select_splits_uses_aliases_from_discarded_duplicate_subject_rows -q

AssertionError: assert 2 == 1
```

The discarded `Q-A` duplicate supplied the only `Bridge` alias to `Q-B`, but
the prior selector retained both subjects as distinct components.

### Green

```text
.venv/bin/python -m pytest tests/test_rlmf_data.py -q
5 passed in 2.82s

.venv/bin/python -m pytest tests/test_rlmf_data.py tests/test_rlmf_cli.py -q
6 passed in 2.86s
```

The fixture snapshot also asserts that every source row and every discarded row
has a non-null `alias_component_id`.

### Full Suite

```text
.venv/bin/python -m pytest -q
226 passed in 13.31s
```

## Remaining Risks

- The real pinned PopQA download was not run in this network-free regression
  pass. Its full alias graph may have fewer than 896 eligible components; the
  existing strict failure behavior intentionally remains in place.
- The audit artifacts now contain additional provenance fields. Consumers that
  enforce a closed JSON schema will need to allow those additive fields.
