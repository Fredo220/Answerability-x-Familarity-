# Task 1 Report: Freeze The New Study And Bind Upstream Provenance

## Implementation summary

- Added a machine-checkable RLMF preregistration contract and a low-compute deviation register. Study 0 remains untouched.
- Pinned the RLMF reference repository to `a087e7a1e49f52aaa701add19cd80699b709fdef` and vendored the exact trainer, rewards, sample config, MIT license, and extracted metacognition prompt.
- Pinned TRL `v0.23.0` and vendored the exact `GRPOTrainer` source. Both manifests record retrieval date, source location, and SHA-256 digests.
- Added exact Colab requirements, the `rlmf-local` optional dependency group while retaining `test`, and RLMF/Colab local-state ignores.

## RED evidence

Command:

```bash
.venv/bin/python -m pytest tests/test_rlmf_preregistration.py -q
```

Output:

```text
FFFF
4 failed in 0.10s
```

The failures were the expected `FileNotFoundError` values for `docs/rlmf_preregistration.md`, `third_party/rlmf/UPSTREAM.json`, and `third_party/trl/UPSTREAM.json`.

## GREEN evidence

Command:

```bash
.venv/bin/python -m pytest tests/test_rlmf_preregistration.py -q
```

Output:

```text
....
4 passed in 0.01s
```

Command:

```bash
.venv/bin/python -m pytest -q
```

Output:

```text
144 passed in 14.58s
```

Vendor hash verification output:

```text
rlmf_trainer.py      d608b198324407f949c07d7f693680951ec62edf6962036ac1afe896f112cfeb
rewards.py           92302296a23ebde6bf37fb765d8fd5e69973c67595f18bd2e90c11006e000d44
sample_config.py     91fa6385bdf1298002b5d3eac1883f856155dc781b836fc1a531438a37cb620c
metacognition_prompt 88ae37b2a730c4fce8687afd53210147afd2a4341e9e1a39dfe962945b3fa360
grpo_trainer.py      4658c3920040f256bc909a3c49263409441f4425c8ba6273585370cf6b025777
```

Required command:

```bash
git diff --check
```

Output: no output; exit code 0.

## Files changed

- `.gitignore`
- `pyproject.toml`
- `docs/rlmf_preregistration.md`
- `docs/rlmf_low_compute_deviations.md`
- `docs/superpowers/plans/2026-07-13-rlmf-mechanistic-replication.md` (pre-existing expected untracked plan, staged as required)
- `requirements-rlmf-colab.txt`
- `third_party/rlmf/LICENSE`
- `third_party/rlmf/UPSTREAM.json`
- `third_party/rlmf/rlmf_trainer.py`
- `third_party/rlmf/rewards.py`
- `third_party/rlmf/sample_config.py`
- `third_party/rlmf/metacognition_prompt.txt`
- `third_party/trl/UPSTREAM.json`
- `third_party/trl/grpo_trainer.py`
- `tests/test_rlmf_preregistration.py`

## Self-review

- The RLMF file hashes match the values required by the task brief, and the copied license and TRL source were independently hashed after vendoring.
- The metacognition prompt is the exact decoded `judgment_prompt` template at line 16 of the pinned upstream `dynamic_score_samples.py`; its source location and digest are recorded in the manifest.
- The preregistration locks the finite-three-seed estimand, Study 1/2 gates, `pre_confidence` primary Study 2 anchor, test-MAE incremental-dynamics metric, and no-test-driven-change rule.
- No existing Study 0 source, registration, result, artifact, or namespace was changed.

## Concerns

- `git diff --cached --check` reports trailing whitespace already present in the exact upstream RLMF files. Those bytes are intentionally retained because trimming them would invalidate the required SHA-256 provenance. The required `git diff --check` command exits 0.
- The required runtime refusal when the installed `GRPOTrainer` hash diverges from the vendored TRL manifest belongs to the later local trainer implementation. Task 1 only establishes the reference source and manifest, and its file-ownership list does not include a runtime trainer module.

## Review follow-up

- Added narrowly scoped `.gitattributes` rules for the exact vendored RLMF files whose upstream bytes contain trailing whitespace. The rules disable only `blank-at-eol` for `rlmf_trainer.py` and `rewards.py`, and both `blank-at-eol` and `blank-at-eof` for `sample_config.py`; no global or non-vendored whitespace checks are disabled.
- Extended `tests/test_rlmf_preregistration.py` to hash `third_party/rlmf/LICENSE` and compare it with `license_sha256` from `third_party/rlmf/UPSTREAM.json`.

Verification:

```text
$ .venv/bin/python -m pytest tests/test_rlmf_preregistration.py -q
....                                                                     [100%]
4 passed in 0.01s

$ .venv/bin/python -m pytest -q
........................................................................ [ 50%]
........................................................................ [100%]
144 passed in 13.21s

$ git diff --check 439594b
<no output>
exit=0
```
