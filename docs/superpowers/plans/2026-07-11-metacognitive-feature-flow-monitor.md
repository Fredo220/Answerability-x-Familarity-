# Metacognitive Feature-Flow Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a CPU-friendly internal reliability score that tests whether causal layerwise and tokenwise dynamics of a contrastive risk direction predict concept-mixing failures beyond the same direction used statically.

**Architecture:** Fit one centered contrastive direction per residual-stream layer on training examples only, standardize its tokenwise projections with training-only statistics, and derive causal layer and token differences. Feed static projections, their dynamics, and an optional stable-class operator residual into the existing causal-prefix evaluator; freeze prefix and threshold choices on validation, evaluate test once, and persist all secondary artifacts outside the frozen primary namespace. This plan establishes the monitoring signal only; activation capping, jailbreak transfer, and multi-model Colab replication are separate plans after this detector is validated.

**Tech Stack:** Python 3.12, NumPy, scikit-learn, PyTorch model artifacts already persisted by the repository, Matplotlib, pytest, Git.

## Global Constraints

- Local target: `meta-llama/Llama-3.2-1B-Instruct`.
- Local hardware: CPU, 8 GB RAM, batch size 1.
- Existing concept data and model artifacts remain valid and are never rewritten to accommodate a new method.
- The existing primary preregistration, primary method selection, and primary acceptance thresholds remain frozen.
- Register this secondary detector before computing any new secondary test metrics.
- At answer position `t`, a score may use only emitted answer tokens before `t` and layer states up to its selected layer.
- Fit directions, centers, standardization statistics, operators, prefix choices, and thresholds on train/validation only.
- Fit the operator baseline only on label-0 training examples.
- Call the result a `metacognitive internal reliability signal` in scientific text; `artificial intuition` is an explanatory metaphor only.
- A secondary detection claim requires at least 20 positive held-out examples spanning at least 10 independent entity families.
- The confirmatory comparison is `contrastive_vector` versus `contrastive_plus_dynamics`; operator residuals appear only in the exploratory `full_metacognitive_monitor`.
- Require secondary AUROC gain of at least `0.03`, a paired cluster-bootstrap 95% interval above zero, and Benjamini-Hochberg adjusted `p < 0.05` before calling the detector supported.
- Never generate or publish new jailbreak recipes in this plan.
- Do not add online gradients, an ODE solver, matrix exponentials, a new SAE, or live activation steering in this plan.
- Compute every score on the existing causal replay artifacts; no secondary method may trigger model re-extraction.

---

## File Map

- Modify `.gitignore`: ignore macOS and interrupted atomic-write files.
- Create `docs/secondary_preregistration.md`: date and freeze the secondary detector hypothesis, comparison, endpoint rule, and decision threshold.
- Create `src/trajectory_extractor/contrastive_directions.py`: train-only centered contrastive directions and projection scores.
- Create `src/trajectory_extractor/vector_dynamics.py`: train-standardized static scores plus causal layer/token differences.
- Create `src/trajectory_extractor/secondary_study.py`: held-out evaluation, cluster bootstrap, endpoint eligibility, and BH correction.
- Create `src/trajectory_extractor/secondary_artifacts.py`: atomic persistence under `runs/<run>/secondary/`.
- Modify `src/trajectory_extractor/cli.py`: add `evaluate-secondary-concept` without changing `evaluate-concept`.
- Modify `src/trajectory_extractor/plotting.py`: add a class-conditioned metacognitive risk-gap heatmap.
- Modify `README.md`, `docs/execution_plan.md`, and `docs/references.md`: document the command, scientific framing, and literature boundary.
- Create focused tests in `tests/test_contrastive_directions.py`, `tests/test_vector_dynamics.py`, `tests/test_secondary_study.py`, `tests/test_secondary_artifacts.py`, and `tests/test_secondary_cli.py`.

### Task 1: Freeze The Secondary Study And Establish A Versioned Baseline

**Files:**
- Modify: `.gitignore`
- Create: `docs/secondary_preregistration.md`
- Include in baseline commit: `README.md`, `configs/`, `data/`, `docs/`, `notebooks/`, `pyproject.toml`, `src/`, `tests/`

**Interfaces:**
- Consumes: the approved design in `docs/superpowers/specs/2026-07-11-dual-track-feature-dynamics-design.md`.
- Produces: a dated secondary registration and a clean Git baseline from which extraction and analysis are reproducible.

- [ ] **Step 1: Extend the ignore rules before staging the existing project**

Append exactly these lines to `.gitignore`:

```gitignore
.DS_Store
*.tmp
```

- [ ] **Step 2: Write the dated secondary registration**

Create `docs/secondary_preregistration.md` with this content:

```markdown
# Secondary Registration: Metacognitive Feature-Flow Monitor

Date frozen: 2026-07-11

## Status

This is a prospective secondary analysis registered after the primary protocol
and before any secondary test metric is computed. It does not modify or replace
the frozen primary comparison in `docs/preregistration.md`.

## Hypothesis

A causal internal reliability score built from the layerwise and tokenwise
evolution of a contrastive error direction predicts held-out concept-mixing
errors better than the same contrastive direction used as a static score.
"Artificial intuition" is an explanatory metaphor; the measured object is a
metacognitive internal reliability signal, not consciousness or ground-truth
access.

## Endpoint And Splits

- Primary secondary endpoint: `exact_error`.
- Mechanistic diagnostic endpoint: `binding_error`.
- Direction, center, and standardization statistics: training split only.
- Prefix and threshold selection: validation split only.
- Test evaluation: once after all choices are frozen.
- Cluster unit: `entity_family`.

The endpoint is confirmatory only when the test split has at least 20 positive
examples spanning at least 10 independent entity families. Otherwise the result
is `not_evaluable` and remains descriptive.

## Registered Methods

1. `contrastive_vector`: centered projection onto the normalized training-only
   risk-minus-control activation direction at each layer, standardized with the
   same training-only layer statistics used by method 2.
2. `contrastive_plus_dynamics`: the static projection plus its train-standardized
   layerwise and causal tokenwise first differences.
3. `full_metacognitive_monitor`: method 2 plus output uncertainty, raw hidden-state
   dynamics, and a PCA-ridge operator residual fitted on label-0 training examples.
   This third method is exploratory and cannot replace the registered comparison.

## Registered Comparison

The confirmatory contrast is:

`contrastive_plus_dynamics - contrastive_vector`

measured as paired test AUROC difference with an entity-family cluster bootstrap.
The claim is supported only if all conditions hold:

- AUROC difference is at least 0.03;
- the paired 95% bootstrap interval excludes zero on the positive side;
- Benjamini-Hochberg adjusted p-value is below 0.05;
- endpoint eligibility requirements are satisfied.

The within-track secondary family has two preregistered hypotheses: this
detection comparison and the later capping-versus-triggered-steering comparison.
Until the intervention p-value exists, reserve its slot with p=1.0. This yields
a conservative adjusted detection p-value; the final intervention plan recomputes
Benjamini-Hochberg across both observed p-values.

Report AUPRC, expected calibration error, false-positive rate, selected causal
token/layer prefix, threshold, and earliest positive threshold crossing even when
the claim is unsupported.

## Falsification And Leakage Controls

- No test example may fit a direction, center, scale, operator, prefix, or threshold.
- Variable response length may not reweight an example during direction fitting.
- Changing a future token activation may not change an earlier token score.
- The operator reference class is label 0 only.
- Existing primary artifacts and metrics are read-only.
- Negative and null effects are reported without changing the dataset or threshold.
```

- [ ] **Step 3: Verify the pre-change test baseline**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: `64 passed` and exit code `0`.

- [ ] **Step 4: Stage the existing project without `.DS_Store` or runtime artifacts**

Run:

```bash
git add .gitignore README.md configs data docs notebooks pyproject.toml src tests
git status --short
```

Expected: project sources and documents are staged, while `.DS_Store`, `.venv/`, `.pytest_cache/`, and `runs/` are absent.

- [ ] **Step 5: Commit the reproducible baseline**

Run:

```bash
git commit -m "chore: establish reproducible research baseline"
```

Expected: one commit containing the previously untracked project plus the secondary registration; `git status --short` is empty.

### Task 2: Fit Train-Only Contrastive Directions

**Files:**
- Create: `src/trajectory_extractor/contrastive_directions.py`
- Test: `tests/test_contrastive_directions.py`

**Interfaces:**
- Consumes: `TrajectoryBatch` with hidden states shaped `[example, token, layer, hidden_dim]` and train indices.
- Produces: `LayerwiseContrastiveDirection.fit(batch, train_indices)`, `transform(batch) -> np.ndarray [N,T,L]`, and fitted `directions`, `centers`, and `fit_example_ids` properties.

- [ ] **Step 1: Write failing tests for sign, normalization, equal example weighting, and leakage rejection**

Create `tests/test_contrastive_directions.py`:

```python
import numpy as np
import pytest

from trajectory_extractor.contrastive_directions import LayerwiseContrastiveDirection
from trajectory_extractor.types import TrajectoryBatch


def make_batch() -> TrajectoryBatch:
    hidden = np.zeros((6, 3, 2, 4), dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 0, 1])
    splits = np.array(["train", "train", "train", "train", "test", "test"])
    mask = np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=bool,
    )
    hidden[labels == 1, :, :, 0] = 3.0
    hidden[labels == 0, :, :, 0] = -1.0
    return TrajectoryBatch(
        example_ids=tuple(f"e{index}" for index in range(6)),
        labels=labels,
        splits=splits,
        hidden_states=hidden,
        token_mask=mask,
        token_logprobs=np.zeros((6, 3), dtype=np.float32),
        token_entropies=np.zeros((6, 3), dtype=np.float32),
    )


def test_direction_is_unit_length_and_risk_projection_is_larger():
    batch = make_batch()
    model = LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 3]))

    scores = model.transform(batch)

    np.testing.assert_allclose(np.linalg.norm(model.directions, axis=1), 1.0)
    assert scores[2, 0, 0] > scores[0, 0, 0]
    assert model.fit_example_ids == ("e0", "e1", "e2", "e3")
    assert np.all(scores[~batch.token_mask] == 0.0)


def test_each_example_is_pooled_before_class_means_are_computed():
    batch = make_batch()
    model = LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 3]))

    np.testing.assert_allclose(model.centers[:, 0], 1.0)


def test_fit_rejects_test_examples_and_missing_classes():
    batch = make_batch()

    with pytest.raises(ValueError, match="training split only"):
        LayerwiseContrastiveDirection().fit(batch, np.array([0, 1, 2, 5]))
    with pytest.raises(ValueError, match="both classes"):
        LayerwiseContrastiveDirection().fit(batch, np.array([0, 1]))
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_contrastive_directions.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'trajectory_extractor.contrastive_directions'`.

- [ ] **Step 3: Implement the smallest train-only direction model that satisfies the contract**

Create `src/trajectory_extractor/contrastive_directions.py`:

```python
from __future__ import annotations

import numpy as np

from trajectory_extractor.types import TrajectoryBatch


class LayerwiseContrastiveDirection:
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.directions = np.empty((0, 0), dtype=np.float32)
        self.centers = np.empty((0, 0), dtype=np.float32)
        self.fit_example_ids: tuple[str, ...] = ()

    def fit(
        self,
        batch: TrajectoryBatch,
        train_indices: np.ndarray,
    ) -> "LayerwiseContrastiveDirection":
        indices = _validated_train_indices(batch, train_indices)
        labels = batch.labels[indices]
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("contrastive direction requires both classes")

        pooled = _pool_valid_tokens_per_example(batch)
        selected = pooled[indices]
        risk_mean = selected[labels == 1].mean(axis=0)
        control_mean = selected[labels == 0].mean(axis=0)
        raw = risk_mean - control_mean
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        if np.any(norms <= self.epsilon):
            raise ValueError("contrastive direction has a near-zero layer")

        self.directions = (raw / norms).astype(np.float32)
        self.centers = selected.mean(axis=0).astype(np.float32)
        self.fit_example_ids = tuple(batch.example_ids[index] for index in indices)
        return self

    def transform(self, batch: TrajectoryBatch) -> np.ndarray:
        if not self.fit_example_ids:
            raise RuntimeError("contrastive direction must be fitted before transform")
        expected = batch.hidden_states.shape[2:]
        if self.directions.shape != expected or self.centers.shape != expected:
            raise ValueError("batch layer and hidden dimensions do not match fitted direction")
        scores = np.zeros(batch.hidden_states.shape[:3], dtype=np.float32)
        for layer in range(batch.hidden_states.shape[2]):
            current = batch.hidden_states[:, :, layer, :].astype(np.float32)
            current -= self.centers[layer][None, None, :]
            scores[:, :, layer] = np.einsum(
                "ntd,d->nt",
                current,
                self.directions[layer],
                optimize=True,
            )
        scores[~batch.token_mask] = 0.0
        return scores


def _pool_valid_tokens_per_example(batch: TrajectoryBatch) -> np.ndarray:
    counts = batch.token_mask.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("every example needs at least one valid token")
    n_examples, _, n_layers, hidden_dim = batch.hidden_states.shape
    pooled = np.empty((n_examples, n_layers, hidden_dim), dtype=np.float32)
    for layer in range(n_layers):
        current = batch.hidden_states[:, :, layer, :].astype(np.float32)
        current *= batch.token_mask[:, :, None]
        pooled[:, layer, :] = current.sum(axis=1) / counts[:, None]
    return pooled


def _validated_train_indices(batch: TrajectoryBatch, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("train_indices must be a non-empty vector")
    if np.unique(values).size != values.size:
        raise ValueError("train_indices must not contain duplicates")
    if values.min() < 0 or values.max() >= len(batch.example_ids):
        raise ValueError("train_indices are out of range")
    if np.any(batch.splits[values] != "train"):
        raise ValueError("contrastive directions may use the training split only")
    selected_ids = [batch.example_ids[index] for index in values]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("fit example IDs must be unique")
    return values
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_contrastive_directions.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit the direction model**

Run:

```bash
git add src/trajectory_extractor/contrastive_directions.py tests/test_contrastive_directions.py
git commit -m "feat: add train-only contrastive directions"
```

Expected: commit succeeds and the focused tests remain green.

### Task 3: Derive Causal Layer And Token Dynamics

**Files:**
- Create: `src/trajectory_extractor/vector_dynamics.py`
- Test: `tests/test_vector_dynamics.py`

**Interfaces:**
- Consumes: `TrajectoryBatch`, projection scores `[N,T,L]`, and train indices.
- Produces: `StandardizedVectorDynamics.fit(batch, scores, train_indices)`, `StandardizedVectorDynamics.transform(batch, scores) -> VectorDynamics`, and `VectorDynamics.as_feature_tensor(include_static=True) -> [N,T,L,F]`.

- [ ] **Step 1: Write failing tests for training-only scale, masking, shape, and causal token dependence**

Create `tests/test_vector_dynamics.py`:

```python
import numpy as np
import pytest

from trajectory_extractor.types import TrajectoryBatch
from trajectory_extractor.vector_dynamics import StandardizedVectorDynamics


def make_batch() -> TrajectoryBatch:
    hidden = np.zeros((4, 3, 3, 2), dtype=np.float32)
    return TrajectoryBatch(
        example_ids=("a", "b", "c", "d"),
        labels=np.array([0, 1, 0, 1]),
        splits=np.array(["train", "train", "val", "test"]),
        hidden_states=hidden,
        token_mask=np.array(
            [[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
            dtype=bool,
        ),
        token_logprobs=np.zeros((4, 3), dtype=np.float32),
        token_entropies=np.zeros((4, 3), dtype=np.float32),
    )


def test_standardized_dynamics_have_expected_values_and_masking():
    batch = make_batch()
    scores = np.array(
        [
            [[0, 1, 3], [1, 2, 4], [0, 0, 0]],
            [[2, 3, 5], [3, 4, 6], [4, 5, 7]],
            [[1, 2, 4], [2, 3, 5], [3, 4, 6]],
            [[8, 9, 11], [9, 10, 12], [10, 11, 13]],
        ],
        dtype=np.float32,
    )
    model = StandardizedVectorDynamics().fit(batch, scores, np.array([0, 1]))

    result = model.transform(batch, scores)

    assert result.static.shape == (4, 3, 3)
    assert result.layer_delta.shape == (4, 3, 3)
    assert result.token_delta.shape == (4, 3, 3)
    assert result.as_feature_tensor().shape == (4, 3, 3, 3)
    assert np.all(result.static[~batch.token_mask] == 0.0)
    assert np.all(result.token_delta[:, 0] == 0.0)
    assert np.all(result.layer_delta[:, :, 0] == 0.0)
    assert model.fit_example_ids == ("a", "b")


def test_future_tokens_cannot_change_an_earlier_score():
    batch = make_batch()
    scores = np.arange(36, dtype=np.float32).reshape(4, 3, 3)
    model = StandardizedVectorDynamics().fit(batch, scores, np.array([0, 1]))
    original = model.transform(batch, scores)
    changed = scores.copy()
    changed[:, 2, :] += 10_000

    modified = model.transform(batch, changed)

    np.testing.assert_allclose(original.static[:, :2], modified.static[:, :2])
    np.testing.assert_allclose(original.token_delta[:, :2], modified.token_delta[:, :2])


def test_fit_rejects_non_training_examples():
    batch = make_batch()
    scores = np.ones((4, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="training split only"):
        StandardizedVectorDynamics().fit(batch, scores, np.array([0, 2]))
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_vector_dynamics.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'trajectory_extractor.vector_dynamics'`.

- [ ] **Step 3: Implement train-standardized causal differences**

Create `src/trajectory_extractor/vector_dynamics.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajectory_extractor.types import TrajectoryBatch


@dataclass(frozen=True)
class VectorDynamics:
    static: np.ndarray
    layer_delta: np.ndarray
    token_delta: np.ndarray

    def as_feature_tensor(self, *, include_static: bool = True) -> np.ndarray:
        values = [self.layer_delta, self.token_delta]
        if include_static:
            values.insert(0, self.static)
        return np.stack(values, axis=-1).astype(np.float32)


class StandardizedVectorDynamics:
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.means = np.empty(0, dtype=np.float32)
        self.scales = np.empty(0, dtype=np.float32)
        self.fit_example_ids: tuple[str, ...] = ()

    def fit(
        self,
        batch: TrajectoryBatch,
        scores: np.ndarray,
        train_indices: np.ndarray,
    ) -> "StandardizedVectorDynamics":
        values = _validated_scores(batch, scores)
        indices = _validated_train_indices(batch, train_indices)
        means = np.zeros(values.shape[2], dtype=np.float32)
        scales = np.ones(values.shape[2], dtype=np.float32)
        selected_mask = batch.token_mask[indices]
        for layer in range(values.shape[2]):
            samples = values[indices, :, layer][selected_mask]
            means[layer] = samples.mean()
            standard_deviation = float(samples.std())
            scales[layer] = standard_deviation if standard_deviation > self.epsilon else 1.0
        self.means = means
        self.scales = scales
        self.fit_example_ids = tuple(batch.example_ids[index] for index in indices)
        return self

    def transform(self, batch: TrajectoryBatch, scores: np.ndarray) -> VectorDynamics:
        values = _validated_scores(batch, scores)
        if not self.fit_example_ids:
            raise RuntimeError("vector dynamics must be fitted before transform")
        if self.means.shape != (values.shape[2],):
            raise ValueError("score layer count does not match fitted dynamics")
        static = (values - self.means[None, None, :]) / self.scales[None, None, :]
        static = static.astype(np.float32)
        static[~batch.token_mask] = 0.0

        layer_delta = np.zeros_like(static)
        layer_delta[:, :, 1:] = static[:, :, 1:] - static[:, :, :-1]
        layer_delta[~batch.token_mask] = 0.0

        token_delta = np.zeros_like(static)
        valid_pair = batch.token_mask[:, 1:] & batch.token_mask[:, :-1]
        difference = static[:, 1:, :] - static[:, :-1, :]
        token_delta[:, 1:, :] = difference * valid_pair[:, :, None]
        token_delta[~batch.token_mask] = 0.0
        return VectorDynamics(static, layer_delta, token_delta)


def _validated_scores(batch: TrajectoryBatch, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != batch.hidden_states.shape[:3]:
        raise ValueError("scores must have shape [example, token, layer]")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return values


def _validated_train_indices(batch: TrajectoryBatch, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("train_indices must be a non-empty vector")
    if np.unique(values).size != values.size:
        raise ValueError("train_indices must not contain duplicates")
    if values.min() < 0 or values.max() >= len(batch.example_ids):
        raise ValueError("train_indices are out of range")
    if np.any(batch.splits[values] != "train"):
        raise ValueError("vector standardization may use the training split only")
    return values
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_vector_dynamics.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit vector dynamics**

Run:

```bash
git add src/trajectory_extractor/vector_dynamics.py tests/test_vector_dynamics.py
git commit -m "feat: add causal contrastive vector dynamics"
```

Expected: commit succeeds and both new focused test files pass.

### Task 4: Evaluate The Registered Metacognitive Score Without Touching Primary Logic

**Files:**
- Create: `src/trajectory_extractor/secondary_study.py`
- Test: `tests/test_secondary_study.py`

**Interfaces:**
- Consumes: `TrajectoryBatch`, `LayerwiseContrastiveDirection`, `StandardizedVectorDynamics`, existing causal-prefix evaluators, and the existing stable-class `LayerwiseOperatorResidual`.
- Produces: `evaluate_concept_secondary(batch, pca_dims=32, ridge_alpha=1e-3, n_bootstrap=2000) -> dict`, `secondary_endpoint_status(batch, test_indices, min_positive=20, min_clusters=10) -> dict`, and `benjamini_hochberg(p_values) -> np.ndarray`.

- [ ] **Step 1: Write failing tests for the registered method set, fit provenance, endpoint eligibility, and FDR correction**

Create `tests/test_secondary_study.py`:

```python
import numpy as np

from trajectory_extractor.secondary_study import (
    benjamini_hochberg,
    causal_output_uncertainty,
    evaluate_concept_secondary,
    secondary_endpoint_status,
)
from trajectory_extractor.types import TrajectoryBatch


def make_batch() -> TrajectoryBatch:
    rng = np.random.default_rng(17)
    n_train, n_val, n_test = 40, 20, 40
    n_examples = n_train + n_val + n_test
    labels = np.arange(n_examples) % 2
    splits = np.array(
        ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    )
    hidden = rng.normal(0, 0.15, size=(n_examples, 2, 4, 6)).astype(np.float32)
    for layer in range(4):
        hidden[:, :, layer, 0] += labels[:, None] * (0.4 + 0.2 * layer)
        hidden[:, 1, layer, 1] += labels * (0.1 * layer)
    provenance = tuple(
        {
            "entity_family": (
                f"test-family-{index // 2}" if splits[index] == "test" else f"fit-{index // 2}"
            )
        }
        for index in range(n_examples)
    )
    return TrajectoryBatch(
        example_ids=tuple(f"e{index:03d}" for index in range(n_examples)),
        labels=labels.astype(np.int64),
        splits=splits,
        hidden_states=hidden,
        token_mask=np.ones((n_examples, 2), dtype=bool),
        token_logprobs=(-1.0 + 0.05 * rng.normal(size=(n_examples, 2))).astype(np.float32),
        token_entropies=(1.0 + 0.05 * rng.normal(size=(n_examples, 2))).astype(np.float32),
        provenance=provenance,
    )


def test_secondary_evaluation_keeps_registered_comparison_and_fit_ids():
    batch = make_batch()

    result = evaluate_concept_secondary(
        batch,
        pca_dims=3,
        ridge_alpha=1e-3,
        n_bootstrap=50,
    )

    assert set(result["methods"]) == {
        "contrastive_vector",
        "contrastive_plus_dynamics",
        "full_metacognitive_monitor",
    }
    assert result["registered_comparison"]["candidate"] == "contrastive_plus_dynamics"
    assert result["registered_comparison"]["baseline"] == "contrastive_vector"
    assert result["endpoint_status"]["evaluable"] is True
    assert result["endpoint_status"]["positive_examples"] == 20
    assert result["endpoint_status"]["positive_clusters"] == 20
    assert all(identifier.startswith("e0") for identifier in result["fit_example_ids"]["direction"])
    assert all(
        batch.labels[batch.example_ids.index(identifier)] == 0
        for identifier in result["fit_example_ids"]["operator"]
    )
    assert result["artifacts"]["metacognitive_risk_probability"].shape == (40,)
    assert result["artifacts"]["metacognitive_risk_surface"].shape == (40, 2, 4)
    assert result["artifacts"]["directions"].shape == (4, 6)


def test_endpoint_is_not_evaluable_when_positive_cluster_count_is_too_small():
    batch = make_batch()
    provenance = tuple({"entity_family": "one-family"} for _ in batch.provenance)
    reduced = TrajectoryBatch(
        example_ids=batch.example_ids,
        labels=batch.labels,
        splits=batch.splits,
        hidden_states=batch.hidden_states,
        token_mask=batch.token_mask,
        token_logprobs=batch.token_logprobs,
        token_entropies=batch.token_entropies,
        provenance=provenance,
    )
    test = np.flatnonzero(reduced.splits == "test")

    status = secondary_endpoint_status(reduced, test)

    assert status["evaluable"] is False
    assert status["positive_examples"] == 20
    assert status["positive_clusters"] == 1
    assert "positive_clusters<10" in status["reasons"]


def test_benjamini_hochberg_preserves_order_and_monotonicity():
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.20]))

    np.testing.assert_allclose(adjusted, np.array([0.03, 0.06, 0.20]))


def test_output_uncertainty_is_shifted_to_prior_tokens():
    batch = make_batch()

    logprobs, entropies = causal_output_uncertainty(batch)

    assert np.all(logprobs[:, 0] == 0.0)
    assert np.all(entropies[:, 0] == 0.0)
    np.testing.assert_allclose(logprobs[:, 1], batch.token_logprobs[:, 0])
    np.testing.assert_allclose(entropies[:, 1], batch.token_entropies[:, 0])
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_secondary_study.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'trajectory_extractor.secondary_study'`.

- [ ] **Step 3: Implement the isolated secondary evaluator**

Create `src/trajectory_extractor/secondary_study.py`:

```python
from __future__ import annotations

import numpy as np

from trajectory_extractor.contrastive_directions import LayerwiseContrastiveDirection
from trajectory_extractor.evaluation import (
    binary_metrics,
    evaluate_and_predict_prefix_surfaces,
    paired_bootstrap_auc_delta,
    select_threshold,
    threshold_metrics,
)
from trajectory_extractor.features import make_method_tensor
from trajectory_extractor.operator_residual import LayerwiseOperatorResidual
from trajectory_extractor.types import TrajectoryBatch
from trajectory_extractor.vector_dynamics import StandardizedVectorDynamics


REGISTERED_BASELINE = "contrastive_vector"
REGISTERED_CANDIDATE = "contrastive_plus_dynamics"
MIN_AUROC_GAIN = 0.03


def causal_output_uncertainty(batch: TrajectoryBatch) -> tuple[np.ndarray, np.ndarray]:
    logprobs = np.zeros_like(batch.token_logprobs, dtype=np.float32)
    entropies = np.zeros_like(batch.token_entropies, dtype=np.float32)
    logprobs[:, 1:] = batch.token_logprobs[:, :-1]
    entropies[:, 1:] = batch.token_entropies[:, :-1]
    logprobs[~batch.token_mask] = 0.0
    entropies[~batch.token_mask] = 0.0
    return logprobs, entropies


def evaluate_concept_secondary(
    batch: TrajectoryBatch,
    *,
    pca_dims: int = 32,
    ridge_alpha: float = 1e-3,
    n_bootstrap: int = 2000,
) -> dict:
    train = np.flatnonzero(batch.splits == "train")
    validation = np.flatnonzero(batch.splits == "val")
    test = np.flatnonzero(batch.splits == "test")
    if min(train.size, validation.size, test.size) == 0:
        raise ValueError("secondary evaluation requires train, val, and test splits")
    for name, indices in (("train", train), ("validation", validation), ("test", test)):
        if np.unique(batch.labels[indices]).size < 2:
            raise ValueError(f"{name} split requires both classes")

    direction = LayerwiseContrastiveDirection().fit(batch, train)
    projection_scores = direction.transform(batch)
    dynamics_model = StandardizedVectorDynamics().fit(batch, projection_scores, train)
    vector_dynamics = dynamics_model.transform(batch, projection_scores)

    stable_train = train[batch.labels[train] == 0]
    operator = LayerwiseOperatorResidual(pca_dims, ridge_alpha).fit(batch, stable_train)
    operator_scores = operator.transform(batch)
    causal_batch = _with_causal_output_uncertainty(batch)
    full_features = make_method_tensor(
        causal_batch,
        static_scores=vector_dynamics.static,
        operator_residuals=operator_scores,
    )
    full_features = np.concatenate(
        [
            full_features,
            vector_dynamics.layer_delta[..., None],
            vector_dynamics.token_delta[..., None],
        ],
        axis=-1,
    )
    methods = {
        REGISTERED_BASELINE: vector_dynamics.static[..., None],
        REGISTERED_CANDIDATE: vector_dynamics.as_feature_tensor(include_static=True),
        "full_metacognitive_monitor": full_features.astype(np.float32),
    }

    method_results: dict[str, dict] = {}
    selected_test_probabilities: dict[str, np.ndarray] = {}
    test_probability_surfaces: dict[str, np.ndarray] = {}
    for name, features in methods.items():
        surface, validation_probabilities, test_probabilities = (
            evaluate_and_predict_prefix_surfaces(
                features,
                batch.labels,
                token_mask=batch.token_mask,
                train_indices=train,
                validation_indices=validation,
                test_indices=test,
            )
        )
        selected = np.unravel_index(int(np.nanargmax(surface.auroc)), surface.auroc.shape)
        validation_scores = validation_probabilities[:, selected[0], selected[1]]
        test_scores = test_probabilities[:, selected[0], selected[1]]
        threshold = select_threshold(batch.labels[validation], validation_scores)
        metrics = binary_metrics(batch.labels[test], test_scores, threshold=threshold)
        causal_scores = test_probabilities.copy()
        valid_scores = np.broadcast_to(batch.token_mask[test, :, None], causal_scores.shape)
        causal_scores[~valid_scores] = -np.inf
        crossings = threshold_metrics(causal_scores, batch.labels[test], threshold=threshold)
        positive_crossings = crossings.earliest_crossing[batch.labels[test] == 1]
        observed_crossings = positive_crossings[positive_crossings >= 0]
        positive_tokens = crossings.earliest_token[batch.labels[test] == 1]
        observed_tokens = positive_tokens[positive_tokens >= 0]
        positive_layers = crossings.earliest_layer[batch.labels[test] == 1]
        observed_layers = positive_layers[positive_layers >= 0]
        method_results[name] = {
            "selected_token": int(selected[0]),
            "selected_layer": int(selected[1]),
            "validation_auroc": float(surface.auroc[selected]),
            "validation_auprc": float(surface.auprc[selected]),
            "test": metrics,
            "test_auroc": metrics["auroc"],
            "test_auprc": metrics["auprc"],
            "test_calibration_error": metrics["calibration_error"],
            "test_false_positive_rate": crossings.false_positive_rate,
            "median_positive_crossing": (
                float(np.median(observed_crossings)) if observed_crossings.size else None
            ),
            "median_positive_crossing_token": (
                float(np.median(observed_tokens)) if observed_tokens.size else None
            ),
            "median_positive_crossing_layer": (
                float(np.median(observed_layers)) if observed_layers.size else None
            ),
            "validation_surface": {
                "auroc": surface.auroc.tolist(),
                "auprc": surface.auprc.tolist(),
            },
        }
        selected_test_probabilities[name] = test_scores.astype(np.float32)
        test_probability_surfaces[name] = test_probabilities.astype(np.float32)

    cluster_groups = _bootstrap_groups(batch, test)
    bootstrap = paired_bootstrap_auc_delta(
        batch.labels[test],
        selected_test_probabilities[REGISTERED_CANDIDATE],
        selected_test_probabilities[REGISTERED_BASELINE],
        n_bootstrap=n_bootstrap,
        groups=cluster_groups,
    )
    raw_p = _two_sided_bootstrap_p(bootstrap.samples)
    adjusted_p = float(benjamini_hochberg(np.array([raw_p, 1.0]))[0])
    endpoint = secondary_endpoint_status(batch, test)
    supported = bool(
        endpoint["evaluable"]
        and bootstrap.delta >= MIN_AUROC_GAIN
        and bootstrap.lower > 0.0
        and adjusted_p < 0.05
    )
    claim_status = (
        "not_evaluable"
        if not endpoint["evaluable"]
        else "provisional_supported" if supported else "not_supported"
    )

    return {
        "scientific_name": "metacognitive_internal_reliability_signal",
        "user_facing_metaphor": "artificial_intuition",
        "claim_status": claim_status,
        "methods": method_results,
        "registered_comparison": {
            "candidate": REGISTERED_CANDIDATE,
            "baseline": REGISTERED_BASELINE,
            "delta_auroc": bootstrap.delta,
            "lower": bootstrap.lower,
            "upper": bootstrap.upper,
            "raw_p": raw_p,
            "bh_adjusted_p": adjusted_p,
            "fdr_family": [
                "detection_vector_dynamics",
                "intervention_capping_vs_triggered_pending",
            ],
            "minimum_effect": MIN_AUROC_GAIN,
            "supported": supported,
        },
        "endpoint_status": endpoint,
        "fit_example_ids": {
            "direction": list(direction.fit_example_ids),
            "vector_standardization": list(dynamics_model.fit_example_ids),
            "operator": list(operator.fit_example_ids),
        },
        "operator_reference_class": 0,
        "artifacts": {
            "directions": direction.directions,
            "centers": direction.centers,
            "vector_means": dynamics_model.means,
            "vector_scales": dynamics_model.scales,
            "test_indices": test.astype(np.int64),
            "test_labels": batch.labels[test].astype(np.int64),
            "contrastive_vector_probability": selected_test_probabilities[REGISTERED_BASELINE],
            "metacognitive_risk_probability": selected_test_probabilities[REGISTERED_CANDIDATE],
            "metacognitive_risk_surface": test_probability_surfaces[REGISTERED_CANDIDATE],
            "full_monitor_probability": selected_test_probabilities["full_metacognitive_monitor"],
            "bootstrap_delta_samples": bootstrap.samples,
        },
    }


def secondary_endpoint_status(
    batch: TrajectoryBatch,
    test_indices: np.ndarray,
    *,
    min_positive: int = 20,
    min_clusters: int = 10,
) -> dict:
    indices = np.asarray(test_indices, dtype=int)
    positive = indices[batch.labels[indices] == 1]
    clusters = {
        str(batch.provenance[index].get("entity_family", ""))
        for index in positive
        if batch.provenance and batch.provenance[index].get("entity_family")
    }
    reasons: list[str] = []
    if positive.size < min_positive:
        reasons.append(f"positive_examples<{min_positive}")
    if len(clusters) < min_clusters:
        reasons.append(f"positive_clusters<{min_clusters}")
    return {
        "evaluable": not reasons,
        "positive_examples": int(positive.size),
        "positive_clusters": len(clusters),
        "minimum_positive_examples": min_positive,
        "minimum_positive_clusters": min_clusters,
        "reasons": reasons,
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.empty_like(ranked)
    running = 1.0
    count = len(ranked)
    for position in range(count - 1, -1, -1):
        rank = position + 1
        running = min(running, ranked[position] * count / rank)
        adjusted_ranked[position] = min(1.0, running)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_ranked
    return adjusted


def _two_sided_bootstrap_p(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=float)
    lower = (np.count_nonzero(values <= 0.0) + 1) / (values.size + 1)
    upper = (np.count_nonzero(values >= 0.0) + 1) / (values.size + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def _bootstrap_groups(batch: TrajectoryBatch, indices: np.ndarray) -> np.ndarray:
    groups = []
    for index in indices:
        family = batch.provenance[index].get("entity_family") if batch.provenance else None
        groups.append(str(family) if family else batch.example_ids[index])
    return np.asarray(groups)


def _with_causal_output_uncertainty(batch: TrajectoryBatch) -> TrajectoryBatch:
    logprobs, entropies = causal_output_uncertainty(batch)
    return TrajectoryBatch(
        example_ids=batch.example_ids,
        labels=batch.labels,
        splits=batch.splits,
        hidden_states=batch.hidden_states,
        token_mask=batch.token_mask,
        token_logprobs=logprobs,
        token_entropies=entropies,
        provenance=batch.provenance,
    )
```

- [ ] **Step 4: Run the focused secondary tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_secondary_study.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Run primary and secondary unit suites together to catch shared-API regressions**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation.py tests/test_features_and_operator.py tests/test_secondary_study.py -q
```

Expected: all selected tests pass; no existing primary test changes are needed.

- [ ] **Step 6: Commit the isolated secondary evaluator**

Run:

```bash
git add src/trajectory_extractor/secondary_study.py tests/test_secondary_study.py
git commit -m "feat: evaluate metacognitive feature dynamics"
```

Expected: commit succeeds with no modification to `src/trajectory_extractor/study.py`.

### Task 5: Persist Secondary Models And Scores Atomically

**Files:**
- Create: `src/trajectory_extractor/secondary_artifacts.py`
- Test: `tests/test_secondary_artifacts.py`

**Interfaces:**
- Consumes: JSON-compatible metrics and NumPy arrays returned by `evaluate_concept_secondary`.
- Produces: `SecondaryArtifactStore.write_json`, `write_npz`, `read_json`, and `read_npz` under an isolated `secondary/<section>/` namespace.

- [ ] **Step 1: Write failing round-trip and namespace tests**

Create `tests/test_secondary_artifacts.py`:

```python
import numpy as np
import pytest

from trajectory_extractor.secondary_artifacts import SecondaryArtifactStore


def test_secondary_artifacts_round_trip_in_isolated_namespace(tmp_path):
    store = SecondaryArtifactStore(tmp_path)

    json_path = store.write_json(
        "concept-main",
        "comparisons",
        "detection_exact_error",
        {"supported": False, "delta": 0.01},
    )
    array_path = store.write_npz(
        "concept-main",
        "contrastive_vectors",
        "exact_error",
        directions=np.eye(2, dtype=np.float32),
        centers=np.ones((2, 2), dtype=np.float32),
    )

    assert json_path == tmp_path / "concept-main" / "secondary" / "comparisons" / "detection_exact_error.json"
    assert array_path == tmp_path / "concept-main" / "secondary" / "contrastive_vectors" / "exact_error.npz"
    assert store.read_json("concept-main", "comparisons", "detection_exact_error")["delta"] == 0.01
    arrays = store.read_npz("concept-main", "contrastive_vectors", "exact_error")
    np.testing.assert_array_equal(arrays["directions"], np.eye(2, dtype=np.float32))


def test_secondary_artifacts_reject_unknown_sections_and_unsafe_empty_ids(tmp_path):
    store = SecondaryArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="secondary section"):
        store.write_json("concept-main", "metrics", "x", {})
    with pytest.raises(ValueError, match="safe character"):
        store.write_json("___", "comparisons", "x", {})


def test_replacement_leaves_a_complete_readable_artifact(tmp_path):
    store = SecondaryArtifactStore(tmp_path)
    store.write_json("run", "comparisons", "result", {"version": 1})
    store.write_json("run", "comparisons", "result", {"version": 2})

    assert store.read_json("run", "comparisons", "result") == {"version": 2}
    assert list((tmp_path / "run" / "secondary" / "comparisons").glob("*.tmp")) == []
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_secondary_artifacts.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'trajectory_extractor.secondary_artifacts'`.

- [ ] **Step 3: Implement atomic JSON and NPZ writes**

Create `src/trajectory_extractor/secondary_artifacts.py`:

```python
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np


_SECTIONS = {
    "contrastive_vectors",
    "vector_dynamics",
    "activation_capping",
    "comparisons",
}


class SecondaryArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_json(self, run_id: str, section: str, name: str, value) -> Path:
        destination = self._path(run_id, section, name, ".json")
        payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
        _atomic_write_bytes(destination, payload)
        return destination

    def write_npz(self, run_id: str, section: str, name: str, **arrays: np.ndarray) -> Path:
        if not arrays:
            raise ValueError("write_npz requires at least one named array")
        destination = self._path(run_id, section, name, ".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return destination

    def read_json(self, run_id: str, section: str, name: str):
        return json.loads(self._path(run_id, section, name, ".json").read_text())

    def read_npz(self, run_id: str, section: str, name: str) -> dict[str, np.ndarray]:
        with np.load(self._path(run_id, section, name, ".npz")) as arrays:
            return {key: arrays[key].copy() for key in arrays.files}

    def _path(self, run_id: str, section: str, name: str, suffix: str) -> Path:
        if section not in _SECTIONS:
            raise ValueError(f"unknown secondary section: {section}")
        return (
            self.root
            / _safe_id(run_id)
            / "secondary"
            / section
            / f"{_safe_id(name)}{suffix}"
        )


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError("identifier must contain at least one safe character")
    return safe
```

- [ ] **Step 4: Run the focused artifact tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_secondary_artifacts.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit secondary artifact storage**

Run:

```bash
git add src/trajectory_extractor/secondary_artifacts.py tests/test_secondary_artifacts.py
git commit -m "feat: persist secondary study artifacts atomically"
```

Expected: commit succeeds and primary `RunStore` remains unchanged.

### Task 6: Expose The Secondary Evaluation Through The CLI

**Files:**
- Modify: `src/trajectory_extractor/cli.py`
- Modify: `src/trajectory_extractor/plotting.py`
- Test: `tests/test_secondary_cli.py`

**Interfaces:**
- Consumes: `feature-dynamics evaluate-secondary-concept --config configs/llama32_1b.json --run-id concept-main --endpoint exact_error`.
- Produces: atomic model states, test predictions, bootstrap samples, comparison JSON, runtime fields, and two data-derived figures.

- [ ] **Step 1: Write a failing CLI integration test with all expensive work replaced by deterministic fakes**

Create `tests/test_secondary_cli.py`:

```python
import json
from pathlib import Path

import numpy as np

from trajectory_extractor import cli


def test_evaluate_secondary_concept_writes_isolated_artifacts(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_id": "test/model",
                "model_revision": "revision",
                "device": "cpu",
                "dtype": "float32",
                "seed": 42,
                "max_new_tokens": 2,
                "pca_dims": 2,
                "ridge_alpha": 0.001,
                "temperature": 0.0,
                "output_dir": str(tmp_path / "runs"),
            }
        )
    )
    fake_batch = object()
    monkeypatch.setattr(cli.RunStore, "load_batch", lambda self, run_id, label_key: fake_batch)

    def fake_evaluation(batch, **kwargs):
        assert batch is fake_batch
        return {
            "methods": {
                "contrastive_vector": {"test_auroc": 0.60, "validation_surface": {"auroc": [[0.60]]}},
                "contrastive_plus_dynamics": {"test_auroc": 0.70, "validation_surface": {"auroc": [[0.70]]}},
                "full_metacognitive_monitor": {"test_auroc": 0.72, "validation_surface": {"auroc": [[0.72]]}},
            },
            "registered_comparison": {"supported": True},
            "endpoint_status": {"evaluable": True},
            "claim_status": "provisional_supported",
            "artifacts": {
                "directions": np.eye(2, dtype=np.float32),
                "centers": np.zeros((2, 2), dtype=np.float32),
                "vector_means": np.zeros(2, dtype=np.float32),
                "vector_scales": np.ones(2, dtype=np.float32),
                "test_indices": np.array([0, 1]),
                "test_labels": np.array([0, 1]),
                "contrastive_vector_probability": np.array([0.2, 0.6]),
                "metacognitive_risk_probability": np.array([0.1, 0.9]),
                "metacognitive_risk_surface": np.array([[[0.1]], [[0.9]]]),
                "full_monitor_probability": np.array([0.1, 0.95]),
                "bootstrap_delta_samples": np.array([0.05, 0.10]),
            },
        }

    monkeypatch.setattr(cli, "evaluate_concept_secondary", fake_evaluation)
    monkeypatch.setattr(cli, "_write_secondary_figures", lambda *args, **kwargs: None)

    exit_code = cli.main(
        [
            "evaluate-secondary-concept",
            "--config",
            str(config_path),
            "--run-id",
            "concept-main",
            "--bootstrap",
            "20",
            "--endpoint",
            "exact_error",
        ]
    )

    root = tmp_path / "runs" / "concept-main" / "secondary"
    assert exit_code == 0
    assert (root / "comparisons" / "detection_exact_error.json").exists()
    assert (root / "comparisons" / "predictions_exact_error.npz").exists()
    assert (root / "contrastive_vectors" / "exact_error.npz").exists()
    assert (root / "vector_dynamics" / "exact_error.npz").exists()
    assert not (tmp_path / "runs" / "concept-main" / "metrics" / "detection.json").exists()
```

- [ ] **Step 2: Run the CLI test and confirm the parser rejects the new command**

Run:

```bash
.venv/bin/python -m pytest tests/test_secondary_cli.py -q
```

Expected: failure because `evaluate-secondary-concept` and its imports do not exist yet.

- [ ] **Step 3: Add the secondary imports and parser without changing existing command definitions**

Add these imports near the other project imports in `src/trajectory_extractor/cli.py`:

```python
from trajectory_extractor.secondary_artifacts import SecondaryArtifactStore
from trajectory_extractor.secondary_study import evaluate_concept_secondary
```

Add this parser immediately after the existing `evaluate-concept` parser:

```python
    evaluate_secondary = subparsers.add_parser("evaluate-secondary-concept")
    evaluate_secondary.add_argument("--config", default="configs/llama32_1b.json")
    evaluate_secondary.add_argument("--run-id", default="concept-main")
    evaluate_secondary.add_argument("--bootstrap", type=int, default=2000)
    evaluate_secondary.add_argument(
        "--endpoint",
        choices=("exact_error", "binding_error"),
        default="exact_error",
    )
```

- [ ] **Step 4: Add the command handler and keep arrays out of JSON**

Insert this handler immediately after the existing `evaluate-concept` handler in `src/trajectory_extractor/cli.py`:

```python
    if args.command == "evaluate-secondary-concept":
        config = ExperimentConfig.from_json(args.config)
        batch = RunStore(config.output_dir).load_batch(args.run_id, label_key=args.endpoint)
        started = time.perf_counter()
        result = evaluate_concept_secondary(
            batch,
            pca_dims=config.pca_dims,
            ridge_alpha=config.ridge_alpha,
            n_bootstrap=args.bootstrap,
        )
        result["runtime"] = {
            "seconds": time.perf_counter() - started,
            "max_resident_set_size": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        arrays = result.pop("artifacts")
        secondary = SecondaryArtifactStore(config.output_dir)
        secondary.write_npz(
            args.run_id,
            "contrastive_vectors",
            args.endpoint,
            directions=arrays["directions"],
            centers=arrays["centers"],
        )
        secondary.write_npz(
            args.run_id,
            "vector_dynamics",
            args.endpoint,
            means=arrays["vector_means"],
            scales=arrays["vector_scales"],
        )
        secondary.write_npz(
            args.run_id,
            "comparisons",
            f"predictions_{args.endpoint}",
            test_indices=arrays["test_indices"],
            test_labels=arrays["test_labels"],
            contrastive_vector_probability=arrays["contrastive_vector_probability"],
            metacognitive_risk_probability=arrays["metacognitive_risk_probability"],
            metacognitive_risk_surface=arrays["metacognitive_risk_surface"],
            full_monitor_probability=arrays["full_monitor_probability"],
            bootstrap_delta_samples=arrays["bootstrap_delta_samples"],
        )
        metrics_path = secondary.write_json(
            args.run_id,
            "comparisons",
            f"detection_{args.endpoint}",
            result,
        )
        _write_secondary_figures(
            secondary,
            args.run_id,
            result,
            arrays,
            endpoint=args.endpoint,
        )
        print(
            json.dumps(
                {
                    "metrics": str(metrics_path),
                    "supported": result["registered_comparison"]["supported"],
                    "evaluable": result["endpoint_status"]["evaluable"],
                    "claim_status": result["claim_status"],
                }
            )
        )
        return 0
```

- [ ] **Step 5: Add a data-derived risk-gap plot**

Add this function to `src/trajectory_extractor/plotting.py`:

```python
def plot_class_risk_gap(
    probabilities: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    values = np.asarray(probabilities, dtype=float)
    target = np.asarray(labels, dtype=int)
    if values.ndim != 3 or values.shape[0] != target.size:
        raise ValueError("probabilities must have shape [example, token, layer]")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("risk-gap plot requires both classes")
    gap = np.median(values[target == 1], axis=0) - np.median(values[target == 0], axis=0)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(gap, aspect="auto", origin="lower", cmap="coolwarm")
    axis.set_xlabel("Layer prefix")
    axis.set_ylabel("Answer-token prefix")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Median risk gap")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
```

Ensure `src/trajectory_extractor/plotting.py` imports `Path`, `matplotlib.pyplot as plt`, and `numpy as np`; reuse existing imports when already present.

- [ ] **Step 6: Add the secondary figure writer using persisted result values**

Add this helper near `_write_detection_figures` in `src/trajectory_extractor/cli.py`:

```python
def _write_secondary_figures(
    store: SecondaryArtifactStore,
    run_id: str,
    result: dict,
    arrays: dict[str, np.ndarray],
    *,
    endpoint: str,
) -> None:
    from trajectory_extractor.plotting import plot_class_risk_gap, plot_method_comparison

    directory = store.root / run_id / "secondary" / "comparisons" / "figures"
    plot_method_comparison(
        {name: values["test_auroc"] for name, values in result["methods"].items()},
        directory / f"method_comparison_{endpoint}.png",
    )
    plot_class_risk_gap(
        arrays["metacognitive_risk_surface"],
        arrays["test_labels"],
        directory / f"metacognitive_risk_gap_{endpoint}.png",
        title="Held-out metacognitive risk gap",
    )
```

This figure uses the actual held-out causal-prefix probability surface and does not invent green/yellow/red zones or a predetermined tipping layer.

- [ ] **Step 7: Run the CLI test and the complete secondary test set**

Run:

```bash
.venv/bin/python -m pytest tests/test_secondary_cli.py tests/test_secondary_artifacts.py tests/test_secondary_study.py tests/test_vector_dynamics.py tests/test_contrastive_directions.py -q
```

Expected: `14 passed`.

- [ ] **Step 8: Commit the CLI and plotting integration**

Run:

```bash
git add src/trajectory_extractor/cli.py src/trajectory_extractor/plotting.py tests/test_secondary_cli.py
git commit -m "feat: expose metacognitive monitor evaluation"
```

Expected: commit succeeds and no file under `runs/` is staged.

### Task 7: Document The Scientific Boundary And Run Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/execution_plan.md`
- Modify: `docs/references.md`

**Interfaces:**
- Consumes: the implemented CLI and the approved scientific framing.
- Produces: a reproducible user command, an explicit non-anthropomorphic score definition, and literature attribution that does not overclaim novelty.

- [ ] **Step 1: Add the secondary command and score interpretation to the README**

Add this command directly after the existing `evaluate-concept` command in `README.md`:

```bash
feature-dynamics evaluate-secondary-concept \
  --config configs/llama32_1b.json \
  --run-id concept-main \
  --bootstrap 2000 \
  --endpoint exact_error
```

Add this paragraph below the concept command block:

```markdown
The secondary command evaluates a **metacognitive internal reliability signal**:
a causal risk score derived from a contrastive error direction and its evolution
across answer-token and layer prefixes. "Artificial intuition" is a user-facing
metaphor only. The score does not imply consciousness or ground-truth access, and
it is not a failure proof. The registered question is whether dynamics add held-out
predictive value beyond the same contrastive direction used statically.
```

- [ ] **Step 2: Add the secondary analysis stage to the execution plan**

Add this subsection after the existing concept detection stage in `docs/execution_plan.md`:

```markdown
### Stage 3b: Prospective metacognitive feature-flow monitor

1. Keep `docs/preregistration.md` and `evaluate-concept` unchanged.
2. Fit centered risk-minus-control activation directions on training examples only.
3. Standardize direction projections per layer with training-only statistics.
4. Derive causal cross-layer and prior-token differences.
5. Select the prefix and threshold on validation only.
6. Compare `contrastive_plus_dynamics` against `contrastive_vector` once on test.
7. Persist results under `runs/concept-main/secondary/`; never overwrite primary metrics.
8. Mark the result `not_evaluable` unless test has 20 positive examples across 10
   independent entity families.

This stage validates a monitoring signal only. External-fact transfer, activation
control, JailbreakBench transfer, and multi-model replication require separate
plans before any broad reliability or safety claim.
```

- [ ] **Step 3: Add the Anthropic baselines that constrain the novelty claim**

Append these entries to `docs/references.md`:

```markdown
- [Anthropic, Persona Vectors](https://www.anthropic.com/research/persona-vectors)
- [Anthropic, The Assistant Axis](https://www.anthropic.com/research/assistant-axis)
- [Anthropic, Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html)
```

Append this paragraph after the reference list:

```markdown
Persona Vectors, the Assistant Axis, and the emotion-concepts work establish that
distributed activation directions can monitor and causally affect behavior. This
project therefore does not claim the first activation-vector monitor, dynamic
internal monitor, or safety steering method. Its narrower registered question is
whether causal layer/token dynamics of a contrastive direction add held-out value
for entity-relation-object binding failures in a small open model.
```

- [ ] **Step 4: Run the complete test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: `78 passed` and exit code `0`.

- [ ] **Step 5: Run formatting and repository integrity checks**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` emits no output; only the three intended documentation files are modified.

- [ ] **Step 6: Commit the documentation boundary**

Run:

```bash
git add README.md docs/execution_plan.md docs/references.md
git commit -m "docs: define the metacognitive monitor boundary"
```

Expected: commit succeeds and the working tree is clean.

### Task 8: Complete The Frozen Corpus And Execute Primary Then Secondary Evaluation

**Files:**
- Runtime artifacts only: `runs/concept-main/`
- No source file modifications.

**Interfaces:**
- Consumes: the unchanged 1,200-example dataset, existing resumable extraction, frozen primary evaluator, and new secondary evaluator.
- Produces: primary and secondary held-out artifacts with an auditable proof that secondary analysis did not alter the primary result.

- [ ] **Step 1: Resume the unchanged concept extraction**

Run:

```bash
.venv/bin/feature-dynamics extract-concept \
  --config configs/llama32_1b.json \
  --data data/processed/concept_mixing.jsonl \
  --run-id concept-main
```

Expected: the command skips complete examples, resumes from the existing checkpoint, exits `0`, and does not rewrite already complete `.json`/`.npz` pairs.

- [ ] **Step 2: Verify that every dataset row has both artifact halves**

Run:

```bash
find runs/concept-main/examples -name '*.json' | wc -l
find runs/concept-main/examples -name '*.npz' | wc -l
```

Expected: both commands print `1200`.

- [ ] **Step 3: Run the frozen primary evaluation first**

Run:

```bash
.venv/bin/feature-dynamics evaluate-concept \
  --config configs/llama32_1b.json \
  --run-id concept-main \
  --bootstrap 2000 \
  --endpoint exact_error
```

Expected: exit code `0` and a complete `runs/concept-main/metrics/detection_exact_error.json`.

- [ ] **Step 4: Record the primary artifact hash before secondary evaluation**

Run:

```bash
shasum -a 256 runs/concept-main/metrics/detection_exact_error.json
```

Expected: one SHA-256 line. Keep this value for Step 7.

- [ ] **Step 5: Run the registered metacognitive detector on exact error**

Run:

```bash
.venv/bin/feature-dynamics evaluate-secondary-concept \
  --config configs/llama32_1b.json \
  --run-id concept-main \
  --bootstrap 2000 \
  --endpoint exact_error
```

Expected: exit code `0` and JSON output containing `metrics`, `supported`, and `evaluable`.

- [ ] **Step 6: Run the same frozen pipeline on the mechanistic binding diagnostic**

Run:

```bash
.venv/bin/feature-dynamics evaluate-secondary-concept \
  --config configs/llama32_1b.json \
  --run-id concept-main \
  --bootstrap 2000 \
  --endpoint binding_error
```

Expected: exit code `0`; the endpoint may legitimately report `evaluable: false` if its held-out positive count is below the frozen rule.

- [ ] **Step 7: Prove that secondary execution did not modify the primary artifact**

Run:

```bash
shasum -a 256 runs/concept-main/metrics/detection_exact_error.json
```

Expected: the SHA-256 value exactly matches Step 4.

- [ ] **Step 8: Validate the secondary artifact set**

Run:

```bash
test -f runs/concept-main/secondary/comparisons/detection_exact_error.json
test -f runs/concept-main/secondary/comparisons/predictions_exact_error.npz
test -f runs/concept-main/secondary/contrastive_vectors/exact_error.npz
test -f runs/concept-main/secondary/vector_dynamics/exact_error.npz
test -f runs/concept-main/secondary/comparisons/figures/method_comparison_exact_error.png
```

Expected: every command exits `0`.

- [ ] **Step 9: Render and inspect the registered decision payload**

Run:

```bash
.venv/bin/python -m json.tool \
  runs/concept-main/secondary/comparisons/detection_exact_error.json
```

Expected: valid JSON containing `registered_comparison`, `endpoint_status`, `methods`, `fit_example_ids`, and `runtime`. Report the result exactly as `provisional_supported`, `not_supported`, or `not_evaluable`; do not reinterpret a null result as success.

## Completion Boundary

This plan is complete when the secondary detector is tested, versioned, and executed on the frozen held-out synthetic concept-mixing split. It establishes whether the proposed score is a useful **additional internal warning signal**. It does not establish general factuality, jailbreak prevention, consciousness, or a universal LLM intuition mechanism.

Even when `registered_comparison.supported` is true, treat the result as
`provisional_supported` until the required shuffled-label, shuffled-layer,
random-projection, matched-capacity static, and metadata-confounder controls are
complete. Those controls are grouped with external transfer in the next concept
validation plan so they use the frozen detector without reopening this test split.

Before a fellowship-facing final claim, write and execute separate plans for:

1. source-documented external-fact transfer and frozen secondary falsification controls;
2. validation-selected projection capping and capability-preservation controls;
3. independent JailbreakBench monitoring and Llama Guard/human evaluation;
4. Qwen multi-model replication on Colab after hardware selection is frozen.
