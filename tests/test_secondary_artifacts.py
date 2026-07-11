import numpy as np
import pytest

import trajectory_extractor.secondary_artifacts as secondary_artifacts
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


def test_npz_replacement_leaves_version_two_complete_and_readable(tmp_path):
    store = SecondaryArtifactStore(tmp_path)
    first = np.array([1, 2], dtype=np.int64)
    second = np.array([3, 4], dtype=np.int64)

    store.write_npz("run", "vector_dynamics", "result", values=first)
    store.write_npz("run", "vector_dynamics", "result", values=second)

    arrays = store.read_npz("run", "vector_dynamics", "result")
    np.testing.assert_array_equal(arrays["values"], second)
    assert list((tmp_path / "run" / "secondary" / "vector_dynamics").glob("*.tmp")) == []


def test_npz_write_failure_preserves_existing_artifact_and_cleans_temp(tmp_path, monkeypatch):
    store = SecondaryArtifactStore(tmp_path)
    original = np.array([1, 2], dtype=np.int64)
    replacement = np.array([3, 4], dtype=np.int64)
    store.write_npz("run", "vector_dynamics", "result", values=original)

    def write_partial_then_raise(handle, **arrays):
        handle.write(b"partial npz")
        raise RuntimeError("injected npz failure")

    monkeypatch.setattr(secondary_artifacts.np, "savez_compressed", write_partial_then_raise)
    with pytest.raises(RuntimeError, match="injected npz failure"):
        store.write_npz("run", "vector_dynamics", "result", values=replacement)

    arrays = store.read_npz("run", "vector_dynamics", "result")
    np.testing.assert_array_equal(arrays["values"], original)
    assert list((tmp_path / "run" / "secondary" / "vector_dynamics").glob("*.tmp")) == []


def test_json_replace_failure_preserves_existing_artifact_and_cleans_temp(tmp_path, monkeypatch):
    store = SecondaryArtifactStore(tmp_path)
    store.write_json("run", "comparisons", "result", {"version": 1})

    def raise_on_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(secondary_artifacts.os, "replace", raise_on_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        store.write_json("run", "comparisons", "result", {"version": 2})

    assert store.read_json("run", "comparisons", "result") == {"version": 1}
    assert list((tmp_path / "run" / "secondary" / "comparisons").glob("*.tmp")) == []


def test_traversal_like_ids_stay_under_exact_secondary_namespace(tmp_path):
    store = SecondaryArtifactStore(tmp_path)

    path = store.write_json("../outside", "comparisons", "../../result", {"ok": True})

    relative = path.resolve().relative_to(tmp_path.resolve())
    assert relative.parts[1:3] == ("secondary", "comparisons")
    assert relative.name.endswith(".json")
    assert path.resolve().is_relative_to(tmp_path.resolve())
