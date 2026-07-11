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
