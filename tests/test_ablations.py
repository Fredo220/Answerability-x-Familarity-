import numpy as np

from trajectory_extractor.ablations import last_token_only, random_projection, shuffled_layers, steering_grid
from trajectory_extractor.types import TrajectoryBatch


def batch():
    hidden = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(np.float16)
    return TrajectoryBatch(
        example_ids=("a", "b"),
        labels=np.array([0, 1]),
        splits=np.array(["train", "test"]),
        hidden_states=hidden,
        token_mask=np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
        token_logprobs=np.zeros((2, 3), dtype=np.float32),
        token_entropies=np.zeros((2, 3), dtype=np.float32),
    )


def test_required_structural_ablations_preserve_alignment():
    original = batch()
    assert last_token_only(original).token_mask.tolist() == [[False, True, False], [False, False, True]]
    assert shuffled_layers(original, seed=1).hidden_states.shape == original.hidden_states.shape
    assert random_projection(original, dimensions=3).hidden_states.shape == (2, 3, 4, 3)
    assert len(steering_grid([3, 5], [0.5, 1.0])) == 4
