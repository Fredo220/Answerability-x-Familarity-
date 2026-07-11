from collections.abc import Mapping

import numpy as np

from trajectory_extractor.evaluator import TrajectoryEvaluator


def extract_layer_scores(
    model,
    evaluator: TrajectoryEvaluator,
    prompt: str,
) -> list[float]:
    _, cache = model.run_with_cache(prompt)
    num_layers = int(model.cfg.n_layers)
    scores: list[float] = []
    previous_vec: np.ndarray | None = None

    for layer_idx in range(num_layers - 1):
        current_vec = _last_token_vector(cache, layer_idx)
        next_vec = _last_token_vector(cache, layer_idx + 1)

        if previous_vec is None:
            scores.append(0.0)
        else:
            scores.append(evaluator.evaluate_step(previous_vec, current_vec, next_vec))

        previous_vec = current_vec

    return scores


def run_conditions(
    model,
    prompts: Mapping[str, str],
    evaluator: TrajectoryEvaluator,
) -> dict[str, list[float]]:
    return {
        condition_name: extract_layer_scores(model, evaluator, prompt)
        for condition_name, prompt in prompts.items()
    }


def _last_token_vector(cache, layer_idx: int) -> np.ndarray:
    value = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1, :]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)
