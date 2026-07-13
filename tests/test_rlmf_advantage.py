import pytest
import torch

from trajectory_extractor.rlmf_advantage import (
    RewardBatch,
    compute_group_advantages,
    rlmf_advantage,
    standard_grpo_advantage,
)


# Copied from the official piecewise formula in yale-nlp/RLMF at
# a087e7a1e49f52aaa701add19cd80699b709fdef,
# src/exp2_rlmf/c_rl_training/rlmf_trainer.py:625-653 (vendored SHA-256
# d608b198324407f949c07d7f693680951ec62edf6962036ac1afe896f112cfeb).
UPSTREAM_PARITY_FIXTURE = {
    "other_reward": [2.0, 6.0, -2.0, 4.0],
    "faith_reward": [1.0, 5.0, 3.0, 7.0],
    "metacognitive_reward": [0.25, 0.50, -3.0, 2.0],
    "k": 1.0,
    "expected": [-3.5, 5.0, -5.5, 10.5],
}


def tensor(values, *, dtype=torch.float64, requires_grad=False):
    return torch.tensor(values, dtype=dtype, requires_grad=requires_grad)


def test_standard_grpo_advantage_uses_weighted_total_reward_without_std_normalization():
    other = tensor([2.0, 6.0])
    faith = tensor([1.0, 5.0])

    advantage = standard_grpo_advantage(other, faith)

    torch.testing.assert_close(advantage, tensor([-4.0, 4.0]))


def test_rlmf_advantage_matches_pinned_upstream_piecewise_fixture():
    fixture = UPSTREAM_PARITY_FIXTURE

    advantage = rlmf_advantage(
        tensor(fixture["other_reward"]),
        tensor(fixture["faith_reward"]),
        tensor(fixture["metacognitive_reward"]),
        k=fixture["k"],
    )

    torch.testing.assert_close(advantage, tensor(fixture["expected"]))


def test_rlmf_scales_only_strictly_above_mean_faithfulness_rewards():
    other = tensor([0.0, 0.0])
    faith = tensor([1.0, 3.0])
    metascore = tensor([100.0, 2.0])

    advantage = rlmf_advantage(other, faith, metascore)

    torch.testing.assert_close(advantage, tensor([-1.0, 3.0]))


def test_rlmf_does_not_add_metascore_as_a_reward():
    zeros = tensor([0.0, 0.0])

    advantage = rlmf_advantage(zeros, zeros, tensor([100.0, -100.0]))

    torch.testing.assert_close(advantage, zeros)


def test_group_advantages_center_each_complete_rollout_group_independently():
    batch = RewardBatch(
        other_reward=tensor([2.0, 6.0, 10.0, 14.0]),
        faith_reward=tensor([1.0, 5.0, 3.0, 7.0]),
        metacognitive_reward=tensor([0.0, 0.0, 0.0, 0.0]),
        group_size=2,
    )

    advantage = compute_group_advantages(batch, "standard")

    torch.testing.assert_close(advantage, tensor([-4.0, 4.0, -4.0, 4.0]))


def test_equal_rewards_produce_zero_advantages_in_both_arms():
    batch = RewardBatch(
        other_reward=tensor([2.0, 2.0, 5.0, 5.0]),
        faith_reward=tensor([3.0, 3.0, 7.0, 7.0]),
        metacognitive_reward=tensor([-10.0, 10.0, 4.0, 1.0]),
        group_size=2,
    )

    for arm in ("standard", "rlmf"):
        torch.testing.assert_close(
            compute_group_advantages(batch, arm),
            torch.zeros_like(batch.other_reward),
        )


def test_group_advantages_are_permutation_equivariant_within_each_group():
    batch = RewardBatch(
        other_reward=tensor([2.0, 6.0, -2.0, 4.0]),
        faith_reward=tensor([1.0, 5.0, 3.0, 7.0]),
        metacognitive_reward=tensor([0.25, 0.50, -3.0, 2.0]),
        group_size=2,
    )
    permutation = torch.tensor([1, 0, 3, 2])
    inverse = torch.argsort(permutation)
    permuted = RewardBatch(
        other_reward=batch.other_reward[permutation],
        faith_reward=batch.faith_reward[permutation],
        metacognitive_reward=batch.metacognitive_reward[permutation],
        group_size=2,
    )

    for arm in ("standard", "rlmf"):
        expected = compute_group_advantages(batch, arm)
        actual = compute_group_advantages(permuted, arm)[inverse]
        torch.testing.assert_close(actual, expected)


def test_advantages_preserve_dtype_and_device_and_detach_from_rewards():
    other = tensor([1.0, 3.0], dtype=torch.float32, requires_grad=True)
    faith = tensor([2.0, 4.0], dtype=torch.float32, requires_grad=True)
    metascore = tensor([0.5, 0.5], dtype=torch.float32, requires_grad=True)

    advantage = rlmf_advantage(other, faith, metascore)
    policy_log_prob = tensor([-0.3, -0.7], dtype=torch.float32, requires_grad=True)
    loss = -(advantage * policy_log_prob).mean()
    loss.backward()

    assert advantage.dtype is torch.float32
    assert advantage.device == other.device
    assert not advantage.requires_grad
    assert other.grad is None
    assert faith.grad is None
    assert metascore.grad is None
    assert policy_log_prob.grad is not None
    assert torch.isfinite(policy_log_prob.grad).all()


@pytest.mark.parametrize(
    ("other", "faith"),
    [
        pytest.param(["max", "max"], ["max", "max"], id="total-overflow"),
        pytest.param(["max", 0.0], [0.0, "max"], id="mean-overflow"),
        pytest.param(["max", "-max", "-max"], [0.0, 0.0, 0.0], id="centering-overflow"),
    ],
)
def test_standard_rejects_finite_inputs_that_overflow_derived_values(other, faith):
    maximum = torch.finfo(torch.float32).max
    other = tensor(
        [maximum if value == "max" else -maximum if value == "-max" else value for value in other],
        dtype=torch.float32,
    )
    faith = tensor(
        [maximum if value == "max" else -maximum if value == "-max" else value for value in faith],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="finite"):
        standard_grpo_advantage(other, faith)


@pytest.mark.parametrize(
    ("other", "faith", "metascore", "k"),
    [
        pytest.param(["max", "max"], [0.0, 0.0], [0.0, 0.0], 1.0, id="mean-overflow"),
        pytest.param(
            ["max", "-max", "-max"],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
            id="centering-overflow",
        ),
        pytest.param([0.0, 0.0], [0.0, "max"], ["max", "max"], 1.0, id="scaling-overflow"),
        pytest.param(["max", "-max"], ["max", 0.0], [0.0, 0.0], 1.0, id="final-addition-overflow"),
        pytest.param([0.0, 0.0], [0.0, 1.0], [0.0, 0.0], 1e308, id="unrepresentable-k"),
    ],
)
def test_rlmf_rejects_finite_inputs_that_overflow_derived_values(other, faith, metascore, k):
    maximum = torch.finfo(torch.float32).max

    def make(values):
        return tensor(
            [maximum if value == "max" else -maximum if value == "-max" else value for value in values],
            dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="finite"):
        rlmf_advantage(make(other), make(faith), make(metascore), k=k)


@pytest.mark.parametrize("arm", ["standard", "rlmf"])
def test_group_advantages_reject_finite_inputs_that_overflow_derived_values(arm):
    maximum = torch.finfo(torch.float32).max
    batch = RewardBatch(
        other_reward=tensor([maximum, maximum], dtype=torch.float32),
        faith_reward=tensor([0.0, 0.0], dtype=torch.float32),
        metacognitive_reward=tensor([0.0, 0.0], dtype=torch.float32),
        group_size=2,
    )

    with pytest.raises(ValueError, match="finite"):
        compute_group_advantages(batch, arm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_advantages_preserve_cuda_device():
    device = torch.device("cuda")
    batch = RewardBatch(
        other_reward=tensor([1.0, 3.0], dtype=torch.float32).to(device),
        faith_reward=tensor([2.0, 4.0], dtype=torch.float32).to(device),
        metacognitive_reward=tensor([0.5, 0.5], dtype=torch.float32).to(device),
        group_size=2,
    )

    advantage = compute_group_advantages(batch, "rlmf")

    assert advantage.device == device
    assert advantage.dtype is torch.float32


@pytest.mark.parametrize(
    "batch",
    [
        RewardBatch(
            tensor([1.0, 2.0, 3.0]),
            tensor([1.0, 2.0, 3.0]),
            tensor([0.0, 0.0, 0.0]),
            2,
        ),
        RewardBatch(
            tensor([1.0, float("nan")]),
            tensor([1.0, 2.0]),
            tensor([0.0, 0.0]),
            2,
        ),
    ],
)
def test_group_advantages_reject_incomplete_or_nonfinite_reward_batches(batch):
    with pytest.raises(ValueError):
        compute_group_advantages(batch, "rlmf")


def test_group_advantages_rejects_unknown_arm():
    batch = RewardBatch(
        tensor([1.0, 2.0]),
        tensor([1.0, 2.0]),
        tensor([0.0, 0.0]),
        2,
    )

    with pytest.raises(ValueError, match="arm"):
        compute_group_advantages(batch, "unknown")
