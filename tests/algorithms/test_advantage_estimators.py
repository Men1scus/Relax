# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Golden-value tests for the extracted advantage estimators."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.advantages import (  # noqa: E402
    ADVANTAGE_FNS,
    compute_advantages_and_returns,
    whiten_scalar,
)


def _args(estimator, **overrides):
    base = dict(advantage_estimator=estimator, kl_coef=0.0, gamma=1.0, lambd=1.0)
    base.update(overrides)
    return SimpleNamespace(**base)


def _inputs(lengths=(3, 2)):
    return dict(
        kl=[torch.zeros(n, dtype=torch.float32) for n in lengths],
        loss_masks=[torch.ones(n, dtype=torch.float32) for n in lengths],
        response_lengths=list(lengths),
        total_lengths=[n + 2 for n in lengths],
        values=None,
    )


# ---------------- dispatch ----------------


def test_registry_covers_every_spec_advantage_id():
    from relax.algorithms import list_algorithm_names
    from relax.algorithms.spec import get_algorithm

    for name in list_algorithm_names():
        assert get_algorithm(name).advantage_fn in ADVANTAGE_FNS


def test_unknown_advantage_fn_is_not_silently_tolerated():
    with pytest.raises(KeyError):
        ADVANTAGE_FNS["not_a_real_fn"]


# ---------------- grpo family ----------------


def test_grpo_broadcast_repeats_the_scalar_over_tokens():
    adv, ret = compute_advantages_and_returns(_args("grpo"), rewards=[1.5, -2.0], **_inputs())
    assert torch.equal(adv[0], torch.full((3,), 1.5))
    assert torch.equal(adv[1], torch.full((2,), -2.0))
    assert torch.equal(ret[0], adv[0])


def test_grpo_accepts_a_tensor_as_well_as_a_list():
    from_list = compute_advantages_and_returns(_args("grpo"), rewards=[1.5, -2.0], **_inputs())[0]
    from_tensor = compute_advantages_and_returns(_args("grpo"), rewards=torch.tensor([1.5, -2.0]), **_inputs())[0]
    assert torch.equal(from_list[0], from_tensor[0])


def test_grpo_advantages_is_a_distinct_list_from_returns():
    """Legacy did `advantages = list(returns)`; rebinding one must not touch the other."""
    adv, ret = compute_advantages_and_returns(_args("grpo"), rewards=[1.0, 1.0], **_inputs())
    assert adv is not ret
    adv[0] = torch.zeros(3)
    assert not torch.equal(ret[0], adv[0])


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "sapo", "cispo"])
def test_grpo_family_produces_identical_advantages(estimator):
    baseline, _ = compute_advantages_and_returns(_args("grpo"), rewards=[0.5, -0.5], **_inputs())
    actual, _ = compute_advantages_and_returns(_args(estimator), rewards=[0.5, -0.5], **_inputs())
    for left, right in zip(baseline, actual, strict=True):
        assert torch.equal(left, right)


# ---------------- reinforce++ baseline ----------------


def test_reinforce_plus_plus_baseline_aliases_returns_to_advantages():
    """Legacy did `returns = advantages` (the same list object). Preserve it."""
    adv, ret = compute_advantages_and_returns(_args("reinforce_plus_plus_baseline"), rewards=[1.0, 1.0], **_inputs())
    assert ret is adv


def test_reinforce_plus_plus_baseline_subtracts_kl():
    inputs = _inputs(lengths=(2,))
    inputs["kl"] = [torch.tensor([2.0, 4.0])]
    inputs["loss_masks"] = [torch.ones(2)]
    adv, _ = compute_advantages_and_returns(
        _args("reinforce_plus_plus_baseline", kl_coef=0.5), rewards=[3.0], **inputs
    )
    # ones * 3.0 - 0.5 * [2, 4] = [2.0, 1.0]
    assert torch.equal(adv[0], torch.tensor([2.0, 1.0]))


# ---------------- gdpo step 3 ----------------


def test_whiten_scalar_produces_zero_mean_unit_std():
    out = whiten_scalar(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert abs(out.mean().item()) < 1e-6
    assert abs(out.std().item() - 1.0) < 1e-3


def test_whiten_scalar_zero_variance_returns_zeros_not_noise():
    """A collapsed batch must give exactly 0, not the fp32 residual over
    eps."""
    assert torch.equal(whiten_scalar(torch.full((7,), 0.7)), torch.zeros(7))


def test_whiten_scalar_single_element_returns_zero():
    """std of one element is NaN under Bessel correction."""
    assert torch.equal(whiten_scalar(torch.tensor([3.0])), torch.zeros(1))


def test_whiten_scalar_empty_returns_empty():
    assert whiten_scalar(torch.tensor([])).numel() == 0


def test_whiten_scalar_preserves_ordering():
    values = torch.tensor([-3.0, 0.5, 0.0, 9.0])
    out = whiten_scalar(values)
    assert torch.equal(values.argsort(), out.argsort())


def test_gdpo_advantage_whitens_before_broadcasting():
    rewards = [2.0, -2.0]
    adv, _ = compute_advantages_and_returns(_args("gdpo"), rewards=rewards, **_inputs())
    expected = whiten_scalar(torch.tensor(rewards, dtype=torch.float32))
    assert torch.allclose(adv[0], expected[0].expand(3))
    assert torch.allclose(adv[1], expected[1].expand(2))


def test_gdpo_differs_from_grpo_by_a_positive_scalar():
    rewards = [1.0, -1.0, 0.5, -0.5]
    inputs = _inputs(lengths=(1, 1, 1, 1))
    grpo, _ = compute_advantages_and_returns(_args("grpo"), rewards=rewards, **inputs)
    gdpo, _ = compute_advantages_and_returns(_args("gdpo"), rewards=rewards, **inputs)

    grpo_flat = torch.cat(grpo)
    gdpo_flat = torch.cat(gdpo)
    ratio = gdpo_flat / grpo_flat
    assert (ratio > 0).all()
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-4)
    assert not torch.allclose(gdpo_flat, grpo_flat, atol=1e-3)


def test_gdpo_collapsed_batch_gives_zero_advantages():
    adv, _ = compute_advantages_and_returns(_args("gdpo"), rewards=[0.7, 0.7], **_inputs())
    assert torch.equal(adv[0], torch.zeros(3))
    assert torch.equal(adv[1], torch.zeros(2))
