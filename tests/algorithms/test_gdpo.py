# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GDPO: per-reward group standardisation, weighted sum, batch whitening.

The reference in ``_manual_gdpo`` implements arXiv 2601.05242 Eq. 4 and Eq. 7
directly in plain Python, independently of the tensor implementation, so a
mistake in one is unlikely to be mirrored in the other.
"""

import math
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms import get_algorithm  # noqa: E402
from relax.algorithms.advantages import whiten_scalar  # noqa: E402
from relax.algorithms.rewards import REWARD_NORMALIZERS, extract_reward_components  # noqa: E402


def _args(keys=("correctness", "format"), weights=None, n=4):
    return SimpleNamespace(
        advantage_estimator="gdpo",
        n_samples_per_prompt=n,
        rewards_normalization=True,
        grpo_std_normalization=True,
        gdpo_reward_keys=list(keys) if keys is not None else None,
        gdpo_reward_weights=weights,
    )


class _S:
    """Minimal stand-in for Sample with the same reward-component contract."""

    def __init__(self, group_index, reward):
        self.group_index = group_index
        self.reward = reward

    def get_reward_components(self, keys):
        if not isinstance(self.reward, dict):
            raise ValueError("Sample.reward must be a dict")
        values = []
        for key in keys:
            if key not in self.reward:
                raise ValueError(f"Reward key {key!r} missing from sample reward")
            values.append(self.reward[key])
        return values


def _normalize(args, samples):
    spec = get_algorithm(args.advantage_estimator)
    return REWARD_NORMALIZERS[spec.reward_normalizer](args, samples, [0.0] * len(samples))


def _mk(groups, correctness, fmt):
    return [_S(g, {"correctness": c, "format": f}) for g, c, f in zip(groups, correctness, fmt, strict=True)]


def _manual_gdpo(correctness, fmt, groups, weights=(1.0, 1.0)):
    """Plain-Python reference for Eq.

    4 + Eq. 7.
    """
    per_key = []
    for column in (correctness, fmt):
        out = [0.0] * len(column)
        for g in sorted(set(groups)):
            idx = [i for i, gg in enumerate(groups) if gg == g]
            vals = [column[i] for i in idx]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
            scale = max(abs(v) for v in vals)
            collapsed = std <= 1e-6 * scale
            # 1e-4, hardcoded on purpose: this oracle exists to pin parity with
            # the reference implementation (trl grpo_trainer.py's scale_rewards
            # GDPO path divides by std + 1e-4 at both steps), so importing our
            # own constant here would make the test agree with itself.
            for i in idx:
                out[i] = 0.0 if collapsed else (column[i] - mean) / (std + 1e-4)
        per_key.append(out)
    return [weights[0] * a + weights[1] * b for a, b in zip(per_key[0], per_key[1], strict=True)]


# ---------------- registration ----------------


def test_gdpo_is_registered():
    spec = get_algorithm("gdpo")
    assert spec.reward_normalizer == "gdpo_decoupled"
    assert spec.advantage_fn == "gdpo"
    assert spec.policy_loss_fn == "ppo_clip"
    assert spec.kl_level == "token"


def test_gdpo_spec_guards():
    spec = get_algorithm("gdpo")
    assert spec.min_group_size == 2
    assert spec.allows_custom_reward_post_process is False
    assert spec.forbids_normalize_advantages is True
    assert spec.requires_rewards_normalization is True
    assert spec.uses_reward_components is True
    assert spec.disabled_reason is None


# ---------------- step 1 + 2 ----------------


def test_matches_hand_computed_reference():
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    correctness = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    fmt = [1.0, 1.0, 0.0, 0.0, 0.5, 0.25, 0.75, 1.0]
    samples = _mk(groups, correctness, fmt)

    got = _normalize(_args(), samples)
    want = _manual_gdpo(correctness, fmt, groups)

    assert torch.allclose(torch.tensor(got), torch.tensor(want), atol=1e-6)


def test_weights_apply_to_normalized_advantages_not_raw_rewards():
    groups = [0, 0, 0, 0]
    correctness = [1.0, 0.0, 1.0, 0.0]
    fmt = [0.0, 0.0, 10.0, 10.0]
    samples = _mk(groups, correctness, fmt)

    got = _normalize(_args(weights=[2.0, 0.5]), samples)
    want = _manual_gdpo(correctness, fmt, groups, weights=(2.0, 0.5))

    assert torch.allclose(torch.tensor(got), torch.tensor(want), atol=1e-6)


def test_component_scale_does_not_leak_into_the_combination():
    """After step 1 every component is unit-variance, so rescaling one is a no-
    op."""
    groups = [0, 0, 0, 0]
    correctness = [1.0, 0.0, 1.0, 0.0]
    unit = [1.0, 2.0, 3.0, 4.0]
    thousandfold = [1000.0, 2000.0, 3000.0, 4000.0]

    with_unit = _normalize(_args(), _mk(groups, correctness, unit))
    with_large = _normalize(_args(), _mk(groups, correctness, thousandfold))

    # Not exactly a no-op: the additive epsilon does not scale with the data, so
    # a 1000x rescale leaks about eps/std = 1e-4/1.291 = 7.7e-5. Measured 9.0e-5.
    assert torch.allclose(torch.tensor(with_unit), torch.tensor(with_large), atol=1e-4)


def test_eps_makes_scale_invariance_approximate_for_tiny_rewards():
    """Documented limitation of the additive epsilon, shared with the GRPO
    path.

    Dividing by ``std + eps`` is only scale-free while ``std >> eps``, and GDPO
    uses ``eps = 1e-4`` to match the reference implementation rather than the
    ``1e-6`` the GRPO path uses. That choice is not free: a component whose
    spread is ~1e-3 is shrunk by roughly eps/std, which at 1e-4 is **7.2%**
    against 0.08% at 1e-6.

    So a continuous reward with a very narrow spread -- the paper's maths setup
    scores response length -- is damped noticeably more here than a reader
    coming from the GRPO path would expect. Pinning the number is the point;
    if it ever needs to be configurable, this is the test that says why.
    """
    groups = [0, 0, 0, 0]
    correctness = [1.0, 0.0, 1.0, 0.0]

    normal = _normalize(_args(), _mk(groups, correctness, [1.0, 2.0, 3.0, 4.0]))
    tiny = _normalize(_args(), _mk(groups, correctness, [0.001, 0.002, 0.003, 0.004]))

    deviation = (torch.tensor(normal) - torch.tensor(tiny)).abs().max()
    assert 0.07 < float(deviation) < 0.09, f"expected ~7% damping from eps=1e-4, got {float(deviation)}"


def test_default_weights_are_all_ones():
    groups = [0, 0, 0, 0]
    samples = _mk(groups, [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    assert _normalize(_args(weights=None), samples) == _normalize(_args(weights=[1.0, 1.0]), samples)


def test_grouping_uses_group_index():
    correctness = [1.0, 0.0, 1.0, 0.0, 5.0, 5.0, 5.0, 5.0]
    fmt = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    contiguous = _normalize(_args(), _mk([0, 0, 0, 0, 1, 1, 1, 1], correctness, fmt))
    # group 1 has collapsed correctness, so only format contributes there
    assert abs(sum(contiguous[4:])) < 1e-5


# ---------------- reward collapse ----------------


def test_collapsed_component_contributes_zero_but_others_keep_signal():
    """This is GDPO's core benefit over GRPO."""
    groups = [0, 0, 0, 0]
    correctness = [1.0, 1.0, 1.0, 1.0]  # collapsed
    fmt = [1.0, 0.0, 1.0, 0.0]
    samples = _mk(groups, correctness, fmt)

    got = _normalize(_args(), samples)
    fmt_only = _manual_gdpo([0.0] * 4, fmt, groups)

    assert torch.allclose(torch.tensor(got), torch.tensor(fmt_only), atol=1e-6)
    assert max(abs(v) for v in got) > 0.5


def test_fully_collapsed_group_yields_exact_zeros_without_nan():
    samples = _mk([0, 0, 0, 0], [0.7] * 4, [0.7] * 4)
    got = _normalize(_args(), samples)
    assert got == [0.0, 0.0, 0.0, 0.0]


def test_collapsed_group_does_not_leak_fp32_residual():
    """0.7 repeated 7x produces a nonzero residual if you rely on `/(std + eps)` alone."""
    samples = _mk([0] * 7, [0.7] * 7, [0.7] * 7)
    got = _normalize(_args(n=7), samples)
    assert got == [0.0] * 7


@pytest.mark.parametrize("value", [0.0, 0.1, 0.7, 1.0, 1000.0, -3.5])
def test_collapse_is_exact_at_any_magnitude(value):
    samples = _mk([0] * 5, [value] * 5, [value] * 5)
    assert _normalize(_args(n=5), samples) == [0.0] * 5


def _grpo_reference(summed, groups):
    """What GRPO does: sum the components first, then standardise once."""
    out = [0.0] * len(summed)
    for g in sorted(set(groups)):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        vals = [summed[i] for i in idx]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        std = math.sqrt(var)
        for i in idx:
            out[i] = 0.0 if std == 0 else (summed[i] - mean) / (std + 1e-4)
    return out


def test_gdpo_distinguishes_reward_patterns_that_grpo_flattens():
    """The paper's core argument (arXiv 2601.05242, Sec. 3.1).

    With G=2, standardising collapses any two distinct values to +-1/sqrt(2),
    so GRPO maps "one component fires" and "both components fire" to the exact
    same advantage. Normalising per component first keeps the two apart.
    """
    groups = [0, 0]
    one_fires = _mk(groups, [0.0, 1.0], [0.0, 0.0])
    both_fire = _mk(groups, [0.0, 1.0], [0.0, 1.0])

    grpo_one = _grpo_reference([0.0, 1.0], groups)
    grpo_both = _grpo_reference([0.0, 2.0], groups)
    assert torch.allclose(torch.tensor(grpo_one), torch.tensor(grpo_both), atol=1e-4), (
        "GRPO should be unable to tell these apart"
    )

    gdpo_one = _normalize(_args(n=2), one_fires)
    gdpo_both = _normalize(_args(n=2), both_fire)
    assert not torch.allclose(torch.tensor(gdpo_one), torch.tensor(gdpo_both), atol=1e-2)
    # both_fire carries twice the signal: two components at +-1/sqrt(2) each
    assert math.isclose(gdpo_both[1], 2 * gdpo_one[1], rel_tol=1e-4)


# ---------------- error contracts ----------------


def test_missing_reward_key_raises():
    samples = [_S(0, {"correctness": 1.0}) for _ in range(4)]
    with pytest.raises(ValueError, match="format"):
        _normalize(_args(), samples)


def test_non_dict_reward_raises():
    samples = [_S(0, 1.0) for _ in range(4)]
    with pytest.raises(ValueError, match="must be a dict"):
        _normalize(_args(), samples)


def test_non_numeric_reward_raises():
    samples = [_S(0, {"correctness": 1.0, "format": "good"}) for _ in range(4)]
    with pytest.raises(TypeError, match="must be a real number"):
        _normalize(_args(), samples)


def test_bool_reward_raises():
    """bool subclasses int, so it would slip through a naive isinstance
    check."""
    samples = [_S(0, {"correctness": True, "format": 1.0}) for _ in range(4)]
    with pytest.raises(TypeError, match="must be a real number"):
        _normalize(_args(), samples)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_or_inf_reward_raises(bad):
    samples = [_S(0, {"correctness": bad, "format": 1.0}) for _ in range(4)]
    with pytest.raises(ValueError, match="not finite"):
        _normalize(_args(), samples)


def test_fewer_than_two_keys_raises():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="at least two"):
        _normalize(_args(keys=("correctness",)), samples)


def test_no_keys_raises():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="at least two"):
        _normalize(_args(keys=None), samples)


def test_duplicate_keys_raise():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="duplicates"):
        _normalize(_args(keys=("correctness", "correctness")), samples)


def test_weight_count_mismatch_raises():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="gdpo-reward-weights"):
        _normalize(_args(weights=[1.0]), samples)


def test_wrong_group_size_raises():
    samples = _mk([0, 0, 1, 1], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="expected 4"):
        _normalize(_args(n=4), samples)


# ---------------- numerical properties ----------------


def test_group_of_two_always_normalizes_to_plus_minus_one_over_sqrt_two():
    """With G=2 an unbiased std absorbs the magnitude entirely."""
    samples = _mk([0, 0], [0.0, 100.0], [0.0, 1.0])
    got = _normalize(_args(n=2), samples)
    expected = 2 * (1.0 / math.sqrt(2.0))
    assert math.isclose(got[1], expected, rel_tol=1e-4)
    assert math.isclose(got[0], -expected, rel_tol=1e-4)


def test_extract_reward_components_shape_and_dtype():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    out = extract_reward_components(samples, ["correctness", "format"])
    assert out.shape == (4, 2)
    assert out.dtype == torch.float32


def test_extract_reward_components_preserves_key_order():
    samples = _mk([0, 0, 0, 0], [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    forward = extract_reward_components(samples, ["correctness", "format"])
    reversed_ = extract_reward_components(samples, ["format", "correctness"])
    assert torch.equal(forward, reversed_.flip(dims=[1]))


def test_extract_accepts_python_ints():
    samples = [_S(0, {"correctness": 1, "format": 0}) for _ in range(2)]
    out = extract_reward_components(samples, ["correctness", "format"])
    assert out.dtype == torch.float32


def test_single_reward_gdpo_is_a_constant_positive_multiple_of_grpo():
    """Documented deviation: GDPO does NOT reduce to GRPO for one reward."""
    torch.manual_seed(0)
    b, g = 16, 4
    rewards = (torch.rand(b * g) < 0.5).float().view(b, g)
    grpo = ((rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-6)).flatten()
    gdpo = whiten_scalar(grpo)

    mask = grpo.abs() > 1e-6
    ratio = gdpo[mask] / grpo[mask]
    assert ratio.min() > 0
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-4)
    assert not torch.allclose(gdpo, grpo, atol=1e-3)


def test_sample_get_reward_components_matches_the_test_double():
    """The real Sample must honour the same contract as _S above."""
    from relax.utils.types import Sample

    sample = Sample(group_index=0, reward={"correctness": 1.0, "format": 0.5})
    assert sample.get_reward_components(["format", "correctness"]) == [0.5, 1.0]

    with pytest.raises(ValueError, match="missing from sample reward"):
        sample.get_reward_components(["nope"])

    scalar = Sample(group_index=0, reward=1.0)
    with pytest.raises(ValueError, match="must be a dict"):
        scalar.get_reward_components(["correctness"])


def test_warns_once_when_every_component_collapses_in_every_group(caplog):
    """A batch that produces no gradient at all must not be silent."""
    import logging

    samples = _mk([0, 0, 0, 0], [1.0] * 4, [0.5] * 4)
    with caplog.at_level(logging.WARNING):
        out = _normalize(_args(), samples)

    assert out == [0.0] * 4
    assert sum("all reward components collapsed" in r.message for r in caplog.records) == 1


def test_does_not_warn_when_some_signal_survives(caplog):
    import logging

    samples = _mk([0, 0, 0, 0], [1.0] * 4, [1.0, 0.0, 1.0, 0.0])
    with caplog.at_level(logging.WARNING):
        _normalize(_args(), samples)

    assert not any("all reward components collapsed" in r.message for r in caplog.records)


def test_does_not_warn_when_only_some_groups_collapse(caplog):
    import logging

    samples = _mk(
        [0, 0, 0, 0, 1, 1, 1, 1],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 1.0, 0.0],
    )
    with caplog.at_level(logging.WARNING):
        _normalize(_args(), samples)

    assert not any("all reward components collapsed" in r.message for r in caplog.records)


# ---------------- reward value contract ----------------


def test_numpy_scalars_are_accepted():
    """Reward functions routinely return numpy scalars; only float64 subclasses
    float."""
    np = pytest.importorskip("numpy")

    for dtype in (np.float64, np.float32, np.int64, np.int32, np.float16):
        samples = [
            _S(0, {"correctness": dtype(1), "format": dtype(0)}),
            _S(0, {"correctness": dtype(0), "format": dtype(1)}),
        ]
        out = extract_reward_components(samples, ["correctness", "format"])
        assert out.dtype == torch.float32, dtype
        assert out.shape == (2, 2), dtype


def test_numpy_bool_is_still_rejected():
    np = pytest.importorskip("numpy")

    samples = [_S(0, {"correctness": np.bool_(True), "format": 1.0}) for _ in range(4)]
    with pytest.raises(TypeError, match="must be a real number"):
        _normalize(_args(), samples)


def test_numpy_nan_is_still_rejected():
    np = pytest.importorskip("numpy")

    samples = [_S(0, {"correctness": np.float32("nan"), "format": 1.0}) for _ in range(4)]
    with pytest.raises(ValueError, match="not finite"):
        _normalize(_args(), samples)


# ---------------- weight contract ----------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_weights_raise_instead_of_zeroing_the_batch(bad):
    """Left unchecked these produce an all-zero batch with no error at all."""
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="not finite"):
        _normalize(_args(weights=[bad, 1.0]), samples)


# ---------------- collapse detection is exact ----------------


def test_component_with_small_relative_spread_is_kept():
    """A relative tolerance would have erased this component entirely."""
    groups = [0, 0, 0, 0]
    correctness = [10000.0, 10000.005, 10000.010, 10000.015]
    fmt = [1.0, 0.0, 1.0, 0.0]
    got = _normalize(_args(), _mk(groups, correctness, fmt))

    fmt_only = _manual_gdpo([0.0] * 4, fmt, groups)
    assert not torch.allclose(torch.tensor(got), torch.tensor(fmt_only), atol=1e-3)


# ---------------- batch statistics must survive a large offset ----------------
#
# distributed_mean_std used the one-pass E[x^2] - E[x]^2 form in the caller's
# float32. That subtracts two nearly equal large numbers whenever the values sit
# far from zero, and it failed silently in two different directions. Neither is
# hypothetical: the GDPO paper's maths setup uses a length reward, and token
# counts live exactly in this range.


def test_batch_std_survives_a_large_offset():
    """One-pass returned exactly 0 here, which reads as a collapsed batch."""
    from relax.algorithms.numerics import distributed_mean_std

    values = torch.tensor([1000.0, 1000.01, 1000.02, 1000.03])
    _mean, std = distributed_mean_std(values)

    assert std > 0, "a batch with real spread was reported as collapsed"
    # rtol at float32 resolution, not float64: the statistic is computed in
    # float64 but handed back in the caller's dtype.
    torch.testing.assert_close(std.double(), values.double().std(), rtol=1e-6, atol=0)


def test_batch_std_is_not_inflated_by_a_large_offset():
    """One-pass returned 4.6 here against a true std of 1.3e-3."""
    from relax.algorithms.numerics import distributed_mean_std

    values = torch.tensor([10000.0, 10000.001, 10000.002, 10000.003])
    _mean, std = distributed_mean_std(values)

    torch.testing.assert_close(std.double(), values.double().std(), rtol=1e-6, atol=0)
    assert std < 1e-2, f"std inflated to {float(std)}; advantages would be rescaled by ~1/{float(std):.3g}"


@pytest.mark.parametrize("offset", [0.0, 1e2, 1e3, 1e4])
def test_whitening_is_shift_invariant(offset):
    """Whitening centres before scaling, so adding a constant to every value
    must not change the result.

    One-pass broke this well before float32 ran out of significand.
    """
    from relax.algorithms.advantages import whiten_scalar

    base = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    torch.testing.assert_close(whiten_scalar(base + offset), whiten_scalar(base), rtol=1e-5, atol=1e-6)


def test_batch_std_matches_torch_std_on_ordinary_input():
    """The fix must not move the numbers the existing algorithms already
    produce."""
    from relax.algorithms.numerics import distributed_mean_std

    torch.manual_seed(20260726)
    for n in (2, 4, 8, 64):
        values = torch.randn(n)
        mean, std = distributed_mean_std(values)
        torch.testing.assert_close(mean, values.mean(), rtol=1e-6, atol=0)
        torch.testing.assert_close(std, values.std(), rtol=1e-6, atol=0)
