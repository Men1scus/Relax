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
            for i in idx:
                out[i] = 0.0 if collapsed else (column[i] - mean) / (std + 1e-6)
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

    assert torch.allclose(torch.tensor(with_unit), torch.tensor(with_large), atol=1e-5)


def test_eps_makes_scale_invariance_approximate_for_tiny_rewards():
    """Documented limitation of the additive epsilon, shared with the GRPO
    path.

    Dividing by ``std + 1e-6`` is only scale-free while ``std >> eps``.  A
    reward component whose spread is ~1e-3 is shrunk by roughly eps/std, here
    ~0.08%. Rewards that small are unusual, but the behaviour should be pinned
    down rather than discovered later.
    """
    groups = [0, 0, 0, 0]
    correctness = [1.0, 0.0, 1.0, 0.0]

    normal = _normalize(_args(), _mk(groups, correctness, [1.0, 2.0, 3.0, 4.0]))
    tiny = _normalize(_args(), _mk(groups, correctness, [0.001, 0.002, 0.003, 0.004]))

    assert not torch.allclose(torch.tensor(normal), torch.tensor(tiny), atol=1e-5)
    assert torch.allclose(torch.tensor(normal), torch.tensor(tiny), atol=5e-3)


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
    """0.7 repeated 7x produces +-5.6e-2 if you rely on `/(std + 1e-6)` alone."""
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
            out[i] = 0.0 if std == 0 else (summed[i] - mean) / (std + 1e-6)
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
