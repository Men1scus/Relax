# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""post_process_rewards must dispatch through the registry, not an if/elif
chain."""

import inspect
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import relax.utils.utils as utils_mod  # noqa: E402


def _args(estimator="grpo", **overrides):
    base = dict(
        advantage_estimator=estimator,
        n_samples_per_prompt=4,
        rewards_normalization=True,
        grpo_std_normalization=True,
        custom_reward_post_process_path=None,
        reward_key=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _Sample:
    def __init__(self, group_index, reward):
        self.group_index = group_index
        self.reward = reward

    def get_reward_value(self, args):
        return self.reward if not args.reward_key else self.reward[args.reward_key]


def test_source_has_no_algorithm_name_literals():
    """The estimator whitelists must be gone from post_process_rewards."""
    src = inspect.getsource(utils_mod.post_process_rewards)
    for banned in ('"grpo"', '"gspo"', '"sapo"', '"cispo"', '"reinforce_plus_plus_baseline"'):
        assert banned not in src, f"post_process_rewards still hardcodes {banned}"


def test_debug_subsample_source_has_no_algorithm_name_literals():
    src = inspect.getsource(utils_mod.get_debug_data)
    for banned in ('"grpo"', '"gspo"', '"sapo"', '"cispo"', '"reinforce_plus_plus_baseline"'):
        assert banned not in src, f"get_debug_data still hardcodes {banned}"


def test_returns_raw_and_normalized():
    args = _args("grpo")
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, normalized = utils_mod.post_process_rewards(args, samples)
    assert raw == [0.0, 1.0, 2.0, 3.0]
    assert normalized != raw
    assert abs(sum(normalized)) < 1e-5


def test_identity_path_returns_raw_twice():
    args = _args("reinforce_plus_plus")
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, normalized = utils_mod.post_process_rewards(args, samples)
    assert normalized is raw


def test_rewards_normalization_off_returns_raw_twice():
    args = _args("grpo", rewards_normalization=False)
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, normalized = utils_mod.post_process_rewards(args, samples)
    assert normalized is raw


def test_baseline_estimator_centres_without_dividing_by_std():
    args = _args("reinforce_plus_plus_baseline")
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    _, normalized = utils_mod.post_process_rewards(args, samples)
    assert normalized == pytest.approx([-1.5, -0.5, 0.5, 1.5])


def test_custom_path_still_short_circuits(monkeypatch):
    sentinel = (["raw"], ["norm"])
    monkeypatch.setattr(utils_mod, "load_function", lambda path: lambda a, s: sentinel)
    args = _args("grpo", custom_reward_post_process_path="pkg.mod.fn")
    assert utils_mod.post_process_rewards(args, []) is sentinel


def test_reward_key_selects_from_dict():
    args = _args("grpo", reward_key="score")
    samples = [_Sample(0, {"score": r, "other": 99.0}) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, _ = utils_mod.post_process_rewards(args, samples)
    assert raw == [0.0, 1.0, 2.0, 3.0]


def test_unknown_estimator_raises_from_the_registry():
    args = _args("not_an_algorithm")
    samples = [_Sample(0, 1.0) for _ in range(4)]
    with pytest.raises(KeyError, match="Unknown advantage estimator"):
        utils_mod.post_process_rewards(args, samples)
