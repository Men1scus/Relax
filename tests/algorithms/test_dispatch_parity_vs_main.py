# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The registry must route each algorithm exactly where main's if/elif did.

Scope note, because it is easy to over-claim here. `relax/utils/training/
ppo_utils.py` is byte-identical to main on this branch, so no estimator's or
policy loss's *maths* changed — every one of them still calls the same function
object it always did. Feeding both implementations the same tensors and
comparing outputs would therefore pass by construction and prove nothing.

What the refactor did change is *routing*: which kernel each algorithm name
resolves to, and which capability flags gate the surrounding code. That is what
these tables pin down. They are transcribed from
main @ 039ce876d25540adad847d4223b4de4722d8f425:

    relax/components/advantages.py      lines 165-208 (advantage if/elif)
    relax/backends/megatron/loss.py     lines 569-613 (the duplicate of it)
    relax/backends/megatron/loss.py     line  825     (need_full_log_probs)
    relax/backends/megatron/loss.py     line  884     (sequence-level KL)
    relax/backends/megatron/loss.py     lines 899-914 (policy loss if/elif)
    relax/utils/utils.py                lines 182,203 (reward normalisation)

Do not regenerate these from the current implementation — that would turn the
comparison into the implementation checking itself.
"""

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms import get_algorithm, list_algorithm_names  # noqa: E402
from relax.algorithms.advantages import ADVANTAGE_FNS  # noqa: E402
from relax.algorithms.policy import POLICY_LOSS_FNS  # noqa: E402
from relax.algorithms.rewards import REWARD_NORMALIZERS  # noqa: E402
from relax.utils.training import ppo_utils  # noqa: E402


MAIN_SHA = "039ce876d25540adad847d4223b4de4722d8f425"

# main: `if estimator in ["grpo", "gspo", "sapo", "cispo"]` -> get_grpo_returns, etc.
MAIN_ADVANTAGE_KERNEL = {
    "grpo": ppo_utils.get_grpo_returns,
    "gspo": ppo_utils.get_grpo_returns,
    "sapo": ppo_utils.get_grpo_returns,
    "cispo": ppo_utils.get_grpo_returns,
    "ppo": ppo_utils.get_advantages_and_returns_batch,
    "reinforce_plus_plus": ppo_utils.get_reinforce_plus_plus_returns,
    "reinforce_plus_plus_baseline": ppo_utils.get_reinforce_plus_plus_baseline_advantages,
}

# main loss.py:899-914
MAIN_POLICY_KERNEL = {
    "grpo": ppo_utils.compute_policy_loss,
    "gspo": ppo_utils.compute_policy_loss,
    "sapo": ppo_utils.compute_sapo_loss,
    "cispo": ppo_utils.compute_cispo_loss,
    "ppo": ppo_utils.compute_policy_loss,
    "reinforce_plus_plus": ppo_utils.compute_policy_loss,
    "reinforce_plus_plus_baseline": ppo_utils.compute_policy_loss,
}

# main loss.py:884 — only gspo took the sequence-level KL branch.
MAIN_SEQUENCE_LEVEL_KL = {"gspo"}

# main loss.py:825 — `args.use_opsm or estimator == "gspo"`.
MAIN_NEEDS_FULL_LOG_PROBS = {"gspo"}

# main arguments.py:2739 — `use_critic = estimator == "ppo"`.
MAIN_NEEDS_CRITIC = {"ppo"}

# main arguments.py:2594 — reinforce_plus_plus{,_baseline} asserted normalize_advantages.
MAIN_REQUIRES_NORMALIZE_ADVANTAGES = {"reinforce_plus_plus", "reinforce_plus_plus_baseline"}

# main utils.py:182 — group-mean whitelist; :203 — the subset that also divides by std.
MAIN_GROUP_NORMALIZED = {"grpo", "gspo", "sapo", "cispo", "reinforce_plus_plus_baseline"}
MAIN_GROUP_STD_NORMALIZED = {"grpo", "gspo", "sapo", "cispo"}

MAIN_ALGORITHMS = sorted(MAIN_ADVANTAGE_KERNEL)


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_advantage_routes_to_the_same_kernel_as_main(name):
    """The estimator's maths is unchanged; only the lookup moved."""
    spec = get_algorithm(name)
    fn = ADVANTAGE_FNS[spec.advantage_fn]
    source = fn.__code__.co_names
    expected = MAIN_ADVANTAGE_KERNEL[name].__name__
    assert expected in source, f"{name} no longer reaches {expected}; it calls {source}"


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_policy_loss_routes_to_the_same_kernel_as_main(name):
    spec = get_algorithm(name)
    fn = POLICY_LOSS_FNS[spec.policy_loss_fn]
    expected = MAIN_POLICY_KERNEL[name].__name__
    assert expected in fn.__code__.co_names, f"{name} no longer reaches {expected}"


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_sequence_level_kl_matches_main(name):
    assert (get_algorithm(name).kl_level == "sequence") is (name in MAIN_SEQUENCE_LEVEL_KL)


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_needs_full_log_probs_matches_main(name):
    assert get_algorithm(name).needs_full_log_probs is (name in MAIN_NEEDS_FULL_LOG_PROBS)


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_needs_critic_matches_main(name):
    assert get_algorithm(name).needs_critic is (name in MAIN_NEEDS_CRITIC)


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_requires_normalize_advantages_matches_main(name):
    spec = get_algorithm(name)
    assert spec.requires_normalize_advantages is (name in MAIN_REQUIRES_NORMALIZE_ADVANTAGES)


@pytest.mark.parametrize("name", MAIN_ALGORITHMS)
def test_group_normalization_matches_main(name):
    """utils.py had two overlapping whitelists; both are now spec fields."""
    normalizer = get_algorithm(name).reward_normalizer
    assert (normalizer != "none") is (name in MAIN_GROUP_NORMALIZED)
    assert (normalizer == "group_mean_std") is (name in MAIN_GROUP_STD_NORMALIZED)


def test_every_algorithm_main_supported_is_still_registered():
    """A migration that quietly dropped an algorithm would pass every other
    test."""
    missing = set(MAIN_ALGORITHMS) - set(list_algorithm_names())
    assert not missing, f"{missing} were reachable on main {MAIN_SHA[:7]} and are gone now"


def test_only_gdpo_was_added():
    """Keeps the reference tables honest: any new algorithm must be listed
    here."""
    added = set(list_algorithm_names()) - set(MAIN_ALGORITHMS)
    assert added == {"gdpo"}, f"unexpected new algorithms {added}; update this file's tables"


def test_reward_normalizer_identifiers_all_resolve():
    for name in list_algorithm_names():
        assert get_algorithm(name).reward_normalizer in REWARD_NORMALIZERS
