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


# ---------------- the adapters must be identity wrappers ----------------
#
# The co_names checks above only prove the kernel's name appears in the adapter's
# bytecode. They would still pass if the adapter scaled its input, dropped an
# argument, or threw the result away. These compare the adapter's output against
# calling the kernel directly with main's argument list, which is what actually
# pins "the wrapper adds nothing".


def _kl(lengths=(3, 2)):
    return [torch.zeros(n, dtype=torch.float32) for n in lengths]


def _masks(lengths=(3, 2)):
    return [torch.ones(n, dtype=torch.float32) for n in lengths]


def _args(estimator, **overrides):
    from types import SimpleNamespace

    base = dict(
        advantage_estimator=estimator,
        kl_coef=0.0,
        gamma=1.0,
        lambd=1.0,
        eps_clip=0.2,
        eps_clip_high=0.3,
        sapo_tau_pos=1.0,
        sapo_tau_neg=1.05,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("name", ["grpo", "gspo", "sapo", "cispo"])
def test_grpo_family_adapter_is_the_bare_kernel(name):
    """main: torch.tensor(rewards, float32, device) then get_grpo_returns(...)."""
    from relax.algorithms.advantages import compute_advantages_and_returns

    rewards, kl = [1.5, -2.0], _kl()
    got, _ = compute_advantages_and_returns(_args(name), rewards=rewards, kl=kl, loss_masks=_masks())
    want = ppo_utils.get_grpo_returns(torch.tensor(rewards, dtype=torch.float32, device=kl[0].device), kl)

    assert len(got) == len(want)
    for left, right in zip(got, want, strict=True):
        assert torch.equal(left, right), name


def test_reinforce_plus_plus_baseline_adapter_is_the_bare_kernel():
    from relax.algorithms.advantages import compute_advantages_and_returns

    rewards, kl, masks = [3.0], [torch.tensor([2.0, 4.0])], [torch.ones(2)]
    args = _args("reinforce_plus_plus_baseline", kl_coef=0.5)

    got, returns = compute_advantages_and_returns(args, rewards=rewards, kl=kl, loss_masks=masks)
    want = ppo_utils.get_reinforce_plus_plus_baseline_advantages(
        rewards=torch.tensor(rewards, dtype=torch.float32, device=kl[0].device),
        kl=[torch.tensor([2.0, 4.0])],
        loss_masks=masks,
        kl_coef=0.5,
    )

    for left, right in zip(got, want, strict=True):
        assert torch.equal(left, right)
    assert returns is got, "main aliased returns to advantages for this estimator"


def _loss_inputs():
    torch.manual_seed(20260726)
    return torch.randn(8), torch.randn(8), torch.randn(8)


def test_ppo_clip_adapter_passes_mains_arguments():
    from relax.algorithms.policy import compute_policy_loss_for

    log_probs, ppo_kl, advantages = _loss_inputs()
    args = _args("grpo")
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = ppo_utils.compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_sapo_adapter_passes_mains_arguments():
    from relax.algorithms.policy import compute_policy_loss_for

    log_probs, ppo_kl, advantages = _loss_inputs()
    args = _args("sapo", sapo_tau_pos=1.3, sapo_tau_neg=1.7)
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = ppo_utils.compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=1.3, tau_neg=1.7)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_sapo_adapter_uses_mains_defaults_when_args_omit_the_taus():
    from types import SimpleNamespace

    from relax.algorithms.policy import compute_policy_loss_for

    log_probs, ppo_kl, advantages = _loss_inputs()
    bare = SimpleNamespace(advantage_estimator="sapo", eps_clip=0.2, eps_clip_high=0.3)
    got = compute_policy_loss_for(bare, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = ppo_utils.compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=1.0, tau_neg=1.05)
    assert torch.equal(got[0], want[0])


def test_cispo_adapter_passes_mains_arguments():
    """The one adapter taking four kernel arguments — most room to drop one."""
    from relax.algorithms.policy import compute_policy_loss_for

    log_probs, ppo_kl, advantages = _loss_inputs()
    args = _args("cispo", eps_clip=0.15, eps_clip_high=9.0)
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = ppo_utils.compute_cispo_loss(
        log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages, eps_clip=0.15, eps_clip_high=9.0
    )
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


@pytest.fixture
def cp_disabled(monkeypatch):
    """Minimal megatron.core.mpu so the reinforce++ kernel runs on CPU.

    get_reinforce_plus_plus_returns imports mpu inside the function and reads
    only get_context_parallel_world_size(); at 1 it takes the non-gathering
    branch, which is the configuration the rest of this file already assumes.
    Stubbing it is what makes the adapter testable at all -- the alternative is
    leaving the one selectable estimator with no numerical check, which is how
    it got here.
    """
    import sys
    import types

    core = types.ModuleType("megatron.core")
    core.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
    )
    megatron = types.ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    yield


def test_reinforce_plus_plus_adapter_is_the_bare_kernel(cp_disabled):
    """The one live estimator whose adapter had only a co_names check.

    Coverage said it plainly: advantage_reinforce_plus_plus was never executed
    by any test, so nothing would have caught the adapter dropping an argument
    or reordering the keyword-only ones -- and unlike gae, this estimator is
    selectable (no disabled_reason).
    """
    from relax.algorithms.advantages import compute_advantages_and_returns

    rewards = [1.5, -2.0]
    kl = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])]
    loss_masks = [torch.ones(3), torch.ones(2)]
    response_lengths, total_lengths = [3, 2], [5, 4]
    args = _args("reinforce_plus_plus", kl_coef=0.3, gamma=0.95)

    got, returns = compute_advantages_and_returns(
        args,
        rewards=rewards,
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
    )
    want = ppo_utils.get_reinforce_plus_plus_returns(
        rewards=torch.tensor(rewards, dtype=torch.float32, device=kl[0].device),
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        kl_coef=0.3,
        gamma=0.95,
    )

    assert len(got) == len(want)
    for left, right in zip(got, want, strict=True):
        assert torch.equal(left, right)
    assert returns is not got, "main copied the list before returning it"


def test_reinforce_plus_plus_adapter_forwards_gamma_and_kl_coef(cp_disabled):
    """Both are read off args rather than passed through, so a swap or a
    hardcoded default would survive the identity test above if it used the
    kernel's defaults."""
    from relax.algorithms.advantages import compute_advantages_and_returns

    inputs = dict(
        rewards=[1.0, 1.0],
        kl=[torch.tensor([0.5, 0.5]), torch.tensor([0.5])],
        loss_masks=[torch.ones(2), torch.ones(1)],
        response_lengths=[2, 1],
        total_lengths=[3, 2],
    )
    low, _ = compute_advantages_and_returns(_args("reinforce_plus_plus", kl_coef=0.0, gamma=1.0), **inputs)
    high, _ = compute_advantages_and_returns(_args("reinforce_plus_plus", kl_coef=0.9, gamma=1.0), **inputs)
    assert not torch.equal(low[0], high[0]), "kl_coef is not reaching the kernel"

    g_one, _ = compute_advantages_and_returns(_args("reinforce_plus_plus", kl_coef=0.0, gamma=1.0), **inputs)
    g_half, _ = compute_advantages_and_returns(_args("reinforce_plus_plus", kl_coef=0.0, gamma=0.5), **inputs)
    assert not torch.equal(g_one[0], g_half[0]), "gamma is not reaching the kernel"


def test_loss_py_actually_forwards_the_mini_batch_boundaries():
    """Source-level, because loss.py needs megatron to import.

    The per-batch whitening tests all call advantage_gdpo directly, so removing
    the wiring in loss.py left every one of them green while GDPO silently went
    back to whitening the merged rollout. This is the assertion that fails when
    that happens.
    """
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parents[2] / "relax" / "backends" / "megatron" / "loss.py").read_text(
        encoding="utf-8"
    )

    call = re.search(r"compute_advantages_and_returns_impl\((.*?)\n    \)", src, re.DOTALL)
    assert call, "compute_advantages_and_returns_impl call not found"
    assert "mini_batch_sizes=rollout_data.get(ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY)" in call.group(1), (
        "loss.py must pass the per-training-batch counts; without them GDPO's step 3 "
        "normalises over the whole merged rollout again"
    )
