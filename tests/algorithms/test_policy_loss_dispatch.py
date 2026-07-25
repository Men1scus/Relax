# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Policy loss selection must come from the registry."""

import pathlib
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.policy import POLICY_LOSS_FNS, compute_policy_loss_for  # noqa: E402
from relax.algorithms.spec import get_algorithm, list_algorithm_names  # noqa: E402
from relax.utils.training.ppo_utils import (  # noqa: E402
    compute_cispo_loss,
    compute_policy_loss,
    compute_sapo_loss,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOSS_PATH = REPO_ROOT / "relax" / "backends" / "megatron" / "loss.py"
SERVE_PATH = REPO_ROOT / "relax" / "components" / "advantages.py"


def _args(estimator, **overrides):
    base = dict(
        advantage_estimator=estimator,
        eps_clip=0.2,
        eps_clip_high=0.2,
        sapo_tau_pos=1.0,
        sapo_tau_neg=1.05,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _tensors():
    torch.manual_seed(0)
    return torch.randn(8), torch.randn(8), torch.randn(8)


# ---------------- registry ----------------


def test_registry_has_all_three_losses():
    assert set(POLICY_LOSS_FNS) == {"ppo_clip", "sapo", "cispo"}


def test_every_spec_policy_loss_id_is_registered():
    for name in list_algorithm_names():
        assert get_algorithm(name).policy_loss_fn in POLICY_LOSS_FNS


# ---------------- adapters match their kernels ----------------


def test_ppo_clip_matches_the_underlying_kernel():
    log_probs, ppo_kl, advantages = _tensors()
    args = _args("grpo")
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


def test_sapo_matches_the_underlying_kernel():
    log_probs, ppo_kl, advantages = _tensors()
    got = compute_policy_loss_for(_args("sapo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=1.0, tau_neg=1.05)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


def test_cispo_matches_the_underlying_kernel():
    log_probs, ppo_kl, advantages = _tensors()
    got = compute_policy_loss_for(_args("cispo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_cispo_loss(
        log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages, eps_clip=0.2, eps_clip_high=0.2
    )
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


def test_sapo_defaults_when_args_lack_tau_fields():
    log_probs, ppo_kl, advantages = _tensors()
    args = SimpleNamespace(advantage_estimator="sapo", eps_clip=0.2, eps_clip_high=0.2)
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=1.0, tau_neg=1.05)
    assert torch.equal(got[0], want[0])


def test_sapo_taus_are_read_from_args():
    log_probs, ppo_kl, advantages = _tensors()
    args = _args("sapo", sapo_tau_pos=2.0, sapo_tau_neg=3.0)
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=2.0, tau_neg=3.0)
    assert torch.equal(got[0], want[0])


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "gdpo", "reinforce_plus_plus"])
def test_ppo_clip_family_share_one_loss(estimator):
    log_probs, ppo_kl, advantages = _tensors()
    reference = compute_policy_loss_for(_args("grpo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    actual = compute_policy_loss_for(_args(estimator), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    assert torch.equal(reference[0], actual[0])


# ---------------- call sites no longer branch on names ----------------


def test_loss_py_no_longer_branches_on_estimator_names():
    src = LOSS_PATH.read_text(encoding="utf-8")
    for banned in (
        'args.advantage_estimator == "gspo"',
        'args.advantage_estimator == "sapo"',
        'args.advantage_estimator == "cispo"',
        'args.advantage_estimator == "ppo"',
        "args.advantage_estimator in [",
    ):
        assert banned not in src, f"loss.py still contains: {banned}"


def test_serve_path_no_longer_branches_on_estimator_names():
    src = SERVE_PATH.read_text(encoding="utf-8")
    for banned in (
        "self.config.advantage_estimator == ",
        "self.config.advantage_estimator in [",
    ):
        assert banned not in src, f"components/advantages.py still contains: {banned}"


def test_both_paths_delegate_to_the_shared_estimator():
    for path in (LOSS_PATH, SERVE_PATH):
        src = path.read_text(encoding="utf-8")
        assert "from relax.algorithms.advantages import" in src, f"{path.name} does not use the shared estimator"


def test_loss_py_reads_kl_level_and_full_log_probs_from_the_spec():
    src = LOSS_PATH.read_text(encoding="utf-8")
    assert 'kl_level == "sequence"' in src
    assert "needs_full_log_probs" in src


def test_neither_call_site_still_raises_not_implemented_for_estimators():
    for path in (LOSS_PATH, SERVE_PATH):
        src = path.read_text(encoding="utf-8")
        assert "advantage_estimator {" not in src, f"{path.name} still formats an estimator into NotImplementedError"
