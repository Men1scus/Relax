# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Policy loss variants behind one uniform signature.

The kernels in :mod:`relax.utils.training.ppo_utils` take different argument
lists.  These adapters normalise them to ``fn(args, *, log_probs, ppo_kl,
advantages)`` so the caller can look one up by name instead of branching on the
algorithm.  Adding a variant means adding an adapter and one registry entry; no
call site changes.
"""

from typing import Any, Callable

import torch

from relax.algorithms.spec import get_algorithm
from relax.utils.training.ppo_utils import compute_cispo_loss, compute_policy_loss, compute_sapo_loss


def policy_loss_ppo_clip(args: Any, *, log_probs, ppo_kl, advantages):
    """Standard clipped surrogate objective (GRPO, GSPO, GDPO, REINFORCE++)."""
    return compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)


def policy_loss_sapo(args: Any, *, log_probs, ppo_kl, advantages):
    """Smooth trust region: sigmoid gating instead of a hard clip."""
    return compute_sapo_loss(
        ppo_kl=ppo_kl,
        advantages=advantages,
        tau_pos=getattr(args, "sapo_tau_pos", 1.0),
        tau_neg=getattr(args, "sapo_tau_neg", 1.05),
    )


def policy_loss_cispo(args: Any, *, log_probs, ppo_kl, advantages):
    """Clipped importance ratio that preserves the gradient direction."""
    return compute_cispo_loss(
        log_probs=log_probs,
        ppo_kl=ppo_kl,
        advantages=advantages,
        eps_clip=args.eps_clip,
        eps_clip_high=args.eps_clip_high,
    )


POLICY_LOSS_FNS: dict[str, Callable[..., tuple[torch.Tensor, torch.Tensor]]] = {
    "ppo_clip": policy_loss_ppo_clip,
    "sapo": policy_loss_sapo,
    "cispo": policy_loss_cispo,
}


def compute_policy_loss_for(args: Any, *, log_probs, ppo_kl, advantages):
    """Dispatch to the policy loss registered for
    ``args.advantage_estimator``."""
    spec = get_algorithm(args.advantage_estimator)
    return POLICY_LOSS_FNS[spec.policy_loss_fn](args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
