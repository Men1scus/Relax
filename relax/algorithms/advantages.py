# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Advantage estimators as pure functions shared by both execution paths.

The colocate path calls these from ``relax.backends.megatron.loss`` inside the
Megatron worker; the fully-async path calls them from the
``relax.components.advantages`` Ray Serve deployment.  Keeping the maths here
means the two call sites differ only in what surrounds them: the pipeline-stage
early return, in-place write-back versus nested-tensor packing, and the
optional advantage whitening.

Note that the group-wise reward standardisation happened earlier, on the
rollout side (see :mod:`relax.algorithms.rewards`).  By the time an estimator
runs, ``rewards`` already holds one normalised scalar per sample.
"""

from typing import Any, Callable

import torch
import torch.distributed as dist

from relax.algorithms.numerics import GDPO_EPS, distributed_mean_std, is_collapsed
from relax.algorithms.spec import get_algorithm
from relax.utils.training.ppo_utils import (
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)


def whiten_scalar(values: torch.Tensor, *, process_group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """Sequence-level whitening of one scalar per sample.

    This is GDPO's batch-wise normalisation (arXiv 2601.05242, Eq. 6).  It is
    deliberately *not* the token-level ``distributed_masked_whiten`` used by
    ``--normalize-advantages``: weighting by token count would let long
    responses dominate the statistics, which Eq. 6 does not do.

    ``process_group`` must be supplied wherever the caller holds only a shard of
    the batch.  Each data-parallel rank owns ``global_batch_size / dp_size``
    samples, so whitening locally would give every rank its own mean and scale —
    not the "one global scale factor" the maths assumes.  Callers that already
    hold the whole batch (the single-replica Ray Serve deployment) pass ``None``.

    A batch where every value is identical returns exact zeros; see
    :func:`relax.algorithms.numerics.is_collapsed`.
    """
    if is_collapsed(values, process_group=process_group):
        return torch.zeros_like(values)
    mean, std = distributed_mean_std(values, process_group=process_group)
    if not torch.isfinite(std):
        return torch.zeros_like(values)
    return (values - mean) / (std + GDPO_EPS)


def _as_reward_tensor(rewards: Any, kl: list[torch.Tensor]) -> torch.Tensor:
    if isinstance(rewards, torch.Tensor):
        return rewards.to(dtype=torch.float32, device=kl[0].device)
    return torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)


def advantage_grpo_broadcast(args: Any, *, rewards, kl, **_unused):
    """Broadcast the already group-normalised scalar reward over tokens."""
    reward_tensor = _as_reward_tensor(rewards, kl)
    returns = get_grpo_returns(reward_tensor, kl)
    advantages = list(returns)  # separate list so rebinding one does not move the other
    return advantages, returns


def advantage_gdpo(args: Any, *, rewards, kl, process_group=None, **_unused):
    """GDPO step 3: whiten the combined per-sample advantage, then broadcast.

    Steps 1 and 2 (per-reward group standardisation and the weighted sum) ran
    on the rollout side, so ``rewards`` already holds one ``A_sum`` per sample.
    Doing step 3 here rather than there keeps it out of reach of ``--custom-
    reward-post-process-path``, which short-circuits reward post-processing
    entirely, and off the streaming transfer-batch boundary with its undersized
    tail flushes.
    """
    reward_tensor = _as_reward_tensor(rewards, kl)
    returns = get_grpo_returns(whiten_scalar(reward_tensor, process_group=process_group), kl)
    advantages = list(returns)
    return advantages, returns


def advantage_reinforce_plus_plus(args: Any, *, rewards, kl, loss_masks, response_lengths, total_lengths, **_unused):
    """Discounted returns for REINFORCE++
    (https://arxiv.org/pdf/2501.03262)."""
    reward_tensor = _as_reward_tensor(rewards, kl)
    returns = get_reinforce_plus_plus_returns(
        rewards=reward_tensor,
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        kl_coef=args.kl_coef,
        gamma=args.gamma,
    )
    advantages = list(returns)
    return advantages, returns


def advantage_reinforce_plus_plus_baseline(args: Any, *, rewards, kl, loss_masks, **_unused):
    """REINFORCE++ with a group baseline already subtracted upstream."""
    reward_tensor = _as_reward_tensor(rewards, kl)
    advantages = get_reinforce_plus_plus_baseline_advantages(
        rewards=reward_tensor,
        kl=kl,
        loss_masks=loss_masks,
        kl_coef=args.kl_coef,
    )
    # NOTE(dev): returns aliases the same list object, matching the pre-refactor
    # behaviour. On-policy distillation rebinds slots of the advantages list and
    # both views are expected to observe that.
    return advantages, advantages


def advantage_gae(args: Any, *, rewards, kl, values, response_lengths, total_lengths, **_unused):
    """Generalised advantage estimation (PPO).

    Unreachable in practice: the spec carries ``disabled_reason`` and argument
    validation rejects the estimator before training starts.
    """
    from megatron.core import mpu

    shaped_rewards = []
    cp_rank = mpu.get_context_parallel_rank()
    for reward, k in zip(rewards, kl, strict=False):
        k *= -args.kl_coef
        if cp_rank == 0:
            k[-1] += reward
        shaped_rewards.append(k)
    return get_advantages_and_returns_batch(
        total_lengths, response_lengths, values, shaped_rewards, args.gamma, args.lambd
    )


ADVANTAGE_FNS: dict[str, Callable[..., tuple[list[torch.Tensor], list[torch.Tensor]]]] = {
    "grpo_broadcast": advantage_grpo_broadcast,
    "gdpo": advantage_gdpo,
    "reinforce_plus_plus": advantage_reinforce_plus_plus,
    "reinforce_plus_plus_baseline": advantage_reinforce_plus_plus_baseline,
    "gae": advantage_gae,
}


def compute_advantages_and_returns(
    args: Any,
    *,
    rewards,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor] | None = None,
    response_lengths: list[int] | None = None,
    total_lengths: list[int] | None = None,
    values: list[torch.Tensor] | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Dispatch to the estimator registered for ``args.advantage_estimator``.

    ``process_group`` is the group across which the batch is sharded, or
    ``None`` when the caller holds every sample. Estimators that compute batch-
    level statistics need it to see the whole batch.
    """
    spec = get_algorithm(args.advantage_estimator)
    fn = ADVANTAGE_FNS[spec.advantage_fn]
    return fn(
        args,
        rewards=rewards,
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        values=values,
        process_group=process_group,
    )
