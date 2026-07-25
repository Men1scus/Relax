# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward normalisation strategies, one per algorithm family.

These run on the rollout side (CPU) from
``relax.utils.utils.post_process_rewards``.  Every normaliser takes the raw
per-sample scalar rewards and returns the per-sample scalars written into the
TransferQueue ``rewards`` column, so adding a strategy never changes the data
schema — even for multi-reward algorithms, which collapse their components to a
single scalar here.
"""

from typing import Any, Callable

import torch


GROUP_EPS = 1e-6
ZERO_VARIANCE_THRESHOLD = 1e-12


def group_positions(samples: list[Any], expected_size: int) -> dict[int, list[int]]:
    """Map ``Sample.group_index`` to the positions it occupies in ``samples``.

    Grouping follows ``group_index`` rather than position, so how the caller
    orders the batch does not affect the result.
    """
    positions_by_group: dict[int, list[int]] = {}
    for position, sample in enumerate(samples):
        if sample.group_index is None:
            raise ValueError("Sample.group_index is required for group reward normalization.")
        if sample.group_index not in positions_by_group:
            positions_by_group[sample.group_index] = []
        positions_by_group[sample.group_index].append(position)

    for group_index, positions in positions_by_group.items():
        if len(positions) != expected_size:
            raise ValueError(f"Reward group {group_index} has {len(positions)} samples, expected {expected_size}.")
    return positions_by_group


def _group_normalize(args: Any, samples: list[Any], raw_rewards: list[float], *, use_std: bool) -> list[float]:
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    positions_by_group = group_positions(samples, args.n_samples_per_prompt)

    normalized_rewards = torch.empty_like(rewards)
    for positions in positions_by_group.values():
        group_rewards = rewards[positions]
        group_rewards = group_rewards - group_rewards.mean()
        if use_std:
            group_rewards = group_rewards / (group_rewards.std() + GROUP_EPS)
        normalized_rewards[positions] = group_rewards

    return normalized_rewards.tolist()


def normalize_none(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """No normalisation — the estimator consumes raw rewards (REINFORCE++,
    PPO)."""
    return raw_rewards


def normalize_group_mean(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """Subtract the group mean only (REINFORCE++ baseline)."""
    return _group_normalize(args, samples, raw_rewards, use_std=False)


def normalize_group_mean_std(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """Subtract the group mean, then optionally divide by the group std.

    ``--disable-grpo-std-normalization`` (Dr.GRPO) turns the division off.
    """
    return _group_normalize(args, samples, raw_rewards, use_std=args.grpo_std_normalization)


REWARD_NORMALIZERS: dict[str, Callable[[Any, list[Any], list[float]], list[float]]] = {
    "none": normalize_none,
    "group_mean": normalize_group_mean,
    "group_mean_std": normalize_group_mean_std,
}
