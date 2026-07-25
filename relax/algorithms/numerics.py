# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared numerical constants and guards for the algorithm implementations.

Both the reward stage and the advantage stage standardise values by dividing by
a standard deviation, so they need the same epsilon and the same notion of
"this group carries no signal".  Keeping those here prevents the two stages
from drifting apart.
"""

import torch
import torch.distributed as dist

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_LOGGED_GROUP_SIZE = False


def _log_group_once(process_group: dist.ProcessGroup | None) -> None:
    """Report the reduction group once per process.

    Whether a batch statistic is global or per-shard is invisible in the loss
    curve: a misconfigured run where tensor parallelism ate the extra GPUs
    leaves the data-parallel group at size 1, the all-reduce becomes an
    identity, and everything still trains. This line is what makes that
    distinguishable in a log.
    """
    global _LOGGED_GROUP_SIZE
    if _LOGGED_GROUP_SIZE:
        return
    _LOGGED_GROUP_SIZE = True
    if process_group is None:
        logger.info("Batch statistics are local (no process group); the caller owns the whole batch.")
    else:
        logger.info(
            "Batch statistics reduce over dp_world=%d (this rank is dp_rank=%d).",
            dist.get_world_size(process_group),
            dist.get_rank(process_group),
        )


STD_EPS = 1e-6
"""Epsilon added to a standard deviation before dividing by it.

Matches the value the pre-registry GRPO path used, so group standardisation is
numerically identical across every algorithm in this repository.  Upstream
implementations disagree (the GDPO paper writes no epsilon at all, TRL uses
1e-4, verl 1e-6, ms-swift 1e-8); internal consistency wins over cross-framework
parity because it is what makes the equivalence tests meaningful.
"""


def is_collapsed(values: torch.Tensor, *, process_group: dist.ProcessGroup | None = None) -> bool:
    """Whether every value is identical, i.e. the spread carries no signal.

    Tested by exact equality rather than by comparing the standard deviation
    against a tolerance.  A tolerance has to be relative to the magnitude (the
    mean of N equal float32 values does not come back exactly equal to them, so
    a collapsed group still shows std ~= 1e-8 times its magnitude), and any
    relative tolerance large enough to catch that also erases real signal: with
    ``std <= 1e-6 * max|x|``, the perfectly informative batch
    ``[10000, 10000.005, 10000.010, 10000.015]`` is thrown away.  Exact equality
    has no such false positives, and near-equality is already damped by
    :data:`STD_EPS`.
    """
    if process_group is None:
        if values.numel() == 0:
            return True
        return bool(values.min() == values.max())

    # Every rank in the group has to reach the collective, including one whose
    # shard came out empty — returning early there would hang the others. An
    # empty shard contributes -inf to both halves, which is the identity for MAX
    # and therefore leaves the reduction to the ranks that do have samples.
    empty = values.numel() == 0
    neg_infinity = torch.tensor(float("-inf"), dtype=values.dtype, device=values.device)
    bounds = torch.stack(
        [neg_infinity if empty else -values.min(), neg_infinity if empty else values.max()],
    )
    dist.all_reduce(bounds, op=dist.ReduceOp.MAX, group=process_group)
    low, high = -bounds[0], bounds[1]
    if not torch.isfinite(low):
        return True  # every rank was empty
    return bool(low == high)


def collapsed_columns(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Per-column version of :func:`is_collapsed` for a ``[G, K]`` group.

    Returns a boolean tensor of shape ``[K]``: ``True`` where that reward
    component took the same value across the whole group.
    """
    return values.amax(dim=dim) == values.amin(dim=dim)


def distributed_mean_std(
    values: torch.Tensor, *, process_group: dist.ProcessGroup | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and unbiased std of ``values``, optionally across
    ``process_group``.

    Each rank holds its own shard of the batch, so a local ``std()`` would give
    every rank a different scale factor.  Reducing ``count/sum/sumsq`` makes the
    statistics describe the whole batch, which is what the framework already
    does for ``--normalize-advantages`` (see
    ``relax.utils.distributed_utils.distributed_masked_whiten``).
    """
    _log_group_once(process_group)

    total = values.sum()
    total_sq = (values * values).sum()
    count = torch.tensor(float(values.numel()), dtype=values.dtype, device=values.device)

    if process_group is not None:
        stats = torch.stack([count, total, total_sq])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=process_group)
        count, total, total_sq = stats[0], stats[1], stats[2]

    if count == 0:
        zero = torch.zeros((), dtype=values.dtype, device=values.device)
        return zero, zero

    mean = total / count
    # Bessel-corrected, matching torch.std()'s default so the single-reward GDPO
    # scale factor stays derivable from the GRPO group statistics.
    variance = (total_sq - count * mean * mean) / torch.clamp(count - 1, min=1.0)
    return mean, torch.sqrt(torch.clamp(variance, min=0.0))
