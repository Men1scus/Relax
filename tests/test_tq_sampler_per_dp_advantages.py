# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reproduce the argument mismatch between the Advantages deployment and
``SeqlenBalancedSampler``.

This file only demonstrates the problem. It does not fix it.

``relax/components/advantages.py`` fetches with::

    self.data_system_client.async_get_meta(
        data_fields=...,
        batch_size=self.config.global_batch_size // self.config.num_iters_per_train_update,
        partition_id=f"train_{step}",
        task_name="compute_advantages_and_returns",
    )

and **no** ``sampling_config``. Every other consumer (rollout, actor,
actor_fwd) supplies one. The controller installs a single sampler shared by all
of them, so under ``--balance-data`` -- which
``examples/gdpo/run-qwen3-0.6B-1xgpu-gdpo.sh`` sets -- that fetch is served by
``SeqlenBalancedSampler``, whose contract it does not satisfy: the sampler
documents ``batch_size`` as **per data-parallel rank** and requires ``dp_rank``
and ``batch_index`` in kwargs, both of which default to 0 when absent.

This matters beyond GDPO. The deployment computes advantages for every
estimator, so any algorithm run with ``--fully-async --balance-data`` and
without ``--use-dynamic-batch-size`` takes this path.

The three tests below were written *after* measuring the pinned build rather
than from reading it: an earlier draft asserted that the fetch comes back short
and that distinct ``batch_index`` values yield distinct batches, and both turned
out to be false. What follows is what the sampler actually does.

``transfer_queue`` is not on PyPI -- it is pinned to a git SHA in
``docker/Dockerfile`` and exists only in the training image -- so these skip
anywhere else. Run them inside that image; ``relax-modal/repro_tq_sampler_app.py``
does exactly that.
"""

import pytest


try:
    from transfer_queue.sampler.seqlen_balanced_sampler import SeqlenBalancedSampler

    HAS_TQ = True
except ImportError:  # pragma: no cover - depends on the runner
    SeqlenBalancedSampler = None
    HAS_TQ = False

requires_tq = pytest.mark.skipif(not HAS_TQ, reason="transfer_queue is only installed in the training image")

# What relax/components/advantages.py passes, verbatim.
TASK_NAME = "compute_advantages_and_returns"
PARTITION_ID = "train_0"


class _Partition:
    """Minimal stand-in for DataPartitionStatus.

    ``seqlen_balanced_sampler.py:133-136`` hands the whole selected list to
    ``get_custom_meta`` and then indexes the result, so this returns a mapping
    rather than one sample's metadata.
    """

    def __init__(self, n):
        self._n = n

    def get_custom_meta(self, indexes):
        # Uneven lengths, so the balancing step has something to balance.
        return {i: {"total_lengths": 128 + (i % 5) * 64} for i in indexes}


def _fetch(sampler, ready, batch_size, **overrides):
    """Issue the fetch exactly as the Advantages deployment issues it."""
    kwargs = dict(task_name=TASK_NAME, partition_id=PARTITION_ID, partition=_Partition(len(ready)))
    kwargs.update(overrides)
    return sampler.sample(ready, batch_size, **kwargs)


@requires_tq
@pytest.mark.parametrize(
    ("dp_size", "pool", "expected"),
    [
        (4, 32, 0),  # 16 * 4 = 64 wanted globally, only 32 ready -> nothing
        (4, 64, 16),  # exactly enough
        (2, 32, 16),  # 16 * 2 = 32 ready -> the deployment's own share
        (1, 32, 16),  # single rank: the multiplier is 1, so no shortfall
    ],
)
def test_fetch_yields_nothing_until_batch_size_times_dp_size_is_ready(dp_size, pool, expected):
    """The deployment asks for B and gets 0 unless B * dp_size are ready.

    B samples being available is not enough: the sampler reads B as a per-rank
    figure and looks for B * dp_size before it will select anything
    (seqlen_balanced_sampler.py:109). relax/components/advantages.py:88-89
    treats an empty result as `continue` with no sleep, so the deployment busy
    spins on a partition that already holds everything it needs.
    """
    sampler = SeqlenBalancedSampler(n_samples_per_prompt=8, dp_size=dp_size)
    sampled, _ = _fetch(sampler, list(range(pool)), 16)
    assert len(sampled) == expected


@requires_tq
def test_repeated_fetches_replay_indices_that_are_no_longer_ready():
    """Every fetch for one step collides on a single cache key.

    The deployment loops `while not drain_check(...)`, always passing the same
    partition_id and task_name and never a batch_index, so `cache_key =
    (partition_id, task_name, 0)` is constant across the loop. On a cache hit
    the sampler returns `partitions[dp_rank]` without looking at
    `ready_indexes` at all (seqlen_balanced_sampler.py:96-102) -- so it hands
    back the same indices even after those samples have been consumed and
    dropped out of the ready set.
    """
    sampler = SeqlenBalancedSampler(n_samples_per_prompt=8, dp_size=2)
    ready = list(range(64))

    first, _ = _fetch(sampler, ready, 16)
    assert first, "precondition: the first fetch must return something"

    # Simulate the loop making progress: those samples are gone now.
    remaining = [i for i in ready if i not in set(first)]
    second, _ = _fetch(sampler, remaining, 16)

    assert second == first, "the cache replayed a batch none of whose samples are still ready"
    assert not set(second).issubset(remaining), "precondition: the replayed indices really are gone"


@requires_tq
def test_the_other_ranks_shares_are_selected_and_then_unreachable():
    """dp_rank defaults to 0, so partitions[1:] are computed and then stranded.

    Nothing marks them consumed and nothing can ask for them: a caller without
    a sampling_config always reads slot 0.
    """
    sampler = SeqlenBalancedSampler(n_samples_per_prompt=8, dp_size=2)
    ready = list(range(64))

    default_rank, _ = _fetch(sampler, ready, 16)
    rank_1, _ = _fetch(sampler, ready, 16, dp_rank=1)

    assert rank_1, "rank 1 holds samples"
    assert set(default_rank).isdisjoint(rank_1), "and they are not the ones the deployment receives"
