# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Declarative descriptions of the RL algorithms Relax supports.

A single ``--advantage-estimator`` value used to be interpreted independently by
six different files: role lookup, reward normalisation, the advantage formula,
the policy loss formula, and two rounds of argument validation.  Adding an
algorithm meant finding all of them.  ``AlgorithmSpec`` collects that metadata
in one place, so a new algorithm is one dict entry plus its pure functions.

Fields hold **string identifiers**, not callables.  The advantage formula runs
inside the Ray Serve ``Advantages`` deployment while the policy loss runs inside
the Megatron worker; those two processes import different module subsets, so we
ship the algorithm name and let each process resolve the identifier against its
own table.  That also keeps this module free of heavy imports, which is what
makes the registry testable on a CPU-only runner.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmSpec:
    """Everything the training pipeline needs to know about one algorithm."""

    name: str

    # --- reward stage (rollout side, CPU, scalar in / scalar out) ---
    reward_normalizer: str
    """Key into :data:`relax.algorithms.rewards.REWARD_NORMALIZERS`."""

    # --- advantage stage ---
    advantage_fn: str
    """Key into :data:`relax.algorithms.advantages.ADVANTAGE_FNS`."""

    # --- policy loss stage ---
    policy_loss_fn: str
    """Key into :data:`relax.algorithms.policy.POLICY_LOSS_FNS`."""

    kl_level: str = "token"
    """``"token"`` or ``"sequence"``; GSPO constrains the sequence as a whole."""

    needs_full_log_probs: bool = False
    """Whether the loss needs CP-gathered full-response log probs."""

    # --- orchestration and validation ---
    needs_critic: bool = False
    requires_normalize_advantages: bool = False
    forbids_normalize_advantages: bool = False
    """Set when the algorithm already whitens advantages itself, so the
    token-level ``--normalize-advantages`` pass would whiten a second time."""

    requires_rewards_normalization: bool = False
    supports_fully_async: bool = True
    """Whether the algorithm is correct under ``--fully-async``.

    That mode routes advantage computation to the single-replica
    ``relax.components.advantages`` deployment, which owns no data-parallel
    group and consumes one ``global_batch_size / num_iters_per_train_update``
    slice at a time. An algorithm whose advantages depend on batch-level
    statistics computes them over that slice instead of the batch, and at
    slice size 1 gets no signal at all — silently, since the run still
    converges and exits cleanly. Note ``--hybrid`` is *not* affected: it uses
    the colocate role set, so advantages are computed in the Megatron worker
    where the data-parallel group exists.

    Do not be tempted to relax this into "allow it when the slice happens to
    equal the batch". The slice the deployment actually receives also depends
    on which TransferQueue sampler the controller installed
    (``relax/core/controller.py``), and that choice is global to every
    consumer. Under ``--balance-data`` it is ``SeqlenBalancedSampler``, whose
    ``batch_size`` is *per data-parallel rank*; the deployment passes no
    ``sampling_config``, so it would receive one rank's share of a
    token-balanced split rather than the batch.
    """

    uses_reward_components: bool = False
    """Whether the algorithm consumes several named reward components rather
    than the single scalar ``--reward-key`` selects. Drives the
    ``--gdpo-reward-keys`` / ``--gdpo-reward-weights`` validation.

    Those two options stay GDPO-prefixed on purpose: every algorithm-specific
    option in Relax names its algorithm (``--sapo-tau-pos``,
    ``--disable-grpo-std-normalization``), and GDPO is so far the only member
    of this category — renaming now would mean guessing the abstraction from a
    single example. When a second ``uses_reward_components`` algorithm lands,
    rename both to algorithm-neutral options and keep the old spellings as
    deprecated aliases for one release, the way ``--loss-type sft_loss`` is
    handled in ``relax/utils/arguments.py``.
    """

    min_group_size: int = 1
    allows_custom_reward_post_process: bool = True
    """False when ``--custom-reward-post-process-path`` would silently disable
    the algorithm: that hook short-circuits reward post-processing entirely."""

    disabled_reason: str | None = None
    """Set for algorithms kept for backwards compatibility but not runnable."""

    @property
    def is_group_normalized(self) -> bool:
        """Whether rewards get normalised per prompt group on the rollout
        side."""
        return self.reward_normalizer != "none"


_PPO_DISABLED = (
    "PPO (Proximal Policy Optimization) is no longer supported in Relax. "
    "Please use one of the following advantage estimators instead: "
    "'grpo', 'gspo', 'sapo', 'cispo', 'reinforce_plus_plus', or 'reinforce_plus_plus_baseline'."
)


# NOTE(dev): explicit dict literal, deliberately not decorator-based registration.
# The advantage formula and the policy loss execute in two different processes
# whose import graphs differ; decorator registration depends on "was this module
# imported?" and silently loses an algorithm when one side misses the import.
ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "grpo": AlgorithmSpec(
        name="grpo",
        reward_normalizer="group_mean_std",
        advantage_fn="grpo_broadcast",
        policy_loss_fn="ppo_clip",
    ),
    "gspo": AlgorithmSpec(
        name="gspo",
        reward_normalizer="group_mean_std",
        advantage_fn="grpo_broadcast",
        policy_loss_fn="ppo_clip",
        kl_level="sequence",
        needs_full_log_probs=True,
    ),
    "sapo": AlgorithmSpec(
        name="sapo",
        reward_normalizer="group_mean_std",
        advantage_fn="grpo_broadcast",
        policy_loss_fn="sapo",
    ),
    "cispo": AlgorithmSpec(
        name="cispo",
        reward_normalizer="group_mean_std",
        advantage_fn="grpo_broadcast",
        policy_loss_fn="cispo",
    ),
    "gdpo": AlgorithmSpec(
        name="gdpo",
        reward_normalizer="gdpo_decoupled",
        advantage_fn="gdpo",
        policy_loss_fn="ppo_clip",
        # Step 3 already whitens per sequence; --normalize-advantages would add a
        # second, token-level pass on top of it.
        forbids_normalize_advantages=True,
        requires_rewards_normalization=True,
        # Step 3 needs the training batch. Under --fully-async it would only ever
        # see one slice of it, and a slice of one sample yields zero advantages.
        supports_fully_async=False,
        uses_reward_components=True,
        # Step 1 divides by an unbiased group std, undefined for a single sample.
        min_group_size=2,
        # The custom hook short-circuits reward post-processing, which would
        # silently skip steps 1 and 2 while the run still reports itself as GDPO.
        allows_custom_reward_post_process=False,
    ),
    "ppo": AlgorithmSpec(
        name="ppo",
        reward_normalizer="none",
        advantage_fn="gae",
        policy_loss_fn="ppo_clip",
        needs_critic=True,
        disabled_reason=_PPO_DISABLED,
    ),
    "reinforce_plus_plus": AlgorithmSpec(
        name="reinforce_plus_plus",
        reward_normalizer="none",
        advantage_fn="reinforce_plus_plus",
        policy_loss_fn="ppo_clip",
        requires_normalize_advantages=True,
    ),
    "reinforce_plus_plus_baseline": AlgorithmSpec(
        name="reinforce_plus_plus_baseline",
        reward_normalizer="group_mean",
        advantage_fn="reinforce_plus_plus_baseline",
        policy_loss_fn="ppo_clip",
        requires_normalize_advantages=True,
    ),
}


def get_algorithm(name: str) -> AlgorithmSpec:
    """Look up an algorithm spec by its ``--advantage-estimator`` value."""
    try:
        return ALGORITHM_SPECS[name]
    except KeyError:
        available = ", ".join(ALGORITHM_SPECS)
        raise KeyError(f"Unknown advantage estimator {name!r}. Available: {available}") from None


def list_algorithm_names() -> list[str]:
    """All registered algorithm names, in definition order."""
    return list(ALGORITHM_SPECS)
