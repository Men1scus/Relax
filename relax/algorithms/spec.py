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
    uses_reward_components: bool = False
    """Whether the algorithm consumes several named reward components rather
    than the single scalar ``--reward-key`` selects. Drives the
    ``--gdpo-reward-keys`` / ``--gdpo-reward-weights`` validation."""

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
