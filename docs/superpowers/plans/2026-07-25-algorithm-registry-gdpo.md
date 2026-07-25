# 算法注册表与 GDPO 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立统一的算法注册与分发机制，把散落在 6 个文件、13 处的算法名 if/elif 收编进注册表，并通过该机制接入 GDPO。

**Architecture:** 新建零重依赖模块 `relax/algorithms/`，用 frozen dataclass `AlgorithmSpec` 描述每个算法的能力，字段存字符串标识符而非 callable（跨进程只传算法名，各进程本地查表）。GDPO 采用 split-stage：Step 1+2（逐 reward 组内标准化 + 加权求和）在 rollout 侧的 reward 后处理层完成，产出仍是每样本一个标量，TransferQueue 零改动；Step 3（sequence 级 batch 白化）在 advantage 层完成。

**Tech Stack:** Python 3.10+，PyTorch，pytest。新模块只依赖 stdlib + torch。

## Global Constraints

- 所有 `relax/` 下新建 `.py` 文件首行必须是：`# Copyright (c) 2026 Relax Authors. All Rights Reserved.`
- `relax/algorithms/` 下**禁止顶层 import** `megatron`、`ray`、`transfer_queue`、`tensordict`、`relax.components`、`relax.backends`。重依赖一律函数内 import。
- 日志只用 `from relax.utils.logging_utils import get_logger`；禁止 `print()` 与 `logging.getLogger()`。
- 禁止通配符导入。显式类型注解。
- Ruff 格式化，行宽 119，双引号，isort 以 `relax` 为 first-party。
- 热路径禁止 GPU-CPU 同步（`.item()` / `.tolist()` / 打印张量）。
- 组合优于继承，最多 2 层。
- 每个 Task 结束前跑 `pre-commit run --all-files`。
- 只创建本地 commit，**不 push**。
- 本机环境只有 `torch` + `numpy`，没有 megatron / ray / transfer_queue / sglang。测试必须在这个前提下可跑。

## 数值常量（全局统一，不得各处硬编码）

- 组内标准化 eps：`1e-6`（与现有 `relax/utils/utils.py:204` 一致）
- Step 3 batch 白化 eps：`1e-6`
- `torch.std` 一律使用默认的无偏（Bessel，n-1）修正

---

### Task 1: AlgorithmSpec 与注册表骨架

**Files:**
- Create: `relax/algorithms/__init__.py`
- Create: `relax/algorithms/spec.py`
- Test: `tests/algorithms/__init__.py`, `tests/algorithms/test_algorithm_registry.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `relax.algorithms.spec.AlgorithmSpec`（frozen dataclass，字段见下）
  - `relax.algorithms.spec.ALGORITHM_SPECS: dict[str, AlgorithmSpec]`
  - `relax.algorithms.get_algorithm(name: str) -> AlgorithmSpec`（未注册名抛 `KeyError`，消息列出可用名）
  - `relax.algorithms.list_algorithm_names() -> list[str]`（保持定义顺序）

- [ ] **Step 1: 写失败的测试**

创建 `tests/algorithms/__init__.py`（空文件）与 `tests/algorithms/test_algorithm_registry.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the algorithm registry."""

import pytest

from relax.algorithms import get_algorithm, list_algorithm_names
from relax.algorithms.spec import ALGORITHM_SPECS, AlgorithmSpec


EXPECTED_NAMES = [
    "grpo",
    "gspo",
    "sapo",
    "cispo",
    "ppo",
    "reinforce_plus_plus",
    "reinforce_plus_plus_baseline",
]


def test_all_expected_algorithms_registered():
    for name in EXPECTED_NAMES:
        assert name in ALGORITHM_SPECS, f"{name} missing from ALGORITHM_SPECS"


def test_spec_name_matches_dict_key():
    for key, spec in ALGORITHM_SPECS.items():
        assert spec.name == key


def test_spec_is_frozen():
    spec = get_algorithm("grpo")
    with pytest.raises(Exception):
        spec.name = "mutated"


def test_get_algorithm_unknown_name_raises_with_available_names():
    with pytest.raises(KeyError) as exc:
        get_algorithm("does_not_exist")
    assert "grpo" in str(exc.value)


def test_list_algorithm_names_matches_registry_keys():
    assert list_algorithm_names() == list(ALGORITHM_SPECS.keys())


def test_grpo_family_shares_one_advantage_fn():
    """grpo/gspo/sapo/cispo 在 advantage 层完全等价，必须共享同一个标识符。"""
    ids = {get_algorithm(n).advantage_fn for n in ("grpo", "gspo", "sapo", "cispo")}
    assert ids == {"grpo_broadcast"}


def test_reward_normalizer_ids_match_current_behavior():
    for name in ("grpo", "gspo", "sapo", "cispo"):
        assert get_algorithm(name).reward_normalizer == "group_mean_std"
    assert get_algorithm("reinforce_plus_plus_baseline").reward_normalizer == "group_mean"
    for name in ("ppo", "reinforce_plus_plus"):
        assert get_algorithm(name).reward_normalizer == "none"


def test_gspo_is_the_only_sequence_level_kl():
    seq = {n for n in list_algorithm_names() if get_algorithm(n).kl_level == "sequence"}
    assert seq == {"gspo"}


def test_gspo_is_the_only_one_needing_full_log_probs():
    need = {n for n in list_algorithm_names() if get_algorithm(n).needs_full_log_probs}
    assert need == {"gspo"}


def test_only_ppo_needs_critic_and_is_disabled():
    critic = {n for n in list_algorithm_names() if get_algorithm(n).needs_critic}
    assert critic == {"ppo"}
    assert get_algorithm("ppo").disabled_reason is not None


def test_reinforce_family_requires_normalize_advantages():
    for name in ("reinforce_plus_plus", "reinforce_plus_plus_baseline"):
        assert get_algorithm(name).requires_normalize_advantages is True
    assert get_algorithm("grpo").requires_normalize_advantages is False


def test_policy_loss_ids_match_current_behavior():
    assert get_algorithm("sapo").policy_loss_fn == "sapo"
    assert get_algorithm("cispo").policy_loss_fn == "cispo"
    for name in ("grpo", "gspo", "ppo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"):
        assert get_algorithm(name).policy_loss_fn == "ppo_clip"


def test_defaults_are_permissive():
    spec = AlgorithmSpec(name="x", reward_normalizer="none", advantage_fn="a", policy_loss_fn="ppo_clip")
    assert spec.kl_level == "token"
    assert spec.min_group_size == 1
    assert spec.allows_custom_reward_post_process is True
    assert spec.requires_rewards_normalization is False
    assert spec.forbids_normalize_advantages is False


def test_spec_module_has_no_heavy_imports():
    """注册表必须能在只有 torch 的 CPU 环境里 import。"""
    import sys

    for banned in ("megatron", "ray", "transfer_queue", "tensordict"):
        assert banned not in sys.modules or True  # 允许别处已导入
    # 真正的断言：spec 模块的源码里不出现顶层重依赖
    import inspect

    import relax.algorithms.spec as spec_mod

    src = inspect.getsource(spec_mod)
    for banned in ("import megatron", "import ray", "import transfer_queue", "from megatron", "from ray"):
        assert banned not in src, f"spec.py must not import {banned}"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/algorithms/test_algorithm_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'relax.algorithms'`

- [ ] **Step 3: 实现 `relax/algorithms/spec.py`**

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Declarative descriptions of the RL algorithms Relax supports.

A single ``--advantage-estimator`` value used to be interpreted by six
different files (role lookup, reward normalisation, advantage formula, policy
loss formula, argument validation).  ``AlgorithmSpec`` centralises all of that
metadata so adding an algorithm means adding one dict entry.

Fields hold **string identifiers**, not callables.  The advantage formula runs
inside the Ray Serve ``Advantages`` deployment while the policy loss runs inside
the Megatron worker; those two processes import different module subsets, so we
ship the algorithm name across the wire and let each process resolve the
identifier against its own table.  This also keeps this module free of heavy
imports, which is what makes it testable on a CPU-only CI runner.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmSpec:
    """Everything the training pipeline needs to know about one algorithm."""

    name: str

    # --- reward stage (rollout side, CPU, scalar in / scalar out) ---
    reward_normalizer: str
    """Key into ``relax.algorithms.rewards.REWARD_NORMALIZERS``."""

    # --- advantage stage ---
    advantage_fn: str
    """Key into ``relax.algorithms.advantages.ADVANTAGE_FNS``."""

    # --- policy loss stage ---
    policy_loss_fn: str
    """Key into ``relax.algorithms.policy.POLICY_LOSS_FNS``."""

    kl_level: str = "token"
    """``"token"`` or ``"sequence"``. GSPO constrains the whole sequence."""

    needs_full_log_probs: bool = False
    """Whether the loss needs CP-gathered full-response log probs."""

    # --- orchestration / validation ---
    needs_critic: bool = False
    requires_normalize_advantages: bool = False
    forbids_normalize_advantages: bool = False
    requires_rewards_normalization: bool = False
    min_group_size: int = 1
    allows_custom_reward_post_process: bool = True
    """False when the algorithm's reward normaliser must not be bypassed by
    ``--custom-reward-post-process-path`` (which short-circuits the whole
    function)."""

    disabled_reason: str | None = None
    """Set for algorithms kept for backwards compatibility but not runnable."""

    @property
    def is_group_normalized(self) -> bool:
        """Whether rewards are normalised per prompt group on the rollout side."""
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
```

创建 `relax/algorithms/__init__.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Algorithm registry: names, capabilities and implementations in one place."""

from relax.algorithms.spec import ALGORITHM_SPECS, AlgorithmSpec, get_algorithm, list_algorithm_names


__all__ = ["ALGORITHM_SPECS", "AlgorithmSpec", "get_algorithm", "list_algorithm_names"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/algorithms/test_algorithm_registry.py -v`
Expected: PASS（14 passed）

- [ ] **Step 5: 跑格式化并提交**

```bash
pre-commit run --all-files
git add relax/algorithms tests/algorithms
git commit -m "feat(algorithms): add AlgorithmSpec registry"
```

---

### Task 2: reward normalizer 纯函数 + 对旧实现逐位等价

**Files:**
- Create: `relax/algorithms/rewards.py`
- Test: `tests/algorithms/test_reward_normalizers.py`

**Interfaces:**
- Consumes: `relax.algorithms.spec.get_algorithm`
- Produces:
  - `relax.algorithms.rewards.REWARD_NORMALIZERS: dict[str, Callable]`
  - 每个 normalizer 签名统一为 `fn(args, samples, raw_rewards: list[float]) -> list[float]`
  - `relax.algorithms.rewards.group_positions(samples, expected_size) -> dict[int, list[int]]`

被冻结的旧实现（来自 `relax/utils/utils.py:181-207`，**逐字保留运算顺序**）：

```python
rewards = torch.tensor(raw_rewards, dtype=torch.float)
positions_by_group: dict[int, list[int]] = {}
for position, sample in enumerate(samples):
    if sample.group_index is None:
        raise ValueError("Sample.group_index is required for group reward normalization.")
    if sample.group_index not in positions_by_group:
        positions_by_group[sample.group_index] = []
    positions_by_group[sample.group_index].append(position)

normalized_rewards = torch.empty_like(rewards)
for group_index, positions in positions_by_group.items():
    if len(positions) != args.n_samples_per_prompt:
        raise ValueError(
            f"Reward group {group_index} has {len(positions)} samples, expected {args.n_samples_per_prompt}."
        )
    group_rewards = rewards[positions]
    group_rewards = group_rewards - group_rewards.mean()
    if <est in ["grpo","gspo","sapo","cispo"]> and args.grpo_std_normalization:
        group_rewards = group_rewards / (group_rewards.std() + 1e-6)
    normalized_rewards[positions] = group_rewards

return raw_rewards, normalized_rewards.tolist()
```

- [ ] **Step 1: 写失败的测试**

创建 `tests/algorithms/test_reward_normalizers.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bit-exact characterization tests for the reward normalizers.

The reference implementation below is a frozen copy of
``relax.utils.utils.post_process_rewards`` as of main@039ce87.  Any refactor
that changes a single bit of output must fail these tests.
"""

import random
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms import get_algorithm  # noqa: E402
from relax.algorithms.rewards import REWARD_NORMALIZERS  # noqa: E402


_GROUP_NORM_ESTIMATORS = ["grpo", "gspo", "sapo", "cispo", "reinforce_plus_plus_baseline"]
_STD_NORM_ESTIMATORS = ["grpo", "gspo", "sapo", "cispo"]
ALL_ESTIMATORS = [
    "grpo",
    "gspo",
    "sapo",
    "cispo",
    "ppo",
    "reinforce_plus_plus",
    "reinforce_plus_plus_baseline",
]


def _legacy_post_process_rewards(args, samples, raw_rewards):
    """Frozen copy of the pre-refactor logic. Do not 'clean up'."""
    if args.advantage_estimator in _GROUP_NORM_ESTIMATORS and args.rewards_normalization:
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        positions_by_group: dict[int, list[int]] = {}
        for position, sample in enumerate(samples):
            if sample.group_index is None:
                raise ValueError("Sample.group_index is required for group reward normalization.")
            if sample.group_index not in positions_by_group:
                positions_by_group[sample.group_index] = []
            positions_by_group[sample.group_index].append(position)

        normalized_rewards = torch.empty_like(rewards)
        for group_index, positions in positions_by_group.items():
            if len(positions) != args.n_samples_per_prompt:
                raise ValueError(
                    f"Reward group {group_index} has {len(positions)} samples, "
                    f"expected {args.n_samples_per_prompt}."
                )
            group_rewards = rewards[positions]
            group_rewards = group_rewards - group_rewards.mean()
            if args.advantage_estimator in _STD_NORM_ESTIMATORS and args.grpo_std_normalization:
                group_rewards = group_rewards / (group_rewards.std() + 1e-6)
            normalized_rewards[positions] = group_rewards

        return normalized_rewards.tolist()

    return raw_rewards


def _new_normalize(args, samples, raw_rewards):
    if not args.rewards_normalization:
        return raw_rewards
    spec = get_algorithm(args.advantage_estimator)
    return REWARD_NORMALIZERS[spec.reward_normalizer](args, samples, raw_rewards)


def _args(estimator, *, n=4, rewards_normalization=True, grpo_std_normalization=True):
    return SimpleNamespace(
        advantage_estimator=estimator,
        n_samples_per_prompt=n,
        rewards_normalization=rewards_normalization,
        grpo_std_normalization=grpo_std_normalization,
    )


def _samples(group_indices):
    return [SimpleNamespace(group_index=g) for g in group_indices]


def _assert_bitwise_equal(left, right):
    lt = torch.tensor(left, dtype=torch.float32)
    rt = torch.tensor(right, dtype=torch.float32)
    assert lt.shape == rt.shape
    assert torch.equal(lt.view(torch.int32), rt.view(torch.int32)), f"{left} != {right}"


def _reward_fixtures():
    rng = random.Random(20260725)
    cases = {
        "normal": [rng.uniform(-3, 3) for _ in range(12)],
        "binary": [float(rng.randint(0, 1)) for _ in range(12)],
        "all_equal": [0.7] * 12,
        "one_group_collapsed": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.25, -1.0, 2.0, -0.5, 3.0],
        "negatives": [-rng.uniform(0, 5) for _ in range(12)],
        "duplicates": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
        "large": [1e6, 1e6 + 1, 1e6 - 1, 1e6, 2e6, 2e6, 2e6, 2e6, 0.0, 1.0, 2.0, 3.0],
    }
    return cases


CONTIGUOUS_GROUPS = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
INTERLEAVED_GROUPS = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
@pytest.mark.parametrize("rewards_normalization", [True, False])
@pytest.mark.parametrize("grpo_std_normalization", [True, False])
@pytest.mark.parametrize("fixture_name", sorted(_reward_fixtures()))
@pytest.mark.parametrize("groups", [CONTIGUOUS_GROUPS, INTERLEAVED_GROUPS])
def test_normalizer_is_bitwise_identical_to_legacy(
    estimator, rewards_normalization, grpo_std_normalization, fixture_name, groups
):
    raw = _reward_fixtures()[fixture_name]
    args = _args(
        estimator,
        rewards_normalization=rewards_normalization,
        grpo_std_normalization=grpo_std_normalization,
    )
    samples = _samples(groups)

    expected = _legacy_post_process_rewards(args, samples, raw)
    actual = _new_normalize(args, samples, raw)

    _assert_bitwise_equal(expected, actual)


def test_identity_normalizer_returns_the_same_list_object():
    """Non-normalising estimators must not copy — legacy returned raw_rewards itself."""
    raw = [1.0, 2.0, 3.0, 4.0]
    args = _args("reinforce_plus_plus")
    samples = _samples([0, 0, 0, 0])
    assert _new_normalize(args, samples, raw) is raw


def test_missing_group_index_raises():
    args = _args("grpo")
    samples = [SimpleNamespace(group_index=None)] * 4
    with pytest.raises(ValueError, match="group_index is required"):
        _new_normalize(args, samples, [1.0, 2.0, 3.0, 4.0])


def test_wrong_group_size_raises():
    args = _args("grpo", n=4)
    samples = _samples([0, 0, 1, 1])
    with pytest.raises(ValueError, match="expected 4"):
        _new_normalize(args, samples, [1.0, 2.0, 3.0, 4.0])


def test_group_mean_normalizer_never_divides_by_std():
    """reinforce_plus_plus_baseline only centres, even with std normalisation on."""
    args = _args("reinforce_plus_plus_baseline", n=4, grpo_std_normalization=True)
    samples = _samples([0, 0, 0, 0])
    raw = [0.0, 1.0, 2.0, 3.0]
    out = _new_normalize(args, samples, raw)
    _assert_bitwise_equal(out, [-1.5, -0.5, 0.5, 1.5])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/algorithms/test_reward_normalizers.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'relax.algorithms.rewards'`

- [ ] **Step 3: 实现 `relax/algorithms/rewards.py`**

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward normalisation strategies, one per algorithm family.

These run on the rollout side (CPU) inside
``relax.utils.utils.post_process_rewards``.  Every normaliser takes the raw
per-sample scalar rewards and returns the per-sample scalars that get written
into the TransferQueue ``rewards`` column, so adding a strategy never changes
the data schema.
"""

from typing import Any, Callable

import torch


GROUP_EPS = 1e-6


def group_positions(samples: list[Any], expected_size: int) -> dict[int, list[int]]:
    """Map ``Sample.group_index`` to the positions it occupies in ``samples``.

    Grouping is driven by ``group_index`` rather than by position, so the
    caller's ordering does not affect the result.
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
            raise ValueError(
                f"Reward group {group_index} has {len(positions)} samples, expected {expected_size}."
            )
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
    """No normalisation — the estimator consumes raw rewards (REINFORCE++, PPO)."""
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/algorithms/test_reward_normalizers.py -q`
Expected: PASS（约 400 个参数化用例全绿）

若出现 `all_equal` fixture 下的不等，说明改动了运算顺序，回退重做——不要放宽断言。

- [ ] **Step 5: 提交**

```bash
pre-commit run --all-files
git add relax/algorithms/rewards.py tests/algorithms/test_reward_normalizers.py
git commit -m "feat(algorithms): extract reward normalizers as pure functions"
```

---

### Task 3: `post_process_rewards` 与 `get_debug_data` 改为查表分发

**Files:**
- Modify: `relax/utils/utils.py:169-209`（`post_process_rewards`）
- Modify: `relax/utils/utils.py:428-431`（`get_debug_data` 的第三份白名单）
- Test: `tests/algorithms/test_post_process_rewards_dispatch.py`

**Interfaces:**
- Consumes: `relax.algorithms.get_algorithm`、`relax.algorithms.rewards.REWARD_NORMALIZERS`
- Produces: `post_process_rewards` 保持原签名 `(args, samples) -> tuple[list[float], list[float]]`

- [ ] **Step 1: 写失败的测试**

创建 `tests/algorithms/test_post_process_rewards_dispatch.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""post_process_rewards must dispatch through the registry, not an if/elif chain."""

import inspect
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import relax.utils.utils as utils_mod  # noqa: E402


def _args(estimator="grpo", **overrides):
    base = dict(
        advantage_estimator=estimator,
        n_samples_per_prompt=4,
        rewards_normalization=True,
        grpo_std_normalization=True,
        custom_reward_post_process_path=None,
        reward_key=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _Sample:
    def __init__(self, group_index, reward):
        self.group_index = group_index
        self.reward = reward

    def get_reward_value(self, args):
        return self.reward if not args.reward_key else self.reward[args.reward_key]


def test_source_has_no_algorithm_name_literals():
    """The whitelist string lists must be gone from post_process_rewards."""
    src = inspect.getsource(utils_mod.post_process_rewards)
    for banned in ('"grpo"', '"gspo"', '"sapo"', '"cispo"', '"reinforce_plus_plus_baseline"'):
        assert banned not in src, f"post_process_rewards still hardcodes {banned}"


def test_returns_raw_and_normalized():
    args = _args("grpo")
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, normalized = utils_mod.post_process_rewards(args, samples)
    assert raw == [0.0, 1.0, 2.0, 3.0]
    assert normalized != raw
    assert abs(sum(normalized)) < 1e-5


def test_identity_path_returns_raw_twice():
    args = _args("reinforce_plus_plus")
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, normalized = utils_mod.post_process_rewards(args, samples)
    assert normalized is raw


def test_rewards_normalization_off_returns_raw_twice():
    args = _args("grpo", rewards_normalization=False)
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, normalized = utils_mod.post_process_rewards(args, samples)
    assert normalized is raw


def test_custom_path_still_short_circuits(monkeypatch):
    sentinel = (["raw"], ["norm"])
    monkeypatch.setattr(utils_mod, "load_function", lambda path: (lambda a, s: sentinel))
    args = _args("grpo", custom_reward_post_process_path="pkg.mod.fn")
    assert utils_mod.post_process_rewards(args, []) is sentinel


def test_reward_key_selects_from_dict():
    args = _args("grpo", reward_key="score")
    samples = [_Sample(0, {"score": r, "other": 99.0}) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, _ = utils_mod.post_process_rewards(args, samples)
    assert raw == [0.0, 1.0, 2.0, 3.0]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/algorithms/test_post_process_rewards_dispatch.py -v`
Expected: FAIL — `test_source_has_no_algorithm_name_literals` 失败（当前源码里有 `"grpo"` 等字面量）

- [ ] **Step 3: 改写 `relax/utils/utils.py`**

把 `relax/utils/utils.py:169-209` 整个函数体替换为：

```python
def post_process_rewards(args: Any, samples: list[Sample] | list[list[Sample]]):
    """Post-process rewards and return (raw_rewards, possibly-normalized
    rewards).

    The normalisation strategy comes from the algorithm registry
    (``AlgorithmSpec.reward_normalizer``); this function no longer knows any
    algorithm names.

    Returns:
        Tuple[List[float], List[float]]
    """
    if args.custom_reward_post_process_path is not None:
        custom_reward_post_process_func = load_function(args.custom_reward_post_process_path)
        return custom_reward_post_process_func(args, samples)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if not args.rewards_normalization:
        return raw_rewards, raw_rewards

    spec = get_algorithm(args.advantage_estimator)
    normalizer = REWARD_NORMALIZERS[spec.reward_normalizer]
    return raw_rewards, normalizer(args, samples, raw_rewards)
```

在 `relax/utils/utils.py` 顶部 import 区加入（isort 会归到 first-party 组）：

```python
from relax.algorithms import get_algorithm
from relax.algorithms.rewards import REWARD_NORMALIZERS
```

把 `relax/utils/utils.py:428-431` 的条件：

```python
        if (
            args.custom_reward_post_process_path is None
            and args.advantage_estimator in ["grpo", "gspo", "sapo", "cispo", "reinforce_plus_plus_baseline"]
            and args.rewards_normalization
        ):
```

替换为：

```python
        if (
            args.custom_reward_post_process_path is None
            and get_algorithm(args.advantage_estimator).is_group_normalized
            and args.rewards_normalization
        ):
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/algorithms/ -q`
Expected: PASS（Task 1–3 全部测试）

- [ ] **Step 5: 提交**

```bash
pre-commit run --all-files
git add relax/utils/utils.py tests/algorithms/test_post_process_rewards_dispatch.py
git commit -m "refactor(rewards): dispatch reward normalization through the registry"
```

---

### Task 4: advantage 纯函数模块

**Files:**
- Create: `relax/algorithms/advantages.py`
- Test: `tests/algorithms/test_advantage_estimators.py`

**Interfaces:**
- Consumes: `relax.utils.training.ppo_utils`（顶层只依赖 torch，已实测可导入）
- Produces:
  - `relax.algorithms.advantages.ADVANTAGE_FNS: dict[str, Callable]`
  - 统一签名：`fn(args, *, rewards, kl, loss_masks, response_lengths, total_lengths, values) -> tuple[list[Tensor], list[Tensor]]`，返回 `(advantages, returns)`
  - `relax.algorithms.advantages.compute_advantages_and_returns(args, *, rewards, kl, loss_masks, response_lengths, total_lengths, values) -> tuple[list[Tensor], list[Tensor]]`（按 spec 查表分发）
  - `relax.algorithms.advantages.whiten_scalar(x: Tensor) -> Tensor`

- [ ] **Step 1: 写失败的测试**

创建 `tests/algorithms/test_advantage_estimators.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Golden-value tests for the extracted advantage estimators."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.advantages import ADVANTAGE_FNS, compute_advantages_and_returns, whiten_scalar  # noqa: E402


def _args(estimator, **overrides):
    base = dict(
        advantage_estimator=estimator,
        kl_coef=0.0,
        gamma=1.0,
        lambd=1.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _inputs(lengths=(3, 2)):
    kl = [torch.zeros(n, dtype=torch.float32) for n in lengths]
    loss_masks = [torch.ones(n, dtype=torch.float32) for n in lengths]
    return dict(
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=list(lengths),
        total_lengths=[n + 2 for n in lengths],
        values=None,
    )


def test_grpo_broadcast_repeats_the_scalar_over_tokens():
    args = _args("grpo")
    adv, ret = compute_advantages_and_returns(args, rewards=[1.5, -2.0], **_inputs())
    assert torch.equal(adv[0], torch.full((3,), 1.5))
    assert torch.equal(adv[1], torch.full((2,), -2.0))
    assert torch.equal(ret[0], adv[0])


def test_grpo_advantages_is_a_distinct_list_from_returns():
    """Legacy did `advantages = list(returns)`; rebinding one must not touch the other."""
    args = _args("grpo")
    adv, ret = compute_advantages_and_returns(args, rewards=[1.0, 1.0], **_inputs())
    assert adv is not ret
    adv[0] = torch.zeros(3)
    assert not torch.equal(ret[0], adv[0])


def test_reinforce_plus_plus_baseline_aliases_returns_to_advantages():
    """Legacy did `returns = advantages` (same list object). Preserve it."""
    args = _args("reinforce_plus_plus_baseline")
    adv, ret = compute_advantages_and_returns(args, rewards=[1.0, 1.0], **_inputs())
    assert ret is adv


def test_reinforce_plus_plus_baseline_subtracts_kl():
    args = _args("reinforce_plus_plus_baseline", kl_coef=0.5)
    inputs = _inputs(lengths=(2,))
    inputs["kl"] = [torch.tensor([2.0, 4.0])]
    inputs["loss_masks"] = [torch.ones(2)]
    adv, _ = compute_advantages_and_returns(args, rewards=[3.0], **inputs)
    # ones * 3.0 - 0.5 * [2, 4] = [2.0, 1.0]
    assert torch.equal(adv[0], torch.tensor([2.0, 1.0]))


def test_unknown_advantage_fn_raises():
    args = _args("grpo")
    with pytest.raises(KeyError):
        ADVANTAGE_FNS["not_a_real_fn"]


def test_whiten_scalar_produces_zero_mean_unit_std():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = whiten_scalar(x)
    assert abs(out.mean().item()) < 1e-6
    assert abs(out.std().item() - 1.0) < 1e-3


def test_whiten_scalar_zero_variance_returns_zeros_not_noise():
    """A collapsed batch must give exactly 0, not the fp32 residual / eps."""
    x = torch.full((7,), 0.7)
    out = whiten_scalar(x)
    assert torch.equal(out, torch.zeros(7))


def test_whiten_scalar_single_element_returns_zero():
    """std of one element is NaN under Bessel correction."""
    out = whiten_scalar(torch.tensor([3.0]))
    assert torch.equal(out, torch.zeros(1))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/algorithms/test_advantage_estimators.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'relax.algorithms.advantages'`

- [ ] **Step 3: 实现 `relax/algorithms/advantages.py`**

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Advantage estimators as pure functions shared by both execution paths.

The colocate path calls these from ``relax.backends.megatron.loss`` inside the
Megatron worker; the fully-async path calls them from the
``relax.components.advantages`` Ray Serve deployment.  Keeping the maths here
means the two call sites only differ in what surrounds them (pipeline-stage
early return, in-place write-back vs nested-tensor packing, advantage
whitening).
"""

from typing import Any, Callable

import torch

from relax.algorithms.spec import get_algorithm
from relax.utils.training.ppo_utils import (
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)


WHITEN_EPS = 1e-6
_ZERO_VARIANCE_THRESHOLD = 1e-12


def whiten_scalar(values: torch.Tensor) -> torch.Tensor:
    """Sequence-level whitening of one scalar per sample.

    Used for GDPO's batch-wise normalisation (Eq. 6).  A degenerate batch
    returns exact zeros: dividing the fp32 centring residual by ``eps`` would
    otherwise manufacture a spurious signal (a batch of 0.7 repeated 7 times
    yields +-5.6e-2 that way).
    """
    if values.numel() < 2:
        return torch.zeros_like(values)
    std = values.std()
    if not torch.isfinite(std) or std <= _ZERO_VARIANCE_THRESHOLD:
        return torch.zeros_like(values)
    return (values - values.mean()) / (std + WHITEN_EPS)


def _as_reward_tensor(rewards: Any, kl: list[torch.Tensor]) -> torch.Tensor:
    if isinstance(rewards, torch.Tensor):
        return rewards.to(dtype=torch.float32, device=kl[0].device)
    return torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)


def advantage_grpo_broadcast(args: Any, *, rewards, kl, **_unused):
    """Broadcast the (already group-normalised) scalar reward over tokens."""
    reward_tensor = _as_reward_tensor(rewards, kl)
    returns = get_grpo_returns(reward_tensor, kl)
    advantages = list(returns)  # make a copy
    return advantages, returns


def advantage_gdpo(args: Any, *, rewards, kl, **_unused):
    """GDPO step 3: whiten the combined per-sample advantage, then broadcast.

    Steps 1 and 2 (per-reward group standardisation and the weighted sum) ran
    on the rollout side, so ``rewards`` already holds one ``A_sum`` per sample.
    """
    reward_tensor = _as_reward_tensor(rewards, kl)
    returns = get_grpo_returns(whiten_scalar(reward_tensor), kl)
    advantages = list(returns)
    return advantages, returns


def advantage_reinforce_plus_plus(args: Any, *, rewards, kl, loss_masks, response_lengths, total_lengths, **_unused):
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
    reward_tensor = _as_reward_tensor(rewards, kl)
    advantages = get_reinforce_plus_plus_baseline_advantages(
        rewards=reward_tensor,
        kl=kl,
        loss_masks=loss_masks,
        kl_coef=args.kl_coef,
    )
    # NOTE(dev): legacy aliased returns to the same list object; downstream OPD
    # rebinds list slots and relies on both views observing the change.
    return advantages, advantages


def advantage_gae(args: Any, *, rewards, kl, values, response_lengths, total_lengths, **_unused):
    """Generalised advantage estimation (PPO). Currently unreachable: the spec
    carries ``disabled_reason`` and argument validation rejects it."""
    from megatron.core import mpu

    old_rewards = rewards
    shaped_rewards = []
    cp_rank = mpu.get_context_parallel_rank()
    for reward, k in zip(old_rewards, kl, strict=False):
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
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Dispatch to the estimator registered for ``args.advantage_estimator``."""
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
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/algorithms/test_advantage_estimators.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: 提交**

```bash
pre-commit run --all-files
git add relax/algorithms/advantages.py tests/algorithms/test_advantage_estimators.py
git commit -m "feat(algorithms): extract advantage estimators as shared pure functions"
```

---

### Task 5: 两个 caller 改用共享函数

**Files:**
- Modify: `relax/components/advantages.py:165-208`
- Modify: `relax/backends/megatron/loss.py:569-613`
- Test: `tests/algorithms/test_advantage_callers_agree.py`

**Interfaces:**
- Consumes: `relax.algorithms.advantages.compute_advantages_and_returns`
- Produces: 两个 caller 的行为不变，仅删除各自的 if/elif 链

- [ ] **Step 1: 写失败的测试**

创建 `tests/algorithms/test_advantage_callers_agree.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Both execution paths must delegate the estimator maths to the registry."""

import inspect
import pathlib

import pytest


torch = pytest.importorskip("torch")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SERVE_PATH = REPO_ROOT / "relax" / "components" / "advantages.py"
LOSS_PATH = REPO_ROOT / "relax" / "backends" / "megatron" / "loss.py"

BANNED_LITERALS = ('"grpo"', '"gspo"', '"sapo"', '"cispo"', '"reinforce_plus_plus"')


def _estimator_block(path, func_name):
    src = path.read_text(encoding="utf-8")
    marker = f"def {func_name}("
    start = src.index(marker)
    return src[start : start + 6000]


def test_serve_path_has_no_estimator_if_elif():
    block = _estimator_block(SERVE_PATH, "compute_advantages_and_returns")
    for literal in BANNED_LITERALS:
        assert literal not in block, f"components/advantages.py still branches on {literal}"


def test_loss_path_has_no_estimator_if_elif():
    block = _estimator_block(LOSS_PATH, "compute_advantages_and_returns")
    for literal in BANNED_LITERALS:
        assert literal not in block, f"megatron/loss.py still branches on {literal}"


def test_both_paths_import_the_shared_helper():
    for path in (SERVE_PATH, LOSS_PATH):
        src = path.read_text(encoding="utf-8")
        assert "from relax.algorithms.advantages import" in src, f"{path} does not use the shared helper"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/algorithms/test_advantage_callers_agree.py -v`
Expected: FAIL — 三个测试全挂（当前两处仍是 if/elif）

- [ ] **Step 3: 改写两个 caller**

在 `relax/components/advantages.py` 中，把 `:165-208` 的整段 if/elif（从 `if self.config.advantage_estimator in [...]` 到 `raise NotImplementedError(...)`）替换为：

```python
        advantages, returns = compute_advantages_and_returns(
            self.config,
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            values=values,
        )
```

并在该文件 import 区加入：

```python
from relax.algorithms.advantages import compute_advantages_and_returns
```

在 `relax/backends/megatron/loss.py` 中，把 `:569-613` 的整段 if/elif 替换为：

```python
    advantages, returns = compute_advantages_and_returns_impl(
        args,
        rewards=rewards,
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        values=values,
    )
```

并在 import 区加入（用别名避免与本文件同名函数冲突）：

```python
from relax.algorithms.advantages import compute_advantages_and_returns as compute_advantages_and_returns_impl
```

`loss.py` 中 `:615-689` 的 OPD 后处理、`normalize_advantages` 白化、in-place 写回**保持不动**。
`components/advantages.py` 中 `:210-222` 的 OPD 后处理与 nested_tensor 打包**保持不动**。

同时删除两个文件里因此不再使用的 import：`get_grpo_returns`、`get_reinforce_plus_plus_returns`、`get_reinforce_plus_plus_baseline_advantages`、`get_advantages_and_returns_batch`（ruff 会报 F401，按报告删）。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/algorithms/ -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
pre-commit run --all-files
git add relax/components/advantages.py relax/backends/megatron/loss.py tests/algorithms/test_advantage_callers_agree.py
git commit -m "refactor(advantages): remove the duplicated estimator if/elif chains"
```

---

### Task 6: policy loss 适配器

**Files:**
- Create: `relax/algorithms/policy.py`
- Modify: `relax/backends/megatron/loss.py:825`、`:884`、`:899-914`
- Test: `tests/algorithms/test_policy_loss_dispatch.py`

**Interfaces:**
- Consumes: `relax.utils.training.ppo_utils` 的 `compute_policy_loss` / `compute_sapo_loss` / `compute_cispo_loss`
- Produces:
  - `relax.algorithms.policy.POLICY_LOSS_FNS: dict[str, Callable]`
  - 统一签名 `fn(args, *, log_probs, ppo_kl, advantages) -> tuple[Tensor, Tensor]`
  - `relax.algorithms.policy.compute_policy_loss_for(args, *, log_probs, ppo_kl, advantages) -> tuple[Tensor, Tensor]`

- [ ] **Step 1: 写失败的测试**

创建 `tests/algorithms/test_policy_loss_dispatch.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Policy loss selection must come from the registry."""

import pathlib
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.policy import POLICY_LOSS_FNS, compute_policy_loss_for  # noqa: E402
from relax.utils.training.ppo_utils import compute_cispo_loss, compute_policy_loss, compute_sapo_loss  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOSS_PATH = REPO_ROOT / "relax" / "backends" / "megatron" / "loss.py"


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
    ppo_kl = torch.randn(8)
    advantages = torch.randn(8)
    log_probs = torch.randn(8)
    return log_probs, ppo_kl, advantages


def test_registry_has_all_three_losses():
    assert set(POLICY_LOSS_FNS) == {"ppo_clip", "sapo", "cispo"}


def test_ppo_clip_matches_the_underlying_kernel():
    log_probs, ppo_kl, advantages = _tensors()
    args = _args("grpo")
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


def test_sapo_matches_the_underlying_kernel():
    log_probs, ppo_kl, advantages = _tensors()
    args = _args("sapo")
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=1.0, tau_neg=1.05)
    assert torch.equal(got[0], want[0])


def test_cispo_matches_the_underlying_kernel():
    log_probs, ppo_kl, advantages = _tensors()
    args = _args("cispo")
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_cispo_loss(
        log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages, eps_clip=0.2, eps_clip_high=0.2
    )
    assert torch.equal(got[0], want[0])


def test_sapo_defaults_when_args_lack_tau_fields():
    log_probs, ppo_kl, advantages = _tensors()
    args = SimpleNamespace(advantage_estimator="sapo", eps_clip=0.2, eps_clip_high=0.2)
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=1.0, tau_neg=1.05)
    assert torch.equal(got[0], want[0])


def test_loss_py_no_longer_branches_on_estimator_names():
    src = LOSS_PATH.read_text(encoding="utf-8")
    for banned in (
        'args.advantage_estimator == "gspo"',
        'args.advantage_estimator == "sapo"',
        'args.advantage_estimator == "cispo"',
    ):
        assert banned not in src, f"loss.py still contains: {banned}"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/algorithms/test_policy_loss_dispatch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'relax.algorithms.policy'`

- [ ] **Step 3: 实现 `relax/algorithms/policy.py` 并改 `loss.py`**

创建 `relax/algorithms/policy.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Policy loss variants behind one uniform signature.

The underlying kernels in ``relax.utils.training.ppo_utils`` take different
argument lists; these adapters normalise them to
``fn(args, *, log_probs, ppo_kl, advantages)`` so the caller can look one up by
name instead of branching on the algorithm.
"""

from typing import Any, Callable

import torch

from relax.algorithms.spec import get_algorithm
from relax.utils.training.ppo_utils import compute_cispo_loss, compute_policy_loss, compute_sapo_loss


def policy_loss_ppo_clip(args: Any, *, log_probs, ppo_kl, advantages):
    """Standard clipped surrogate objective (GRPO, GSPO, REINFORCE++)."""
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
    """Clipped importance ratio that keeps the gradient direction."""
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
    """Dispatch to the policy loss registered for ``args.advantage_estimator``."""
    spec = get_algorithm(args.advantage_estimator)
    return POLICY_LOSS_FNS[spec.policy_loss_fn](args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
```

在 `relax/backends/megatron/loss.py`：

`:825` 改为：

```python
    need_full_log_probs = args.use_opsm or get_algorithm(args.advantage_estimator).needs_full_log_probs
```

`:884` 改为：

```python
    if get_algorithm(args.advantage_estimator).kl_level == "sequence":
```

`:899-914` 的三分支替换为：

```python
    pg_loss, pg_clipfrac = compute_policy_loss_for(
        args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages
    )
```

import 区加入：

```python
from relax.algorithms import get_algorithm
from relax.algorithms.policy import compute_policy_loss_for
```

删除因此不再使用的 `compute_cispo_loss`、`compute_policy_loss`、`compute_sapo_loss` import（ruff F401 会指出）。`compute_gspo_kl` 仍在用，保留。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/algorithms/ -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
pre-commit run --all-files
git add relax/algorithms/policy.py relax/backends/megatron/loss.py tests/algorithms/test_policy_loss_dispatch.py
git commit -m "refactor(loss): dispatch policy loss and KL level through the registry"
```

---

### Task 7: `ALGOS` 角色表由 spec 派生（修掉 reinforce_plus_plus 崩溃）

**Files:**
- Modify: `relax/core/registry.py:59-92`
- Test: `tests/algorithms/test_algos_roles.py`

**Interfaces:**
- Consumes: `relax.algorithms.list_algorithm_names`、`relax.algorithms.get_algorithm`
- Produces: `ALGOS` 的 key 集合 = 所有已注册 RL 算法名 ∪ `{"sft"}`

设计要点：`ALGOS` 是**角色拓扑表**，与 `AlgorithmSpec` 是两个命名空间。`"sft"` 由 `loss_type` 触发、与 `advantage_estimator` 正交，不进 spec。

- [ ] **Step 1: 写失败的测试**

创建 `tests/algorithms/test_algos_roles.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""ALGOS role topology must cover every registered algorithm."""

import pytest


pytest.importorskip("megatron.core")

from relax.algorithms import list_algorithm_names  # noqa: E402
from relax.core.registry import ALGOS, ROLES  # noqa: E402


def test_every_registered_algorithm_has_roles():
    """reinforce_plus_plus used to be missing here and crashed the controller."""
    for name in list_algorithm_names():
        assert name in ALGOS, f"{name} has no role mapping; controller would raise ValueError"


def test_sft_stays_a_separate_key():
    assert "sft" in ALGOS
    assert "sft" not in list_algorithm_names()


def test_rl_algorithms_share_the_standard_role_set():
    expected = {ROLES.rollout, ROLES.actor, ROLES.advantages, ROLES.reference, ROLES.actor_fwd}
    for name in list_algorithm_names():
        assert set(ALGOS[name]) == expected, f"{name} has unexpected roles"


def test_sft_roles_unchanged():
    assert set(ALGOS["sft"]) == {ROLES.sft, ROLES.actor}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/algorithms/test_algos_roles.py -v`
Expected: 若本机无 megatron 则整体 SKIP；在有 megatron 的环境下 `test_every_registered_algorithm_has_roles` FAIL（`reinforce_plus_plus` / `reinforce_plus_plus_baseline` / `ppo` 缺失）

本机无 megatron，此 Task 的验证靠 Step 4 的替代测试 + CI。

- [ ] **Step 3: 改写 `relax/core/registry.py`**

把 `:59-92` 的 `ALGOS` 字面量替换为：

```python
def _standard_rl_roles() -> dict:
    """Role topology shared by every RL algorithm.

    Role composition is orthogonal to the advantage formula, so all registered
    RL algorithms get the same set; SFT is keyed separately because it is
    selected by ``loss_type``, not by ``advantage_estimator``.
    """
    return {
        ROLES.rollout: Rollout,
        ROLES.actor: Actor,
        ROLES.advantages: Advantages,
        ROLES.reference: ActorFwd,
        ROLES.actor_fwd: ActorFwd,
    }


ALGOS = {name: _standard_rl_roles() for name in list_algorithm_names()}
ALGOS["sft"] = {
    ROLES.sft: SFT,
    ROLES.actor: Actor,
}
```

import 区加入：

```python
from relax.algorithms import list_algorithm_names
```

- [ ] **Step 4: 加一个不需要 megatron 的替代验证并运行**

在 `tests/algorithms/test_algos_roles.py` 末尾追加：

```python
def test_registry_source_derives_algos_from_spec_names():
    """Source-level check that runs even without megatron installed."""
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "relax" / "core" / "registry.py"
    src = path.read_text(encoding="utf-8")
    assert "list_algorithm_names()" in src
    assert '"grpo": {' not in src, "ALGOS still hardcodes per-algorithm role dicts"
```

把这个函数移到文件顶部 `pytest.importorskip` **之前**（否则无 megatron 时会被整体跳过）。调整后文件结构：模块顶部先定义 `test_registry_source_derives_algos_from_spec_names`，再 `pytest.importorskip("megatron.core")`，再定义其余测试。

Run: `pytest tests/algorithms/test_algos_roles.py -v`
Expected: 1 passed, 4 skipped（本机无 megatron）

- [ ] **Step 5: 提交**

```bash
pre-commit run --all-files
git add relax/core/registry.py tests/algorithms/test_algos_roles.py
git commit -m "fix(registry): derive ALGOS role map from the algorithm registry"
```

---

### Task 8: GDPO 的 reward 归一化（Step 1 + Step 2）

**Files:**
- Modify: `relax/algorithms/rewards.py`
- Modify: `relax/algorithms/spec.py`（新增 `gdpo` 条目）
- Modify: `relax/utils/types.py`（新增 `get_reward_components`）
- Test: `tests/algorithms/test_gdpo.py`

**Interfaces:**
- Consumes: `relax.algorithms.rewards.group_positions`
- Produces:
  - `relax.algorithms.rewards.normalize_gdpo_decoupled(args, samples, raw_rewards) -> list[float]`
  - `relax.algorithms.rewards.extract_reward_components(samples, keys) -> torch.Tensor`（`[B, K]`，float32）
  - `relax.algorithms.rewards.resolve_gdpo_weights(args, keys) -> list[float]`
  - `relax.utils.types.Sample.get_reward_components(keys) -> list[float]`
  - `ALGORITHM_SPECS["gdpo"]`

- [ ] **Step 1: 写失败的测试**

创建 `tests/algorithms/test_gdpo.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GDPO: per-reward group standardisation, weighted sum, batch whitening."""

import math
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms import get_algorithm  # noqa: E402
from relax.algorithms.advantages import whiten_scalar  # noqa: E402
from relax.algorithms.rewards import REWARD_NORMALIZERS, extract_reward_components  # noqa: E402


def _args(keys=("correctness", "format"), weights=None, n=4):
    return SimpleNamespace(
        advantage_estimator="gdpo",
        n_samples_per_prompt=n,
        rewards_normalization=True,
        grpo_std_normalization=True,
        gdpo_reward_keys=list(keys),
        gdpo_reward_weights=weights,
    )


class _S:
    def __init__(self, group_index, reward):
        self.group_index = group_index
        self.reward = reward

    def get_reward_components(self, keys):
        values = []
        for key in keys:
            if not isinstance(self.reward, dict) or key not in self.reward:
                raise ValueError(f"missing reward key {key!r}")
            values.append(self.reward[key])
        return values


def _normalize(args, samples):
    spec = get_algorithm(args.advantage_estimator)
    return REWARD_NORMALIZERS[spec.reward_normalizer](args, samples, [0.0] * len(samples))


def _mk(groups, correctness, fmt):
    return [
        _S(g, {"correctness": c, "format": f})
        for g, c, f in zip(groups, correctness, fmt, strict=True)
    ]


def _manual_gdpo(correctness, fmt, groups, weights=(1.0, 1.0)):
    """Independent NumPy-free reference implementing Eq. 4-6 directly."""
    per_key = []
    for column in (correctness, fmt):
        out = [0.0] * len(column)
        for g in sorted(set(groups)):
            idx = [i for i, gg in enumerate(groups) if gg == g]
            vals = [column[i] for i in idx]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
            for i in idx:
                out[i] = 0.0 if std <= 1e-12 else (column[i] - mean) / (std + 1e-6)
        per_key.append(out)
    combined = [weights[0] * a + weights[1] * b for a, b in zip(per_key[0], per_key[1], strict=True)]
    return combined


# ---------------- registration ----------------


def test_gdpo_is_registered():
    spec = get_algorithm("gdpo")
    assert spec.reward_normalizer == "gdpo_decoupled"
    assert spec.advantage_fn == "gdpo"
    assert spec.policy_loss_fn == "ppo_clip"


def test_gdpo_spec_guards():
    spec = get_algorithm("gdpo")
    assert spec.min_group_size == 2
    assert spec.allows_custom_reward_post_process is False
    assert spec.forbids_normalize_advantages is True
    assert spec.requires_rewards_normalization is True


# ---------------- step 1 + 2 ----------------


def test_matches_hand_computed_three_steps():
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    correctness = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    fmt = [1.0, 1.0, 0.0, 0.0, 0.5, 0.25, 0.75, 1.0]
    samples = _mk(groups, correctness, fmt)

    got = _normalize(_args(), samples)
    want = _manual_gdpo(correctness, fmt, groups)

    assert torch.allclose(torch.tensor(got), torch.tensor(want), atol=1e-6)


def test_weights_apply_to_normalized_advantages_not_raw_rewards():
    groups = [0, 0, 0, 0]
    correctness = [1.0, 0.0, 1.0, 0.0]
    fmt = [0.0, 0.0, 10.0, 10.0]
    samples = _mk(groups, correctness, fmt)

    got = _normalize(_args(weights=[2.0, 0.5]), samples)
    want = _manual_gdpo(correctness, fmt, groups, weights=(2.0, 0.5))

    assert torch.allclose(torch.tensor(got), torch.tensor(want), atol=1e-6)


def test_default_weights_are_all_ones():
    groups = [0, 0, 0, 0]
    samples = _mk(groups, [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    assert _normalize(_args(weights=None), samples) == _normalize(_args(weights=[1.0, 1.0]), samples)


# ---------------- reward collapse ----------------


def test_collapsed_key_contributes_zero_but_others_keep_signal():
    """This is GDPO's core benefit over GRPO."""
    groups = [0, 0, 0, 0]
    correctness = [1.0, 1.0, 1.0, 1.0]  # collapsed
    fmt = [1.0, 0.0, 1.0, 0.0]
    samples = _mk(groups, correctness, fmt)

    got = _normalize(_args(), samples)
    fmt_only = _manual_gdpo([0.0] * 4, fmt, groups)

    assert torch.allclose(torch.tensor(got), torch.tensor(fmt_only), atol=1e-6)
    assert max(abs(v) for v in got) > 0.5


def test_fully_collapsed_group_yields_exact_zeros_without_nan():
    groups = [0, 0, 0, 0]
    samples = _mk(groups, [0.7] * 4, [0.7] * 4)
    got = _normalize(_args(), samples)
    assert got == [0.0, 0.0, 0.0, 0.0]


def test_collapsed_group_does_not_leak_fp32_residual():
    """0.7 repeated 7x used to produce +-5.6e-2 via `/(std + 1e-6)`."""
    groups = [0] * 7
    samples = _mk(groups, [0.7] * 7, [0.7] * 7)
    got = _normalize(_args(n=7), samples)
    assert got == [0.0] * 7


# ---------------- error contracts ----------------


def test_missing_reward_key_raises():
    samples = [_S(0, {"correctness": 1.0}) for _ in range(4)]
    with pytest.raises(ValueError, match="format"):
        _normalize(_args(), samples)


def test_non_numeric_reward_raises():
    samples = [_S(0, {"correctness": 1.0, "format": "good"}) for _ in range(4)]
    with pytest.raises(TypeError):
        _normalize(_args(), samples)


def test_bool_reward_raises():
    samples = [_S(0, {"correctness": True, "format": 1.0}) for _ in range(4)]
    with pytest.raises(TypeError):
        _normalize(_args(), samples)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_or_inf_reward_raises(bad):
    samples = [_S(0, {"correctness": bad, "format": 1.0}) for _ in range(4)]
    with pytest.raises(ValueError):
        _normalize(_args(), samples)


def test_fewer_than_two_keys_raises():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="at least two"):
        _normalize(_args(keys=("correctness",)), samples)


def test_weight_count_mismatch_raises():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="weights"):
        _normalize(_args(weights=[1.0]), samples)


# ---------------- numerical properties ----------------


def test_group_of_two_always_normalizes_to_plus_minus_one_over_sqrt_two():
    groups = [0, 0]
    samples = _mk(groups, [0.0, 100.0], [0.0, 1.0])
    got = _normalize(_args(n=2), samples)
    expected = 2 * (1.0 / math.sqrt(2.0))
    assert math.isclose(got[1], expected, rel_tol=1e-4)
    assert math.isclose(got[0], -expected, rel_tol=1e-4)


def test_extract_reward_components_shape_and_dtype():
    samples = _mk([0, 0, 0, 0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    out = extract_reward_components(samples, ["correctness", "format"])
    assert out.shape == (4, 2)
    assert out.dtype == torch.float32


def test_single_reward_gdpo_is_a_constant_positive_multiple_of_grpo():
    """Documented deviation: GDPO does NOT reduce to GRPO for one reward."""
    torch.manual_seed(0)
    b, g = 16, 4
    rewards = (torch.rand(b * g) < 0.5).float().view(b, g)
    grpo = ((rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-6)).flatten()
    gdpo = whiten_scalar(grpo)

    mask = grpo.abs() > 1e-6
    ratio = gdpo[mask] / grpo[mask]
    assert ratio.min() > 0
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-4)
    assert not torch.allclose(gdpo, grpo, atol=1e-3)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/algorithms/test_gdpo.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_reward_components'`

- [ ] **Step 3: 实现**

在 `relax/utils/types.py` 的 `Sample` 类中，`get_reward_value`（`:169-170`）**之后**新增（不改动现有方法）：

```python
    def get_reward_components(self, keys: list[str]) -> list[Any]:
        """Return the named reward components, preserving ``keys`` order.

        Multi-reward algorithms (GDPO) need the individual components before
        ``get_reward_value`` collapses ``reward`` to a single scalar.
        """
        if not isinstance(self.reward, dict):
            raise ValueError(
                f"Sample.reward must be a dict to read components {keys}, got {type(self.reward).__name__}. "
                "Make the reward function return a dict of named rewards."
            )
        values = []
        for key in keys:
            if key not in self.reward:
                raise ValueError(
                    f"Reward key {key!r} missing from sample reward (available: {sorted(self.reward)})."
                )
            values.append(self.reward[key])
        return values
```

在 `relax/algorithms/rewards.py` 末尾追加：

```python
def resolve_gdpo_weights(args: Any, keys: list[str]) -> list[float]:
    """Per-reward weights, defaulting to 1.0 each.

    Note the weights multiply the **normalised** advantages (Eq. 7), not the
    raw rewards — that is what makes them scale-free across reward components.
    """
    weights = getattr(args, "gdpo_reward_weights", None)
    if weights is None:
        return [1.0] * len(keys)
    if len(weights) != len(keys):
        raise ValueError(
            f"--gdpo-reward-weights has {len(weights)} entries but --gdpo-reward-keys has {len(keys)}."
        )
    return [float(w) for w in weights]


def extract_reward_components(samples: list[Any], keys: list[str]) -> torch.Tensor:
    """Build the ``[B, K]`` component matrix, rejecting malformed rewards.

    Contract violations raise instead of defaulting to 0.0: a silently zeroed
    component is indistinguishable from a genuinely collapsed reward, which
    would hide a broken reward function behind plausible-looking training.
    """
    rows: list[list[float]] = []
    for position, sample in enumerate(samples):
        values = sample.get_reward_components(keys)
        row: list[float] = []
        for key, value in zip(keys, values, strict=True):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"Reward {key!r} of sample {position} must be a real number, "
                    f"got {value!r} ({type(value).__name__})."
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Reward {key!r} of sample {position} is {value}, which is not finite.")
            row.append(value)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def normalize_gdpo_decoupled(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """GDPO steps 1 and 2 (arXiv 2601.05242, Eq. 4 and Eq. 7).

    Step 1 standardises each reward component within its prompt group; step 2
    combines them with the configured weights.  The result is one scalar per
    sample, so it travels through the existing ``rewards`` column and needs no
    TransferQueue schema change.  Step 3 (batch whitening) happens later, in
    :func:`relax.algorithms.advantages.advantage_gdpo`.
    """
    keys = list(getattr(args, "gdpo_reward_keys", None) or [])
    if len(keys) < 2:
        raise ValueError(f"--gdpo-reward-keys needs at least two reward keys, got {keys}.")
    weights = resolve_gdpo_weights(args, keys)

    components = extract_reward_components(samples, keys)
    positions_by_group = group_positions(samples, args.n_samples_per_prompt)

    normalized = torch.zeros_like(components)
    for positions in positions_by_group.values():
        group = components[positions]
        centered = group - group.mean(dim=0, keepdim=True)
        std = group.std(dim=0)
        # A collapsed component contributes exactly 0. Relying on
        # `centered / (0 + eps)` would amplify the fp32 centring residual into
        # a spurious signal (0.7 repeated 7x yields +-5.6e-2 that way).
        scaled = torch.where(
            std > _ZERO_VARIANCE_THRESHOLD,
            centered / (std + GROUP_EPS),
            torch.zeros_like(centered),
        )
        normalized[positions] = scaled

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    return (normalized * weight_tensor).sum(dim=1).tolist()
```

在 `relax/algorithms/rewards.py` 顶部加 `import math`，并加常量：

```python
_ZERO_VARIANCE_THRESHOLD = 1e-12
```

把 `REWARD_NORMALIZERS` 补上一项：

```python
    "gdpo_decoupled": normalize_gdpo_decoupled,
```

在 `relax/algorithms/spec.py` 的 `ALGORITHM_SPECS` 中，`"cispo"` 之后加入：

```python
    "gdpo": AlgorithmSpec(
        name="gdpo",
        reward_normalizer="gdpo_decoupled",
        advantage_fn="gdpo",
        policy_loss_fn="ppo_clip",
        forbids_normalize_advantages=True,
        requires_rewards_normalization=True,
        min_group_size=2,
        allows_custom_reward_post_process=False,
    ),
```

同时更新 `tests/algorithms/test_algorithm_registry.py` 的 `EXPECTED_NAMES` 加入 `"gdpo"`，并把 `test_reward_normalizer_ids_match_current_behavior` 里对 gdpo 的期望补上。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/algorithms/ -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
pre-commit run --all-files
git add relax/algorithms tests/algorithms relax/utils/types.py
git commit -m "feat(gdpo): per-reward group standardization and weighted combination"
```

---

### Task 9: 参数与校验全部改为读 spec

**Files:**
- Modify: `relax/utils/arguments.py:1456-1473`（choices）
- Modify: `relax/utils/arguments.py`（新增 GDPO 参数，接在 `--sapo-tau-neg` 之后）
- Modify: `relax/utils/arguments.py:2593-2598`、`:2739`、`:3000-3005`
- Test: `tests/algorithms/test_arguments_spec_driven.py`

**Interfaces:**
- Consumes: `relax.algorithms.get_algorithm`、`relax.algorithms.list_algorithm_names`
- Produces: `--gdpo-reward-keys`（`nargs="+"`, dest `gdpo_reward_keys`, default `None`）、`--gdpo-reward-weights`（`nargs="+"`, `type=float`, dest `gdpo_reward_weights`, default `None`）

- [ ] **Step 1: 写失败的测试**

创建 `tests/algorithms/test_arguments_spec_driven.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Argument parsing and validation must read the registry, not string lists."""

import pathlib

import pytest


ARGS_PATH = pathlib.Path(__file__).resolve().parents[2] / "relax" / "utils" / "arguments.py"


def _source():
    return ARGS_PATH.read_text(encoding="utf-8")


def test_choices_come_from_the_registry():
    src = _source()
    assert "choices=list_algorithm_names()" in src


def test_no_hardcoded_estimator_choice_list():
    src = _source()
    assert '"reinforce_plus_plus_baseline",\n                    "ppo",' not in src


def test_use_critic_reads_the_spec():
    src = _source()
    assert 'args.advantage_estimator == "ppo"' not in src
    assert "needs_critic" in src


def test_normalize_advantages_rules_read_the_spec():
    src = _source()
    assert "requires_normalize_advantages" in src
    assert "forbids_normalize_advantages" in src


def test_gdpo_arguments_declared():
    src = _source()
    assert "--gdpo-reward-keys" in src
    assert "--gdpo-reward-weights" in src


def test_disabled_reason_drives_the_ppo_rejection():
    src = _source()
    assert "disabled_reason" in src


def test_guard_rails_present():
    src = _source()
    assert "allows_custom_reward_post_process" in src
    assert "min_group_size" in src
    assert "requires_rewards_normalization" in src
```

补一个行为测试 `tests/algorithms/test_gdpo_validation.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Spec-driven validation rules, exercised without importing arguments.py."""

from types import SimpleNamespace

import pytest

from relax.algorithms import get_algorithm


def validate_algorithm_args(args):
    """Mirror of the rules added to slime_validate_args (see Task 9 Step 3)."""
    from relax.utils.arguments import validate_algorithm_args as impl

    return impl(args)


def _args(estimator="gdpo", **overrides):
    base = dict(
        advantage_estimator=estimator,
        normalize_advantages=False,
        rewards_normalization=True,
        custom_reward_post_process_path=None,
        n_samples_per_prompt=4,
        gdpo_reward_keys=["correctness", "format"],
        gdpo_reward_weights=None,
        use_critic=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_gdpo_rejects_custom_reward_post_process():
    with pytest.raises(ValueError, match="custom-reward-post-process-path"):
        validate_algorithm_args(_args(custom_reward_post_process_path="pkg.mod.fn"))


def test_gdpo_rejects_normalize_advantages():
    with pytest.raises(ValueError, match="normalize-advantages"):
        validate_algorithm_args(_args(normalize_advantages=True))


def test_gdpo_rejects_group_size_below_two():
    with pytest.raises(ValueError, match="n-samples-per-prompt"):
        validate_algorithm_args(_args(n_samples_per_prompt=1))


def test_gdpo_rejects_disabled_rewards_normalization():
    with pytest.raises(ValueError, match="rewards-normalization"):
        validate_algorithm_args(_args(rewards_normalization=False))


def test_gdpo_requires_at_least_two_keys():
    with pytest.raises(ValueError, match="at least two"):
        validate_algorithm_args(_args(gdpo_reward_keys=["correctness"]))


def test_gdpo_happy_path():
    args = _args()
    validate_algorithm_args(args)
    assert args.use_critic is False


def test_reinforce_plus_plus_requires_normalize_advantages():
    with pytest.raises(ValueError, match="normalize-advantages"):
        validate_algorithm_args(_args("reinforce_plus_plus", normalize_advantages=False, gdpo_reward_keys=None))


def test_ppo_is_rejected_with_its_disabled_reason():
    with pytest.raises(ValueError, match="no longer supported"):
        validate_algorithm_args(_args("ppo", gdpo_reward_keys=None))


def test_grpo_sets_use_critic_false():
    args = _args("grpo", gdpo_reward_keys=None)
    validate_algorithm_args(args)
    assert args.use_critic is False
    assert get_algorithm("grpo").needs_critic is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/algorithms/test_arguments_spec_driven.py tests/algorithms/test_gdpo_validation.py -v`
Expected: FAIL — 源码断言全挂；`validate_algorithm_args` 不存在

- [ ] **Step 3: 改写 `relax/utils/arguments.py`**

顶部 import 区加入：

```python
from relax.algorithms import get_algorithm, list_algorithm_names
```

`:1456-1473` 的 `--advantage-estimator` 改为：

```python
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=list_algorithm_names(),
                default="grpo",
                help=(
                    "Advantage estimator to use. The choices come from the algorithm registry "
                    "(relax/algorithms/spec.py). Note: on-policy distillation (OPD) is orthogonal "
                    "to the advantage estimator. Use --opd-kl-coef > 0 to enable OPD on top of any estimator."
                ),
            )
```

在 `--sapo-tau-neg`（`:1480-1485`）之后插入：

```python
            parser.add_argument(
                "--gdpo-reward-keys",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "Names of the reward components GDPO normalizes independently, e.g. "
                    "`--gdpo-reward-keys correctness format`. The reward function must return a dict "
                    "containing every key. At least two keys are required."
                ),
            )
            parser.add_argument(
                "--gdpo-reward-weights",
                type=float,
                nargs="+",
                default=None,
                help=(
                    "Per-component weights for GDPO, matching --gdpo-reward-keys in length. "
                    "Defaults to 1.0 for each. The weights multiply the *normalized* advantages, "
                    "not the raw rewards."
                ),
            )
```

在 `slime_validate_args` 之前新增一个模块级函数：

```python
def validate_algorithm_args(args: Namespace) -> None:
    """Apply the registry-declared constraints for the selected algorithm.

    Everything here used to be an `if args.advantage_estimator == "..."` chain
    scattered across this file; the rules now come from AlgorithmSpec fields so
    adding an algorithm cannot forget one.
    """
    spec = get_algorithm(args.advantage_estimator)

    if spec.disabled_reason:
        raise ValueError(spec.disabled_reason)

    args.use_critic = spec.needs_critic

    if spec.requires_normalize_advantages and not args.normalize_advantages:
        raise ValueError(
            f"The {spec.name!r} advantage estimator requires --normalize-advantages to be set."
        )
    if spec.forbids_normalize_advantages and args.normalize_advantages:
        raise ValueError(
            f"The {spec.name!r} advantage estimator already whitens advantages; "
            "--normalize-advantages would apply a second, token-level whitening. Remove it."
        )
    if spec.requires_rewards_normalization and not args.rewards_normalization:
        raise ValueError(
            f"The {spec.name!r} advantage estimator needs reward normalization; "
            "remove --disable-rewards-normalization."
        )
    if not spec.allows_custom_reward_post_process and args.custom_reward_post_process_path is not None:
        raise ValueError(
            f"--custom-reward-post-process-path short-circuits reward post-processing, which would "
            f"silently disable {spec.name!r}'s reward normalization. Drop one of the two."
        )
    if args.n_samples_per_prompt < spec.min_group_size:
        raise ValueError(
            f"The {spec.name!r} advantage estimator needs --n-samples-per-prompt >= "
            f"{spec.min_group_size}, got {args.n_samples_per_prompt}."
        )
    if spec.reward_normalizer == "gdpo_decoupled":
        keys = args.gdpo_reward_keys or []
        if len(keys) < 2:
            raise ValueError(f"--gdpo-reward-keys needs at least two reward keys, got {keys}.")
        weights = args.gdpo_reward_weights
        if weights is not None and len(weights) != len(keys):
            raise ValueError(
                f"--gdpo-reward-weights has {len(weights)} entries but --gdpo-reward-keys has {len(keys)}."
            )
        # Multi-component rewards arrive as dicts; without --reward-key the raw_reward
        # column would hold dicts and blow up dict_to_tensordict downstream.
        if not getattr(args, "reward_key", None):
            raise ValueError(
                f"The {spec.name!r} advantage estimator needs --reward-key to pick the scalar reward "
                "used for metrics and for the raw_reward column."
            )
```

删除 `:2593-2598` 的 `reinforce_plus_plus` assert 块，删除 `:2739` 的 `args.use_critic = args.advantage_estimator == "ppo"`，删除 `:3000-3005` 的 ppo raise 块。在 `slime_validate_args` 里 `args.use_critic` **首次被读取之前**（即原 `:2739` 的位置）插入：

```python
    if not is_sft:
        validate_algorithm_args(args)
    else:
        args.use_critic = False
```

`:2600-2604` 的 `fully_async` 禁止 `normalize_advantages` 的 assert **保持不动**。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/algorithms/ -q`
Expected: PASS。若 `test_gdpo_validation.py` 因 `relax.utils.arguments` 拉入 sglang 等重依赖而无法 import，按 `tests/utils/test_arguments_opd_teacher_colocate.py:1-40` 的 stub 模式在测试文件顶部补 stub module 再 import。

- [ ] **Step 5: 提交**

```bash
pre-commit run --all-files
git add relax/utils/arguments.py tests/algorithms/test_arguments_spec_driven.py tests/algorithms/test_gdpo_validation.py
git commit -m "feat(args): drive estimator choices and validation from the registry"
```

---

### Task 10: 最小训练示例（correctness + format 双奖励）

**Files:**
- Create: `examples/gdpo/reward_gdpo.py`
- Create: `examples/gdpo/run-qwen3-0.6B-1xgpu-gdpo.sh`
- Create: `examples/gdpo/README.md`
- Test: `tests/algorithms/test_example_reward_gdpo.py`

**Interfaces:**
- Consumes: 无
- Produces: `examples.gdpo.reward_gdpo.reward_func(args, sample, **kwargs) -> dict`，返回 `{"score", "correctness", "format"}`

- [ ] **Step 1: 写失败的测试**

创建 `tests/algorithms/test_example_reward_gdpo.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The shipped GDPO example reward must produce the two configured components."""

import importlib.util
import pathlib
from types import SimpleNamespace

import pytest


REWARD_PATH = pathlib.Path(__file__).resolve().parents[2] / "examples" / "gdpo" / "reward_gdpo.py"


def _load():
    spec = importlib.util.spec_from_file_location("reward_gdpo", REWARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(response, label="42"):
    return SimpleNamespace(response=response, label=label, metadata={})


def test_returns_all_three_keys():
    module = _load()
    out = module.compute_gdpo_reward("<think>x</think><answer>42</answer>", "42")
    assert set(out) == {"score", "correctness", "format"}


def test_correct_and_well_formatted():
    module = _load()
    out = module.compute_gdpo_reward("<think>reasoning</think><answer>42</answer>", "42")
    assert out["correctness"] == 1.0
    assert out["format"] == 1.0


def test_wrong_answer_but_well_formatted():
    module = _load()
    out = module.compute_gdpo_reward("<think>reasoning</think><answer>7</answer>", "42")
    assert out["correctness"] == 0.0
    assert out["format"] == 1.0


def test_correct_answer_but_malformed():
    """The case GDPO is designed for: one component collapses, the other does not."""
    module = _load()
    out = module.compute_gdpo_reward("42", "42")
    assert out["correctness"] == 1.0
    assert out["format"] == 0.0


def test_components_are_plain_floats():
    module = _load()
    out = module.compute_gdpo_reward("<answer>42</answer>", "42")
    for key in ("score", "correctness", "format"):
        assert isinstance(out[key], float)
        assert not isinstance(out[key], bool)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/algorithms/test_example_reward_gdpo.py -v`
Expected: FAIL — 文件不存在

- [ ] **Step 3: 实现示例**

创建 `examples/gdpo/reward_gdpo.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-component reward for the GDPO example: correctness and format.

GDPO normalizes each component within its prompt group before combining them,
so a group where every rollout is correct but only some are well-formatted
still carries a learning signal — under GRPO the summed reward would collapse
and the whole group would be discarded.

Wire it up with:

    --custom-rm-path examples.gdpo.reward_gdpo.reward_func
    --reward-key score
    --advantage-estimator gdpo
    --gdpo-reward-keys correctness format
"""

import re
from typing import Any


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_answer(response: str) -> str | None:
    match = _ANSWER_RE.search(response)
    return match.group(1).strip() if match else None


def compute_gdpo_reward(response: str, label: str) -> dict[str, float]:
    """Score one response on answer correctness and on output format."""
    answer = _extract_answer(response)
    correctness = 1.0 if answer is not None and answer == str(label).strip() else 0.0

    has_think = _THINK_RE.search(response) is not None
    has_answer = answer is not None
    format_score = 0.5 * float(has_think) + 0.5 * float(has_answer)

    return {
        # `score` is what --reward-key selects for metrics and for non-GDPO runs.
        "score": correctness,
        "correctness": correctness,
        "format": format_score,
    }


async def reward_func(args: Any, sample: Any, **kwargs: Any) -> dict[str, float]:
    """Entry point for --custom-rm-path."""
    return compute_gdpo_reward(sample.response, sample.label)
```

创建 `examples/gdpo/run-qwen3-0.6B-1xgpu-gdpo.sh`，照抄 `examples/algorithms/run-qwen35-9B-8xgpu-openr1mm-cispo-async.sh` 的骨架，算法块为：

```bash
GDPO_ARGS=(
   --advantage-estimator gdpo
   --gdpo-reward-keys correctness format
   --gdpo-reward-weights 1.0 1.0
   --custom-rm-path examples.gdpo.reward_gdpo.reward_func
   --reward-key score
   --n-samples-per-prompt 8
   --eps-clip 0.2
   --kl-coef 0.00
   --entropy-coef 0.00
)
```

创建 `examples/gdpo/README.md`：说明双奖励结构、为什么 GDPO 在部分 collapse 时优于 GRPO、以及 §7「已知偏差」三条。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/algorithms/test_example_reward_gdpo.py -v && bash -n examples/gdpo/run-qwen3-0.6B-1xgpu-gdpo.sh`
Expected: 5 passed；bash 语法检查无输出

- [ ] **Step 5: 提交**

```bash
pre-commit run --all-files
git add examples/gdpo tests/algorithms/test_example_reward_gdpo.py
git commit -m "docs(examples): add a two-reward GDPO training example"
```

---

### Task 11: 文档

**Files:**
- Modify: `docs/zh/examples/algorithms.md`
- Modify: `docs/en/examples/algorithms.md`
- Modify: `examples/algorithms/README.md`
- Create: `docs/zh/guide/adding-an-algorithm.md`
- Create: `docs/en/guide/adding-an-algorithm.md`
- Modify: `docs/.vitepress/config.*`（sidebar 注册新页）

**Interfaces:**
- Consumes: 无
- Produces: 无代码接口

- [ ] **Step 1: 在两个 algorithms.md 增加 GDPO 小节**

内容需包含：一句话定位、三步公式、`--gdpo-reward-keys` / `--gdpo-reward-weights` 参数表、reward 函数必须返回 dict 的要求、以及设计文档 §7 的三条已知偏差（Step 3 的 batch 边界、单 reward 不退化为 GRPO、G=2 时幅度信息丢失）。参考文献加 `[GDPO - arXiv 2601.05242](https://arxiv.org/abs/2601.05242)`。

- [ ] **Step 2: 更新 `examples/algorithms/README.md`**

「支持的算法」表格加一行 GDPO；`--advantage-estimator` 的取值说明加 `gdpo`；新增「GDPO 专用参数」表。

- [ ] **Step 3: 写「新算法接入」文档**

按四步组织：(1) 在 `relax/algorithms/spec.py` 的 `ALGORITHM_SPECS` 加一条；(2) 若需要新的 reward 归一化 / advantage / policy loss，在对应模块加纯函数并登记到表里；(3) 写单测；(4) 加示例脚本与文档。明确写出三条硬约束：`relax/algorithms/` 禁止顶层重依赖；字段存字符串标识符而非 callable；`ALGOS` 角色表自动派生、无需手改。

- [ ] **Step 4: 验证文档能构建**

Run: `cd docs && npm install && npm run docs:build`
Expected: 构建成功，无 dead link 警告。若本机无 node 环境则跳过，并在提交信息里注明未验证。

- [ ] **Step 5: 提交**

```bash
pre-commit run --all-files
git add docs examples/algorithms/README.md
git commit -m "docs: document GDPO and how to add an algorithm"
```

---

### Task 12: 全量回归与 Modal GPU 冒烟

**Files:**
- Create: `../relax-modal/gdpo_app.py`（在 relax-modal 仓库，不进 Relax）
- Create: `../relax-modal/gdpo-configs/run-qwen3-0.6B-1xgpu-gdpo.sh`

**Interfaces:**
- Consumes: Task 1–11 的全部产出
- Produces: 一次 Qwen3-0.6B 单卡 H100 的 GDPO 小步训练日志与曲线

- [ ] **Step 1: 跑全量 CPU 测试**

Run: `pytest tests/ -v --tb=short`
Expected: 新增测试全绿；既有测试的 pass/skip 数量与改动前一致。

基线必须在 **Task 1 开始之前**就记录下来（此时已有 commit，`git stash` 无用）：

```bash
# 在动第一行代码前执行一次，把结果存到 scratchpad
pytest tests/ -q --tb=no 2>&1 | tail -3 > /tmp/relax-test-baseline.txt
```

Task 12 时用 `git worktree add /tmp/relax-base main` 起一个基线工作区再跑一遍对比，或直接比对上面存下的数字。任何由 pass 变 fail 的既有测试都必须定位根因，不得放宽断言、不得改测试去迁就实现。

- [ ] **Step 2: 跑 lint 与格式化**

Run: `pre-commit run --all-files`
Expected: 全部 Passed。

- [ ] **Step 3: 准备 Modal 冒烟**

复制 `../relax-modal/app.py` 为 `gdpo_app.py`，改动清单：

- `modal.App("relax-gdpo-smoke")`
- 删除 `add_local_file("model-configs/qwen3-0.6B.sh", ...)`（上游 `039ce87` 已合入 `scripts/models/qwen3-0.6B.sh`）
- `add_local_dir("../Relax", ...)` 保持不变——它会把当前工作区（含本次 GDPO 改动）烤进镜像
- sed 生成脚本时把 `--wandb-group` 改为 `gdpo-qwen3-0.6b-gsm8k`，后置 grep 校验同步改
- 环境变量新增 `"PROJECT_NAME": "relax-gdpo-smoke"`（规避 `metrics/service.py:169` 回落到含 `/` 的 `tb_project_name` 导致的 W&B 初始化失败）
- `WANDB_PROJECT` 改为 `"relax-gdpo-smoke"`
- 默认 `num_rollout=4`

冒烟用的启动脚本必须放在 `/workspace/Relax/contributor-program/2026-cohort-1/` 下——原脚本用 `SCRIPT_DIR/../..` 推导 `RELAX_ROOT` 且不可用 env 覆盖。

**先复核**（单份报告来源，未交叉验证）：`community` 仓库的启动脚本是否仍有 `--use-clearml` 与 `--use-metrics-service`（sed 依赖）；`train_clean.parquet` 是否需要在 `prepare_assets.py` 里预生成并 `assets.commit()`（`app.py` 只 commit outputs）。

- [ ] **Step 4: 跑冒烟并取产物**

```bash
cd ../relax-modal
modal run prepare_assets.py            # relax-assets 已就绪可跳过
modal run gdpo_app.py --num-rollout 4  # 约 8 train steps
modal volume get relax-outputs /log/grpo-nr4.log report/gdpo-train.log
```

Expected: 训练跑完退出码 0；日志里能看到 `advantage_estimator ... gdpo`；`rollout/raw_reward` 与 `train/pg_loss` 有非零变化。

- [ ] **Step 5: 汇报**

写一份简短结果：CPU 测试通过数、pre-commit 结果、Modal 冒烟的步数/耗时/关键指标；明确列出**未验证**的部分（如无 node 环境未构建文档、无 megatron 环境未跑 `tests/algorithms/test_algos_roles.py` 的 megatron 分支）。

不 push，不开 PR。
