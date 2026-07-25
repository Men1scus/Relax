# 解耦算法配置与训练流程，接入 GDPO — 设计文档

日期：2026-07-25
分支：`feat/algorithm-registry-gdpo`
基线：`main @ 039ce87`

## 1. 问题

一个算法名（`--advantage-estimator` 的值）当前同时承担四种互不相干的职责，被 6 个文件、13 处 if/elif 与白名单列表分别解读：

| 职责 | 承载位置 |
|---|---|
| `ALGOS` 查表 key，决定起哪些 Ray Serve 角色 | `relax/core/registry.py:59-92` |
| reward 组归一化开关 | `relax/utils/utils.py:182`、`:203`、`:430` |
| advantage 公式选择 | `relax/components/advantages.py:165-208` + `relax/backends/megatron/loss.py:569-613`（**两份重复实现**） |
| policy loss 公式选择 | `relax/backends/megatron/loss.py:825`、`:884`、`:899`、`:905` |

其余散落点：`relax/utils/arguments.py:1456`（choices）、`:2594`、`:2739`、`:3000`。

后果：

- 新增算法需同时改 6 个文件，容易遗漏。
- `reinforce_plus_plus` / `reinforce_plus_plus_baseline` 有完整实现、在 choices 里，但不在 `ALGOS` 里，一用就在 `relax/core/controller.py:333` 抛 `ValueError`。这是可触发的运行期崩溃。
- `tests/` 下没有任何 advantage estimator / policy loss 单测。

同时，Relax 不支持 GDPO 所需的多奖励独立归一化。

## 2. GDPO 是什么

GDPO: Group reward-Decoupled Normalization Policy Optimization，arXiv 2601.05242（NVIDIA，ICML 2026），官方实现 github.com/NVlabs/GDPO。

设第 `i` 个 prompt 采 `G` 条 rollout，共 `n` 个 reward 分量：

1. **Step 1 — 逐 reward 组内标准化**（Eq.4）：`A_k = (r_k - mean_G(r_k)) / std_G(r_k)`
2. **Step 2 — 加权求和**（Eq.5/7）：`A_sum = Σ_k w_k · A_k`，默认 `w_k = 1`。权重加在 **normalized advantage** 上，不是加在 raw reward 上。
3. **Step 3 — batch 级白化**（Eq.6）：`Â = (A_sum - mean_batch) / (std_batch + ε)`，减均值**且**除标准差。

相对 GRPO 的收益：GRPO 下总 reward 组内全同则整条样本 advantage 归零、样本被丢弃；GDPO 下只有**该维**贡献 0，其它维仍有信号。

## 3. 关键约束（均经实测或读码核实）

### 3.1 GRPO 的组内归一化不在 advantage 层

`relax/utils/training/ppo_utils.py:324-331` 的 `get_grpo_returns` 只做标量广播，无分组、无 mean/std。真正的组内标准化在 rollout 侧 `relax/utils/utils.py:169-209 post_process_rewards`，按 `Sample.group_index` 分组，组大小必须等于 `args.n_samples_per_prompt`。

### 3.2 多分量 reward 无法穿过 TransferQueue

`Sample.reward` 已可以是 dict（`relax/utils/types.py:25`），但在 `get_reward_value`（`:169-170`）处被 `--reward-key` 塌缩成单标量。

若新增 `[B,K]` 字段贯穿：`relax/utils/utils.py:250-253 _to_tensor_2d` 会把 depth-2 列表做成 jagged NestedTensor，而 `relax/utils/data/stream_dataloader.py:814` 的 `if "lengths" in k or "reward" in k` 是**子串匹配**，会对它调用 `.tolist()`。

实测（torch 2.10）：

```
RuntimeError: .tolist() is not supported for tensor subclasses, got NestedTensor
```

即该路径会在运行期崩溃，且本机无 transfer_queue / megatron，无法完整验证修复。**因此不走这条路。**

### 3.3 `post_process_rewards` 有一条整段短路

`relax/utils/utils.py:176-178`：`--custom-reward-post-process-path` 在函数第一行就 return。多 reward 场景（GenRM、`examples/deepeyes` 式 dict）恰恰是这个 hook 最可能的使用者。若把 GDPO 三步全放在此函数内，用户一开 custom path，GDPO 静默失效且无报错。

### 3.4 transfer batch 是流式变长的

`relax/engine/rollout/sglang_rollout.py:832-836`：

```python
transfer_batch_size = (args.global_batch_size // args.num_iters_per_train_update // args.n_samples_per_prompt
                       if args.fully_async else args.rollout_batch_size)
```

colocate 下等于 `rollout_batch_size`（完整训练批）；fully-async 下是切片，且 `:902` 还有可能只含个位数样本的尾批 flush。在尾批上做白化会给少量样本乘一个噪声标量。

### 3.5 已核实为**不成立**的担忧

- `dynamic_sampling_filter` 在 `sglang_rollout.py:798` 执行，早于 transfer（`:844+`），所以「归一化后丢样本」不成立。
- `RolloutManager` 是单个 Ray actor，多 SGLang engine 不执行 `post_process_rewards`，所以「多 worker 各自白化」当前不成立（属潜在脆弱性，非现实缺陷）。

### 3.6 数值行为实测

```
collapse 组 fp32 残差：0.7 重复 7 次 → (x-mean)/(std+1e-6) = ±5.6e-2，不是 0
G=1：torch.std 无偏 → NaN
G=2：任意两值归一化后恒为 ±0.7071（幅度信息全丢）
单 reward：GDPO/GRPO 逐元素比值为严格正常数（实测 1.2105863），但 ≠ 理论 1.1547（collapse 组拉低 batch std）
```

第一条同样存在于现有 GRPO 路径（`relax/utils/utils.py:204`），属既有隐患。修它会改变现有 GRPO 数值，与「保持现有算法行为不变」冲突，**本次不动**，仅在 GDPO 内显式置零。

### 3.7 环境

`relax/utils/training/ppo_utils.py` 顶层只 import torch / dist / F，所有 megatron import 都在函数内，实测仅靠 torch 即可导入。CI 是 ubuntu 无 GPU；本机开发环境只有 torch + numpy。因此新模块必须零重依赖才可单测。

## 4. 设计

### 4.1 新模块 `relax/algorithms/`

```
relax/algorithms/
├── __init__.py    公开 API：get_algorithm() / list_algorithm_names()
├── spec.py        AlgorithmSpec (frozen dataclass) + ALGORITHM_SPECS 显式 dict
├── rewards.py     reward normalizer 纯函数 + REWARD_NORMALIZERS 表
├── advantages.py  advantage 纯函数 + ADVANTAGE_FNS 表
└── policy.py      policy loss 统一签名适配器 + POLICY_LOSS_FNS 表
```

硬约束：禁止顶层 import megatron / ray / transfer_queue / `relax.components`。

### 4.2 AlgorithmSpec

字段存**字符串标识符**而非 callable。理由：advantage 在 Ray Serve `Advantages` 进程执行、policy loss 在 Megatron worker 进程执行，两者 import 图不同；跨进程只传算法名、各自本地查表，既规避 cloudpickle 闭包陷阱，也让 spec 模块保持零重依赖。

```python
@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    reward_normalizer: str                          # none | group_mean | group_mean_std | gdpo_decoupled
    advantage_fn: str                               # grpo_broadcast | reinforce_plus_plus | rpp_baseline | gdpo
    policy_loss_fn: str                             # ppo_clip | sapo | cispo
    kl_level: str = "token"                         # token | sequence
    needs_full_log_probs: bool = False
    needs_critic: bool = False
    requires_normalize_advantages: bool = False
    forbids_normalize_advantages: bool = False
    min_group_size: int = 1
    allows_custom_reward_post_process: bool = True
    disabled_reason: str | None = None
```

注册用**显式 dict 字面量**，不用装饰器——装饰器依赖「模块被 import 过」，双进程下漏一处就静默缺算法。

各字段收编的原有分支：

| 字段 | 取代 |
|---|---|
| `reward_normalizer` | `utils.py:182`、`:203`、`:430` 三份白名单 |
| `advantage_fn` | `advantages.py:165-208` + `loss.py:569-613` 两份 if/elif |
| `policy_loss_fn` | `loss.py:899`/`:905`/`:914` |
| `kl_level` | `loss.py:884` 的 gspo 特判 |
| `needs_full_log_probs` | `loss.py:825` |
| `needs_critic` | `arguments.py:2739` |
| `requires_normalize_advantages` | `arguments.py:2594` |
| `disabled_reason` | `arguments.py:3000` 的 ppo |

### 4.3 GDPO 三步的归属（split-stage）

| 步骤 | 位置 | 说明 |
|---|---|---|
| Step 1 逐 key 组内标准化 | `rewards.gdpo_decoupled`，rollout 侧 CPU | 无偏 std + eps 1e-6，零方差**显式置 0** |
| Step 2 加权求和 | 同上 | 产出 `[B]` 标量 `A_sum`，写进现有 `rewards` 字段；**TransferQueue 零改动** |
| Step 3 sequence 级白化 | `advantages.gdpo`，advantage 层 | 对 `[B]` 标量白化后再 `get_grpo_returns` 广播 |

Step 3 放 advantage 层而非 reward 层的三条理由：

1. 不受 §3.3 的 `custom_reward_post_process_path` 短路影响。
2. 避开 §3.4 的流式尾批。
3. 与 `--normalize-advantages`（token 级、跨 DP）的互斥关系可在同一层显式断言，不会双重白化。

### 4.4 消除两份 `compute_advantages_and_returns` 重复

抽成 `relax/algorithms/advantages.py` 的纯函数（输入 rewards/kl/masks/lengths，返回 `(advantages, returns)` 两个 `list[Tensor]`）。两个 caller 各自保留自身差异：

| 差异 | `components/advantages.py` | `backends/megatron/loss.py` |
|---|---|---|
| early-return 条件 | `log_probs is None and values is None and rollout_log_probs is None` | `not mpu.is_pipeline_last_stage()` |
| kl 兜底 | 多一路 `rollout_log_probs` 回退 | 无 |
| 结果写回 | 返回 nested_tensor dict | in-place 写回 rollout_data |
| `normalize_advantages` 白化 | 无（`arguments.py:2600-2604` 禁止） | 有（`:628-686`） |

### 4.5 `ALGOS` 角色表

保持 `relax/core/registry.py` 的角色拓扑表独立于 `AlgorithmSpec`：`sft` 由 `loss_type` 触发，与 `advantage_estimator` 正交，混入同一张表会产生大量无意义空字段。

RL 条目改为从 spec 名派生标准 RL 角色映射。副作用：`reinforce_plus_plus` / `reinforce_plus_plus_baseline` / `gdpo` 自动获得角色，§1 的运行期崩溃消失。

### 4.6 参数（已获授权修改 `relax/utils/arguments.py`）

- `--advantage-estimator` 的 `choices` 改为 `list_algorithm_names()`
- 新增 `--gdpo-reward-keys`（`nargs="+"`，estimator 为 gdpo 时必填，至少 2 个）
- 新增 `--gdpo-reward-weights`（`nargs="+"`，`type=float`，默认全 1.0）
- 校验全部改为读 spec，无算法名字符串比较：
  - `args.use_critic = spec.needs_critic`
  - `spec.requires_normalize_advantages` → assert `normalize_advantages`
  - `spec.forbids_normalize_advantages` → assert not `normalize_advantages`（GDPO 防双重白化）
  - `not spec.allows_custom_reward_post_process and args.custom_reward_post_process_path` → raise（GDPO 防静默失效）
  - `args.n_samples_per_prompt < spec.min_group_size` → raise（GDPO 需 G ≥ 2）
  - `spec.disabled_reason` → raise（ppo）

### 4.7 数值裁决

| 项 | 取值 | 理由 |
|---|---|---|
| Step 1 std | 无偏（n-1）+ eps 1e-6 | 对齐 Relax 现有 GRPO（`utils.py:204`），而非论文（无 eps）/ TRL（1e-4）/ verl（1e-6）/ ms-swift（1e-8）。同库一致优先于跨库保真 |
| Step 1 零方差 | 显式置 0 | 不靠 `/(0+eps)`——§3.6 实测该写法会产生 ±5.6e-2 伪信号 |
| Step 3 粒度 | sequence 级 | 论文 Eq.6 / TRL / ms-swift 口径。verl 的 token 级 `masked_whiten` 让长序列主导统计量，不等价 |
| Step 3 eps | 1e-6 | 与 Step 1 一致 |
| 缺 reward key | raise，报 sample index + key 名 | 填 0 会把契约违约伪装成 reward collapse |
| 非数值 / bool / NaN / Inf | raise | 同上 |
| `G < 2` | 校验期 raise | 无偏 std 在 G=1 时为 NaN，无法靠「关掉 std 归一化」退化 |
| 单 reward 的正标量 | 保留，不提供关闭开关 | Step 3 是 GDPO 定义的一部分；关掉得到的是变体而非论文 GDPO |

## 5. 测试

`tests/algorithms/`：

| 文件 | 内容 |
|---|---|
| `test_post_process_rewards_equivalence.py` | 冻结当前实现为 `_legacy_reference`；7 estimator × `rewards_normalization` × `grpo_std_normalization` × {满组 / 全等组 / 交错 group_index / 负值 / 重复值} 笛卡尔积；**逐位比对**（`view(torch.int32).equal`，不用 `allclose`） |
| `test_algorithm_registry.py` | 每个 CLI 名恰有一个 spec；grpo/gspo/sapo/cispo 共享同一 `advantage_fn` 标识符；未注册名 raise；`choices == list(specs)`；spec 字段与现状一致 |
| `test_advantage_estimators.py` | 六个 estimator 的 golden 值；两份 caller 喂同输入断言输出一致；`reinforce_plus_plus_baseline` 的 `returns is advantages` 别名行为回归 |
| `test_gdpo.py` | 手算 2 组 × G=4 × K=2 三步；单维 collapse（该维贡献 0、他维有信号）；全 collapse 输出全 0 且无 NaN；缺 key / 非数值 / bool / NaN / Inf → raise；G=1 校验期拒绝；G=2 恒 ±0.7071；单 reward 时 GDPO/GRPO 比值为**正常数**（不断言等于 GRPO） |

测试模式沿用 `tests/backends/megatron/test_opd_loss_aggregation.py` 的 fake-megatron 注入 + `tests/core/test_registry_sft.py` 的 `SimpleNamespace` args factory。新模块零重依赖，多数测试不需要任何 mock。

必须避免的假阳性：全零 / 全相同 reward（任何公式都出 0）；只测 grpo；`allclose` 容差吞掉无偏/有偏 std 差异；不测 `custom_reward_post_process` 旁路；不测 `get_debug_data` 的第三份白名单。

## 6. 交付物

- 算法注册与分发模块 `relax/algorithms/`
- 现有 7 个 estimator 迁移到注册表
- GDPO 实现（Step 1/2 在 rewards，Step 3 在 advantages）
- 单测（§5）
- `examples/gdpo/`：reward 函数返回 `{"score", "correctness", "format"}` + 启动脚本
- 文档：`docs/{zh,en}/examples/algorithms.md` 增加 GDPO 节；新增「新算法接入」文档
- Modal GPU 冒烟：基于 `../relax-modal/` 的 Qwen3-0.6B 单卡 H100 小步训练

## 7. 已知偏差（写进用户文档，不隐藏）

1. **Step 3 的 D_Batch**。colocate 下 `rollout_data` 是本 DP rank 的 `global_batch_size / dp_size` 分片，因此 `whiten_scalar` 接收 `mpu.get_data_parallel_group()` 并对 `count/sum/sumsq` 做 all-reduce，统计量等于完整训练批。fully-async 下 Advantages 是单副本、无需归约，但每次消费 `global_batch_size / num_iters_per_train_update` 的切片，统计窗口小于训练批——这层偏差保留。
2. **单 reward 时 GDPO ≠ GRPO**，差一个数据相关的正标量（实测 1.2105863）。Axolotl 文档称「单 reward 时两者等价」不严谨。
3. **G=2 时逐 key 归一化后恒为 ±0.7071**，幅度信息全丢，权重成为唯一区分度来源。

## 8. 明确不做

- 不让多分量 reward 贯穿 TransferQueue（§3.2 已实测会崩）
- 不修 collapse 组的 fp32 伪信号在**现有 GRPO 路径**上的表现（会改变现有算法数值）
- 不删 `ppo` 死分支（无关清理）
- 不改 `Sample.get_reward_value` 的现有签名（`dynamic_sampling_filters.py:13`、`agentic/rollout.py:1304` 依赖它）
