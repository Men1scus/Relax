# 算法参考

Relax 支持多种策略梯度算法，均通过 `--advantage-estimator` 参数选择。本文档覆盖所有已集成的算法（OPD 在线策略蒸馏请参阅[单独文档](./on-policy-distillation.md)）。

所有算法共享同一套训练脚本——只需替换脚本中的 `GRPO_ARGS` 为对应算法的参数块即可。

---

## GRPO

GRPO（Group Relative Policy Optimization）是 Relax 的默认算法。将组内标量奖励广播到每个 token，使用 PPO 风格的裁剪目标函数。

参考论文：[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)。

### 算法原理

GRPO 目标函数为标准 PPO-Clip：

$$J_\text{GRPO}(\theta) = \mathbb{E} \left[ \min\!\left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$，$\hat{A}_t$ 是组相对 advantage（组内奖励减去组均值，按组标准差归一化）。当 ratio 超出裁剪边界时，梯度被直接置零。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator grpo` | 默认 | 启用 GRPO |
| `--eps-clip` | `0.2` | 裁剪边距（ratio 范围 = `[1-ε, 1+ε]`） |
| `--eps-clip-high` | 与 `--eps-clip` 相同 | 上方裁剪边距，可设为不同值以构造非对称裁剪 |
| `--clip-grad` | — | 梯度裁剪范数 |

### 快速开始

GRPO 是默认算法，无需修改参数。直接使用训练脚本即可：

```bash
MODEL_DIR=/path/to/model \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/exp \
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

---

## CISPO

CISPO（Clipped Importance-ratio Soft Policy Optimization）对超出信任域的 token 保留梯度信号，而非将其清零。通过 stop-gradient 系数限制梯度幅度，同时保留梯度方向。

参考论文：[MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)。

### 算法原理

CISPO 目标函数为：

$$J_\text{CISPO}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\ \{o_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \text{sg}\!\left(\hat{r}_{i,t}(\theta)\right) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \right]$$

其中 $\hat{r}_{i,t}(\theta)$ 是裁剪后的重要性采样权重：

$$\hat{r}_{i,t}(\theta) = \text{clip}\!\left(r_{i,t}(\theta),\ 1 - \varepsilon_\text{low}^\text{IS},\ 1 + \varepsilon_\text{high}^\text{IS}\right)$$

$r_{i,t}(\theta) = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_\text{old}}(o_{i,t} \mid q, o_{i,<t})$。梯度**只**流过 $\log\pi_\theta$，$\hat{r}_{i,t}$ 和 $\hat{A}_{i,t}$ 均被 stop-gradient 处理。

### 关键参数

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--advantage-estimator cispo` | — | — | 启用 CISPO |
| `--eps-clip` | `0.2` | `0.2` | 下方裁剪边距（ratio 下界 = `1 - eps_clip`） |
| `--eps-clip-high` | 与 `--eps-clip` 相同 | `10` | 上方裁剪边距（ratio 上界 = `1 + eps_clip_high`）。设为 `10` 可近似取消上侧裁剪 |
| `--kl-loss-coef` | `0.0` | `0.001` | KL 损失系数。推荐设为 `0.001`，添加小幅 KL 惩罚以约束策略偏移 |
| `--use-kl-loss` | 关闭 | 开启 | 启用 KL 损失计算（`--kl-loss-coef` 生效的前提） |
| `--use-tis` | 关闭 | 开启 | Token Importance Sampling，推荐与 CISPO 同时开启 |
| `--clip-grad` | — | `1.0` | 梯度裁剪范数 |

### 快速开始

使用任意 GRPO 训练脚本，将 `GRPO_ARGS` 替换为 `CISPO_ARGS`：

```bash
CISPO_ARGS=(
   --advantage-estimator cispo
   --use-kl-loss
   --kl-loss-coef 0.001
   --eps-clip 0.2
   --eps-clip-high 10
   --use-tis
)
```

---

## GSPO

GSPO（Group-wise Sequence-level Policy Optimization）与 GRPO 的区别在于 KL 散度的计算方式：GSPO 使用**序列级** KL 而非逐 token KL。每个 token 的 KL 值等于该序列所有 token KL 的均值，这为序列内所有 token 提供统一的约束强度。

### 算法原理

GSPO 使用与 GRPO 相同的 PPO-Clip 目标函数，但 ratio 的计算基于序列级 KL：

$$\text{KL}_\text{seq} = \frac{1}{|o|} \sum_{t=1}^{|o|} \left(\log\pi_{\theta_\text{old}}(o_t) - \log\pi_\theta(o_t)\right)$$

每个 token 的 ratio 均为 $r_t = \exp(-\text{KL}_\text{seq})$，而非各自独立的 token 级 ratio。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator gspo` | — | 启用 GSPO |
| `--eps-clip` | `0.2` | 裁剪边距 |
| `--eps-clip-high` | 与 `--eps-clip` 相同 | 上方裁剪边距 |
| `--clip-grad` | — | 梯度裁剪范数 |

### 快速开始

```bash
GSPO_ARGS=(
   --advantage-estimator gspo
   --eps-clip 0.2
)
```

---

## SAPO

SAPO（Soft Adaptive Policy Optimization）用平滑的 sigmoid 门控替代硬裁剪。通过温度参数控制门控曲线的陡峭程度，实现可微的信任域约束。

### 算法原理

SAPO 的核心是一个以 ratio=1 为中心的 sigmoid 门控函数：

$$f(r) = \frac{4}{\tau} \cdot \sigma\!\left(\tau(r - 1)\right)$$

其中 $\sigma$ 是 sigmoid 函数，$\tau$ 根据 advantage 的符号选择不同的温度：

- $A > 0$: 使用 $\tau_\text{pos}$（默认 1.0）
- $A \leq 0$: 使用 $\tau_\text{neg}$（默认 1.05，对 negative token 更强的抑制）

SAPO 目标：$J_\text{SAPO}(\theta) = f(r) \cdot A$

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator sapo` | — | 启用 SAPO |
| `--sapo-tau-pos` | `1.0` | positive advantage 的温度参数 |
| `--sapo-tau-neg` | `1.05` | negative advantage 的温度参数（更高 = 更强抑制） |
| `--clip-grad` | — | 梯度裁剪范数 |

### 快速开始

```bash
SAPO_ARGS=(
   --advantage-estimator sapo
   --sapo-tau-pos 1.0
   --sapo-tau-neg 1.05
)
```

---

## GDPO

GDPO（Group reward-Decoupled Normalization Policy Optimization，[arXiv 2601.05242](https://arxiv.org/abs/2601.05242)）面向**多奖励**训练。它对每个奖励分量分别做组内标准化，再合并——而不是像 GRPO 那样先把多个奖励加起来再归一化。

### 算法原理

设第 $i$ 个 prompt 采样 $G$ 条 rollout，共 $n$ 个奖励分量。

**第一步 —— 逐奖励组内标准化**：

$$A_k^{(i,j)} = \frac{r_k^{(i,j)} - \mathrm{mean}_j\{r_k^{(i,\cdot)}\}}{\mathrm{std}_j\{r_k^{(i,\cdot)}\} + \epsilon}$$

**第二步 —— 加权求和**：

$$A_\text{sum}^{(i,j)} = \sum_k w_k A_k^{(i,j)}$$

注意权重乘在**归一化后的 advantage** 上，不是乘在原始 reward 上。经过第一步各分量已在同一尺度，权重表达的是相对重要性，而不是分量的量纲。

**第三步 —— batch 级白化**：

$$\hat{A}^{(i,j)} = \frac{A_\text{sum}^{(i,j)} - \mathrm{mean}_\text{batch}}{\mathrm{std}_\text{batch} + \epsilon}$$

**相对 GRPO 的收益**：某个分量在组内全相同时（reward collapse），GRPO 的总奖励也会塌缩，整组样本 advantage 归零、被白白丢弃；GDPO 下只有**该分量**贡献 0，其它分量仍然提供学习信号。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator gdpo` | — | 启用 GDPO |
| `--gdpo-reward-keys` | — | **必填**，至少两个。奖励函数返回的 dict 中要独立归一化的 key，如 `correctness format` |
| `--gdpo-reward-weights` | 全 1.0 | 各分量权重，长度须与 `--gdpo-reward-keys` 一致 |
| `--reward-key` | — | **必填**，选出用于 metrics 与 `raw_reward` 列的标量 |
| `--n-samples-per-prompt` | — | 必须 ≥ 2（组内无偏标准差在 $G=1$ 时无定义） |

奖励函数必须返回包含全部 key 的 dict。缺 key、非数值、bool、NaN/Inf 都会直接报错而不是填 0——静默填 0 会把契约违约伪装成真实的 reward collapse。

### 快速开始

```bash
GDPO_ARGS=(
   --advantage-estimator gdpo
   --gdpo-reward-keys correctness format
   --gdpo-reward-weights 1.0 1.0
   --custom-rm-path examples.gdpo.reward_gdpo.reward_func
   --reward-key score
   --n-samples-per-prompt 8
)
```

完整可运行示例见 [`examples/gdpo/`](https://github.com/redai-infra/Relax/tree/main/examples/gdpo)。

### 已知偏差

以下三点是实现与论文之间的实际差异，训练前请确认可以接受：

1. **第三步的 batch 边界**。colocate（及 hybrid）模式下，每个数据并行 rank 只持有 `global_batch_size / dp_size` 的分片，因此第三步会对 `count/sum/sumsq` 做一次跨 DP 的 all-reduce，统计量覆盖完整训练批，与论文一致。**`--fully-async` 不受支持，参数校验阶段会直接拒绝**：那条路径把 advantage 计算交给单副本的 Advantages 服务，它没有数据并行通信域，且每次只消费 `global_batch_size / num_iters_per_train_update` 的一个切片；当这个商为 1 时白化输出恒为 0，训练会安静地在零信号上跑完。要支持它需要按训练批边界做 cohort 累积，属于后续工作。
2. **单个奖励时 GDPO 不退化为 GRPO**。第三步仍然生效，结果与 GRPO 相差一个正标量（与数据相关，实测约 1.21）。要 GRPO 语义就直接用 `--advantage-estimator grpo`。
3. **$G=2$ 时幅度信息丢失**。任意两个不同值经无偏标准化后恒为 $\pm 1/\sqrt{2}$，此时分量之间的区分度只来自权重。

### 互斥项

- 不能与 `--normalize-advantages` 同用：第三步已经做过序列级白化，再叠加 token 级白化没有意义。
- 不能与 `--custom-reward-post-process-path` 同用：该钩子会整段短路奖励后处理，导致第一、二步被静默跳过，而训练日志仍然显示算法是 GDPO。
- 不能与 `--fully-async` 同用（见上）。

三者都会在参数校验阶段直接报错。配合 `--dynamic-sampling-filter-path` 时会给出警告：内置的 `check_reward_nonzero_std` 只看 `--reward-key` 那一个标量，可能丢掉只存在于其它分量的信号。

---

## 算法对比

| 算法 | Advantage 计算 | 策略损失 | KL 约束方式 |
|------|---------------|---------|-----------|
| **GRPO** | 组相对奖励 | PPO-Clip（硬裁剪） | 可选 KL loss |
| **CISPO** | 组相对奖励 | Stop-gradient 系数 | 推荐 KL loss |
| **GSPO** | 组相对奖励 | PPO-Clip + 序列级 KL | 序列级 ratio |
| **SAPO** | 组相对奖励 | Sigmoid 门控 | 温度控制 |
| **GDPO** | 逐奖励组内标准化 + 加权求和 + batch 白化 | PPO-Clip（硬裁剪） | 可选 KL loss |

## 下一步

- [快速开始](./quick-start.md)
- [在线策略蒸馏](./on-policy-distillation.md)
- [生成式奖励模型](./generative-reward-model.md)
