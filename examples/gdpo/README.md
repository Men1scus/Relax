# GDPO 示例：correctness + format 双奖励

本目录是 GDPO（[arXiv 2601.05242](https://arxiv.org/abs/2601.05242)）的最小可运行示例，用 Qwen3-0.6B 单卡在 GSM8K 上训练。

## 为什么需要 GDPO

奖励函数 `reward_gdpo.py` 返回两个分量：

- `correctness` —— `<answer>` 标签里的答案是否正确（0 或 1）
- `format` —— 输出是否同时带 `<think>` 与 `<answer>`（0、0.5 或 1）

这两个分量会**不同步**：模型可能答对但没按格式输出，也可能格式完美但答错。

GRPO 把它们加起来再做一次组内归一化。一旦某组的**总奖励**全相同（比如 8 条 rollout 全对、格式也全对），整组 advantage 归零，这 8 条样本白采了。

GDPO 分别对每个分量做组内标准化，再合并。同样这一组：`correctness` 塌缩、贡献 0，但只要 `format` 还有差异，梯度信号就还在。

## 运行

```bash
export MODEL_DIR=/path/to/models      # 需含 Qwen3-0.6B
export DATA_DIR=/path/to/data         # 需含 gsm8k/train.jsonl
export EXP_DIR=/path/to/experiments

bash examples/gdpo/run-qwen3-0.6B-1xgpu-gdpo.sh
```

### 数据要求

每条 prompt 需要**自带**格式要求，否则基座模型不会产出 `<think>`/`<answer>` 标签，`format` 分量会恒为 0（组内塌缩），GDPO 就退化成只看 `correctness`。准备数据时给 question 追加一句即可：

```python
instruction = (
    "\n\nThink step by step inside <think> </think> tags, then give only the "
    "final number inside <answer> </answer> tags."
)
df["question"] = df["question"] + instruction
```

**不要用 `--system-prompt` 代替**：`relax/utils/data/data_utils.py:181` 把 system message 的 content 构造成多模态 list（`content: [{"type": "text", ...}]`），Qwen3-0.6B 这类纯文本 chat template 渲染时会报
`TypeError: can only concatenate str (not "list") to str`。这是既有的框架限制，与 GDPO 无关。

## 参数说明

```bash
GDPO_ARGS=(
   --advantage-estimator gdpo
   --gdpo-reward-keys correctness format   # 独立归一化的分量，至少两个
   --gdpo-reward-weights 1.0 1.0           # 可省略，默认全 1
   --custom-rm-path examples.gdpo.reward_gdpo.reward_func
   --reward-key score                      # 必填：metrics 与 raw_reward 用的标量
   --n-samples-per-prompt 8                # 必须 ≥ 2
)
```

`--gdpo-reward-weights` 乘的是**归一化之后**的 advantage，不是原始 reward。经过第一步各分量已经是单位方差，所以权重表达的是相对重要性，与分量本身的量纲无关——把 `format` 的取值范围从 `[0,1]` 改成 `[0,100]` 不会改变训练结果。

## 换成自己的奖励

改 `reward_gdpo.py` 的 `compute_gdpo_reward`，返回的 dict 需要包含 `--gdpo-reward-keys` 列出的全部 key，外加 `--reward-key` 指定的那个标量。

分量缺失、非数值、bool、NaN/Inf 都会直接报错。这是有意的：静默填 0 会让一个坏掉的奖励函数看起来像是「这一维恰好塌缩了」，训练照跑，问题要很久以后才暴露。

## 已知偏差

1. **第三步的 batch 边界**。batch 白化作用在 advantage 层拿到的那一批上。本示例是 colocate 模式，等于完整训练批，与论文一致；fully-async 下会变成训练批的切片，影响一个全局正标量缩放。
2. **单个奖励时 GDPO 不等于 GRPO**，差一个正标量。要 GRPO 语义就用 `--advantage-estimator grpo`。
3. **`--n-samples-per-prompt 2` 时幅度信息丢失**：任意两个不同值标准化后恒为 ±0.7071。示例用 8 就是为了避开这一点。

## 冲突项

`--normalize-advantages` 和 `--custom-reward-post-process-path` 都不能与 GDPO 同用，参数校验阶段会直接报错。前者会造成双重白化；后者会整段短路奖励后处理，导致 GDPO 的前两步被静默跳过。
