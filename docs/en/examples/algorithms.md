# Algorithm Reference

Relax supports multiple policy gradient algorithms, all selected via the `--advantage-estimator` flag. This document covers all integrated algorithms (for On-Policy Distillation, see the [dedicated page](./on-policy-distillation.md)).

All algorithms share the same training scripts — simply replace the `GRPO_ARGS` block in the script with the corresponding algorithm's arguments.

---

## GRPO

GRPO (Group Relative Policy Optimization) is the default algorithm in Relax. It broadcasts the group-relative scalar reward to every token and uses a standard PPO-Clip objective.

Reference: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300).

### How It Works

The GRPO objective is the standard PPO-Clip:

$$J_\text{GRPO}(\theta) = \mathbb{E} \left[ \min\!\left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\hat{A}_t \right) \right]$$

where $r_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$, and $\hat{A}_t$ is the group-relative advantage (reward minus group mean, normalized by group standard deviation). Gradients are zeroed out when the ratio exceeds the clipping bounds.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator grpo` | default | Enable GRPO |
| `--eps-clip` | `0.2` | Clipping margin (ratio range = `[1-ε, 1+ε]`) |
| `--eps-clip-high` | same as `--eps-clip` | Upper clipping margin; can be set differently for asymmetric clipping |
| `--clip-grad` | — | Gradient clipping norm |

### Quick Start

GRPO is the default algorithm — no parameter changes needed. Just run the training script directly:

```bash
MODEL_DIR=/path/to/model \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/exp \
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

---

## CISPO

CISPO (Clipped Importance-ratio Soft Policy Optimization) preserves gradient signal for out-of-trust-region tokens instead of zeroing it out. It caps gradient magnitude via a stop-gradient'd coefficient while keeping the gradient direction alive.

Reference: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585).

### How It Works

The CISPO objective is:

$$J_\text{CISPO}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\ \{o_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \text{sg}\!\left(\hat{r}_{i,t}(\theta)\right) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \right]$$

where $\hat{r}_{i,t}(\theta)$ is the clipped importance-sampling weight:

$$\hat{r}_{i,t}(\theta) = \text{clip}\!\left(r_{i,t}(\theta),\ 1 - \varepsilon_\text{low}^\text{IS},\ 1 + \varepsilon_\text{high}^\text{IS}\right)$$

and $r_{i,t}(\theta) = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_\text{old}}(o_{i,t} \mid q, o_{i,<t})$. Gradients flow **only** through $\log\pi_\theta$; both $\hat{r}_{i,t}$ and $\hat{A}_{i,t}$ are stop-gradient'd.

### Key Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--advantage-estimator cispo` | — | — | Enable CISPO |
| `--eps-clip` | `0.2` | `0.2` | Lower clipping margin (ratio lower bound = `1 - eps_clip`) |
| `--eps-clip-high` | same as `--eps-clip` | `10` | Upper clipping margin (ratio upper bound = `1 + eps_clip_high`). Set to `10` to effectively unclamp the upper side |
| `--kl-loss-coef` | `0.0` | `0.001` | KL loss coefficient. Recommended: `0.001` to add a small KL penalty that constrains policy drift |
| `--use-kl-loss` | off | on | Enable KL loss computation (required for `--kl-loss-coef` to take effect) |
| `--use-tis` | off | on | Token Importance Sampling — recommended to enable with CISPO |
| `--clip-grad` | — | `1.0` | Gradient clipping norm |

### Quick Start

Use any existing GRPO training script and replace `GRPO_ARGS` with `CISPO_ARGS`:

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

GSPO (Group-wise Sequence-level Policy Optimization) differs from GRPO in how KL divergence is computed: GSPO uses **sequence-level** KL instead of per-token KL. Every token in a sequence shares the same KL value (the mean over all tokens in that sequence), providing uniform constraint strength within a sequence.

### How It Works

GSPO uses the same PPO-Clip objective as GRPO, but the ratio is computed from sequence-level KL:

$$\text{KL}_\text{seq} = \frac{1}{|o|} \sum_{t=1}^{|o|} \left(\log\pi_{\theta_\text{old}}(o_t) - \log\pi_\theta(o_t)\right)$$

Every token's ratio is $r_t = \exp(-\text{KL}_\text{seq})$, rather than an independent per-token ratio.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator gspo` | — | Enable GSPO |
| `--eps-clip` | `0.2` | Clipping margin |
| `--eps-clip-high` | same as `--eps-clip` | Upper clipping margin |
| `--clip-grad` | — | Gradient clipping norm |

### Quick Start

```bash
GSPO_ARGS=(
   --advantage-estimator gspo
   --eps-clip 0.2
)
```

---

## SAPO

SAPO (Soft Adaptive Policy Optimization) replaces hard clipping with a smooth sigmoid gate. The gate's steepness is controlled by a temperature parameter, implementing a differentiable trust region constraint.

### How It Works

SAPO's core is a sigmoid gate centered at ratio=1:

$$f(r) = \frac{4}{\tau} \cdot \sigma\!\left(\tau(r - 1)\right)$$

where $\sigma$ is the sigmoid function, and $\tau$ is selected based on the advantage sign:

- $A > 0$: use $\tau_\text{pos}$ (default 1.0)
- $A \leq 0$: use $\tau_\text{neg}$ (default 1.05, stronger suppression for negative tokens)

SAPO objective: $J_\text{SAPO}(\theta) = f(r) \cdot A$

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator sapo` | — | Enable SAPO |
| `--sapo-tau-pos` | `1.0` | Temperature for positive advantages |
| `--sapo-tau-neg` | `1.05` | Temperature for negative advantages (higher = stronger suppression) |
| `--clip-grad` | — | Gradient clipping norm |

### Quick Start

```bash
SAPO_ARGS=(
   --advantage-estimator sapo
   --sapo-tau-pos 1.0
   --sapo-tau-neg 1.05
)
```

---

## GDPO

GDPO (Group reward-Decoupled Normalization Policy Optimization, [arXiv 2601.05242](https://arxiv.org/abs/2601.05242)) targets **multi-reward** training. It standardizes each reward component within its prompt group and only then combines them, instead of summing the rewards first and normalizing once as GRPO does.

### How It Works

For prompt $i$ with $G$ rollouts and $n$ reward components:

**Step 1 — per-reward group standardization:**

$$A_k^{(i,j)} = \frac{r_k^{(i,j)} - \mathrm{mean}_j\{r_k^{(i,\cdot)}\}}{\mathrm{std}_j\{r_k^{(i,\cdot)}\} + \epsilon}$$

**Step 2 — weighted sum:**

$$A_\text{sum}^{(i,j)} = \sum_k w_k A_k^{(i,j)}$$

The weights multiply the **normalized** advantages, not the raw rewards. After step 1 every component is on the same scale, so a weight expresses relative importance rather than the component's units.

**Step 3 — batch-wise whitening:**

$$\hat{A}^{(i,j)} = \frac{A_\text{sum}^{(i,j)} - \mathrm{mean}_\text{batch}}{\mathrm{std}_\text{batch} + \epsilon}$$

**Why this beats GRPO:** when one component is constant across a group (reward collapse), GRPO's summed reward collapses too, the whole group's advantages go to zero, and the samples are wasted. Under GDPO only *that component* contributes zero while the others still carry signal.

**On $\epsilon$:** GDPO uses $\epsilon = 10^{-4}$ at both steps, matching the reference implementation (the `scale_rewards` GDPO branch of TRL's `GRPOTrainer`), whereas GRPO / GSPO / SAPO / CISPO keep this repository's existing $10^{-6}$. The two only diverge on near-degenerate groups: with binary rewards and a group of 8 the within-group standard deviation is around 0.4 and the constants differ by 0.02%, but a continuous reward (the paper's maths setup scores response length) can leave a group at a standard deviation of ~$10^{-3}$, where $10^{-4}$ damps that group's signal by about 7% against 0.08% for $10^{-6}$. Groups that collapse *exactly* never reach this division; they are detected by exact equality and zeroed.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator gdpo` | — | Enable GDPO |
| `--gdpo-reward-keys` | — | **Required**, at least two. Keys in the reward dict to standardize independently, e.g. `correctness format` |
| `--gdpo-reward-weights` | all 1.0 | Per-component weights; length must match `--gdpo-reward-keys` |
| `--reward-key` | — | **Required**; selects the scalar used for metrics and the `raw_reward` column |
| `--n-samples-per-prompt` | — | Must be >= 2 (the unbiased group std is undefined at $G=1$) |

The reward function must return a dict containing every key. A missing key, a non-numeric value, a bool, or NaN/Inf raises rather than defaulting to 0.0 — a silently zeroed component is indistinguishable from a genuinely collapsed one.

### Quick Start

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

A complete runnable example lives in [`examples/gdpo/`](https://github.com/redai-infra/Relax/tree/main/examples/gdpo).

### Known Deviations

Two differences between this implementation and the paper. Confirm they are acceptable before training. Step 3's batch boundary used to be a third; it has since been corrected — see below.

**Step 3's batch boundary (now aligned).** Eq. 6 normalises over one training batch. The caller merges `num_rollout_minis` of them with `concat_rollout_batches` before the advantage stage, so step 3 has to be told where the boundaries are. They travel in `ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY`, which all three actor paths (colocate and hybrid) set; `loss.py` forwards them to the advantage dispatcher as `mini_batch_sizes`, and **only GDPO reads it** — every other estimator absorbs it in `**_unused` and is bit-identical either way. Each segment all-reduces across the data-parallel group, so the statistics cover both a whole training batch and every rank. Why it matters: whitening merged batches centres them all on a pooled mean, and on a measured example four of eight samples **change sign** — a different objective, not a precision difference.

**`--fully-async` remains unsupported** and is rejected during argument validation: it hands advantage computation to the single-replica Advantages deployment, which has no data-parallel group, never sees the batch boundaries, and consumes one `global_batch_size / num_iters_per_train_update` slice at a time — when that quotient is 1 the whitened output is identically zero and the run trains on no signal at all, quietly.

1. **A single reward does not reduce to GRPO.** Step 3 still applies, leaving a positive scalar difference from GRPO (data-dependent, measured around 1.21). Use `--advantage-estimator grpo` if you want GRPO semantics.
2. **$G=2$ discards magnitude.** Any two distinct values standardize to exactly $\pm 1/\sqrt{2}$, so with a group of two the only thing distinguishing components is their weights.

### Mutually Exclusive Options

- `--normalize-advantages`: step 3 already whitens per sequence; adding the token-level pass on top is not meaningful.
- `--custom-reward-post-process-path`: that hook short-circuits reward post-processing entirely, silently skipping steps 1 and 2 while the run still reports itself as GDPO.
- `--fully-async`: see above.

All three fail during argument validation. Combining it with `--dynamic-sampling-filter-path` logs a warning instead: the built-in `check_reward_nonzero_std` judges a group by the single `--reward-key` scalar and may drop groups whose signal lives in the other components.

---

## Algorithm Comparison

| Algorithm | Advantage Computation | Policy Loss | KL Constraint |
|-----------|----------------------|-------------|---------------|
| **GRPO** | Group-relative reward | PPO-Clip (hard clip) | Optional KL loss |
| **CISPO** | Group-relative reward | Stop-gradient coefficient | Recommended KL loss |
| **GSPO** | Group-relative reward | PPO-Clip + sequence-level KL | Sequence-level ratio |
| **SAPO** | Group-relative reward | Sigmoid gate | Temperature-controlled |
| **GDPO** | Per-reward group standardization + weighted sum + batch whitening | PPO-Clip (hard clip) | Optional KL loss |

## Next Steps

- [Quick Start](./quick-start.md)
- [On-Policy Distillation](./on-policy-distillation.md)
- [Generative Reward Model](./generative-reward-model.md)
