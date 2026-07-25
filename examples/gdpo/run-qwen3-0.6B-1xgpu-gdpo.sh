#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-0.6B single-GPU GDPO training on GSM8K.
#
# GDPO (arXiv 2601.05242) standardizes each reward component within its prompt
# group before combining them, so a group where every rollout is correct but
# only some are well-formatted still carries signal. The reward function in
# reward_gdpo.py returns both components; --gdpo-reward-keys names them.
#
# Usage:
#   bash examples/gdpo/run-qwen3-0.6B-1xgpu-gdpo.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:=relax-gdpo}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=20}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-0.6B
   --ref-load ${MODEL_DIR}/Qwen3-0.6B
   --load ${EXP_DIR}/Qwen3-0.6B_mcore_gdpo/
   --save ${EXP_DIR}/Qwen3-0.6B_mcore_gdpo/
   --save-interval 100
   --max-actor-ckpt-to-keep 1
   --megatron-to-hf-mode bridge
)

SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/gsm8k/train.jsonl
   --input-key question
   --label-key answer
   --apply-chat-template
   --system-prompt "${SYSTEM_PROMPT}"
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1.0
   --global-batch-size 16
)

# The two reward components come from examples/gdpo/reward_gdpo.py.
# --reward-key selects the scalar used for metrics and the raw_reward column;
# --gdpo-reward-keys names the components GDPO standardizes independently.
GDPO_ARGS=(
   --advantage-estimator gdpo
   --gdpo-reward-keys correctness format
   --gdpo-reward-weights 1.0 1.0
   --custom-rm-path examples.gdpo.reward_gdpo.reward_func
   --reward-key score
   --eps-clip 0.2
   --kl-coef 0.00
   --entropy-coef 0.00
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

WANDB_ARGS=(
   --use-tensorboard
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen3-0.6b-gdpo-gpu1-${now}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 1], "rollout": [1, 1]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --use-health-check \
    --balance-data \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GDPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee log/qwen3-0.6b-gdpo-gpu1-${now}.log
