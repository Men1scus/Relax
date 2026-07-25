# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-component reward for the GDPO example: correctness and format.

GDPO standardizes each component within its prompt group before combining them,
so a group where every rollout is correct but only some are well-formatted still
carries a learning signal. Under GRPO the summed reward would collapse and the
whole group would contribute nothing.

Wire it up with::

    --advantage-estimator gdpo
    --gdpo-reward-keys correctness format
    --custom-rm-path examples.gdpo.reward_gdpo.reward_func
    --reward-key score

``score`` is what ``--reward-key`` selects for metrics and for the ``raw_reward``
column; the training signal comes from the two components.
"""

import re
from typing import Any


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_answer(response: str) -> str | None:
    match = _ANSWER_RE.search(response)
    return match.group(1).strip() if match else None


def compute_gdpo_reward(response: str, label: Any) -> dict[str, float]:
    """Score one response on answer correctness and on output format.

    The two components are deliberately decorrelated: a response can be correct
    without the expected tags, and well-formatted while wrong. That is the
    situation GDPO handles better than a summed reward.
    """
    answer = _extract_answer(response)
    correctness = 1.0 if answer is not None and answer == str(label).strip() else 0.0

    has_think = _THINK_RE.search(response) is not None
    format_score = 0.5 * float(has_think) + 0.5 * float(answer is not None)

    return {
        "score": correctness,
        "correctness": correctness,
        "format": format_score,
    }


async def reward_func(args: Any, sample: Any, **kwargs: Any) -> dict[str, float]:
    """Entry point for ``--custom-rm-path``."""
    return compute_gdpo_reward(sample.response, sample.label)
