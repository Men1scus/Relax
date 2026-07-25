# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared numerical constants and guards for the algorithm implementations.

Both the reward stage and the advantage stage standardise values by dividing by
a standard deviation, so they need the same epsilon and the same notion of
"this group carries no signal".  Keeping those here prevents the two stages
from drifting apart.
"""

import torch


STD_EPS = 1e-6
"""Epsilon added to a standard deviation before dividing by it.

Matches the value the pre-registry GRPO path used, so group standardisation is
numerically identical across every algorithm in this repository.  Upstream
implementations disagree (the GDPO paper writes no epsilon at all, TRL uses
1e-4, verl 1e-6, ms-swift 1e-8); internal consistency wins over cross-framework
parity because it is what makes the equivalence tests meaningful.
"""

COLLAPSE_RTOL = 1e-6
"""Relative tolerance for calling a spread "no signal at all".

This has to be relative rather than absolute.  Summing N equal float32 values
and dividing by N does not return the value exactly, so a genuinely collapsed
group still has a residual standard deviation proportional to its magnitude:
0.7 repeated seven times gives std ~= 6.4e-8.  An absolute threshold tight
enough for small rewards would miss that, and ``residual / (residual + 1e-6)``
then amplifies pure rounding noise into a +-5.6e-2 "advantage".  float32
epsilon is ~1.2e-7, so 1e-6 leaves room for accumulated rounding while staying
far below any real reward spread.
"""


def is_collapsed(values: torch.Tensor, std: torch.Tensor) -> bool:
    """Whether ``values`` carry only rounding noise rather than usable
    signal."""
    if not torch.isfinite(std):
        return True
    return bool(std <= COLLAPSE_RTOL * values.abs().max())


def collapse_mask(values: torch.Tensor, std: torch.Tensor, dim: int) -> torch.Tensor:
    """Per-column version of :func:`is_collapsed` for a ``[G, K]`` group.

    Returns a boolean tensor shaped like ``std``: ``True`` where that column's
    spread is indistinguishable from rounding noise.
    """
    scale = values.abs().amax(dim=dim)
    return ~torch.isfinite(std) | (std <= COLLAPSE_RTOL * scale)
