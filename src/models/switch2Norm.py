from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

NormType = Literal["bn", "gn", "none"]


def _best_group_count(num_channels: int, max_groups: int = 32) -> int:
    g_max = min(max_groups, num_channels)
    for g in range(g_max, 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def make_norm(norm: NormType, num_channels: int, *, num_groups: int = 32) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "gn":
        g = _best_group_count(num_channels, max_groups=num_groups)
        return nn.GroupNorm(g, num_channels)
    if norm == "none":
        return nn.Identity()


def freeze_bn_stats(m: nn.Module) -> None:
    if isinstance(m, _BatchNorm):
        m.eval()
