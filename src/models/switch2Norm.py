from typing import Literal

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

NormType = Literal["bn", "syncbn", "none"]


def make_norm(norm: NormType, num_channels: int) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "syncbn":
        return nn.SyncBatchNorm(num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm}")


def freeze_bn_stats(m: nn.Module) -> None:
    if isinstance(m, _BatchNorm):
        m.eval()
