from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.focalloss import FocalLoss


class CombinedCEFocalLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.focal_weight = float(focal_weight)
        self.ignore_index = int(ignore_index)
        self.focal = FocalLoss(gamma=focal_gamma, alpha=None, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        focal = self.focal(logits, targets)
        return self.ce_weight * ce + self.focal_weight * focal
