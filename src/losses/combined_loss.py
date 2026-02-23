from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.focalloss import FocalLoss


class CombinedCEFocalLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.focal_weight = float(focal_weight)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.focal = FocalLoss(
            gamma=focal_gamma,
            alpha=self.class_weights,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        focal = self.focal(logits, targets)
        return self.ce_weight * ce + self.focal_weight * focal
