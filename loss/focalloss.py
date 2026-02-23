from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | torch.Tensor | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index,
        )

        valid_mask = targets != self.ignore_index
        if not torch.any(valid_mask):
            return logits.new_tensor(0.0)

        ce_loss = ce_loss[valid_mask]
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(logits.device, dtype=logits.dtype)
                alpha_t = alpha[targets[valid_mask]]
                focal_loss = alpha_t * focal_loss
            else:
                focal_loss = self.alpha * focal_loss

        return focal_loss.mean()
