from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, kernel_size: int = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.kernel_size = max(1, int(kernel_size))
        self.eps = float(eps)

    def _target_boundary_map(self, labels: torch.Tensor) -> torch.Tensor:
        labels_f = labels.float().unsqueeze(1)
        pooled = F.max_pool2d(labels_f, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        return (pooled != labels_f).float()

    def _pred_boundary_map(self, probs: torch.Tensor) -> torch.Tensor:
        score = probs.max(dim=1, keepdim=True).values
        pooled_max = F.max_pool2d(score, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        pooled_min = -F.max_pool2d(-score, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        return (pooled_max - pooled_min).clamp(0.0, 1.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)

        valid = (targets != self.ignore_index).float().unsqueeze(1)
        tgt = targets.clone()
        tgt[targets == self.ignore_index] = 0

        pred_boundary = self._pred_boundary_map(probs)
        target_boundary = self._target_boundary_map(tgt)

        pred_boundary_logits = torch.logit(pred_boundary.clamp(self.eps, 1.0 - self.eps))            
        loss_map = F.binary_cross_entropy_with_logits(pred_boundary_logits, target_boundary, reduction="none")
        loss_map = loss_map * valid
        denom = valid.sum().clamp_min(self.eps)
        return loss_map.sum() / denom
