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

    def update_class_weights(self, new_weights: torch.Tensor) -> None:
        if self.class_weights is None:
            self.register_buffer("class_weights", new_weights.detach().clone())
        else:
            self.class_weights.copy_(new_weights.detach())

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


class OHEMCELoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = 255,
        ohem_ratio: float = 0.25,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.ohem_ratio = float(ohem_ratio)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def update_class_weights(self, new_weights: torch.Tensor) -> None:
        if self.class_weights is None:
            self.register_buffer("class_weights", new_weights.detach().clone())
        else:
            self.class_weights.copy_(new_weights.detach())

    def _ohem_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_pixel = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = targets != self.ignore_index
        losses = per_pixel[valid]
        if losses.numel() == 0:
            return per_pixel.new_tensor(0.0)
        k = max(1, int(losses.numel() * self.ohem_ratio))
        topk_loss, _ = torch.topk(losses, k=k, largest=True)
        return topk_loss.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, return_components: bool = False):
        ohem_ce = self._ohem_ce(logits, targets)
        if return_components:
            return ohem_ce, {
                "ohem_ce": float(ohem_ce.detach().item()),
                "total": float(ohem_ce.detach().item()),
            }
        return ohem_ce
