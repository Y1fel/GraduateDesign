import torch
import torch.nn as nn
import torch.nn.functional as F

from .Dice import SoftDiceLoss
from .Focal import FocalLoss


class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, min_kept: int = 100000, thresh: float = 0.7):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.min_kept = int(min_kept)
        self.thresh = float(thresh)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, class_weight: torch.Tensor | None = None) -> torch.Tensor:
        n, c, h, w = logits.shape
        if target.shape != (n, h, w):
            raise ValueError(f"target shape must be {(n, h, w)}, got {tuple(target.shape)}")

        per_pixel_loss = F.cross_entropy(
            logits,
            target.long(),
            weight=class_weight,
            ignore_index=self.ignore_index,
            reduction="none",
        ).reshape(-1)
        target_f = target.reshape(-1)
        valid = target_f != self.ignore_index
        if valid.sum() == 0:
            return per_pixel_loss.sum() * 0.0

        valid_loss = per_pixel_loss[valid]
        kept = min(self.min_kept, valid_loss.numel())
        if kept <= 0:
            return valid_loss.mean()

        topk_loss, _ = torch.topk(valid_loss, k=kept, largest=True, sorted=True)
        if topk_loss.numel() > 0:
            hard_mask = valid_loss >= max(self.thresh, topk_loss[-1].item())
            hard = valid_loss[hard_mask]
            if hard.numel() > 0:
                return hard.mean()
        return topk_loss.mean()


class CrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.0,
        dice_include_background: bool = True,
        ce_variant: str = "ce",
        class_weights: torch.Tensor | None = None,
        focal_gamma: float = 2.0,
        ohem_min_kept: int = 100000,
        ohem_thresh: float = 0.7,
        boundary_weight: float = 0.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.label_smoothing = float(label_smoothing)
        self.ce_variant = str(ce_variant).lower()
        self.boundary_weight = float(boundary_weight)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        self.focal = FocalLoss(
            gamma=focal_gamma,
            alpha=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="mean",
        )
        self.ohem = OHEMCrossEntropyLoss(
            ignore_index=self.ignore_index,
            min_kept=ohem_min_kept,
            thresh=ohem_thresh,
        )

        self.dice = SoftDiceLoss(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            include_background=dice_include_background,
            smooth=1.0,
            eps=1e-6,
            reduction="mean",
        )

    def _compute_boundary_mask(self, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        t = target.clone()
        t[~valid] = 0

        boundary = torch.zeros_like(valid)

        diff_h = (t[:, 1:, :] != t[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
        boundary[:, 1:, :] |= diff_h
        boundary[:, :-1, :] |= diff_h

        diff_w = (t[:, :, 1:] != t[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
        boundary[:, :, 1:] |= diff_w
        boundary[:, :, :-1] |= diff_w

        return boundary & valid

    def _boundary_weighted_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_pixel_ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        valid = target != self.ignore_index
        if valid.sum() == 0:
            return per_pixel_ce.sum() * 0.0

        pixel_weight = torch.ones_like(per_pixel_ce)
        if self.boundary_weight > 0:
            boundary_mask = self._compute_boundary_mask(target)
            pixel_weight = pixel_weight + self.boundary_weight * boundary_mask.float()

        weighted_loss = (per_pixel_ce * pixel_weight * valid.float()).sum()
        normalizer = (pixel_weight * valid.float()).sum().clamp_min(1.0)
        return weighted_loss / normalizer

    def forward_components(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        target = target.long()
        if self.ce_variant == "ce":
            ce = self._boundary_weighted_ce(logits, target)
        elif self.ce_variant == "focal":
            ce = self.focal(logits, target)
        elif self.ce_variant == "ohem":
            ce = self.ohem(logits, target, class_weight=self.class_weights)
        else:
            raise ValueError(f"Unknown ce_variant: {self.ce_variant}")

        dice = self.dice(logits, target)
        total = self.ce_weight * ce + self.dice_weight * dice
        return {"total": total, "ce": ce, "dice": dice}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward_components(logits, target)["total"]
