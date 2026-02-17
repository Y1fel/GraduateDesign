# src/losses/Dice.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Multi-class soft Dice loss for logits.
    Key fixes:
      - supports ignore_index
      - only computes dice on classes that appear in target (avoid penalizing absent classes)
      - optional exclude background (class 0)
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        smooth: float = 1.0,
        eps: float = 1e-6,
        include_background: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.smooth = float(smooth)
        self.eps = float(eps)
        self.include_background = bool(include_background)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, C, H, W)
        target: (N, H, W) int64
        """
        if logits.dim() != 4:
            raise ValueError(f"logits must be 4D (N,C,H,W), got {logits.shape}")
        if target.dim() != 3:
            raise ValueError(f"target must be 3D (N,H,W), got {target.shape}")

        n, c, h, w = logits.shape
        if c != self.num_classes:
            raise ValueError(f"num_classes={self.num_classes} but logits has C={c}")

        # probs
        probs = F.softmax(logits, dim=1)

        # ignore mask
        valid_mask = (target != self.ignore_index)  # (N,H,W)
        # clamp ignore pixels to 0 before one_hot (won't matter because masked out)
        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        # one-hot target
        target_1h = F.one_hot(target_safe, num_classes=self.num_classes)  # (N,H,W,C)
        target_1h = target_1h.permute(0, 3, 1, 2).contiguous().float()   # (N,C,H,W)

        # apply valid mask to both
        valid_mask_f = valid_mask.unsqueeze(1).float()  # (N,1,H,W)
        probs = probs * valid_mask_f
        target_1h = target_1h * valid_mask_f

        # dice per class
        dims = (0, 2, 3)
        intersection = (probs * target_1h).sum(dims)                # (C,)
        cardinality = probs.sum(dims) + target_1h.sum(dims)         # (C,)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)  # (C,)

        # only keep classes that appear in target (avoid punishing absent classes)
        target_pixels_per_class = target_1h.sum(dims)  # (C,)
        present = target_pixels_per_class > 0

        if not self.include_background:
            present = present & (torch.arange(self.num_classes, device=present.device) != 0)

        if present.any():
            dice = dice[present]
            loss = 1.0 - dice
        else:
            # extremely rare: whole batch is ignore_index
            loss = dice.new_tensor(0.0)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
