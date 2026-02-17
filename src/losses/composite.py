import torch
import torch.nn as nn
import torch.nn.functional as F

from .Dice import SoftDiceLoss


class CrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.0,
        dice_include_background: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.label_smoothing = float(label_smoothing)

        self.dice = SoftDiceLoss(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            include_background=dice_include_background,
            smooth=1.0,
            eps=1e-6,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target.long(),
            weight=None,  # ✅ no class weights
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        dice = self.dice(logits, target.long())
        return self.ce_weight * ce + self.dice_weight * dice
