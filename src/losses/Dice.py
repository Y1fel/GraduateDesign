import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLossMulticlass(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n, c, h, w = logits.shape
        probs = F.softmax(logits, dim=1)

        # mask ignore
        valid = (target != self.ignore_index)  # (N,H,W)
        if valid.sum() == 0:
            return logits.new_tensor(0.0)

        tgt = target.clone()
        tgt[~valid] = 0
        tgt_oh = F.one_hot(tgt, num_classes=self.num_classes).permute(0, 3, 1, 2).float()  # (N,C,H,W)

        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        tgt_oh = tgt_oh * valid_f

        # dice per class
        inter = (probs * tgt_oh).sum(dim=(0, 2, 3))
        denom = (probs + tgt_oh).sum(dim=(0, 2, 3))

        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()
