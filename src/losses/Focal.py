from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    FL = - alpha_t * (1 - p_t)^gamma * log(p_t)
    - logits: (N, C, H, W)
    - targets: (N, H, W) with int64 class indices
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        self.eps = float(eps)

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 4:
            raise ValueError(f"logits must be (N,C,H,W), got {tuple(logits.shape)}")
        if targets.dim() != 3:
            raise ValueError(f"targets must be (N,H,W), got {tuple(targets.shape)}")

        n, c, h, w = logits.shape
        if targets.shape != (n, h, w):
            raise ValueError(f"targets shape must be (N,H,W)={(n,h,w)}, got {tuple(targets.shape)}")

        # flatten
        logits = logits.permute(0, 2, 3, 1).reshape(-1, c)    # (NHW, C)
        targets = targets.reshape(-1)                         # (NHW,)

        valid = targets != self.ignore_index
        if valid.sum() == 0:
            # no valid pixels
            return logits.sum() * 0.0

        logits_v = logits[valid]
        targets_v = targets[valid]

        log_probs = F.log_softmax(logits_v, dim=1)           # (M, C)
        probs = log_probs.exp()                               # (M, C)

        # pick p_t / log(p_t)
        idx = targets_v.unsqueeze(1)                          # (M,1)
        log_pt = log_probs.gather(1, idx).squeeze(1)          # (M,)
        pt = probs.gather(1, idx).squeeze(1).clamp(self.eps, 1.0 - self.eps)

        focal_factor = (1.0 - pt).pow(self.gamma)            # (M,)

        if self.alpha is None:
            alpha_t = 1.0
        else:
            # alpha per class: (C,)
            if self.alpha.numel() != c:
                raise ValueError(f"alpha must have shape (C,) with C={c}, got {tuple(self.alpha.shape)}")
            alpha_t = self.alpha.gather(0, targets_v)         # (M,)

        loss = -alpha_t * focal_factor * log_pt               # (M,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            # return per-pixel loss in original shape (N,H,W), fill ignored with 0
            out = torch.zeros(n * h * w, device=loss.device, dtype=loss.dtype)
            out[valid] = loss
            return out.view(n, h, w)
        raise ValueError(f"Invalid reduction: {self.reduction}")
