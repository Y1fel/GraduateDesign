from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.focalloss import FocalLoss


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    c = probas.size(1)
    losses = []
    valid = labels != ignore_index
    if valid.sum() == 0:
        return probas.new_tensor(0.0)
    probas = probas[valid]
    labels = labels[valid]
    for class_id in range(c):
        fg = (labels == class_id).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, class_id]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = _lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if not losses:
        return probas.new_tensor(0.0)
    return torch.stack(losses).mean()


def _safe_odd_kernel_size(kernel_size: int) -> int:
    k = max(1, int(kernel_size))
    # even kernel + same padding 会引发输出尺寸偏移，强制转为奇数避免边界图与标签尺寸冲突。
    if k % 2 == 0:
        k += 1
    return k


def _morphological_gradient(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    k = _safe_odd_kernel_size(kernel_size)
    pad = k // 2
    mask = mask.float().unsqueeze(1)
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0).squeeze(1)


def _label_boundary_map(labels: torch.Tensor, valid: torch.Tensor, boundary_width: int) -> torch.Tensor:
    # 基于 4 邻域类别变化提取语义边界，避免把 class id 当作连续灰度值做形态学造成的伪边界。
    edges = torch.zeros_like(labels, dtype=torch.bool)

    # 水平方向相邻像素类别变化（两侧都必须是有效标签）
    valid_h = valid[:, :, :-1] & valid[:, :, 1:]
    diff_h = (labels[:, :, :-1] != labels[:, :, 1:]) & valid_h
    edges[:, :, :-1] |= diff_h
    edges[:, :, 1:] |= diff_h

    # 垂直方向相邻像素类别变化（两侧都必须是有效标签）
    valid_v = valid[:, :-1, :] & valid[:, 1:, :]
    diff_v = (labels[:, :-1, :] != labels[:, 1:, :]) & valid_v
    edges[:, :-1, :] |= diff_v
    edges[:, 1:, :] |= diff_v

    edge_map = edges.float()
    k = _safe_odd_kernel_size(boundary_width)
    if k > 1:
        # 与 boundary_width 语义一致：将边界监督带适度膨胀。
        pad = k // 2
        edge_map = F.max_pool2d(edge_map.unsqueeze(1), kernel_size=k, stride=1, padding=pad).squeeze(1)
    return edge_map


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


class CompositeSegLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        ohem_ratio: float = 0.25,
        ohem_weight: float = 0.5,
        lovasz_weight: float = 0.3,
        boundary_weight: float = 0.2,
        boundary_width: int = 3,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.ohem_ratio = float(ohem_ratio)
        self.ohem_weight = float(ohem_weight)
        self.lovasz_weight = float(lovasz_weight)
        self.boundary_weight = float(boundary_weight)
        self.boundary_width = int(boundary_width)
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

    def _lovasz_softmax(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probas = F.softmax(logits, dim=1).permute(0, 2, 3, 1).reshape(-1, logits.size(1))
        labels = targets.reshape(-1)
        return _lovasz_softmax_flat(probas, labels, ignore_index=self.ignore_index)

    def _boundary_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid = targets != self.ignore_index

        pred_edges = _morphological_gradient(probs.max(dim=1).values, kernel_size=self.boundary_width)
        gt_edges = _label_boundary_map(targets, valid, boundary_width=self.boundary_width)
        edge_mask = gt_edges * valid.float()
        if edge_mask.sum() == 0:
            return logits.new_tensor(0.0)
        return ((pred_edges - gt_edges).abs() * edge_mask).sum() / edge_mask.sum().clamp_min(1.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, return_components: bool = False):
        ohem_ce = self._ohem_ce(logits, targets)
        lovasz = self._lovasz_softmax(logits, targets)
        boundary = self._boundary_loss(logits, targets)
        total = self.ohem_weight * ohem_ce + self.lovasz_weight * lovasz + self.boundary_weight * boundary
        if return_components:
            return total, {
                "ohem_ce": float(ohem_ce.detach().item()),
                "lovasz": float(lovasz.detach().item()),
                "boundary": float(boundary.detach().item()),
                "total": float(total.detach().item()),
            }
        return total
