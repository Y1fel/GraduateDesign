import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable


@torch.no_grad()
def update_confusion_matrix(
    conf: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> None:
    pred = pred.view(-1)
    target = target.view(-1)

    m = target != ignore_index
    pred = pred[m]
    target = target[m]
    if pred.numel() == 0:
        return

    k = (target * num_classes + pred).to(torch.int64)
    bins = torch.bincount(k, minlength=num_classes * num_classes)
    conf += bins.view(num_classes, num_classes)


@torch.no_grad()
def _boundary_map(mask: torch.Tensor, ignore_index: int) -> torch.Tensor:
    valid = mask != ignore_index
    t = mask.clone()
    t[~valid] = 0

    b = torch.zeros_like(valid)
    diff_h = (t[:, 1:, :] != t[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    b[:, 1:, :] |= diff_h
    b[:, :-1, :] |= diff_h

    diff_w = (t[:, :, 1:] != t[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    b[:, :, 1:] |= diff_w
    b[:, :, :-1] |= diff_w

    return b & valid


@torch.no_grad()
def _boundary_fscore(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    dilation: int,
) -> tuple[float, float, float]:
    pb = _boundary_map(pred, ignore_index)
    tb = _boundary_map(target, ignore_index)

    if dilation > 0:
        k = 2 * dilation + 1
        pb_d = F.max_pool2d(pb.float().unsqueeze(1), kernel_size=k, stride=1, padding=dilation) > 0
        tb_d = F.max_pool2d(tb.float().unsqueeze(1), kernel_size=k, stride=1, padding=dilation) > 0
        matched_p = (pb.unsqueeze(1) & tb_d).sum().item()
        matched_t = (tb.unsqueeze(1) & pb_d).sum().item()
    else:
        matched_p = (pb & tb).sum().item()
        matched_t = matched_p

    pred_count = pb.sum().item()
    target_count = tb.sum().item()
    precision = matched_p / pred_count if pred_count > 0 else 1.0
    recall = matched_t / target_count if target_count > 0 else 1.0
    if precision + recall <= 0:
        return 0.0, precision, recall
    return 2.0 * precision * recall / (precision + recall), precision, recall


@torch.no_grad()
def _trimap_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    trimap_width: int,
) -> float:
    valid = target != ignore_index
    tb = _boundary_map(target, ignore_index)

    if trimap_width > 0:
        k = 2 * trimap_width + 1
        band = F.max_pool2d(tb.float().unsqueeze(1), kernel_size=k, stride=1, padding=trimap_width).squeeze(1) > 0
    else:
        band = tb

    region = band & valid
    union = region
    if union.sum().item() == 0:
        return float("nan")

    inter = (pred == target) & region
    return inter.sum().item() / max(union.sum().item(), 1)


@torch.no_grad()
def compute_segmentation_metrics(
    model,
    loader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    boundary_dilation: int = 2,
    trimap_width: int = 3,
    postprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict:
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    bf_scores = []
    bf_precisions = []
    bf_recalls = []
    trimap_ious = []

    for imgs, masks, _names in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        pred = torch.argmax(logits, dim=1)
        if postprocess_fn is not None:
            pred = postprocess_fn(pred)
        update_confusion_matrix(conf, pred, masks, num_classes, ignore_index)

        bf, bp, br = _boundary_fscore(pred, masks, ignore_index=ignore_index, dilation=boundary_dilation)
        bf_scores.append(bf)
        bf_precisions.append(bp)
        bf_recalls.append(br)
        trimap_ious.append(_trimap_iou(pred, masks, ignore_index=ignore_index, trimap_width=trimap_width))

    c = conf.detach().cpu().numpy().astype(np.float64)
    tp = np.diag(c)
    fp = c.sum(axis=0) - tp
    fn = c.sum(axis=1) - tp
    denom = tp + fp + fn

    iou = np.full(tp.shape, np.nan, dtype=np.float64)
    iou_valid = denom > 0
    iou[iou_valid] = tp[iou_valid] / denom[iou_valid]

    recall_denom = tp + fn
    recall = np.full(tp.shape, np.nan, dtype=np.float64)
    rec_valid = recall_denom > 0
    recall[rec_valid] = tp[rec_valid] / recall_denom[rec_valid]

    precision_denom = tp + fp
    precision = np.full(tp.shape, np.nan, dtype=np.float64)
    pre_valid = precision_denom > 0
    precision[pre_valid] = tp[pre_valid] / precision_denom[pre_valid]

    miou = float(np.nanmean(iou)) if np.any(iou_valid) else float("nan")
    return {
        "miou": miou,
        "iou_per_class": iou,
        "recall_per_class": recall,
        "precision_per_class": precision,
        "boundary_fscore": float(np.nanmean(np.array(bf_scores, dtype=np.float64))) if bf_scores else float("nan"),
        "boundary_precision": float(np.nanmean(np.array(bf_precisions, dtype=np.float64))) if bf_precisions else float("nan"),
        "boundary_recall": float(np.nanmean(np.array(bf_recalls, dtype=np.float64))) if bf_recalls else float("nan"),
        "trimap_iou": float(np.nanmean(np.array(trimap_ious, dtype=np.float64))) if trimap_ious else float("nan"),
    }

