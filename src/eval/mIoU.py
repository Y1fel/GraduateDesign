import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def update_confusion_matrix(
    conf: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    debug_stats: dict[str, int] | None = None,
) -> None:
    pred = pred.view(-1)
    target = target.view(-1)

    m = target != ignore_index
    target_in_range = (target >= 0) & (target < num_classes)
    pred_in_range = (pred >= 0) & (pred < num_classes)
    valid = m & target_in_range & pred_in_range

    if debug_stats is not None:
        debug_stats["total_pixels"] = debug_stats.get("total_pixels", 0) + int(target.numel())
        debug_stats["ignored_pixels"] = debug_stats.get("ignored_pixels", 0) + int((~m).sum().item())
        debug_stats["target_out_of_range"] = debug_stats.get("target_out_of_range", 0) + int((m & ~target_in_range).sum().item())
        debug_stats["pred_out_of_range"] = debug_stats.get("pred_out_of_range", 0) + int((m & ~pred_in_range).sum().item())
        debug_stats["kept_pixels"] = debug_stats.get("kept_pixels", 0) + int(valid.sum().item())

    pred = pred[valid]
    target = target[valid]
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
    scales: list[float] | tuple[float, ...] | None = None,
    flip: bool = False,
) -> dict:
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    bf_scores = []
    bf_precisions = []
    bf_recalls = []
    trimap_ious = []

    eval_scales = list(scales) if scales else [1.0]

    for imgs, masks, _names in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        base_h, base_w = imgs.shape[-2], imgs.shape[-1]
        ms_logits: list[torch.Tensor] = []

        for scale in eval_scales:
            if float(scale) <= 0:
                raise ValueError(f"Scale must be positive, got {scale}")

            if float(scale) == 1.0:
                scaled_imgs = imgs
            else:
                scaled_h = max(1, int(round(base_h * float(scale))))
                scaled_w = max(1, int(round(base_w * float(scale))))
                scaled_imgs = F.interpolate(
                    imgs,
                    size=(scaled_h, scaled_w),
                    mode="bilinear",
                    align_corners=False,
                )

            logits = model(scaled_imgs)

            if flip:
                flipped_imgs = torch.flip(scaled_imgs, dims=[3])
                flipped_logits = model(flipped_imgs)
                flipped_logits = torch.flip(flipped_logits, dims=[3])
                logits = 0.5 * (logits + flipped_logits)

            if logits.shape[-2:] != (base_h, base_w):
                logits = F.interpolate(
                    logits,
                    size=(base_h, base_w),
                    mode="bilinear",
                    align_corners=False,
                )
            ms_logits.append(logits)

        logits = torch.stack(ms_logits, dim=0).mean(dim=0)
        pred = torch.argmax(logits, dim=1)
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
