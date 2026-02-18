import numpy as np
import torch


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
def compute_segmentation_metrics(
    model,
    loader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> dict:
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for imgs, masks, _names in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        pred = torch.argmax(logits, dim=1)
        update_confusion_matrix(conf, pred, masks, num_classes, ignore_index)

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
    }


@torch.no_grad()
def compute_miou(
    model,
    loader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> float:
    return float(
        compute_segmentation_metrics(
            model=model,
            loader=loader,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )["miou"]
    )
