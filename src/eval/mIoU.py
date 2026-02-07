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
def compute_miou(
    model,
    loader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> float:
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
    valid = denom > 0
    iou[valid] = tp[valid] / denom[valid]
    return float(np.nanmean(iou)) if np.any(valid) else float("nan")
