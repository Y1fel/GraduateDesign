from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from src.utils.Id2Mask import id_mask_to_color

RGB = Tuple[int, int, int]


@torch.no_grad()
def save_predictions_triplet(
    model,
    loader,
    device: torch.device,
    out_dir: Path,
    id2color: List[RGB],
    ignore_index: int,
    epoch: int,
    max_items: int = 8,
) -> None:
    model.eval()
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for imgs, masks, names in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        pred = torch.argmax(logits, dim=1)

        imgs_cpu = imgs.detach().cpu()
        masks_cpu = masks.detach().cpu().numpy()
        pred_cpu = pred.detach().cpu().numpy()

        for i in range(imgs_cpu.shape[0]):
            if saved >= max_items:
                return

            # de-normalize (approx)
            img = imgs_cpu[i]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img * std + mean).clamp(0, 1)
            img_np = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

            gt_id = masks_cpu[i].astype(np.uint8)
            pr_id = pred_cpu[i].astype(np.uint8)

            gt_rgb = id_mask_to_color(gt_id, id2color, ignore_index)
            pr_rgb = id_mask_to_color(pr_id, id2color, ignore_index)

            triplet = np.concatenate([img_np, gt_rgb, pr_rgb], axis=1)
            stem = Path(names[i]).stem

            Image.fromarray(triplet).save(epoch_dir / f"{stem}_triplet.png")
            Image.fromarray(pr_rgb).save(epoch_dir / f"{stem}_pred_color.png")
            saved += 1
