from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.transforms import (
    maybe_hflip_pair,
    pil_to_tensor,
    random_scale_pair,
    resize_pair,
    normalize_img,
)


class CityscapesDataset(Dataset):
    """Cityscapes loader based on official folder layout.

    Expected root structure:
      root/
        leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png
        gtFine/{train,val}/<city>/*_gtFine_labelTrainIds.png

    Note:
      - split=test has no public gt labels, so training/eval should use train/val.
      - masks are single-channel trainId maps in [0,18] with ignored pixels usually 255.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        resize_w: int,
        resize_h: int,
        ignore_index: int,
        training: bool,
        hflip_prob: float = 0.0,
        multi_scale_range: Tuple[float, float] = (1.0, 1.0),
        random_crop_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"

        self.root = Path(root)
        self.split = split
        self.resize_w = int(resize_w)
        self.resize_h = int(resize_h)
        self.ignore_index = int(ignore_index)
        self.training = bool(training)

        self.hflip_prob = float(hflip_prob)
        self.multi_scale_range = (float(multi_scale_range[0]), float(multi_scale_range[1]))
        self.random_crop_size = random_crop_size

        self.images_root = self.root / "leftImg8bit" / split
        self.labels_root = self.root / "gtFine" / split

        if not self.images_root.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_root}")
        if split != "test" and not self.labels_root.exists():
            raise FileNotFoundError(f"Labels dir not found: {self.labels_root}")

        self.img_paths = sorted(self.images_root.glob("*/*_leftImg8bit.png"))
        if not self.img_paths:
            raise RuntimeError(f"No Cityscapes images found in {self.images_root}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def _resolve_mask(self, img_path: Path) -> Path:
        if self.split == "test":
            raise RuntimeError("Cityscapes test split has no public labels.")

        city = img_path.parent.name
        stem = img_path.name.replace("_leftImg8bit.png", "")
        mask_path = self.labels_root / city / f"{stem}_gtFine_labelIds.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path.name}: {mask_path}")
        return mask_path

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]

        img = Image.open(img_path).convert("RGB")
        if self.split == "test":
            mask_id = Image.fromarray(
                np.full((img.height, img.width), fill_value=self.ignore_index, dtype=np.uint8),
                mode="L",
            )
        else:
            mask_path = self._resolve_mask(img_path)
            mask_id = Image.open(mask_path).convert("L")

        img, mask_id = resize_pair(img, mask_id, (self.resize_w, self.resize_h))

        if self.training:
            if self.multi_scale_range != (1.0, 1.0):
                img, mask_id = random_scale_pair(img, mask_id, self.multi_scale_range)
            img, mask_id = maybe_hflip_pair(img, mask_id, self.hflip_prob)

        mask_new = np.asarray(mask_id, dtype=np.uint8)

        if self.training and self.random_crop_size is not None:
            crop_w, crop_h = int(self.random_crop_size[0]), int(self.random_crop_size[1])
            img_np = np.asarray(img, dtype=np.uint8)
            h, w = img_np.shape[:2]

            if h < crop_h or w < crop_w:
                pad_h = max(0, crop_h - h)
                pad_w = max(0, crop_w - w)
                img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
                mask_new = np.pad(
                    mask_new,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=self.ignore_index,
                )
                h, w = img_np.shape[:2]

            y1 = np.random.randint(0, h - crop_h + 1)
            x1 = np.random.randint(0, w - crop_w + 1)
            img_np = img_np[y1:y1 + crop_h, x1:x1 + crop_w]
            mask_new = mask_new[y1:y1 + crop_h, x1:x1 + crop_w]
            img = Image.fromarray(img_np, mode="RGB")

        img_t = pil_to_tensor(img)
        img_t = normalize_img(img_t)
        mask_t = torch.from_numpy(mask_new.astype(np.int64))

        rel_name = str(img_path.relative_to(self.images_root))
        return img_t, mask_t, rel_name

