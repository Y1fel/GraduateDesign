from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.transforms import resize_pair, maybe_hflip_pair, pil_to_tensor, normalize_img
from src.utils.Id2Mask import color_mask_to_id

RGB = Tuple[int, int, int]


class CamVidFolderDataset(Dataset):
    """
    期望目录：
      root/
        train/         train_labels/
        val/           val_labels/
        test/          test_labels/
        class_dict.csv
    """
    def __init__(
        self,
        root: Path,
        split: str,
        color2id: Dict[RGB, int],
        resize_w: int,
        resize_h: int,
        hflip_prob: float,
        ignore_index: int,
        training: bool,
    ) -> None:
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"

        self.root = root
        self.split = split
        self.color2id = color2id
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.hflip_prob = hflip_prob
        self.ignore_index = ignore_index
        self.training = training

        # ✅ 保留你喜欢的写法（不放 config 展开）
        self.train_images_dir = root / "train"
        self.train_masks_dir = root / "train_labels"
        self.val_images_dir = root / "val"
        self.val_masks_dir = root / "val_labels"
        self.test_images_dir = root / "test"
        self.test_masks_dir = root / "test_labels"

        if split == "train":
            self.images_dir, self.masks_dir = self.train_images_dir, self.train_masks_dir
        elif split == "val":
            self.images_dir, self.masks_dir = self.val_images_dir, self.val_masks_dir
        else:
            self.images_dir, self.masks_dir = self.test_images_dir, self.test_masks_dir

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks dir not found: {self.masks_dir}")

        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.img_paths = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in exts])
        if not self.img_paths:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def _resolve_mask(self, img_path: Path) -> Path:
        # 1) 同名
        p1 = self.masks_dir / img_path.name
        if p1.exists():
            return p1

        # 2) 常见命名：xxx_L.png
        p2 = self.masks_dir / f"{img_path.stem}_L{img_path.suffix}"
        if p2.exists():
            return p2

        # 3) 兜底：同 stem 任意扩展名
        cand = list(self.masks_dir.glob(f"{img_path.stem}.*"))
        if cand:
            return cand[0]

        raise FileNotFoundError(f"Mask not found for {img_path.name} in {self.masks_dir}")

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        mask_path = self._resolve_mask(img_path)

        img = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path).convert("RGB")

        img, mask_rgb = resize_pair(img, mask_rgb, (self.resize_w, self.resize_h))

        if self.training:
            img, mask_rgb = maybe_hflip_pair(img, mask_rgb, self.hflip_prob)

        img_t = pil_to_tensor(img)
        img_t = normalize_img(img_t)

        mask_id = color_mask_to_id(mask_rgb, self.color2id, self.ignore_index)  # (H,W) uint8
        mask_t = torch.from_numpy(mask_id.astype(np.int64))  # (H,W) long

        return img_t, mask_t, img_path.name
