from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.transforms import (
    resize_pair,
    random_scale_pair,
    maybe_hflip_pair,
    pil_to_tensor,
    normalize_img,
)
from src.utils.Id2Mask import color_mask_to_id

RGB = Tuple[int, int, int]


def _build_default_train_preprocess() -> Dict[str, Any]:
    return {
        "multi_scale_range": (1.0, 1.0),
        "random_crop_size": None,
        "hflip_prob": 0.0,
    }


def _build_default_eval_preprocess() -> Dict[str, Any]:
    return {}


class CamVidFolderDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        color2id: Dict[RGB, int],
        resize_w: int,
        resize_h: int,
        ignore_index: int,
        training: bool,
        label_lut: Optional[np.ndarray] = None,  # shape (256,), old_id -> new_id or ignore
        train_preprocess: Optional[Dict[str, Any]] = None,
        eval_preprocess: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"

        self.root = Path(root)
        self.split = split
        self.color2id = color2id
        self.resize_w = int(resize_w)
        self.resize_h = int(resize_h)
        self.ignore_index = int(ignore_index)
        self.training = bool(training)

        train_cfg = _build_default_train_preprocess()
        if train_preprocess is not None:
            train_cfg.update(train_preprocess)

        eval_cfg = _build_default_eval_preprocess()
        if eval_preprocess is not None:
            eval_cfg.update(eval_preprocess)

        self.train_preprocess = train_cfg
        self.eval_preprocess = eval_cfg

        self.hflip_prob = float(self.train_preprocess["hflip_prob"])
        self.multi_scale_range = (
            float(self.train_preprocess["multi_scale_range"][0]),
            float(self.train_preprocess["multi_scale_range"][1]),
        )
        self.random_crop_size = self.train_preprocess["random_crop_size"]


        if label_lut is not None:
            lut = np.asarray(label_lut)
            if lut.shape != (256,):
                raise ValueError(f"label_lut must have shape (256,), got {lut.shape}")
            # 用 uint8 存，后面映射时保持 0..K-1 或 255
            self.label_lut = lut.astype(np.uint8, copy=False)
        else:
            self.label_lut = None

        self.train_images_dir = self.root / "train"
        self.train_masks_dir = self.root / "train_labels"
        self.val_images_dir = self.root / "val"
        self.val_masks_dir = self.root / "val_labels"
        self.test_images_dir = self.root / "test"
        self.test_masks_dir = self.root / "test_labels"

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

        # 3) 任意扩展名
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
            if self.multi_scale_range != (1.0, 1.0):
                img, mask_rgb = random_scale_pair(img, mask_rgb, self.multi_scale_range)
            img, mask_rgb = maybe_hflip_pair(img, mask_rgb, self.hflip_prob)

        mask_old = color_mask_to_id(mask_rgb, self.color2id, self.ignore_index)

        if self.label_lut is not None:
            mask_new = self.label_lut[mask_old]
        else:
            mask_new = mask_old

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

        mask_t = torch.from_numpy(mask_new.astype(np.int64))  # long for CE

        return img_t, mask_t, img_path.name
