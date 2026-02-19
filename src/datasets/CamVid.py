from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from src.datasets.transforms import (
    resize_pair,
    random_scale_pair,
    maybe_hflip_pair,
    pil_to_tensor,
    normalize_img,
    photometric_augment,
)
from src.utils.Id2Mask import color_mask_to_id

RGB = Tuple[int, int, int]


class CamVidFolderDataset(Dataset):
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
        label_lut: Optional[np.ndarray] = None,  # shape (256,), old_id -> new_id or ignore
        photo_aug_prob: float = 0.0,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
        saturation_jitter: float = 0.0,
        gamma_range: Tuple[float, float] = (1.0, 1.0),
        photo_op_prob: float = 0.5,
        blur_prob: float = 0.0,
        blur_radius_range: Tuple[float, float] = (0.0, 0.0),
        jpeg_prob: float = 0.0,
        jpeg_quality_range: Tuple[int, int] = (95, 100),
        multi_scale_range: Tuple[float, float] = (1.0, 1.0),
        random_crop_size: Optional[Tuple[int, int]] = None,
        auto_contrast: bool = False,
        auto_contrast_cutoff: float = 1.0,
        ignore_filename_prefixes: Optional[Sequence[str]] = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"

        self.root = Path(root)
        self.split = split
        self.color2id = color2id
        self.resize_w = int(resize_w)
        self.resize_h = int(resize_h)
        self.hflip_prob = float(hflip_prob)
        self.ignore_index = int(ignore_index)
        self.training = bool(training)

        self.photo_aug_prob = float(photo_aug_prob)
        self.photo_aug_prob_current = float(photo_aug_prob)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)
        self.saturation_jitter = float(saturation_jitter)
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))
        self.photo_op_prob = float(photo_op_prob)
        self.blur_prob = float(blur_prob)
        self.blur_radius_range = (float(blur_radius_range[0]), float(blur_radius_range[1]))
        self.jpeg_prob = float(jpeg_prob)
        self.jpeg_quality_range = (int(jpeg_quality_range[0]), int(jpeg_quality_range[1]))
        self.multi_scale_range = (float(multi_scale_range[0]), float(multi_scale_range[1]))
        self.random_crop_size = random_crop_size
        self.auto_contrast = bool(auto_contrast)
        self.auto_contrast_cutoff = float(auto_contrast_cutoff)
        self.ignore_filename_prefixes = tuple(ignore_filename_prefixes or ())

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

        if self.ignore_filename_prefixes:
            before = len(self.img_paths)
            self.img_paths = [
                p for p in self.img_paths
                if not any(p.name.startswith(prefix) for prefix in self.ignore_filename_prefixes)
            ]
            ignored = before - len(self.img_paths)
            if ignored > 0:
                print(f"[DATA] split={self.split}: ignored {ignored} files by prefix filter {self.ignore_filename_prefixes}")

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

    def set_photo_aug_scale(self, scale: float) -> None:
        s = max(0.0, min(1.0, float(scale)))
        self.photo_aug_prob_current = self.photo_aug_prob * s

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

        if self.auto_contrast:
            img = ImageOps.autocontrast(img, cutoff=self.auto_contrast_cutoff)

        if self.training:
            img = photometric_augment(
                img,
                prob=self.photo_aug_prob_current,
                brightness=self.brightness_jitter,
                contrast=self.contrast_jitter,
                saturation=self.saturation_jitter,
                gamma_range=self.gamma_range,
                op_prob=self.photo_op_prob,
                blur_prob=self.blur_prob,
                blur_radius_range=self.blur_radius_range,
                jpeg_prob=self.jpeg_prob,
                jpeg_quality_range=self.jpeg_quality_range,
            )

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
