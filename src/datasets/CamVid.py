from pathlib import Path
from typing import Dict, Tuple, Optional, Any

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
    low_light_preprocess,
)
from src.utils.Id2Mask import color_mask_to_id

RGB = Tuple[int, int, int]


def _build_default_train_preprocess() -> Dict[str, Any]:
    return {
        "auto_contrast": False,
        "auto_contrast_cutoff": 1.0,
        "low_light_preprocess_enable": False,
        "low_light_gamma": 1.0,
        "low_light_brightness_gain": 1.0,
        "photo_aug_prob": 0.0,
        "brightness_jitter": 0.0,
        "contrast_jitter": 0.0,
        "saturation_jitter": 0.0,
        "gamma_range": (1.0, 1.0),
        "photo_op_prob": 0.5,
        "blur_prob": 0.0,
        "blur_radius_range": (0.0, 0.0),
        "jpeg_prob": 0.0,
        "jpeg_quality_range": (95, 100),
        "multi_scale_range": (1.0, 1.0),
        "random_crop_size": None,
        "hflip_prob": 0.0,
    }


def _build_default_eval_preprocess() -> Dict[str, Any]:
    return {
        "auto_contrast": False,
        "auto_contrast_cutoff": 1.0,
        "low_light_preprocess_enable": False,
        "low_light_gamma": 1.0,
        "low_light_brightness_gain": 1.0,
    }


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
        self.photo_aug_prob = float(self.train_preprocess["photo_aug_prob"])
        self.photo_aug_prob_current = float(self.train_preprocess["photo_aug_prob"])
        self.brightness_jitter = float(self.train_preprocess["brightness_jitter"])
        self.contrast_jitter = float(self.train_preprocess["contrast_jitter"])
        self.saturation_jitter = float(self.train_preprocess["saturation_jitter"])
        self.gamma_range = (
            float(self.train_preprocess["gamma_range"][0]),
            float(self.train_preprocess["gamma_range"][1]),
        )
        self.photo_op_prob = float(self.train_preprocess["photo_op_prob"])
        self.blur_prob = float(self.train_preprocess["blur_prob"])
        self.blur_radius_range = (
            float(self.train_preprocess["blur_radius_range"][0]),
            float(self.train_preprocess["blur_radius_range"][1]),
        )
        self.jpeg_prob = float(self.train_preprocess["jpeg_prob"])
        self.jpeg_quality_range = (
            int(self.train_preprocess["jpeg_quality_range"][0]),
            int(self.train_preprocess["jpeg_quality_range"][1]),
        )
        self.multi_scale_range = (
            float(self.train_preprocess["multi_scale_range"][0]),
            float(self.train_preprocess["multi_scale_range"][1]),
        )
        self.random_crop_size = self.train_preprocess["random_crop_size"]

        active_eval = self.eval_preprocess
        self.auto_contrast = bool(active_eval["auto_contrast"])
        self.auto_contrast_cutoff = float(active_eval["auto_contrast_cutoff"])
        self.low_light_preprocess_enable = bool(active_eval["low_light_preprocess_enable"])
        self.low_light_gamma = float(active_eval["low_light_gamma"])
        self.low_light_brightness_gain = float(active_eval["low_light_brightness_gain"])

        if self.training:
            active_train = self.train_preprocess
            self.auto_contrast = bool(active_train["auto_contrast"])
            self.auto_contrast_cutoff = float(active_train["auto_contrast_cutoff"])
            self.low_light_preprocess_enable = bool(active_train["low_light_preprocess_enable"])
            self.low_light_gamma = float(active_train["low_light_gamma"])
            self.low_light_brightness_gain = float(active_train["low_light_brightness_gain"])

        self._aug_stats = {
            "samples_seen": 0,
            "photometric_applied": 0,
            "blur_applied": 0,
            "jpeg_applied": 0,
        }

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

    def set_photo_aug_scale(self, scale: float) -> None:
        s = max(0.0, min(1.0, float(scale)))
        self.photo_aug_prob_current = self.photo_aug_prob * s

    def reset_aug_stats(self) -> None:
        for key in self._aug_stats:
            self._aug_stats[key] = 0

    def consume_aug_stats(self) -> dict[str, int]:
        out = {k: int(v) for k, v in self._aug_stats.items()}
        self.reset_aug_stats()
        return out

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

        if self.low_light_preprocess_enable:
            img = low_light_preprocess(
                img,
                gamma=self.low_light_gamma,
                brightness_gain=self.low_light_brightness_gain,
            )

        if self.auto_contrast:
            img = ImageOps.autocontrast(img, cutoff=self.auto_contrast_cutoff)

        if self.training:
            img, aug_stats = photometric_augment(
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
                return_stats=True,
            )
            self._aug_stats["samples_seen"] += 1
            if aug_stats["photometric_applied"]:
                self._aug_stats["photometric_applied"] += 1
            if aug_stats["blur_applied"]:
                self._aug_stats["blur_applied"] += 1
            if aug_stats["jpeg_applied"]:
                self._aug_stats["jpeg_applied"] += 1

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
