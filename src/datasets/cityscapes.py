from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.cityscapes_labels import CITYSCAPES_34_TO_19
from src.datasets.transforms import (
    maybe_hflip_pair,
    pil_to_tensor,
    random_scale_pair,
    normalize_img,
    maybe_color_jitter,
    maybe_gaussian_blur,
)


class CityscapesDataset(Dataset):
    """Cityscapes loader based on official folder layout.

    Expected root structure:
      root/
        leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png
        gtFine/{train,val}/<city>/*_gtFine_labelIds.png

    Note:
      - split=test has no public gt labels, so training/eval should use train/val.
      - masks are single-channel labelId maps in [0,33] with ignored pixels usually 255.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        ignore_index: int,
        training: bool,
        hflip_prob: float = 0.0,
        multi_scale_range: Tuple[float, float] = (1.0, 1.0),
        random_crop_size: Optional[Tuple[int, int]] = None,
        crop_retry: int = 1,
        crop_max_class_ratio: float = 1.0,
        remap_to_19: bool = True,
        color_jitter_prob: float = 0.0,
        color_jitter_brightness: float = 0.0,
        color_jitter_contrast: float = 0.0,
        color_jitter_saturation: float = 0.0,
        gaussian_blur_prob: float = 0.0,
        gaussian_blur_radius_range: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"

        self.root = Path(root)
        self.split = split
        self.ignore_index = int(ignore_index)
        self.training = bool(training)

        self.hflip_prob = float(hflip_prob)
        self.multi_scale_range = (float(multi_scale_range[0]), float(multi_scale_range[1]))
        self.random_crop_size = random_crop_size
        self.crop_retry = max(1, int(crop_retry))
        self.crop_max_class_ratio = float(crop_max_class_ratio)
        self.remap_to_19 = bool(remap_to_19)

        self.color_jitter_prob = float(color_jitter_prob)
        self.color_jitter_brightness = float(color_jitter_brightness)
        self.color_jitter_contrast = float(color_jitter_contrast)
        self.color_jitter_saturation = float(color_jitter_saturation)
        self.gaussian_blur_prob = float(gaussian_blur_prob)
        self.gaussian_blur_radius_range = (
            float(gaussian_blur_radius_range[0]),
            float(gaussian_blur_radius_range[1]),
        )
        self._label_id_to_train_id = np.asarray(CITYSCAPES_34_TO_19, dtype=np.uint8)

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

        if self.training:
            if self.multi_scale_range != (1.0, 1.0):
                img, mask_id = random_scale_pair(img, mask_id, self.multi_scale_range)
            img, mask_id = maybe_hflip_pair(img, mask_id, self.hflip_prob)
            img = maybe_color_jitter(
                img,
                prob=self.color_jitter_prob,
                brightness=self.color_jitter_brightness,
                contrast=self.color_jitter_contrast,
                saturation=self.color_jitter_saturation,
            )
            img = maybe_gaussian_blur(
                img,
                prob=self.gaussian_blur_prob,
                radius_range=self.gaussian_blur_radius_range,
            )

        mask_new = np.asarray(mask_id, dtype=np.uint8)
        if self.remap_to_19:
            valid = mask_new <= 33
            remapped = np.full(mask_new.shape, fill_value=self.ignore_index, dtype=np.uint8)
            remapped[valid] = self._label_id_to_train_id[mask_new[valid]]
            mask_new = remapped

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

            crop_img = None
            crop_mask = None
            for _ in range(self.crop_retry):
                y1 = np.random.randint(0, h - crop_h + 1)
                x1 = np.random.randint(0, w - crop_w + 1)
                candidate_img = img_np[y1:y1 + crop_h, x1:x1 + crop_w]
                candidate_mask = mask_new[y1:y1 + crop_h, x1:x1 + crop_w]

                valid = candidate_mask != self.ignore_index
                if not np.any(valid):
                    crop_img, crop_mask = candidate_img, candidate_mask
                    break

                _, counts = np.unique(candidate_mask[valid], return_counts=True)
                max_ratio = float(counts.max() / counts.sum())
                crop_img, crop_mask = candidate_img, candidate_mask
                if max_ratio < self.crop_max_class_ratio:
                    break

            img_np = crop_img
            mask_new = crop_mask
            img = Image.fromarray(img_np, mode="RGB")

        img_t = pil_to_tensor(img)
        img_t = normalize_img(img_t)
        mask_t = torch.from_numpy(mask_new.astype(np.int64))

        rel_name = str(img_path.relative_to(self.images_root))
        return img_t, mask_t, rel_name
