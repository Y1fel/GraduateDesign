from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.comma10k_labels import COMMA10K_CLASS_NAMES, COMMA10K_ID2COLOR
from src.datasets.meta import SegmentationDatasetMeta
from src.datasets.splits import match_split_entry, read_split_entries
from src.datasets.transforms import (
    color_jitter,
    gaussian_blur,
    hflip_pair,
    normalize_img,
    pil_to_tensor,
    random_scale_pair,
)
from src.utils.Id2Mask import color_mask_to_id


def _stem_base(stem: str) -> str:
    base = stem
    if "_" in base:
        prefix, suffix = base.rsplit("_", 1)
        if suffix.isalpha() and len(suffix) == 1:
            base = prefix
    return base


class Comma10KDataset(Dataset):
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

        self.color_jitter_prob = float(color_jitter_prob)
        self.color_jitter_brightness = float(color_jitter_brightness)
        self.color_jitter_contrast = float(color_jitter_contrast)
        self.color_jitter_saturation = float(color_jitter_saturation)
        self.gaussian_blur_prob = float(gaussian_blur_prob)
        self.gaussian_blur_radius_range = (
            float(gaussian_blur_radius_range[0]),
            float(gaussian_blur_radius_range[1]),
        )

        self.meta = SegmentationDatasetMeta(
            dataset_name="comma10k",
            num_classes=len(COMMA10K_CLASS_NAMES),
            class_names=tuple(COMMA10K_CLASS_NAMES),
            id2color=tuple(COMMA10K_ID2COLOR),
        )
        self._color2id = {tuple(color): idx for idx, color in enumerate(COMMA10K_ID2COLOR)}

        self._split_entries = read_split_entries(self.root, self.split)
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No comma10k samples found for split={split} under {self.root}")
        self.img_paths = [img for img, _mask in self.samples]

    def _iter_dir_pairs(self) -> list[tuple[Path, Optional[Path]]]:
        return [
            (self.root / "imgs", self.root / "masks"),
            (self.root / "imgs2", self.root / "masks2"),
            (self.root / "imgsd", self.root / "masksd"),
        ]

    def _default_split_match(self, img_path: Path) -> bool:
        base = _stem_base(img_path.stem)
        numeric_tail = "".join(ch for ch in base if ch.isdigit())
        ends_with_nine = bool(numeric_tail) and numeric_tail.endswith("9")

        if self.split == "val":
            return ends_with_nine
        if self.split == "train":
            return not ends_with_nine
        return True

    def _collect_samples(self) -> list[tuple[Path, Optional[Path]]]:
        samples: list[tuple[Path, Optional[Path]]] = []
        for img_dir, mask_dir in self._iter_dir_pairs():
            if not img_dir.exists():
                continue

            mask_index = {}
            if mask_dir is not None and mask_dir.exists():
                for mask_path in mask_dir.glob("*.png"):
                    mask_index[mask_path.name] = mask_path

            for img_path in sorted(path for path in img_dir.glob("*.png") if path.is_file()):
                if self._split_entries is not None:
                    if not match_split_entry(img_path, img_dir, self._split_entries):
                        continue
                elif not self._default_split_match(img_path):
                    continue

                mask_path = mask_index.get(img_path.name)
                if self.split != "test" and mask_path is None:
                    continue
                samples.append((img_path, mask_path))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_mask(self, img_path: Path) -> Path:
        for sample_img, sample_mask in self.samples:
            if sample_img == img_path and sample_mask is not None:
                return sample_mask
        raise FileNotFoundError(f"Mask not found for {img_path}")

    def load_mask_ids(self, mask_path: Path) -> np.ndarray:
        mask = Image.open(mask_path)
        arr = np.asarray(mask)

        if arr.ndim == 3:
            return color_mask_to_id(arr, self._color2id, ignore_index=self.ignore_index)

        raw = np.asarray(mask.convert("L"), dtype=np.uint8)
        mapped = np.full(raw.shape, fill_value=self.ignore_index, dtype=np.uint8)
        valid_vals = raw[raw != self.ignore_index]

        if valid_vals.size > 0 and int(valid_vals.max()) <= (self.meta.num_classes - 1):
            direct_valid = raw < self.meta.num_classes
            mapped[direct_valid] = raw[direct_valid]
        else:
            one_based_valid = (raw >= 1) & (raw <= self.meta.num_classes)
            mapped[one_based_valid] = raw[one_based_valid] - 1

        ignore_valid = raw == self.ignore_index
        mapped[ignore_valid] = self.ignore_index
        return mapped

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if mask_path is None:
            mask_new = np.full((img.height, img.width), fill_value=self.ignore_index, dtype=np.uint8)
        else:
            mask_new = self.load_mask_ids(mask_path)

        mask_id = Image.fromarray(mask_new, mode="L")

        if self.training:
            if self.multi_scale_range != (1.0, 1.0):
                img, mask_id = random_scale_pair(img, mask_id, self.multi_scale_range)
            img, mask_id = hflip_pair(img, mask_id, self.hflip_prob)
            img = color_jitter(
                img,
                prob=self.color_jitter_prob,
                brightness=self.color_jitter_brightness,
                contrast=self.color_jitter_contrast,
                saturation=self.color_jitter_saturation,
            )
            img = gaussian_blur(
                img,
                prob=self.gaussian_blur_prob,
                radius_range=self.gaussian_blur_radius_range,
            )

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
        rel_name = img_path.name
        return img_t, mask_t, rel_name
