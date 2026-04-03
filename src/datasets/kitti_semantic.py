from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.cityscapes_labels import (
    CITYSCAPES_19_CLASS_NAMES,
    CITYSCAPES_19_ID2COLOR,
    CITYSCAPES_34_ID2COLOR,
    CITYSCAPES_34_TO_19,
)
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


def _find_existing_dir(root: Path, candidates: list[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class KITTISemanticDataset(Dataset):
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
            dataset_name="kitti_semantic",
            num_classes=19,
            class_names=tuple(CITYSCAPES_19_CLASS_NAMES),
            id2color=tuple(CITYSCAPES_19_ID2COLOR),
        )
        self._cityscapes_34_to_19 = np.asarray(CITYSCAPES_34_TO_19, dtype=np.uint8)
        color2id = {}
        for raw_id, color in enumerate(CITYSCAPES_34_ID2COLOR):
            train_id = int(CITYSCAPES_34_TO_19[raw_id])
            if train_id == self.ignore_index:
                continue
            color2id[tuple(color)] = train_id
        self._cityscapes_color2id = color2id

        self._setup_paths()
        self.img_paths = self._collect_split_images()
        if not self.img_paths:
            raise RuntimeError(f"No KITTI semantic images found for split={split} under {self.root}")

    def _setup_paths(self) -> None:
        split_entries = read_split_entries(self.root, self.split)

        if self.split == "test":
            self.images_root = _find_existing_dir(
                self.root,
                [
                    self.root / "testing" / "image_2",
                    self.root / "test" / "image_2",
                    self.root / "images" / "test",
                ],
            )
            self.labels_root = None
            self._split_entries = split_entries
            return

        self.images_root = _find_existing_dir(
            self.root,
            [
                self.root / "training" / "image_2",
                self.root / "train" / "image_2",
                self.root / "images" / "training",
                self.root / "images" / "train",
            ],
        )
        self.labels_root = _find_existing_dir(
            self.root,
            [
                self.root / "training" / "semantic",
                self.root / "training" / "semantic_rgb",
                self.root / "training" / "semantics",
                self.root / "train" / "semantic",
                self.root / "labels" / "training",
                self.root / "labels" / "train",
            ],
        )
        self._split_entries = split_entries

        if self.images_root is None:
            raise FileNotFoundError(f"KITTI semantic images dir not found under {self.root}")
        if self.labels_root is None:
            raise FileNotFoundError(f"KITTI semantic labels dir not found under {self.root}")

    def _collect_split_images(self) -> list[Path]:
        if self.images_root is None:
            raise FileNotFoundError(f"KITTI semantic images dir not found under {self.root}")

        img_paths = sorted(path for path in self.images_root.glob("*.png") if path.is_file())
        if self._split_entries is not None:
            return [path for path in img_paths if match_split_entry(path, self.images_root, self._split_entries)]

        if self.split == "test":
            return img_paths

        if self.split == "val":
            return [path for idx, path in enumerate(img_paths) if idx % 5 == 0]
        return [path for idx, path in enumerate(img_paths) if idx % 5 != 0]

    def __len__(self) -> int:
        return len(self.img_paths)

    def _resolve_mask(self, img_path: Path) -> Path:
        if self.labels_root is None:
            raise RuntimeError(f"KITTI semantic split={self.split} has no labels directory")

        mask_path = self.labels_root / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path.name}: {mask_path}")
        return mask_path

    def load_mask_ids(self, mask_path: Path) -> np.ndarray:
        mask = Image.open(mask_path)
        arr = np.asarray(mask)

        if arr.ndim == 3:
            return color_mask_to_id(arr, self._cityscapes_color2id, ignore_index=self.ignore_index)

        raw = np.asarray(mask.convert("L"), dtype=np.uint8)
        mapped = np.full(raw.shape, fill_value=self.ignore_index, dtype=np.uint8)

        train_id_valid = raw < self.meta.num_classes
        mapped[train_id_valid] = raw[train_id_valid]

        city_valid = (raw <= 33) & (~train_id_valid)
        mapped[city_valid] = self._cityscapes_34_to_19[raw[city_valid]]

        ignore_valid = raw == self.ignore_index
        mapped[ignore_valid] = self.ignore_index
        return mapped

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.labels_root is None:
            mask_new = np.full((img.height, img.width), fill_value=self.ignore_index, dtype=np.uint8)
        else:
            mask_path = self._resolve_mask(img_path)
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
        rel_name = str(img_path.relative_to(self.images_root))
        return img_t, mask_t, rel_name
