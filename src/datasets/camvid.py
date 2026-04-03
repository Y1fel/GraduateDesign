from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.camvid_labels import CAMVID_IGNORE_LABEL_NAMES, CAMVID_LABELS
from src.datasets.meta import SegmentationDatasetMeta
from src.datasets.transforms import (
    color_jitter,
    gaussian_blur,
    hflip_pair,
    normalize_img,
    pil_to_tensor,
    random_scale_pair,
)
from src.utils.Id2Mask import color_mask_to_id, load_class_dict_csv


def _normalize_label_name(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _normalize_stem(stem: str) -> str:
    norm = stem.lower()
    suffixes = ("_labelids", "_labels", "_label", "_mask", "_trainids", "_trainid", "_l")
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if norm.endswith(suffix):
                norm = norm[: -len(suffix)]
                changed = True
    return norm


def _find_existing_dir(root: Path, candidates: list[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class CamVidDataset(Dataset):
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

        self.images_root = _find_existing_dir(
            self.root,
            [
                self.root / split,
                self.root / f"{split}_images",
                self.root / "images" / split,
            ],
        )
        if self.images_root is None:
            raise FileNotFoundError(f"CamVid images dir not found for split={split} under {self.root}")

        self.labels_root = _find_existing_dir(
            self.root,
            [
                self.root / f"{split}_labels",
                self.root / f"{split}annot",
                self.root / f"{split}_annotations",
                self.root / f"{split}_masks",
                self.root / "labels" / split,
                self.root / "annotations" / split,
            ],
        )
        if split != "test" and self.labels_root is None:
            raise FileNotFoundError(f"CamVid labels dir not found for split={split} under {self.root}")

        self._init_metadata()
        self.img_paths = self._collect_image_paths()
        self._mask_index = self._build_mask_index() if self.labels_root is not None else {}
        if not self.img_paths:
            raise RuntimeError(f"No CamVid images found in {self.images_root}")

    def _init_metadata(self) -> None:
        class_dict_path = self.root / "class_dict.csv"
        loaded = load_class_dict_csv(class_dict_path)
        if loaded is None:
            rows = CAMVID_LABELS
        else:
            _, csv_colors, csv_names = loaded
            rows = list(zip(csv_names, csv_colors))

        class_names = []
        id2color = []
        raw_id_to_train_id = []
        color2id = {}

        ignore_name_set = {_normalize_label_name(name) for name in CAMVID_IGNORE_LABEL_NAMES}

        for raw_id, (name, color) in enumerate(rows):
            label_name = name.strip() or f"class_{raw_id}"
            normalized_name = _normalize_label_name(label_name)
            if normalized_name in ignore_name_set:
                raw_id_to_train_id.append(self.ignore_index)
                continue

            train_id = len(class_names)
            raw_id_to_train_id.append(train_id)
            class_names.append(label_name)
            id2color.append(tuple(int(v) for v in color))
            color2id[tuple(int(v) for v in color)] = train_id

        self.meta = SegmentationDatasetMeta(
            dataset_name="camvid",
            num_classes=len(class_names),
            class_names=tuple(class_names),
            id2color=tuple(id2color),
        )
        self._color2id = color2id
        self._raw_id_to_train_id = np.asarray(raw_id_to_train_id, dtype=np.int64)

    def _collect_image_paths(self) -> list[Path]:
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        img_paths = []
        for pattern in patterns:
            img_paths.extend(self.images_root.rglob(pattern))
        return sorted(path for path in img_paths if path.is_file())

    def _build_mask_index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        for pattern in patterns:
            for mask_path in self.labels_root.rglob(pattern):
                rel = mask_path.relative_to(self.labels_root)
                key = "/".join((*rel.parts[:-1], _normalize_stem(rel.stem)))
                index[key.lower()] = mask_path
                index.setdefault(_normalize_stem(mask_path.stem), mask_path)
        return index

    def __len__(self) -> int:
        return len(self.img_paths)

    def _resolve_mask(self, img_path: Path) -> Path:
        if self.labels_root is None:
            raise RuntimeError(f"CamVid split={self.split} has no labels directory")

        rel = img_path.relative_to(self.images_root)
        rel_key = "/".join((*rel.parts[:-1], _normalize_stem(rel.stem))).lower()
        mask_path = self._mask_index.get(rel_key) or self._mask_index.get(_normalize_stem(img_path.stem))
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {img_path.name} under {self.labels_root}")
        return mask_path

    def load_mask_ids(self, mask_path: Path) -> np.ndarray:
        mask = Image.open(mask_path)
        arr = np.asarray(mask)

        if arr.ndim == 3:
            return color_mask_to_id(arr, self._color2id, ignore_index=self.ignore_index)

        mask_id = np.asarray(mask.convert("L"), dtype=np.int64)
        mapped = np.full(mask_id.shape, fill_value=self.ignore_index, dtype=np.uint8)
        valid = (mask_id >= 0) & (mask_id < len(self._raw_id_to_train_id))
        mapped[valid] = self._raw_id_to_train_id[mask_id[valid]].astype(np.uint8)
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
