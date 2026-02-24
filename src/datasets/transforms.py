import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from src.commom.constants import IMAGENET_MEAN, IMAGENET_STD


def normalize_img(img_t: torch.Tensor) -> torch.Tensor:
    """
    Normalize image tensor with ImageNet statistics.
    img_t: (3,H,W), float in [0,1]
    """
    # Pair with save_predictions_triplet denormalization in src/viz/visualizer.py
    mean = torch.tensor(IMAGENET_MEAN, dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    return (img_t - mean) / std


def pil_hflip(im: Image.Image) -> Image.Image:
    # Pillow 新写法
    if hasattr(Image, "Transpose"):
        return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    # 兼容旧写法
    return im.transpose(Image.FLIP_LEFT_RIGHT)



def random_scale_pair(
    img: Image.Image,
    mask: Image.Image,
    scale_range: Tuple[float, float],
) -> Tuple[Image.Image, Image.Image]:
    lo, hi = float(scale_range[0]), float(scale_range[1])
    if lo <= 0:
        lo = 1e-3
    if hi < lo:
        lo, hi = hi, lo

    s = random.uniform(lo, hi)
    w, h = img.size
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    mask = mask.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
    return img, mask


def maybe_hflip_pair(
    img: Image.Image,
    mask: Image.Image,
    prob: float,
) -> Tuple[Image.Image, Image.Image]:
    if random.random() < prob:
        return pil_hflip(img), pil_hflip(mask)
    return img, mask


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.uint8)  # (H,W,3)
    t = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0
    return t
