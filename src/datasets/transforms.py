import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def normalize_img(img_t: torch.Tensor) -> torch.Tensor:
    """
    img_t: (3,H,W), float in [0,1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_t - mean) / std


def pil_hflip(im: Image.Image) -> Image.Image:
    # Pillow 新写法
    if hasattr(Image, "Transpose"):
        return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    # 兼容旧写法
    return im.transpose(Image.FLIP_LEFT_RIGHT)


def resize_pair(
    img: Image.Image,
    mask: Image.Image,
    size_wh: Tuple[int, int],
) -> Tuple[Image.Image, Image.Image]:
    w, h = size_wh
    img = img.resize((w, h), resample=Image.Resampling.BILINEAR)
    mask = mask.resize((w, h), resample=Image.Resampling.NEAREST)
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
