import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from PIL import ImageEnhance


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


def _clamp_unit(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def photometric_augment(
    img: Image.Image,
    *,
    prob: float,
    brightness: float,
    contrast: float,
    saturation: float,
    gamma_range: Tuple[float, float],
    op_prob: float = 0.5,
) -> Image.Image:
    if random.random() >= _clamp_unit(prob):
        return img

    out = img
    op_prob = _clamp_unit(op_prob)

    def _sample_factor(span: float) -> float:
        span = max(0.0, float(span))
        return random.uniform(1.0 - span, 1.0 + span)

    if brightness > 0 and random.random() < op_prob:
        out = ImageEnhance.Brightness(out).enhance(_sample_factor(brightness))
    if contrast > 0 and random.random() < op_prob:
        out = ImageEnhance.Contrast(out).enhance(_sample_factor(contrast))
    if saturation > 0 and random.random() < op_prob:
        out = ImageEnhance.Color(out).enhance(_sample_factor(saturation))

    g0, g1 = float(gamma_range[0]), float(gamma_range[1])
    lo, hi = (g0, g1) if g0 <= g1 else (g1, g0)
    lo = max(1e-3, lo)
    hi = max(lo, hi)
    if random.random() < op_prob:
        gamma = random.uniform(lo, hi)
        arr = np.asarray(out, dtype=np.float32) / 255.0
        arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
    else:
        arr = np.asarray(out, dtype=np.float32) / 255.0
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")




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
