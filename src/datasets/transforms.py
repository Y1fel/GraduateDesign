import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from src.commom.constants import IMAGENET_MEAN, IMAGENET_STD


def normalize_img(img_t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    return (img_t - mean) / std


def pil_hflip(im: Image.Image) -> Image.Image:
    if hasattr(Image, "Transpose"):
        return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
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


def hflip_pair(
    img: Image.Image,
    mask: Image.Image,
    prob: float,
) -> Tuple[Image.Image, Image.Image]:
    if random.random() < prob:
        return pil_hflip(img), pil_hflip(mask)
    return img, mask

def color_jitter(
    img: Image.Image,
    prob: float,
    brightness: float,
    contrast: float,
    saturation: float,
) -> Image.Image:
    if random.random() >= prob:
        return img

    if brightness > 0:
        factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
        img = ImageEnhance.Brightness(img).enhance(factor)
    if contrast > 0:
        factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
        img = ImageEnhance.Contrast(img).enhance(factor)
    if saturation > 0:
        factor = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
        img = ImageEnhance.Color(img).enhance(factor)
    return img

def gaussian_blur(
    img: Image.Image,
    prob: float,
    radius_range: Tuple[float, float],
) -> Image.Image:
    if random.random() >= prob:
        return img

    lo, hi = float(radius_range[0]), float(radius_range[1])
    if hi < lo:
        lo, hi = hi, lo
    radius = random.uniform(max(0.0, lo), max(0.0, hi))
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.uint8)
    chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
    t = torch.from_numpy(chw).clone().float() / 255.0
    return t
