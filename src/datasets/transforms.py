import random
from typing import Any, Tuple

import numpy as np
import torch
from PIL import Image
from PIL import ImageEnhance, ImageFilter


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


def low_light_preprocess(
    img: Image.Image,
    *,
    gamma: float = 1.0,
    brightness_gain: float = 1.0,
) -> Image.Image:
    gamma = max(1e-3, float(gamma))
    brightness_gain = max(1e-3, float(brightness_gain))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
    arr = np.clip(arr * brightness_gain, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def photometric_augment(
    img: Image.Image,
    *,
    prob: float,
    brightness: float,
    contrast: float,
    saturation: float,
    gamma_range: Tuple[float, float],
    op_prob: float = 0.5,
    blur_prob: float = 0.0,
    blur_radius_range: Tuple[float, float] = (0.0, 0.0),
    jpeg_prob: float = 0.0,
    jpeg_quality_range: Tuple[int, int] = (95, 100),
    return_stats: bool = False,
) -> Image.Image | tuple[Image.Image, dict[str, Any]]:
    stats = {"photometric_applied": False, "blur_applied": False, "jpeg_applied": False}
    if random.random() >= _clamp_unit(prob):
        return (img, stats) if return_stats else img

    stats["photometric_applied"] = True
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


    blur_prob = _clamp_unit(blur_prob)
    if blur_prob > 0 and random.random() < blur_prob:
        br0, br1 = float(blur_radius_range[0]), float(blur_radius_range[1])
        lo_b, hi_b = (br0, br1) if br0 <= br1 else (br1, br0)
        lo_b = max(0.0, lo_b)
        hi_b = max(lo_b, hi_b)
        radius = random.uniform(lo_b, hi_b)
        if radius > 0:
            out = out.filter(ImageFilter.GaussianBlur(radius=radius))
            stats["blur_applied"] = True

    jpeg_prob = _clamp_unit(jpeg_prob)
    if jpeg_prob > 0 and random.random() < jpeg_prob:
        q0, q1 = int(jpeg_quality_range[0]), int(jpeg_quality_range[1])
        lo_q, hi_q = (q0, q1) if q0 <= q1 else (q1, q0)
        lo_q = max(40, min(100, lo_q))
        hi_q = max(lo_q, min(100, hi_q))
        import io

        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=random.randint(lo_q, hi_q))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        stats["jpeg_applied"] = True

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
    out_img = Image.fromarray(arr, mode="RGB")
    return (out_img, stats) if return_stats else out_img




def resize_pair(
    img: Image.Image,
    mask: Image.Image,
    size_wh: Tuple[int, int],
) -> Tuple[Image.Image, Image.Image]:
    w, h = size_wh
    img = img.resize((w, h), resample=Image.Resampling.BILINEAR)
    mask = mask.resize((w, h), resample=Image.Resampling.NEAREST)
    return img, mask


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
    return resize_pair(img, mask, (new_w, new_h))


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
