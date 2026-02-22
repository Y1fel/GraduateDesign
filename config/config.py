from dataclasses import dataclass
from pathlib import Path

from src.models.switch2Norm import NormType

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    data_root: Path = PROJECT_ROOT / "data" / "archive" / "CamVid"

    num_classes: int = 32
    ignore_index: int = 255

    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    lr_0: float = 0.01
    weight_decay: float = 1e-4

    label_smoothing: float = 0.0

    output_stride: int = 16
    backbone_pretrained: bool = True
    head_norm: NormType = "bn"

    resize_h: int = 720
    resize_w: int = 960
    crop_h: int = 720
    crop_w: int = 960
    train_multi_scale_min: float = 1.0
    train_multi_scale_max: float = 1.0
    hflip_prob: float = 0.5

    photo_aug_prob: float = 0.35
    brightness_jitter: float = 0.10
    contrast_jitter: float = 0.08
    saturation_jitter: float = 0.06
    gamma_min: float = 0.95
    gamma_max: float = 1.05
    photo_op_prob: float = 0.50
    blur_prob: float = 0.03
    blur_radius_min: float = 0.1
    blur_radius_max: float = 0.6
    jpeg_prob: float = 0.03
    jpeg_quality_min: int = 90
    jpeg_quality_max: int = 98
    photo_aug_warmup_epochs: int = 20

    train_auto_contrast_enable: bool = False
    train_auto_contrast_cutoff: float = 1.0
    eval_auto_contrast_enable: bool = False
    eval_auto_contrast_cutoff: float = 1.0

    train_low_light_preprocess_enable: bool = False
    eval_low_light_preprocess_enable: bool = False
    low_light_gamma: float = 0.85
    low_light_brightness_gain: float = 1.10

    save_vis_every: int = 50
    save_vis_max_items: int = 8

    outputs_root: Path = PROJECT_ROOT / "outputs"
    seed: int = 42
    use_amp: bool = False
