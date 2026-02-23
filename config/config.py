from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.models.switch2Norm import NormType

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    data_root: Path = PROJECT_ROOT / "data"

    num_classes: int = 19
    ignore_index: int = 255

    epochs: int = 100
    batch_size: int = 12
    num_workers: int = 12
    persistent_workers: bool = True
    prefetch_factor: int = 6
    lr_0: float = 3e-4
    weight_decay: float = 1e-3
    lr_eta_min: float = 1e-6
    # Freeze BN only for small-batch settings after stability is verified.
    freeze_bn: bool = False
    dominant_class_warn_ratio: float = 0.9

    label_smoothing: float = 0.0

    output_stride: int = 16
    backbone_pretrained: bool = True
    head_norm: NormType = "bn"
    aspp_dropout: float = 0.1
    decoder_dropout: float = 0.2

    resize_h: int = 1024
    resize_w: int = 2048
    crop_h: int = 768
    crop_w: int = 768
    train_multi_scale_min: float = 1.0
    train_multi_scale_max: float = 1.0
    hflip_prob: float = 0.5


    save_vis_every: int = 50
    save_vis_max_items: int = 10

    outputs_root: Path = PROJECT_ROOT / "outputs"
    seed: int = 42
    use_amp: bool = True

    # Rare-class-aware sampling for long-tail classes.
    use_rare_class_sampler: bool = True
    rare_class_ids: tuple[int, ...] = (14, 16, 17, 12)
    rare_class_weight_multiplier: float = 3.0
    sampler_num_samples_factor: float = 1.0

    # Hybrid loss: ce_weight * CE + focal_weight * Focal.
    ce_weight: float = 0.5
    focal_weight: float = 0.5
    focal_gamma: float = 2.0

    # Inference post-processing.
    # Keep eval off by default to avoid CPU bottlenecks during validation.
    enable_postprocess_eval: bool = False
    enable_postprocess_vis: bool = True
    postprocess_min_component_area: int = 20
    postprocess_filter: Literal["majority", "median"] = "majority"
    postprocess_kernel_size: int = 3
