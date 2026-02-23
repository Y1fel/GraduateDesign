from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    data_root: Path = PROJECT_ROOT / "data"

    num_classes: int = 19
    ignore_index: int = 255

    epochs: int = 120
    batch_size: int = 8
    num_workers: int = 8
    persistent_workers: bool = True
    prefetch_factor: int = 4
    lr_0: float = 2.5e-4
    weight_decay: float = 1e-3
    lr_eta_min: float = 1e-6
    lr_policy: str = "poly"
    poly_power: float = 0.9
    warmup_iters: int = 1500
    warmup_ratio: float = 0.1
    # Freeze BN only for small-batch settings after stability is verified.
    freeze_bn: bool = False
    dominant_class_warn_ratio: float = 0.9

    label_smoothing: float = 0.03
    use_class_weights: bool = True
    class_weight_power: float = 0.6
    class_weight_min: float = 0.3
    class_weight_max: float = 3.0

    output_stride: int = 16
    backbone_pretrained: bool = True
    aspp_dropout: float = 0.1
    decoder_dropout: float = 0.2

    resize_h: int = 1024
    resize_w: int = 2048
    crop_h: int = 896
    crop_w: int = 896
    crop_retry: int = 10
    crop_max_class_ratio: float = 0.75
    train_multi_scale_min: float = 0.5
    train_multi_scale_max: float = 2
    hflip_prob: float = 0.5

    # Eval-time TTA.
    eval_multi_scale: bool = True
    eval_scales: tuple[float, ...] = (0.75, 1.0, 1.25)
    eval_flip: bool = True

    save_vis_every: int = 50
    save_vis_max_items: int = 10

    outputs_root: Path = PROJECT_ROOT / "outputs"
    seed: int = 42
    use_amp: bool = True

    # Rare-class-aware sampling for long-tail classes.
    use_rare_class_sampler: bool = True
    rare_class_ids: tuple[int, ...] = (3, 5, 6, 7, 17, 15, 14, 12)
    rare_class_weight_multiplier: float = 4.0
    sampler_num_samples_factor: float = 1.0

    # Hybrid loss: ce_weight * CE + focal_weight * Focal.
    ce_weight: float = 0.45
    focal_weight: float = 0.55
    focal_gamma: float = 2.0
