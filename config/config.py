from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    data_root: Path = PROJECT_ROOT / "data"
    dataset_name: str = "cityscapes"  # "cityscapes" / "camvid" / "kitti_semantic" / "comma10k"
    use_dataset_profile: bool = True  # True: apply dataset-specific num_classes and crop size defaults
    cityscapes_root: Path = PROJECT_ROOT / "data" / "cityscapes"
    camvid_root: Path = PROJECT_ROOT / "data" / "CamVid"
    kitti_semantic_root: Path = PROJECT_ROOT / "data" / "KITTI_semantic"
    comma10k_root: Path = PROJECT_ROOT / "data" / "comma10k"

    num_classes: int = 19
    ignore_index: int = 255

    # Base training
    epochs: int = 150
    batch_size: int = 8
    num_workers: int = 12
    persistent_workers: bool = True
    prefetch_factor: int = 4

    # Optimizer / schedule
    lr_0: float = 0.01
    weight_decay: float = 5e-4
    lr_eta_min: float = 5e-5
    lr_policy: str = "poly"
    poly_power: float = 0.9
    warmup_iters: int = 1500
    warmup_ratio: float = 0.1

    # BatchNorm / monitoring
    freeze_bn: bool = False
    dominant_class_warn_ratio: float = 0.9

    # Class-weight-related options
    label_smoothing: float = 0.0
    use_class_weights: bool = False
    class_weight_strategy: str = "median_frequency"
    class_weight_power: float = 0.6
    class_weight_min: float = 0.8
    class_weight_max: float = 2.0
    class_weight_rare_cap: float = 2.2
    class_weight_boost_low_iou_every: int = 100
    class_weight_boost_low_iou_threshold: float = 0.2
    class_weight_boost_factor: float = 1.05

    # Rare-class sampler
    use_rare_class_sampler: bool = False
    rare_class_ids: tuple[int, ...] = (3, 4, 5, 6, 9, 12, 16, 17)
    rare_class_weight_multiplier: float = 2.0
    sampler_num_samples_factor: float = 1.0

    # Model
    output_stride: int = 8
    backbone_pretrained: bool = True
    backbone_name: str = "rsnet-100"
    segmentation_head: str = "aspp"  # "aspp" / "ocr"
    aspp_dropout: float = 0.05
    ocr_mid_channels: int = 512
    ocr_key_channels: int = 256
    ocr_dropout: float = 0.05
    decoder_upsample_mode: str = "learnable"  # "learnable" / "bilinear"
    decoder_dropout: float = 0.15
    aux_loss_weight: float = 0.4
    use_aux_loss: bool = True

    # Augmentation
    crop_h: int = 769
    crop_w: int = 769
    crop_retry: int = 1   #10
    crop_max_class_ratio: float = 1.0    #0.7
    train_multi_scale_min: float = 0.5
    train_multi_scale_max: float = 2.0
    hflip_prob: float = 0.5
    color_jitter_prob: float = 0.0
    color_jitter_brightness: float = 0.0
    color_jitter_contrast: float = 0.0
    color_jitter_saturation: float = 0.0
    gaussian_blur_prob: float = 0.0
    gaussian_blur_radius_min: float = 0.1
    gaussian_blur_radius_max: float = 1.3

    # Outputs
    save_vis_every: int = 20
    save_vis_max_items: int = 10
    outputs_root: Path = PROJECT_ROOT / "outputs"

    # Loss mode:
    # - "ce": plain cross entropy
    # - "baseline": CE + focal
    # - "ohem": OHEM CE
    # - "ohem_boundary": OHEM + boundary
    loss_mode: str = "ce"

    # CE + Focal
    ce_weight: float = 0.75
    focal_weight: float = 0.25
    focal_gamma: float = 2.0

    # OHEM
    ohem_ratio: float = 0.20
    ohem_weight: float = 1.0
    boundary_weight: float = 0.15
    boundary_kernel_size: int = 3
    report_loss_every: int = 5

    seed: int = 42
    use_amp: bool = True


@dataclass
class MobileTrainConfig(TrainConfig):
    epochs: int = 90

    color_jitter_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    gaussian_blur_prob: float = 0.2
    gaussian_blur_radius_min: float = 0.1
    gaussian_blur_radius_max: float = 1.3

    class_weight_boost_factor: float = 1.1
    output_stride: int = 16

    use_distillation: bool = True
    distill_teacher_ckpt: Path = PROJECT_ROOT / "outputs" / "best.pth"
    distill_teacher_arch: str = "resnet"
    distill_teacher_backbone_name: str = "rsnet-100"
    distill_teacher_backbone_pretrained: bool = False
    distill_teacher_output_stride: int = 8
    distill_type: str = "cwd"  # "cwd" / "kl"
    distill_temperature: float = 4.0
    distill_loss_weight: float = 0.7
    distill_aux_weight: float = 0.4
