from pathlib import Path

from src.datasets.comma10k import Comma10KDataset
from src.datasets.camvid import CamVidDataset
from src.datasets.cityscapes import CityscapesDataset
from src.datasets.kitti_semantic import KITTISemanticDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def normalize_dataset_name(name: str) -> str:
    normalized = str(name).strip().lower()
    aliases = {
        "cityscapes": "cityscapes",
        "camvid": "camvid",
        "kitti_semantic": "kitti_semantic",
        "kitti-semantic": "kitti_semantic",
        "kitti": "kitti_semantic",
        "comma10k": "comma10k",
        "comma": "comma10k",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported dataset_name={name}. Use cityscapes/camvid/kitti_semantic/comma10k.")
    return aliases[normalized]


def apply_dataset_profile(cfg) -> None:
    dataset_name = normalize_dataset_name(getattr(cfg, "dataset_name", "cityscapes"))
    cfg.dataset_name = dataset_name

    if not bool(getattr(cfg, "use_dataset_profile", True)):
        return

    if dataset_name == "cityscapes":
        cfg.num_classes = 19
        cfg.crop_h = 769
        cfg.crop_w = 769
        return

    if dataset_name == "camvid":
        cfg.num_classes = 11
        cfg.crop_h = 360
        cfg.crop_w = 480
        return

    if dataset_name == "kitti_semantic":
        cfg.num_classes = 19
        cfg.crop_h = 352
        cfg.crop_w = 1024
        return

    if dataset_name == "comma10k":
        cfg.num_classes = 6
        cfg.crop_h = 512
        cfg.crop_w = 1024
        return

    raise ValueError(f"Unsupported dataset_name={dataset_name}. Use cityscapes/camvid/kitti_semantic/comma10k.")


def resolve_dataset_root(cfg) -> Path:
    configured_root = Path(cfg.data_root)
    if configured_root != DEFAULT_DATA_ROOT:
        return configured_root

    dataset_name = normalize_dataset_name(getattr(cfg, "dataset_name", "cityscapes"))
    if dataset_name == "cityscapes":
        return Path(cfg.cityscapes_root)
    if dataset_name == "camvid":
        return Path(cfg.camvid_root)
    if dataset_name == "kitti_semantic":
        return Path(cfg.kitti_semantic_root)
    return Path(cfg.comma10k_root)


def build_dataset(cfg, split: str, training: bool):
    dataset_name = normalize_dataset_name(getattr(cfg, "dataset_name", "cityscapes"))
    root = resolve_dataset_root(cfg)
    common_kwargs = dict(
        root=root,
        split=split,
        ignore_index=cfg.ignore_index,
        training=training,
        hflip_prob=cfg.hflip_prob,
        multi_scale_range=(cfg.train_multi_scale_min, cfg.train_multi_scale_max),
        random_crop_size=(cfg.crop_w, cfg.crop_h),
        crop_retry=cfg.crop_retry,
        crop_max_class_ratio=cfg.crop_max_class_ratio,
        color_jitter_prob=cfg.color_jitter_prob,
        color_jitter_brightness=cfg.color_jitter_brightness,
        color_jitter_contrast=cfg.color_jitter_contrast,
        color_jitter_saturation=cfg.color_jitter_saturation,
        gaussian_blur_prob=cfg.gaussian_blur_prob,
        gaussian_blur_radius_range=(cfg.gaussian_blur_radius_min, cfg.gaussian_blur_radius_max),
    )

    if not training:
        common_kwargs["hflip_prob"] = 0.0
        common_kwargs["multi_scale_range"] = (1.0, 1.0)
        common_kwargs["random_crop_size"] = None
        common_kwargs["crop_retry"] = 1
        common_kwargs["crop_max_class_ratio"] = 1.0
        common_kwargs["color_jitter_prob"] = 0.0
        common_kwargs["gaussian_blur_prob"] = 0.0

    if dataset_name == "cityscapes":
        return CityscapesDataset(**common_kwargs, remap_to_19=True)
    if dataset_name == "camvid":
        return CamVidDataset(**common_kwargs)
    if dataset_name == "kitti_semantic":
        return KITTISemanticDataset(**common_kwargs)
    return Comma10KDataset(**common_kwargs)
