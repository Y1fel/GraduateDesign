from __future__ import annotations

from config.config import TrainConfig
from src.models.comparison_segmentors import (
    FCNSegmentor,
    PSPNetSegmentor,
    UNetSegmentor,
)
from src.models.deeplabv3_plus import DeepLabV3Plus


def normalize_model_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "deeplabv3plus": "deeplabv3plus",
        "deeplab": "deeplabv3plus",
        "unet": "unet",
        "pspnet": "pspnet",
        "fcn": "fcn",
    }
    if normalized not in aliases:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported model_name={name}. Use one of: {supported}")
    return aliases[normalized]


def build_segmentation_model(cfg: TrainConfig):
    model_name = normalize_model_name(getattr(cfg, "model_name", "deeplabv3plus"))
    num_classes = int(cfg.num_classes)

    if model_name == "deeplabv3plus":
        return DeepLabV3Plus(
            num_classes=num_classes,
            backbone_pretrained=cfg.backbone_pretrained,
            backbone_name=cfg.backbone_name,
            output_stride=cfg.output_stride,
            segmentation_head=cfg.segmentation_head,
            aspp_dropout=cfg.aspp_dropout,
            hybrid_variant=cfg.hybrid_variant,
            hybrid_use_strip=cfg.hybrid_use_strip,
            hybrid_strip_kernel=cfg.hybrid_strip_kernel,
            hybrid_mid_kernel=cfg.hybrid_mid_kernel,
            hybrid_large_kernel=cfg.hybrid_large_kernel,
            hybrid_gate_reduction=cfg.hybrid_gate_reduction,
            hybrid_residual_channels=cfg.hybrid_residual_channels,
            hybrid_residual_init=cfg.hybrid_residual_init,
            hybrid_dropout=cfg.hybrid_dropout,
            decoder_upsample_mode=cfg.decoder_upsample_mode,
            decoder_dropout=cfg.decoder_dropout,
        )

    if model_name == "unet":
        return UNetSegmentor(
            num_classes=num_classes,
            backbone_pretrained=cfg.backbone_pretrained,
            backbone_name=cfg.backbone_name,
            output_stride=cfg.output_stride,
            decoder_dropout=cfg.decoder_dropout,
        )

    if model_name == "pspnet":
        return PSPNetSegmentor(
            num_classes=num_classes,
            backbone_pretrained=cfg.backbone_pretrained,
            backbone_name=cfg.backbone_name,
            output_stride=cfg.output_stride,
            decoder_dropout=cfg.decoder_dropout,
        )

    if model_name == "fcn":
        return FCNSegmentor(
            num_classes=num_classes,
            backbone_pretrained=cfg.backbone_pretrained,
            backbone_name=cfg.backbone_name,
            output_stride=cfg.output_stride,
            decoder_dropout=cfg.decoder_dropout,
        )
