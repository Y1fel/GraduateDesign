from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from src.models.aspp import ASPP
from src.models.decoder import DeepLabV3PlusDecoder


class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 backbone for DeepLabV3+.

    Returns:
      - low_level: stride-4 feature (C=24)
      - mid_level: stride-8 feature (C=32)
      - high_level: stride-16 feature (C=96)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        m = mobilenet_v2(weights=weights)
        self.features = m.features

        self.low_level_channels = 24
        self.out_channels = 96

    def forward(self, x: torch.Tensor):
        low_level = None
        mid_level = None
        high_level = None

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 3:
                low_level = x
            elif idx == 6:
                mid_level = x
            elif idx == 13:
                high_level = x
                break

        if low_level is None or mid_level is None or high_level is None:
            raise RuntimeError("Failed to extract MobileNetV2 feature maps.")

        return low_level, mid_level, high_level


class DeepLabV3PlusMobileNetV2(nn.Module):
    """DeepLabV3+ with MobileNetV2 backbone, intended for student model."""

    def __init__(
        self,
        num_classes: int,
        backbone_pretrained: bool = True,
        aspp_out_channels: int = 256,
        decoder_channels: int = 256,
        aspp_dropout: float = 0.1,
        decoder_dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone = MobileNetV2Backbone(pretrained=backbone_pretrained)

        self.aspp = ASPP(
            in_channels=self.backbone.out_channels,
            out_channels=aspp_out_channels,
            atrous_rates=(6, 12, 18),
            dropout=aspp_dropout,
        )

        self.decoder = DeepLabV3PlusDecoder(
            low_level_in_channels=self.backbone.low_level_channels,
            aspp_out_channels=aspp_out_channels,
            decoder_channels=decoder_channels,
            dropout=decoder_dropout,
        )

        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x: torch.Tensor):
        input_size = x.shape[-2:]
        low_level, _, high_level = self.backbone(x)
        aspp_feat = self.aspp(high_level)
        dec_feat = self.decoder(low_level, aspp_feat)
        logits = self.classifier(dec_feat)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits
