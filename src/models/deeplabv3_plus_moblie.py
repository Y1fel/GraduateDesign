import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.MobileNetV2 import MobileNetV2
from src.models.aspp import ASPP
from src.models.decoder import DeepLabV3PlusDecoder


class MobileNetV2Backbone(nn.Module):
    def __init__(self, output_stride: int = 16):
        super().__init__()
        if output_stride not in (16, 32):
            raise ValueError("output_stride error")

        backbone = MobileNetV2()
        self.features = backbone.features
        self.output_stride = output_stride

        self.low_level_channels = 24
        self.out_channels = 96 if output_stride == 16 else 1280

    def forward(self, x: torch.Tensor):
        low_level = None
        high_level = None

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 3:
                low_level = x
            if self.output_stride == 16 and idx == 13:
                high_level = x

        if self.output_stride == 32:
            high_level = x

        return low_level, high_level


class DeepLabV3PlusMobile(nn.Module):
    def __init__(
        self,
        num_classes: int,
        output_stride: int = 16,
        aspp_out_channels: int = 256,
        decoder_channels: int = 256,
        aspp_dropout: float = 0.1,
        decoder_dropout: float = 0.2,
        decoder_upsample_mode: str = "learnable",
    ):
        super().__init__()

        self.backbone = MobileNetV2Backbone(output_stride=output_stride)

        rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)
        self.aspp = ASPP(
            in_channels=self.backbone.out_channels,
            out_channels=aspp_out_channels,
            atrous_rates=rates,
            dropout=aspp_dropout,
        )

        self.decoder = DeepLabV3PlusDecoder(
            low_level_in_channels=self.backbone.low_level_channels,
            aspp_out_channels=aspp_out_channels,
            decoder_channels=decoder_channels,
            dropout=decoder_dropout,
            upsample_mode=decoder_upsample_mode,
        )

        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        self.aux_classifier = nn.Conv2d(self.backbone.out_channels, num_classes, kernel_size=1)

        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

        nn.init.normal_(self.aux_classifier.weight, mean=0.0, std=0.01)
        if self.aux_classifier.bias is not None:
            nn.init.constant_(self.aux_classifier.bias, 0.0)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        input_size = x.shape[-2:]

        low_level, high_level = self.backbone(x)
        aspp_feat = self.aspp(high_level)
        dec_feat = self.decoder(low_level, aspp_feat)
        logits = self.classifier(dec_feat)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        if not return_aux:
            return logits

        aux_logits = self.aux_classifier(high_level)
        aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)
        return logits, aux_logits
