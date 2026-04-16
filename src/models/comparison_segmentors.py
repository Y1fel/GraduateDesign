from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder import ConvNormReLU
from src.models.encoder import ResNetBackbone


def _dropout_layer(p: float) -> nn.Module:
    return nn.Dropout(p=float(p)) if float(p) > 0.0 else nn.Identity()


class UpsampleFuseBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormReLU(in_channels + skip_channels, out_channels, k=3, p=1),
            ConvNormReLU(out_channels, out_channels, k=3, p=1),
            _dropout_layer(dropout),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 512, bins: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        if len(bins) == 0:
            raise ValueError("bins must not be empty")
        branch_channels = max(1, int(out_channels) // len(bins))
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    ConvNormReLU(in_channels, branch_channels, k=1),
                )
                for bin_size in bins
            ]
        )
        merged_channels = int(in_channels) + branch_channels * len(bins)
        self.bottleneck = ConvNormReLU(merged_channels, out_channels, k=3, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for branch in self.branches:
            pooled = branch(x)
            pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
            feats.append(pooled)
        return self.bottleneck(torch.cat(feats, dim=1))


class UNetSegmentor(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_pretrained: bool = True,
        backbone_name: str = "rsnet-50",
        output_stride: int = 16,
        decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = ResNetBackbone(
            pretrained=backbone_pretrained,
            output_stride=output_stride,
            backbone_name=backbone_name,
        )
        self.center = nn.Sequential(
            ConvNormReLU(self.backbone.out_channels, 512, k=3, p=1),
            ConvNormReLU(512, 512, k=3, p=1),
        )
        self.dec4 = UpsampleFuseBlock(512, self.backbone.high_mid_channels, 512, decoder_dropout)
        self.dec3 = UpsampleFuseBlock(512, self.backbone.mid_level_channels, 256, decoder_dropout)
        self.dec2 = UpsampleFuseBlock(256, self.backbone.low_level_channels, 128, decoder_dropout)
        self.dec1 = UpsampleFuseBlock(128, self.backbone.stem_channels, 64, decoder_dropout)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        self.aux_classifier = nn.Sequential(
            ConvNormReLU(self.backbone.high_mid_channels, 256, k=3, p=1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_preupsample: bool = False,
        return_features: bool = False,
    ):
        input_size = x.shape[-2:]
        feats = self.backbone.forward_features(x)
        center = self.center(feats["layer4"])
        d4 = self.dec4(center, feats["layer3"])
        d3 = self.dec3(d4, feats["layer2"])
        d2 = self.dec2(d3, feats["layer1"])
        d1 = self.dec1(d2, feats["stem"])

        logits_preupsample = self.classifier(d1)
        logits = F.interpolate(logits_preupsample, size=input_size, mode="bilinear", align_corners=False)
        aux_preupsample = self.aux_classifier(feats["layer3"])
        aux_logits = F.interpolate(aux_preupsample, size=input_size, mode="bilinear", align_corners=False)
        features = {
            "context": center,
            "decoder": d1,
            "high_level": feats["layer4"],
            "low_level": feats["layer1"],
        }
        if not return_aux:
            if return_preupsample:
                return (logits, logits_preupsample, features) if return_features else (logits, logits_preupsample)
            return (logits, features) if return_features else logits
        if return_preupsample:
            if return_features:
                return logits, aux_logits, logits_preupsample, aux_preupsample, features
            return logits, aux_logits, logits_preupsample, aux_preupsample
        if return_features:
            return logits, aux_logits, features
        return logits, aux_logits


class PSPNetSegmentor(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_pretrained: bool = True,
        backbone_name: str = "rsnet-50",
        output_stride: int = 16,
        decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = ResNetBackbone(
            pretrained=backbone_pretrained,
            output_stride=output_stride,
            backbone_name=backbone_name,
        )
        self.ppm = PyramidPoolingModule(self.backbone.out_channels, out_channels=512)
        self.head = nn.Sequential(
            ConvNormReLU(512, 512, k=3, p=1),
            _dropout_layer(decoder_dropout),
        )
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        self.aux_classifier = nn.Sequential(
            ConvNormReLU(self.backbone.high_mid_channels, 256, k=3, p=1),
            _dropout_layer(decoder_dropout),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_preupsample: bool = False,
        return_features: bool = False,
    ):
        input_size = x.shape[-2:]
        feats = self.backbone.forward_features(x)
        ppm_feat = self.ppm(feats["layer4"])
        dec_feat = self.head(ppm_feat)
        logits_preupsample = self.classifier(dec_feat)
        logits = F.interpolate(logits_preupsample, size=input_size, mode="bilinear", align_corners=False)
        aux_preupsample = self.aux_classifier(feats["layer3"])
        aux_logits = F.interpolate(aux_preupsample, size=input_size, mode="bilinear", align_corners=False)
        features = {
            "context": ppm_feat,
            "decoder": dec_feat,
            "high_level": feats["layer4"],
            "low_level": feats["layer1"],
        }
        if not return_aux:
            if return_preupsample:
                return (logits, logits_preupsample, features) if return_features else (logits, logits_preupsample)
            return (logits, features) if return_features else logits
        if return_preupsample:
            if return_features:
                return logits, aux_logits, logits_preupsample, aux_preupsample, features
            return logits, aux_logits, logits_preupsample, aux_preupsample
        if return_features:
            return logits, aux_logits, features
        return logits, aux_logits


class FCNSegmentor(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_pretrained: bool = True,
        backbone_name: str = "rsnet-50",
        output_stride: int = 16,
        decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = ResNetBackbone(
            pretrained=backbone_pretrained,
            output_stride=output_stride,
            backbone_name=backbone_name,
        )
        self.head = nn.Sequential(
            ConvNormReLU(self.backbone.out_channels, 512, k=3, p=1),
            _dropout_layer(decoder_dropout),
        )
        self.score_high = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_mid = nn.Conv2d(self.backbone.mid_level_channels, num_classes, kernel_size=1)
        self.score_low = nn.Conv2d(self.backbone.low_level_channels, num_classes, kernel_size=1)
        self.aux_classifier = nn.Conv2d(self.backbone.high_mid_channels, num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_preupsample: bool = False,
        return_features: bool = False,
    ):
        input_size = x.shape[-2:]
        feats = self.backbone.forward_features(x)
        context = self.head(feats["layer4"])
        score = self.score_high(context)
        score = F.interpolate(score, size=feats["layer2"].shape[-2:], mode="bilinear", align_corners=False)
        score = score + self.score_mid(feats["layer2"])
        score = F.interpolate(score, size=feats["layer1"].shape[-2:], mode="bilinear", align_corners=False)
        logits_preupsample = score + self.score_low(feats["layer1"])
        logits = F.interpolate(logits_preupsample, size=input_size, mode="bilinear", align_corners=False)
        aux_preupsample = self.aux_classifier(feats["layer3"])
        aux_logits = F.interpolate(aux_preupsample, size=input_size, mode="bilinear", align_corners=False)
        features = {
            "context": context,
            "decoder": logits_preupsample,
            "high_level": feats["layer4"],
            "low_level": feats["layer1"],
        }
        if not return_aux:
            if return_preupsample:
                return (logits, logits_preupsample, features) if return_features else (logits, logits_preupsample)
            return (logits, features) if return_features else logits
        if return_preupsample:
            if return_features:
                return logits, aux_logits, logits_preupsample, aux_preupsample, features
            return logits, aux_logits, logits_preupsample, aux_preupsample
        if return_features:
            return logits, aux_logits, features
        return logits, aux_logits
