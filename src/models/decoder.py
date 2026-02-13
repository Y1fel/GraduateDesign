from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.switch2Norm import NormType, make_norm


class ConvNormReLU(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int,
        s: int = 1,
        p: int = 0,
        d: int = 1,
        norm: NormType = "bn",
        num_groups: int = 32,
    ):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False),
            make_norm(norm, out_ch, num_groups=num_groups),
            nn.ReLU(inplace=True),
        )


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        low_level_in_channels: int,
        aspp_out_channels: int = 256,
        low_level_out_channels: int = 48,
        decoder_channels: int = 256,
        dropout: float = 0.1,
        norm: NormType = "bn",
        num_groups: int = 32,
    ):
        super().__init__()
        self.low_reduce = ConvNormReLU(
            low_level_in_channels, low_level_out_channels, k=1, norm=norm, num_groups=num_groups
        )

        in_ch = aspp_out_channels + low_level_out_channels
        self.refine = nn.Sequential(
            ConvNormReLU(in_ch, decoder_channels, k=3, p=1, norm=norm, num_groups=num_groups),
            ConvNormReLU(decoder_channels, decoder_channels, k=3, p=1, norm=norm, num_groups=num_groups),
            nn.Dropout(p=dropout),
        )

    def forward(self, low_level: torch.Tensor, aspp_feat: torch.Tensor):
        low = self.low_reduce(low_level)
        aspp_up = F.interpolate(aspp_feat, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([aspp_up, low], dim=1)
        return self.refine(x)
