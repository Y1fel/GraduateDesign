from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormReLU(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int,
        s: int = 1,
        p: int = 0,
        d: int = 1,
    ):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class LearnableUpsampleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pre = ConvNormReLU(channels, channels, k=1)
        self.post = ConvNormReLU(channels, channels, k=3, p=1)

    def forward(self, x: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        x = self.pre(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return self.post(x)


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        low_level_in_channels: int,
        aspp_out_channels: int = 256,
        low_level_out_channels: int = 48,
        decoder_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.low_reduce = ConvNormReLU(
            low_level_in_channels, low_level_out_channels, k=1
        )
        self.aspp_upsample = LearnableUpsampleBlock(aspp_out_channels)

        in_ch = aspp_out_channels + low_level_out_channels
        self.refine = nn.Sequential(
            ConvNormReLU(in_ch, decoder_channels, k=3, p=1),
            ConvNormReLU(decoder_channels, decoder_channels, k=3, p=1),
            nn.Dropout(p=dropout),
        )

    def forward(self, low_level: torch.Tensor, aspp_feat: torch.Tensor):
        low = self.low_reduce(low_level)
        out_size = low.shape[-2:]
        aspp_up = self.aspp_upsample(aspp_feat, out_size)

        x = torch.cat([aspp_up, low], dim=1)
        return self.refine(x)
