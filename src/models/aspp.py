from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.switch2Norm import NormType, make_norm


class Conv(nn.Sequential):
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


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates=(6, 12, 18),
        dropout: float = 0.1,
        norm: NormType = "bn",
        num_groups: int = 32,
    ):
        super().__init__()
        r1, r2, r3 = atrous_rates

        self.conv = Conv(in_channels, out_channels, k=1, norm=norm, num_groups=num_groups)

        self.conv_1 = Conv(in_channels, out_channels, k=3, p=r1, d=r1, norm=norm, num_groups=num_groups)
        self.conv_2 = Conv(in_channels, out_channels, k=3, p=r2, d=r2, norm=norm, num_groups=num_groups)
        self.conv_3 = Conv(in_channels, out_channels, k=3, p=r3, d=r3, norm=norm, num_groups=num_groups)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(in_channels, out_channels, k=1, norm=norm, num_groups=num_groups),
        )

        self.project = nn.Sequential(
            Conv(out_channels * 5, out_channels, k=1, norm=norm, num_groups=num_groups),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2], x.shape[-1]
        b1 = self.conv(x)
        b2 = self.conv_1(x)
        b3 = self.conv_2(x)
        b4 = self.conv_3(x)

        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)

        y = torch.cat([b1, b2, b3, b4, gp], dim=1)
        y = self.project(y)
        return y
