from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
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


class ChannelReweight(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.mlp(w)
        return x * w


class SpatialReweight(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg, max_val], dim=1))
        return x * attn


class ContextEnhancementBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        dilations: tuple[int, int] = (3, 6),
        dropout: float = 0.1,
    ):
        super().__init__()
        d1, d2 = dilations

        self.local_branch = ConvBNReLU(channels, channels, k=3, p=1)
        self.large_rf_branch = nn.Sequential(
            ConvBNReLU(channels, channels, k=3, p=d1, d=d1),
            ConvBNReLU(channels, channels, k=3, p=d2, d=d2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = ConvBNReLU(channels, channels, k=1)

        self.fuse = nn.Sequential(
            ConvBNReLU(channels * 3, channels, k=1),
            nn.Dropout(p=dropout),
        )
        self.channel_reweight = ChannelReweight(channels=channels, reduction=reduction)
        self.spatial_reweight = SpatialReweight(kernel_size=7)

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        h, w = x.shape[-2:]

        local_feat = self.local_branch(x)
        large_rf_feat = self.large_rf_branch(x)

        global_feat = self.global_pool(x)
        global_feat = self.global_proj(global_feat)
        global_feat = F.interpolate(global_feat, size=(h, w), mode="bilinear", align_corners=False)

        fused = torch.cat([local_feat, large_rf_feat, global_feat], dim=1)
        fused = self.fuse(fused)
        fused = self.channel_reweight(fused)
        fused = self.spatial_reweight(fused)
        fused = self.out_proj(fused)

        return self.act(identity + fused)
