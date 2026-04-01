from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SpatialGatherModule(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, feats: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        batch_size, feat_channels, _, _ = feats.shape
        _, num_classes, _, _ = probs.shape

        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, feat_channels, -1).permute(0, 2, 1)

        probs = F.softmax(self.scale * probs, dim=2)
        context = torch.matmul(probs, feats)
        return context.permute(0, 2, 1).unsqueeze(-1).contiguous()


class ObjectAttentionBlock2D(nn.Module):
    def __init__(self, in_channels: int, key_channels: int) -> None:
        super().__init__()
        self.key_channels = int(key_channels)

        self.query_proj = nn.Sequential(
            ConvBNReLU(in_channels, key_channels, kernel_size=1),
            ConvBNReLU(key_channels, key_channels, kernel_size=1),
        )
        self.key_proj = nn.Sequential(
            ConvBNReLU(in_channels, key_channels, kernel_size=1),
            ConvBNReLU(key_channels, key_channels, kernel_size=1),
        )
        self.value_proj = ConvBNReLU(in_channels, key_channels, kernel_size=1)
        self.out_proj = ConvBNReLU(key_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, proxy: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape

        query = self.query_proj(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        key = self.key_proj(proxy).view(batch_size, self.key_channels, -1)
        value = self.value_proj(proxy).view(batch_size, self.key_channels, -1).permute(0, 2, 1)

        sim_map = torch.matmul(query, key) * (self.key_channels ** -0.5)
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous().view(batch_size, self.key_channels, height, width)
        return self.out_proj(context)


class SpatialOCRModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        key_channels: int,
        out_channels: int,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.object_context = ObjectAttentionBlock2D(
            in_channels=in_channels,
            key_channels=key_channels,
        )
        self.conv_bn_dropout = nn.Sequential(
            ConvBNReLU(in_channels * 2, out_channels, kernel_size=1),
            nn.Dropout2d(p=float(dropout)),
        )

    def forward(self, feats: torch.Tensor, proxy_feats: torch.Tensor) -> torch.Tensor:
        context = self.object_context(feats, proxy_feats)
        return self.conv_bn_dropout(torch.cat([context, feats], dim=1))
