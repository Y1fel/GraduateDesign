from __future__ import annotations

import torch
import torch.nn as nn

from src.models.aspp import ASPP


def _prob_to_logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-4), 1.0 - 1e-4)
    return float(torch.logit(torch.tensor(prob)).item())


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwisePointwiseBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int | tuple[int, int],
        padding: int | tuple[int, int],
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
            ),
            ConvBNReLU(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LargeKernelContextBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        pad = kernel_size // 2
        self.pre = ConvBNReLU(in_channels, hidden_channels, kernel_size=1)
        self.large_kernel = DepthwisePointwiseBlock(
            hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
        )
        self.project = nn.Sequential(
            ConvBN(hidden_channels * 2, out_channels, kernel_size=1),
            nn.Dropout2d(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.pre(x)
        large = self.large_kernel(base)
        return self.project(torch.cat([base, large], dim=1))


class StripContextBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        strip_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if strip_kernel % 2 == 0:
            raise ValueError(f"strip_kernel must be odd, got {strip_kernel}")

        pad = strip_kernel // 2
        self.pre = ConvBNReLU(in_channels, hidden_channels, kernel_size=1)
        self.horizontal = DepthwisePointwiseBlock(
            hidden_channels,
            kernel_size=(1, strip_kernel),
            padding=(0, pad),
        )
        self.vertical = DepthwisePointwiseBlock(
            hidden_channels,
            kernel_size=(strip_kernel, 1),
            padding=(pad, 0),
        )
        self.project = nn.Sequential(
            ConvBN(hidden_channels * 3, out_channels, kernel_size=1),
            nn.Dropout2d(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.pre(x)
        horiz = self.horizontal(base)
        vert = self.vertical(base)
        return self.project(torch.cat([base, horiz, vert], dim=1))


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduction = max(int(reduction), 1)
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class HybridContextNeck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: tuple[int, int, int],
        use_strip: bool = False,
        strip_kernel: int = 11,
        mid_kernel: int = 7,
        large_kernel: int = 15,
        gate_reduction: int = 16,
        residual_channels: int = 128,
        residual_init: float = 0.02,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.use_strip = bool(use_strip)
        if self.use_strip and strip_kernel % 2 == 0:
            raise ValueError(f"strip_kernel must be odd, got {strip_kernel}")
        if mid_kernel % 2 == 0:
            raise ValueError(f"mid_kernel must be odd, got {mid_kernel}")
        if large_kernel % 2 == 0:
            raise ValueError(f"large_kernel must be odd, got {large_kernel}")
        if int(mid_kernel) >= int(large_kernel):
            raise ValueError(
                f"mid_kernel must be smaller than large_kernel, got mid_kernel={mid_kernel}, large_kernel={large_kernel}"
            )

        self.aspp = ASPP(
            in_channels=in_channels,
            out_channels=out_channels,
            atrous_rates=atrous_rates,
            dropout=dropout,
        )
        residual_channels = max(int(residual_channels), 64)
        self.strip_branch = (
            StripContextBranch(
                in_channels=out_channels,
                hidden_channels=residual_channels,
                out_channels=out_channels,
                strip_kernel=strip_kernel,
                dropout=dropout,
            )
            if self.use_strip
            else None
        )
        self.mid_kernel_branch = LargeKernelContextBranch(
            in_channels=out_channels,
            hidden_channels=residual_channels,
            out_channels=out_channels,
            kernel_size=mid_kernel,
            dropout=dropout,
        )
        self.large_kernel_branch = LargeKernelContextBranch(
            in_channels=out_channels,
            hidden_channels=residual_channels,
            out_channels=out_channels,
            kernel_size=large_kernel,
            dropout=dropout,
        )
        self.strip_gate = ChannelGate(out_channels, reduction=gate_reduction) if self.use_strip else None
        self.mid_gate = ChannelGate(out_channels, reduction=gate_reduction)
        self.large_gate = ChannelGate(out_channels, reduction=gate_reduction)
        init_logit = _prob_to_logit(residual_init)
        self.strip_scale_logit = nn.Parameter(torch.tensor(init_logit)) if self.use_strip else None
        self.mid_scale_logit = nn.Parameter(torch.tensor(init_logit))
        self.large_scale_logit = nn.Parameter(torch.tensor(init_logit))
        self.refine = nn.Sequential(
            ConvBNReLU(out_channels, out_channels, kernel_size=1),
            nn.Dropout2d(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aspp_feat = self.aspp(x)
        enhanced = aspp_feat
        if self.strip_branch is not None and self.strip_gate is not None and self.strip_scale_logit is not None:
            strip_feat = self.strip_gate(self.strip_branch(aspp_feat))
            strip_scale = torch.sigmoid(self.strip_scale_logit)
            enhanced = enhanced + strip_scale * strip_feat
        mid_feat = self.mid_gate(self.mid_kernel_branch(aspp_feat))
        large_feat = self.large_gate(self.large_kernel_branch(aspp_feat))
        mid_scale = torch.sigmoid(self.mid_scale_logit)
        large_scale = torch.sigmoid(self.large_scale_logit)
        enhanced = enhanced + mid_scale * mid_feat + large_scale * large_feat
        return self.refine(enhanced)
