import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetBackbone(nn.Module):
    """
    ResNet backbone for DeepLabv3+.
    Returns:
      - low_level: feature map at stride 4 (layer1 output)
      - high_level: feature map at stride 16 or 8 (layer4 output, with dilation)
    """
    def __init__(self, pretrained: bool = True, output_stride: int = 16):
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError("output_stride must be 8 or 16")

        # For torchvision resnet:
        # replace_stride_with_dilation has 3 elems for layer2, layer3, layer4
        # Default strides: layer2(stride=2), layer3(stride=2), layer4(stride=2)
        # To get OS=16: keep layer2 stride, keep layer3 stride, dilate layer4 (no stride)
        # To get OS=8: keep layer2 stride, dilate layer3, dilate layer4
        if output_stride == 16:
            replace = [False, False, True]
        else:  # output_stride == 8
            replace = [False, True, True]

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = resnet50(weights=weights, replace_stride_with_dilation=replace)

        # Stem
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool

        # Stages
        self.layer1 = m.layer1  # stride 4
        self.layer2 = m.layer2  # stride 8 (kept)
        self.layer3 = m.layer3  # stride 16 or 8 (if dilated)
        self.layer4 = m.layer4  # stride 16 or 8 (dilated)

        self.out_channels = 2048  # resnet50 layer4 channels
        self.low_level_channels = 256  # layer1 output channels

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level = x  # (N, 256, H/4, W/4)

        x = self.layer2(x)
        x = self.layer3(x)
        high_level = self.layer4(x)  # (N, 2048, H/OS, W/OS)

        return low_level, high_level