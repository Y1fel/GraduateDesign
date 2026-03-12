import torch
import torch.nn as nn
from torchvision.models import (
    ResNet101_Weights,
    ResNet50_Weights,
    resnet101,
    resnet50,
)


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, output_stride: int = 16, backbone_name: str = "rsnet-50"):
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError("output_stride must be 8 or 16")

        if output_stride == 16:
            replace = [False, False, True]
        else:                      
            replace = [False, True, True]

        backbone_key = str(backbone_name).lower().replace("_", "-")
        if backbone_key == "resnet-50":
            backbone_key = "rsnet-50"
        if backbone_key in {"resnet-100", "rsnet-100"}:
            backbone_key = "rs-net-100"

        builders = {
            "rsnet-50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
                                                              
            "rs-net-100": (resnet101, ResNet101_Weights.IMAGENET1K_V2),
        }
        if backbone_key not in builders:
            supported = ", ".join(builders.keys())
            raise ValueError(f"Unsupported backbone_name: {backbone_name}. Use one of: {supported}")

        builder, default_weights = builders[backbone_key]
        weights = default_weights if pretrained else None
        m = builder(weights=weights, replace_stride_with_dilation=replace)

              
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool

                
        self.layer1 = m.layer1            
        self.layer2 = m.layer2            
        self.layer3 = m.layer3               
        self.layer4 = m.layer4               

        self.out_channels = 2048
        self.low_level_channels = 256

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        low_level = x1                      

        x2 = self.layer2(x1)                      
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)                         

        return low_level, x2, x4
