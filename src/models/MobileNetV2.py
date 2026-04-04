import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.models import mobilenet_v2

try:
    from torchvision.models import MobileNet_V2_Weights
except ImportError:  # torchvision<0.13
    MobileNet_V2_Weights = None


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, pretrained: bool = False):
        super().__init__()
        cfg = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        layers = []

        input_channel = 32
        layers.append(
            nn.Sequential(
                nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True)
            )
        )

        for t, c, n, s in cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    InvertedResidual(
                        input_channel,
                        c,
                        stride,
                        t
                    )
                )

                input_channel = c
                   
        layers.append(
            nn.Sequential(
                nn.Conv2d(input_channel, 1280, 1, bias=False),
                nn.BatchNorm2d(1280),
                nn.ReLU6(inplace=True)
            )
        )

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

        if pretrained:
            self._load_imagenet_pretrained()

    def _load_imagenet_pretrained(self) -> None:
        if MobileNet_V2_Weights is not None:
            ref_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            ref_model = mobilenet_v2(pretrained=True)

        ref_state = ref_model.state_dict()
        cur_state = self.state_dict()
        matched_state = {
            key: value
            for key, value in ref_state.items()
            if key in cur_state and tuple(cur_state[key].shape) == tuple(value.shape)
        }
        cur_state.update(matched_state)
        self.load_state_dict(cur_state, strict=False)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
