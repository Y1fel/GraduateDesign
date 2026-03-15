import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.mobilenetv2 import InvertedResidual


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
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

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x