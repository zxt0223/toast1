"""DS-MSAN model used by both training and evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        if in_planes < ratio:
            raise ValueError("in_planes 必须大于或等于 ratio。")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = self.fc(self.avg_pool(inputs))
        maximum = self.fc(self.max_pool(inputs))
        return self.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为奇数。")
        self.conv1 = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = inputs.mean(dim=1, keepdim=True)
        maximum = inputs.amax(dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([average, maximum], dim=1)))


class MultiScalePhysicsBlock(nn.Module):
    """Pathology-inspired two-stream block for transient and sustained sounds."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError("out_channels 必须为偶数。")
        branch_channels = out_channels // 2

        self.crackle_stream = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=(3, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.wheeze_stream = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=(3, 3),
                padding=(2, 2),
                dilation=(2, 2),
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.fusion_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.fusion_bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        merged = torch.cat(
            [self.crackle_stream(inputs), self.wheeze_stream(inputs)],
            dim=1,
        )
        return F.relu(self.fusion_bn(self.fusion_conv(merged)))


class InnovativeResNet(nn.Module):
    """
    Compact DS-MSAN classifier.

    The historical class name is retained so the public 0.6439 checkpoint and
    existing imports remain compatible. The architecture has 421,766 trainable
    parameters and produces four respiratory-cycle logits.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = MultiScalePhysicsBlock(32, 64)
        self.layer2 = MultiScalePhysicsBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.layer3 = MultiScalePhysicsBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.ca = ChannelAttention(256)
        self.sa = SpatialAttention()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.layer1(features)
        features = self.pool2(self.layer2(features))
        features = self.pool3(self.layer3(features))
        features = self.ca(features) * features
        features = self.sa(features) * features
        return self.global_pool(features).flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


# Clear paper-facing name without changing the legacy checkpoint structure.
DSMSAN = InnovativeResNet