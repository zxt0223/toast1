from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int, preferred: int = 8) -> int:
    """选择能够整除通道数的 GroupNorm 分组数。"""
    for groups in (preferred, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        layers = [
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
            nn.GroupNorm(_num_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class PathologyResidualBlock(nn.Module):
    """
    病理声学双分支残差块。

    transient 分支：较宽频率感受野、较短时间感受野，偏向 Crackle；
    sustained 分支：时间方向长卷积与扩张，偏向 Wheeze。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int] = (1, 1),
        time_dilation: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        half = out_channels // 2

        self.transient = nn.Sequential(
            ConvGNAct(
                in_channels,
                half,
                kernel_size=(5, 3),
                stride=stride,
                padding=(2, 1),
            ),
            ConvGNAct(
                half,
                half,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
        )

        self.sustained = nn.Sequential(
            ConvGNAct(
                in_channels,
                half,
                kernel_size=(3, 7),
                stride=stride,
                padding=(1, 3 * time_dilation),
                dilation=(1, time_dilation),
            ),
            ConvGNAct(
                half,
                half,
                kernel_size=(3, 5),
                padding=(1, 2 * time_dilation),
                dilation=(1, time_dilation),
            ),
        )

        self.fusion = ConvGNAct(
            out_channels,
            out_channels,
            kernel_size=1,
            activation=False,
        )

        if in_channels != out_channels or stride != (1, 1):
            self.shortcut = ConvGNAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                activation=False,
            )
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        merged = torch.cat([self.transient(x), self.sustained(x)], dim=1)
        merged = self.dropout(self.fusion(merged))
        return self.activation(merged + residual)


class FrequencyBandGate(nn.Module):
    """根据当前样本自适应强调有效频带。"""

    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=5, padding=2, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_channels, 1, kernel_size=5, padding=2, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, F, T] -> [B, 1, F]
        descriptor = x.mean(dim=(1, 3), keepdim=False).unsqueeze(1)
        gate = torch.sigmoid(self.net(descriptor)).unsqueeze(-1)
        # 0.5~1.5 的残差式门控，避免完全抹掉某个频带。
        return x * (0.5 + gate)


class MultiHeadAttentiveStatsPool(nn.Module):
    """多头时间注意力统计池化，保留短暂事件和持续事件。"""

    def __init__(self, channels: int, num_heads: int = 2) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        hidden = max(32, channels // 4)
        self.attention = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, num_heads, kernel_size=1),
        )

    @property
    def output_dim(self) -> int:
        # 每个 head 输出 mean + std，再拼接一个全局 max。
        return self.channels * (2 * self.num_heads + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先聚合频率，保留时间序列：[B, C, T]
        sequence = x.mean(dim=2)
        weights = torch.softmax(self.attention(sequence), dim=-1)  # [B,H,T]

        expanded = sequence.unsqueeze(1)  # [B,1,C,T]
        weights_expanded = weights.unsqueeze(2)  # [B,H,1,T]

        mean = (weights_expanded * expanded).sum(dim=-1)  # [B,H,C]
        second = (weights_expanded * expanded.square()).sum(dim=-1)
        variance = (second - mean.square()).clamp_min(1e-5)
        std = variance.sqrt()

        maximum = sequence.amax(dim=-1)  # [B,C]
        return torch.cat(
            [mean.flatten(1), std.flatten(1), maximum],
            dim=1,
        )


class HierarchicalRespiratoryNet(nn.Module):
    """
    HATS-Net：Hierarchical Abnormality-Then-Subtype Network。

    最终决策分两步：
      1) Normal / Abnormal；
      2) 若为 Abnormal，再预测 Crackle / Wheeze / Both。

    auxiliary_head 只在训练时辅助四分类表征，不参与最终决策。
    """

    def __init__(
        self,
        input_channels: int = 2,
        feature_channels: int = 256,
        embedding_dim: int = 384,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvGNAct(
                input_channels,
                32,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
            ),
            PathologyResidualBlock(32, 64, time_dilation=1, dropout=0.03),
        )

        self.stage1 = nn.Sequential(
            PathologyResidualBlock(64, 96, stride=(2, 2), time_dilation=2, dropout=0.05),
            PathologyResidualBlock(96, 96, time_dilation=2, dropout=0.05),
        )
        self.freq_gate1 = FrequencyBandGate(hidden_channels=12)

        self.stage2 = nn.Sequential(
            PathologyResidualBlock(96, 160, stride=(2, 2), time_dilation=3, dropout=0.08),
            PathologyResidualBlock(160, 160, time_dilation=4, dropout=0.08),
        )

        self.stage3 = nn.Sequential(
            PathologyResidualBlock(
                160,
                feature_channels,
                stride=(2, 1),
                time_dilation=5,
                dropout=0.10,
            ),
            PathologyResidualBlock(
                feature_channels,
                feature_channels,
                time_dilation=6,
                dropout=0.10,
            ),
        )
        self.freq_gate2 = FrequencyBandGate(hidden_channels=16)

        self.pool = MultiHeadAttentiveStatsPool(feature_channels, num_heads=2)
        pooled_dim = self.pool.output_dim

        self.embedding = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
        )

        self.abnormal_head = nn.Linear(embedding_dim, 1)
        self.subtype_head = nn.Linear(embedding_dim, 3)
        self.auxiliary_head = nn.Linear(embedding_dim, 4)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.freq_gate1(self.stage1(x))
        x = self.stage2(x)
        x = self.freq_gate2(self.stage3(x))
        return self.embedding(self.pool(x))

    def forward(self, x: torch.Tensor):
        embedding = self.extract_embedding(x)
        abnormal_logit = self.abnormal_head(embedding).squeeze(1)
        subtype_logits = self.subtype_head(embedding)
        auxiliary_logits = self.auxiliary_head(embedding)
        return abnormal_logit, subtype_logits, auxiliary_logits

    @staticmethod
    def hierarchical_probabilities(
        abnormal_logit: torch.Tensor,
        subtype_logits: torch.Tensor,
    ) -> torch.Tensor:
        p_abnormal = torch.sigmoid(abnormal_logit).unsqueeze(1)
        subtype_probs = torch.softmax(subtype_logits, dim=1)
        return torch.cat(
            [1.0 - p_abnormal, p_abnormal * subtype_probs],
            dim=1,
        )


# 兼容旧代码可能使用的名称。
InnovativeResNet = HierarchicalRespiratoryNet