import torch
from torch import nn

from .CBAM import CBAM
from .LWT import LearnableWaveletTransformReg


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.fc(self.avg_pool(x).flatten(1))
        return x * weights[:, :, None, None]


class WAFM(nn.Module):
    """Approximation averaging, adaptive detail fusion, and encoder residual."""

    def __init__(self, channels, reduction=16, learnable=True, adaptive=True):
        super().__init__()
        self.lwt = LearnableWaveletTransformReg(channels, learnable)
        detail_channels = 3 * channels
        hidden = max(1, detail_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv1d(detail_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, detail_channels, 1),
            nn.Sigmoid(),
        ) if adaptive else None
        self.cbam = CBAM(detail_channels)
        self.se = SqueezeExcitation(channels, reduction)

    def forward(self, x1, x2):
        ll1, hf1 = self.lwt(self.se(x1))
        ll2, hf2 = self.lwt(self.se(x2))
        if self.mlp is None:
            hf = (hf1 + hf2) / 2
        else:
            differences = (hf1 - hf2).abs().mean(dim=(2, 3))
            weights = self.mlp(differences.unsqueeze(-1)).unsqueeze(-1)
            hf = weights * hf1 + (1 - weights) * hf2
        fused = self.lwt.synthesis((ll1 + ll2) / 2, self.cbam(hf))
        return fused + (x1 + x2) / 2
