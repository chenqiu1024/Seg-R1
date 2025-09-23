from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


@dataclass
class ModelConfig:
    backbone: Literal["resnet18"] = "resnet18"
    pretrained: bool = False
    upsample_to_input: bool = True


class HeatmapHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PointHeatmapModel(nn.Module):
    """Backbone + 1-channel heatmap logits head.

    Output: logits of shape [B,1,H',W'] optionally upsampled to input HxW.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if cfg.pretrained else None
            base = resnet18(weights=weights)
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2  # stride 8 total
            self.layer3 = base.layer3  # stride 16
            self.layer4 = base.layer4  # stride 32
            in_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {cfg.backbone}")

        self.head = HeatmapHead(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits_low = self.head(x)  # [B,1,h',w']
        if self.cfg.upsample_to_input:
            logits = F.interpolate(logits_low, size=(h, w), mode="bilinear", align_corners=False)
            return logits
        return logits_low


def argmax_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Get integer pixel coordinates from logits [B,1,H,W] -> [B,2] (x,y)."""
    b, _, h, w = logits.shape
    idx = logits.view(b, -1).argmax(dim=1)
    y = (idx // w).float()
    x = (idx % w).float()
    return torch.stack([x, y], dim=1)


def soft_argmax_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Soft-argmax to get sub-pixel expected coordinates [B,1,H,W] -> [B,2] (x,y)."""
    b, _, h, w = logits.shape
    p = F.softmax(logits.view(b, -1) / max(temperature, 1e-6), dim=1).view(b, 1, h, w)
    xs = torch.linspace(0, w - 1, w, device=logits.device).view(1, 1, 1, w)
    ys = torch.linspace(0, h - 1, h, device=logits.device).view(1, 1, h, 1)
    x_exp = (p * xs).sum(dim=(2, 3))
    y_exp = (p * ys).sum(dim=(2, 3))
    return torch.cat([x_exp, y_exp], dim=1)


