from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


@dataclass
class ModelConfig:
    """模型配置
    
    backbone: 选择骨干架构
        - "unet_s": 轻量UNet，默认推荐，输出高分辨率平滑热力图，适合软分布学习
        - "resnet18": ResNet18 + 插值上采样，速度快但空间精度略低
    pretrained: 是否使用预训练权重（仅对resnet18有效）
    upsample_to_input: resnet18时是否上采样到输入分辨率
    """
    backbone: Literal["resnet18", "unet_s"] = "unet_s"
    pretrained: bool = False
    upsample_to_input: bool = True
    in_channels: int = 3  # 支持RGB(3)+Gray(1)成对输入，默认4通道


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


class LabelHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 256, num_classes: int = 2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.fc(x).flatten(1)

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSmall(nn.Module):
    """轻量级U-Net，输出输入分辨率的单通道logits热力图
    
    设计用于生成平滑、空间精确的热力图，通过跳连保持细节信息。
    相比ResNet+上采样，能更好地学习软分布，避免插值带来的空间失真。
    特别适合配合高斯软目标(KL/MSE)训练距离衰减的热力图。
    
    参数:
        in_channels: 输入通道数，默认3(RGB)
        base_ch: 基础通道数，控制模型容量，默认64
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 64) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.out_ch = c1
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c2, c1)

        self.final = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        # expose last decoder feature for classification head
        self.last_dec = d1
        return self.final(d1)


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
            # 调整conv1以适配任意输入通道
            if cfg.in_channels != 3:
                old_conv1 = base.conv1
                new_conv1 = nn.Conv2d(cfg.in_channels, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                                       stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
                with torch.no_grad():
                    if old_conv1.weight.shape[1] == 3:
                        # 将预训练权重映射到新通道：前3通道拷贝，其余通道取均值
                        new_conv1.weight[:, :3] = old_conv1.weight
                        if cfg.in_channels > 3:
                            mean_w = old_conv1.weight.mean(dim=1, keepdim=True)
                            new_conv1.weight[:, 3:cfg.in_channels] = mean_w.repeat(1, cfg.in_channels - 3, 1, 1)
                    else:
                        nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
                base.conv1 = new_conv1
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2  # stride 8 total
            self.layer3 = base.layer3  # stride 16
            self.layer4 = base.layer4  # stride 32
            in_channels = 512
            self.head = HeatmapHead(in_channels)
            self.label_head = LabelHead(in_channels)
            self._is_unet = False
        elif cfg.backbone == "unet_s":
            self.unet = UNetSmall(in_channels=cfg.in_channels, base_ch=64)
            self.label_head_unet = LabelHead(self.unet.out_ch)
            self._is_unet = True
        else:
            raise ValueError(f"Unsupported backbone: {cfg.backbone}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        if getattr(self, "_is_unet", False):
            logits = self.unet(x)
            # use last decoder feature for label
            feat = getattr(self.unet, "last_dec", None)
            if feat is None:
                # fallback to using input averaged with logits if feature missing
                feat = x
            label_logits = self.label_head_unet(feat)
            return logits, label_logits
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits_low = self.head(x)  # [B,1,h',w']
        label_logits = self.label_head(x)  # [B,2]
        if self.cfg.upsample_to_input:
            logits = F.interpolate(logits_low, size=(h, w), mode="bilinear", align_corners=False)
            return logits, label_logits
        return logits_low, label_logits


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


# ==========================
# 层级策略采样与log_prob计算
# 将像素分布分解为：cell分布 + cell内子像素分布
# ==========================

def _crop_to_stride(logits: torch.Tensor, stride: int) -> tuple[torch.Tensor, int, int]:
    B, C, H, W = logits.shape
    Hc = H // stride
    Wc = W // stride
    Ht = Hc * stride
    Wt = Wc * stride
    return logits[:, :, :Ht, :Wt], Hc, Wc


def _cell_logsumexp(logits: torch.Tensor, stride: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    logits, Hc, Wc = _crop_to_stride(logits, stride)
    B, _, Ht, Wt = logits.shape
    s = stride
    patches = F.unfold(logits, kernel_size=(s, s), stride=(s, s))  # [B, s*s, Hc*Wc]
    lse = torch.logsumexp(patches, dim=1)  # [B, Hc*Wc]
    lse_map = lse.view(B, 1, Hc, Wc)
    return lse_map, patches, Hc, Wc


@torch.no_grad()
def sample_cell_and_offset(
    logits: torch.Tensor,
    stride: int,
    temperature: float = 1.0,
) -> dict:
    device = logits.device
    T = max(temperature, 1e-6)
    lse_map, patches, Hc, Wc = _cell_logsumexp(logits / T, stride)
    B = logits.size(0)
    s = stride

    # p(cell)
    cell_logits_flat = lse_map.view(B, -1)
    cell_dist = torch.distributions.Categorical(logits=cell_logits_flat)
    cell_idx = cell_dist.sample()                       # [B]
    cell_logp = cell_dist.log_prob(cell_idx)            # [B]

    # p(subpixel | cell)
    within_logits = patches.gather(2, cell_idx.view(B, 1, 1).expand(B, s*s, 1)).squeeze(-1)
    sub_dist = torch.distributions.Categorical(logits=within_logits)
    sub_idx = sub_dist.sample()
    sub_logp = sub_dist.log_prob(sub_idx)

    # index -> (i,j)
    ci = (cell_idx // Wc)              # [B]
    cj = (cell_idx % Wc)               # [B]
    ui = (sub_idx // s)
    uj = (sub_idx % s)

    # pixel coordinates within cropped region
    yi = ci * s + ui
    xj = cj * s + uj

    # offset in [-0.5, 0.5]
    off_x = (uj.to(torch.float32) + 0.5) / s - 0.5
    off_y = (ui.to(torch.float32) + 0.5) / s - 0.5

    total_logp = cell_logp + sub_logp

    return {
        "cell_idx": cell_idx,
        "cell_ij": torch.stack([cj, ci], dim=1),
        "sub_idx": sub_idx,
        "sub_ij": torch.stack([uj, ui], dim=1),
        "pixel_xy": torch.stack([xj, yi], dim=1),
        "offset": torch.stack([off_x, off_y], dim=1).to(device),
        "log_prob": total_logp,
        "grid_hw": torch.tensor([Hc, Wc], device=device),
        "stride": torch.tensor(s, device=device),
    }


def action_to_continuous_xy(sample_out: dict) -> torch.Tensor:
    s = int(sample_out["stride"].item())
    x = sample_out["pixel_xy"][:, 0].to(torch.float32) + 0.5
    y = sample_out["pixel_xy"][:, 1].to(torch.float32) + 0.5
    return torch.stack([x, y], dim=1)


def log_prob_of_action(
    logits: torch.Tensor,
    stride: int,
    cell_idx: torch.Tensor,
    sub_idx: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    T = max(temperature, 1e-6)
    lse_map, patches, Hc, Wc = _cell_logsumexp(logits / T, stride)
    B = logits.size(0)
    s = stride

    cell_logits_flat = lse_map.view(B, -1)
    cell_logp = torch.log_softmax(cell_logits_flat, dim=1).gather(1, cell_idx.view(B, 1)).squeeze(1)

    within_logits = patches.gather(2, cell_idx.view(B, 1, 1).expand(B, s*s, 1)).squeeze(-1)
    sub_logp = torch.log_softmax(within_logits, dim=1).gather(1, sub_idx.view(B, 1)).squeeze(1)

    return cell_logp + sub_logp


# ==========================
# 联合策略：标签 + cell + cell内子像素
# 提供采样与log_prob计算，便于GRPO等算法直接使用
# ==========================

@torch.no_grad()
def sample_joint_label_cell_offset(
    logits: torch.Tensor,
    label_logits: torch.Tensor,
    stride: int,
    temperature_pixel: float = 1.0,
    temperature_label: float = 1.0,
) -> dict:
    """Sample a joint action (label, cell, subpixel) and return combined log_prob.

    Args:
        logits: [B,1,H,W] heatmap logits over pixels
        label_logits: [B,2] binary label logits
        stride: cell stride size (pixels)
        temperature_pixel: temperature for pixel distribution
        temperature_label: temperature for label distribution
    Returns:
        dict with keys:
          - label_idx: [B]
          - cell_idx, cell_ij, sub_idx, sub_ij, pixel_xy, offset (same as sample_cell_and_offset)
          - log_prob_label: [B]
          - log_prob_pixel: [B]  (cell + sub)
          - log_prob: [B]        (sum)
          - grid_hw, stride
    """
    T_label = max(temperature_label, 1e-6)
    label_dist = torch.distributions.Categorical(logits=label_logits / T_label)
    label_idx = label_dist.sample()
    logp_label = label_dist.log_prob(label_idx)

    pixel = sample_cell_and_offset(logits, stride=stride, temperature=temperature_pixel)
    logp_pixel = pixel["log_prob"]

    out = dict(pixel)
    out["label_idx"] = label_idx
    out["log_prob_label"] = logp_label
    out["log_prob_pixel"] = logp_pixel
    out["log_prob"] = logp_label + logp_pixel
    return out


def log_prob_of_joint_action(
    logits: torch.Tensor,
    label_logits: torch.Tensor,
    stride: int,
    label_idx: torch.Tensor,
    cell_idx: torch.Tensor,
    sub_idx: torch.Tensor,
    temperature_pixel: float = 1.0,
    temperature_label: float = 1.0,
) -> torch.Tensor:
    """Compute log_prob(label) + log_prob(cell) + log_prob(sub|cell) for a given action."""
    # label term
    T_label = max(temperature_label, 1e-6)
    ll = torch.log_softmax(label_logits / T_label, dim=1)
    logp_label = ll.gather(1, label_idx.view(-1, 1)).squeeze(1)

    # pixel terms
    logp_pixel = log_prob_of_action(
        logits=logits,
        stride=stride,
        cell_idx=cell_idx,
        sub_idx=sub_idx,
        temperature=temperature_pixel,
    )
    return logp_label + logp_pixel

