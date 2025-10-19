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
    # 明确两路输入通道数：主图(灰度或RGB)与条件灰度图
    main_in_channels: int = 3
    cond_in_channels: int = 1


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
        # 使用 GroupNorm 代替 BatchNorm2d，以兼容 batch=1 且空间为 1x1 的情况
        # 选择能整除 mid_channels 的最大分组数，保证有效分组
        gn_groups = 1
        for g in (32, 16, 8, 4, 2, 1):
            if mid_channels % g == 0:
                gn_groups = g
                break
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        y = self.fc(x)
        z = y.flatten(1)
        return z

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
        total_in = int(cfg.main_in_channels) + int(cfg.cond_in_channels)
        if cfg.backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if cfg.pretrained else None
            base = resnet18(weights=weights)
            # 调整conv1以适配任意输入通道
            if total_in != 3:
                old_conv1 = base.conv1
                new_conv1 = nn.Conv2d(total_in, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                                       stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
                with torch.no_grad():
                    if old_conv1.weight.shape[1] == 3:
                        # 将预训练权重映射到新通道：前3通道拷贝，其余通道取均值
                        new_conv1.weight[:, :3] = old_conv1.weight
                        if total_in > 3:
                            mean_w = old_conv1.weight.mean(dim=1, keepdim=True)
                            new_conv1.weight[:, 3:total_in] = mean_w.repeat(1, total_in - 3, 1, 1)
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
            self.unet = UNetSmall(in_channels=total_in, base_ch=64)
            self.label_head_unet = LabelHead(self.unet.out_ch)
            self._is_unet = True
        else:
            raise ValueError(f"Unsupported backbone: {cfg.backbone}")

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 接受两路输入；为兼容旧用法，若cond为None则认为x已是拼接后的总通道输入
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
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
    # 读取输入张量的批大小、通道数与空间尺寸 [B, C, H, W]
    B, C, H, W = logits.shape
    # 计算按 stride 划分后的网格大小（cell 网格的高宽）
    Hc = H // stride
    Wc = W // stride
    # 计算裁剪后的有效像素尺寸，使其恰好能被 stride 整除
    Ht = Hc * stride
    Wt = Wc * stride
    # 返回裁剪到 [Ht, Wt] 的 logits 以及 cell 网格尺寸 Hc、Wc
    return logits[:, :, :Ht, :Wt], Hc, Wc


def _cell_logsumexp(logits: torch.Tensor, stride: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    # 先裁剪 logits 到能够整除 stride 的有效区域，并得到 cell 网格大小
    logits, Hc, Wc = _crop_to_stride(logits, stride)
    # 读取裁剪后张量的形状（Ht, Wt 是有效空间分辨率）
    B, _, Ht, Wt = logits.shape
    s = stride
    # 使用 unfold 将每个不重叠的 s×s 区域拉平成长度为 s*s 的列，形成 patches
    # 输出形状为 [B, s*s, Hc*Wc]，其中 Hc*Wc 表示所有 cell 的数目
    patches = F.unfold(logits, kernel_size=(s, s), stride=(s, s))  # [B, s*s, Hc*Wc]
    # 对每个 cell（每一列）在像素维度上做 log-sum-exp 聚合，得到 cell 级别的 log 质量
    lse = torch.logsumexp(patches, dim=1)  # [B, Hc*Wc]
    # 还原为网格形状 [B, 1, Hc, Wc] 以便后续按网格进行采样与索引
    lse_map = lse.view(B, 1, Hc, Wc)
    # 返回：cell 级 logits 图（lse_map）、原始 cell 内像素 logits（patches）、以及网格大小
    return lse_map, patches, Hc, Wc


@torch.no_grad()
def sample_cell_and_offset(
    logits: torch.Tensor,
    stride: int,
    temperature: float = 1.0,
) -> dict:
    # 采样辅助：从像素 logits 中分两级（cell 与 cell 内像素）采样，并返回坐标、偏移与对数概率
    device = logits.device
    # 温度下限保护，避免数值不稳定；随后将 logits 除以温度以控制分布陡峭程度
    T = max(temperature, 1e-6)
    # 计算 cell 级别的 log-sum-exp 地图与每个 cell 内像素的展开表示
    lse_map, patches, Hc, Wc = _cell_logsumexp(logits / T, stride)
    # 批大小 B 与 stride 的别名 s
    B = logits.size(0)
    s = stride

    # 先在 cell 网格上采样：p(cell)
    # 将 [B,1,Hc,Wc] 展平成 [B, Hc*Wc] 作为 Categorical 的 logits 输入
    cell_logits_flat = lse_map.view(B, -1).float()
    cell_dist = torch.distributions.Categorical(logits=cell_logits_flat)
    # 采样得到每个样本的 cell 索引（扁平索引）与对应的 log 概率
    cell_idx = cell_dist.sample()                       # [B]
    cell_logp = cell_dist.log_prob(cell_idx)            # [B]

    # 在选中的 cell 内再次采样子像素：p(subpixel | cell)
    # 先用 gather 取出被选中 cell 的长度为 s*s 的像素 logits 列向量
    within_logits = patches.gather(2, cell_idx.view(B, 1, 1).expand(B, s*s, 1)).squeeze(-1).float()
    # 基于该列构造条件分布并采样像素内索引 sub_idx（范围 [0, s*s)）以及其对数概率
    sub_dist = torch.distributions.Categorical(logits=within_logits)
    sub_idx = sub_dist.sample()
    sub_logp = sub_dist.log_prob(sub_idx)

    # 将扁平索引还原为二维坐标
    ci = (cell_idx // Wc)              # [B] 选中 cell 的行索引（在 Hc 维度）
    cj = (cell_idx % Wc)               # [B] 选中 cell 的列索引（在 Wc 维度）
    ui = (sub_idx // s)                # [B] cell 内的行偏移（0..s-1）
    uj = (sub_idx % s)                 # [B] cell 内的列偏移（0..s-1）

    # 合成裁剪区域内的像素整点坐标（以像素为单位）
    yi = ci * s + ui
    xj = cj * s + uj

    # 提供归一化到 [-0.5, 0.5] 的亚像素偏移，中心对齐，便于连续坐标建模
    off_x = (uj.to(torch.float32) + 0.5) / s - 0.5
    off_y = (ui.to(torch.float32) + 0.5) / s - 0.5

    # 联合对数概率：log p(cell) + log p(subpixel | cell)
    total_logp = cell_logp + sub_logp

    # 返回包含索引、坐标、偏移以及对数概率等信息的字典
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
    # 标签分布采样：对 label_logits 施加温度缩放并构造类别分布
    T_label = max(temperature_label, 1e-6)
    label_dist = torch.distributions.Categorical(logits=(label_logits / T_label).float())
    # 采样得到每个样本的标签索引与其对数概率
    label_idx = label_dist.sample()
    logp_label = label_dist.log_prob(label_idx)

    # 像素分布采样：调用两级像素采样（cell + 子像素），带像素温度
    pixel = sample_cell_and_offset(logits, stride=stride, temperature=temperature_pixel)
    # 取得像素部分的联合对数概率（cell + subpixel）
    logp_pixel = pixel["log_prob"]

    # 汇总输出，将标签和像素的采样结果与对数概率拼装在一个字典中
    out = dict(pixel)
    out["label_idx"] = label_idx
    out["log_prob_label"] = logp_label
    out["log_prob_pixel"] = logp_pixel
    # 联合动作的总 log_prob = 标签部分 + 像素部分
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

