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
    
    SAM LoRA 相关配置:
    use_sam_encoder: 是否使用 SAM image encoder 作为特征提取器
    sam_checkpoint: SAM 模型检查点路径
    sam_lora_enabled: 是否启用 Late LoRA 微调
    sam_lora_rank: LoRA 秩（通常为 4-16）
    sam_lora_alpha: LoRA 缩放因子
    sam_lora_dropout: LoRA dropout 概率
    """
    backbone: Literal["resnet18", "unet_s"] = "unet_s"
    pretrained: bool = False
    upsample_to_input: bool = True
    # 明确两路输入通道数：主图(灰度或RGB)与条件灰度图
    main_in_channels: int = 3
    cond_in_channels: int = 1
    
    # SAM Late LoRA 配置
    use_sam_encoder: bool = False
    sam_checkpoint: Optional[str] = None
    sam_lora_enabled: bool = False
    sam_lora_rank: int = 8
    sam_lora_alpha: float = 16.0
    sam_lora_dropout: float = 0.0
    sam_freeze_encoder: bool = True  # 是否冻结 SAM encoder（不启用 LoRA 时）


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


# ==========================
# SAM Encoder 集成
# ==========================

class SAMEncoderWrapper(nn.Module):
    """SAM Image Encoder 包装器，用于特征提取
    
    可选地应用 Late LoRA 进行参数高效微调
    """
    
    def __init__(
        self,
        sam_checkpoint: str,
        device: torch.device,
        lora_enabled: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        
        # 延迟导入以避免不必要的依赖
        try:
            import sys
            import os
            # 添加 SAM2 路径
            sam2_path = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "sam2")
            if os.path.exists(sam2_path) and sam2_path not in sys.path:
                sys.path.insert(0, sam2_path)
            
            from sam2.build_sam import build_sam2
        except ImportError as e:
            raise ImportError(
                "SAM2 is required but not found. Please install it:\n"
                "git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2\n"
                "cd third_party/sam2 && pip install -e ."
            ) from e
        
        # 加载 SAM 模型
        print(f"[SAM] Loading SAM checkpoint from {sam_checkpoint}")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam_model = build_sam2(model_cfg, sam_checkpoint, device=str(device))
        
        # 提取 image encoder
        self.image_encoder = self.sam_model.image_encoder
        
        # 冻结或启用 LoRA
        if lora_enabled:
            print(f"[SAM] Enabling Late LoRA: rank={lora_rank}, alpha={lora_alpha}")
            try:
                from .sam_lora import apply_late_lora_to_sam_encoder, print_lora_info
                self.lora_modules = apply_late_lora_to_sam_encoder(
                    self.sam_model,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                )
                print_lora_info(self.sam_model)
            except Exception as e:
                print(f"[SAM] Warning: Failed to apply LoRA: {e}")
                self.lora_modules = {}
        else:
            self.lora_modules = {}
            if freeze_encoder:
                print("[SAM] Freezing SAM image encoder")
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
            else:
                print("[SAM] SAM image encoder is trainable (no LoRA)")
        
        # 获取输出特征维度
        self.feature_dim = self._get_feature_dim()
        print(f"[SAM] Encoder feature dimension: {self.feature_dim}")
    
    def _get_feature_dim(self) -> int:
        """通过前向传播一个假输入来获取特征维度"""
        with torch.no_grad():
            # SAM 期望输入为 1024x1024
            dummy = torch.randn(1, 3, 1024, 1024, device=next(self.image_encoder.parameters()).device)
            feat = self.image_encoder(dummy)
            
            # SAM2 可能返回 dict，需要提取实际的特征张量
            if isinstance(feat, dict):
                # 通常在 'vision_features' 或类似的键中
                if 'vision_features' in feat:
                    feat = feat['vision_features']
                elif 'high_res_feats' in feat:
                    feat = feat['high_res_feats']
                else:
                    # 取第一个张量值
                    for v in feat.values():
                        if isinstance(v, torch.Tensor) and v.ndim == 4:
                            feat = v
                            break
            
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            
            return feat.shape[1]  # [B, C, H, W] -> C
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取图像特征
        
        Args:
            x: 输入图像 [B, 3, H, W]，任意尺寸（将被调整为 1024x1024）
            
        Returns:
            特征图 [B, C, H', W']
        """
        # SAM encoder 需要 1024x1024 输入
        orig_size = x.shape[-2:]
        if orig_size != (1024, 1024):
            x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        
        # 提取特征
        feat = self.image_encoder(x)
        
        # SAM2 可能返回 dict，需要提取实际的特征张量
        if isinstance(feat, dict):
            # 通常在 'vision_features' 或类似的键中
            if 'vision_features' in feat:
                feat = feat['vision_features']
            elif 'high_res_feats' in feat:
                feat = feat['high_res_feats']
            else:
                # 取第一个张量值
                for v in feat.values():
                    if isinstance(v, torch.Tensor) and v.ndim == 4:
                        feat = v
                        break
        
        # 如果输出是多尺度特征，取第一个
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        
        return feat


class PointHeatmapModelWithSAM(nn.Module):
    """结合 SAM encoder 的点热力图预测模型
    
    使用 SAM image encoder 提取特征，然后通过热力图头预测点位置
    """
    
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        
        if not cfg.use_sam_encoder:
            raise ValueError("This model requires use_sam_encoder=True")
        
        if cfg.sam_checkpoint is None:
            raise ValueError("sam_checkpoint must be provided when use_sam_encoder=True")
        
        # 确定设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建 SAM encoder
        self.sam_encoder = SAMEncoderWrapper(
            sam_checkpoint=cfg.sam_checkpoint,
            device=device,
            lora_enabled=cfg.sam_lora_enabled,
            lora_rank=cfg.sam_lora_rank,
            lora_alpha=cfg.sam_lora_alpha,
            lora_dropout=cfg.sam_lora_dropout,
            freeze_encoder=cfg.sam_freeze_encoder,
        )
        
        # 获取 SAM encoder 输出维度
        sam_feat_dim = self.sam_encoder.feature_dim
        
        # 条件输入处理：独立的小网络处理条件图像（灰度 mask）
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cfg.cond_in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 特征融合：将 SAM 特征和条件特征拼接
        # SAM 输出通常是 256 通道 @ 64x64（对于 1024x1024 输入）
        # 条件特征需要匹配空间尺寸
        self.feature_fusion = nn.Conv2d(sam_feat_dim + 128, 512, kernel_size=1)
        
        # 热力图头和标签头
        self.head = HeatmapHead(512, mid_channels=256)
        self.label_head = LabelHead(512, mid_channels=256)
        
    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 主图像 [B, 3, H, W]
            cond: 条件图像（灰度 mask）[B, 1, H, W]
            
        Returns:
            logits: 热力图 logits [B, 1, H, W]
            label_logits: 标签 logits [B, 2]
        """
        b, _, h, w = x.shape
        
        # 提取 SAM 特征
        sam_feat = self.sam_encoder(x)  # [B, C, H', W']
        
        # 处理条件输入
        if cond is None:
            cond = torch.zeros(b, self.cfg.cond_in_channels, h, w, device=x.device)
        
        cond_feat = self.cond_encoder(cond)  # [B, 128, H'', W'']
        
        # 匹配空间尺寸
        if cond_feat.shape[-2:] != sam_feat.shape[-2:]:
            cond_feat = F.interpolate(
                cond_feat, 
                size=sam_feat.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
        
        # 融合特征
        fused = torch.cat([sam_feat, cond_feat], dim=1)
        fused = self.feature_fusion(fused)
        
        # 预测热力图
        logits = self.head(fused)  # [B, 1, H', W']
        
        # 上采样到输入分辨率
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        
        # 预测标签
        label_logits = self.label_head(fused)  # [B, 2]
        
        return logits, label_logits

