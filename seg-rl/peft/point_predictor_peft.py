#!/usr/bin/env python3
"""
点预测网络（PEFT版本） - 基于SAM特征预测下一个提示点

该网络接受SAM2编码器的特征和当前掩模作为输入，输出热力图logits和标签logits。
网络架构包括：掩模编码器、特征融合层、上采样解码器和输出头。

调用示例:
    from seg-rl.peft import PointPredictorFromSAMFeatures
    
    # 初始化网络
    predictor = PointPredictorFromSAMFeatures(
        sam_feature_dim=256,
        output_size=(512, 512),
        fusion_mode="film"  # 或 "concat"
    )
    
    # 前向传播
    sam_features = sam2_lora.get_image_features(image)  # [B, 256, H/8, W/8]
    prev_mask = ...  # [B, 1, H, W]
    heatmap_logits, label_logits = predictor(sam_features, prev_mask)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用现有的输出头
try:
    from ..heatmap.model import HeatmapHead, LabelHead
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from heatmap.model import HeatmapHead, LabelHead


class MaskEncoder(nn.Module):
    """
    掩模编码器：将掩模下采样并编码到特征维度
    
    将 [B, 1, H, W] 的掩模编码为 [B, mask_channels, H/scale, W/scale]
    
    Args:
        output_channels: 输出通道数
        output_scale: 输出尺度（相对输入的下采样倍数）
    """
    
    def __init__(self, output_channels: int = 64, output_scale: int = 8):
        super().__init__()
        self.output_scale = output_scale
        
        # 使用卷积逐步下采样
        # H, W -> H/2, W/2 -> H/4, W/4 -> H/8, W/8
        layers = []
        in_ch = 1
        scales_needed = [2, 4, 8, 16]
        target_scale = output_scale
        
        if target_scale not in scales_needed:
            raise ValueError(f"output_scale must be one of {scales_needed}")
        
        # 构建下采样路径
        channels_progression = [16, 32, 64, 64]
        for i, scale in enumerate(scales_needed):
            out_ch = channels_progression[i] if i < len(channels_progression) else output_channels
            if i == len(scales_needed) - 1:
                out_ch = output_channels
            
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
            
            if scale == target_scale:
                break
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask: [B, 1, H, W]
            
        Returns:
            encoded_mask: [B, output_channels, H/scale, W/scale]
        """
        return self.encoder(mask)


class FiLMFusion(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) 融合层
    
    使用掩模特征调制SAM特征: 
    output = sam_features * (1 + gamma) + beta
    
    Args:
        sam_channels: SAM特征通道数
        mask_channels: 掩模特征通道数
    """
    
    def __init__(self, sam_channels: int, mask_channels: int):
        super().__init__()
        # 从掩模特征生成gamma和beta
        self.to_gamma = nn.Sequential(
            nn.Conv2d(mask_channels, sam_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.to_beta = nn.Sequential(
            nn.Conv2d(mask_channels, sam_channels, kernel_size=1),
        )
    
    def forward(self, sam_features: torch.Tensor, mask_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sam_features: [B, sam_channels, H, W]
            mask_features: [B, mask_channels, H, W]
            
        Returns:
            fused_features: [B, sam_channels, H, W]
        """
        gamma = self.to_gamma(mask_features)
        beta = self.to_beta(mask_features)
        return sam_features * (1.0 + gamma) + beta


class ConcatFusion(nn.Module):
    """
    Concatenation 融合层
    
    将SAM特征和掩模特征在通道维度拼接，然后用1x1卷积融合
    
    Args:
        sam_channels: SAM特征通道数
        mask_channels: 掩模特征通道数
        output_channels: 输出通道数
    """
    
    def __init__(self, sam_channels: int, mask_channels: int, output_channels: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(sam_channels + mask_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, sam_features: torch.Tensor, mask_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sam_features: [B, sam_channels, H, W]
            mask_features: [B, mask_channels, H, W]
            
        Returns:
            fused_features: [B, output_channels, H, W]
        """
        concatenated = torch.cat([sam_features, mask_features], dim=1)
        return self.fusion(concatenated)


class Decoder(nn.Module):
    """
    上采样解码器：将融合特征上采样到目标分辨率
    
    使用转置卷积 + skip connections（如有）
    
    Args:
        in_channels: 输入通道数
        output_size: 目标输出尺寸 (H, W)
        current_scale: 当前特征相对输出的尺度（如8表示当前是H/8）
    """
    
    def __init__(self, in_channels: int, output_size: Tuple[int, int], current_scale: int = 8):
        super().__init__()
        self.output_size = output_size
        self.current_scale = current_scale
        
        # 计算需要上采样的次数
        # current_scale=8 -> 需要3次上采样 (8->4->2->1)
        num_upsamples = 0
        scale = current_scale
        while scale > 1:
            scale = scale // 2
            num_upsamples += 1
        
        # 构建上采样路径
        layers = []
        ch = in_channels
        for i in range(num_upsamples):
            out_ch = max(ch // 2, 64)  # 逐渐减少通道数，但不低于64
            layers.extend([
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            ch = out_ch
        
        self.decoder = nn.Sequential(*layers)
        self.out_channels = ch
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H/scale, W/scale]
            
        Returns:
            decoded: [B, out_channels, H, W]
        """
        x = self.decoder(x)
        
        # 如果尺寸不完全匹配，用插值调整
        if x.shape[2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        return x


class PointPredictorFromSAMFeatures(nn.Module):
    """
    基于SAM特征的点预测网络
    
    架构流程:
    1. MaskEncoder: 编码当前掩模
    2. FeatureFusion: 融合SAM特征和掩模特征
    3. Decoder: 上采样到目标分辨率
    4. Heads: 输出热力图logits和标签logits
    
    Args:
        sam_feature_dim: SAM特征的通道数（默认256）
        mask_encoder_channels: 掩模编码器的输出通道数
        output_size: 输出热力图的尺寸 (H, W)
        fusion_mode: 特征融合模式，"film"或"concat"
        feature_scale: SAM特征相对输出的尺度（8表示H/8）
    """
    
    def __init__(
        self,
        sam_feature_dim: int = 256,
        mask_encoder_channels: int = 64,
        output_size: Tuple[int, int] = (512, 512),
        fusion_mode: str = "film",
        feature_scale: int = 8,
    ):
        super().__init__()
        
        self.sam_feature_dim = sam_feature_dim
        self.mask_encoder_channels = mask_encoder_channels
        self.output_size = output_size
        self.fusion_mode = fusion_mode
        self.feature_scale = feature_scale
        
        # 1. 掩模编码器
        self.mask_encoder = MaskEncoder(
            output_channels=mask_encoder_channels,
            output_scale=feature_scale
        )
        
        # 2. 特征融合
        if fusion_mode == "film":
            self.fusion = FiLMFusion(sam_feature_dim, mask_encoder_channels)
            decoder_in_channels = sam_feature_dim
        elif fusion_mode == "concat":
            self.fusion = ConcatFusion(sam_feature_dim, mask_encoder_channels, sam_feature_dim)
            decoder_in_channels = sam_feature_dim
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}. Use 'film' or 'concat'.")
        
        # 3. 解码器
        self.decoder = Decoder(
            in_channels=decoder_in_channels,
            output_size=output_size,
            current_scale=feature_scale
        )
        
        # 4. 输出头（复用现有实现）
        decoder_out_channels = self.decoder.out_channels
        self.heatmap_head = HeatmapHead(in_channels=decoder_out_channels, mid_channels=128)
        self.label_head = LabelHead(in_channels=decoder_out_channels, mid_channels=128, num_classes=2)
    
    def forward(
        self,
        sam_features: torch.Tensor,
        prev_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            sam_features: SAM编码器特征 [B, sam_feature_dim, H/scale, W/scale]
            prev_mask: 当前掩模 [B, 1, H, W]
            
        Returns:
            heatmap_logits: [B, 1, H, W]
            label_logits: [B, 2]
        """
        # 1. 编码掩模
        mask_features = self.mask_encoder(prev_mask)  # [B, mask_channels, H/scale, W/scale]
        
        # 2. 融合特征
        fused_features = self.fusion(sam_features, mask_features)  # [B, channels, H/scale, W/scale]
        
        # 3. 解码上采样
        decoded = self.decoder(fused_features)  # [B, decoder_out_channels, H, W]
        
        # 4. 输出头
        heatmap_logits = self.heatmap_head(decoded)  # [B, 1, H, W]
        label_logits = self.label_head(decoded)      # [B, 2]
        
        return heatmap_logits, label_logits
    
    def extra_repr(self) -> str:
        return (f'sam_feature_dim={self.sam_feature_dim}, '
                f'output_size={self.output_size}, '
                f'fusion_mode={self.fusion_mode}, '
                f'feature_scale={self.feature_scale}')


# 便捷函数
def create_point_predictor(
    output_size: Tuple[int, int] = (512, 512),
    fusion_mode: str = "film",
    feature_scale: int = 8,
) -> PointPredictorFromSAMFeatures:
    """
    创建点预测网络（便捷函数）
    
    Args:
        output_size: 输出尺寸
        fusion_mode: 融合模式
        feature_scale: 特征尺度
        
    Returns:
        PointPredictorFromSAMFeatures实例
    """
    return PointPredictorFromSAMFeatures(
        sam_feature_dim=256,
        mask_encoder_channels=64,
        output_size=output_size,
        fusion_mode=fusion_mode,
        feature_scale=feature_scale,
    )

