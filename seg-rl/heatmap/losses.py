from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def ce_over_pixels(logits: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over all pixels treating each pixel as a class.

    Args:
        logits: [B,1,H,W] unnormalized scores
        target_xy: [B,2] target coordinates in pixels (x,y)
    Returns:
        scalar loss
    """
    b, _, h, w = logits.shape
    flat_logits = logits.view(b, -1)
    x = target_xy[:, 0].clamp(0, w - 1).long()
    y = target_xy[:, 1].clamp(0, h - 1).long()
    idx = y * w + x
    return F.cross_entropy(flat_logits, idx)


def gaussian_heatmap_targets(
    target_xy: torch.Tensor,
    height: int,
    width: int,
    sigma: float = 3.0,
    normalize: bool = True,
) -> torch.Tensor:
    """生成以真值点为中心的高斯软目标热力图
    
    产生距离真值点逐渐衰减的概率分布，避免硬分类的类别不平衡问题。
    
    Args:
        target_xy: [B,2] 真值坐标 (x,y)
        height, width: 输出热力图尺寸
        sigma: 高斯标准差(像素)
            - 小图或精确目标: 3.0-5.0
            - 大图或模糊目标: 6.0-12.0  
            - 高分辨率(1024+): 8.0-15.0
        normalize: 是否归一化为概率分布(和为1)
    Returns:
        heatmaps: [B,1,H,W] 软目标热力图
    """
    device = target_xy.device
    b = target_xy.shape[0]
    xs = torch.arange(width, device=device).view(1, 1, 1, width).float()
    ys = torch.arange(height, device=device).view(1, 1, height, 1).float()
    x0 = target_xy[:, 0].view(b, 1, 1, 1)
    y0 = target_xy[:, 1].view(b, 1, 1, 1)
    dist2 = (xs - x0) ** 2 + (ys - y0) ** 2
    heat = torch.exp(-0.5 * dist2 / (sigma ** 2))
    if normalize:
        heat = heat / (heat.sum(dim=(2, 3), keepdim=True) + 1e-8)
    return heat


def kl_to_gaussian_targets(logits: torch.Tensor, target_xy: torch.Tensor, sigma: float = 3.0, tau: float = 1.0) -> torch.Tensor:
    """模型分布与高斯软目标间的KL散度损失
    
    让模型学习与真值距离衰减的概率分布，而非硬分类。
    
    Args:
        logits: [B,1,H,W] 模型输出的未归一化分数
        target_xy: [B,2] 真值坐标
        sigma: 高斯目标的标准差，建议:
            - 512x512: 6.0-8.0
            - 1024x1024: 10.0-12.0
        tau: 模型softmax的温度，控制分布尖锐度:
            - tau=1.0: 标准softmax  
            - tau>1.0: 更平缓分布
            - tau<1.0: 更尖锐分布
    Returns:
        scalar loss
    """
    b, _, h, w = logits.shape
    with torch.no_grad():
        p = gaussian_heatmap_targets(target_xy, h, w, sigma=sigma, normalize=True)  # [B,1,H,W]
        p = p.clamp_min(1e-12)
    log_q = F.log_softmax((logits / max(tau, 1e-6)).view(b, -1), dim=1).view(b, 1, h, w)
    kl = (p * (p.log() - log_q)).sum(dim=(1, 2, 3))
    return kl.mean()


def mse_to_gaussian_targets(logits: torch.Tensor, target_xy: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """模型分布与高斯软目标间的MSE损失
    
    更稳定的形状匹配损失，鼓励平滑的距离衰减热力图。
    相比KL散度，MSE对分布形状的匹配更直接，训练更稳定。
    
    Args:
        logits: [B,1,H,W] 模型输出
        target_xy: [B,2] 真值坐标  
        sigma: 高斯目标标准差，建议值同KL损失
    Returns:
        scalar loss
    """
    b, _, h, w = logits.shape
    with torch.no_grad():
        target = gaussian_heatmap_targets(target_xy, h, w, sigma=sigma, normalize=True)  # [B,1,H,W]
    # normalize predicted to probability via softmax over pixels
    prob = F.softmax(logits.view(b, -1), dim=1).view(b, 1, h, w)
    return F.mse_loss(prob, target)


