#!/usr/bin/env python3
"""
PEFT训练辅助工具函数

包括：
- Metrics计算（Dice, IoU, PCK等）
- Checkpoint管理（保存/加载）
- 可视化辅助函数
- 学习率调度器
"""

from __future__ import annotations

import glob
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Metrics计算
# ============================================================================

def compute_dice(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    计算Dice系数
    
    Args:
        pred: 预测掩模 [H, W] 或 [B, H, W]，二值
        target: 真值掩模 [H, W] 或 [B, H, W]，二值
        epsilon: 平滑项
        
    Returns:
        Dice系数 [0, 1]
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    计算IoU (Intersection over Union)
    
    Args:
        pred: 预测掩模
        target: 真值掩模
        epsilon: 平滑项
        
    Returns:
        IoU [0, 1]
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.item()


def compute_pck(pred_point: torch.Tensor, target_point: torch.Tensor, threshold: float) -> bool:
    """
    计算PCK (Percentage of Correct Keypoints)
    
    如果预测点与真值点的距离小于阈值，则认为正确
    
    Args:
        pred_point: 预测点 [2] (x, y)
        target_point: 真值点 [2] (x, y)
        threshold: 距离阈值（像素）
        
    Returns:
        是否正确（距离 < 阈值）
    """
    distance = torch.norm(pred_point - target_point, p=2)
    return distance.item() < threshold


def compute_batch_pck(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    threshold: float
) -> Tuple[float, torch.Tensor]:
    """
    批量计算PCK
    
    Args:
        pred_points: 预测点 [B, 2]
        target_points: 真值点 [B, 2]
        threshold: 距离阈值
        
    Returns:
        (PCK准确率, 距离向量 [B])
    """
    distances = torch.norm(pred_points - target_points, p=2, dim=1)
    correct = (distances < threshold).float()
    pck = correct.mean().item()
    return pck, distances


# ============================================================================
# Checkpoint管理
# ============================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    step: int,
    point_predictor: nn.Module,
    sam_lora_wrapper,
    optimizer_sam: Optional[torch.optim.Optimizer] = None,
    optimizer_point: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    config: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
):
    """
    保存完整checkpoint
    
    Args:
        path: 保存路径
        epoch: 当前epoch
        step: 当前step
        point_predictor: 点预测网络
        sam_lora_wrapper: SAM2 LoRA包装器
        optimizer_sam: SAM优化器（可选）
        optimizer_point: 点网络优化器（可选）
        scheduler: 学习率调度器（可选）
        config: 配置字典（可选）
        metrics: 指标字典（可选）
    """
    checkpoint = {
        'epoch': epoch,
        'step': step,
        
        # 模型状态
        'point_predictor_state': point_predictor.state_dict(),
        'sam_lora_state': sam_lora_wrapper.get_lora_state_dict(),
        
        # 配置
        'config': config or {},
        
        # 指标
        'metrics': metrics or {},
        
        # 随机状态
        'random_state': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
    }
    
    # 优化器状态
    if optimizer_sam is not None:
        checkpoint['optimizer_sam_state'] = optimizer_sam.state_dict()
    if optimizer_point is not None:
        checkpoint['optimizer_point_state'] = optimizer_point.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    point_predictor: nn.Module,
    sam_lora_wrapper,
    optimizer_sam: Optional[torch.optim.Optimizer] = None,
    optimizer_point: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda"
) -> Tuple[int, int, Dict, Dict]:
    """
    加载checkpoint
    
    Args:
        path: checkpoint路径
        point_predictor: 点预测网络
        sam_lora_wrapper: SAM2 LoRA包装器
        optimizer_sam: SAM优化器（可选）
        optimizer_point: 点网络优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
        
    Returns:
        (epoch, step, config, metrics)
    """
    checkpoint = torch.load(path, map_location=device)
    
    # 加载模型状态
    point_predictor.load_state_dict(checkpoint['point_predictor_state'])
    sam_lora_wrapper.load_lora_state_dict(checkpoint['sam_lora_state'])
    
    # 加载优化器状态
    if optimizer_sam is not None and 'optimizer_sam_state' in checkpoint:
        optimizer_sam.load_state_dict(checkpoint['optimizer_sam_state'])
    if optimizer_point is not None and 'optimizer_point_state' in checkpoint:
        optimizer_point.load_state_dict(checkpoint['optimizer_point_state'])
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    # 恢复随机状态
    if 'random_state' in checkpoint:
        torch.set_rng_state(checkpoint['random_state']['torch'])
        np.random.set_state(checkpoint['random_state']['numpy'])
        random.setstate(checkpoint['random_state']['python'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    config = checkpoint.get('config', {})
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {path} (epoch={epoch}, step={step})")
    return epoch, step, config, metrics


def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    """
    查找输出目录下最新的checkpoint
    
    Args:
        out_dir: 输出目录
        
    Returns:
        最新checkpoint路径，如果没有则返回None
    """
    ckpt_pattern = os.path.join(out_dir, 'checkpoint_*.pt')
    ckpts = glob.glob(ckpt_pattern)
    if not ckpts:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(ckpts, key=os.path.getmtime)
    return latest


# ============================================================================
# 学习率调度
# ============================================================================

def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    创建warmup + cosine学习率调度器
    
    Args:
        optimizer: 优化器
        warmup_epochs: warmup的epoch数
        total_epochs: 总epoch数
        min_lr_ratio: 最小学习率相对初始学习率的比例
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 可视化辅助
# ============================================================================

def heatmap_to_rgb(heatmap: torch.Tensor, colormap: str = 'jet') -> np.ndarray:
    """
    将热力图转换为RGB可视化
    
    Args:
        heatmap: 热力图 [H, W] 或 [1, H, W]
        colormap: 颜色映射 ('jet', 'viridis', 'hot')
        
    Returns:
        RGB图像 [H, W, 3], uint8
    """
    if heatmap.dim() == 3:
        heatmap = heatmap[0]
    
    heatmap_np = heatmap.cpu().numpy()
    
    # 归一化到 [0, 1]
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    
    # 转换为uint8
    heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
    
    # 应用颜色映射
    try:
        import cv2
        if colormap == 'jet':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        elif colormap == 'viridis':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
        elif colormap == 'hot':
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
        else:
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # cv2返回BGR，转为RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    except ImportError:
        # 如果没有cv2，使用简单的灰度图
        colored = np.stack([heatmap_uint8] * 3, axis=-1)
    
    return colored


def draw_point_on_image(
    image: np.ndarray,
    point: Tuple[float, float],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 5,
    thickness: int = -1
) -> np.ndarray:
    """
    在图像上绘制点
    
    Args:
        image: 输入图像 [H, W, 3], uint8
        point: 点坐标 (x, y)
        color: 颜色 (R, G, B)
        radius: 半径
        thickness: 线条粗细，-1表示填充
        
    Returns:
        绘制后的图像
    """
    try:
        import cv2
        image_copy = image.copy()
        cv2.circle(image_copy, (int(point[0]), int(point[1])), radius, color, thickness)
        return image_copy
    except ImportError:
        return image


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    将tensor转换为numpy图像用于可视化
    
    Args:
        tensor: [3, H, W], [0, 1] float tensor
        
    Returns:
        [H, W, 3], uint8 numpy array
    """
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


# ============================================================================
# 训练辅助
# ============================================================================

def set_seed(seed: int):
    """
    设置随机种子以保证可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        (总参数数, 可训练参数数)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

