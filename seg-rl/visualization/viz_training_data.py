#!/usr/bin/env python3
"""
可视化训练数据脚本

用于可视化PEFT训练数据中的图像、点标注和真值掩模。
支持随机采样或遍历所有样本。
如果提供了checkpoint，还会生成预测掩模对比图。

用法:
    # 随机采样5个样本（仅显示标注）
    python -m seg-rl.visualization.viz_training_data \
        --jsonl outputs/braintumour/peft_train-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode random \
        --num_samples 5

    # 遍历所有样本（带预测对比，需要sam_masks_dir）
    python -m seg-rl.visualization.viz_training_data \
        --jsonl outputs/braintumour/peft_test-251107.jsonl \
        --out_dir outputs/braintumour/visualizations \
        --mode all \
        --show_prediction
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

# 使用相对导入或绝对导入
try:
    from visualization.utils import compute_dice, compute_iou
except ImportError:
    # 如果绝对导入失败，尝试相对导入
    from .utils import compute_dice, compute_iou


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """加载JSONL文件"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON数组格式
            data = json.loads(content)
        else:
            # JSONL格式（逐行）
            data = []
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def visualize_sample(
    sample: Dict,
    output_path: str,
    show_mask: bool = True,
    dpi: int = 150,
    show_prediction: bool = True,
) -> None:
    """
    可视化单个样本
    
    Args:
        sample: 样本字典，包含 'image', 'gt_mask', 'points', 'labels', 'sam_masks_dir' 等字段
        output_path: 输出图像路径
        show_mask: 是否显示真值掩模
        dpi: 图像分辨率
        show_prediction: 是否显示预测掩模对比（需要sam_masks_dir）
    """
    # 加载图像
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # 获取点和标签
    points = sample.get('points', [])
    labels = sample.get('labels', [])
    
    if len(points) != len(labels):
        print(f"Warning: Points and labels length mismatch: {len(points)} vs {len(labels)}")
        return
    
    # 加载预测掩模和计算指标（如果sam_masks_dir存在）
    pred_mask = None
    pred_mask_binary = None
    gt_mask_binary_for_metrics = None
    metrics = {}
    do_prediction = False
    
    if show_prediction:
        sam_masks_dir = sample.get('sam_masks_dir', None)
        gt_mask_path = sample.get('gt_mask', None)
        
        if sam_masks_dir and gt_mask_path:
            try:
                # 获取图像stem（文件名不含扩展名）
                image_stem = Path(image_path).stem
                mask_dir = os.path.join(sam_masks_dir, image_stem)
                
                # 加载最后一个步骤的预测掩模（len(points)-1.png）
                if len(points) > 0:
                    last_mask_idx = len(points) - 1
                    pred_mask_path = os.path.join(mask_dir, f"{last_mask_idx}.png")
                    
                    if os.path.exists(pred_mask_path):
                        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
                        pred_mask_binary = (pred_mask > 127).astype(float)
                        
                        # 加载真值掩模并计算指标
                        if os.path.exists(gt_mask_path):
                            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                            gt_mask_binary_for_metrics = (gt_mask > 127).astype(float)
                            
                            # 调整尺寸以匹配（如果需要）
                            if gt_mask_binary_for_metrics.shape != pred_mask_binary.shape:
                                scale_y = pred_mask_binary.shape[0] / gt_mask_binary_for_metrics.shape[0]
                                scale_x = pred_mask_binary.shape[1] / gt_mask_binary_for_metrics.shape[1]
                                gt_mask_binary_for_metrics = zoom(gt_mask_binary_for_metrics, (scale_y, scale_x), order=0)
                            
                            # 转换为tensor计算指标
                            pred_tensor = torch.from_numpy(pred_mask_binary).float()
                            gt_tensor = torch.from_numpy(gt_mask_binary_for_metrics).float()
                            
                            dice = compute_dice(pred_tensor, gt_tensor)
                            iou = compute_iou(pred_tensor, gt_tensor)
                            
                            metrics['dice'] = dice
                            metrics['iou'] = iou
                            metrics['num_points'] = len(points)
                            
                            do_prediction = True
                        else:
                            print(f"Warning: GT mask not found: {gt_mask_path}")
                    else:
                        print(f"Warning: Predicted mask not found: {pred_mask_path}")
                else:
                    print(f"Warning: No points in sample, cannot load predicted mask")
            except Exception as e:
                print(f"Warning: Failed to load predicted mask: {e}")
                import traceback
                traceback.print_exc()
    
    # 创建图形
    if do_prediction:
        # 两幅图：第一幅显示点和标注，第二幅显示掩模对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        ax1, ax2 = axes
    else:
        # 单幅图：只显示点和标注
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax2 = None
    
    # 第一幅图：显示点和标注
    ax1.imshow(img_np)
    
    # 如果有真值掩模，叠加显示
    if show_mask:
        gt_mask_path = sample.get('gt_mask', None)
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                mask_binary = (gt_mask > 127).astype(float)
                ax1.imshow(mask_binary, alpha=0.3, cmap='Greens')
            except Exception as e:
                print(f"Warning: Failed to load mask {gt_mask_path}: {e}")
    
    # 绘制点
    for j, ((x, y), label) in enumerate(zip(points, labels)):
        if label == 1:
            # 前景点：红色，实心圆
            color = 'red'
            size = 150
            point_type = 'FG'
        else:
            # 背景点：蓝色，实心圆
            color = 'blue'
            size = 120
            point_type = 'BG'
        
        # 绘制点
        ax1.scatter(x, y, c=color, s=size, marker='o', 
                   edgecolors='white', linewidths=2, zorder=10)
        
        # 添加标签文本
        ax1.text(x + 15, y, f'{j}({point_type})', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 设置第一幅图的标题
    stem = Path(image_path).stem
    title1 = f'{stem}: {len(points)} points (Ground Truth)'
    ax1.set_title(title1, fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 第二幅图：显示掩模对比（如果进行预测）
    if do_prediction and ax2 is not None:
        ax2.imshow(img_np)
        
        gt_mask_path = sample.get('gt_mask', None)
        gt_mask_binary = None
        
        # 加载并调整真值掩模尺寸（用于显示）
        if gt_mask_path and os.path.exists(gt_mask_path):
            try:
                gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                gt_mask_binary = (gt_mask > 127).astype(float)
                # 调整尺寸以匹配图像
                if gt_mask_binary.shape != img_np.shape[:2]:
                    scale_y = img_np.shape[0] / gt_mask_binary.shape[0]
                    scale_x = img_np.shape[1] / gt_mask_binary.shape[1]
                    gt_mask_binary = zoom(gt_mask_binary, (scale_y, scale_x), order=0)
            except Exception as e:
                print(f"Warning: Failed to load GT mask: {e}")
        
        # 调整预测掩模尺寸（用于显示）
        pred_mask_binary_display = None
        if pred_mask_binary is not None:
            # pred_mask_binary已经是二值化的，但需要调整尺寸以匹配图像
            if pred_mask_binary.shape != img_np.shape[:2]:
                scale_y = img_np.shape[0] / pred_mask_binary.shape[0]
                scale_x = img_np.shape[1] / pred_mask_binary.shape[1]
                pred_mask_binary_display = zoom(pred_mask_binary, (scale_y, scale_x), order=0)
            else:
                pred_mask_binary_display = pred_mask_binary
        
        # 优化显示：先显示重叠区域（黄色），再显示真值（绿色），最后显示预测（红色）
        if gt_mask_binary is not None and pred_mask_binary_display is not None:
            # 重叠区域（黄色）
            overlap = (gt_mask_binary * pred_mask_binary_display).astype(float)
            overlap_colored = np.zeros_like(img_np)
            overlap_colored[:, :, 0] = overlap * 255  # 红色
            overlap_colored[:, :, 1] = overlap * 255  # 绿色
            ax2.imshow(overlap_colored, alpha=0.5)
            
            # 真值掩模的未重叠部分（绿色）
            gt_only = (gt_mask_binary * (1 - pred_mask_binary_display)).astype(float)
            gt_only_colored = np.zeros_like(img_np)
            gt_only_colored[:, :, 1] = gt_only * 255
            ax2.imshow(gt_only_colored, alpha=0.4)
            
            # 预测掩模的未重叠部分（红色）
            pred_only = (pred_mask_binary_display * (1 - gt_mask_binary)).astype(float)
            pred_only_colored = np.zeros_like(img_np)
            pred_only_colored[:, :, 0] = pred_only * 255
            ax2.imshow(pred_only_colored, alpha=0.4)
        elif gt_mask_binary is not None:
            # 只有真值掩模（绿色）
            gt_mask_colored = np.zeros_like(img_np)
            gt_mask_colored[:, :, 1] = gt_mask_binary * 255
            ax2.imshow(gt_mask_colored, alpha=0.4)
        elif pred_mask_binary_display is not None:
            # 只有预测掩模（红色）
            pred_mask_colored = np.zeros_like(img_np)
            pred_mask_colored[:, :, 0] = pred_mask_binary_display * 255
            ax2.imshow(pred_mask_colored, alpha=0.4)
        
        # 绘制预测点（使用JSONL中的points）
        for j, ((x, y), label) in enumerate(zip(points, labels)):
            if label == 1:
                color = 'red'
                size = 150
                point_type = 'FG'
            else:
                color = 'blue'
                size = 120
                point_type = 'BG'
            
            ax2.scatter(x, y, c=color, s=size, marker='o', 
                       edgecolors='white', linewidths=2, zorder=10)
            ax2.text(x + 15, y, f'{j}({point_type})', 
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # 设置第二幅图的标题和指标
        title2 = f'Prediction: {len(points)} points'
        if metrics:
            metrics_text = f"Dice: {metrics.get('dice', 0):.3f}, IoU: {metrics.get('iou', 0):.3f}"
            title2 += f'\n{metrics_text}'
        ax2.set_title(title2, fontsize=14, weight='bold')
        ax2.axis('off')
        
        # 在图像角落添加指标文本
        if metrics:
            metrics_str = f"Dice: {metrics.get('dice', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}\nPoints: {metrics.get('num_points', 0)}"
            ax2.text(0.02, 0.98, metrics_str, 
                    transform=ax2.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化PEFT训练数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='输出图像文件夹路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['random', 'all'],
        default='random',
        help='采样模式: random=随机采样, all=遍历所有样本'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='随机采样模式下的样本数量（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（仅当mode=random时有效）'
    )
    
    parser.add_argument(
        '--show_mask',
        action='store_true',
        default=True,
        help='是否显示真值掩模（默认True）'
    )
    
    parser.add_argument(
        '--no_mask',
        dest='show_mask',
        action='store_false',
        help='不显示真值掩模'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='输出图像分辨率（DPI）'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='输出文件名前缀'
    )
    
    parser.add_argument(
        '--show_prediction',
        action='store_true',
        default=True,
        help='是否显示预测掩模对比（需要sam_masks_dir，默认True）'
    )
    
    parser.add_argument(
        '--no_prediction',
        dest='show_prediction',
        action='store_false',
        help='不显示预测掩模对比'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading data from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} samples")
    
    # 选择样本
    if args.mode == 'random':
        random.seed(args.seed)
        selected_samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"Randomly selected {len(selected_samples)} samples")
    else:
        selected_samples = data
        print(f"Processing all {len(selected_samples)} samples")
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化每个样本
    print(f"\nGenerating visualizations...")
    for i, sample in enumerate(selected_samples):
        # 生成输出文件名
        if args.mode == 'random':
            output_filename = f'{args.prefix}_{i:03d}.png'
        else:
            # 使用图像文件名作为输出文件名
            image_path = sample.get('image', '')
            if image_path:
                stem = Path(image_path).stem
                output_filename = f'{args.prefix}_{stem}.png'
            else:
                output_filename = f'{args.prefix}_{i:04d}.png'
        
        output_path = os.path.join(args.out_dir, output_filename)
        
        # 可视化
        try:
            visualize_sample(
                sample, output_path, 
                show_mask=args.show_mask, 
                dpi=args.dpi,
                show_prediction=args.show_prediction,
            )
            print(f"  [{i+1}/{len(selected_samples)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_samples)}] Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")
    print(f"Total: {len(selected_samples)} images generated")


if __name__ == '__main__':
    main()

