#!/usr/bin/env python3
"""
PEFT模型评估脚本

在测试集上评估训练好的PEFT模型性能

主要功能:
- 加载训练好的checkpoint
- 执行完整rollout（迭代预测点序列）
- 计算指标：Dice, IoU, PCK, 点数等
- 生成可视化：点序列图、分割结果对比

调用示例:
    python -m seg-rl.peft.eval_peft_model \\
      --test_json datasets/seg_r1_md/Task01_BrainTumour/test.jsonl \\
      --checkpoint outputs/braintumour/peft_supervised/checkpoint_best.pt \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --out_dir outputs/braintumour/peft_eval \\
      --max_rollout_steps 16 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import compute_dice, compute_iou, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PEFT Model")
    
    p.add_argument("--test_json", type=str, required=True, help="Test JSONL file")
    p.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint to evaluate")
    p.add_argument("--sam_checkpoint", type=str, required=True, help="SAM2 checkpoint")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for results")
    
    p.add_argument("--image_size", type=int, nargs=2, default=[512, 512], help="Image size")
    p.add_argument("--max_rollout_steps", type=int, default=16, help="Max points per image")
    p.add_argument("--dice_threshold", type=float, default=0.95, help="Stop if Dice > threshold")
    p.add_argument("--improvement_threshold", type=float, default=0.001,
                   help="Stop if improvement < threshold")
    
    p.add_argument("--device", type=str, default="cuda", help="Device")
    p.add_argument("--save_visualizations", action="store_true", help="Save visualization images")
    
    return p.parse_args()


@torch.no_grad()
def rollout_single_image(
    image_path: str,
    gt_mask_path: str,
    sam2_lora: LoRASAM2Wrapper,
    point_predictor: PointPredictorFromSAMFeatures,
    args,
) -> Dict:
    """
    对单张图像执行完整rollout
    
    Returns:
        结果字典，包含点序列、Dice曲线等
    """
    device = args.device
    
    # 加载图像和真值
    image = Image.open(image_path).convert('RGB')
    gt_mask = Image.open(gt_mask_path).convert('L')
    
    orig_w, orig_h = image.size
    
    # Resize
    image_resized = image.resize(tuple(args.image_size), Image.BILINEAR)
    gt_mask_resized = gt_mask.resize(tuple(args.image_size), Image.NEAREST)
    
    # 转换为tensor
    import torchvision.transforms.functional as TF
    image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)  # [1, 3, H, W]
    gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)  # [1, H, W]
    gt_mask_binary = (gt_mask_tensor > 0.5).float()
    
    # Rollout
    points_seq = []
    labels_seq = []
    dice_history = []
    iou_history = []
    
    prev_dice = 0.0
    
    for step in range(args.max_rollout_steps):
        # 生成当前掩模
        if step == 0:
            prev_mask = torch.zeros_like(gt_mask_binary)
        else:
            # 用SAM2生成掩模
            # 需要将点坐标缩放回原始尺寸
            points_orig = [(px * orig_w / args.image_size[1], py * orig_h / args.image_size[0])
                          for px, py in points_seq]
            prev_mask = sam2_lora.predict_mask(
                np.array(image), points_orig, labels_seq
            )
            # predict_mask 返回的是 torch.Tensor，不是 numpy 数组
            if isinstance(prev_mask, torch.Tensor):
                prev_mask = prev_mask.unsqueeze(0).to(device)
            else:
                prev_mask = torch.from_numpy(prev_mask).unsqueeze(0).to(device)
            prev_mask = torch.nn.functional.interpolate(
                prev_mask.unsqueeze(0), size=tuple(args.image_size), mode='nearest'
            )[0]
        
        # 预测下一个点
        sam_features = sam2_lora.get_image_features(image_tensor, feature_scale=8)
        heatmap_logits, label_logits = point_predictor(sam_features, prev_mask.unsqueeze(0))
        
        # Argmax获取点
        B, _, H, W = heatmap_logits.shape
        heatmap_flat = heatmap_logits.view(B, -1)
        pred_idx = heatmap_flat.argmax(dim=1)
        pred_y = (pred_idx // W).float()
        pred_x = (pred_idx % W).float()
        
        # 标签
        pred_label = label_logits.argmax(dim=1).item()
        
        # 记录
        points_seq.append((pred_x.item(), pred_y.item()))
        labels_seq.append(pred_label)
        
        # 计算指标
        dice = compute_dice(prev_mask[0], gt_mask_binary[0])
        iou = compute_iou(prev_mask[0], gt_mask_binary[0])
        dice_history.append(dice)
        iou_history.append(iou)
        
        # 停止条件
        if dice > args.dice_threshold:
            break
        if step > 0 and (dice - prev_dice) < args.improvement_threshold:
            break
        
        prev_dice = dice
    
    # 将点坐标从resize后的尺寸缩放回原始图像尺寸
    # points_seq中的坐标是在args.image_size坐标系中的，需要缩放回orig_w x orig_h
    # args.image_size是[H, W]，所以args.image_size[0]是H，args.image_size[1]是W
    points_orig = [
        (px * orig_w / args.image_size[1], py * orig_h / args.image_size[0])
        for px, py in points_seq
    ]
    
    return {
        'image_path': image_path,
        'points': points_orig,  # 返回原始图像尺寸的坐标
        'labels': labels_seq,
        'dice_history': dice_history,
        'iou_history': iou_history,
        'final_dice': dice_history[-1] if dice_history else 0.0,
        'final_iou': iou_history[-1] if iou_history else 0.0,
        'num_points': len(points_seq),
    }


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载checkpoint配置
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    
    # 提取配置
    lora_rank = config.get('lora_rank', 16)
    lora_alpha = config.get('lora_alpha', 32)
    fusion_mode = config.get('fusion_mode', 'film')
    feature_scale = config.get('feature_scale', 8)
    
    print(f"Model config: LoRA rank={lora_rank}, alpha={lora_alpha}, fusion={fusion_mode}")
    
    # 1. 初始化SAM2 + LoRA
    print("\nInitializing SAM2 with LoRA...")
    sam2_lora = LoRASAM2Wrapper(
        sam_checkpoint=args.sam_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=str(device)
    )
    
    # 2. 初始化点预测网络
    print("Initializing Point Predictor...")
    
    # 根据 feature_scale 确定 SAM 特征通道数
    sam_feature_dim_map = {4: 32, 8: 64, 16: 256}
    if feature_scale not in sam_feature_dim_map:
        raise ValueError(f"Unsupported feature_scale: {feature_scale}. Use 4, 8, or 16.")
    sam_feature_dim = sam_feature_dim_map[feature_scale]
    
    point_predictor = PointPredictorFromSAMFeatures(
        sam_feature_dim=sam_feature_dim,
        output_size=tuple(args.image_size),
        fusion_mode=fusion_mode,
        feature_scale=feature_scale,
    ).to(device)
    
    # 3. 加载权重
    print("Loading model weights...")
    load_checkpoint(
        args.checkpoint, point_predictor, sam2_lora, None, None, None, str(device)
    )
    
    point_predictor.eval()
    
    # 4. 加载测试数据
    print(f"\nLoading test data from {args.test_json}")
    with open(args.test_json, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            test_samples = json.loads(content)
        else:
            test_samples = [json.loads(line) for line in content.splitlines() if line.strip()]
    
    print(f"Loaded {len(test_samples)} test samples")
    
    # 5. 评估
    print("\n" + "="*80)
    print("Evaluating...")
    print("="*80)
    
    results = []
    
    for sample in tqdm(test_samples):
        image_path = sample['image']
        gt_mask_path = sample.get('gt_mask', None)
        
        if gt_mask_path is None or not os.path.exists(gt_mask_path):
            print(f"Warning: GT mask not found for {image_path}, skipping")
            continue
        
        result = rollout_single_image(
            image_path, gt_mask_path, sam2_lora, point_predictor, args
        )
        results.append(result)
    
    # 6. 汇总统计
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    
    final_dices = [r['final_dice'] for r in results]
    final_ious = [r['final_iou'] for r in results]
    num_points = [r['num_points'] for r in results]
    
    print(f"Number of samples: {len(results)}")
    print(f"\nDice Score:")
    print(f"  Mean: {np.mean(final_dices):.4f}")
    print(f"  Std:  {np.std(final_dices):.4f}")
    print(f"  Median: {np.median(final_dices):.4f}")
    
    print(f"\nIoU Score:")
    print(f"  Mean: {np.mean(final_ious):.4f}")
    print(f"  Std:  {np.std(final_ious):.4f}")
    print(f"  Median: {np.median(final_ious):.4f}")
    
    print(f"\nNumber of Points:")
    print(f"  Mean: {np.mean(num_points):.2f}")
    print(f"  Std:  {np.std(num_points):.2f}")
    print(f"  Median: {np.median(num_points):.0f}")
    
    # 7. 保存结果
    results_json_path = os.path.join(args.out_dir, 'eval_results.json')
    with open(results_json_path, 'w') as f:
        json.dump({
            'summary': {
                'num_samples': len(results),
                'mean_dice': float(np.mean(final_dices)),
                'std_dice': float(np.std(final_dices)),
                'median_dice': float(np.median(final_dices)),
                'mean_iou': float(np.mean(final_ious)),
                'mean_num_points': float(np.mean(num_points)),
            },
            'per_sample_results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to {results_json_path}")


if __name__ == "__main__":
    main()

