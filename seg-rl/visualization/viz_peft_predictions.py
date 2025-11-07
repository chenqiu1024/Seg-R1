#!/usr/bin/env python3
"""
PEFT模型预测可视化

可视化PEFT模型的预测热力图和点位置

调用示例:
    python -m seg-rl.visualization.viz_peft_predictions \\
      --image path/to/image.jpg \\
      --prev_mask path/to/prev_mask.png \\
      --checkpoint outputs/braintumour/peft_supervised/checkpoint_best.pt \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --out_dir outputs/viz/predictions
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import load_checkpoint, heatmap_to_rgb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--prev_mask", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sam_checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/viz")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 加载checkpoint配置
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config = ckpt.get('config', {})
    
    lora_rank = config.get('lora_rank', 16)
    lora_alpha = config.get('lora_alpha', 32)
    fusion_mode = config.get('fusion_mode', 'film')
    feature_scale = config.get('feature_scale', 8)
    image_size = config.get('image_size', [512, 512])
    
    # 初始化模型
    print("Initializing models...")
    sam2_lora = LoRASAM2Wrapper(
        args.sam_checkpoint, lora_rank, lora_alpha, args.device
    )
    point_predictor = PointPredictorFromSAMFeatures(
        sam_feature_dim=256,
        output_size=tuple(image_size),
        fusion_mode=fusion_mode,
        feature_scale=feature_scale,
    ).to(args.device)
    
    # 加载权重
    load_checkpoint(args.checkpoint, point_predictor, sam2_lora, None, None, None, args.device)
    point_predictor.eval()
    
    # 加载图像
    image = Image.open(args.image).convert('RGB')
    image_resized = image.resize(tuple(image_size))
    
    # 加载掩模
    if args.prev_mask and os.path.exists(args.prev_mask):
        prev_mask = Image.open(args.prev_mask).convert('L')
    else:
        prev_mask = Image.new('L', image.size, 0)
    
    prev_mask_resized = prev_mask.resize(tuple(image_size), Image.NEAREST)
    
    # 转换为tensor
    import torchvision.transforms.functional as TF
    image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(args.device)
    prev_mask_tensor = TF.to_tensor(prev_mask_resized).to(args.device)
    prev_mask_tensor = (prev_mask_tensor > 0.5).float().unsqueeze(0)
    
    # 预测
    print("Predicting...")
    with torch.no_grad():
        sam_features = sam2_lora.get_image_features(image_tensor, feature_scale)
        heatmap_logits, label_logits = point_predictor(sam_features, prev_mask_tensor)
    
    # Argmax获取点
    B, _, H, W = heatmap_logits.shape
    heatmap_flat = heatmap_logits.view(B, -1)
    pred_idx = heatmap_flat.argmax(dim=1)
    pred_y = (pred_idx // W).item()
    pred_x = (pred_idx % W).item()
    pred_label = label_logits.argmax(dim=1).item()
    
    print(f"Predicted point: ({pred_x}, {pred_y}), label: {pred_label}")
    
    # 可视化
    # 1. 热力图
    heatmap_np = heatmap_logits[0, 0].cpu().numpy()
    heatmap_rgb = heatmap_to_rgb(torch.from_numpy(heatmap_np), colormap='jet')
    heatmap_path = os.path.join(args.out_dir, "heatmap.png")
    Image.fromarray(heatmap_rgb).save(heatmap_path)
    print(f"Saved heatmap to {heatmap_path}")
    
    # 2. 叠加在原图上
    image_np = np.array(image_resized)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_rgb, 0.4, 0)
    
    # 绘制预测点
    color = (0, 255, 0) if pred_label == 1 else (255, 0, 0)
    cv2.circle(overlay, (int(pred_x), int(pred_y)), 5, color, -1)
    cv2.circle(overlay, (int(pred_x), int(pred_y)), 7, (255, 255, 255), 2)
    
    overlay_path = os.path.join(args.out_dir, "overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved overlay to {overlay_path}")
    
    # 3. 掩模可视化
    if args.prev_mask:
        mask_np = np.array(prev_mask_resized)
        mask_viz = cv2.applyColorMap(mask_np, cv2.COLORMAP_VIRIDIS)
        mask_path = os.path.join(args.out_dir, "prev_mask.png")
        cv2.imwrite(mask_path, mask_viz)
        print(f"Saved mask to {mask_path}")
    
    print(f"\nVisualization saved to {args.out_dir}")


if __name__ == "__main__":
    main()

