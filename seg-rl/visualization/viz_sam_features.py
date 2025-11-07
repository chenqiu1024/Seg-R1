#!/usr/bin/env python3
"""
SAM特征可视化

使用PCA降维可视化SAM编码器的特征图，对比LoRA微调前后的差异

调用示例:
    python -m seg-rl.visualization.viz_sam_features \\
      --image path/to/image.jpg \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --lora_checkpoint outputs/braintumour/peft_supervised/checkpoint_best.pt \\
      --out_dir outputs/viz/sam_features
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--sam_checkpoint", type=str, required=True)
    p.add_argument("--lora_checkpoint", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/viz")
    p.add_argument("--feature_scale", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def visualize_features_pca(features: torch.Tensor, title: str, save_path: str):
    """
    使用PCA将特征降维到3通道并可视化
    
    Args:
        features: [1, C, H, W]
        title: 标题
        save_path: 保存路径
    """
    # [1, C, H, W] -> [H*W, C]
    B, C, H, W = features.shape
    features_flat = features[0].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    
    # PCA降到3维
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_flat)  # [H*W, 3]
    
    # Reshape回 [H, W, 3]
    features_pca = features_pca.reshape(H, W, 3)
    
    # 归一化到 [0, 255]
    for i in range(3):
        channel = features_pca[:, :, i]
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        features_pca[:, :, i] = channel * 255
    
    features_pca = features_pca.astype(np.uint8)
    
    # 添加标题
    h, w = features_pca.shape[:2]
    canvas = np.zeros((h + 30, w, 3), dtype=np.uint8)
    canvas[30:, :, :] = features_pca
    cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"Saved {save_path}")


def main():
    args = parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 加载图像
    image = Image.open(args.image).convert('RGB')
    image = image.resize((512, 512))
    
    import torchvision.transforms.functional as TF
    image_tensor = TF.to_tensor(image).unsqueeze(0).to(args.device)
    
    # 1. 原始SAM特征（无LoRA）
    print("Extracting features from original SAM...")
    sam_original = LoRASAM2Wrapper(
        args.sam_checkpoint, lora_rank=16, lora_alpha=32, device=args.device
    )
    features_original = sam_original.get_image_features(image_tensor, args.feature_scale)
    
    viz_path_original = os.path.join(args.out_dir, "sam_features_original.png")
    visualize_features_pca(features_original, "Original SAM", viz_path_original)
    
    # 2. 如果提供了LoRA checkpoint，提取微调后的特征
    if args.lora_checkpoint:
        print(f"Extracting features from LoRA-finetuned SAM...")
        sam_lora = LoRASAM2Wrapper(
            args.sam_checkpoint, lora_rank=16, lora_alpha=32, device=args.device
        )
        sam_lora.load_lora_checkpoint(args.lora_checkpoint)
        
        features_lora = sam_lora.get_image_features(image_tensor, args.feature_scale)
        
        viz_path_lora = os.path.join(args.out_dir, "sam_features_lora.png")
        visualize_features_pca(features_lora, "LoRA-finetuned SAM", viz_path_lora)
        
        # 差异图
        diff = torch.abs(features_lora - features_original)
        viz_path_diff = os.path.join(args.out_dir, "sam_features_diff.png")
        visualize_features_pca(diff, "Difference (LoRA - Original)", viz_path_diff)
    
    print(f"\nVisualization saved to {args.out_dir}")


if __name__ == "__main__":
    main()

