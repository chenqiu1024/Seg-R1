#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割 (简化版本)

直接复用seg-r1/src/open_r1/grpo.py中的SAMWrapper类进行分割

用法:
    python seg-rl/sam2_segment_simple.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/mask_salient_points-0.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks-1 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 添加seg-r1和third_party路径以导入SAMWrapper及sam2依赖
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "seg-r1" / "src"))
# Ensure Python sees the inner package directory (third_party/sam2)
sys.path.insert(0, str(project_root / "third_party" / "sam2"))

try:
    from open_r1.grpo import SAMWrapper
except ImportError as e:
    print(f"Error importing SAMWrapper from grpo.py: {e}")
    print("Please ensure the seg-r1 module is properly set up.")
    print("You may need to:")
    print("1. Install SAM2 dependencies")
    print("2. Check the import path")
    sys.exit(1)


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签"""
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像"""
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts (using grpo.py SAMWrapper)")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(args.sam_checkpoint, args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 处理JSONL文件
    print(f"Processing JSONL file: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                num_errors += 1
                continue
            
            # 验证并提取点信息
            is_valid, points, labels = validate_and_extract_points(obj, line_num)
            if not is_valid:
                num_errors += 1
                continue
            
            image_path = obj["image"]
            
            # 检查图像文件是否存在
            if not os.path.isfile(image_path):
                print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
                num_errors += 1
                continue
            
            # 生成输出路径
            output_path = get_output_path(image_path, args.output_dir)
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.isfile(output_path):
                print(f"[SKIP] Line {line_num}: Output already exists: {output_path}")
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                
                # 调整大小（如果指定）
                if args.resize:
                    image = image.resize(args.resize, PILImage.BILINEAR)
                
                # 运行SAM2预测
                mask, confidence = sam_wrapper.predict(image, points, labels)
                
                # 保存mask
                save_mask_as_grayscale(mask, output_path)
                
                print(f"[OK] Line {line_num}: {image_path} -> {output_path} (confidence: {confidence:.3f})")
                num_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: Failed to process {image_path} - {e}")
                num_errors += 1
                continue
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
