#!/usr/bin/env python3

"""
从mask图片目录计算包围盒并生成JSON描述文件

读取指定目录下的所有mask图片，计算每个mask的最小包围盒，
并将结果保存为JSON格式文件。

功能:
- 支持多种图片格式 (.png, .jpg, .jpeg, .bmp, .tiff)
- 自动计算每个mask的最小包围盒
- 输出标准JSON格式文件

用法:
    python seg-rl/calculate_bbox_from_masks.py \
      --mask_dir /path/to/masks \
      --output_json /path/to/output.json

示例:
    python seg-rl/calculate_bbox_from_masks.py \
      --mask_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/masks_step1 \
      --output_json /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json

输出JSON格式:
    [
      {
        "mask_path": "/path/to/mask1.png",
        "bbox": [10, 20, 100, 150]
      },
      {
        "mask_path": "/path/to/mask2.png", 
        "bbox": [50, 80, 200, 300]
      }
    ]

其中bbox格式为 [x_min, y_min, x_max, y_max]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

# 支持的图片格式
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    if len(mask.shape) == 3:
        # 如果是彩色图像，转换为灰度
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # 二值化处理
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def get_mask_files(mask_dir: str) -> List[str]:
    """获取目录下所有支持格式的mask文件
    
    Args:
        mask_dir: mask文件目录
        
    Returns:
        mask文件路径列表，按文件名排序
    """
    mask_files = []
    
    if not os.path.isdir(mask_dir):
        print(f"Error: Directory not found: {mask_dir}")
        return mask_files
    
    for file_path in Path(mask_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            mask_files.append(str(file_path))
    
    # 按文件名排序
    mask_files.sort()
    return mask_files


def process_mask_file(mask_path: str) -> Tuple[bool, int, int, int, int]:
    """处理单个mask文件
    
    Args:
        mask_path: mask文件路径
        
    Returns:
        (success, x_min, y_min, x_max, y_max) 元组
    """
    try:
        # 读取mask图像
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[ERROR] Failed to load image: {mask_path}")
            return False, 0, 0, 0, 0
        
        # 计算包围盒
        x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
        return True, x_min, y_min, x_max, y_max
        
    except Exception as e:
        print(f"[ERROR] Failed to process {mask_path}: {e}")
        return False, 0, 0, 0, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate bounding boxes from mask images and generate JSON output"
    )
    parser.add_argument(
        "--mask_dir", 
        type=str, 
        required=True,
        help="Directory containing mask images"
    )
    parser.add_argument(
        "--output_json", 
        type=str, 
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--relative_paths", 
        action="store_true",
        help="Use relative paths in JSON output (relative to mask_dir)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed processing information"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.mask_dir):
        print(f"Error: Mask directory not found: {args.mask_dir}")
        return 1
    
    # 获取所有mask文件
    print(f"Scanning mask directory: {args.mask_dir}")
    mask_files = get_mask_files(args.mask_dir)
    
    if not mask_files:
        print(f"No supported image files found in: {args.mask_dir}")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        return 1
    
    print(f"Found {len(mask_files)} mask files")
    
    # 处理所有mask文件
    json_results = []
    num_processed = 0
    num_errors = 0
    
    for i, mask_path in enumerate(mask_files, 1):
        if args.verbose:
            print(f"[{i}/{len(mask_files)}] Processing: {mask_path}")
        
        success, x_min, y_min, x_max, y_max = process_mask_file(mask_path)
        
        if success:
            # 决定使用相对路径还是绝对路径
            if args.relative_paths:
                # 相对于mask_dir的相对路径
                output_path = os.path.relpath(mask_path, args.mask_dir)
            else:
                # 使用绝对路径
                output_path = os.path.abspath(mask_path)
            
            json_results.append({
                "mask_path": output_path,
                "bbox": [x_min, y_min, x_max, y_max]
            })
            
            if args.verbose:
                print(f"  -> bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
            
            num_processed += 1
        else:
            num_errors += 1
    
    # 保存JSON输出
    if json_results:
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(args.output_json)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(args.output_json, "w", encoding="utf-8") as json_file:
                json.dump(json_results, json_file, indent=2, ensure_ascii=False)
            
            print(f"\nJSON output saved to: {args.output_json}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON output: {e}")
            return 1
    else:
        print("No valid masks processed, no JSON output generated.")
        return 1
    
    # 打印总结
    print(f"\nProcessing completed:")
    print(f"  Total files: {len(mask_files)}")
    print(f"  Processed: {num_processed}")
    print(f"  Errors: {num_errors}")
    print(f"  Output file: {args.output_json}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
