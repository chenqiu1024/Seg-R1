#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取包含图像路径和正/负类点坐标的JSONL文件，使用SAM2进行分割，
并将预测的mask保存为灰度图像。

输入JSONL格式:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    与输入图像对应的灰度mask图像，保存到指定目录
    - 0 (黑色): 背景
    - 255 (白色): 前景

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --device cuda

高级用法:
    python seg-rl/sam2_segment_from_points.py \\
      --input_jsonl /path/to/points.jsonl \\
      --output_dir /path/to/masks \\
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \\
      --device cuda \\
      --resize 512 512 \\
      --skip_existing
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

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


class SAMWrapper:
    """SAM2包装器，用于图像分割预测"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """初始化SAM2模型和预测器
        
        Args:
            model_path: SAM2模型检查点路径
            device: 运行设备 (e.g. "cuda", "cuda:0", "cpu")
                   如果为None，将自动检测可用设备
        """
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        # 检查配置文件是否存在
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / model_cfg
        if not config_path.exists():
            # 尝试相对于当前目录
            config_path = Path(model_cfg)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        
        sam_model = build_sam2(str(config_path), model_path)
        
        # 自动检测设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # 移动到指定设备
        sam_model = sam_model.to(self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, 
                image: PILImage.Image, 
                points: Optional[List[Tuple[int, int]]] = None, 
                labels: Optional[List[int]] = None,
                bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """使用给定提示运行SAM2预测
        
        Args:
            image: 输入PIL图像
            points: 点坐标列表 [(x,y), ...]
            labels: 点标签列表 (1=前景, 0=背景)
            bbox: 可选边界框 [x1,y1,x2,y2]
            
        Returns:
            (predicted_mask, confidence_score)的元组
        """
        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bboxes = np.array([bbox]) if bbox else None

        # 转换为numpy数组
        image_np = np.array(image)
        
        # 确保是RGB格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_np
        
        # 设置图像
        self.predictor.set_image(rgb_image)
        
        # 预测
        mask_pred, score, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bboxes,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签
    
    Args:
        obj: JSON对象
        line_num: 行号（用于错误报告）
        
    Returns:
        (is_valid, points, labels)的元组
    """
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
    """生成输出mask文件路径
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像
    
    Args:
        mask: 二值mask数组
        output_path: 输出文件路径
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--config_path", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="Path to SAM2 config file")
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
