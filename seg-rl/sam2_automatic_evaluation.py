#!/usr/bin/env python3

"""
使用SAM2 Automatic(EVERYTHING)模式批量分割，并基于规范化JSON描述生成标准化输出以便评估与可视化

功能:
- 根据 --refer_json 指定的JSON数组描述，按EVERYTHING模式对每个样本进行分割
- 使用每个候选区域的包围框与该样本真值mask的包围框计算IoU，选择IoU最大的区域作为预测mask
- 将预测mask与包围框以规范形式写入 --output_dir：
  - <output_dir>/<stem>/0.png
  - <output_dir>/<stem>.jsonl  内容：{"count":1, "bboxes":[[x1,y1,x2,y2]]}

判定规则:
1. 对每张原图使用SAM2 Automatic产生多个区域
2. 将每个区域的 [x,y,w,h] 转为 [x1,y1,x2,y2]
3. 与真值mask的包围框计算IoU，取IoU最大的区域作为预测mask（若无区域，跳过该样本）

输出可视化:
- 本脚本不再负责可视化；请使用 visualization/viz_sam_segmentation.py 对标准化结果进行可视化

用法:
    python seg-rl/sam2_automatic_evaluation.py \
      --refer_json /root/datasets/segrl_pretrain_braintumour.jsonl \
      --output_dir /root/outputs/sam_automatic \
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
      --device cuda

示例:
    python seg-rl/sam2_automatic_evaluation.py \
      --refer_json /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --output_dir /root/autodl-tmp/outputs/seg_r1_md/Task01_BrainTumour/sam_automatic \
      --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import time

import cv2
import numpy as np
from PIL import Image as PILImage
import torch

# 可选的matplotlib导入
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not found. Visualization features will be disabled.")
    HAS_MATPLOTLIB = False

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


# 支持的图片格式
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask之间的IoU
    
    Args:
        mask1: 第一个mask (binary)
        mask2: 第二个mask (binary)
        
    Returns:
        IoU值 (0-1)
    """
    # 确保是二值mask
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 计算交集和并集
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个mask之间的DICE系数
    
    Args:
        mask1: 第一个mask (binary)
        mask2: 第二个mask (binary)
        
    Returns:
        DICE值 (0-1)
    """
    # 确保是二值mask
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 计算交集
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    total = mask1_bin.sum() + mask2_bin.sum()
    
    if total == 0:
        return 0.0
    
    return 2.0 * float(intersection) / float(total)


def bbox_overlap(bbox1: List[int], bbox2: List[int]) -> bool:
    """检查两个包围盒是否有重叠
    
    Args:
        bbox1: [x_min, y_min, x_max, y_max]
        bbox2: [x_min, y_min, x_max, y_max]
        
    Returns:
        是否有重叠
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 检查是否不重叠
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return False
    
    return True


def _read_refer_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        arr = json.load(f)
    if not isinstance(arr, list):
        raise RuntimeError("refer_json must be a JSON array")
    return arr


def _compute_bbox_from_mask(mask: np.ndarray) -> List[int]:
    mask_bin = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask_bin)
    if ys.size == 0 or xs.size == 0:
        return [0, 0, 0, 0]
    x1 = int(xs.min()); x2 = int(xs.max())
    y1 = int(ys.min()); y2 = int(ys.max())
    return [x1, y1, x2, y2]


def load_mask_image(mask_path: str) -> Optional[np.ndarray]:
    """加载mask图像
    
    Args:
        mask_path: mask文件路径
        
    Returns:
        mask数组或None
    """
    try:
        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found: {mask_path}")
            return None
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Warning: Failed to load mask: {mask_path}")
            return None
        
        # 如果是彩色图像，转换为灰度
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        return mask
    except Exception as e:
        print(f"Warning: Error loading mask {mask_path}: {e}")
        return None


def evaluate_image(image_path: str, gt_mask_path: str, mask_generator) -> Tuple[float, Dict[str, Any]]:
    """对单张图片运行EVERYTHING分割，按与GT bbox的IoU选择最佳区域"""
    try:
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            return 0.0, {'error': f'Failed to load image: {image_path}'}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载真值mask
        gt_mask = load_mask_image(gt_mask_path)
        if gt_mask is None:
            return 0.0, {'error': f'Failed to load ground truth mask: {gt_mask_path}'}
        # 由GT mask计算bbox
        gt_bbox = _compute_bbox_from_mask(gt_mask)
        
        # 使用SAM2 Automatic模式生成mask
        masks = mask_generator.generate(image_rgb)
        
        if not masks:
            return 0.0, {'error': 'No masks generated by SAM2', 'num_generated': 0}
        
        # 遍历所有mask，按bbox IoU选择最佳
        best_iou = 0.0
        best_mask_info = None
        candidate_count = 0
        for mask_info in masks:
            pred_bbox = mask_info['bbox']  # [x, y, w, h]
            pred_bbox_xyxy = [
                int(pred_bbox[0]),
                int(pred_bbox[1]),
                int(pred_bbox[0] + pred_bbox[2]),
                int(pred_bbox[1] + pred_bbox[3])
            ]
            # bbox IoU
            # compute IoU between gt_bbox and pred_bbox_xyxy
            x1 = max(gt_bbox[0], pred_bbox_xyxy[0])
            y1 = max(gt_bbox[1], pred_bbox_xyxy[1])
            x2 = min(gt_bbox[2], pred_bbox_xyxy[2])
            y2 = min(gt_bbox[3], pred_bbox_xyxy[3])
            inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
            area_gt = max(0, gt_bbox[2] - gt_bbox[0] + 1) * max(0, gt_bbox[3] - gt_bbox[1] + 1)
            area_pd = max(0, pred_bbox_xyxy[2] - pred_bbox_xyxy[0] + 1) * max(0, pred_bbox_xyxy[3] - pred_bbox_xyxy[1] + 1)
            union = area_gt + area_pd - inter
            iou = float(inter) / float(union) if union > 0 else 0.0
            candidate_count += 1
            if iou > best_iou:
                best_iou = iou
                best_mask_info = {
                    'mask': mask_info['segmentation'],
                    'bbox': pred_bbox_xyxy,
                    'area': mask_info.get('area', 0),
                    'stability_score': mask_info.get('stability_score', 0.0)
                }
        if best_mask_info is None:
            return 0.0, {
                'error': 'No masks found',
                'num_generated': len(masks),
                'num_candidates': candidate_count
            }
        return best_iou, {
            'num_generated': len(masks),
            'num_candidates': candidate_count,
            'best_mask': best_mask_info,
            'all_generated_masks': masks,
            'gt_bbox': gt_bbox,
            'gt_mask': gt_mask
        }
        
    except Exception as e:
        return 0.0, {'error': f'Exception during evaluation: {str(e)}'}


def create_visualization(image_path: str, eval_info: Dict[str, Any], best_score: float, 
                        metric: str, output_path: str) -> None:
    """创建可视化图片
    
    Args:
        image_path: 原图路径
        eval_info: 评估信息
        best_score: 最佳分数
        metric: 评估指标
        output_path: 输出路径（无扩展名）
    """
    if not HAS_MATPLOTLIB:
        print(f"Skipping visualization for {Path(image_path).name} (matplotlib not available)")
        return
    
    try:
        # 加载原图
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建图1: 多区域分割展示
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax1.imshow(image_rgb)
        ax1.set_title(f'SAM2 Automatic Segmentation Results\n{len(eval_info["all_generated_masks"])} masks generated')
        
        # 显示所有生成的mask
        colors = plt.cm.Set3(np.linspace(0, 1, len(eval_info["all_generated_masks"])))
        for i, mask_info in enumerate(eval_info["all_generated_masks"]):
            mask = mask_info['segmentation']
            # 创建彩色mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = [*colors[i][:3], 0.6]  # 半透明
            ax1.imshow(colored_mask)
            
            # 添加包围盒
            bbox = mask_info['bbox']  # [x, y, w, h]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                   linewidth=1, edgecolor=colors[i], facecolor='none')
            ax1.add_patch(rect)
        
        ax1.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_path}_all_masks.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建图2: 最佳预测与真值对比
        if eval_info.get('best_mask') is not None and eval_info.get('gt_mask') is not None:
            fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
            
            # 转换为灰度图作为背景
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ax2.imshow(image_gray, cmap='gray', alpha=0.7)
            
            # 显示真值mask（绿色）
            gt_mask = eval_info['gt_mask']
            gt_colored = np.zeros((*gt_mask.shape, 4))
            gt_colored[gt_mask > 0] = [0, 1, 0, 0.5]  # 绿色半透明
            ax2.imshow(gt_colored)
            
            # 显示最佳预测mask（红色）
            best_mask = eval_info['best_mask']['mask']
            pred_colored = np.zeros((*best_mask.shape, 4))
            pred_colored[best_mask > 0] = [1, 0, 0, 0.5]  # 红色半透明
            ax2.imshow(pred_colored)
            
            # 添加包围盒
            gt_bbox = eval_info['gt_bbox']
            pred_bbox = eval_info['best_mask']['bbox']
            
            # 真值包围盒（绿色）
            gt_rect = patches.Rectangle((gt_bbox[0], gt_bbox[1]), 
                                      gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1],
                                      linewidth=2, edgecolor='green', facecolor='none', 
                                      linestyle='--', label='Ground Truth')
            ax2.add_patch(gt_rect)
            
            # 预测包围盒（红色）
            pred_rect = patches.Rectangle((pred_bbox[0], pred_bbox[1]), 
                                        pred_bbox[2]-pred_bbox[0], pred_bbox[3]-pred_bbox[1],
                                        linewidth=2, edgecolor='red', facecolor='none', 
                                        label='Best Prediction')
            ax2.add_patch(pred_rect)
            
            ax2.set_title(f'Best Prediction vs Ground Truth\n'
                         f'{metric.upper()}: {best_score:.3f} | '
                         f'Candidates: {eval_info["num_candidates"]}/{eval_info["num_generated"]}')
            ax2.legend()
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_path}_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
        
    except Exception as e:
        print(f"Warning: Failed to create visualization for {image_path}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM2 EVERYTHING mode and save standardized outputs for evaluation/visualization"
    )
    parser.add_argument(
        "--refer_json",
        type=str,
        required=True,
        help="Path to JSON array describing samples: image, gt_mask, sam_masks_dir (ignored here)"
    )
    parser.add_argument(
        "--sam_checkpoint", 
        type=str, 
        required=True,
        help="Path to SAM2 model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory for standardized masks and bbox logs"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run on (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--points_per_side", 
        type=int, 
        default=32,
        help="Number of points per side for automatic mask generation"
    )
    parser.add_argument(
        "--pred_iou_thresh", 
        type=float, 
        default=0.88,
        help="IoU threshold for mask prediction"
    )
    parser.add_argument(
        "--stability_score_thresh", 
        type=float, 
        default=0.95,
        help="Stability score threshold for mask filtering"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed processing information"
    )
    parser.add_argument(
        "--no_visualization", 
        action="store_true",
        help="Skip creating visualization images"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入
    if not os.path.isfile(args.refer_json):
        print(f"Error: refer_json not found: {args.refer_json}")
        return 1
    
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取refer_json数组
    try:
        records = _read_refer_json(args.refer_json)
    except Exception as e:
        print(f"Error reading refer_json: {e}")
        return 1
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {args.sam_checkpoint}")
    
    # 自动检测设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model = build_sam2(model_cfg, args.sam_checkpoint, device=device)
        # config_path = Path(__file__).parent.parent / "third_party" / "sam2" / args.config_path
        # if not config_path.exists():
        #     config_path = Path(args.config_path)
        #     if not config_path.exists():
        #         raise FileNotFoundError(f"SAM2 config file not found: {args.config_path}") 
        # sam_model = build_sam2(str(config_path), args.sam_checkpoint)
        sam_model = sam_model.to(device)
        
        # 创建自动mask生成器
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam_model,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
        )
        
        print("SAM2 initialized successfully")
        
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 评估所有图片
    print("\nStarting SAM2 EVERYTHING inference and export...")
    
    num_processed = 0
    num_errors = 0
    start_time = time.time()
    
    for i, rec in enumerate(records, 1):
        if not isinstance(rec, dict):
            continue
        image_path = rec.get('image')
        gt_mask_path = rec.get('gt_mask')
        if not (image_path and gt_mask_path):
            num_errors += 1
            continue
        stem = Path(image_path).stem
        out_mask_dir = os.path.join(args.output_dir, stem)
        os.makedirs(out_mask_dir, exist_ok=True)
        mask_png_path = os.path.join(out_mask_dir, '0.png')
        bbox_json_path = os.path.join(args.output_dir, f"{stem}.jsonl")

        score, eval_info = evaluate_image(image_path, gt_mask_path, mask_generator)
        if 'error' in eval_info:
            if args.verbose:
                print(f"[{i}/{len(records)}] {stem}: {eval_info['error']}")
            num_errors += 1
            continue

        best = eval_info.get('best_mask')
        if best is None or best.get('mask') is None:
            num_errors += 1
            continue

        # 保存预测mask为灰度png（0/255）
        mask_bin = (best['mask'] > 0).astype(np.uint8) * 255
        cv2.imwrite(mask_png_path, mask_bin)

        # 写bbox日志（单对象字典）
        bbox = best.get('bbox', [0, 0, 0, 0])
        rec_obj = {"count": 1, "bboxes": [[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]]}
        with open(bbox_json_path, 'w', encoding='utf-8') as f:
            json.dump(rec_obj, f, ensure_ascii=False)

        num_processed += 1
        if args.verbose:
            print(f"[{i}/{len(records)}] {stem}: saved 0.png and bbox json")

    elapsed = time.time() - start_time
    print(f"\nDone. processed={num_processed} errors={num_errors} in {elapsed:.1f}s")
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
