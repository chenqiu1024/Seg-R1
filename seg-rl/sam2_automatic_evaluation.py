#!/usr/bin/env python3

"""
使用SAM2 Automatic模式批量分割并评估与真值mask的吻合度

功能:
- 使用SAM2 Automatic模式对指定目录下的图片进行批量分割
- 与JSON描述的真值masks进行吻合度评估
- 支持IoU和DICE指标评估
- 生成分割效果可视化图片
- 统计整个数据集的最好/最差/平均分割效果

评估方法:
1. 对每张原图使用SAM2 Automatic模式分割得到多个区域
2. 通过包围盒重叠筛选候选预测mask
3. 计算候选mask与真值mask的IoU/DICE，取最大值作为该图的评估结果
4. 统计整个数据集的分割效果

输出可视化:
- 图1: Automatic模式分割的多区域多色展示
- 图2: 最佳预测mask与真值mask叠加显示（半透明）+ 评估指标

用法:
    python seg-rl/sam2_automatic_evaluation.py \
      --image_dir /path/to/images \
      --json_file /path/to/masks.json \
      --sam_checkpoint /path/to/sam2.1_hiera_large.pt \
      --output_dir /path/to/results \
      --device cuda

示例:
    python seg-rl/sam2_automatic_evaluation.py \
      --image_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/imagesTr \
      --json_file /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pred_masks-0.json \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/evaluation_results \
      --device cuda \
      --metric dice
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


def load_ground_truth_masks(json_file: str) -> Dict[str, Dict[str, Any]]:
    """加载真值mask信息
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        字典，键为图片文件名（无扩展名），值为mask信息
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_masks = {}
    for item in data:
        mask_path = item['mask_path']
        bbox = item['bbox']
        
        # 从mask路径提取文件名（无扩展名）
        mask_filename = Path(mask_path).stem
        
        gt_masks[mask_filename] = {
            'mask_path': mask_path,
            'bbox': bbox
        }
    
    return gt_masks


def get_image_files(image_dir: str) -> List[str]:
    """获取图片目录下所有支持的图片文件
    
    Args:
        image_dir: 图片目录
        
    Returns:
        图片文件路径列表，按文件名排序
    """
    image_files = []
    
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return image_files
    
    for file_path in Path(image_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(str(file_path))
    
    image_files.sort()
    return image_files


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


def evaluate_image(image_path: str, gt_info: Dict[str, Any], mask_generator, metric: str = 'iou') -> Tuple[float, Dict[str, Any]]:
    """评估单张图片的分割效果
    
    Args:
        image_path: 图片路径
        gt_info: 真值信息
        mask_generator: SAM2 mask生成器
        metric: 评估指标 ('iou' 或 'dice')
        
    Returns:
        (最佳分数, 详细信息字典)
    """
    try:
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            return 0.0, {'error': f'Failed to load image: {image_path}'}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载真值mask
        gt_mask = load_mask_image(gt_info['mask_path'])
        if gt_mask is None:
            return 0.0, {'error': f'Failed to load ground truth mask: {gt_info["mask_path"]}'}
        
        gt_bbox = gt_info['bbox']
        
        # 使用SAM2 Automatic模式生成mask
        masks = mask_generator.generate(image_rgb)
        
        if not masks:
            return 0.0, {'error': 'No masks generated by SAM2', 'num_generated': 0}
        
        # 筛选与真值包围盒重叠的预测mask
        candidate_masks = []
        for mask_info in masks:
            pred_bbox = mask_info['bbox']  # SAM2返回的格式应该是[x, y, w, h]
            # 转换为[x_min, y_min, x_max, y_max]格式
            pred_bbox_xyxy = [
                int(pred_bbox[0]), 
                int(pred_bbox[1]), 
                int(pred_bbox[0] + pred_bbox[2]), 
                int(pred_bbox[1] + pred_bbox[3])
            ]
            
            if bbox_overlap(gt_bbox, pred_bbox_xyxy):
                candidate_masks.append({
                    'mask': mask_info['segmentation'],
                    'bbox': pred_bbox_xyxy,
                    'area': mask_info['area'],
                    'stability_score': mask_info.get('stability_score', 0.0)
                })
        
        if not candidate_masks:
            return 0.0, {
                'error': 'No overlapping masks found',
                'num_generated': len(masks),
                'num_candidates': 0
            }
        
        # 计算每个候选mask与真值的重合度
        best_score = 0.0
        best_mask_info = None
        
        for mask_info in candidate_masks:
            pred_mask = mask_info['mask']
            
            if metric == 'dice':
                score = calculate_dice(gt_mask, pred_mask)
            else:  # iou
                score = calculate_iou(gt_mask, pred_mask)
            
            if score > best_score:
                best_score = score
                best_mask_info = mask_info
        
        return best_score, {
            'num_generated': len(masks),
            'num_candidates': len(candidate_masks),
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
        description="Evaluate SAM2 Automatic segmentation against ground truth masks"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--json_file", 
        type=str, 
        required=True,
        help="JSON file containing ground truth mask information"
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
        help="Output directory for results and visualizations"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run on (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        choices=['iou', 'dice'], 
        default='iou',
        help="Evaluation metric (default: iou)"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to SAM2 config file"
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
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return 1
    
    if not os.path.isfile(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return 1
    
    if not os.path.isfile(args.sam_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载真值数据
    print(f"Loading ground truth data from: {args.json_file}")
    gt_masks = load_ground_truth_masks(args.json_file)
    print(f"Loaded {len(gt_masks)} ground truth masks")
    
    # 获取图片文件
    print(f"Scanning image directory: {args.image_dir}")
    image_files = get_image_files(args.image_dir)
    print(f"Found {len(image_files)} images")
    
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
        config_path = Path(__file__).parent.parent / "third_party" / "sam2" / args.config_path
        if not config_path.exists():
            config_path = Path(args.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"SAM2 config file not found: {args.config_path}")
        
        sam_model = build_sam2(str(config_path), args.sam_checkpoint)
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
    print(f"\nStarting evaluation with {args.metric.upper()} metric...")
    
    all_scores = []
    num_processed = 0
    num_errors = 0
    results_log = []
    
    start_time = time.time()
    
    for i, image_path in enumerate(image_files, 1):
        image_name = Path(image_path).stem
        
        if args.verbose:
            print(f"[{i}/{len(image_files)}] Processing: {image_name}")
        
        # 检查是否有对应的真值
        if image_name not in gt_masks:
            if args.verbose:
                print(f"  -> No ground truth found, skipping")
            num_errors += 1
            continue
        
        gt_info = gt_masks[image_name]
        
        # 评估图片
        score, eval_info = evaluate_image(image_path, gt_info, mask_generator, args.metric)
        
        if 'error' in eval_info:
            if args.verbose:
                print(f"  -> Error: {eval_info['error']}")
            num_errors += 1
            continue
        
        all_scores.append(score)
        num_processed += 1
        
        # 记录结果
        result_entry = {
            'image_name': image_name,
            'score': score,
            'num_generated': eval_info['num_generated'],
            'num_candidates': eval_info['num_candidates']
        }
        results_log.append(result_entry)
        
        if args.verbose:
            print(f"  -> {args.metric.upper()}: {score:.3f} "
                  f"({eval_info['num_candidates']}/{eval_info['num_generated']} candidates)")
        
        # 创建可视化（如果启用）
        if not args.no_visualization:
            vis_output_path = os.path.join(args.output_dir, image_name)
            create_visualization(image_path, eval_info, score, args.metric, vis_output_path)
    
    # 计算统计信息
    if all_scores:
        best_score = max(all_scores)
        worst_score = min(all_scores)
        avg_score = sum(all_scores) / len(all_scores)
        
        # 保存详细结果
        results_summary = {
            'metric': args.metric,
            'total_images': len(image_files),
            'processed': num_processed,
            'errors': num_errors,
            'statistics': {
                'best': best_score,
                'worst': worst_score,
                'average': avg_score
            },
            'detailed_results': results_log,
            'parameters': {
                'points_per_side': args.points_per_side,
                'pred_iou_thresh': args.pred_iou_thresh,
                'stability_score_thresh': args.stability_score_thresh
            }
        }
        
        results_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        elapsed_time = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed_time:.1f} seconds:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Processed: {num_processed}")
        print(f"  Errors: {num_errors}")
        print(f"  Metric: {args.metric.upper()}")
        print(f"  Best score: {best_score:.3f}")
        print(f"  Worst score: {worst_score:.3f}")
        print(f"  Average score: {avg_score:.3f}")
        print(f"  Results saved to: {results_file}")
        print(f"  Visualizations saved to: {args.output_dir}")
        
        return 0
    else:
        print("No images were successfully processed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
