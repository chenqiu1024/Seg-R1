#!/usr/bin/env python3

"""
Evaluate SAM-predicted masks against ground-truth masks.

该脚本会循环评估从1到N的所有提示点序列长度，统计每个长度对应的平均性能指标，
并绘制指标随提示点数量变化的曲线图。

Inputs:
  --input_json: Path to JSON array with entries like:
    [
      {"image": "/path/img.jpg", "gt_mask": "/path/gt.png", "sam_masks_dir": "/out/masks",
       "points": [[x1,y1], ...], "labels": [1, 0, ...]},
      ...
    ]

  --max_prompts (required, int > 0): 最大提示点数量N，脚本会评估从1到N的所有提示点序列长度

  --output_plot (optional): 输出曲线图的文件路径（默认: <input_json>_metrics_curve.png）

Outputs:
  - 控制台输出：每个提示点序列长度对应的平均性能指标（DICE, IoU, Precision, Recall, F1, S_MEASURE）
  - 曲线图：保存为PNG文件，显示各指标随提示点数量增长的变化曲线
  - 每个样本的指标文件（per-sample）：
    <sam_masks_dir>/<stem>-metrics.jsonl

Example:
 python seg-rl/evaluation/eval_sam_masks.py \
   --input_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-1.jsonl \
   --max_prompts 8 \
   --output_plot outputs/braintumour/metrics_curve.png
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate predicted masks vs ground truth for multiple prompt counts")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to JSON array file with image, gt_mask, sam_masks_dir entries")
    p.add_argument("--max_prompts", type=int, required=True,
                   help="Maximum number of prompts N. Script will evaluate from 1 to N prompts")
    p.add_argument("--output_plot", type=str, default=None,
                   help="Output path for metrics curve plot (default: <input_json>_metrics_curve.png)")
    return p.parse_args()


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def _find_pred_mask(sam_masks_dir: str, image_path: str, num_prompts: Optional[int]) -> Tuple[Optional[str], Optional[int]]:
    """Return (mask_path, effective_num_prompts) for the sample.
    If num_prompts provided, look for <sam_masks_dir>/<stem>/<num_prompts-1>.png.
    Else, find the highest numeric PNG index in that directory and return k+1.
    """
    stem = _stem(image_path)
    d = os.path.join(sam_masks_dir, stem)
    if not os.path.isdir(d):
        return None, None
    if num_prompts is not None and num_prompts > 0:
        k = num_prompts - 1
        p = os.path.join(d, f"{k}.png")
        return (p if os.path.isfile(p) else None), (k + 1 if os.path.isfile(p) else None)
    # fallback: scan for max index
    best_idx = -1
    for name in os.listdir(d):
        if not name.lower().endswith(".png"):
            continue
        s = os.path.splitext(name)[0]
        try:
            idx = int(s)
        except Exception:
            continue
        if idx > best_idx:
            best_idx = idx
    if best_idx < 0:
        return None, None
    p = os.path.join(d, f"{best_idx}.png")
    return p, best_idx + 1


def _load_mask_as_bool(path: str) -> np.ndarray:
    """Load mask image and return boolean array (True for foreground)."""
    arr = np.array(Image.open(path), dtype=np.uint8)
    if arr.ndim == 3:
        arr = (arr.any(axis=2)).astype(np.uint8)
    return arr > 0


def _resize_to(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.array(Image.fromarray(arr.astype(np.uint8) * 255).resize((w, h), resample=Image.NEAREST)) > 0


def _s_measure(pred_mask: np.ndarray, gt_mask: np.ndarray, alpha: float = 0.5) -> float:
    """S-measure (structure measure) adapted from mask_comparison.py."""
    pred = (pred_mask > 0).astype(np.float32)
    gt = (gt_mask > 0).astype(np.float32)

    def ssim_object(p: np.ndarray, g: np.ndarray) -> float:
        fg_p = p * g
        fg_g = g
        if np.sum(fg_g) == 0:
            return 1.0 if np.sum(fg_p) == 0 else 0.0
        mean_p = np.mean(fg_p)
        mean_g = np.mean(fg_g)
        var_p = np.var(fg_p)
        var_g = np.var(fg_g)
        cov = np.mean((fg_p - mean_p) * (fg_g - mean_g))
        c1, c2 = 0.01, 0.03
        denom = (mean_p**2 + mean_g**2 + c1) * (var_p + var_g + c2)
        if denom == 0:
            return 0.0
        ssim = ((2 * mean_p * mean_g + c1) * (2 * cov + c2)) / denom
        return float(max(0.0, ssim))

    def ssim_region(p: np.ndarray, g: np.ndarray) -> float:
        h, w = p.shape
        if h < 2 or w < 2:
            return float(np.mean(p == g))
        h_mid, w_mid = h // 2, w // 2
        pairs = [
            (p[:h_mid, :w_mid], g[:h_mid, :w_mid]),
            (p[:h_mid, w_mid:], g[:h_mid, w_mid:]),
            (p[h_mid:, :w_mid], g[h_mid:, :w_mid]),
            (p[h_mid:, w_mid:], g[h_mid:, w_mid:]),
        ]
        scores: List[float] = []
        for pr, gr in pairs:
            if pr.size == 0:
                continue
            mean_p = float(np.mean(pr))
            mean_g = float(np.mean(gr))
            if mean_g == 0 and mean_p == 0:
                scores.append(1.0)
            elif mean_g == 0 or mean_p == 0:
                scores.append(0.0)
            else:
                var_p = float(np.var(pr))
                var_g = float(np.var(gr))
                cov = float(np.cov(pr.flatten(), gr.flatten())[0, 1]) if pr.size > 1 else 0.0
                c1, c2 = 0.01, 0.03
                denom = (mean_p**2 + mean_g**2 + c1) * (var_p + var_g + c2)
                if denom == 0:
                    ssim = 0.0
                else:
                    ssim = ((2 * mean_p * mean_g + c1) * (2 * cov + c2)) / denom
                scores.append(max(0.0, float(ssim)))
        return float(np.mean(scores)) if scores else 0.0

    s_obj = ssim_object(pred, gt)
    s_reg = ssim_region(pred, gt)
    return float(alpha * s_obj + (1.0 - alpha) * s_reg)


def _compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute segmentation metrics on boolean foreground maps of equal shape."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum(dtype=np.float64)
    gt_sum = gt.sum(dtype=np.float64)
    pred_sum = pred.sum(dtype=np.float64)
    union = np.logical_or(gt, pred).sum(dtype=np.float64)

    dice = (2.0 * inter) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else (1.0 if union == 0 else 0.0)
    iou = inter / union if union > 0 else 1.0
    precision = inter / pred_sum if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
    recall = inter / gt_sum if gt_sum > 0 else (1.0 if pred_sum == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    s_meas = _s_measure(pred, gt)

    return {
        "DICE": float(dice),
        "IOU": float(iou),
        "PRECISION": float(precision),
        "RECALL": float(recall),
        "F1": float(f1),
        "S_MEASURE": float(s_meas),
    }


def _update_metrics_file(sam_masks_dir: str, image_path: str, num_prompts: int, metrics: Dict[str, float]) -> None:
    stem = _stem(image_path)
    out_path = os.path.join(sam_masks_dir, f"{stem}-metrics.jsonl")
    data: Dict[str, Any] = {}
    if os.path.isfile(out_path):
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        data = parsed
        except Exception:
            data = {}
    data[str(int(num_prompts))] = metrics
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _evaluate_for_num_prompts(
    arr: List[Dict[str, Any]],
    num_prompts: int
) -> Tuple[List[Dict[str, float]], int, int, int]:
    """对指定的提示点数量进行评估，返回指标列表和统计信息"""
    all_metrics: List[Dict[str, float]] = []
    processed = 0
    skipped = 0
    errors = 0

    for i, rec in enumerate(arr):
        if not isinstance(rec, dict):
            skipped += 1
            continue
        image_path = rec.get("image")
        gt_mask_path = rec.get("gt_mask")
        sam_masks_dir = rec.get("sam_masks_dir")
        if not (image_path and gt_mask_path and sam_masks_dir):
            skipped += 1
            continue

        mask_path, eff_num_prompts = _find_pred_mask(sam_masks_dir, image_path, num_prompts)
        if not mask_path or eff_num_prompts is None or not os.path.isfile(mask_path):
            skipped += 1
            continue

        try:
            gt = _load_mask_as_bool(gt_mask_path)
            pred = _load_mask_as_bool(mask_path)
            h, w = gt.shape[:2]
            if pred.shape[:2] != (h, w):
                pred = _resize_to(pred, w, h)
            metrics = _compute_metrics(gt, pred)
            _update_metrics_file(sam_masks_dir, image_path, eff_num_prompts, metrics)
            processed += 1
            all_metrics.append(metrics)
        except Exception as e:
            errors += 1

    return all_metrics, processed, skipped, errors


def _plot_metrics_curves(
    metrics_by_prompts: Dict[int, Dict[str, float]],
    output_path: str
) -> None:
    """绘制指标随提示点数量变化的曲线图"""
    if not metrics_by_prompts:
        print("[WARN] No metrics to plot")
        return

    # 获取所有指标名称
    first_metrics = next(iter(metrics_by_prompts.values()))
    metric_names = sorted(first_metrics.keys())
    num_metrics = len(metric_names)

    # 准备数据
    prompt_counts = sorted(metrics_by_prompts.keys())
    metric_values = {name: [metrics_by_prompts[n][name] for n in prompt_counts] for name in metric_names}

    # 创建图表：根据指标数量动态调整布局
    if num_metrics <= 3:
        nrows, ncols = 1, num_metrics
    elif num_metrics <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = (num_metrics + 2) // 3, 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, num_metrics))

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        ax.plot(prompt_counts, metric_values[metric_name], marker='o', linewidth=2, markersize=6, color=colors[idx])
        ax.set_xlabel('Number of Prompts', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} vs Number of Prompts', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([min(prompt_counts) - 0.5, max(prompt_counts) + 0.5])
        ax.set_ylim([0, 1.05])
    
    # 隐藏多余的子图
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图形以释放内存
    print(f"\n[INFO] Metrics curve plot saved to: {output_path}")


def main() -> int:
    args = parse_args()

    if args.max_prompts < 1:
        print(f"Error: max_prompts must be >= 1, got {args.max_prompts}")
        return 1

    if not os.path.isfile(args.input_json):
        print(f"Error: input_json not found: {args.input_json}")
        return 1

    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            arr = json.load(f)
    except Exception as e:
        print(f"Error reading JSON array: {e}")
        return 1

    if not isinstance(arr, list):
        print("Error: input_json must contain a JSON array")
        return 1

    # 确定输出图表路径
    if args.output_plot is None:
        input_path = Path(args.input_json)
        output_plot = str(input_path.parent / f"{input_path.stem}_metrics_curve.png")
    else:
        output_plot = args.output_plot
    os.makedirs(os.path.dirname(output_plot) or ".", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Evaluating metrics for prompts from 1 to {args.max_prompts}")
    print(f"Total samples: {len(arr)}")
    print(f"Output plot: {output_plot}")
    print(f"{'='*60}\n")

    # 存储每个提示点数量对应的平均指标
    metrics_by_prompts: Dict[int, Dict[str, float]] = {}
    total_processed = 0
    total_skipped = 0
    total_errors = 0

    # 循环评估从1到max_prompts的所有提示点数量
    for num_prompts in range(1, args.max_prompts + 1):
        print(f"[评估中] num_prompts = {num_prompts}/{args.max_prompts}...", end="", flush=True)
        
        all_metrics, processed, skipped, errors = _evaluate_for_num_prompts(arr, num_prompts)
        
        total_processed += processed
        total_skipped += skipped
        total_errors += errors

        if all_metrics:
            # 计算平均指标
            keys = sorted(all_metrics[0].keys())
            means = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
            metrics_by_prompts[num_prompts] = means
            print(f" 完成 (processed={processed}, skipped={skipped}, errors={errors})")
        else:
            print(f" 跳过 (无有效数据: processed={processed}, skipped={skipped}, errors={errors})")

    # 输出每个提示点数量对应的平均指标
    print(f"\n{'='*60}")
    print("Average Metrics by Number of Prompts:")
    print(f"{'='*60}")
    print(f"{'Num Prompts':<12} {'DICE':<8} {'IOU':<8} {'PRECISION':<10} {'RECALL':<8} {'F1':<8} {'S_MEASURE':<10}")
    print("-" * 70)

    for num_prompts in sorted(metrics_by_prompts.keys()):
        m = metrics_by_prompts[num_prompts]
        print(f"{num_prompts:<12} {m['DICE']:<8.4f} {m['IOU']:<8.4f} {m['PRECISION']:<10.4f} "
              f"{m['RECALL']:<8.4f} {m['F1']:<8.4f} {m['S_MEASURE']:<10.4f}")

    print(f"{'='*60}")

    # 绘制曲线图
    if metrics_by_prompts:
        _plot_metrics_curves(metrics_by_prompts, output_plot)

    print(f"\nDone. Total: processed={total_processed} skipped={total_skipped} errors={total_errors}")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


