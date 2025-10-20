#!/usr/bin/env python3

"""
Evaluate SAM-predicted masks against ground-truth masks.

Inputs:
  --input_json: Path to JSON array with entries like:
    [
      {"image": "/path/img.jpg", "gt_mask": "/path/gt.png", "sam_masks_dir": "/out/masks",
       "points": [[x1,y1], ...], "labels": [1, 0, ...]},
      ...
    ]

  --num_prompts (optional, int > 0): choose the predicted mask index=k=num_prompts-1
    If omitted, select the highest index PNG under sam_masks_dir/<stem>/.

Outputs (per-sample):
  Writes/updates a JSON dict file at:
    <sam_masks_dir>/<stem>-metrics.jsonl
  Example content:
    {
      "1": {"DICE": 0.18, "IOU": 0.09},
      "2": {"DICE": 0.93, "IOU": 0.86}
    }

 Example:
 python seg-rl/evaluation/eval_sam_masks.py --input_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-1.jsonl --num_prompts 1
 
 /opt/anaconda3/envs/seg-r1/bin/python seg-rl/evaluation/eval_sam_masks.py --input_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl --num_prompts 1
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate predicted masks vs ground truth")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to JSON array file with image, gt_mask, sam_masks_dir entries")
    p.add_argument("--num_prompts", type=int, default=None,
                   help="Number of prompts to evaluate (k = num_prompts-1). If omitted, use max available index")
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


def main() -> int:
    args = parse_args()

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

    processed = 0
    skipped = 0
    errors = 0
    all_metrics: List[Dict[str, float]] = []

    for i, rec in enumerate(arr):
        if not isinstance(rec, dict):
            skipped += 1
            continue
        image_path = rec.get("image")
        gt_mask_path = rec.get("gt_mask")
        sam_masks_dir = rec.get("sam_masks_dir")
        if not (image_path and gt_mask_path and sam_masks_dir):
            print(f"[WARN] idx {i}: missing required fields")
            skipped += 1
            continue

        mask_path, eff_num_prompts = _find_pred_mask(sam_masks_dir, image_path, args.num_prompts)
        if not mask_path or eff_num_prompts is None or not os.path.isfile(mask_path):
            print(f"[WARN] idx {i}: predicted mask not found")
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
            msg = f"ok idx {i}: {Path(image_path).stem} prompts={eff_num_prompts} | " \
                f"DICE={metrics['DICE']:.4f} IOU={metrics['IOU']:.4f} " \
                f"P={metrics['PRECISION']:.4f} R={metrics['RECALL']:.4f} F1={metrics['F1']:.4f} " \
                f"S={metrics['S_MEASURE']:.4f}"
            print(f"\r{msg}", end="", flush=True)
        except Exception as e:
            print(f"[ERROR] idx {i}: failed to evaluate - {e}")
            errors += 1

    # Dataset-wide averages
    if all_metrics:
        keys = sorted(all_metrics[0].keys())
        means = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
        print("\nDataset averages:")
        for k in keys:
            print(f"  {k}: {means[k]:.4f}")

    print(f"\nDone. processed={processed} skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


