#!/usr/bin/env python3

"""
Visualize heuristic prompt generation effect across prompt counts for SAM segmentation.

For each sample in --input_json, generate a single image composed of K columns, where K
is the number of prompts available for that sample. Column k (1-based) shows:
  - Grayscale original image as background
  - Semi-transparent GT mask (green)
  - Semi-transparent predicted mask for k prompts (red), loaded from sam_masks_dir/<stem>/<k-1>.png
  - The last prompt point at step k drawn: white '^' for foreground, white 'X' for background

Output per-sample image to --out_dir/<stem>.jpg

Example usage:
python seg-rl/visualization/viz_heuristic_sam_points.py \
    --input_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl \
    --out_dir outputs/heuristic_points \
    --alpha_gt 0.4 \
    --alpha_pred 0.4 \
    --marker_size 8 \
    --font_scale 0.6
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize heuristic SAM prompt sequence effects")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to JSON array with fields: image, gt_mask, sam_masks_dir, points, labels")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory to save composed visualizations as <stem>.jpg")
    p.add_argument("--alpha_gt", type=float, default=0.4, help="Alpha for GT mask overlay (green)")
    p.add_argument("--alpha_pred", type=float, default=0.4, help="Alpha for predicted mask overlay (red)")
    p.add_argument("--marker_size", type=int, default=8, help="Marker size in pixels for the last point")
    p.add_argument("--font_scale", type=float, default=0.6, help="Font scale for optional labels")
    return p.parse_args()


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def _read_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    arr = json.loads(content)
    if not isinstance(arr, list):
        raise RuntimeError("input_json must be a JSON array")
    return arr


def _load_image_bgr(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _load_mask_bool(path: str) -> Optional[np.ndarray]:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m > 0


def _resize_bool(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0


def _overlay_mask(base_bgr: np.ndarray, mask_bool: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float) -> np.ndarray:
    out = base_bgr.astype(np.float32).copy()
    color_layer = np.zeros_like(out, dtype=np.float32)
    color_layer[mask_bool.astype(bool)] = np.array(color_bgr, dtype=np.float32)
    out = np.clip(out + alpha * color_layer, 0, 255)
    return out.astype(np.uint8)


def _draw_caret(img: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int], thickness: int = 2) -> None:
    p_top = (int(x), int(y - size))
    p_left = (int(x - size), int(y + size))
    p_right = (int(x + size), int(y + size))
    cv2.line(img, p_left, p_top, color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, p_right, p_top, color, thickness, lineType=cv2.LINE_AA)


def _draw_cross(img: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int], thickness: int = 2) -> None:
    cv2.line(img, (int(x - size), int(y - size)), (int(x + size), int(y + size)), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (int(x - size), int(y + size)), (int(x + size), int(y - size)), color, thickness, lineType=cv2.LINE_AA)


def _compose_panel(image_bgr: np.ndarray,
                   gt_bool: np.ndarray,
                   pred_bool: np.ndarray,
                   last_point: Optional[Tuple[float, float]],
                   last_label: Optional[int],
                   alpha_gt: float,
                   alpha_pred: float,
                   marker_size: int) -> np.ndarray:
    # background: grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # overlay GT then Pred
    vis = _overlay_mask(bg, gt_bool, (0, 255, 0), alpha_gt)
    vis = _overlay_mask(vis, pred_bool, (0, 0, 255), alpha_pred)
    # draw last point
    if last_point is not None and last_label is not None:
        x, y = int(round(last_point[0])), int(round(last_point[1]))
        if int(last_label) == 1:
            _draw_caret(vis, x, y, marker_size, (255, 255, 255), 2)
        else:
            _draw_cross(vis, x, y, marker_size, (255, 255, 255), 2)
    return vis


def _find_pred_mask(sam_masks_dir: str, stem: str, k: int) -> Optional[str]:
    p = os.path.join(sam_masks_dir, stem, f"{k-1}.png")
    return p if os.path.isfile(p) else None


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.input_json):
        print(f"Error: input_json not found: {args.input_json}")
        return 1
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        records = _read_json_array(args.input_json)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return 1

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        image_path = rec.get("image")
        gt_mask_path = rec.get("gt_mask")
        sam_masks_dir = rec.get("sam_masks_dir")
        points = rec.get("points", [])
        labels = rec.get("labels", [])
        if not (image_path and gt_mask_path and sam_masks_dir):
            print(f"[WARN] skip idx {idx}: missing fields")
            continue

        stem = _stem(image_path)
        img = _load_image_bgr(image_path)
        gt = _load_mask_bool(gt_mask_path)
        if img is None or gt is None:
            print(f"[WARN] skip {stem}: failed to load image or gt mask")
            continue
        h, w = img.shape[:2]
        if gt.shape[:2] != (h, w):
            gt = _resize_bool(gt, w, h)

        panels: List[np.ndarray] = []
        K = len(points)
        for k in range(1, K + 1):
            pred_path = _find_pred_mask(sam_masks_dir, stem, k)
            if pred_path is None:
                continue
            pred = _load_mask_bool(pred_path)
            if pred is None:
                continue
            if pred.shape[:2] != (h, w):
                pred = _resize_bool(pred, w, h)

            # last point and label for this k
            last_pt = None
            last_lb = None
            if len(points) >= k:
                pt = points[k - 1]
                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                    last_pt = (float(pt[0]), float(pt[1]))
            if len(labels) >= k:
                try:
                    last_lb = int(labels[k - 1])
                except Exception:
                    last_lb = None

            panel = _compose_panel(
                image_bgr=img,
                gt_bool=gt,
                pred_bool=pred,
                last_point=last_pt,
                last_label=last_lb,
                alpha_gt=args.alpha_gt,
                alpha_pred=args.alpha_pred,
                marker_size=args.marker_size,
            )
            panels.append(panel)

        if not panels:
            print(f"[WARN] no panels for {stem}")
            continue

        # Concatenate horizontally
        canvas = np.concatenate(panels, axis=1)
        out_path = os.path.join(args.out_dir, f"{stem}.jpg")
        cv2.imwrite(out_path, canvas)
        print(f"wrote: {out_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


