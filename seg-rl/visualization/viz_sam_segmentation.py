#!/usr/bin/env python3

"""
Batch visualization of SAM segmentation results per number of prompts.

Usage:
  python seg-rl/visualization/viz_sam_segmentation.py \
    --input_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl \
    --output_dir outputs/sam_everything \
    --alpha_gt 0.35 \
    --alpha_pred 0.35 \
    --marker_size 8 \
    --font_scale 0.5

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/visualization/viz_sam_segmentation.py \
    --input_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-1.jsonl \
    --output_dir output/braintumour/sam_everything \
    --alpha_gt 0.65 \
    --alpha_pred 0.65 \
    --marker_size 4 \
    --font_scale 0.25

Input JSON format (array):
  [
    {"image": "/path/img.jpg", "gt_mask": "/path/gt.png", "sam_masks_dir": "/out/masks",
     "points": [[x1,y1], [x2,y2], ...], "labels": [1,0,...]},
    ...
  ]

For each distinct number of prompts K observed across samples, this script creates
<output_dir>/<K>/ and writes one visualization per sample: <stem>.jpg

Each visualization contains three columns:
  1) Original image overlaid with semi-transparent GT mask (green)
  2) Original image overlaid with semi-transparent predicted mask for K prompts (red),
     and the first K prompt points drawn (FG as caret '^', BG as 'X'), numbered by order
  3) Grayscale original image overlaid with both GT (green) and predicted (red)

Additionally, metrics for the chosen K are shown as overlaid text, loaded from
<sam_masks_dir>/<stem>-metrics.jsonl if available.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize SAM segmentation results by prompt count")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to JSON array with image, gt_mask, sam_masks_dir, points, labels")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write visualization images (subdirs by prompt count)")
    p.add_argument("--alpha_gt", type=float, default=0.35, help="Alpha for GT mask overlay")
    p.add_argument("--alpha_pred", type=float, default=0.35, help="Alpha for predicted mask overlay")
    p.add_argument("--marker_size", type=int, default=8, help="Marker size in pixels for prompt points")
    p.add_argument("--font_scale", type=float, default=0.5, help="Font scale for index annotations")
    return p.parse_args()


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def _read_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    data = json.loads(content)
    if not isinstance(data, list):
        raise RuntimeError("input_json must contain a JSON array")
    return data


def _load_image_bgr(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def _load_mask_bool(path: str) -> Optional[np.ndarray]:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m > 0


def _resize_bool(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0


def _overlay_mask(image_bgr: np.ndarray, mask_bool: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float) -> np.ndarray:
    overlay = image_bgr.copy().astype(np.float32)
    color_layer = np.zeros_like(overlay, dtype=np.float32)
    color_layer[mask_bool.astype(bool)] = np.array(color_bgr, dtype=np.float32)
    out = overlay * 1.0 + (alpha * color_layer)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _draw_caret(img: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int], thickness: int = 2) -> None:
    # Draw a caret '^' with apex at (x, y)
    p_top = (int(x), int(y - size))
    p_left = (int(x - size), int(y + size))
    p_right = (int(x + size), int(y + size))
    cv2.line(img, p_left, p_top, color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, p_right, p_top, color, thickness, lineType=cv2.LINE_AA)


def _draw_cross(img: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int], thickness: int = 2) -> None:
    # Draw 'X' centered at (x, y)
    cv2.line(img, (int(x - size), int(y - size)), (int(x + size), int(y + size)), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (int(x - size), int(y + size)), (int(x + size), int(y - size)), color, thickness, lineType=cv2.LINE_AA)


def _annotate_text(img: np.ndarray, text_lines: List[str], org: Tuple[int, int],
                   font_scale: float = 0.5, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    x0, y0 = int(org[0]), int(org[1])
    for idx, line in enumerate(text_lines):
        y = y0 + idx * int(18 * font_scale + 6)
        # Draw outline for readability
        cv2.putText(img, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


def _load_metrics(sam_masks_dir: str, image_path: str) -> Dict[str, Any]:
    stem = _stem(image_path)
    p = os.path.join(sam_masks_dir, f"{stem}-metrics.jsonl")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _available_counts_for_record(rec: Dict[str, Any]) -> Set[int]:
    counts: Set[int] = set()
    sam_masks_dir = rec.get("sam_masks_dir")
    image_path = rec.get("image")
    if not (sam_masks_dir and image_path):
        return counts
    # 1) from metrics keys
    metrics = _load_metrics(sam_masks_dir, image_path)
    for k in metrics.keys():
        try:
            counts.add(int(k))
        except Exception:
            pass
    # 2) from predicted files
    d = os.path.join(sam_masks_dir, _stem(image_path))
    if os.path.isdir(d):
        for name in os.listdir(d):
            if name.lower().endswith('.png'):
                s = os.path.splitext(name)[0]
                try:
                    idx = int(s)
                    counts.add(idx + 1)
                except Exception:
                    pass
    # 3) from points length
    pts = rec.get("points", [])
    if isinstance(pts, list) and len(pts) > 0:
        counts.add(len(pts))
    return counts


def _find_pred_mask(sam_masks_dir: str, image_path: str, num_prompts: int) -> Optional[str]:
    stem = _stem(image_path)
    p = os.path.join(sam_masks_dir, stem, f"{num_prompts - 1}.png")
    return p if os.path.isfile(p) else None


def _compose_three_columns(img_bgr: np.ndarray,
                           gt_bool: np.ndarray,
                           pred_bool: np.ndarray,
                           points: List[List[float]],
                           labels: List[int],
                           k: int,
                           alpha_gt: float,
                           alpha_pred: float,
                           marker_size: int,
                           font_scale: float,
                           metrics: Dict[str, Any]) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    # Column 1: image + GT
    col1 = _overlay_mask(img_bgr, gt_bool, (0, 255, 0), alpha_gt)

    # Column 2: image + Pred + points
    col2 = _overlay_mask(img_bgr, pred_bool, (0, 0, 255), alpha_pred)
    upto = min(k, len(points), len(labels))
    for i in range(upto):
        x, y = float(points[i][0]), float(points[i][1])
        lab = int(labels[i])
        if lab == 1:
            _draw_caret(col2, int(round(x)), int(round(y)), marker_size, (0, 255, 255), 2)
        else:
            _draw_cross(col2, int(round(x)), int(round(y)), marker_size, (0, 255, 255), 2)
        # index label
        cv2.putText(col2, str(i + 1), (int(round(x + marker_size + 2)), int(round(y - marker_size - 2))),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(col2, str(i + 1), (int(round(x + marker_size + 2)), int(round(y - marker_size - 2))),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    # Column 3: grayscale image + GT + Pred
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    col3 = _overlay_mask(gray_bgr, gt_bool, (0, 255, 0), alpha_gt)
    col3 = _overlay_mask(col3, pred_bool, (0, 0, 255), alpha_pred)

    # Concatenate
    canvas = np.concatenate([col1, col2, col3], axis=1)

    # Metrics text (from metrics dict)
    lines: List[str] = []
    for key in ["DICE", "IOU", "PRECISION", "RECALL", "F1", "S_MEASURE"]:
        if key in metrics:
            lines.append(f"{key}: {metrics[key]:.4f}")
    if lines:
        _annotate_text(canvas, lines, (10, 20), font_scale=max(0.5, font_scale))

    return canvas


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.input_json):
        print(f"Error: input_json not found: {args.input_json}")
        return 1
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        records = _read_json_array(args.input_json)
    except Exception as e:
        print(f"Error reading input_json: {e}")
        return 1

    # Determine all distinct counts across records
    counts_all: Set[int] = set()
    for rec in records:
        if isinstance(rec, dict):
            counts_all |= _available_counts_for_record(rec)
    if not counts_all:
        print("No available prompt counts found. Nothing to visualize.")
        return 0

    for k in sorted(counts_all):
        out_k_dir = os.path.join(args.output_dir, str(k))
        os.makedirs(out_k_dir, exist_ok=True)

        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            image_path = rec.get("image")
            gt_mask_path = rec.get("gt_mask")
            sam_masks_dir = rec.get("sam_masks_dir")
            points = rec.get("points", [])
            labels = rec.get("labels", [])
            if not (image_path and gt_mask_path and sam_masks_dir):
                continue

            img_bgr = _load_image_bgr(image_path)
            gt_bool = _load_mask_bool(gt_mask_path)
            pred_path = _find_pred_mask(sam_masks_dir, image_path, k)
            pred_bool = _load_mask_bool(pred_path) if pred_path else None
            if img_bgr is None or gt_bool is None or pred_bool is None:
                print(f"[WARN] skip idx {idx}: missing data for k={k}")
                continue

            h, w = img_bgr.shape[:2]
            if gt_bool.shape[:2] != (h, w):
                gt_bool = _resize_bool(gt_bool, w, h)
            if pred_bool.shape[:2] != (h, w):
                pred_bool = _resize_bool(pred_bool, w, h)

            metrics_dict = _load_metrics(sam_masks_dir, image_path)
            metrics_k = metrics_dict.get(str(k), {}) if isinstance(metrics_dict, dict) else {}

            vis = _compose_three_columns(
                img_bgr=img_bgr,
                gt_bool=gt_bool,
                pred_bool=pred_bool,
                points=points,
                labels=labels,
                k=k,
                alpha_gt=args.alpha_gt,
                alpha_pred=args.alpha_pred,
                marker_size=args.marker_size,
                font_scale=args.font_scale,
                metrics=metrics_k,
            )

            out_path = os.path.join(out_k_dir, f"{_stem(image_path)}.jpg")
            cv2.imwrite(out_path, vis)
            print(f"wrote: {out_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


