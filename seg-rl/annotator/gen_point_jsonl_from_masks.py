#!/usr/bin/env python3

"""
Generate JSONL supervision (image,x,y) by computing a "deep interior" point of
binary masks using the Euclidean Distance Transform (EDT) maximum.

Each output line is a JSON object like:
  {"image": "/abs/path/to/image.jpg", "gt_mask": "/abs/path/to/mask.png", "points": [[123.4,456.7]], "labels": [1]}

Coordinates are in pixel space of the original image/mask pair.
Designed for Seg-R1 heatmap-classification pretraining pipelines.

Example usage:
  # 1) 首次生成（计算首个最佳提示点）
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --images_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir  /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251002.jsonl

  # 2) 追加模式（为每条记录计算后续的第2个及以后提示点）
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl

  # 3) 调试模式（生成调试图像）
  # 若 sam2_segment_from_points.py 已生成 heatmap-{i}.png，本脚本会在调试图中追加第4幅：
  # “概率热力图 + 当前提示点”，用于核对提示点与热区的一致性。
  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --debug_json outputs/braintumour/pred_251023.jsonl \
    --debug_output_dir outputs/braintumour/dbg_gen_points

Notes:
- Foreground is defined as any non-zero pixel in the mask.
- Target point is the pixel farthest from the background (EDT argmax), which
  is guaranteed to lie inside the foreground, robust to holes and thin parts.
- Samples with empty masks are skipped by default (override with --skip_empty false).
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
# from PIL import Image, ImageDraw
from PIL import Image as _PIL
from scipy.ndimage import label, distance_transform_edt
import cv2  # type: ignore

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create/Append deepest-point prompts from masks using EDT")
    # 互斥：首写输出 vs 追加到已存在的JSON数组
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_jsonl", type=str, help="Path to write JSONL lines for first prompts")
    group.add_argument("--appendto_jsonl", type=str, help="Path to existing JSON array to append next prompts")
    group.add_argument("--debug_json", type=str, help="Path to JSON array to read for debug visualization only")

    # 首次生成模式需要的参数（在追加模式下将被忽略）
    p.add_argument("--images_dir", type=str, default=None, help="Directory containing images (e.g., JPG)")
    p.add_argument("--masks_dir", type=str, default=None, help="Directory containing PNG masks aligned to images")
    p.add_argument("--abs_paths", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                   help="Write absolute image paths (default: True)")
    p.add_argument("--skip_empty", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                   help="Skip samples where mask is empty (default: True)")
    p.add_argument("--image_exts", type=str, default=".jpg,.jpeg,.png",
                   help="Comma-separated acceptable image extensions (default: .jpg,.jpeg,.png)")
    p.add_argument("--viz_dir", type=str, default=None, help="Optional directory to save overlay visualizations")
    p.add_argument("--viz_alpha", type=float, default=0.35, help="Alpha for mask overlay (0-1)")
    p.add_argument("--viz_radius", type=int, default=4, help="Radius (pixels) for deepest-point circle")
    p.add_argument("--debug_output_dir", type=str, default=None,
                   help="If set, write per-sample debug images for diff component selection")
    return p.parse_args()


def list_files_with_exts(directory: str, allowed_exts: Tuple[str, ...]) -> List[str]:
    directory = to_abs(directory) or directory
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    out: List[str] = []
    for name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, name)):
            continue
        if any(name.lower().endswith(ext) for ext in allowed_exts):
            out.append(name)
    return sorted(out)


def stem(path: str) -> str:
    name = os.path.basename(path)
    # Remove common two-part extensions like .nii.gz by iterative splitext
    base, ext = os.path.splitext(name)
    return base


def to_abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path) if not os.path.isabs(path) else path


def _find_latest_mask_path(sam_masks_dir: str, image_stem: str, current_points_len: int) -> Optional[str]:
    """返回该图像在sam_masks_dir下的最新预测mask路径。
    优先尝试 sam_masks_dir/<stem>/<current_points_len-1>.png；若不存在，
    则在该目录下寻找最大数字文件名的png。
    """
    candidate_dir = os.path.join(sam_masks_dir, image_stem)
    if not os.path.isdir(candidate_dir):
        return None
    k = max(0, int(current_points_len - 1))
    preferred = os.path.join(candidate_dir, f"{k}.png")
    if os.path.isfile(preferred):
        return preferred
    # fallback: find max numeric png
    best_idx = -1
    best_path = None
    for name in os.listdir(candidate_dir):
        if not name.lower().endswith(".png"):
            continue
        s = os.path.splitext(name)[0]
        try:
            idx = int(s)
        except Exception:
            continue
        if idx > best_idx:
            best_idx = idx
            best_path = os.path.join(candidate_dir, name)
    return best_path

from scipy import ndimage as ndi

def farthest_point_from_boundary(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    # mask: 二值数组，区域为 True/1，背景为 False/0
    mask = (mask > 0)
    dist = ndi.distance_transform_edt(mask)  # 区域内每点到背景的欧氏距离
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    # r = float(dist[y, x])  # 到边界的最大最小距离
    # return (y, x), r
    return (float(x), float(y))

def compute_centroid(mask_u8: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute a robust interior point (x,y) for the foreground mask as the
    Euclidean Distance Transform (EDT) maximum (i.e., the pixel farthest from
    the background). Returns None if the mask has no foreground.
    """
    if mask_u8.ndim == 3:
        # If mask has channels, convert to single channel by any non-zero across channels
        mask_u8 = (mask_u8.any(axis=2)).astype(np.uint8) * 255
    m = (mask_u8 > 0).astype(np.uint8)
    # Preprocess: keep largest connected component and fill holes for robustness
    try:
        from scipy import ndimage as ndi  # type: ignore
        if m.sum() > 0:
            labels, nlab = ndi.label(m, structure=np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8))
            if nlab > 1:
                sizes = ndi.sum(m, labels, index=range(1, nlab + 1))
                largest_idx = int(np.argmax(sizes)) + 1
                m = (labels == largest_idx).astype(np.uint8)
            # Fill holes
            m = ndi.binary_fill_holes(m > 0).astype(np.uint8)
    except Exception:
        try:
            import cv2  # type: ignore
            if m.sum() > 0:
                # Connected components with stats; background is label 0
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                if num_labels > 2:
                    # Find largest non-background component by area (stats[:, cv2.CC_STAT_AREA])
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    largest_idx = 1 + int(np.argmax(areas))
                    m = (labels == largest_idx).astype(np.uint8)
                elif num_labels == 2:
                    # Only one foreground component
                    m = (labels == 1).astype(np.uint8)
                # Fill holes via flood fill on background
                h, w = m.shape
                inv = (1 - m) * 255
                ff = inv.copy()
                mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                cv2.floodFill(ff, mask, seedPoint=(0, 0), newVal=128)
                holes = (ff != 128)  # interior background
                m = ((1 - inv/255).astype(np.uint8) | holes.astype(np.uint8))
        except Exception:
            # If no morphology libs, proceed without preprocessing
            pass
    if m.sum() == 0:
        return None
    # Prefer SciPy's true Euclidean distance; fallback to OpenCV if SciPy unavailable
    dist = None
    try:
        from scipy import ndimage as ndi  # type: ignore
        dist = ndi.distance_transform_edt(m)
    except Exception:
        try:
            import cv2  # type: ignore
            dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
        except Exception:
            # As a last resort, use cityblock (L1) via iterative distance on uint8
            # This is a coarse fallback and less accurate than EDT.
            # Here we approximate by growing distance in four-neighborhood.
            h, w = m.shape
            dist = np.zeros_like(m, dtype=np.float32)
            # Forward pass
            for y in range(h):
                for x in range(w):
                    if m[y, x] == 0:
                        continue
                    v = dist[y, x]
                    if y > 0:
                        v = max(v, dist[y - 1, x] + 1.0)
                    if x > 0:
                        v = max(v, dist[y, x - 1] + 1.0)
                    dist[y, x] = v
            # Backward pass
            for y in range(h - 1, -1, -1):
                for x in range(w - 1, -1, -1):
                    if m[y, x] == 0:
                        continue
                    v = dist[y, x]
                    if y + 1 < h:
                        v = max(v, dist[y + 1, x] + 1.0)
                    if x + 1 < w:
                        v = max(v, dist[y, x + 1] + 1.0)
                    dist[y, x] = v
    # Argmax returns (y, x)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (float(x), float(y))

def largest_diff_component_representative(A, B, connectivity=2, min_area=0, dbg_out_path: Optional[str] = None):
    # A, B: 2D uint8 arrays, values {0,1} 或 {0,255}
    A = (A > 0).astype(np.uint8)
    B = (B > 0).astype(np.uint8)
    ### C = A.astype(np.int16) - B.astype(np.int16)  # {-1, 0, +1}
    C = A.astype(np.int8) - B.astype(np.int8)  # {-1, 0, +1}

    regions = []
    # 8 邻域结构元（connectivity=2 表示 8 邻域；=1 表示 4 邻域）
    st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    for sign, mask in ((1, C > 0), (-1, C < 0)):
        if not mask.any():
            continue
        lab, n = ndi.label(mask, structure=st)
        if n == 0:
            continue
        areas = np.bincount(lab.ravel())[1:]  # 忽略背景0
        if min_area > 0:
            keep_ids = np.where(areas >= min_area)[0] + 1
            if keep_ids.size == 0:
                continue
            mask2 = np.isin(lab, keep_ids)
            lab, n = ndi.label(mask2, structure=st)
            if n == 0:
                continue
            areas = np.bincount(lab.ravel())[1:]

        k = np.argmax(areas) + 1
        comp = (lab == k)
        regions.append((sign, int(areas[k-1]), comp))

    if not regions:
        return None  # 无非零差异

    # 选面积最大的分量；若需以“最深”优先，可在 key 中加入 max EDT 作为次序
    sign, area, comp = max(regions, key=lambda t: t[1])

    # centroid = farthest_point_from_boundary(comp)
    # # centroid = compute_centroid(comp)
    # if centroid is None:
    #     return None
    D = distance_transform_edt(comp)
    (centroidY, centroidX) = np.unravel_index(np.argmax(D), D.shape)

    # 调试可视化（改用统一的渲染函数，保持风格一致）
    if dbg_out_path:
        try:
            _render_diff_panels_with_point(
                A=A,
                B=B,
                point_xy=(float(centroidX), float(centroidY)),
                label=1 if int(sign) > 0 else 0,
                dbg_out_path=dbg_out_path,
                connectivity=connectivity,
            )
        except Exception:
            pass

    return dict(sign=sign, area=area, y=int(centroidY), x=int(centroidX), max_radius=float(D[centroidY, centroidX]))


def _render_diff_panels_with_point(
    A: np.ndarray,
    B: np.ndarray,
    point_xy: Tuple[float, float],
    label: int,
    dbg_out_path: Optional[str],
    connectivity: int = 2,
    base_image_bgr: Optional[np.ndarray] = None,
    heatmap_rgb: Optional[np.ndarray] = None,
) -> None:
    """Render 3-panel debug image reusing the visualization style in
    largest_diff_component_representative, but mark a provided point instead of
    the recomputed centroid.
    - Panel 1: A (green) vs B (red) overlay
    - Panel 2: signed diff C (positive green, negative red)
    - Panel 3: connected components colored, with marker at point_xy
    - Panel 4 (optional): probability heatmap (RGB) + current point marker
    """
    rng = np.random.default_rng(12345)
    try:
        if dbg_out_path is None:
            return
        h, w = A.shape[:2]
        # Panel 1: A/B overlay (A=green, B=red), optionally blended on base image
        overlay1 = np.zeros((h, w, 3), dtype=np.uint8)
        overlay1[..., 1] = (A > 0).astype(np.uint8) * 255  # G
        overlay1[..., 2] = (B > 0).astype(np.uint8) * 255  # R
        if base_image_bgr is not None:
            base = base_image_bgr
            if base.ndim == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            if base.shape[:2] != (h, w):
                base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
            panel1 = cv2.addWeighted(base, 1.0, overlay1, 0.6, 0.0)
        else:
            bg = np.zeros_like(overlay1)
            panel1 = cv2.addWeighted(bg, 1.0, overlay1, 0.6, 0.0)

        # Panel 2: signed diff C (positive green, negative red)
        C = A.astype(np.int16) - B.astype(np.int16)
        panel2 = np.zeros((h, w, 3), dtype=np.uint8)
        pos = (C > 0)
        neg = (C < 0)
        panel2[pos, 1] = 255
        panel2[neg, 2] = 255

        # Panel 3: connected components colored + provided point marker
        panel3 = np.zeros((h, w, 3), dtype=np.uint8)
        st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
        lab, n = ndi.label(C, structure=st)
        if n != 0:
            colors = (rng.integers(0, 256, size=(n, 3))).astype(np.uint8)
            colors = np.clip(colors + 100, 0, 255)
            for i in range(1, n + 1):
                panel3[lab == i] = colors[i - 1]

        # draw marker at provided (x,y): foreground -> caret '^', background -> 'X'
        cx = int(round(float(point_xy[0])))
        cy = int(round(float(point_xy[1])))
        def _draw_cross(img, x, y, size=6, color=(255,255,255), thickness=2):
            cv2.line(img, (x - size, y - size), (x + size, y + size), (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y + size), (x + size, y - size), (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness, lineType=cv2.LINE_AA)
        def _draw_caret(img, x, y, size=6, color=(255,255,255), thickness=2):
            p_top = (int(x), int(y - size))
            p_left = (int(x - size), int(y + size))
            p_right = (int(x + size), int(y + size))
            cv2.line(img, p_left, p_top, (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, p_right, p_top, (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, p_left, p_top, color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p_right, p_top, color, thickness, lineType=cv2.LINE_AA)
        if int(label) > 0:
            _draw_caret(panel3, cx, cy, size=6, color=(255,255,255), thickness=2)
        else:
            _draw_cross(panel3, cx, cy, size=6, color=(255,255,255), thickness=2)

        # Compute and draw metrics comparing B (pred) vs A (gt) on panel1
        def _compute_metrics(gt_u8: np.ndarray, pr_u8: np.ndarray) -> dict:
            gt = (gt_u8 > 0)
            pr = (pr_u8 > 0)
            tp = int(np.logical_and(gt, pr).sum())
            fp = int(np.logical_and(~gt, pr).sum())
            fn = int(np.logical_and(gt, ~pr).sum())
            denom_dice = 2 * tp + fp + fn
            dice = (2 * tp / denom_dice) if denom_dice > 0 else 1.0
            denom_iou = tp + fp + fn
            iou = (tp / denom_iou) if denom_iou > 0 else 1.0
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Lightweight S-measure implementation adapted for binary masks
            def _s_measure_binary(pred: np.ndarray, gt: np.ndarray) -> float:
                pred_f = pred.astype(np.float64)
                gt_f = gt.astype(np.float64)
                y_mean = gt_f.mean()
                if y_mean == 0:
                    return float(1.0 - pred_f.mean())
                if y_mean == 1:
                    return float(pred_f.mean())

                def s_object(p: np.ndarray, g: np.ndarray) -> float:
                    vals = p[g == 1]
                    if vals.size == 0:
                        return 0.0
                    mx = float(vals.mean())
                    sx = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                    return float(2 * mx / (mx * mx + 1 + sx + np.spacing(1)))

                def ssim(a: np.ndarray, b: np.ndarray) -> float:
                    h_, w_ = a.shape
                    N = h_ * w_
                    if N <= 1:
                        return 1.0
                    mx = float(a.mean()); my = float(b.mean())
                    sx = float(((a - mx) ** 2).sum() / (N - 1))
                    sy = float(((b - my) ** 2).sum() / (N - 1))
                    sxy = float(((a - mx) * (b - my)).sum() / (N - 1))
                    alpha = 4 * mx * my * sxy
                    beta = (mx * mx + my * my) * (sx + sy)
                    if alpha != 0:
                        return float(alpha / (beta + np.spacing(1)))
                    if alpha == 0 and beta == 0:
                        return 1.0
                    return 0.0

                coords = np.argwhere(gt_f == 1)
                h_, w_ = gt_f.shape
                if coords.size == 0:
                    cx, cy = int(round(w_ / 2)), int(round(h_ / 2))
                else:
                    cy, cx = coords.mean(axis=0).round().astype(int).tolist()
                cx = int(cx) + 1; cy = int(cy) + 1
                gt_LT = gt_f[0:cy, 0:cx]; gt_RT = gt_f[0:cy, cx:w_]
                gt_LB = gt_f[cy:h_, 0:cx]; gt_RB = gt_f[cy:h_, cx:w_]
                pr_LT = pred_f[0:cy, 0:cx]; pr_RT = pred_f[0:cy, cx:w_]
                pr_LB = pred_f[cy:h_, 0:cx]; pr_RB = pred_f[cy:h_, cx:w_]
                area = float(h_ * w_)
                w1 = (cx * cy) / area
                w2 = (cy * (w_ - cx)) / area
                w3 = ((h_ - cy) * cx) / area
                w4 = 1.0 - w1 - w2 - w3
                region_score = (
                    w1 * ssim(pr_LT, gt_LT) +
                    w2 * ssim(pr_RT, gt_RT) +
                    w3 * ssim(pr_LB, gt_LB) +
                    w4 * ssim(pr_RB, gt_RB)
                )
                alpha = 0.5
                return float(max(0.0, alpha * s_object(pred_f, gt_f) + (1 - alpha) * region_score))

            s_measure = _s_measure_binary(pr, gt)
            return {
                "DICE": dice,
                "IOU": iou,
                "PRECISION": precision,
                "RECALL": recall,
                "S_MEASURE": s_measure,
                "F1_SCORE": f1,
            }

        def _draw_metrics_footer(img: np.ndarray, metrics: dict, keys: List[str]) -> None:
            parts: List[str] = []
            for k in keys:
                if k in metrics:
                    parts.append(f"{k}: {metrics[k]:.3f}")
            if not parts:
                return
            text = "  ".join(parts)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            max_width = img.shape[1] - 16
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            if tw > max_width and tw > 0:
                scale = max(0.3, scale * (max_width / tw))
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = 8
            y = img.shape[0] - 6
            # stroke for readability
            cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        try:
            m = _compute_metrics(A, B)
            _draw_metrics_footer(panel1, m, ["DICE", "IOU"])
            _draw_metrics_footer(panel2, m, ["PRECISION", "RECALL"])
            _draw_metrics_footer(panel3, m, ["S_MEASURE", "F1_SCORE"])
        except Exception:
            pass

        # Panel 4: probability heatmap + point (若提供 heatmap_rgb)
        if heatmap_rgb is not None:
            hm = heatmap_rgb
            if hm.ndim == 2:
                hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2BGR)
            if hm.shape[:2] != (h, w):
                hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
            panel4 = hm.copy()
            # draw current point on heatmap
            if int(label) > 0:
                _draw_caret(panel4, cx, cy, size=6, color=(255,255,255), thickness=2)
            else:
                _draw_cross(panel4, cx, cy, size=6, color=(255,255,255), thickness=2)

            # Optional overlay for context: blend grayscale base image under the heatmap
            if base_image_bgr is not None:
                base = base_image_bgr
                if base.ndim == 2:
                    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
                if base.shape[:2] != (h, w):
                    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
                base_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                alpha = 0.5
                panel4 = cv2.addWeighted(base_gray_bgr, 1.0 - alpha, panel4, alpha, 0.0)

        # Titles
        def _add_title(img: np.ndarray, text: str) -> None:
            band_h = max(20, min(48, img.shape[0] // 20))
            roi = img[0:band_h, :, :]
            overlay = roi.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1] - 1, band_h - 1), (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, 0.5, roi, 0.5, 0, dst=roi)
            org = (8, band_h - 6)
            font_scale = 0.4
            font_thickness = 1
            # Truncate text to fit within image width
            max_width = img.shape[1] - 16  # Leave some padding
            (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            if text_width > max_width:
                # Binary search to find the right truncation point
                ellipsis = "..."
                for i in range(len(text), 0, -1):
                    truncated = text[:i] + ellipsis
                    (trunc_width, _), _ = cv2.getTextSize(truncated, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    if trunc_width <= max_width:
                        text = truncated
                        break
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        _add_title(panel1, "Overlay A (green) vs B (red)")
        _add_title(panel2, "Signed diff C: + (green), - (red)")
        _add_title(panel3, "Components + selected point")
        if heatmap_rgb is not None:
            _add_title(panel4, "Probability heatmap + selected point")

        panels = [panel1, panel2, panel3]
        if heatmap_rgb is not None:
            panels.append(panel4)
        canvas = np.concatenate(panels, axis=1)
        os.makedirs(os.path.dirname(dbg_out_path) or ".", exist_ok=True)
        _PIL.fromarray(canvas[..., ::-1]).save(dbg_out_path)
    except Exception:
        pass

def find_corresponding_image(
    images_dir: str,
    sample_stem: str,
    allowed_exts: Tuple[str, ...],
) -> Optional[str]:
    images_dir_abs = to_abs(images_dir)
    for ext in allowed_exts:
        candidate = os.path.join(images_dir_abs, sample_stem + ext)
        if os.path.isfile(candidate):
            return candidate
    # Fallback: scan directory and match by stem (handles mixed-case extensions or uncommon extensions)
    for name in os.listdir(images_dir_abs):
        if stem(name).lower() == sample_stem.lower():
            return os.path.join(images_dir_abs, name)
    return None

def calculate_next_point(gt_mask_path, sam_masks_dir, current_points_len, dbg_out_dir: Optional[str] = None):
    # 读取gt与pred
    try:
        gt_u8 = np.array(_PIL.open(to_abs(gt_mask_path)), dtype=np.uint8)
        if current_points_len > 0:
            last_mask_path = _find_latest_mask_path(sam_masks_dir, stem(gt_mask_path), current_points_len)
            if not last_mask_path or not os.path.isfile(last_mask_path):
                print(f"[WARN] Skip: last mask not found under {sam_masks_dir}")
                return None, None, None
            pred_u8 = np.array(_PIL.open(to_abs(last_mask_path)), dtype=np.uint8)
        else:
            pred_u8 = np.zeros_like(gt_u8)
    except Exception as e:
        print(f"[WARN] Skip: failed to read masks ({e})")
        return None, None, None

    # 若形状不同，将pred按最近邻缩放到gt尺寸
    h_g, w_g = gt_u8.shape[:2]
    h_p, w_p = pred_u8.shape[:2]
    if (h_g, w_g) != (h_p, w_p):
        try:
            # from PIL import Image as _PIL
            pred_u8 = np.array(_PIL.fromarray(pred_u8).resize((w_g, h_g), resample=_PIL.NEAREST), dtype=np.uint8)
        except Exception:
            pass

    ### For Debug Only:
    # if stem(gt_mask_path) == "BRATS_001_z0088":
    #     print("#Debug#")
    ### :For Debug Only

    # 计算差异的最大连通分量代表点
    dbg_path = None
    if dbg_out_dir:
        try:
            subdir = os.path.join(dbg_out_dir, stem(gt_mask_path))
            os.makedirs(subdir, exist_ok=True)
            dbg_path = os.path.join(subdir, f"dbg_{max(0, int(current_points_len))}.png")
        except Exception:
            dbg_path = None
    rep = largest_diff_component_representative(gt_u8, pred_u8, connectivity=2, min_area=0, dbg_out_path=dbg_path)
    if rep is None:
        print(f"[INFO] no diff component; skip append")
        return None, None, None
    x_next = int(rep["x"])  # 列
    y_next = int(rep["y"])  # 行
    label_next = 1 if int(rep["sign"]) > 0 else 0
    return x_next, y_next, label_next

def main() -> None:
    global _PIL
    args = parse_args()

    # Debug-only mode: read JSON, generate debug images but do not write JSON
    if args.debug_json:
        if not args.debug_output_dir:
            raise RuntimeError("--debug_output_dir is required when --debug_json is set")
        json_path = args.debug_json
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Debug JSON not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        try:
            arr = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Debug JSON must be a JSON array: {e}")
        if not isinstance(arr, list):
            raise RuntimeError("Debug JSON root must be a JSON array")

        num_generated = 0
        for rec in arr:
            if not isinstance(rec, dict):
                continue

            image_path = rec.get("image")
            gt_mask_path = rec.get("gt_mask")
            sam_masks_dir = rec.get("sam_masks_dir")
            if not (image_path and sam_masks_dir):
                continue
            sample_stem = stem(image_path)
            ### For Debug Only:
            # if not sample_stem == "BRATS_001_z0088":
            #     continue
            ### :For Debug Only
            try:
                subdir = os.path.join(args.debug_output_dir, sample_stem)
                os.makedirs(subdir, exist_ok=True)
            except Exception:
                continue

            points = rec.get("points", [])
            labels = rec.get("labels", [])
            if not (isinstance(points, list) and isinstance(labels, list)):
                continue
            if len(points) != len(labels):
                continue

            gt_mask_u8 = np.array(_PIL.open(to_abs(gt_mask_path)), dtype=np.uint8)
            # Load base image if exists
            base_img_bgr = None
            try:
                img_arr = np.array(_PIL.open(to_abs(image_path)).convert("RGB"))
                base_img_bgr = img_arr[:, :, ::-1]
            except Exception:
                base_img_bgr = None
            # heatmap will be loaded per-step inside the loop (fix: previously attempted before loop)
            # Load gt as all-zero array to preserve panel structure; we visualize differences of predicted masks only
            for i, (pt, lb) in enumerate(zip(points, labels)):
                # For step i, compare masks of ground truth and step i-1 (i==0 uses empty prev)
                if i == 0:
                    prev = np.zeros_like(gt_mask_u8, dtype=np.uint8)
                else:
                    try:
                        prev_path = os.path.join(sam_masks_dir, sample_stem, f"{i-1}.png")
                        prev = np.array(_PIL.open(to_abs(prev_path)).convert("L"), dtype=np.uint8) if os.path.isfile(prev_path) else np.zeros_like(gt_mask_u8, dtype=np.uint8)
                    except Exception:
                        prev = np.zeros_like(gt_mask_u8, dtype=np.uint8)
                # Load heatmap for this step: prefer heatmap-{i}.png; fallback to heatmap-{i-1}.png if missing
                heatmap_rgb = None
                try:
                    hm_dir = os.path.join(sam_masks_dir, sample_stem)
                    cand = [f"heatmap-{i}.png"]
                    if i - 1 >= 0:
                        cand.append(f"heatmap-{i-1}.png")
                    for name in cand:
                        hm_path = os.path.join(hm_dir, name)
                        if os.path.isfile(hm_path):
                            hm = np.array(_PIL.open(to_abs(hm_path)).convert("RGB"))
                            heatmap_rgb = hm[:, :, ::-1]  # to BGR
                            break
                except Exception:
                    heatmap_rgb = None
                # Render panels with provided point
                dbg_path = os.path.join(subdir, f"dbg_{i}.png")
                _render_diff_panels_with_point(
                    gt_mask_u8,
                    prev,
                    (float(pt[0]), float(pt[1])),
                    int(lb),
                    dbg_path,
                    base_image_bgr=base_img_bgr,
                    heatmap_rgb=heatmap_rgb,
                )
                num_generated += 1
        print(f"Generated debug images for {num_generated} steps into {args.debug_output_dir}")
        return

    # 追加模式：读取已有JSON数组，计算下一提示点并写回
    if args.appendto_jsonl:
        json_path = args.appendto_jsonl
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Append target not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        try:
            arr = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Append target must be a JSON array: {e}")
        if not isinstance(arr, list):
            raise RuntimeError("Append target root must be a JSON array")

        updated = 0
        for idx, rec in enumerate(arr):
            if not isinstance(rec, dict):
                continue
            image_path = rec.get("image")
            gt_mask_path = rec.get("gt_mask")
            sam_masks_dir = rec.get("sam_masks_dir")
            points_list = rec.get("points", [])
            labels_list = rec.get("labels", [])

            if not (image_path and gt_mask_path and sam_masks_dir):
                print(f"[WARN] Skip idx {idx}: missing fields")
                continue

            x_next, y_next, label_next = calculate_next_point(
                gt_mask_path,
                sam_masks_dir,
                len(points_list),
                dbg_out_dir=args.debug_output_dir,
            )
            if x_next is None or y_next is None or label_next is None:
                continue

            # 追加points与labels
            if not isinstance(points_list, list):
                points_list = []
            if not isinstance(labels_list, list):
                labels_list = []
            points_list.append([x_next, y_next])
            labels_list.append(int(label_next))
            rec["points"] = points_list
            rec["labels"] = labels_list
            updated += 1

        # 回写同一文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(arr, f, indent=2, ensure_ascii=False)
        print(f"Appended next prompts to {json_path} (updated {updated} items)")
        return

    # 否则：首写模式，生成首个提示点（与之前逻辑一致）
    if not args.images_dir or not args.masks_dir or not args.output_jsonl:
        raise RuntimeError("images_dir, masks_dir and output_jsonl are required for initial generation mode")

    images_dir = args.images_dir
    masks_dir = args.masks_dir
    output_jsonl = args.output_jsonl
    allowed_image_exts = tuple(e.strip().lower() for e in args.image_exts.split(",") if e.strip())

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    # Visualization output disabled
    # if args.viz_dir:
    #     os.makedirs(args.viz_dir, exist_ok=True)

    # Gather masks as the source of truth
    mask_files = list_files_with_exts(masks_dir, allowed_exts=(".png",))
    if len(mask_files) == 0:
        raise RuntimeError(f"No mask PNG files found under {masks_dir}")

    num_written = 0
    num_skipped_empty = 0
    num_missing_images = 0
    num_size_mismatch = 0

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for mask_name in mask_files:
            sample_stem = stem(mask_name)
            mask_path = os.path.join(masks_dir, mask_name)
            
            # Load mask
            try:
                mask_img = _PIL.open(to_abs(mask_path))
                mask_u8 = np.array(mask_img, dtype=np.uint8)
            except Exception as e:
                print(f"[WARN] Skipping unreadable mask: {mask_path} ({e})")
                continue
            
            x_next, y_next, label_next = calculate_next_point(
                mask_path,
                None,
                0,
                dbg_out_dir=args.debug_output_dir,
            )
            if x_next is None or y_next is None or label_next is None:
                if args.skip_empty:
                    num_skipped_empty += 1
                    continue
                else:
                    # For empty masks and skip_empty=False, place centroid at image center
                    h, w = mask_u8.shape[:2]
                    centroid = (float(w - 1) / 2.0, float(h - 1) / 2.0)
            centroid = (x_next, y_next)

            # Find corresponding image
            img_path = find_corresponding_image(images_dir, sample_stem, allowed_image_exts)
            if img_path is None:
                num_missing_images += 1
                print(f"[WARN] Missing image for mask stem '{sample_stem}' in {images_dir}")
                continue
           
            image_field = os.path.abspath(img_path) if args.abs_paths else img_path
            mask_field = os.path.abspath(mask_path) if args.abs_paths else mask_path
            record = {"image": image_field, "gt_mask": mask_field, "points": [[float(centroid[0]), float(centroid[1])]], "labels": [1]}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

            # Visualization
            if False and args.viz_dir:
                try:
                    # Load image RGB
                    with _PIL.open(to_abs(img_path)) as im_rgb:
                        im_rgb = im_rgb.convert("RGB")
                        w_im, h_im = im_rgb.size
                        
                        # Ensure mask size matches visualization image
                        m = mask_u8
                        if (m.shape[1], m.shape[0]) != (w_im, h_im):
                            # resize nearest for mask
                            m_img = Image.fromarray(m)
                            m_img = m_img.resize((w_im, h_im), resample=Image.NEAREST)
                            m = np.array(m_img, dtype=np.uint8)
                        
                        # Create colored mask overlay
                        overlay = Image.new("RGBA", (w_im, h_im), (0, 0, 0, 0))
                        mask_alpha = int(max(0.0, min(1.0, args.viz_alpha)) * 255)
                        color = (255, 0, 0, mask_alpha)
                        # Build alpha mask where foreground>0
                        fg = (m > 0).astype(np.uint8) * mask_alpha
                        alpha_img = Image.fromarray(fg, mode="L")
                        color_img = Image.new("RGBA", (w_im, h_im), color)
                        overlay.paste(color_img, (0, 0), mask=alpha_img)
                        
                        # Compose overlay on image
                        comp = im_rgb.convert("RGBA")
                        comp = Image.alpha_composite(comp, overlay)
                        
                        # Draw the deepest point
                        draw = ImageDraw.Draw(comp)
                        px, py = float(centroid[0]), float(centroid[1])
                        r = max(1, int(args.viz_radius))
                        draw.ellipse((px - r, py - r, px + r, py + r), outline=(0, 255, 0, 255), width=2)
                        
                        # Save
                        out_name = f"{sample_stem}.png"
                        comp.convert("RGB").save(os.path.join(args.viz_dir, out_name))
                except Exception as e:
                    print(f"[WARN] Visualization failed for {img_path}: {e}")

    print(f"Wrote {num_written} samples to {output_jsonl}")
    if num_skipped_empty:
        print(f"Skipped empty masks: {num_skipped_empty}")
    if num_missing_images:
        print(f"Masks with missing images: {num_missing_images}")
    if num_size_mismatch:
        print(f"Image/Mask size mismatches adjusted: {num_size_mismatch}")


if __name__ == "__main__":
    main()


