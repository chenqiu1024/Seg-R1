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

  # 2) 追加模式（为每条记录计算后续的第2个及以后提示点）
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl /root/datasets/segrl_pretrain_braintumour.jsonl

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
from PIL import Image, ImageDraw
from scipy.ndimage import label, distance_transform_edt

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create/Append deepest-point prompts from masks using EDT")
    # 互斥：首写输出 vs 追加到已存在的JSON数组
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_jsonl", type=str, help="Path to write JSONL lines for first prompts")
    group.add_argument("--appendto_jsonl", type=str, help="Path to existing JSON array to append next prompts")

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
    return p.parse_args()


def list_files_with_exts(directory: str, allowed_exts: Tuple[str, ...]) -> List[str]:
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


def _find_latest_mask_path(sam_masks_dir: str, image_path: str, current_points_len: int) -> Optional[str]:
    """返回该图像在sam_masks_dir下的最新预测mask路径。
    优先尝试 sam_masks_dir/<stem>/<current_points_len-1>.png；若不存在，
    则在该目录下寻找最大数字文件名的png。
    """
    image_stem = stem(image_path)
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

def largest_diff_component_representative(A, B, connectivity=2, min_area=0):
    # A, B: 2D uint8 arrays, values {0,1} 或 {0,255}
    A = (A > 0).astype(np.uint8)
    B = (B > 0).astype(np.uint8)
    C = A.astype(np.int8) - B.astype(np.int8)  # {-1, 0, +1}

    regions = []
    # 8 邻域结构元（connectivity=2 表示 8 邻域；=1 表示 4 邻域）
    st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    for sign, mask in ((1, C > 0), (-1, C < 0)):
        if not mask.any():
            continue
        lab, n = label(mask, structure=st)
        if n == 0:
            continue
        areas = np.bincount(lab.ravel())[1:]  # 忽略背景0
        if min_area > 0:
            keep_ids = np.where(areas >= min_area)[0] + 1
            if keep_ids.size == 0:
                continue
            mask2 = np.isin(lab, keep_ids)
            lab, n = label(mask2, structure=st)
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

    D = distance_transform_edt(comp)
    y, x = np.unravel_index(np.argmax(D), D.shape)
    return dict(sign=sign, area=area, y=int(y), x=int(x), max_radius=float(D[y, x]))

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


def find_corresponding_image(
    images_dir: str,
    sample_stem: str,
    allowed_exts: Tuple[str, ...],
) -> Optional[str]:
    for ext in allowed_exts:
        candidate = os.path.join(images_dir, sample_stem + ext)
        if os.path.isfile(candidate):
            return candidate
    # Fallback: scan directory and match by stem (handles mixed-case extensions or uncommon extensions)
    for name in os.listdir(images_dir):
        if stem(name).lower() == sample_stem.lower():
            return os.path.join(images_dir, name)
    return None


def main() -> None:
    args = parse_args()

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

            # 找最新预测mask
            last_mask_path = _find_latest_mask_path(sam_masks_dir, image_path, len(points_list))
            if not last_mask_path or not os.path.isfile(last_mask_path):
                print(f"[WARN] Skip idx {idx}: last mask not found under {sam_masks_dir}")
                continue

            # 读取gt与pred
            try:
                gt_u8 = np.array(Image.open(gt_mask_path), dtype=np.uint8)
                pred_u8 = np.array(Image.open(last_mask_path), dtype=np.uint8)
            except Exception as e:
                print(f"[WARN] Skip idx {idx}: failed to read masks ({e})")
                continue

            # 若形状不同，将pred按最近邻缩放到gt尺寸
            h_g, w_g = gt_u8.shape[:2]
            h_p, w_p = pred_u8.shape[:2]
            if (h_g, w_g) != (h_p, w_p):
                try:
                    from PIL import Image as _PIL
                    pred_u8 = np.array(_PIL.fromarray(pred_u8).resize((w_g, h_g), resample=Image.NEAREST), dtype=np.uint8)
                except Exception:
                    pass

            # 计算差异的最大连通分量代表点
            rep = largest_diff_component_representative(gt_u8, pred_u8, connectivity=2, min_area=0)
            if rep is None:
                print(f"[INFO] idx {idx}: no diff component; skip append")
                continue
            x_next = int(rep["x"])  # 列
            y_next = int(rep["y"])  # 行
            label_next = 1 if int(rep["sign"]) > 0 else 0

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
                mask_img = Image.open(mask_path)
                mask_u8 = np.array(mask_img, dtype=np.uint8)
            except Exception as e:
                print(f"[WARN] Skipping unreadable mask: {mask_path} ({e})")
                continue
            
            # Compute centroid
            centroid = compute_centroid(mask_u8)
            if centroid is None:
                if args.skip_empty:
                    num_skipped_empty += 1
                    continue
                else:
                    # For empty masks and skip_empty=False, place centroid at image center
                    h, w = mask_u8.shape[:2]
                    centroid = (float(w - 1) / 2.0, float(h - 1) / 2.0)
            
            # Find corresponding image
            img_path = find_corresponding_image(images_dir, sample_stem, allowed_image_exts)
            if img_path is None:
                num_missing_images += 1
                print(f"[WARN] Missing image for mask stem '{sample_stem}' in {images_dir}")
                continue
            
            # Optional: verify size match and warn if not
            try:
                with Image.open(img_path) as im:
                    w_im, h_im = im.size
                h_m, w_m = mask_u8.shape[:2]
                if (w_im, h_im) != (w_m, h_m):
                    num_size_mismatch += 1
                    # Map centroid from mask to image coordinate if sizes differ
                    scale_x = w_im / float(max(w_m, 1))
                    scale_y = h_im / float(max(h_m, 1))
                    centroid = (centroid[0] * scale_x, centroid[1] * scale_y)
            except Exception:
                pass
            
            image_field = os.path.abspath(img_path) if args.abs_paths else img_path
            mask_field = os.path.abspath(mask_path) if args.abs_paths else mask_path
            record = {"image": image_field, "gt_mask": mask_field, "points": [[float(centroid[0]), float(centroid[1])]], "labels": [1]}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

            # Visualization
            if False and args.viz_dir:
                try:
                    # Load image RGB
                    with Image.open(img_path) as im_rgb:
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


