#!/usr/bin/env python3

"""
Generate JSONL supervision (image,x,y) by computing a "deep interior" point of
binary masks using the Euclidean Distance Transform (EDT) maximum.

Each output line is a JSON object like:
  {"image": "/abs/path/to/image.jpg", "points": [[123.4,456.7], [124.5,457.8]], "labels": [1, 0]}

Coordinates are in pixel space of the original image/mask pair.
Designed for Seg-R1 heatmap-classification pretraining pipelines.

Example usage:
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --images_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir  /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/mask_salient_points-0.jsonl \
    --viz_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/mask_salient_points-0 \
    --viz_alpha 0.7 \
    --viz_radius 6

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create JSONL of image->deepest-point(x,y) from masks using EDT")
    p.add_argument("--images_dir", type=str, required=True, help="Directory containing images (e.g., JPG)")
    p.add_argument("--masks_dir", type=str, required=True, help="Directory containing PNG masks aligned to images")
    p.add_argument("--output_jsonl", type=str, required=True, help="Path to write JSONL output")
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
    images_dir = args.images_dir
    masks_dir = args.masks_dir
    output_jsonl = args.output_jsonl
    allowed_image_exts = tuple(e.strip().lower() for e in args.image_exts.split(",") if e.strip())

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)

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
            record = {"image": image_field, "points": [[float(centroid[0]), float(centroid[1])]], "labels": [1]}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

            # Visualization
            if args.viz_dir:
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


