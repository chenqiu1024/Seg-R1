# 算法选择与像素聚类
_Exported on 9/28/2025 at 14:46:47 PDT from Cursor (1.6.45)_

---

**User**

假设我们要解决这样一类问题：给出两个相同尺度的二值2D图像A和B，即二者宽高均分别为W和H个像素，每个像素只有0和1（或0和255）两种取值。每个图像上的非零值区域都有相似的特征：绝大部分非零像素形成同一个连通区域，没有或只有很少的与主要部分不连通的孤立非零像素区域。现在要计算图像A与图像B之差，即令C=A-B，显然此时C中的像素值只有3种情况：0，正值，或负值。然后用某种算法（算法1），把C中取相同值且连通的像素聚类在一起，于是就可能形成零个或多个连通的相似像素区域。接下来取其中面积最大的一个区域，用某种算法（算法2）选取该区域中最具“代表性”的像素点（即该像素点属于该区域，并且要尽量“最深入”该区域内部）作为输出。请对于上述的算法1和算法2应该分别使用什么算法，给出你能找到的最好的建议。

---

**Cursor**

### 结论（直接可用的最佳建议）
- 算法1（连通区域聚类）: 使用连通域标记（Connected-Component Labeling, CCL）。具体做法是分别对 `C>0` 与 `C<0` 两个二值掩膜做 8 邻域 CCL（建议 8 邻域以避免对角断裂），获得所有连通分量及面积统计，然后在两类里选出面积最大的那个分量作为候选。通常忽略 `C==0`（它往往是大面积“相同区域”，不是差异的关注点）。
- 算法2（选“最深入”的代表性像素）: 对选中的连通分量掩膜做欧氏距离变换（Euclidean Distance Transform, EDT），取距离值最大的像素作为输出点。这个点等价于该区域最大内切圆的圆心位置（离边界最远、最“深入”）。

### 细节与实践要点
- 预处理（可选但强烈建议）
  - 二值归一化：先把输入的 0/255 转为 0/1，再做 `C = A - B` 得到 {-1, 0, +1}。
  - 去噪与去孤岛：在对 `C>0`/`C<0` 做 CCL 前，对每个掩膜做小面积滤除（如阈值为若干像素或占图像面积的一个很小比例），或使用轻微形态学开运算，避免把极小噪点当作最大区域。
  - 孔洞填充（可选）：若区域内部存在微小孔洞，会影响 EDT；可视情况对孔洞做填充或只在 EDT 时使用外部背景作为边界。
- CCL 实现建议
  - 使用两遍/并查集（SAUF）等线性时间 CCL；工程上直接用 OpenCV `connectedComponentsWithStats` 或 scikit-image `measure.label` 即可。
  - 选择 8 邻域（更鲁棒）；如果任务明确需要严格格点连通，可改为 4 邻域。
- 代表点（EDT）实现与细节
  - 使用 `distance_transform_edt`（SciPy）或 OpenCV `distanceTransform`（L2）得到区域内到背景的欧氏距离图，取全局最大值位置即为代表点。
  - 若出现并列最大（平顶），可用次级规则打破平局：例如选距离图最大值点中距离形心最近者，或选对应最大内切圆半径更大的区域。
  - 需要亚像素精度时，可在距离图的 3×3 局部最大邻域上做二次曲面拟合得到亚像素中心。
- 复杂度
  - CCL 与 EDT 都是 O(W×H) 时间与空间复杂度，适合大图像的批量处理。
  
### 极简参考代码（NumPy + SciPy）
```python
import numpy as np
from scipy.ndimage import label, distance_transform_edt

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
```

### 可能的变体与优化
- 若任务希望分别输出“新增区域”和“缺失区域”的代表点，可分别对 `C>0` 与 `C<0` 各取最大分量与 EDT 极大点。
- 若区域存在大量锯齿/细碎边界，可先对掩膜做轻微闭运算或小孔洞填充，再做 EDT，得到更稳定的“最深点”。
- 如果最终要与矢量几何一致，可把分量边界矢量化后用“最大内切圆”或“Polylabel”思想；在栅格世界中，EDT 已是高效且精确的等价做法。

- 输出点物理意义明确：它是到外部背景的最远点，等同于该区域最大内切圆的圆心，最符合“尽量深入区域内部”的需求。

- 以上方案在准确性、速度和稳定性上是业界常用且权衡最优的做法。

```python
import numpy as np
from scipy.ndimage import label, distance_transform_edt

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
```

---

**User**

好的。下面请修改seg-rl/annotator/gen_point_jsonl_from_masks.py：1.去除（或暂时注释掉）其输出可视化图像的部分；2.使其生成的JSONL文件中的JSON符合以下形式：[
  {"image":IMAGE_PATH, "gt_mask":GT_MASK_PATH, "points":[[X0,Y0]], "labels":[L0]},
  ... ...
]  ，其中IMAGE_PATH为一个输入样本的原图像文件所在路径，gt_mask是输入样本的mask图像文件所在路径


---

**Cursor**

I’m going to add a short TODO list and start by removing or commenting out the visualization outputs in `seg-rl/annotator/gen_point_jsonl_from_masks.py`, then I’ll update the JSONL writing to match the requested schema. Next, I’ll open the file to find the visualization code and current JSON writing logic.

I'll open the file to locate visualization code and current JSON writing, then adjust them.

I'll disable the visualization writes by gating the block with a constant false condition and stop creating the viz output directory. I'll also update the JSONL record to include the `gt_mask` path (respecting the `--abs_paths` flag). Then I’ll mark the visualization task done and update the JSON schema task.

```python
#!/usr/bin/env python3

"""
Generate JSONL supervision (image,x,y) by computing a "deep interior" point of
binary masks using the Euclidean Distance Transform (EDT) maximum.

Each output line is a JSON object like:
  {"image": "/abs/path/to/image.jpg", "gt_mask": "/abs/path/to/mask.png", "points": [[123.4,456.7]], "labels": [1]}

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
from scipy.ndimage import label, distance_transform_edt

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



```

I disabled visualization output and updated the JSONL schema as requested, then verified no code issues beyond environment import warnings.

- Completed: turned off creating `viz_dir` and gated the viz block with `if False and args.viz_dir`.
- Completed: JSON lines now include `image`, `gt_mask`, `points` with one `[x, y]`, and `labels` with one integer. The `--abs_paths` flag applies to both paths.

You can run it like:
```bash
python /Users/donchiu/Works/Seg-R1/seg-rl/annotator/gen_point_jsonl_from_masks.py --images_dir ... --masks_dir ... --output_jsonl ... --abs_paths true
```

```shellscript
python /Users/donchiu/Works/Seg-R1/seg-rl/annotator/gen_point_jsonl_from_masks.py --images_dir ... --masks_dir ... --output_jsonl ... --abs_paths true
```

---

**User**

接下来修改sam2_segment_from_points.py使其满足以下要求（需要先仔细阅读原有代码，找到与新的要求有所差异的部分并修改，不要修改已有的不与新要求冲突的代码逻辑）：
  1. 它从input_jsonl参数得到的JSON文件里会包含一个如同上述segrl_pretrain_braintumour.jsonl一样形式的JSON数组，对数组中每一条，将其中的”points”和”labels”数组所表示的点坐标与相应的前/背景标签值作为提供给sam2的点提示序列，对”image”文件路径所指向的输入图像作分割，得到预测mask。在output_dir所指向的目录下（若不存在则创建），对于每个image，a) 创建一个与image文件同名的子目录，把预测mask以.png文件形式存放在其下，其文件名是如下形式: {$输入JSON中对应条目所包含points数组的长度 - 1}.png ; b) 创建一个同名的.jsonl文件其中包含一个JSON数组（若文件已存在则仅向后添加），其中的每一条目为如下形式：{“count””:{$a)所述同名子目录下mask文件的总数量}, “bboxes”:[{$第1个mask的包围盒坐标数组}, {$第2个mask的包围盒坐标数组}, …]}。
2. 最后，将input_jsonl指向的JSON文件内容拷贝到json_output所指向的JSON文件上，并在其中每条上加入一个键值对："sam_masks_dir”:$output_dir (如若已存在该键值对则覆盖)
3. 对其它已有参数的处理保持不变。
举例：假设用户以以下命令行参数执行该脚本：
python seg-rl/sam2_segment_from_points.py \ --input_jsonl /root/datasets/segrl_pretrain_braintumour.jsonl \ —json_output /root/datasets/segrl_pretrain_braintumour.jsonl \ --output_dir /root/datasets/pretrain_gt_masks \ --sam_checkpoint /root/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \ --device cuda \ --resize 512 512 \ --skip_existing
，其中/root/datasets/segrl_pretrain_braintumour.jsonl的内容如下(具体坐标数值仅为示意)：
[
{"image”:“/root/datasets/braintumour/images/BRAIN_00.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_00.jpg”, "points”:[[123,456]], "labels”:[1]},
{"image”:“/root/datasets/braintumour/images/BRAIN_01.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_01.jpg”, "points”:[[312,576]], "labels”:[1]}
]
, 那么执行后/root/datasets/segrl_pretrain_braintumour.jsonl的内容应该是：
[
{"image”:“/root/datasets/braintumour/images/BRAIN_00.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_00.jpg”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, ”points”:[[123,456]], "labels”:[1]},
{"image”:“/root/datasets/braintumour/images/BRAIN_01.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_01.jpg”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, "points”:[[312,576]], "labels”:[1]}
]
, /root/datasets/pretrain_gt_masks目录应该存在并包含以下内容：
pretrain_gt_masks/
 ├── BRAIN_00/
 ├──── 0.png
 ├── BRAIN_01/
 ├──── 0.png
 ├── BRAIN_00.jsonl
 ├── BRAIN_01.jsonl
其中每个png文件均是预测mask图像，而BRAIN_00.jsonl应具备内容(具体坐标数值仅为示意)：
{"count":1, "bboxes”:[[13,75,256,489]]}
，BRAIN_01.jsonl应具备内容(具体坐标数值仅为示意)：
{"count":1, "bboxes”:[[213,375,412,611]]}
而假如下次再以同样参数运行脚本时segrl_pretrain_braintumour.jsonl里的points和labels数组的长度为2，例如：
[
{"image”:“/root/datasets/braintumour/images/BRAIN_00.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_00.jpg”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, "points”:[[123,456],[256,312]], "labels”:[1,0]},
{"image”:“/root/datasets/braintumour/images/BRAIN_01.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_01.jpg”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, "points”:[[312,576],[168,35]], "labels”:[1,0]}
]
，则执行后pretrain_gt_masks/BRAIN_00/和pretrain_gt_masks/BRAIN_01/下就会分别多出其各自对应于提示点序列长度为2时的预测mask图像文件 1.png，并且 BRAIN_00.jsonl和BRAIN_01.jsonl里的count会变为2，bboxes数组长度也相应为2。
我的描述应该足够清晰了，接下来请按要求修改sam2_segment_from_points.py代码

---

**Cursor**

I'll open and scan the current `sam2_segment_from_points.py` to understand its parameters and flow, then implement the new behavior while preserving existing logic where possible.

I'll implement the new behavior in `seg-rl/sam2_segment_from_points.py`: adapt input reading to support a JSON array, change output structure to write masks into per-image subdirectories named by image stem with filenames based on points length minus one, compute and append bounding boxes to a per-image JSON array log, and copy the input array to `json_output` with `sam_masks_dir` set to `output_dir`. I’ll preserve other logic and parameters and only update the necessary parts.

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取由点提示组成的输入（支持JSON数组或JSONL逐行），使用SAM2进行分割，
将预测mask保存为灰度图像，并按要求生成/追加每幅图像对应的bbox日志。

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

输入JSON格式（数组）或JSONL（逐行）:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    - 预测mask按图像stem创建子目录保存为灰度图：output_dir/<stem>/<k>.png
      其中k为提示点序列长度-1；像素值：0=背景，255=前景
    - 在output_dir生成<stem>.jsonl（JSON数组）。每次运行在末尾追加一条：
      {"count": 子目录下mask文件总数, "bboxes": [[x1,y1,x2,y2], ...]}
    - 将输入JSON数组拷贝到 --json_output，并为每条加入/覆写
      "sam_masks_dir": output_dir

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/heatmap_train-0/pred_salient_points.jsonl \
      --output_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pred_masks-0 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda

带JSON输出:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/heatmap_train-0/pred_salient_points.jsonl \
      --output_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pred_masks-0 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing \
      --json_output /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pred_masks-0.jsonl

JSON输出格式:
    [
      {
        "mask_path": "/path/to/mask1.png",
        "bbox": [10, 20, 100, 150]
      },
      ...
    ]
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


def _print_progress(num_processed: int, num_skipped: int, num_errors: int, note: str) -> None:
    """在同一行输出进度信息"""
    msg = f"[Progress] processed={num_processed} skipped={num_skipped} errors={num_errors} | {note}"
    print(f"\r{msg}", end="", flush=True)


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
        sam_model = build_sam2(model_cfg, model_path) 
        # # 检查配置文件是否存在
        # config_path = Path(__file__).parent.parent / "third_party" / "sam2" / "sam2" / model_cfg
        # if not config_path.exists():
        #     # 尝试相对于当前目录
        #     config_path = Path(model_cfg)
        #     if not config_path.exists():
        #         raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        # sam_model = build_sam2(str(config_path), model_path)
        
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
    """生成输出mask文件路径（旧：单文件路径；新：保留但未使用）
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def get_mask_dir_for_image(image_path: str, output_dir: str) -> str:
    """返回该图像对应的mask子目录路径 output_dir/<stem>"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, image_name)


def get_mask_index_from_points(points: List[Tuple[float, float]]) -> int:
    """根据点序列长度返回mask索引：len(points)-1"""
    return max(0, int(len(points) - 1))


def get_mask_path_for_points(image_path: str, output_dir: str, points: List[Tuple[float, float]]) -> str:
    """输出mask路径：output_dir/<stem>/<k>.png，k=len(points)-1"""
    mask_dir = get_mask_dir_for_image(image_path, output_dir)
    os.makedirs(mask_dir, exist_ok=True)
    k = get_mask_index_from_points(points)
    return os.path.join(mask_dir, f"{k}.png")


def list_mask_pngs_sorted(mask_dir: str) -> List[str]:
    """列出目录下按数字文件名排序的png路径列表（0.png, 1.png, ...）"""
    if not os.path.isdir(mask_dir):
        return []
    files = []
    for name in os.listdir(mask_dir):
        if not name.lower().endswith('.png'):
            continue
        stem = os.path.splitext(name)[0]
        try:
            idx = int(stem)
        except ValueError:
            continue
        files.append((idx, os.path.join(mask_dir, name)))
    files.sort(key=lambda t: t[0])
    return [p for _, p in files]


def compute_bbox_from_mask_file(mask_path: str) -> Tuple[int, int, int, int]:
    """读取mask文件并计算其包围盒"""
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0, 0, 0, 0
    return calculate_bounding_box(img)


def append_state_log(output_dir: str, image_path: str) -> None:
    """构建并在 <output_dir>/<stem>.jsonl 末尾追加一次快照（JSON数组）。
    每条记录: {"count": N, "bboxes": [...]}，其中bboxes按0..N-1顺序。
    """
    stem = Path(image_path).stem
    log_path = os.path.join(output_dir, f"{stem}.jsonl")
    mask_dir = get_mask_dir_for_image(image_path, output_dir)
    mask_files = list_mask_pngs_sorted(mask_dir)
    bboxes: List[List[int]] = []
    for mp in mask_files:
        x1, y1, x2, y2 = compute_bbox_from_mask_file(mp)
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])

    new_entry = {"count": len(mask_files), "bboxes": bboxes}

    # 读取现有数组（若不存在则初始化为空数组）
    existing: List[Dict[str, Any]] = []
    if os.path.isfile(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    try:
                        existing = json.loads(content)
                        if not isinstance(existing, list):
                            # 若不是数组，尝试按JSONL逐行解析
                            existing = [json.loads(line) for line in content.splitlines() if line.strip()]
                    except json.JSONDecodeError:
                        # 按JSONL逐行
                        existing = [json.loads(line) for line in content.splitlines() if line.strip()]
        except Exception:
            existing = []

    existing.append(new_entry)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False)


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


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
                   help="Input JSON array (preferred) or JSONL file containing image paths and point coordinates")
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
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to write the copied input JSON array with added 'sam_masks_dir'")
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
    
    # 读取输入（优先按JSON数组解析；失败则按JSONL逐行解析）
    print(f"Processing input: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 解析输入记录列表 records: List[Dict]
    records: List[Dict[str, Any]] = []
    try:
        with open(args.input_jsonl, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    records = parsed
                else:
                    print("[ERROR] Input JSON root must be an array")
                    return 1
            else:
                # JSONL逐行
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                        num_errors += 1
                        continue
    except Exception as e:
        print(f"Error reading input: {e}")
        return 1

    for idx, obj in enumerate(records, 1):
        line_num = idx
            
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
            
            # 生成输出路径（子目录/<k>.png）
            output_path = get_mask_path_for_points(image_path, args.output_dir, points)
            
            # 检查是否跳过已存在的文件
            mask_existed = os.path.isfile(output_path)
            if args.skip_existing and mask_existed:
                _print_progress(num_processed, num_skipped + 1, num_errors,
                                f"skip line {line_num}: {Path(output_path).name} exists")
                # 若对应日志不存在，则仍然补齐一次日志快照
                stem_log_path = os.path.join(args.output_dir, f"{Path(image_path).stem}.jsonl")
                if not os.path.isfile(stem_log_path):
                    try:
                        append_state_log(args.output_dir, image_path)
                    except Exception:
                        pass
                num_skipped += 1
                continue
            
            try:
                # 加载图像
                image = PILImage.open(image_path).convert("RGB")
                orig_w, orig_h = image.size

                # 若指定resize，则按比例缩放点坐标并对图像进行resize
                if args.resize:
                    resize_w, resize_h = int(args.resize[0]), int(args.resize[1])
                    scale_x = float(resize_w) / float(orig_w)
                    scale_y = float(resize_h) / float(orig_h)
                    points_resized = [(px * scale_x, py * scale_y) for (px, py) in points]
                    image_for_pred = image.resize((resize_w, resize_h), PILImage.BILINEAR)
                else:
                    points_resized = points
                    image_for_pred = image
                    scale_x = 1.0
                    scale_y = 1.0

                # 运行SAM2预测（在处理后的分辨率上）
                mask, confidence = sam_wrapper.predict(image_for_pred, points_resized, labels)

                # 如果做了resize，则将mask缩放回原图尺寸，并在原图尺寸上计算bbox
                if args.resize:
                    mask_binary = (mask > 0).astype(np.uint8)
                    mask_orig_size = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    x_min, y_min, x_max, y_max = calculate_bounding_box(mask_orig_size)
                    # 保存缩放回原尺寸的mask
                    save_mask_as_grayscale(mask_orig_size, output_path)
                else:
                    # 未resize，直接使用原mask与bbox
                    x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                    save_mask_as_grayscale(mask, output_path)

                # 追加/更新该图像的日志（JSON数组末尾追加一次当前总状态）
                try:
                    append_state_log(args.output_dir, image_path)
                except Exception as e:
                    print(f"[WARN] Failed to append state log for {image_path}: {e}")

                _print_progress(num_processed + 1, num_skipped, num_errors,
                                f"ok line {line_num}: {Path(output_path).name} conf={confidence:.3f}")
                num_processed += 1
                
            except Exception as e:
                _print_progress(num_processed, num_skipped, num_errors + 1,
                                f"error line {line_num}: {Path(image_path).name}")
                num_errors += 1
                continue
    
    # 将输入数组复制到json_output并添加/覆盖 sam_masks_dir
    if args.json_output:
        try:
            # 若前面用JSONL解析得到records，则以records为准写出数组
            out_records = []
            for obj in records:
                if isinstance(obj, dict):
                    obj2 = dict(obj)
                    obj2["sam_masks_dir"] = args.output_dir
                    out_records.append(obj2)
            with open(args.json_output, 'w', encoding='utf-8') as jf:
                json.dump(out_records, jf, indent=2, ensure_ascii=False)
            print(f"Copied input to json_output with 'sam_masks_dir': {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save json_output: {e}")
            return 1
    
    # 打印总结（先补换行清空进度行）
    print()
    print(f"Processing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

I'll fix the indentation error in the processing loop to resolve the linter issue, ensuring the code compiles and runs. Then I'll re-check for linter errors.

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取由点提示组成的输入（支持JSON数组或JSONL逐行），使用SAM2进行分割，
将预测mask保存为灰度图像，并按要求生成/追加每幅图像对应的bbox日志。

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

输入JSON格式（数组）或JSONL（逐行）:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    - 预测mask按图像stem创建子目录保存为灰度图：output_dir/<stem>/<k>.png
      其中k为提示点序列长度-1；像素值：0=背景，255=前景
    - 在output_dir生成<stem>.jsonl（JSON数组）。每次运行在末尾追加一条：
      {"count": 子目录下mask文件总数, "bboxes": [[x1,y1,x2,y2], ...]}
    - 将输入JSON数组拷贝到 --json_output，并为每条加入/覆写
      "sam_masks_dir": output_dir

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/heatmap_train-0/pred_salient_points.jsonl \
      --output_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pred_masks-0 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda

带JSON输出:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/heatmap_train-0/pred_salient_points.jsonl \
      --output_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pred_masks-0 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing \
      --json_output /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pred_masks-0.jsonl

JSON输出格式:
    [
      {
        "mask_path": "/path/to/mask1.png",
        "bbox": [10, 20, 100, 150]
      },
      ...
    ]
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


def _print_progress(num_processed: int, num_skipped: int, num_errors: int, note: str) -> None:
    """在同一行输出进度信息"""
    msg = f"[Progress] processed={num_processed} skipped={num_skipped} errors={num_errors} | {note}"
    print(f"\r{msg}", end="", flush=True)


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
        sam_model = build_sam2(model_cfg, model_path) 
        # # 检查配置文件是否存在
        # config_path = Path(__file__).parent.parent / "third_party" / "sam2" / "sam2" / model_cfg
        # if not config_path.exists():
        #     # 尝试相对于当前目录
        #     config_path = Path(model_cfg)
        #     if not config_path.exists():
        #         raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        # sam_model = build_sam2(str(config_path), model_path)
        
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
    """生成输出mask文件路径（旧：单文件路径；新：保留但未使用）
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def get_mask_dir_for_image(image_path: str, output_dir: str) -> str:
    """返回该图像对应的mask子目录路径 output_dir/<stem>"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, image_name)


def get_mask_index_from_points(points: List[Tuple[float, float]]) -> int:
    """根据点序列长度返回mask索引：len(points)-1"""
    return max(0, int(len(points) - 1))


def get_mask_path_for_points(image_path: str, output_dir: str, points: List[Tuple[float, float]]) -> str:
    """输出mask路径：output_dir/<stem>/<k>.png，k=len(points)-1"""
    mask_dir = get_mask_dir_for_image(image_path, output_dir)
    os.makedirs(mask_dir, exist_ok=True)
    k = get_mask_index_from_points(points)
    return os.path.join(mask_dir, f"{k}.png")


def list_mask_pngs_sorted(mask_dir: str) -> List[str]:
    """列出目录下按数字文件名排序的png路径列表（0.png, 1.png, ...）"""
    if not os.path.isdir(mask_dir):
        return []
    files = []
    for name in os.listdir(mask_dir):
        if not name.lower().endswith('.png'):
            continue
        stem = os.path.splitext(name)[0]
        try:
            idx = int(stem)
        except ValueError:
            continue
        files.append((idx, os.path.join(mask_dir, name)))
    files.sort(key=lambda t: t[0])
    return [p for _, p in files]


def compute_bbox_from_mask_file(mask_path: str) -> Tuple[int, int, int, int]:
    """读取mask文件并计算其包围盒"""
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0, 0, 0, 0
    return calculate_bounding_box(img)


def append_state_log(output_dir: str, image_path: str) -> None:
    """构建并在 <output_dir>/<stem>.jsonl 末尾追加一次快照（JSON数组）。
    每条记录: {"count": N, "bboxes": [...]}，其中bboxes按0..N-1顺序。
    """
    stem = Path(image_path).stem
    log_path = os.path.join(output_dir, f"{stem}.jsonl")
    mask_dir = get_mask_dir_for_image(image_path, output_dir)
    mask_files = list_mask_pngs_sorted(mask_dir)
    bboxes: List[List[int]] = []
    for mp in mask_files:
        x1, y1, x2, y2 = compute_bbox_from_mask_file(mp)
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])

    new_entry = {"count": len(mask_files), "bboxes": bboxes}

    # 读取现有数组（若不存在则初始化为空数组）
    existing: List[Dict[str, Any]] = []
    if os.path.isfile(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    try:
                        existing = json.loads(content)
                        if not isinstance(existing, list):
                            # 若不是数组，尝试按JSONL逐行解析
                            existing = [json.loads(line) for line in content.splitlines() if line.strip()]
                    except json.JSONDecodeError:
                        # 按JSONL逐行
                        existing = [json.loads(line) for line in content.splitlines() if line.strip()]
        except Exception:
            existing = []

    existing.append(new_entry)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False)


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


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
                   help="Input JSON array (preferred) or JSONL file containing image paths and point coordinates")
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
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to write the copied input JSON array with added 'sam_masks_dir'")
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
    
    # 读取输入（优先按JSON数组解析；失败则按JSONL逐行解析）
    print(f"Processing input: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 解析输入记录列表 records: List[Dict]
    records: List[Dict[str, Any]] = []
    try:
        with open(args.input_jsonl, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    records = parsed
                else:
                    print("[ERROR] Input JSON root must be an array")
                    return 1
            else:
                # JSONL逐行
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                        num_errors += 1
                        continue
    except Exception as e:
        print(f"Error reading input: {e}")
        return 1

    for idx, obj in enumerate(records, 1):
        line_num = idx

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

        # 生成输出路径（子目录/<k>.png）
        output_path = get_mask_path_for_points(image_path, args.output_dir, points)

        # 检查是否跳过已存在的文件
        mask_existed = os.path.isfile(output_path)
        if args.skip_existing and mask_existed:
            _print_progress(num_processed, num_skipped + 1, num_errors,
                            f"skip line {line_num}: {Path(output_path).name} exists")
            # 若对应日志不存在，则仍然补齐一次日志快照
            stem_log_path = os.path.join(args.output_dir, f"{Path(image_path).stem}.jsonl")
            if not os.path.isfile(stem_log_path):
                try:
                    append_state_log(args.output_dir, image_path)
                except Exception:
                    pass
            num_skipped += 1
            continue

        try:
            # 加载图像
            image = PILImage.open(image_path).convert("RGB")
            orig_w, orig_h = image.size

            # 若指定resize，则按比例缩放点坐标并对图像进行resize
            if args.resize:
                resize_w, resize_h = int(args.resize[0]), int(args.resize[1])
                scale_x = float(resize_w) / float(orig_w)
                scale_y = float(resize_h) / float(orig_h)
                points_resized = [(px * scale_x, py * scale_y) for (px, py) in points]
                image_for_pred = image.resize((resize_w, resize_h), PILImage.BILINEAR)
            else:
                points_resized = points
                image_for_pred = image
                scale_x = 1.0
                scale_y = 1.0

            # 运行SAM2预测（在处理后的分辨率上）
            mask, confidence = sam_wrapper.predict(image_for_pred, points_resized, labels)

            # 如果做了resize，则将mask缩放回原图尺寸，并在原图尺寸上计算bbox
            if args.resize:
                mask_binary = (mask > 0).astype(np.uint8)
                mask_orig_size = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask_orig_size)
                # 保存缩放回原尺寸的mask
                save_mask_as_grayscale(mask_orig_size, output_path)
            else:
                # 未resize，直接使用原mask与bbox
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                save_mask_as_grayscale(mask, output_path)

            # 追加/更新该图像的日志（JSON数组末尾追加一次当前总状态）
            try:
                append_state_log(args.output_dir, image_path)
            except Exception as e:
                print(f"[WARN] Failed to append state log for {image_path}: {e}")

            _print_progress(num_processed + 1, num_skipped, num_errors,
                            f"ok line {line_num}: {Path(output_path).name} conf={confidence:.3f}")
            num_processed += 1

        except Exception as e:
            _print_progress(num_processed, num_skipped, num_errors + 1,
                            f"error line {line_num}: {Path(image_path).name}")
            num_errors += 1
            continue
    
    # 将输入数组复制到json_output并添加/覆盖 sam_masks_dir
    if args.json_output:
        try:
            # 若前面用JSONL解析得到records，则以records为准写出数组
            out_records = []
            for obj in records:
                if isinstance(obj, dict):
                    obj2 = dict(obj)
                    obj2["sam_masks_dir"] = args.output_dir
                    out_records.append(obj2)
            with open(args.json_output, 'w', encoding='utf-8') as jf:
                json.dump(out_records, jf, indent=2, ensure_ascii=False)
            print(f"Copied input to json_output with 'sam_masks_dir': {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save json_output: {e}")
            return 1
    
    # 打印总结（先补换行清空进度行）
    print()
    print(f"Processing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

I updated `seg-rl/sam2_segment_from_points.py` to: read a JSON array (fallback to JSONL), run SAM2 using provided points/labels per entry, save masks into `output_dir/<stem>/<k>.png` where k=len(points)-1, maintain a per-image `<stem>.jsonl` JSON array log with {"count", "bboxes"}, and copy the input array to `--json_output` with "sam_masks_dir" set to `--output_dir`. I’ll record and complete the implementation tasks.

- I read and updated `seg-rl/sam2_segment_from_points.py` to fulfill your new requirements while keeping other logic intact.
- Implemented:
  - Input parsing supports a JSON array (preferred) with fallback to JSONL lines.
  - For each record, SAM2 runs with `points` and `labels`. Output mask saved to `output_dir/<stem>/<k>.png` where `k = len(points) - 1` and `<stem>` = image filename without extension. Directories are created on demand.
  - A per-image JSON array log at `output_dir/<stem>.jsonl` is appended with a snapshot: `{"count": <mask_count>, "bboxes": [[x1,y1,x2,y2], ...]}` where bboxes are computed from all `0.png..N.png` masks currently present.
  - The input array is copied to `--json_output` with `"sam_masks_dir": <output_dir>` added/overwritten for each entry.
  - `--skip_existing` respected: if `<k>.png` exists, it’s skipped; if the per-image log doesn’t exist yet, a snapshot is still created.
  - Resizing behavior and other params preserved.

Run example:
```bash
python /Users/donchiu/Works/Seg-R1/seg-rl/sam2_segment_from_points.py \
  --input_jsonl /root/datasets/segrl_pretrain_braintumour.jsonl \
  --json_output /root/datasets/segrl_pretrain_braintumour.jsonl \
  --output_dir /root/datasets/pretrain_gt_masks \
  --sam_checkpoint /root/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --skip_existing
```

Notes:
- Editor shows environment import warnings (cv2, numpy, torch, PIL, sam2); they are benign if your env has these installed.
- If your input file is a JSON array, it will be read as a whole; JSONL lines are supported as fallback.

```shellscript
python /Users/donchiu/Works/Seg-R1/seg-rl/sam2_segment_from_points.py \
  --input_jsonl /root/datasets/segrl_pretrain_braintumour.jsonl \
  --json_output /root/datasets/segrl_pretrain_braintumour.jsonl \
  --output_dir /root/datasets/pretrain_gt_masks \
  --sam_checkpoint /root/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --skip_existing
```

---

**User**

有两个问题：1. 没必要在append_state_log中通过重新加载mask图像再去计算包围盒，因为在517行调用calculate_bounding_box时已经计算出了，无需重复计算； 2. append_state_log追加分割图像日志并不是添加一整条{"count": len(mask_files), "bboxes": bboxes}，而是仅需在bboxes所指向的数组中增加一个新的包围盒坐标(len=4的数组）,并把count值递增即可。整个<output_dir>/<stem>.jsonl 里应仅保持{"count": len(mask_files), "bboxes": bboxes}这样一个字典而并非不定长的数组。

---

**Cursor**

I’ll update the per-image logging to append only the new bbox and increment count, reuse the already computed bbox instead of re-reading mask images, and remove the unnecessary helper functions. Then I’ll fix the skip-existing branch so it doesn’t attempt to write a log without a bbox.

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取由点提示组成的输入（支持JSON数组或JSONL逐行），使用SAM2进行分割，
将预测mask保存为灰度图像，并按要求生成/追加每幅图像对应的bbox日志。

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

输入JSON格式（数组）或JSONL（逐行）:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    - 预测mask按图像stem创建子目录保存为灰度图：output_dir/<stem>/<k>.png
      其中k为提示点序列长度-1；像素值：0=背景，255=前景
    - 在output_dir生成<stem>.jsonl（JSON对象，非数组）。该文件内容保持为：
      {"count": N, "bboxes": [[x1,y1,x2,y2], ...]}。每次新增一个mask，仅向
      bboxes追加一个新的bbox并将count递增，无需重写历史bbox。
    - 将输入JSON数组拷贝到 --json_output，并为每条加入/覆写
      "sam_masks_dir": output_dir

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/heatmap_train-0/pred_salient_points.jsonl \
      --output_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pred_masks-0 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda

带JSON输出:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/heatmap_train-0/pred_salient_points.jsonl \
      --output_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pred_masks-0 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing \
      --json_output /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pred_masks-0.jsonl

JSON输出格式:
    [
      {
        "mask_path": "/path/to/mask1.png",
        "bbox": [10, 20, 100, 150]
      },
      ...
    ]
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


def _print_progress(num_processed: int, num_skipped: int, num_errors: int, note: str) -> None:
    """在同一行输出进度信息"""
    msg = f"[Progress] processed={num_processed} skipped={num_skipped} errors={num_errors} | {note}"
    print(f"\r{msg}", end="", flush=True)


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
        sam_model = build_sam2(model_cfg, model_path) 
        # # 检查配置文件是否存在
        # config_path = Path(__file__).parent.parent / "third_party" / "sam2" / "sam2" / model_cfg
        # if not config_path.exists():
        #     # 尝试相对于当前目录
        #     config_path = Path(model_cfg)
        #     if not config_path.exists():
        #         raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        # sam_model = build_sam2(str(config_path), model_path)
        
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
    """生成输出mask文件路径（旧：单文件路径；新：保留但未使用）
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def get_mask_dir_for_image(image_path: str, output_dir: str) -> str:
    """返回该图像对应的mask子目录路径 output_dir/<stem>"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, image_name)


def get_mask_index_from_points(points: List[Tuple[float, float]]) -> int:
    """根据点序列长度返回mask索引：len(points)-1"""
    return max(0, int(len(points) - 1))


def get_mask_path_for_points(image_path: str, output_dir: str, points: List[Tuple[float, float]]) -> str:
    """输出mask路径：output_dir/<stem>/<k>.png，k=len(points)-1"""
    mask_dir = get_mask_dir_for_image(image_path, output_dir)
    os.makedirs(mask_dir, exist_ok=True)
    k = get_mask_index_from_points(points)
    return os.path.join(mask_dir, f"{k}.png")


def append_state_log(output_dir: str, image_path: str, bbox: Tuple[int, int, int, int]) -> None:
    """在 <output_dir>/<stem>.jsonl 中维护单个JSON对象：
    {"count": N, "bboxes": [[x1,y1,x2,y2], ...]}
    仅追加当前bbox并将count递增。不会重新读取或计算历史bbox。
    """
    stem = Path(image_path).stem
    log_path = os.path.join(output_dir, f"{stem}.jsonl")

    record: Dict[str, Any] = {"count": 0, "bboxes": []}
    if os.path.isfile(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "count" in parsed and "bboxes" in parsed:
                        record = parsed
        except Exception:
            pass

    # 追加当前bbox并递增count
    record.setdefault("bboxes", [])
    record["bboxes"].append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
    record["count"] = int(record.get("count", 0)) + 1

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


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
                   help="Input JSON array (preferred) or JSONL file containing image paths and point coordinates")
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
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to write the copied input JSON array with added 'sam_masks_dir'")
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
    
    # 读取输入（优先按JSON数组解析；失败则按JSONL逐行解析）
    print(f"Processing input: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 解析输入记录列表 records: List[Dict]
    records: List[Dict[str, Any]] = []
    try:
        with open(args.input_jsonl, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    records = parsed
                else:
                    print("[ERROR] Input JSON root must be an array")
                    return 1
            else:
                # JSONL逐行
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                        num_errors += 1
                        continue
    except Exception as e:
        print(f"Error reading input: {e}")
        return 1

    for idx, obj in enumerate(records, 1):
        line_num = idx

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

        # 生成输出路径（子目录/<k>.png）
        output_path = get_mask_path_for_points(image_path, args.output_dir, points)

        # 检查是否跳过已存在的文件
        mask_existed = os.path.isfile(output_path)
        if args.skip_existing and mask_existed:
            _print_progress(num_processed, num_skipped + 1, num_errors,
                            f"skip line {line_num}: {Path(output_path).name} exists")
            # 若对应日志不存在，则仍然补齐一次日志快照
            stem_log_path = os.path.join(args.output_dir, f"{Path(image_path).stem}.jsonl")
            if not os.path.isfile(stem_log_path):
                try:
                    append_state_log(args.output_dir, image_path)
                except Exception:
                    pass
            num_skipped += 1
            continue

        try:
            # 加载图像
            image = PILImage.open(image_path).convert("RGB")
            orig_w, orig_h = image.size

            # 若指定resize，则按比例缩放点坐标并对图像进行resize
            if args.resize:
                resize_w, resize_h = int(args.resize[0]), int(args.resize[1])
                scale_x = float(resize_w) / float(orig_w)
                scale_y = float(resize_h) / float(orig_h)
                points_resized = [(px * scale_x, py * scale_y) for (px, py) in points]
                image_for_pred = image.resize((resize_w, resize_h), PILImage.BILINEAR)
            else:
                points_resized = points
                image_for_pred = image
                scale_x = 1.0
                scale_y = 1.0

            # 运行SAM2预测（在处理后的分辨率上）
            mask, confidence = sam_wrapper.predict(image_for_pred, points_resized, labels)

            # 如果做了resize，则将mask缩放回原图尺寸，并在原图尺寸上计算bbox
            if args.resize:
                mask_binary = (mask > 0).astype(np.uint8)
                mask_orig_size = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask_orig_size)
                # 保存缩放回原尺寸的mask
                save_mask_as_grayscale(mask_orig_size, output_path)
            else:
                # 未resize，直接使用原mask与bbox
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                save_mask_as_grayscale(mask, output_path)

            # 追加/更新该图像的日志（JSON数组末尾追加一次当前总状态）
            try:
                append_state_log(args.output_dir, image_path)
            except Exception as e:
                print(f"[WARN] Failed to append state log for {image_path}: {e}")

            _print_progress(num_processed + 1, num_skipped, num_errors,
                            f"ok line {line_num}: {Path(output_path).name} conf={confidence:.3f}")
            num_processed += 1

        except Exception as e:
            _print_progress(num_processed, num_skipped, num_errors + 1,
                            f"error line {line_num}: {Path(image_path).name}")
            num_errors += 1
            continue
    
    # 将输入数组复制到json_output并添加/覆盖 sam_masks_dir
    if args.json_output:
        try:
            # 若前面用JSONL解析得到records，则以records为准写出数组
            out_records = []
            for obj in records:
                if isinstance(obj, dict):
                    obj2 = dict(obj)
                    obj2["sam_masks_dir"] = args.output_dir
                    out_records.append(obj2)
            with open(args.json_output, 'w', encoding='utf-8') as jf:
                json.dump(out_records, jf, indent=2, ensure_ascii=False)
            print(f"Copied input to json_output with 'sam_masks_dir': {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save json_output: {e}")
            return 1
    
    # 打印总结（先补换行清空进度行）
    print()
    print(f"Processing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

I'll update the main loop to pass the already computed bounding box to the logging function, and stop writing logs in the skip-existing branch since no bbox is available. Then I'll re-run lint checks to ensure no new issues.

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取由点提示组成的输入（支持JSON数组或JSONL逐行），使用SAM2进行分割，
将预测mask保存为灰度图像，并按要求生成/追加每幅图像对应的bbox日志。

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

输入JSON格式（数组）或JSONL（逐行）:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    - 预测mask按图像stem创建子目录保存为灰度图：output_dir/<stem>/<k>.png
      其中k为提示点序列长度-1；像素值：0=背景，255=前景
    - 在output_dir生成<stem>.jsonl（JSON对象，非数组）。该文件内容保持为：
      {"count": N, "bboxes": [[x1,y1,x2,y2], ...]}。每次新增一个mask，仅向
      bboxes追加一个新的bbox并将count递增，无需重写历史bbox。
    - 将输入JSON数组拷贝到 --json_output，并为每条加入/覆写
      "sam_masks_dir": output_dir

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/heatmap_train-0/pred_salient_points.jsonl \
      --output_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pred_masks-0 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda

带JSON输出:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/heatmap_train-0/pred_salient_points.jsonl \
      --output_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pred_masks-0 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing \
      --json_output /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pred_masks-0.jsonl

JSON输出格式:
    [
      {
        "mask_path": "/path/to/mask1.png",
        "bbox": [10, 20, 100, 150]
      },
      ...
    ]
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


def _print_progress(num_processed: int, num_skipped: int, num_errors: int, note: str) -> None:
    """在同一行输出进度信息"""
    msg = f"[Progress] processed={num_processed} skipped={num_skipped} errors={num_errors} | {note}"
    print(f"\r{msg}", end="", flush=True)


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
        sam_model = build_sam2(model_cfg, model_path) 
        # # 检查配置文件是否存在
        # config_path = Path(__file__).parent.parent / "third_party" / "sam2" / "sam2" / model_cfg
        # if not config_path.exists():
        #     # 尝试相对于当前目录
        #     config_path = Path(model_cfg)
        #     if not config_path.exists():
        #         raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        # sam_model = build_sam2(str(config_path), model_path)
        
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
    """生成输出mask文件路径（旧：单文件路径；新：保留但未使用）
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def get_mask_dir_for_image(image_path: str, output_dir: str) -> str:
    """返回该图像对应的mask子目录路径 output_dir/<stem>"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, image_name)


def get_mask_index_from_points(points: List[Tuple[float, float]]) -> int:
    """根据点序列长度返回mask索引：len(points)-1"""
    return max(0, int(len(points) - 1))


def get_mask_path_for_points(image_path: str, output_dir: str, points: List[Tuple[float, float]]) -> str:
    """输出mask路径：output_dir/<stem>/<k>.png，k=len(points)-1"""
    mask_dir = get_mask_dir_for_image(image_path, output_dir)
    os.makedirs(mask_dir, exist_ok=True)
    k = get_mask_index_from_points(points)
    return os.path.join(mask_dir, f"{k}.png")


def append_state_log(output_dir: str, image_path: str, bbox: Tuple[int, int, int, int]) -> None:
    """在 <output_dir>/<stem>.jsonl 中维护单个JSON对象：
    {"count": N, "bboxes": [[x1,y1,x2,y2], ...]}
    仅追加当前bbox并将count递增。不会重新读取或计算历史bbox。
    """
    stem = Path(image_path).stem
    log_path = os.path.join(output_dir, f"{stem}.jsonl")

    record: Dict[str, Any] = {"count": 0, "bboxes": []}
    if os.path.isfile(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "count" in parsed and "bboxes" in parsed:
                        record = parsed
        except Exception:
            pass

    # 追加当前bbox并递增count
    record.setdefault("bboxes", [])
    record["bboxes"].append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
    record["count"] = int(record.get("count", 0)) + 1

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


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
                   help="Input JSON array (preferred) or JSONL file containing image paths and point coordinates")
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
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to write the copied input JSON array with added 'sam_masks_dir'")
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
    
    # 读取输入（优先按JSON数组解析；失败则按JSONL逐行解析）
    print(f"Processing input: {args.input_jsonl}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 解析输入记录列表 records: List[Dict]
    records: List[Dict[str, Any]] = []
    try:
        with open(args.input_jsonl, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    records = parsed
                else:
                    print("[ERROR] Input JSON root must be an array")
                    return 1
            else:
                # JSONL逐行
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                        num_errors += 1
                        continue
    except Exception as e:
        print(f"Error reading input: {e}")
        return 1

    for idx, obj in enumerate(records, 1):
        line_num = idx

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

        # 生成输出路径（子目录/<k>.png）
        output_path = get_mask_path_for_points(image_path, args.output_dir, points)

        # 检查是否跳过已存在的文件
        mask_existed = os.path.isfile(output_path)
        if args.skip_existing and mask_existed:
            _print_progress(num_processed, num_skipped + 1, num_errors,
                            f"skip line {line_num}: {Path(output_path).name} exists")
            # 此分支无现成bbox，且不应为不存在的日志补写空记录
            num_skipped += 1
            continue

        try:
            # 加载图像
            image = PILImage.open(image_path).convert("RGB")
            orig_w, orig_h = image.size

            # 若指定resize，则按比例缩放点坐标并对图像进行resize
            if args.resize:
                resize_w, resize_h = int(args.resize[0]), int(args.resize[1])
                scale_x = float(resize_w) / float(orig_w)
                scale_y = float(resize_h) / float(orig_h)
                points_resized = [(px * scale_x, py * scale_y) for (px, py) in points]
                image_for_pred = image.resize((resize_w, resize_h), PILImage.BILINEAR)
            else:
                points_resized = points
                image_for_pred = image
                scale_x = 1.0
                scale_y = 1.0

            # 运行SAM2预测（在处理后的分辨率上）
            mask, confidence = sam_wrapper.predict(image_for_pred, points_resized, labels)

            # 如果做了resize，则将mask缩放回原图尺寸，并在原图尺寸上计算bbox
            if args.resize:
                mask_binary = (mask > 0).astype(np.uint8)
                mask_orig_size = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask_orig_size)
                # 保存缩放回原尺寸的mask
                save_mask_as_grayscale(mask_orig_size, output_path)
            else:
                # 未resize，直接使用原mask与bbox
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                save_mask_as_grayscale(mask, output_path)

            # 追加/更新该图像的日志（单一JSON对象，追加bbox并递增count）
            try:
                append_state_log(args.output_dir, image_path, (x_min, y_min, x_max, y_max))
            except Exception as e:
                print(f"[WARN] Failed to append state log for {image_path}: {e}")

            _print_progress(num_processed + 1, num_skipped, num_errors,
                            f"ok line {line_num}: {Path(output_path).name} conf={confidence:.3f}")
            num_processed += 1

        except Exception as e:
            _print_progress(num_processed, num_skipped, num_errors + 1,
                            f"error line {line_num}: {Path(image_path).name}")
            num_errors += 1
            continue
    
    # 将输入数组复制到json_output并添加/覆盖 sam_masks_dir
    if args.json_output:
        try:
            # 若前面用JSONL解析得到records，则以records为准写出数组
            out_records = []
            for obj in records:
                if isinstance(obj, dict):
                    obj2 = dict(obj)
                    obj2["sam_masks_dir"] = args.output_dir
                    out_records.append(obj2)
            with open(args.json_output, 'w', encoding='utf-8') as jf:
                json.dump(out_records, jf, indent=2, ensure_ascii=False)
            print(f"Copied input to json_output with 'sam_masks_dir': {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save json_output: {e}")
            return 1
    
    # 打印总结（先补换行清空进度行）
    print()
    print(f"Processing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

- Adjusted per-image log writing:
  - Replaced the previous “snapshot array” approach with a single persistent JSON object at `output_dir/<stem>.jsonl` holding only {"count", "bboxes"}.
  - The logger now appends just the new bbox and increments count; it doesn’t re-read masks or recompute historical bboxes.
  - The main loop passes the already computed `(x_min, y_min, x_max, y_max)` to the logger.
  - In the `--skip_existing` branch we no longer write any log entry (no bbox available).

- Removed unnecessary helpers for rescanning directories and recomputing bboxes from files.

- Existing env import warnings remain benign if your dependencies are installed.

This aligns with your two requests exactly: no redundant bbox computation, and maintain a single JSON object updated in place.

---

**User**

接下来，继续修改gen_point_jsonl_from_masks.py: 增加一个参数appendto_jsonl，其与output_jsonl互斥，即二者之中有且仅有其中之一。在指定output_jsonl参数时表示的是计算首个最佳提示点，与之前逻辑不变；在指定了appendto_jsonl参数的情况下表示计算后续（第2个及以后的）最佳提示点，其字符串参数值表示一个前述已经包含sam_masks_dir的JSON数组描述文件的路径，此时其它命令行参数无效，因为必要信息已可完全由该JSON文件获得。假设该JSON描述文件内容为：
[ {"image":IMAGE_PATH, "gt_mask":GT_MASK_PATH, "sam_masks_dir":SAM_MASKS_DIR, "points":[[X0,Y0]], "labels":[L0]}, ... ... ]
，在此情况下，脚本要做的是：对于数组中每一个输入样本，把gt_mask和sam_masks_dir目录下对应的最后一个预测mask分别作为参数列表中的A和B传入你之前已经实现的largest_diff_component_representative函数，把计算得到的点坐标和正负号作为下一个点提示的point和label值（+对应1，-对应0），添加在points和labels数组之后，来更新原JSON数组描述文件。
举例：还是沿用前述例子。假设已有/root/datasets/segrl_pretrain_braintumour.jsonl内容如下：
[
{"image”:“/root/datasets/braintumour/images/BRAIN_00.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_00.png”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, ”points”:[[123,456]], "labels”:[1]},
{"image”:“/root/datasets/braintumour/images/BRAIN_01.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_01.png”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, "points”:[[312,576]], "labels”:[1]}
]
, /root/datasets/pretrain_gt_masks目录存在并包含以下内容：
pretrain_gt_masks/
 ├── BRAIN_00/
 ├──── 0.png
 ├── BRAIN_01/
 ├──── 0.png
 ├── BRAIN_00.jsonl
 ├── BRAIN_01.jsonl
其中每个png文件均是在仅有第一个提示点的条件下作sam分割得到的预测mask图像。BRAIN_00.jsonl具备内容(具体坐标数值仅为示意)：
{"count":1, "bboxes”:[[13,75,256,489]]}
，BRAIN_01.jsonl具备内容(具体坐标数值仅为示意)：
{"count":1, "bboxes”:[[213,375,412,611]]}
，那么执行以下命令：
python seg-rl/annotator/gen_point_jsonl_from_masks.py \ --appendto_jsonl /root/datasets/segrl_pretrain_braintumour.jsonl
之后，对于BRAIN_00这个样本，程序应使用largest_diff_component_representative函数，根据真值mask图像/root/datasets/braintumour/masks/BRAIN_00.png与最后一个预测mask图像/root/datasets/pretrain_gt_masks/BRAIN_00/0.png的差值图像，求下一个最佳提示点。假设得到的下一提示点坐标为[256,312]，label=0即为背景点，则/root/datasets/segrl_pretrain_braintumour.jsonl 中对应于BRAIN_00这一项应相应被更新成：
{"image”:“/root/datasets/braintumour/images/BRAIN_00.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_00.jpg”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, "points”:[[123,456],[256,312]], "labels”:[1,0]}
。对BRAIN_01样本同理。最终整个JSON描述文件被更新成（坐标数值仅为示意）：
[
{"image”:“/root/datasets/braintumour/images/BRAIN_00.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_00.jpg”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, "points”:[[123,456],[256,312]], "labels”:[1,0]},
{"image”:“/root/datasets/braintumour/images/BRAIN_01.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_01.jpg”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, "points”:[[312,576],[168,35]], "labels”:[1,0]}
]
。我已经描述得足够清楚了，请开始修改

---

**Cursor**

I'll add the new mutually-exclusive argument, implement append mode reading the JSON array, find last predicted mask per sample, compute next point via largest_diff_component_representative, update points/labels, and write back the same JSON file. Then I’ll run a quick lint check on the modified file.

I'll add the mutually exclusive argument `--appendto_jsonl`, make `--images_dir`, `--masks_dir`, and `--output_jsonl` only required in generation mode, and implement append mode: read the JSON array, for each item compute the next point using `largest_diff_component_representative` on `gt_mask` and the latest SAM mask in `sam_masks_dir/<stem>/<k>.png` (k defaults to len(points)-1 or falls back to max index), append the new point and label, and overwrite the same JSON file. I'll then run a quick lint check.

```python
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



```

- Added `--appendto_jsonl` mutually exclusive with `--output_jsonl`. In append mode, other CLI args are ignored.
- Append mode behavior:
  - Reads the JSON array at `--appendto_jsonl`.
  - For each record, finds the latest SAM mask under `sam_masks_dir/<stem>/<k>.png` where `k = len(points)-1` (fallback to max numeric).
  - Loads `gt_mask` and the latest predicted mask, resizes predicted mask to GT size if needed, computes the next representative point with `largest_diff_component_representative`, maps sign to label (1 for +, 0 for -), appends to `points` and `labels`.
  - Writes the updated array back to the same file.

- Initial generation mode unchanged, except moved under the new CLI structure and still writes JSONL lines with first point.

Example:
```bash
# Append next prompts in place
python /Users/donchiu/Works/Seg-R1/seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --appendto_jsonl /root/datasets/segrl_pretrain_braintumour.jsonl
```

Notes:
- Env import warnings are expected if the Python env lacks those libs; code changes compile otherwise.

```shellscript
# Append next prompts in place
python /Users/donchiu/Works/Seg-R1/seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --appendto_jsonl /root/datasets/segrl_pretrain_braintumour.jsonl
```

---

**User**

在seg-rl/evaluation下编写一个脚本eval_sam_masks.py用于计算评估mask分割效果。包含几个命令行参数：
—input_json: 即指向前述JSON描述文件的路径。该文件中包含的JSON形式沿用先前各脚本所使用的定义，即形如：
[
{"image”:“/root/datasets/braintumour/images/BRAIN_00.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_00.jpg”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, "points”:[[123,456],[256,312]], "labels”:[1,0]},
{"image”:“/root/datasets/braintumour/images/BRAIN_01.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_01.jpg”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, "points”:[[312,576],[168,35]], "labels”:[1,0]}
]
的JSON数组；
—num_prompts: 可选参数，指定需作评估的masks所对应的提示点数量，为大于0的正整数。在指定了该参数时，程序对每个样本应比较gt_mask对应的真值掩模图像 与 sam_masks_dir/{$样本图像文件名stem}/{$num_prompts - 1}.png 所指向的mask图像之间的差异，计算常见的图像分割性能评估指标，如DICE/IOU/S-measure等。若不指定该参数，则找sam_masks_dir/{$样本图像文件名stem}/下最大标号的.png图像作为与gt_mask所比较的预测mask图像。
程序运行后，对每个样本计算其gt_mask与指定提示点数量对应的预测mask之间的差异指标，写入/更新到{$sam_masks_dir}/{$样本图像文件名stem}-metrics.jsonl中，其内容形如：
{
“1”: {“DICE”:$DICE_1, “IOU”:$IOU_1},
“2”: {“DICE”:$DICE_2, “IOU”:$IOU_2}
}
即其根对象是一个JSON字典，其中每个键-值对的键为所采用的提示点数量num_prompts，值为对 sam_masks_dir/{$样本图像文件名stem}/{$num_prompts - 1}.png 所指向的mask图像 求与gt_mask之间的差异所得出的各项指标。
举例：
假设已有/root/datasets/segrl_pretrain_braintumour.jsonl内容如下：
[
{"image”:“/root/datasets/braintumour/images/BRAIN_00.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_00.jpg”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, "points”:[[123,456],[256,312]], "labels”:[1,0]},
{"image”:“/root/datasets/braintumour/images/BRAIN_01.jpg”, "gt_mask":“/root/datasets/braintumour/masks/BRAIN_01.jpg”, “sam_masks_dir”:”/root/datasets/pretrain_gt_masks”, "points”:[[312,576],[168,35]], "labels”:[1,0]}
]
, /root/datasets/pretrain_gt_masks目录存在并仅包含以下内容：
pretrain_gt_masks/
 ├── BRAIN_00/
 ├──── 0.png
 ├──── 1.png
 ├── BRAIN_01/
 ├──── 0.png
 ├──── 1.png
 ├── BRAIN_00.jsonl
 ├── BRAIN_01.jsonl
，则执行以下命令：
python seg-rl/evaluation/eval_sam_masks.py \ --input_json /root/datasets/segrl_pretrain_braintumour.jsonl —num_prompts 2 之后，应在/root/datasets/pretrain_gt_masks目录下生成BRAIN_00-metrics.jsonl和BRAIN_01-metrics.jsonl，其内容均形如（数值仅为示意）：
{
“2”: {“DICE”:0.93, “IOU”:0.86}
}
。若再执行python seg-rl/evaluation/eval_sam_masks.py \ --input_json /root/datasets/segrl_pretrain_braintumour.jsonl —num_prompts 1 之后，BRAIN_00-metrics.jsonl和BRAIN_01-metrics.jsonl会被相应更新成（数值仅为示意）：
{
“1”: {“DICE”:0.18, “IOU”:0.09}
“2”: {“DICE”:0.93, “IOU”:0.86}
}
请清晰理解需求并开始编写。


---

**Cursor**

I'll create a new evaluation script at `seg-rl/evaluation/eval_sam_masks.py` to compute DICE and IOU for predicted masks against ground truth, supporting `--input_json` and optional `--num_prompts`. It will read the JSON array, locate the appropriate predicted mask per sample, compute metrics, and update `<sam_masks_dir>/<stem>-metrics.jsonl` as a single JSON object mapping num_prompts to metric results. Then I'll check for linter issues.

```python
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


def _compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute DICE and IOU. Arrays are boolean foreground maps of equal shape."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum(dtype=np.float64)
    gt_sum = gt.sum(dtype=np.float64)
    pred_sum = pred.sum(dtype=np.float64)
    union = np.logical_or(gt, pred).sum(dtype=np.float64)

    dice = (2.0 * inter) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else (1.0 if union == 0 else 0.0)
    iou = inter / union if union > 0 else 1.0
    return {"DICE": float(dice), "IOU": float(iou)}


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
            print(f"ok idx {i}: {Path(image_path).stem} prompts={eff_num_prompts} | DICE={metrics['DICE']:.4f} IOU={metrics['IOU']:.4f}")
        except Exception as e:
            print(f"[ERROR] idx {i}: failed to evaluate - {e}")
            errors += 1

    print(f"Done. processed={processed} skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


```

- Created `seg-rl/evaluation/eval_sam_masks.py` to evaluate predicted masks vs ground truth.

What it does:
- Reads `--input_json` JSON array with fields `image`, `gt_mask`, `sam_masks_dir`, `points`, `labels`.
- Chooses predicted mask:
  - If `--num_prompts N` is given, uses `sam_masks_dir/<stem>/<N-1>.png`.
  - Otherwise, finds the highest index `<k>.png` under `sam_masks_dir/<stem>/` and sets effective prompts to `k+1`.
- Loads `gt_mask` and predicted mask, resizes prediction to GT size if needed (nearest), binarizes (>0).
- Computes DICE and IOU.
- Updates `<sam_masks_dir>/<stem>-metrics.jsonl` as a single JSON dict mapping prompt count to metrics:
  - Example:
    {
      "1": {"DICE": 0.18, "IOU": 0.09},
      "2": {"DICE": 0.93, "IOU": 0.86}
    }

Usage examples:
```bash
# Evaluate with 2 prompts (uses .../1.png)
python /Users/donchiu/Works/Seg-R1/seg-rl/evaluation/eval_sam_masks.py \
  --input_json /root/datasets/segrl_pretrain_braintumour.jsonl \
  --num_prompts 2

# Evaluate with max available prompt index
python /Users/donchiu/Works/Seg-R1/seg-rl/evaluation/eval_sam_masks.py \
  --input_json /root/datasets/segrl_pretrain_braintumour.jsonl
```

Notes:
- Script computes DICE and IOU; S-measure can be added later if needed.
- No linter issues detected.

```shellscript
# Evaluate with 2 prompts (uses .../1.png)
python /Users/donchiu/Works/Seg-R1/seg-rl/evaluation/eval_sam_masks.py \
  --input_json /root/datasets/segrl_pretrain_braintumour.jsonl \
  --num_prompts 2

# Evaluate with max available prompt index
python /Users/donchiu/Works/Seg-R1/seg-rl/evaluation/eval_sam_masks.py \
  --input_json /root/datasets/segrl_pretrain_braintumour.jsonl
```

---

**User**

请在其中添加更丰富的metrics计算，例如recall, precision, s-measure, F1-score等，参考seg-rl/mask_comparison.py

---

**Cursor**

I'll first open `seg-rl/mask_comparison.py` to review how metrics like S-measure are calculated so I can implement consistent computations in `eval_sam_masks.py`.

I'll add precision, recall, F1, and S-measure computations to eval_sam_masks.py based on the existing implementations in mask_comparison.py, then update the metrics output and run a quick lint check.

```python
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
            print(
                f"ok idx {i}: {Path(image_path).stem} prompts={eff_num_prompts} | "
                f"DICE={metrics['DICE']:.4f} IOU={metrics['IOU']:.4f} "
                f"P={metrics['PRECISION']:.4f} R={metrics['RECALL']:.4f} F1={metrics['F1']:.4f} "
                f"S={metrics['S_MEASURE']:.4f}"
            )
        except Exception as e:
            print(f"[ERROR] idx {i}: failed to evaluate - {e}")
            errors += 1

    print(f"Done. processed={processed} skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())



```

已在 `seg-rl/evaluation/eval_sam_masks.py` 中补充更丰富的指标，参考了 `seg-rl/mask_comparison.py` 的实现。

- 新增指标
  - DICE
  - IOU
  - PRECISION
  - RECALL
  - F1
  - S_MEASURE（结构相似度，含 object-aware 与 region-aware 组合）

- 输出
  - 仍写入 `sam_masks_dir/<stem>-metrics.jsonl`，键为提示点数量 `num_prompts`，值为上述各项指标：
    {
      "2": {"DICE":0.93,"IOU":0.86,"PRECISION":0.90,"RECALL":0.96,"F1":0.93,"S_MEASURE":0.92}
    }

- 使用示例
```bash
python /Users/donchiu/Works/Seg-R1/seg-rl/evaluation/eval_sam_masks.py \
  --input_json /root/datasets/segrl_pretrain_braintumour.jsonl --num_prompts 2
```

- 说明
  - 若未指定 `--num_prompts`，会选用 `sam_masks_dir/<stem>/` 下最大编号的 `k.png`，并将 `num_prompts=k+1` 作为键记录。

```shellscript
python /Users/donchiu/Works/Seg-R1/seg-rl/evaluation/eval_sam_masks.py \
  --input_json /root/datasets/segrl_pretrain_braintumour.jsonl --num_prompts 2
```

---

**User**

还需补充一点：需计算整个数据集各项metric的平均值，打印到控制台输出

---

**Cursor**

I'll add accumulation of per-sample metrics in `eval_sam_masks.py` and print dataset-wide averages at the end. Then I'll run a quick lint check.

```python
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
            print(
                f"ok idx {i}: {Path(image_path).stem} prompts={eff_num_prompts} | "
                f"DICE={metrics['DICE']:.4f} IOU={metrics['IOU']:.4f} "
                f"P={metrics['PRECISION']:.4f} R={metrics['RECALL']:.4f} F1={metrics['F1']:.4f} "
                f"S={metrics['S_MEASURE']:.4f}"
            )
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



```

已在 `seg-rl/evaluation/eval_sam_masks.py` 增加数据集级别的平均指标输出。现在脚本会在逐样本输出后，汇总并打印全数据集的平均 DICE、IOU、PRECISION、RECALL、F1、S_MEASURE。

示例输出追加段落:
- Dataset averages:
  - DICE: 0.8123
  - IOU: 0.6932
  - PRECISION: 0.8456
  - RECALL: 0.7841
  - F1: 0.8139
  - S_MEASURE: 0.8012

其余逻辑保持不变。

---

**User**

基于前述规范化的各JSON文件及目录结构定义，请在seg-rl/visualization下编写脚本viz_sam_segmentation.py用以批量可视化分割效果，命令行参数：
—input_json: 即指向前述JSON描述文件的路径。该文件中包含的JSON形式沿用先前各脚本所使用的定义；
—output_dir: 输出可视化图像文件的目录，若不存在则创建。
程序通过input_json指向的JSON描述可以找到每个样本相应的原始图、真值mask图，以及存放预测mask和分割效果评估的{$sam_masks_dir}/{$stem}-metrics.jsonl文件，这些信息足以为每个样本生成可视化的预测mask结果比较图，图中应包括如下几列：原图像叠加半透明真值mask图，原图像叠加半透明预测mask及提示点序列图（前景点以^形的尖端指示，背景点以X的交叉点指示，并在旁边用小数字标注序号），原图像叠加半透明真值mask和半透明预测mask图（两个mask用两种不同半透明颜色表示，原图像转换为灰度图以便看清）。并且图上应附有文字，显示该分割预测mask对应的所有metrics（通过{$sam_masks_dir}/{$stem}-metrics.jsonl文件可以获得）。 在output_dir所指向目录中为每个不同的提示点数量创建一个子目录，每个子目录下对每一个样本生成相应的结果比较图。
例如，执行：
python seg-rl/visualization/viz_sam_segmentation \ --input_json /root/datasets/segrl_pretrain_braintumour.jsonl —output_dir /root/output/braintumour 之后，应在/root/output/braintumour目录下生成如下内容：
/root/output/braintumour/
 ├── 1/
 ├──── BRAIN_00.jpg
 ├──── BRAIN_01.jpg
 ├── 2/
 ├──── BRAIN_00.jpg
 ├──── BRAIN_01.jpg


---

**Cursor**

```python
#!/usr/bin/env python3

"""
Batch visualization of SAM segmentation results per number of prompts.

Usage:
  python seg-rl/visualization/viz_sam_segmentation.py \
    --input_json /root/datasets/segrl_pretrain_braintumour.jsonl \
    --output_dir /root/output/braintumour

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


```

已新增可视化脚本 `seg-rl/visualization/viz_sam_segmentation.py`，按规范批量生成对比图，并按提示点数量分目录输出。

功能要点
- 读取 `--input_json`（数组，含 `image`, `gt_mask`, `sam_masks_dir`, `points`, `labels`）。
- 为每个样本、每个可用的提示点数量 k（由 metrics 文件、预测子目录和 points 长度三路综合发现）生成一张可视化图，输出到 `--output_dir/<k>/<stem>.jpg`。
- 三列内容：
  - 原图叠加半透明 GT（绿色）。
  - 原图叠加半透明 Pred（红色）+ 前 k 个提示点序列（前景点用“^”，背景点用“X”，旁以数字标注顺序）。
  - 原图灰度化后叠加半透明 GT（绿）与 Pred（红）。
- 图中附带 `sam_masks_dir/<stem>-metrics.jsonl` 对应 k 的全部指标文本（如 DICE/IOU/P/R/F1/S_MEASURE）。
- 自动创建输出子目录；尺寸不匹配时按最近邻对掩模重采样。

使用示例
```bash
python /Users/donchiu/Works/Seg-R1/seg-rl/visualization/viz_sam_segmentation.py \
  --input_json /root/datasets/segrl_pretrain_braintumour.jsonl \
  --output_dir /root/output/braintumour
```

输出结构（示意）
- /root/output/braintumour/
  - 1/BRAIN_00.jpg, 1/BRAIN_01.jpg
  - 2/BRAIN_00.jpg, 2/BRAIN_01.jpg

```shellscript
python /Users/donchiu/Works/Seg-R1/seg-rl/visualization/viz_sam_segmentation.py \
  --input_json /root/datasets/segrl_pretrain_braintumour.jsonl \
  --output_dir /root/output/braintumour
```

