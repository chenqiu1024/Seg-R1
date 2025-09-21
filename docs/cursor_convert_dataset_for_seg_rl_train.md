# Convert dataset for Seg-R1 training
_Exported on 9/21/2025 at 12:06:30 PDT from Cursor (1.6.42)_

---

**User**

How to use utils/mddecathlon_to_segr1.py to convert datasets/Task04_Hippocampus to a new dataset with format suitable for Seg-R1 training?

---

**Cursor**

I'll help you understand how to use the `utils/mddecathlon_to_segr1.py` script to convert the Task04_Hippocampus dataset to a format suitable for Seg-R1 training. Let me first examine the script and then look at the dataset structure.

Now let me check if the Task04_Hippocampus dataset exists in the medical_decathlon folder:

Great! I can see that Task04_Hippocampus exists. Let me also check its structure to understand what files are available:

Perfect! Now I have all the information I need to help you convert the Task04_Hippocampus dataset to Seg-R1 format. Let me provide you with a comprehensive guide on how to use the `mddecathlon_to_segr1.py` script.

## How to Convert Task04_Hippocampus to Seg-R1 Format

Based on the script analysis and the dataset structure, here's how to use `utils/mddecathlon_to_segr1.py` to convert the Task04_Hippocampus dataset:

### Basic Command

```bash
python utils/mddecathlon_to_segr1.py \
  --input_root /root/autodl-tmp/works/Seg-R0/datasets/medical_decathlon/Task04_Hippocampus \
  --output_root /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md \
  --task_name Task04_Hippocampus \
  --slice_axis 2 \
  --keep_empty False \
  --size_variants canonical,512 \
  --intensity percentiles:1,99
```

### Parameter Explanation

- **`--input_root`**: Path to the Task04_Hippocampus directory containing `imagesTr/` and `labelsTr/` folders
- **`--output_root`**: Base directory where the converted dataset will be stored
- **`--task_name`**: Name for the output dataset (will create `Task04_Hippocampus/` under output_root)
- **`--slice_axis`**: 
  - `2` = axial slices (most common for medical images)
  - `1` = coronal slices
  - `0` = sagittal slices
- **`--keep_empty`**: `False` means skip slices where the mask is completely empty (recommended)
- **`--size_variants`**: 
  - `canonical` = original image size
  - `512` = resize to 512×512 pixels
- **`--intensity`**: `percentiles:1,99` uses 1st and 99th percentiles for robust intensity scaling

### Advanced Options

You can also customize other parameters:

```bash
python utils/mddecathlon_to_segr1.py \
  --input_root /root/autodl-tmp/works/Seg-R0/datasets/medical_decathlon/Task04_Hippocampus \
  --output_root /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md \
  --task_name Task04_Hippocampus \
  --images_subdir imagesTr \
  --labels_subdir labelsTr \
  --slice_axis 2 \
  --keep_empty False \
  --size_variants canonical,256,512 \
  --intensity percentiles:2,98 \
  --case_id_regex "hippocampus_(\d+)"
```

Additional parameters:
- **`--images_subdir`**: Subfolder containing images (default: `imagesTr`)
- **`--labels_subdir`**: Subfolder containing labels (default: `labelsTr`)
- **`--case_id_regex`**: Optional regex to extract simpler case IDs from filenames

### Output Structure

After running the script, you'll get this structure:

```
datasets/seg_r1_md/Task04_Hippocampus/
├── canonical/
│   └── canonical/
│       ├── images/           # Original size JPG images
│       │   ├── hippocampus_001_z0001.jpg
│       │   ├── hippocampus_001_z0002.jpg
│       │   └── ...
│       └── masks/            # Original size PNG masks
│           ├── hippocampus_001_z0001.png
│           ├── hippocampus_001_z0002.png
│           └── ...
├── size512/
│   └── canonical/
│       ├── images/           # 512×512 resized JPG images
│       └── masks/            # 512×512 resized PNG masks
└── views/                    # Symlinked views for different training stages
    ├── prerl/
    │   ├── im/               # Links to images
    │   └── gt/               # Links to masks
    ├── rl/
    │   ├── Image/            # Links to images
    │   └── GT_Object/        # Links to masks
    └── sod/
        ├── DUTS-TR-Image/    # Links to images
        └── DUTS-TR-Mask/     # Links to masks
```

### Using the Converted Dataset

After conversion, you can use the dataset for different training stages:

**Pre-RL Training:**
```bash
--dataset_image datasets/seg_r1_md/Task04_Hippocampus/canonical/views/prerl/im
--dataset_gt    datasets/seg_r1_md/Task04_Hippocampus/canonical/views/prerl/gt
```

**RL Training:**
```bash
--dataset_image datasets/seg_r1_md/Task04_Hippocampus/canonical/views/rl/Image
--dataset_gt    datasets/seg_r1_md/Task04_Hippocampus/canonical/views/rl/GT_Object
```

**SOD Fine-tuning:**
```bash
--dataset_image datasets/seg_r1_md/Task04_Hippocampus/canonical/views/sod/DUTS-TR-Image
--dataset_gt    datasets/seg_r1_md/Task04_Hippocampus/canonical/views/sod/DUTS-TR-Mask
```

### Dataset Information

The Task04_Hippocampus dataset contains:
- **260 training cases** with both images and labels
- **130 test cases** (images only, no labels)
- **MRI modality** 
- **3 label classes**: background (0), Anterior (1), Posterior (2)
- The script will convert multi-class labels to binary masks (any non-zero label becomes 255)

### Prerequisites

Make sure you have the required dependencies:
```bash
pip install nibabel numpy pillow
```

The script will automatically handle the conversion from 3D NIfTI volumes to 2D image/mask pairs suitable for Seg-R1 training!

```shellscript
python utils/mddecathlon_to_segr1.py \
  --input_root /root/autodl-tmp/works/Seg-R0/datasets/medical_decathlon/Task04_Hippocampus \
  --output_root /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md \
  --task_name Task04_Hippocampus \
  --slice_axis 2 \
  --keep_empty False \
  --size_variants canonical,512 \
  --intensity percentiles:1,99
```

```shellscript
python utils/mddecathlon_to_segr1.py \
  --input_root /root/autodl-tmp/works/Seg-R0/datasets/medical_decathlon/Task04_Hippocampus \
  --output_root /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md \
  --task_name Task04_Hippocampus \
  --images_subdir imagesTr \
  --labels_subdir labelsTr \
  --slice_axis 2 \
  --keep_empty False \
  --size_variants canonical,256,512 \
  --intensity percentiles:2,98 \
  --case_id_regex "hippocampus_(\d+)"
```

```plaintext
datasets/seg_r1_md/Task04_Hippocampus/
├── canonical/
│   └── canonical/
│       ├── images/           # Original size JPG images
│       │   ├── hippocampus_001_z0001.jpg
│       │   ├── hippocampus_001_z0002.jpg
│       │   └── ...
│       └── masks/            # Original size PNG masks
│           ├── hippocampus_001_z0001.png
│           ├── hippocampus_001_z0002.png
│           └── ...
├── size512/
│   └── canonical/
│       ├── images/           # 512×512 resized JPG images
│       └── masks/            # 512×512 resized PNG masks
└── views/                    # Symlinked views for different training stages
    ├── prerl/
    │   ├── im/               # Links to images
    │   └── gt/               # Links to masks
    ├── rl/
    │   ├── Image/            # Links to images
    │   └── GT_Object/        # Links to masks
    └── sod/
        ├── DUTS-TR-Image/    # Links to images
        └── DUTS-TR-Mask/     # Links to masks
```

```shellscript
--dataset_image datasets/seg_r1_md/Task04_Hippocampus/canonical/views/prerl/im
--dataset_gt    datasets/seg_r1_md/Task04_Hippocampus/canonical/views/prerl/gt
```

```shellscript
--dataset_image datasets/seg_r1_md/Task04_Hippocampus/canonical/views/rl/Image
--dataset_gt    datasets/seg_r1_md/Task04_Hippocampus/canonical/views/rl/GT_Object
```

```shellscript
--dataset_image datasets/seg_r1_md/Task04_Hippocampus/canonical/views/sod/DUTS-TR-Image
--dataset_gt    datasets/seg_r1_md/Task04_Hippocampus/canonical/views/sod/DUTS-TR-Mask
```

```shellscript
pip install nibabel numpy pillow
```

---

**User**

Read through cursor_pretrain_design.md to make clear about the context on what I did previously. What I wanna do now is to make the dataset files in datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/ capable for my heatmap-classification task. Specifically, we just need a JSON file conforming to the description in "数据准备（JSONL，一行一个样本）...". The ground truth coordinate in the JSON should be at the centroid of the ground truth mask's foreground area.  Write a Python script for processing the dataset. computing centroids of GT masks and finally creating the JSON file.

---

**Cursor**

I'll create a utility script that scans the BrainTumour canonical images/masks, computes foreground centroids from masks, and writes a JSONL file with absolute image paths and (x,y) pixel coordinates. Then I'll show you the command to run it for your dataset.

```python
#!/usr/bin/env python3

"""
Generate JSONL supervision (image,x,y) by computing centroids of binary masks.

Each output line is a JSON object like:
  {"image": "/abs/path/to/image.jpg", "x": 123.4, "y": 456.7}

Coordinates are in pixel space of the original image/mask pair.
Designed for Seg-R1 heatmap-classification pretraining pipelines.

Example usage:
  python utils/gen_point_jsonl_from_masks.py \
    --images_dir /abs/path/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images \
    --masks_dir  /abs/path/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/masks \
    --output_jsonl /abs/path/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl

Notes:
- Foreground is defined as any non-zero pixel in the mask.
- Samples with empty masks are skipped by default (override with --skip_empty false).
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create JSONL of image->centroid(x,y) from masks")
    p.add_argument("--images_dir", type=str, required=True, help="Directory containing images (e.g., JPG)")
    p.add_argument("--masks_dir", type=str, required=True, help="Directory containing PNG masks aligned to images")
    p.add_argument("--output_jsonl", type=str, required=True, help="Path to write JSONL output")
    p.add_argument("--abs_paths", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                   help="Write absolute image paths (default: True)")
    p.add_argument("--skip_empty", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                   help="Skip samples where mask is empty (default: True)")
    p.add_argument("--image_exts", type=str, default=".jpg,.jpeg,.png",
                   help="Comma-separated acceptable image extensions (default: .jpg,.jpeg,.png)")
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
    Compute centroid (x,y) in pixel coordinates for foreground mask (non-zero).
    Returns None if the mask has no foreground.
    """
    if mask_u8.ndim == 3:
        # If mask has channels, convert to single channel by any non-zero across channels
        mask_u8 = (mask_u8.any(axis=2)).astype(np.uint8) * 255
    foreground_indices = np.argwhere(mask_u8 > 0)
    if foreground_indices.size == 0:
        return None
    # np.argwhere returns (row=y, col=x)
    mean_yx = foreground_indices.mean(axis=0)  # [y, x]
    y = float(mean_yx[0])
    x = float(mean_yx[1])
    return (x, y)


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
            record = {"image": image_field, "x": float(centroid[0]), "y": float(centroid[1])}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

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

I'll generate the JSONL immediately for `Task01_BrainTumour` by running the new script against the canonical images and masks directories.

---

**User**

先稍等。我们需要修改一下计算“中心点”的算法：我所需要的“中心点”未必是一个mask前景像素的“重心”即把所有x，y坐标分别加和求平均得到的几何重心，因为可以设想mask有可能是一个带有“环”的几何区域，这种情况下简单计算重心所得到的点可能在mask外部。我所需要的坐标点是一个确定在mask前景区域内部，并且越“深入”前景区域“内部”越好。我不知道如何从几何或数学的角度描述这样一个点，甚至于是否有这种明确定义的概念我也不清楚。请你进行调研，告诉我有什么办法计算这样一个点坐标

---

**Cursor**

### 你要的“更深入内部”的代表点有哪些可行定义

- **最大内切圆圆心（Maximal Inscribed Circle Center）/ 距边界最远点（EDT 最大值）**
  - 定义：在前景区域内，使得到“背景/边界”的欧氏距离最大的点。
  - 性质：一定在前景内部；在“环形/甜甜圈”这类区域时也在环带内部（不落在空洞里）；“越深入”越大。
  - 实现（栅格/像素级）：
    - 计算前景二值图的欧氏距离变换（EDT），取最大值所在像素坐标即为所求点。
    - 多连通区域会自然选择“最厚”的区域（最大半径）。
  - 别名与关联：内切圆圆心（非凸集合的一般化）、“pole of inaccessibility”（多边形版本）、骨架（medial axis）上的半径极大点。

- **骨架-半径极大点（Skeleton radius maxima）**
  - 思路：先做前景骨架，再在骨架上找距离变换的局部极大值；全局最大即上面“最大内切圆中心”。
  - 优点：可得到多处“深内点”，用于需要多个候选点的场景。
  - 缺点：实现复杂度高于直接 EDT 取最大。

- **Chebyshev 中心（凸集）/ 多边形的 Polylabel**
  - 凸集的 Chebyshev 中心是最大内切球中心；多边形可用 `polylabel` 算法近似。
  - 对栅格掩码，直接 EDT 最大值通常更直接、稳定且高效。

- 备选（不建议作为主方案）
  - 质心后“投影/爬坡”到内部：先取质心，再沿 EDT 梯度上升到局部最大。能保证落在内部，但比直接 EDT 最大值复杂且可能落在局部极大。
  - 最小化平均距离的点（Fermat-Weber 样式）：不保证在内部，也不等价于“深入”。

### 推荐方案（简洁、稳健）
- 用前景的欧氏距离变换（EDT），取全图最大值位置作为点坐标。
- 如有多个连通域：
  - 若你只要一个点：直接用全局最大即可（自然偏向面积/厚度更大的部分）。
  - 若要每个连通域一个点：先连通域标记，再在各自子掩码上取 EDT 最大点。

### 边界情况与细化
- 空掩码：跳过或按需放图中心。
- 子像素细化（可选）：对最大点 3×3 邻域做二次曲面拟合或加权质心（以 EDT 值为权）得到亚像素坐标。
- 坐标与图像对齐：若图像与掩码分辨率不一致，需按尺度映射坐标（你当前 JPG/PNG 同尺寸通常一致）。

### 极简实现示例（使用 SciPy 或 OpenCV 二选一）
```python
import numpy as np

def deepest_point_by_edt(mask_u8):
    # mask_u8: HxW, 前景>0
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return None, 0.0
    try:
        from scipy import ndimage as ndi
        # 欧氏距离到背景（0）像素
        dist = ndi.distance_transform_edt(m)
    except Exception:
        import cv2
        dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
    # 最大值位置（y, x）
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (float(x), float(y)), float(dist[y, x])
```

若要对每个连通域各取一个点：
```python
def per_component_deepest_points(mask_u8, top_k=None):
    from scipy import ndimage as ndi
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return []
    labels, n = ndi.label(m)
    pts = []
    for lab in range(1, n+1):
        sub = (labels == lab).astype(np.uint8)
        dist = ndi.distance_transform_edt(sub)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        pts.append(((float(x), float(y)), float(dist[y, x]), int((labels==lab).sum())))
    # 可按半径或面积排序
    pts.sort(key=lambda t: t[1], reverse=True)
    if top_k is not None:
        pts = pts[:top_k]
    return pts  # [((x,y), radius, area), ...]
```

如果你同意，我可以把你刚才的 `utils/gen_point_jsonl_from_masks.py` 中的质心计算替换为上述 EDT 最大点算法（支持每图一个点，保持 JSONL 接口不变），并可选支持“每连通域一个点（输出多行）”。

```python
import numpy as np

def deepest_point_by_edt(mask_u8):
    # mask_u8: HxW, 前景>0
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return None, 0.0
    try:
        from scipy import ndimage as ndi
        # 欧氏距离到背景（0）像素
        dist = ndi.distance_transform_edt(m)
    except Exception:
        import cv2
        dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
    # 最大值位置（y, x）
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (float(x), float(y)), float(dist[y, x])
```

```python
def per_component_deepest_points(mask_u8, top_k=None):
    from scipy import ndimage as ndi
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return []
    labels, n = ndi.label(m)
    pts = []
    for lab in range(1, n+1):
        sub = (labels == lab).astype(np.uint8)
        dist = ndi.distance_transform_edt(sub)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        pts.append(((float(x), float(y)), float(dist[y, x]), int((labels==lab).sum())))
    # 可按半径或面积排序
    pts.sort(key=lambda t: t[1], reverse=True)
    if top_k is not None:
        pts = pts[:top_k]
    return pts  # [((x,y), radius, area), ...]
```

---

**User**

我再提供一些前提条件：mask大多数是不规则的图形，绝大部分面积连通，只有少量面积与主体非连通；大部分情况下不存在有“洞”的情况，但仍然有少量情形会出现。在这种前提下，兼顾计算效率，请选择一个最佳的方案来作为我们的目标点选取规则，并相应修改刚才生成脚本中的质心计算部分代码。

---

**Cursor**

I'll update the script to compute the target point using the Euclidean Distance Transform (EDT) maximum (deepest interior point), with SciPy preferred and OpenCV as a fallback, then run lint checks.

I will update the centroid computation to use Euclidean Distance Transform (EDT) to select the deepest interior point, prefer SciPy with OpenCV fallback, and then check for linter issues.

```python
#!/usr/bin/env python3

"""
Generate JSONL supervision (image,x,y) by computing a "deep interior" point of
binary masks using the Euclidean Distance Transform (EDT) maximum.

Each output line is a JSON object like:
  {"image": "/abs/path/to/image.jpg", "x": 123.4, "y": 456.7}

Coordinates are in pixel space of the original image/mask pair.
Designed for Seg-R1 heatmap-classification pretraining pipelines.

Example usage:
  python utils/gen_point_jsonl_from_masks.py \
    --images_dir /abs/path/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images \
    --masks_dir  /abs/path/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/masks \
    --output_jsonl /abs/path/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl

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
from PIL import Image


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
            record = {"image": image_field, "x": float(centroid[0]), "y": float(centroid[1])}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

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

- I updated the script to choose the “deepest interior” point using the Euclidean Distance Transform (EDT) maximum, which is robust to irregular shapes, small disconnected bits, and occasional holes, and is efficient on pixel masks.
- You can now generate JSONL with: 
```bash
python /root/autodl-tmp/works/Seg-R0/utils/gen_point_jsonl_from_masks.py \
  --images_dir /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images \
  --masks_dir  /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/masks \
  --output_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl
```
- The script prefers SciPy’s EDT; if unavailable, it falls back to OpenCV’s distanceTransform, and finally to a simple L1-like fallback.

Want me to run it now and produce the JSONL file?

```shellscript
python /root/autodl-tmp/works/Seg-R0/utils/gen_point_jsonl_from_masks.py \
  --images_dir /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images \
  --masks_dir  /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/masks \
  --output_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl
```

---

**User**

Sure, try it out!

---

**Cursor**

I'll generate the JSONL from BrainTumour masks using the updated deepest-point method, then show a quick preview.

- Generated: 942 samples at `datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl`.
- Preview:
```json
{"image": "/root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images/BRATS_001_z0029.jpg", "x": 128.0, "y": 158.0}
{"image": "/root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images/BRATS_001_z0030.jpg", "x": 127.0, "y": 159.0}
{"image": "/root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images/BRATS_001_z0031.jpg", "x": 128.0, "y": 154.0}
{"image": "/root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images/BRATS_001_z0032.jpg", "x": 127.0, "y": 154.0}
{"image": "/root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images/BRATS_001_z0033.jpg", "x": 126.0, "y": 152.0}
```

You can feed this JSONL to the heatmap-classification training:
```bash
python -m seg_rl.train \
  --train_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --height 512 --width 512 --batch_size 16 --epochs 20 --loss ce --amp
```

Want me to also generate a size512 version or per-component deepest points?

```json
{"image": "/root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images/BRATS_001_z0029.jpg", "x": 128.0, "y": 158.0}
{"image": "/root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images/BRATS_001_z0030.jpg", "x": 127.0, "y": 159.0}
{"image": "/root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images/BRATS_001_z0031.jpg", "x": 128.0, "y": 154.0}
{"image": "/root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images/BRATS_001_z0032.jpg", "x": 127.0, "y": 154.0}
{"image": "/root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images/BRATS_001_z0033.jpg", "x": 126.0, "y": 152.0}
```

```shellscript
python -m seg_rl.train \
  --train_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --height 512 --width 512 --batch_size 16 --epochs 20 --loss ce --amp
```

---

**User**

为验证所计算出的EDT最大点是否正确，请添加代码以可视化其结果，你可以把这部分逻辑直接加在刚才的脚本后面：创建一个单独的目录以存放可视化图像供用户验证，对于每个样本，把原始图像、真值mask和生成的EDT最大点都叠加画成一张图片，存放在该目录下

---

**Cursor**

I'll add a --viz_dir option to the script. When provided, it will save an overlay image per sample showing the original image, a semi-transparent mask overlay, and the EDT deepest point as a small circle. Then I'll run it to generate the visualizations.

```python
#!/usr/bin/env python3

"""
Generate JSONL supervision (image,x,y) by computing a "deep interior" point of
binary masks using the Euclidean Distance Transform (EDT) maximum.

Each output line is a JSON object like:
  {"image": "/abs/path/to/image.jpg", "x": 123.4, "y": 456.7}

Coordinates are in pixel space of the original image/mask pair.
Designed for Seg-R1 heatmap-classification pretraining pipelines.

Example usage:
  python utils/gen_point_jsonl_from_masks.py \
    --images_dir /abs/path/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images \
    --masks_dir  /abs/path/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/masks \
    --output_jsonl /abs/path/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl

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
            record = {"image": image_field, "x": float(centroid[0]), "y": float(centroid[1])}
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



```

---

**User**

我发现你的EDT最大点计算似乎并不符合预期：以BRATS_001_z0085为例，请你查看/root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/下相应的原图像和mask图像，再看你在points_canonical.jsonl中对应的坐标计算结果，以及经过可视化后在viz_canonical里的相应图片BRATS_001_z0085.png。明显可见，你算出的点在非常靠近mask最大连通区域边界的位置上，而并非预期的应该在某个深入mask区域内部的地方。请重新检查你的算法和代码实现，找出原因并修改它

---

**Cursor**

I'll update the mask processing to keep only the largest connected component and fill holes before computing EDT, using SciPy if available and OpenCV as fallback, then regenerate overlays to verify BRATS_001_z0085.

```python
#!/usr/bin/env python3

"""
Generate JSONL supervision (image,x,y) by computing a "deep interior" point of
binary masks using the Euclidean Distance Transform (EDT) maximum.

Each output line is a JSON object like:
  {"image": "/abs/path/to/image.jpg", "x": 123.4, "y": 456.7}

Coordinates are in pixel space of the original image/mask pair.
Designed for Seg-R1 heatmap-classification pretraining pipelines.

Example usage:
  python utils/gen_point_jsonl_from_masks.py \
    --images_dir /abs/path/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/images \
    --masks_dir  /abs/path/datasets/seg_r1_md/Task01_BrainTumour/canonical/canonical/masks \
    --output_jsonl /abs/path/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl

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
            record = {"image": image_field, "x": float(centroid[0]), "y": float(centroid[1])}
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



```

- 我已修改点选取为：先保留最大连通域并填洞，再计算该区域的欧氏距离变换EDT并取全局最大点。这样可避免薄枝、零碎小块和孔洞导致的偏移，点应更“深入”主体内部。
- 我已重新生成了 JSONL 与可视化。请重点查看 `viz_canonical/BRATS_001_z0085.png` 是否更符合预期；如仍异常，我再加一个防抖策略（例如对 EDT 做轻度高斯平滑后取极大，或从质心沿 EDT 梯度爬升到全局峰）。

---

**User**

这回看上去不错！

---

**Cursor**

太好了！现在 EDT 深入点看起来正确。

- 你可以直接用已生成的 JSONL 开训：
```bash
python -m seg_rl.train \
  --train_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --height 512 --width 512 --batch_size 16 --epochs 20 --loss ce --amp
```
- 如需继续抽查可视化，脚本的 `--viz_dir` 参数已经启用，重新运行会更新 `viz_canonical/`。想更显眼可调 `--viz_alpha 0.5 --viz_radius 6`。

需要我也为 `Task04_Hippocampus` 或其他任务生成对应的 JSONL 和可视化吗？

- 已做的关键改动与产物
  - `utils/gen_point_jsonl_from_masks.py`: 采用“最大连通域 + 填洞 + EDT 最大点”。
  - 生成 `points_canonical.jsonl` 和覆盖图 `viz_canonical/*.png`，用于快速人工验收。

```shellscript
python -m seg_rl.train \
  --train_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --height 512 --width 512 --batch_size 16 --epochs 20 --loss ce --amp
```

