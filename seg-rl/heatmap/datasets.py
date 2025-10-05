from __future__ import annotations

"""
JSONL点定位数据集支持多种格式

新格式 (来自 gen_point_jsonl_from_masks.py):
    {"image": "/path/to/img.jpg", "points": [[x1,y1], [x2,y2], ...], "labels": [1, 0, ...]}
    - points: 坐标点列表，每个点为[x,y]格式
    - labels: 对应的标签，1表示正类（前景），0表示负类（背景）
    - 只使用label=1的点进行训练，如有多个正类点则选择第一个
    
旧格式 (向后兼容):
    {"image": "/path/to/img.jpg", "x": 123.4, "y": 456.7}
    - 直接包含单个点的x,y坐标

生成工具:
    python seg-rl/annotator/gen_point_jsonl_from_masks.py \\
      --images_dir /path/to/images \\
      --masks_dir /path/to/masks \\
      --output_jsonl /path/to/output.jsonl
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import os


@dataclass
class ImageSize:
    height: int
    width: int


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"L", "I;16", "I", "F"}:
        return image.convert("RGB")
    return image.convert("RGB")


def _resize_with_coords(image: Image.Image, xy: Tuple[float, float], out_size: ImageSize) -> Tuple[Image.Image, Tuple[float, float]]:
    orig_w, orig_h = image.size
    x, y = xy
    # Simple direct resize, scale coords accordingly
    scale_x = out_size.width / float(orig_w)
    scale_y = out_size.height / float(orig_h)
    x_resized = x * scale_x
    y_resized = y * scale_y
    image_resized = image.resize((out_size.width, out_size.height), resample=Image.BILINEAR)
    return image_resized, (x_resized, y_resized)


def _random_hflip(image: Image.Image, xy: Tuple[float, float], p: float = 0.5) -> Tuple[Image.Image, Tuple[float, float]]:
    if np.random.rand() >= p:
        return image, xy
    w, _ = image.size
    x, y = xy
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    x_flipped = (w - 1) - x
    return image, (x_flipped, y)


def _random_vflip(image: Image.Image, xy: Tuple[float, float], p: float = 0.0) -> Tuple[Image.Image, Tuple[float, float]]:
    if np.random.rand() >= p:
        return image, xy
    w, h = image.size
    x, y = xy
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    y_flipped = (h - 1) - y
    return image, (x, y_flipped)


class JsonlPointDataset(Dataset):
    """Dataset for JSONL lines supporting multiple formats:
    
    New format (from gen_point_jsonl_from_masks.py):
        {"image": "/path/to/img.jpg", "points": [[x1,y1], [x2,y2], ...], "labels": [1, 0, ...]}
    
    Legacy format (backward compatibility):
        {"image": "/path/to/img.jpg", "x": x1, "y": y1}
    
    For new format:
        - Uses only points with label=1 (positive/foreground)
        - If multiple positive points exist, picks the first one
        - Skips samples with no positive points
    
    Options:
        - resize: resize to fixed HxW and scale coordinates
        - random flips: horizontal/vertical (vflip off by default)
        - normalization: mean/std to Tensor
    """

    def __init__(
        self,
        jsonl_path: str,
        image_size: ImageSize,
        hflip_p: float = 0.5,
        vflip_p: float = 0.0,
        mean_rgb: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std_rgb: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        mean_gray: float = 0.5,
        std_gray: float = 0.5,
        training: bool = True,
    ) -> None:
        super().__init__()
        self.entries: List[Dict[str, object]] = []
        # Read entire file as JSON array if possible; fallback to JSONL per-line
        recs: List[Dict] = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            data = json.loads(content)
            if isinstance(data, list):
                recs = [obj for obj in data if isinstance(obj, dict)]
            else:
                raise ValueError("Root is not a JSON array")
        except Exception:
            # Fallback: JSONL lines
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            recs.append(obj)
                    except Exception:
                        continue
        # Validate/process
        for i, obj in enumerate(recs, 1):
            try:
                if not self._validate_and_process_entry(obj, i):
                    continue
                self.entries.append(obj)
            except (ValueError, KeyError, IndexError) as e:
                print(f"[WARN] Skipping invalid JSON at idx {i}: {e}")
                continue
        
        if len(self.entries) == 0:
            raise ValueError(f"No valid entries found in {jsonl_path}")
        
        self.image_size = image_size
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb
        self.mean_gray = float(mean_gray)
        self.std_gray = float(std_gray)
        self.training = training
        
    def _validate_and_process_entry(self, obj: Dict, line_num: int) -> bool:
        """Validate and normalize entry to internal format. Returns True if valid."""
        if "image" not in obj:
            print(f"[WARN] Line {line_num}: Missing 'image' field")
            return False
            
        # Check format: new format has "points" and "labels", legacy has "x" and "y"
        if "points" in obj and "labels" in obj:
            # New format
            points = obj["points"]
            labels = obj["labels"]
            
            if not isinstance(points, list) or not isinstance(labels, list):
                print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
                return False
                
            if len(points) != len(labels):
                print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
                return False
            
            # Choose the first (point,label) pair deterministically
            first_point = points[0]
            first_label = labels[0]
            if not isinstance(first_point, (list, tuple)) or len(first_point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {first_point}")
                return False
            if first_label not in (0, 1):
                print(f"[WARN] Line {line_num}: Invalid label value {first_label}, expected 0 or 1")
                return False
            # Store in internal format
            obj["_x"] = float(first_point[0])
            obj["_y"] = float(first_point[1])
            obj["_label"] = int(first_label)
            
        elif "x" in obj and "y" in obj:
            # Legacy format
            obj["_x"] = float(obj["x"])
            obj["_y"] = float(obj["y"])
            # If legacy has label field, use it; else default to 1
            obj["_label"] = int(obj.get("label", 1))
            
        else:
            print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
            return False
            
        return True

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        e = self.entries[idx]
        img_path = e["image"]
        # Prefer explicit conditional image path if provided; otherwise fallback to gt_mask as grayscale condition
        cond_path = e.get("cond") if isinstance(e, dict) else None
        if cond_path is None:
            cond_path = e.get("gt_mask") if isinstance(e, dict) else None
        # Use normalized internal coordinates
        x = float(e["_x"])  # pixel coordinates in original image
        y = float(e["_y"])

        image = Image.open(img_path).convert("RGB")
        image = _ensure_rgb(image)
        cond_img: Image.Image
        if cond_path is None:
            # If second image missing, fall back to grayscale of the RGB image to keep channel count stable
            cond_img = image.convert("L")
        else:
            cond_img = Image.open(cond_path).convert("L")

        # Resize and coordinate scaling
        image, (x, y) = _resize_with_coords(image, (x, y), self.image_size)
        cond_img, _ = _resize_with_coords(cond_img, (x, y), self.image_size)

        # Augmentations (flip) - apply identical transforms to both images and update coords once
        if self.training:
            # decide flips once
            do_h = (np.random.rand() < self.hflip_p)
            do_v = (np.random.rand() < self.vflip_p)
            if do_h:
                w, _ = image.size
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                cond_img = cond_img.transpose(Image.FLIP_LEFT_RIGHT)
                x = (w - 1) - x
            if do_v:
                _, h = image.size
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                cond_img = cond_img.transpose(Image.FLIP_TOP_BOTTOM)
                y = (h - 1) - y

        # To tensor and normalize
        image_t = TF.to_tensor(image)  # [3,H,W], 0..1
        image_t = TF.normalize(image_t, mean=self.mean_rgb, std=self.std_rgb)
        cond_t = TF.to_tensor(cond_img)  # [1,H,W]
        cond_t = (cond_t - self.mean_gray) / max(self.std_gray, 1e-6)
        # Stack to 4 channels: RGB + Gray
        image_pair = torch.cat([image_t, cond_t], dim=0)  # [4,H,W]

        # Pack target as tensor [2]
        target_xy = torch.tensor([x, y], dtype=torch.float32)
        target_label = torch.tensor(int(e.get("_label", 1)), dtype=torch.long)

        sample = {
            "image": image_pair,
            "image_rgb": image_t,
            "image_gray": cond_t,
            "target_xy": target_xy,
            "target_label": target_label,
            "meta": {
                "path": img_path,
                "cond_path": cond_path,
                "height": self.image_size.height,
                "width": self.image_size.width,
            },
        }
        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets_xy = torch.stack([b["target_xy"] for b in batch], dim=0)
    targets_label = torch.stack([b["target_label"] for b in batch], dim=0)
    metas = [b["meta"] for b in batch]
    return {"image": images, "target_xy": targets_xy, "target_label": targets_label, "meta": metas}


class SamSequencePointDataset(Dataset):
    """Dataset for self-supervised heatmap training from segmentation sequences.

    For each entry in JSONL containing fields:
      {"image": "/path/img.jpg", "points": [[x,y], ...], "labels": [1,0,...]}

    Generate one sample per index i in 0..len(points)-1:
      - first input image: the image itself (RGB)
      - second input image (gray):
          i == 0  -> all-zero gray image
          i > 0   -> load mask at {sam_dir}/{stem}/{i-1}.png
      - target: (points[i], labels[i])

    Notes:
      - stem is derived from the image file name (without extension)
      - both images are resized to the fixed ImageSize; flips are applied synchronously
    """

    def __init__(
        self,
        jsonl_path: str,
        sam_dir: str,
        image_size: ImageSize,
        hflip_p: float = 0.5,
        vflip_p: float = 0.0,
        mean_rgb: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std_rgb: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        mean_gray: float = 0.5,
        std_gray: float = 0.5,
        training: bool = True,
    ) -> None:
        super().__init__()
        self.entries: List[Dict[str, object]] = []
        # Prefer JSON array; fallback to JSONL
        recs: List[Dict] = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            data = json.loads(content)
            if isinstance(data, list):
                recs = [obj for obj in data if isinstance(obj, dict)]
            else:
                raise ValueError("Root is not a JSON array")
        except Exception:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            recs.append(obj)
                    except Exception:
                        continue
        for obj in recs:
            # minimal validation
            if "image" not in obj or "points" not in obj or "labels" not in obj:
                continue
            points = obj.get("points", [])
            labels = obj.get("labels", [])
            if not isinstance(points, list) or not isinstance(labels, list) or len(points) != len(labels) or len(points) == 0:
                continue
            self.entries.append(obj)

        if len(self.entries) == 0:
            raise ValueError(f"No valid entries found in {jsonl_path}")

        self.sam_dir = os.path.abspath(sam_dir)
        self.image_size = image_size
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb
        self.mean_gray = float(mean_gray)
        self.std_gray = float(std_gray)
        self.training = training

        # Build (entry_idx, point_idx) index
        self.index: List[Tuple[int, int]] = []
        for ei, obj in enumerate(self.entries):
            pts = obj.get("points", [])  # type: ignore
            for i in range(len(pts)):
                self.index.append((ei, i))

    def __len__(self) -> int:
        return len(self.index)

    def _stem(self, path: str) -> str:
        name = os.path.basename(path)
        base, _ext = os.path.splitext(name)
        return base

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ei, pi = self.index[idx]
        e = self.entries[ei]
        assert isinstance(e, dict)
        img_path = str(e["image"])  # absolute or relative
        pts = e.get("points", [])  # type: ignore
        labs = e.get("labels", [])  # type: ignore
        point = pts[pi]
        label = labs[pi]
        x = float(point[0])
        y = float(point[1])
        label_int = int(label)

        # load RGB image
        image = Image.open(img_path).convert("RGB")
        image = _ensure_rgb(image)

        # build conditional gray image
        if pi == 0:
            cond_img = Image.new("L", (image.width, image.height), color=0)
        else:
            stem = self._stem(img_path)
            mask_path = os.path.join(self.sam_dir, stem, f"{pi - 1}.png")
            try:
                cond_img = Image.open(mask_path).convert("L")
            except Exception:
                cond_img = Image.new("L", (image.width, image.height), color=0)

        # Resize and coordinate scaling (use RGB image as reference for scaling)
        image, (x, y) = _resize_with_coords(image, (x, y), self.image_size)
        cond_img = cond_img.resize((self.image_size.width, self.image_size.height), resample=Image.NEAREST)

        # Synchronized flips
        if self.training:
            do_h = (np.random.rand() < self.hflip_p)
            do_v = (np.random.rand() < self.vflip_p)
            if do_h:
                w, _ = image.size
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                cond_img = cond_img.transpose(Image.FLIP_LEFT_RIGHT)
                x = (w - 1) - x
            if do_v:
                _, h = image.size
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                cond_img = cond_img.transpose(Image.FLIP_TOP_BOTTOM)
                y = (h - 1) - y

        # To tensor and normalize
        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, mean=self.mean_rgb, std=self.std_rgb)
        cond_t = TF.to_tensor(cond_img)
        cond_t = (cond_t - self.mean_gray) / max(self.std_gray, 1e-6)
        image_pair = torch.cat([image_t, cond_t], dim=0)

        target_xy = torch.tensor([x, y], dtype=torch.float32)
        target_label = torch.tensor(label_int, dtype=torch.long)

        sample = {
            "image": image_pair,
            "image_rgb": image_t,
            "image_gray": cond_t,
            "target_xy": target_xy,
            "target_label": target_label,
            "meta": {
                "path": img_path,
                "height": self.image_size.height,
                "width": self.image_size.width,
                "step_idx": pi,
            },
        }
        return sample


