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
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        training: bool = True,
    ) -> None:
        super().__init__()
        self.entries: List[Dict[str, object]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if not self._validate_and_process_entry(obj, line_num):
                        continue
                    self.entries.append(obj)
                except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
                    print(f"[WARN] Skipping invalid JSON at line {line_num}: {e}")
                    continue
        
        if len(self.entries) == 0:
            raise ValueError(f"No valid entries found in {jsonl_path}")
        
        self.image_size = image_size
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.mean = mean
        self.std = std
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
            
            # Find first positive point (label=1)
            positive_point = None
            for point, label in zip(points, labels):
                if label == 1:
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        print(f"[WARN] Line {line_num}: Invalid point format {point}")
                        return False
                    positive_point = point
                    break
                    
            if positive_point is None:
                print(f"[WARN] Line {line_num}: No positive points (label=1) found")
                return False
                
            # Store in internal format
            obj["_x"] = float(positive_point[0])
            obj["_y"] = float(positive_point[1])
            
        elif "x" in obj and "y" in obj:
            # Legacy format
            obj["_x"] = float(obj["x"])
            obj["_y"] = float(obj["y"])
            
        else:
            print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
            return False
            
        return True

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        e = self.entries[idx]
        img_path = e["image"]
        # Use normalized internal coordinates
        x = float(e["_x"])  # pixel coordinates in original image
        y = float(e["_y"])

        image = Image.open(img_path).convert("RGB")
        image = _ensure_rgb(image)

        # Resize and coordinate scaling
        image, (x, y) = _resize_with_coords(image, (x, y), self.image_size)

        # Augmentations (flip)
        if self.training:
            image, (x, y) = _random_hflip(image, (x, y), p=self.hflip_p)
            image, (x, y) = _random_vflip(image, (x, y), p=self.vflip_p)

        # To tensor and normalize
        image_t = TF.to_tensor(image)  # [3,H,W], 0..1
        image_t = TF.normalize(image_t, mean=self.mean, std=self.std)

        # Pack target as tensor [2]
        target_xy = torch.tensor([x, y], dtype=torch.float32)

        sample = {
            "image": image_t,
            "target_xy": target_xy,
            "meta": {
                "path": img_path,
                "height": self.image_size.height,
                "width": self.image_size.width,
            },
        }
        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = torch.stack([b["target_xy"] for b in batch], dim=0)
    metas = [b["meta"] for b in batch]
    return {"image": images, "target_xy": targets, "meta": metas}


