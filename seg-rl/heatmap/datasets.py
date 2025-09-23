from __future__ import annotations

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
    """Dataset for JSONL lines: {"image": path, "x": int/float, "y": int/float}.

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
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                assert "image" in obj and "x" in obj and "y" in obj, "Each line must contain image,x,y"
                self.entries.append(obj)
        self.image_size = image_size
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.mean = mean
        self.std = std
        self.training = training

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        e = self.entries[idx]
        img_path = e["image"]
        x = float(e["x"])  # pixel coordinates in original image
        y = float(e["y"])

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


