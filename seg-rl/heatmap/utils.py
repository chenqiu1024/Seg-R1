from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np


@dataclass
class Checkpoint:
    model: Dict
    optimizer: Dict
    scaler: Dict
    epoch: int
    step: int
    metadata: Dict | None = None  # 添加元数据字段


def save_checkpoint(
    path: str, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    scaler: torch.cuda.amp.GradScaler, 
    epoch: int, 
    step: int,
    metadata: Dict | None = None,
) -> None:
    """保存 checkpoint，包含模型元数据
    
    Args:
        path: 保存路径
        model: 模型
        optimizer: 优化器
        scaler: GradScaler
        epoch: 当前 epoch
        step: 当前步数
        metadata: 模型元数据，如 {"use_sam_encoder": True, "sam_lora_enabled": True, ...}
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 自动检测模型类型
    if metadata is None:
        metadata = {}
        # 检测是否使用 SAM encoder
        if hasattr(model, "sam_encoder"):
            metadata["use_sam_encoder"] = True
            # 检测 PEFT 方法
            if hasattr(model.sam_encoder, "peft_method"):
                metadata["sam_peft_method"] = model.sam_encoder.peft_method
                # 向后兼容
                metadata["sam_lora_enabled"] = (model.sam_encoder.peft_method == "late_lora")
            else:
                # 旧版本兼容
                metadata["sam_lora_enabled"] = hasattr(model.sam_encoder, "peft_modules") and len(model.sam_encoder.peft_modules) > 0
                metadata["sam_peft_method"] = "late_lora" if metadata["sam_lora_enabled"] else None
        else:
            metadata["use_sam_encoder"] = False
            metadata["sam_lora_enabled"] = False
            metadata["sam_peft_method"] = None
    
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "metadata": metadata,
        "version": "1.0",  # checkpoint 版本号
    }, path)


def load_checkpoint(
    path: str, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer | None = None, 
    scaler: torch.cuda.amp.GradScaler | None = None,
    strict: bool = True,
) -> Checkpoint:
    """加载 checkpoint，支持向后兼容
    
    Args:
        path: checkpoint 路径
        model: 目标模型
        optimizer: 优化器（可选）
        scaler: GradScaler（可选）
        strict: 是否严格匹配模型参数
        
    Returns:
        Checkpoint 对象
    """
    ckpt = torch.load(path, map_location="cpu")
    
    # 获取元数据
    metadata = ckpt.get("metadata", {})
    
    # 检查兼容性
    model_has_sam = hasattr(model, "sam_encoder")
    ckpt_has_sam = metadata.get("use_sam_encoder", False)
    
    if model_has_sam != ckpt_has_sam:
        print(f"[Warning] Model type mismatch:")
        print(f"  Current model uses SAM encoder: {model_has_sam}")
        print(f"  Checkpoint uses SAM encoder: {ckpt_has_sam}")
        
        if not model_has_sam and ckpt_has_sam:
            print("[Warning] Cannot load SAM-based checkpoint into non-SAM model")
            print("[Warning] Attempting to load with strict=False, some parameters may be missing")
            strict = False
        elif model_has_sam and not ckpt_has_sam:
            print("[Warning] Cannot load non-SAM checkpoint into SAM model")
            print("[Warning] Only loading point prediction head parameters")
            # 只加载非 SAM 部分的参数
            filtered_state = {}
            for k, v in ckpt["model"].items():
                if not k.startswith("sam_encoder"):
                    filtered_state[k] = v
            ckpt["model"] = filtered_state
            strict = False
    
    # 加载模型参数
    try:
        model.load_state_dict(ckpt["model"], strict=strict)
        print(f"[Checkpoint] Loaded model parameters (strict={strict})")
    except Exception as e:
        print(f"[Error] Failed to load model parameters: {e}")
        if strict:
            print("[Info] Retrying with strict=False...")
            model.load_state_dict(ckpt["model"], strict=False)
    
    # 加载优化器和scaler
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("[Checkpoint] Loaded optimizer state")
        except Exception as e:
            print(f"[Warning] Failed to load optimizer state: {e}")
    
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
            print("[Checkpoint] Loaded scaler state")
        except Exception as e:
            print(f"[Warning] Failed to load scaler state: {e}")
    
    return Checkpoint(
        model=ckpt.get("model", {}), 
        optimizer=ckpt.get("optimizer", {}), 
        scaler=ckpt.get("scaler", {}), 
        epoch=ckpt.get("epoch", 0), 
        step=ckpt.get("step", 0),
        metadata=metadata,
    )


def compute_pck(pred_xy: torch.Tensor, target_xy: torch.Tensor, thresh: float) -> float:
    """Percentage of Correct Keypoints under pixel distance threshold."""
    d = torch.linalg.norm(pred_xy - target_xy, dim=1)
    return (d <= thresh).float().mean().item()


def overlay_point_on_image(image: Image.Image, xy: Tuple[float, float], color: Tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    r = 3
    x, y = xy
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
    return img


def draw_cross(image: Image.Image, xy: Tuple[float, float], color: Tuple[int, int, int] = (0, 255, 0), size: int = 6, width: int = 2) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x, y = xy
    draw.line((x - size, y, x + size, y), fill=color, width=width)
    draw.line((x, y - size, x, y + size), fill=color, width=width)
    return img


def draw_triangle(image: Image.Image, xy: Tuple[float, float], color: Tuple[int, int, int] = (255, 0, 0), size: int = 9, width: int = 2) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x, y = xy
    pts = [
        (x, y - size),
        (x - size * 0.866, y + size * 0.5),
        (x + size * 0.866, y + size * 0.5),
    ]
    # filled triangle with black outline to ensure visibility on any background
    draw.polygon(pts, fill=color, outline=(0, 0, 0))
    return img


def draw_diagonal_cross(image: Image.Image, xy: Tuple[float, float], color: Tuple[int, int, int] = (255, 0, 0), size: int = 8, width: int = 2) -> Image.Image:
    """Draw a 45-degree cross (X-shape). Renders a black stroke underlay for visibility."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x, y = xy
    # black under-stroke
    if width >= 2:
        w2 = width + 2
        draw.line((x - size, y - size, x + size, y + size), fill=(0, 0, 0), width=w2)
        draw.line((x - size, y + size, x + size, y - size), fill=(0, 0, 0), width=w2)
    # colored cross
    draw.line((x - size, y - size, x + size, y + size), fill=color, width=width)
    draw.line((x - size, y + size, x + size, y - size), fill=color, width=width)
    return img


def colormap_jet(array01: np.ndarray) -> np.ndarray:
    """Map [0,1] array to Jet colormap RGB uint8.

    Lightweight implementation to avoid heavy deps.
    """
    a = np.clip(array01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * a - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * a - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * a - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def heatmap_to_pil(heatmap: np.ndarray) -> Image.Image:
    """Convert HxW heatmap (float) to color Image via jet colormap."""
    hmin, hmax = float(np.min(heatmap)), float(np.max(heatmap))
    if hmax - hmin < 1e-12:
        norm = np.zeros_like(heatmap, dtype=np.float32)
    else:
        norm = (heatmap - hmin) / (hmax - hmin)
    rgb = colormap_jet(norm)
    return Image.fromarray(rgb)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    hm = heatmap_to_pil(heatmap).resize(image.size)
    return Image.blend(image.convert("RGB"), hm.convert("RGB"), alpha)


def make_grid(images: List[Image.Image], cols: int = 4, bg_color: Tuple[int, int, int] = (0, 0, 0), padding: int = 2) -> Image.Image:
    if not images:
        raise ValueError("Empty images for grid")
    w, h = images[0].size
    rows = int(np.ceil(len(images) / float(cols)))
    grid_w = cols * w + (cols - 1) * padding
    grid_h = rows * h + (rows - 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), color=bg_color)
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        x = c * (w + padding)
        y = r * (h + padding)
        grid.paste(img, (x, y))
    return grid


