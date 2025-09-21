from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw


@dataclass
class Checkpoint:
    model: Dict
    optimizer: Dict
    scaler: Dict
    epoch: int
    step: int


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, epoch: int, step: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
    }, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, scaler: torch.cuda.amp.GradScaler | None = None) -> Checkpoint:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return Checkpoint(model=ckpt.get("model", {}), optimizer=ckpt.get("optimizer", {}), scaler=ckpt.get("scaler", {}), epoch=ckpt.get("epoch", 0), step=ckpt.get("step", 0))


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


