# 分析验证测试效果可视化问题
_Exported on 9/21/2025 at 15:06:15 PDT from Cursor (1.6.42)_

---

**User**

请通过阅读docs/cursor_pretrain_design.md了解并恢复之前对话的上下文。接着请解答我的问题：val_or_train_vis.png中得到的验证测试效果可视化图像看上去很奇怪，并没有看到表示预测坐标的三角形标记，并且热力图与原始输入大脑图像看上去也并不重合。请仔细分析原因并修改

---

**Cursor**

I'll first scan the repository to locate the code that generates the validation visualization and also open the design document to recover context. Then I'll inspect for issues around coordinate rendering and heatmap alignment.

Let me open the core modules that control visualization and coordinate spaces: drawing/overlay helpers, the model's output resolution, and the dataset transforms. This will let me verify coordinate scaling and potential misalignment sources.

I'll open the training script to inspect how the model config is constructed, how images are normalized/unnormalized, and how visualization functions are called.

I’m going to update the visualization to address both issues:
- Use a probability heatmap (softmax over HxW) to ensure overlays align meaningfully and avoid strange color scaling.
- Make the predicted-point triangle highly visible (filled with a contrasting fill and a dark outline) and clamp coordinates to image bounds.

I'll edit `seg-rl/seg_rl/utils.py` to improve `draw_triangle`, and modify `visualize_dataset` in `seg-rl/seg_rl/train.py` to use the probability heatmap and safer drawing.

我已读过 `docs/cursor_pretrain_design.md` 并检查了可视化/数据/模型相关代码。你描述的两个现象（看不到预测三角形、热力图与原图“不重合”）主要来自两点：
- 当前叠加的是“原始logits”的颜色映射，数值范围和对比度不稳定，容易出现看起来像“整幅图被染色”的效果，不易观察峰值位置。
- 预测点只画了1像素描边的空心三角形，且未钳制到图像边界，叠加半透明伪彩后很容易看不清或越界看不到。

我做了两处改动：
- 叠加“概率热力图”（对 H×W 进行 softmax 后的概率），保证热力图视觉稳定、峰值对齐。
- 把预测点的三角形改为“实心填充+黑色描边”，并对 GT/Pred 坐标进行边界钳制，适度增大尺寸，保证明显可见。

### 代码改动

```68:78:/root/autodl-tmp/works/Seg-R0/seg-rl/seg_rl/utils.py
def draw_triangle(image: Image.Image, xy: Tuple[float, float], color: Tuple[int, int, int] = (255, 0, 0), size: int = 9, width: int = 2) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x, y = xy
    pts = [
        (x, y - size),
        (x - size * 0.866, y + size * 0.5),
        (x + size * 0.866, y + size * 0.5),
    ]
    # 填充+黑色描边，提高可见性；width 对 polygon 不生效，但黑色描边已足够
    draw.polygon(pts, fill=color, outline=(0, 0, 0))
    return img
```

```162:201:/root/autodl-tmp/works/Seg-R0/seg-rl/seg_rl/train.py
    def visualize_dataset(loader: DataLoader | None, split_name: str):
        if loader is None or args.vis_mode == "none":
            return
        model.eval()
        import torchvision.transforms.functional as TF
        import numpy as np
        saved_images: List[Image.Image] = []
        count = 0
        max_count = args.vis_count if args.vis_mode == "sample" else float("inf")
        with torch.no_grad():
            for batch in loader:
                img_t = batch["image"]  # [B,3,H,W]
                tgt = batch["target_xy"]  # [B,2]
                logits = model(img_t.to(device))
                # 概率热力图用于可视化，稳定峰值
                bsz, _, H, W = logits.shape
                probs = torch.softmax(logits.view(bsz, -1), dim=1).view(bsz, 1, H, W)
                # 软argmax预测
                pred_xy = soft_argmax_from_logits(logits).cpu()
                for i in range(img_t.size(0)):
                    if count >= max_count:
                        break
                    # 反归一化到 [0,1] 便于显示
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    arr = img_t[i].cpu().float().numpy()
                    arr = (arr * std + mean).clip(0.0, 1.0)
                    pil_img = TF.to_pil_image(torch.from_numpy(arr))
                    # 使用概率图来叠加
                    hm = probs[i, 0].detach().cpu().float().numpy()
                    over = overlay_heatmap(pil_img, hm, alpha=0.5)
                    # 坐标钳制，避免越界导致标记不可见
                    h, w = img_t.size(-2), img_t.size(-1)
                    gx = float(max(0.0, min(w - 1.0, float(tgt[i, 0]))))
                    gy = float(max(0.0, min(h - 1.0, float(tgt[i, 1]))))
                    px = float(max(0.0, min(w - 1.0, float(pred_xy[i, 0]))))
                    py = float(max(0.0, min(h - 1.0, float(pred_xy[i, 1]))))
                    # 先GT十字，再Pred实心三角
                    over = draw_cross(over, (gx, gy), color=(0, 255, 0))
                    over = draw_triangle(over, (px, py), color=(255, 0, 0), size=10)
                    saved_images.append(over)
                    count += 1
                if count >= max_count:
                    break
        # save grid or per-sample
        if args.vis_mode == "sample":
            grid = make_grid(saved_images, cols=4)
            grid.save(os.path.join(vis_dir, f"{split_name}_vis.png"))
        else:  # all
            split_dir = os.path.join(vis_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            for i, img in enumerate(saved_images):
                img.save(os.path.join(split_dir, f"{i:05d}.png"))
```

### 如何复现并检查
- 重新运行（或在下一个epoch自动触发）可视化：
```bash
cd /root/autodl-tmp/works/Seg-R0/seg-rl
python -m seg_rl.train \
  --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --height 240 --width 240 --batch_size 16 --epochs 1 --loss ce \
  --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pretrain \
  --vis_mode sample --vis_count 16
```
- 查看 `outputs/seg_r1_md/Task01_BrainTumour/pretrain/vis/val_or_train_vis.png`。

如果你仍觉得热力图“泛黄/泛蓝”，可以把 `overlay_heatmap(..., alpha=0.5)` 中 `alpha` 调到 0.35～0.4 以更突出原图细节。

- 我将检查可视化与坐标空间处理的相关代码，并对热力图叠加与标记绘制做了稳健化改动，接着给出了可视化重跑方式。
- 以上 edits 会让热力图与图像对齐更直观（概率峰值），并确保预测三角标记醒目可见。

```python
def draw_triangle(image: Image.Image, xy: Tuple[float, float], color: Tuple[int, int, int] = (255, 0, 0), size: int = 9, width: int = 2) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x, y = xy
    pts = [
        (x, y - size),
        (x - size * 0.866, y + size * 0.5),
        (x + size * 0.866, y + size * 0.5),
    ]
    # 填充+黑色描边，提高可见性；width 对 polygon 不生效，但黑色描边已足够
    draw.polygon(pts, fill=color, outline=(0, 0, 0))
    return img
```

```python
    def visualize_dataset(loader: DataLoader | None, split_name: str):
        if loader is None or args.vis_mode == "none":
            return
        model.eval()
        import torchvision.transforms.functional as TF
        import numpy as np
        saved_images: List[Image.Image] = []
        count = 0
        max_count = args.vis_count if args.vis_mode == "sample" else float("inf")
        with torch.no_grad():
            for batch in loader:
                img_t = batch["image"]  # [B,3,H,W]
                tgt = batch["target_xy"]  # [B,2]
                logits = model(img_t.to(device))
                # 概率热力图用于可视化，稳定峰值
                bsz, _, H, W = logits.shape
                probs = torch.softmax(logits.view(bsz, -1), dim=1).view(bsz, 1, H, W)
                # 软argmax预测
                pred_xy = soft_argmax_from_logits(logits).cpu()
                for i in range(img_t.size(0)):
                    if count >= max_count:
                        break
                    # 反归一化到 [0,1] 便于显示
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    arr = img_t[i].cpu().float().numpy()
                    arr = (arr * std + mean).clip(0.0, 1.0)
                    pil_img = TF.to_pil_image(torch.from_numpy(arr))
                    # 使用概率图来叠加
                    hm = probs[i, 0].detach().cpu().float().numpy()
                    over = overlay_heatmap(pil_img, hm, alpha=0.5)
                    # 坐标钳制，避免越界导致标记不可见
                    h, w = img_t.size(-2), img_t.size(-1)
                    gx = float(max(0.0, min(w - 1.0, float(tgt[i, 0]))))
                    gy = float(max(0.0, min(h - 1.0, float(tgt[i, 1]))))
                    px = float(max(0.0, min(w - 1.0, float(pred_xy[i, 0]))))
                    py = float(max(0.0, min(h - 1.0, float(pred_xy[i, 1]))))
                    # 先GT十字，再Pred实心三角
                    over = draw_cross(over, (gx, gy), color=(0, 255, 0))
                    over = draw_triangle(over, (px, py), color=(255, 0, 0), size=10)
                    saved_images.append(over)
                    count += 1
                if count >= max_count:
                    break
        # save grid or per-sample
        if args.vis_mode == "sample":
            grid = make_grid(saved_images, cols=4)
            grid.save(os.path.join(vis_dir, f"{split_name}_vis.png"))
        else:  # all
            split_dir = os.path.join(vis_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            for i, img in enumerate(saved_images):
                img.save(os.path.join(split_dir, f"{i:05d}.png"))
```

```shellscript
cd /root/autodl-tmp/works/Seg-R0/seg-rl
python -m seg_rl.train \
  --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --height 240 --width 240 --batch_size 16 --epochs 1 --loss ce \
  --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pretrain \
  --vis_mode sample --vis_count 16
```

---

**User**

请实际apply你的上述改动

---

**Cursor**

I'll update the todo list for this change and start by modifying the triangle drawing utility to make the predicted marker clearly visible.

I'll implement the code edits now: first update `seg_rl/utils.py` draw_triangle, then adjust `visualize_dataset` in `seg_rl/train.py` to use probability heatmap and clamp coordinates.

```python
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



```

```python
from __future__ import annotations

import argparse
import os
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from PIL import Image

from .datasets import JsonlPointDataset, ImageSize, collate_fn
from .model import ModelConfig, PointHeatmapModel, argmax_from_logits, soft_argmax_from_logits
from .losses import ce_over_pixels, kl_to_gaussian_targets
from .utils import save_checkpoint
from .utils import draw_cross, draw_triangle, overlay_heatmap, make_grid

"""
Sample usage:
cd seg-rl;
python -m seg_rl.train \
  --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
  --height 240 --width 240 --batch_size 16 --epochs 10000 --loss ce --amp \
  --eval_thresh 5.0 --save_every 100 \
  --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pretrain \
  --vis_mode sample --vis_count 16

"""
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train heatmap classification point locator")
    p.add_argument("--data_jsonl", type=str, required=True, help="JSONL containing all samples; splits done in-script")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", type=str, choices=["ce", "kl"], default="ce")
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma for KL loss")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="./outputs/seg_rl")
    p.add_argument("--plots_dir", type=str, default=None, help="Directory to save loss/PCK plots; default under out_dir/plots")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval_thresh", type=float, default=5.0, help="PCK threshold in pixels")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio from the full dataset")
    p.add_argument("--test_ratio", type=float, default=0.0, help="Test split ratio from the full dataset")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument("--vis_mode", type=str, choices=["none", "sample", "all"], default="none", help="Visualization mode")
    p.add_argument("--vis_count", type=int, default=16, help="When vis_mode=sample, number of samples to visualize")
    p.add_argument("--vis_dir", type=str, default=None, help="Directory to save visualization images; default under out_dir/vis")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = args.plots_dir or os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    vis_dir = args.vis_dir or os.path.join(args.out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    image_size = ImageSize(height=args.height, width=args.width)
    # Load once to get length for splitting
    full_ds_for_len = JsonlPointDataset(args.data_jsonl, image_size=image_size, training=True)
    n = len(full_ds_for_len)
    assert 0.0 <= args.val_ratio < 1.0 and 0.0 <= args.test_ratio < 1.0 and args.val_ratio + args.test_ratio < 1.0, "Invalid split ratios"
    n_test = int(round(n * args.test_ratio))
    n_val = int(round(n * args.val_ratio))
    n_train = n - n_val - n_test
    g = torch.Generator()
    g.manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    # Build datasets per split to control augmentation flag
    train_base = JsonlPointDataset(args.data_jsonl, image_size=image_size, training=True)
    val_base = JsonlPointDataset(args.data_jsonl, image_size=image_size, training=False) if n_val > 0 else None
    test_base = JsonlPointDataset(args.data_jsonl, image_size=image_size, training=False) if n_test > 0 else None

    train_ds = Subset(train_base, idx_train)
    val_ds = Subset(val_base, idx_val) if val_base is not None else None
    test_ds = Subset(test_base, idx_test) if test_base is not None else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn) if test_ds is not None else None

    cfg = ModelConfig(pretrained=args.pretrained)
    model = PointHeatmapModel(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        from .utils import load_checkpoint
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler)
        start_epoch = ckpt.epoch
        global_step = ckpt.step

    history = {"train_loss": [], "val_pck": [], "test_pck": []}

    def evaluate(loader: DataLoader | None, split_name: str) -> float:
        if loader is None:
            return float("nan")
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(device, non_blocking=True)
                tgt = batch["target_xy"].to(device, non_blocking=True)
                logits = model(img)
                pred_xy = soft_argmax_from_logits(logits)
                d = torch.linalg.norm(pred_xy - tgt, dim=1)
                correct += (d <= args.eval_thresh).sum().item()
                total += d.numel()
        pck = correct / max(1, total)
        print(f"{split_name} PCK@{args.eval_thresh}: {pck:.4f}")
        return pck

    def plot_curves():
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            epochs_axis = np.arange(1, len(history["train_loss"]) + 1)
            # Loss
            plt.figure()
            plt.plot(epochs_axis, history["train_loss"], label="train_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, ls=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "loss.png"))
            plt.close()
            # PCK
            plt.figure()
            if any(not np.isnan(v) for v in history["val_pck"]):
                plt.plot(epochs_axis, history["val_pck"], label="val_pck")
            if any(not np.isnan(v) for v in history["test_pck"]):
                plt.plot(epochs_axis, history["test_pck"], label="test_pck")
            plt.xlabel("Epoch")
            plt.ylabel(f"PCK@{args.eval_thresh}")
            plt.legend()
            plt.grid(True, ls=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "pck.png"))
            plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")

    def visualize_dataset(loader: DataLoader | None, split_name: str):
        if loader is None or args.vis_mode == "none":
            return
        model.eval()
        import torchvision.transforms.functional as TF
        import numpy as np
        saved_images: List[Image.Image] = []
        count = 0
        max_count = args.vis_count if args.vis_mode == "sample" else float("inf")
        with torch.no_grad():
            for batch in loader:
                img_t = batch["image"]  # [B,3,H,W]
                tgt = batch["target_xy"]  # [B,2]
                logits = model(img_t.to(device))
                # probability heatmap for stable visualization
                bsz, _, H, W = logits.shape
                probs = torch.softmax(logits.view(bsz, -1), dim=1).view(bsz, 1, H, W)
                # compute soft-argmax pred on CPU for simplicity
                pred_xy = soft_argmax_from_logits(logits).cpu()
                for i in range(img_t.size(0)):
                    if count >= max_count:
                        break
                    # unnormalize to [0,1] for visualization (ImageNet stats)
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    arr = img_t[i].cpu().float().numpy()
                    arr = (arr * std + mean).clip(0.0, 1.0)
                    pil_img = TF.to_pil_image(torch.from_numpy(arr))
                    # overlay probability heatmap instead of raw logits
                    hm = probs[i, 0].detach().cpu().float().numpy()
                    over = overlay_heatmap(pil_img, hm, alpha=0.5)
                    # clamp coords to image bounds to ensure visibility
                    h, w = img_t.size(-2), img_t.size(-1)
                    gx = float(max(0.0, min(w - 1.0, float(tgt[i, 0]))))
                    gy = float(max(0.0, min(h - 1.0, float(tgt[i, 1]))))
                    px = float(max(0.0, min(w - 1.0, float(pred_xy[i, 0]))))
                    py = float(max(0.0, min(h - 1.0, float(pred_xy[i, 1]))))
                    # draw GT and Pred (GT cross in green, Pred filled triangle in red)
                    over = draw_cross(over, (gx, gy), color=(0, 255, 0))
                    over = draw_triangle(over, (px, py), color=(255, 0, 0), size=10)
                    saved_images.append(over)
                    count += 1
                if count >= max_count:
                    break
        # save grid or per-sample
        if args.vis_mode == "sample":
            grid = make_grid(saved_images, cols=4)
            grid.save(os.path.join(vis_dir, f"{split_name}_vis.png"))
        else:  # all
            split_dir = os.path.join(vis_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            for i, img in enumerate(saved_images):
                img.save(os.path.join(split_dir, f"{i:05d}.png"))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            tgt = batch["target_xy"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(img)
                if args.loss == "ce":
                    loss = ce_over_pixels(logits, tgt)
                else:
                    loss = kl_to_gaussian_targets(logits, tgt, sigma=args.sigma)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

        avg_loss = running_loss / max(1, len(train_loader))
        history["train_loss"].append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - train loss: {avg_loss:.4f}")

        # Validation
        val_pck = evaluate(val_loader, "val") if val_loader is not None else float("nan")
        history["val_pck"].append(val_pck)
        test_pck = evaluate(test_loader, "test") if test_loader is not None else float("nan")
        history["test_pck"].append(test_pck)

        # plots
        plot_curves()

        # visualizations
        if args.vis_mode != "none":
            visualize_dataset(val_loader or train_loader, split_name="val_or_train")

        # Save
        if ((epoch + 1) % max(1, args.save_every)) == 0:
            save_path = os.path.join(args.out_dir, f"model_epoch_{epoch+1}.pt")
            save_checkpoint(save_path, model, optimizer, scaler, epoch=epoch+1, step=global_step)


if __name__ == "__main__":
    main()



```

已完成修改并通过静态检查。

- 我将 `seg_rl/utils.py` 的 `draw_triangle` 改为实心填充红色、黑色描边，默认尺寸更大，提升可见性。
- 我将 `seg_rl/train.py` 的可视化改为叠加概率热力图（对 H×W 做 softmax），并对 GT/Pred 坐标做边界钳制，再绘制绿色十字与红色实心三角。

你可以直接重跑一次可视化来生成新的图像：
```bash
cd /root/autodl-tmp/works/Seg-R0/seg-rl
python -m seg_rl.train \
  --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --height 240 --width 240 --batch_size 16 --epochs 1 --loss ce \
  --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pretrain \
  --vis_mode sample --vis_count 16
```
查看 `outputs/seg_r1_md/Task01_BrainTumour/pretrain/vis/val_or_train_vis.png` 应能看到清晰的红色三角预测标记，且热力图与原图对齐更直观。

```shellscript
cd /root/autodl-tmp/works/Seg-R0/seg-rl
python -m seg_rl.train \
  --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --height 240 --width 240 --batch_size 16 --epochs 1 --loss ce \
  --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pretrain \
  --vis_mode sample --vis_count 16
```

---

**User**

好。为了能更清晰地看到效果，请在可视化时把原图像以灰度显示，叠加的热力图保持彩色

---

**Cursor**

I'll update the visualization to render the base image in grayscale while keeping the heatmap colored by passing a grayscale-converted image into the overlay function. Then I'll run a quick lint check.

```python
from __future__ import annotations

import argparse
import os
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from PIL import Image

from .datasets import JsonlPointDataset, ImageSize, collate_fn
from .model import ModelConfig, PointHeatmapModel, argmax_from_logits, soft_argmax_from_logits
from .losses import ce_over_pixels, kl_to_gaussian_targets
from .utils import save_checkpoint
from .utils import draw_cross, draw_triangle, overlay_heatmap, make_grid

"""
Sample usage:
cd seg-rl;
python -m seg_rl.train \
  --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
  --height 240 --width 240 --batch_size 16 --epochs 10000 --loss ce --amp \
  --eval_thresh 5.0 --save_every 100 \
  --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pretrain \
  --vis_mode sample --vis_count 16

"""
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train heatmap classification point locator")
    p.add_argument("--data_jsonl", type=str, required=True, help="JSONL containing all samples; splits done in-script")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", type=str, choices=["ce", "kl"], default="ce")
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma for KL loss")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="./outputs/seg_rl")
    p.add_argument("--plots_dir", type=str, default=None, help="Directory to save loss/PCK plots; default under out_dir/plots")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval_thresh", type=float, default=5.0, help="PCK threshold in pixels")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio from the full dataset")
    p.add_argument("--test_ratio", type=float, default=0.0, help="Test split ratio from the full dataset")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument("--vis_mode", type=str, choices=["none", "sample", "all"], default="none", help="Visualization mode")
    p.add_argument("--vis_count", type=int, default=16, help="When vis_mode=sample, number of samples to visualize")
    p.add_argument("--vis_dir", type=str, default=None, help="Directory to save visualization images; default under out_dir/vis")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = args.plots_dir or os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    vis_dir = args.vis_dir or os.path.join(args.out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    image_size = ImageSize(height=args.height, width=args.width)
    # Load once to get length for splitting
    full_ds_for_len = JsonlPointDataset(args.data_jsonl, image_size=image_size, training=True)
    n = len(full_ds_for_len)
    assert 0.0 <= args.val_ratio < 1.0 and 0.0 <= args.test_ratio < 1.0 and args.val_ratio + args.test_ratio < 1.0, "Invalid split ratios"
    n_test = int(round(n * args.test_ratio))
    n_val = int(round(n * args.val_ratio))
    n_train = n - n_val - n_test
    g = torch.Generator()
    g.manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    # Build datasets per split to control augmentation flag
    train_base = JsonlPointDataset(args.data_jsonl, image_size=image_size, training=True)
    val_base = JsonlPointDataset(args.data_jsonl, image_size=image_size, training=False) if n_val > 0 else None
    test_base = JsonlPointDataset(args.data_jsonl, image_size=image_size, training=False) if n_test > 0 else None

    train_ds = Subset(train_base, idx_train)
    val_ds = Subset(val_base, idx_val) if val_base is not None else None
    test_ds = Subset(test_base, idx_test) if test_base is not None else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn) if test_ds is not None else None

    cfg = ModelConfig(pretrained=args.pretrained)
    model = PointHeatmapModel(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        from .utils import load_checkpoint
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler)
        start_epoch = ckpt.epoch
        global_step = ckpt.step

    history = {"train_loss": [], "val_pck": [], "test_pck": []}

    def evaluate(loader: DataLoader | None, split_name: str) -> float:
        if loader is None:
            return float("nan")
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(device, non_blocking=True)
                tgt = batch["target_xy"].to(device, non_blocking=True)
                logits = model(img)
                pred_xy = soft_argmax_from_logits(logits)
                d = torch.linalg.norm(pred_xy - tgt, dim=1)
                correct += (d <= args.eval_thresh).sum().item()
                total += d.numel()
        pck = correct / max(1, total)
        print(f"{split_name} PCK@{args.eval_thresh}: {pck:.4f}")
        return pck

    def plot_curves():
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            epochs_axis = np.arange(1, len(history["train_loss"]) + 1)
            # Loss
            plt.figure()
            plt.plot(epochs_axis, history["train_loss"], label="train_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, ls=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "loss.png"))
            plt.close()
            # PCK
            plt.figure()
            if any(not np.isnan(v) for v in history["val_pck"]):
                plt.plot(epochs_axis, history["val_pck"], label="val_pck")
            if any(not np.isnan(v) for v in history["test_pck"]):
                plt.plot(epochs_axis, history["test_pck"], label="test_pck")
            plt.xlabel("Epoch")
            plt.ylabel(f"PCK@{args.eval_thresh}")
            plt.legend()
            plt.grid(True, ls=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "pck.png"))
            plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")

    def visualize_dataset(loader: DataLoader | None, split_name: str):
        if loader is None or args.vis_mode == "none":
            return
        model.eval()
        import torchvision.transforms.functional as TF
        import numpy as np
        saved_images: List[Image.Image] = []
        count = 0
        max_count = args.vis_count if args.vis_mode == "sample" else float("inf")
        with torch.no_grad():
            for batch in loader:
                img_t = batch["image"]  # [B,3,H,W]
                tgt = batch["target_xy"]  # [B,2]
                logits = model(img_t.to(device))
                # probability heatmap for stable visualization
                bsz, _, H, W = logits.shape
                probs = torch.softmax(logits.view(bsz, -1), dim=1).view(bsz, 1, H, W)
                # compute soft-argmax pred on CPU for simplicity
                pred_xy = soft_argmax_from_logits(logits).cpu()
                for i in range(img_t.size(0)):
                    if count >= max_count:
                        break
                    # unnormalize to [0,1] for visualization (ImageNet stats)
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    arr = img_t[i].cpu().float().numpy()
                    arr = (arr * std + mean).clip(0.0, 1.0)
                    pil_img = TF.to_pil_image(torch.from_numpy(arr))
                    # overlay probability heatmap instead of raw logits
                    hm = probs[i, 0].detach().cpu().float().numpy()
                    # show base image in grayscale while keeping heatmap colored
                    over = overlay_heatmap(pil_img.convert("L"), hm, alpha=0.5)
                    # clamp coords to image bounds to ensure visibility
                    h, w = img_t.size(-2), img_t.size(-1)
                    gx = float(max(0.0, min(w - 1.0, float(tgt[i, 0]))))
                    gy = float(max(0.0, min(h - 1.0, float(tgt[i, 1]))))
                    px = float(max(0.0, min(w - 1.0, float(pred_xy[i, 0]))))
                    py = float(max(0.0, min(h - 1.0, float(pred_xy[i, 1]))))
                    # draw GT and Pred (GT cross in green, Pred filled triangle in red)
                    over = draw_cross(over, (gx, gy), color=(0, 255, 0))
                    over = draw_triangle(over, (px, py), color=(255, 0, 0), size=10)
                    saved_images.append(over)
                    count += 1
                if count >= max_count:
                    break
        # save grid or per-sample
        if args.vis_mode == "sample":
            grid = make_grid(saved_images, cols=4)
            grid.save(os.path.join(vis_dir, f"{split_name}_vis.png"))
        else:  # all
            split_dir = os.path.join(vis_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            for i, img in enumerate(saved_images):
                img.save(os.path.join(split_dir, f"{i:05d}.png"))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            tgt = batch["target_xy"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(img)
                if args.loss == "ce":
                    loss = ce_over_pixels(logits, tgt)
                else:
                    loss = kl_to_gaussian_targets(logits, tgt, sigma=args.sigma)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

        avg_loss = running_loss / max(1, len(train_loader))
        history["train_loss"].append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - train loss: {avg_loss:.4f}")

        # Validation
        val_pck = evaluate(val_loader, "val") if val_loader is not None else float("nan")
        history["val_pck"].append(val_pck)
        test_pck = evaluate(test_loader, "test") if test_loader is not None else float("nan")
        history["test_pck"].append(test_pck)

        # plots
        plot_curves()

        # visualizations
        if args.vis_mode != "none":
            visualize_dataset(val_loader or train_loader, split_name="val_or_train")

        # Save
        if ((epoch + 1) % max(1, args.save_every)) == 0:
            save_path = os.path.join(args.out_dir, f"model_epoch_{epoch+1}.pt")
            save_checkpoint(save_path, model, optimizer, scaler, epoch=epoch+1, step=global_step)


if __name__ == "__main__":
    main()



```

已更新可视化：底图转为灰度后再叠加彩色热力图，标记绘制逻辑保持不变。生成的新图会更清楚地突出热力图颜色对比。

可直接重跑一次可视化生成：
```bash
cd /root/autodl-tmp/works/Seg-R0/seg-rl
python -m seg_rl.train \
  --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --height 240 --width 240 --batch_size 16 --epochs 1 --loss ce \
  --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pretrain \
  --vis_mode sample --vis_count 16
```
查看 `.../pretrain/vis/val_or_train_vis.png`。

```shellscript
cd /root/autodl-tmp/works/Seg-R0/seg-rl
python -m seg_rl.train \
  --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --height 240 --width 240 --batch_size 16 --epochs 1 --loss ce \
  --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pretrain \
  --vis_mode sample --vis_count 16
```

---

**User**

请看生成的结果val_or_train_vis.png，这叠加的彩色热力图明显太小了吧，是不是没有正确做好尺寸缩放和对齐？请再次检查。并且，三角形放在最上层容易遮挡住绿色十字不容易看清，且位置指示不够精确。请把红色三角改成斜45度的红色十字，这样两个十字都能被看清且指示精准。

---

**Cursor**

I'll add a robust size check to upsample logits to image size before computing probabilities and soft-argmax, ensuring heatmap aligns perfectly. I'll also implement a new `draw_diagonal_cross` utility (45° red cross with black stroke) and switch visualization to use it instead of the triangle so both markers are clear and precise.

```python
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



```

```python
from __future__ import annotations

import argparse
import os
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from PIL import Image

from .datasets import JsonlPointDataset, ImageSize, collate_fn
from .model import ModelConfig, PointHeatmapModel, argmax_from_logits, soft_argmax_from_logits
from .losses import ce_over_pixels, kl_to_gaussian_targets
from .utils import save_checkpoint
from .utils import draw_cross, draw_triangle, draw_diagonal_cross, overlay_heatmap, make_grid

"""
Sample usage:
cd seg-rl;
python -m seg_rl.train \
  --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
  --height 240 --width 240 --batch_size 16 --epochs 10000 --loss ce --amp \
  --eval_thresh 5.0 --save_every 100 \
  --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pretrain \
  --vis_mode sample --vis_count 16

"""
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train heatmap classification point locator")
    p.add_argument("--data_jsonl", type=str, required=True, help="JSONL containing all samples; splits done in-script")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", type=str, choices=["ce", "kl"], default="ce")
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma for KL loss")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="./outputs/seg_rl")
    p.add_argument("--plots_dir", type=str, default=None, help="Directory to save loss/PCK plots; default under out_dir/plots")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval_thresh", type=float, default=5.0, help="PCK threshold in pixels")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio from the full dataset")
    p.add_argument("--test_ratio", type=float, default=0.0, help="Test split ratio from the full dataset")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument("--vis_mode", type=str, choices=["none", "sample", "all"], default="none", help="Visualization mode")
    p.add_argument("--vis_count", type=int, default=16, help="When vis_mode=sample, number of samples to visualize")
    p.add_argument("--vis_dir", type=str, default=None, help="Directory to save visualization images; default under out_dir/vis")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = args.plots_dir or os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    vis_dir = args.vis_dir or os.path.join(args.out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    image_size = ImageSize(height=args.height, width=args.width)
    # Load once to get length for splitting
    full_ds_for_len = JsonlPointDataset(args.data_jsonl, image_size=image_size, training=True)
    n = len(full_ds_for_len)
    assert 0.0 <= args.val_ratio < 1.0 and 0.0 <= args.test_ratio < 1.0 and args.val_ratio + args.test_ratio < 1.0, "Invalid split ratios"
    n_test = int(round(n * args.test_ratio))
    n_val = int(round(n * args.val_ratio))
    n_train = n - n_val - n_test
    g = torch.Generator()
    g.manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    # Build datasets per split to control augmentation flag
    train_base = JsonlPointDataset(args.data_jsonl, image_size=image_size, training=True)
    val_base = JsonlPointDataset(args.data_jsonl, image_size=image_size, training=False) if n_val > 0 else None
    test_base = JsonlPointDataset(args.data_jsonl, image_size=image_size, training=False) if n_test > 0 else None

    train_ds = Subset(train_base, idx_train)
    val_ds = Subset(val_base, idx_val) if val_base is not None else None
    test_ds = Subset(test_base, idx_test) if test_base is not None else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn) if test_ds is not None else None

    cfg = ModelConfig(pretrained=args.pretrained)
    model = PointHeatmapModel(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        from .utils import load_checkpoint
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler)
        start_epoch = ckpt.epoch
        global_step = ckpt.step

    history = {"train_loss": [], "val_pck": [], "test_pck": []}

    def evaluate(loader: DataLoader | None, split_name: str) -> float:
        if loader is None:
            return float("nan")
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(device, non_blocking=True)
                tgt = batch["target_xy"].to(device, non_blocking=True)
                logits = model(img)
                pred_xy = soft_argmax_from_logits(logits)
                d = torch.linalg.norm(pred_xy - tgt, dim=1)
                correct += (d <= args.eval_thresh).sum().item()
                total += d.numel()
        pck = correct / max(1, total)
        print(f"{split_name} PCK@{args.eval_thresh}: {pck:.4f}")
        return pck

    def plot_curves():
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            epochs_axis = np.arange(1, len(history["train_loss"]) + 1)
            # Loss
            plt.figure()
            plt.plot(epochs_axis, history["train_loss"], label="train_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, ls=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "loss.png"))
            plt.close()
            # PCK
            plt.figure()
            if any(not np.isnan(v) for v in history["val_pck"]):
                plt.plot(epochs_axis, history["val_pck"], label="val_pck")
            if any(not np.isnan(v) for v in history["test_pck"]):
                plt.plot(epochs_axis, history["test_pck"], label="test_pck")
            plt.xlabel("Epoch")
            plt.ylabel(f"PCK@{args.eval_thresh}")
            plt.legend()
            plt.grid(True, ls=":", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "pck.png"))
            plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")

    def visualize_dataset(loader: DataLoader | None, split_name: str):
        if loader is None or args.vis_mode == "none":
            return
        model.eval()
        import torchvision.transforms.functional as TF
        import numpy as np
        saved_images: List[Image.Image] = []
        count = 0
        max_count = args.vis_count if args.vis_mode == "sample" else float("inf")
        with torch.no_grad():
            for batch in loader:
                img_t = batch["image"]  # [B,3,H,W]
                tgt = batch["target_xy"]  # [B,2]
                logits = model(img_t.to(device))
                # ensure logits match image resolution for precise overlay
                _, _, H_img, W_img = img_t.shape
                if logits.shape[-2] != H_img or logits.shape[-1] != W_img:
                    logits = torch.nn.functional.interpolate(logits, size=(H_img, W_img), mode="bilinear", align_corners=False)
                # probability heatmap for stable visualization
                bsz, _, H, W = logits.shape
                probs = torch.softmax(logits.view(bsz, -1), dim=1).view(bsz, 1, H, W)
                # compute soft-argmax pred on CPU for simplicity
                pred_xy = soft_argmax_from_logits(logits).cpu()
                for i in range(img_t.size(0)):
                    if count >= max_count:
                        break
                    # unnormalize to [0,1] for visualization (ImageNet stats)
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    arr = img_t[i].cpu().float().numpy()
                    arr = (arr * std + mean).clip(0.0, 1.0)
                    pil_img = TF.to_pil_image(torch.from_numpy(arr))
                    # overlay probability heatmap instead of raw logits
                    hm = probs[i, 0].detach().cpu().float().numpy()
                    # show base image in grayscale while keeping heatmap colored
                    over = overlay_heatmap(pil_img.convert("L"), hm, alpha=0.5)
                    # clamp coords to image bounds to ensure visibility
                    h, w = img_t.size(-2), img_t.size(-1)
                    gx = float(max(0.0, min(w - 1.0, float(tgt[i, 0]))))
                    gy = float(max(0.0, min(h - 1.0, float(tgt[i, 1]))))
                    px = float(max(0.0, min(w - 1.0, float(pred_xy[i, 0]))))
                    py = float(max(0.0, min(h - 1.0, float(pred_xy[i, 1]))))
                    # draw GT and Pred (GT + Pred both crosses; Pred is 45-degree red cross)
                    over = draw_cross(over, (gx, gy), color=(0, 255, 0))
                    over = draw_diagonal_cross(over, (px, py), color=(255, 0, 0), size=10)
                    saved_images.append(over)
                    count += 1
                if count >= max_count:
                    break
        # save grid or per-sample
        if args.vis_mode == "sample":
            grid = make_grid(saved_images, cols=4)
            grid.save(os.path.join(vis_dir, f"{split_name}_vis.png"))
        else:  # all
            split_dir = os.path.join(vis_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            for i, img in enumerate(saved_images):
                img.save(os.path.join(split_dir, f"{i:05d}.png"))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            tgt = batch["target_xy"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(img)
                if args.loss == "ce":
                    loss = ce_over_pixels(logits, tgt)
                else:
                    loss = kl_to_gaussian_targets(logits, tgt, sigma=args.sigma)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

        avg_loss = running_loss / max(1, len(train_loader))
        history["train_loss"].append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - train loss: {avg_loss:.4f}")

        # Validation
        val_pck = evaluate(val_loader, "val") if val_loader is not None else float("nan")
        history["val_pck"].append(val_pck)
        test_pck = evaluate(test_loader, "test") if test_loader is not None else float("nan")
        history["test_pck"].append(test_pck)

        # plots
        plot_curves()

        # visualizations
        if args.vis_mode != "none":
            visualize_dataset(val_loader or train_loader, split_name="val_or_train")

        # Save
        if ((epoch + 1) % max(1, args.save_every)) == 0:
            save_path = os.path.join(args.out_dir, f"model_epoch_{epoch+1}.pt")
            save_checkpoint(save_path, model, optimizer, scaler, epoch=epoch+1, step=global_step)


if __name__ == "__main__":
    main()



```

已修复两点并增强标记可读性：
- 确保热力图尺寸对齐：在可视化时若 `logits` 尺寸与图像不一致，会先插值到与输入图像完全一致的 H×W，再做 softmax，可视化叠加不再“变小”或错位。
- 预测标记改为45°红色十字：新增 `draw_diagonal_cross`，在标记下方先绘制黑色加粗底线以保证任何底图上清晰可见；替换原红色三角。绿色正十字与红色斜十字可同时辨识且中心更精准。

重跑一次可视化生成检查：
```bash
cd /root/autodl-tmp/works/Seg-R0/seg-rl
python -m seg_rl.train \
  --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --height 240 --width 240 --batch_size 16 --epochs 1 --loss ce \
  --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pretrain \
  --vis_mode sample --vis_count 16
```
查看 `.../pretrain/vis/val_or_train_vis.png`。

```shellscript
cd /root/autodl-tmp/works/Seg-R0/seg-rl
python -m seg_rl.train \
  --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/points_canonical.jsonl \
  --height 240 --width 240 --batch_size 16 --epochs 1 --loss ce \
  --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/pretrain \
  --vis_mode sample --vis_count 16
```

