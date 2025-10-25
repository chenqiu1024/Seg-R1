# 理解预测下一提示点的逻辑
_Exported on 10/25/2025 at 16:55:35 PDT from Cursor (1.7.54)_

---

**User**

阅读历史对话概要以获取上下文，然后继续解答我的问题：在seg-rl/heatmap/predict_next_point_from_model.py的_predict_point函数里，我发现预测下一提示点坐标似乎是在用整个logits对全部坐标做加权平均而不是选取最大softmax值的点得到的（soft_argmax_from_logits函数，213~214行）。我的理解正确吗？如果是，请解释为什么要这样做而不是选取softmax最大者？

---

**Cursor**

Let me open the relevant files to confirm how `_predict_point` calls `soft_argmax_from_logits` and how that function computes the coordinate. I'll pull both `predict_next_point_from_model.py` and `model.py` in parallel.

I checked `_predict_point` and `soft_argmax_from_logits` in your code to confirm the behavior; citing the relevant lines below.

- Yes, your understanding is correct. `_predict_point` computes the coordinate via soft-argmax, i.e., the expected coordinate under the softmax distribution over all pixels, not the pixel with maximum probability.

Code references:
```187:196:/Users/donchiu/Works/Seg-R1/seg-rl/heatmap/predict_next_point_from_model.py
def _predict_point(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        p = F.softmax(logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob
```

```207:215:/Users/donchiu/Works/Seg-R1/seg-rl/heatmap/model.py
def soft_argmax_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Soft-argmax to get sub-pixel expected coordinates [B,1,H,W] -> [B,2] (x,y)."""
    b, _, h, w = logits.shape
    p = F.softmax(logits.view(b, -1) / max(temperature, 1e-6), dim=1).view(b, 1, h, w)
    xs = torch.linspace(0, w - 1, w, device=logits.device).view(1, 1, 1, w)
    ys = torch.linspace(0, h - 1, h, device=logits.device).view(1, 1, h, 1)
    x_exp = (p * xs).sum(dim=(2, 3))
    y_exp = (p * ys).sum(dim=(2, 3))
    return torch.cat([x_exp, y_exp], dim=1)
```

Why do this instead of hard argmax?
- • Sub-pixel precision: Expected coordinates yield continuous (sub-pixel) points, often improving downstream SAM2 behavior versus integer argmax.
- • Stability/smoothness: Hard argmax is sensitive to small logit noise; soft-argmax varies smoothly and reduces jitter across steps.
- • Matches training targets: The model is trained against soft/gaussian heatmap targets (KL/MSE/CE). The expected value aligns with those soft distributions.
- • Temperature control: `--tau` lets you tune between near-argmax (small tau) and more averaging (larger tau) without code changes.

Caveat and options:
- For multi-modal heatmaps, soft-argmax can land between peaks. If you prefer a single mode:
  - Use hard argmax: replace `soft_argmax_from_logits(...)` with `argmax_from_logits(...)` (available in the same file).
  - Or sample a discrete pixel (and sub-pixel) using the hierarchical samplers (`sample_cell_and_offset` / `sample_joint_label_cell_offset`) for stochastic behavior in RL.

Minimal references for alternatives:
```198:205:/Users/donchiu/Works/Seg-R1/seg-rl/heatmap/model.py
def argmax_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Get integer pixel coordinates from logits [B,1,H,W] -> [B,2] (x,y)."""
    b, _, h, w = logits.shape
    idx = logits.view(b, -1).argmax(dim=1)
    y = (idx // w).float()
    x = (idx % w).float()
    return torch.stack([x, y], dim=1)
```

- If you just want harder selections without editing code, try a smaller `--tau` (approaches argmax as tau→0, with numerical clamp in code).

- I verified the implementation and confirmed the soft-argmax choice is intentional for precision and stability; if you want, I can draft the exact one-line edit to switch to hard argmax or to the hierarchical sampler.

- Answered your question and pointed to exact lines; no changes applied.

```python
def _predict_point(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        p = F.softmax(logits.view(b, -1) / max(float(tau), 1e-6), dim=1).view(b, 1, H, W)
        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob
```

```python
def soft_argmax_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Soft-argmax to get sub-pixel expected coordinates [B,1,H,W] -> [B,2] (x,y)."""
    b, _, h, w = logits.shape
    p = F.softmax(logits.view(b, -1) / max(temperature, 1e-6), dim=1).view(b, 1, h, w)
    xs = torch.linspace(0, w - 1, w, device=logits.device).view(1, 1, 1, w)
    ys = torch.linspace(0, h - 1, h, device=logits.device).view(1, 1, h, 1)
    x_exp = (p * xs).sum(dim=(2, 3))
    y_exp = (p * ys).sum(dim=(2, 3))
    return torch.cat([x_exp, y_exp], dim=1)
```

```python
def argmax_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Get integer pixel coordinates from logits [B,1,H,W] -> [B,2] (x,y)."""
    b, _, h, w = logits.shape
    idx = logits.view(b, -1).argmax(dim=1)
    y = (idx // w).float()
    x = (idx % w).float()
    return torch.stack([x, y], dim=1)
```

---

**User**

以后请用中文回答。我还有个疑问：对于一个sam提示点，除了需要坐标信息以外，还包括是前景点（1）还是背景点（0）。那么按我的理解，应该不止需要一张概率图，而是相应于前景点有一个概率图，相应于背景点有一张概率图，最后通过看两张概率图里最大值的点在哪里而同时输出坐标位置和前/背景标签。而model.py中的设计是只用一张概率图预测坐标，而通过另一个头来生成标签。这样的设计是否合理？你觉得我的设想会不会更合理？请不要拍马屁，而是通过机器学习领域的经验和知识来客观作评价。

---

**Cursor**

### 结论（先回答你的问题）
- 你的理解是对的：当前实现用一张像素分布图预测坐标，另用一个分类头决定前/背景标签，而不是“前景图+背景图各一张再取总体最大”。
- 这种设计是合理的，但在某些场景下你提出的“每类一张热力图”的方案更有表达力，代价是参数更大、监督与标定更复杂，训练更容易不稳。是否更合理取决于你的数据与目标。

代码证据：
```171:185:/Users/donchiu/Works/Seg-R1/seg-rl/heatmap/model.py
def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
    if getattr(self, "_is_unet", False):
        logits = self.unet(x)
        feat = getattr(self.unet, "last_dec", None)
        if feat is None:
            feat = x
        label_logits = self.label_head_unet(feat)
        return logits, label_logits
```

```44:57:/Users/donchiu/Works/Seg-R1/seg-rl/heatmap/model.py
class LabelHead(nn.Module):
    ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.fc(x).flatten(1)
```

```328:366:/Users/donchiu/Works/Seg-R1/seg-rl/heatmap/model.py
def sample_joint_label_cell_offset(...):
    label_dist = torch.distributions.Categorical(logits=label_logits / T_label)
    label_idx = label_dist.sample()
    ...
    pixel = sample_cell_and_offset(logits, stride=stride, temperature=temperature_pixel)
    ...
    out["log_prob"] = logp_label + logp_pixel
```

### 客观对比：单热力图+标签头 vs 每类一张热力图

- 单热力图 + 标签头（当前方案）
  - 优点
    - **稳定与光滑**：坐标用软argmax可得子像素、对噪声不敏感；标签用全局特征池化，数值更稳。
    - **参数/数据友好**：更少参数与监督约束，易训练、对小数据更鲁棒。
    - **目标对齐**：若热力图训练为“哪里点击最有用（与标签无关）”的软目标，高斯/KL/MSE更契合期望值坐标。
  - 局限
    - **独立性假设**：当前 joint 采样里标签与像素基本独立（logp 相加），无法表达“前景点应偏A区、背景点应偏B区”的强耦合偏好。

- 每类一张热力图（你的设想）
  - 优点
    - **表达力更强**：可直接建模 p(x,y | label)，适合前/背景“最佳点击区域”分布差异显著的任务。
    - **推理自然**：跨两张图取总体最大即可同时得到坐标与标签（相当于在 2×H×W 上做一次 argmax）。
  - 代价/风险
    - **监督更复杂**：需要构造两通道目标（对应标签通道是高斯，另一通道应压低）；损失与温度标定更敏感。
    - **训练耦合与竞争**：两张图互相“抢峰值”，更容易出现校准偏差或不稳；对数据量和正负样本平衡更敏感。
    - **推理漂移**：多模态场景下，若两通道峰值接近，选择会更跳变，需要额外温度/置信度门控。

### 什么时候该换成“两张图”？
- 数据显示：最佳点击区域对标签高度依赖（如常见“内点补漏、外点抑伪”且空间分布差异明显）。
- 你要做“联合策略”且希望显式刻画 p(pixel | label)（当前实现是近似独立）。
- 你有足够数据与评测，能承担更复杂的标定与早停策略。

### 折中与落地建议（从易到难）
- 保持现架构，训练与推理按现有流程，但在 RL 阶段使用 `sample_joint_label_cell_offset` 做联合采样；先验证收益。
- 轻改版“条件热力图”：把热力图头改成 2 通道 H∈R^{2×H×W}，训练时对真标签通道用高斯软目标、另一通道用较低常数或小高斯；推理取 2×H×W 全局 argmax。若不想改损失太多，可先只改推理路径做消融。
- 进阶版“条件化”：
  - 用预测的 `label_logits` 对热力图做 FiLM/门控，得到条件化像素分布（避免完全复制两张头部）。
  - 或直接建模联合分布 p(label, cell, subpixel)（你已有分解框架），把像素头做成“按标签条件化”的两套权重或少量调制参数。

简短结论
- 当前“单热力图+标签头”是工程上稳健、易训的合理基线；在很多数据规模与难度下表现往往更稳。
- 你的“两张图”方案更强表达、更贴近联合策略，但需要更细的标定与监督，收益需通过消融验证。
- 建议先做最小代价的条件化/两通道消融，再决定是否全面迁移到类条件热力图。

- 我已客观比较两者的利弊；如果你需要，我可以给出最小改动的两通道推理/训练改动草案用于快速消融。

```python
def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
    if getattr(self, "_is_unet", False):
        logits = self.unet(x)
        feat = getattr(self.unet, "last_dec", None)
        if feat is None:
            feat = x
        label_logits = self.label_head_unet(feat)
        return logits, label_logits
```

```python
class LabelHead(nn.Module):
    ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.fc(x).flatten(1)
```

```python
def sample_joint_label_cell_offset(...):
    label_dist = torch.distributions.Categorical(logits=label_logits / T_label)
    label_idx = label_dist.sample()
    ...
    pixel = sample_cell_and_offset(logits, stride=stride, temperature=temperature_pixel)
    ...
    out["log_prob"] = logp_label + logp_pixel
```

---

**User**

好的，请先实施你的最小改动，并且可以通过命令行参数来区分，使得可以在当前结构和修改结构之前切换，默认时采用当前结构不变。

---

**Cursor**

I'll add a CLI flag to switch coordinate/label selection strategies and implement a minimal "two-map" heuristic in `predict_next_point_from_model.py`, defaulting to the current behavior. Then I'll update the calls accordingly.

```python
#!/usr/bin/env python3

"""
Predict next SAM prompt point (position + binary label) using a trained heatmap model.

This script mirrors seg-rl/annotator/gen_point_jsonl_from_masks.py, but replaces the
EDT-based next-point computation with a model inference that takes only:
  - the RGB image, and
  - the current predicted mask (or zero mask at step 0)
and outputs the next prompt point and its label.

Modes:
  1) Initial generation (write new JSON array):
     - For each image under --images_dir, predict the first prompt (i=0) with a zero mask.
     - Write a JSON array with records like:
         {"image": "/abs/path.jpg", "sam_masks_dir": "/abs/sam_dir", "points": [[x0,y0]], "labels": [l0]}

  2) Append mode (update existing JSON array):
     - Read --appendto_json as a JSON array, for each record take current length k=len(points),
       load the latest predicted mask at {sam_dir}/{stem}/{k-1}.png (k>0; else zero), and predict
       the next point (xk, yk, lk). Append to arrays and write back in-place.

  3) Debug visualization (no file write except images):
     - Read --debug_json (JSON array) that already contains points+labels and a sam_masks_dir.
     - For each step i, load prev and cur predicted masks (i-1 and i; prev is zero for i=0) and
       call the same 3-panel visualization function used by the annotator to render panels with
       the provided point, saving to --debug_output_dir/{stem}/dbg_{i}.png.
     - Ground-truth masks are NOT used anywhere in this script (except indirectly if you supply
       them in your own JSON for reference; they are not required).

Examples:
  # Initial generation
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/heatmap_train-251001-optimized/model_epoch_185.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred_points-sft_e185-251018.jsonl

  python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred_points-e160-251019.jsonl

  # Append mode
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --appendto_json datasets/seg_r1_md/Task01_BrainTumour/pred_points-251001.jsonl

  python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --appendto_json outputs/braintumour/pred_points-e160-251019.jsonl

  # Debug visualization only
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --debug_json datasets/seg_r1_md/Task01_BrainTumour/pred_points-init.jsonl \
    --debug_output_dir outputs/braintumour/dbg_model_pred

Notes:
  - The model expects 4-channel input (RGB normalized + Gray normalized mask). This script
    normalizes RGB with ImageNet mean/std and mask with mean=0.5, std=0.5.
  - Device selection prefers CUDA, then MPS (Apple Silicon), otherwise CPU. AMP is only used on CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

try:
    # package context
    from .model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

import importlib.util as _importlib_util

def _import_render_helper() -> Optional[Callable]:
    """Dynamically import _render_diff_panels_with_point from annotator script by file path.

    This avoids package-name issues due to hyphen in 'seg-rl'.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # seg-rl
        target = os.path.join(base_dir, "annotator", "gen_point_jsonl_from_masks.py")
        if not os.path.isfile(target):
            return None
        spec = _importlib_util.spec_from_file_location("gen_point_jsonl_from_masks", target)
        if spec is None or spec.loader is None:
            return None
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        func = getattr(module, "_render_diff_panels_with_point", None)
        if callable(func):
            return func  # type: ignore[return-value]
    except Exception:
        return None
    return None

_render_diff_panels_with_point: Optional[Callable] = _import_render_helper()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict next SAM prompt point via trained heatmap model")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_json", type=str, help="Write a new JSON array with first predicted points")
    group.add_argument("--appendto_json", type=str, help="Append next predicted point to an existing JSON array")
    group.add_argument("--debug_json", type=str, help="Debug visualization only: read JSON array and render panels")

    p.add_argument("--model_path", type=str, required=False, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--images_dir", type=str, default=None, help="Directory of input images (initial mode)")
    p.add_argument("--masks_dir", type=str, default=None, help="Ground-truth masks directory (initial mode only, to fill 'gt_mask' field)")
    p.add_argument("--sam_dir", type=str, default=None, help="Directory of predicted SAM masks per step: {sam_dir}/{stem}/{k}.png (initial/debug mode)")
    # height/width: 0 means use native image size; if >0, will resize for inference
    p.add_argument("--height", type=int, default=0, help="0 = use native image height")
    p.add_argument("--width", type=int, default=0, help="0 = use native image width")
    p.add_argument("--tau", type=float, default=1.0, help="Softmax temperature for heatmap -> probability")
    # Coordinate/label selection strategy
    p.add_argument(
        "--coord_label_mode",
        type=str,
        choices=["decoupled", "argmax", "two_map"],
        default="decoupled",
        help=(
            "How to pick (x,y,label): decoupled = current default (soft-argmax coords + label head argmax); "
            "argmax = hard argmax coords + label head argmax; two_map = derive foreground/background maps from heatmap "
            "(p_fg = softmax(logits/tau), p_bg = 1 - p_fg), pick global maximum over 2xHxW to jointly decide label and pixel."
        ),
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--debug_output_dir", type=str, default=None)
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
    return p.parse_args()


def _device_and_amp(args: argparse.Namespace) -> Tuple[torch.device, bool, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    amp_enabled = args.amp and (device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    return device, amp_enabled, autocast_device_type


def _load_model(model_path: str, device: torch.device) -> PointHeatmapModel:
    cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
    model = PointHeatmapModel(cfg).to(device)
    ckpt = torch.load(model_path, map_location="cpu") if model_path and os.path.isfile(model_path) else None
    if ckpt is not None:
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _to_tensor_separate(image_path: str, gray_mask: Optional[Image.Image], out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Prepare separate RGB and grayscale tensors for the model; returns (rgb, gray, (orig_h, orig_w))."""
    im0 = Image.open(image_path).convert("RGB")
    orig_w, orig_h = im0.size
    H, W = out_hw
    # decide resize size
    if H <= 0 or W <= 0:
        H, W = orig_h, orig_w
    im = im0.resize((W, H), resample=Image.BILINEAR) if (W, H) != (orig_w, orig_h) else im0
    if gray_mask is None:
        gray0 = Image.new("L", (orig_w, orig_h), 0)
    else:
        gray0 = gray_mask.convert("L")
    gray = gray0.resize((W, H), resample=Image.NEAREST) if (W, H) != (orig_w, orig_h) else gray0
    import torchvision.transforms.functional as TF
    rgb_t = TF.to_tensor(im)
    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    g_t = TF.to_tensor(gray)
    g_t = (g_t - 0.5) / 0.5
    return rgb_t.unsqueeze(0), g_t.unsqueeze(0), (orig_h, orig_w)  # [1,3,H,W], [1,1,H,W], (orig_h,orig_w)


def _predict_point(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float, coord_label_mode: str) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        T = max(float(tau), 1e-6)
        p = F.softmax(logits.view(b, -1) / T, dim=1).view(b, 1, H, W)

        if coord_label_mode == "decoupled":
            xy = soft_argmax_from_logits(logits, temperature=tau)[0]
            lab = int(label_logits.argmax(dim=1).item())
        elif coord_label_mode == "argmax":
            from .model import argmax_from_logits  # local import to avoid circular on fallback
            xy = argmax_from_logits(logits)[0]
            lab = int(label_logits.argmax(dim=1).item())
        elif coord_label_mode == "two_map":
            # Minimal heuristic: build two maps p_fg = p, p_bg = 1 - p; pick global maximum over both
            p_fg = p  # [B,1,H,W]
            p_bg = 1.0 - p_fg
            two_maps = torch.cat([p_fg, p_bg], dim=1)  # [B,2,H,W]
            idx = two_maps.view(b, -1).argmax(dim=1)
            # decode index -> label, y, x
            per_map = H * W
            lab_idx = (idx // per_map).to(torch.int64)
            rem = (idx % per_map)
            y = (rem // W).to(torch.float32)
            x = (rem % W).to(torch.float32)
            xy = torch.stack([x, y], dim=1)
            lab = int(lab_idx.item())
        else:
            # fallback to default
            xy = soft_argmax_from_logits(logits, temperature=tau)[0]
            lab = int(label_logits.argmax(dim=1).item())

        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int) -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"heatmap-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


def _list_images(images_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    out: List[str] = []
    for name in sorted(os.listdir(images_dir)):
        p = os.path.join(images_dir, name)
        if os.path.isdir(p):
            continue
        if os.path.splitext(name)[1].lower() in exts:
            out.append(p)
    return out


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def run_initial(args: argparse.Namespace) -> None:
    if not args.images_dir or not args.output_json:
        raise RuntimeError("--images_dir and --output_json are required for initial mode")
    if not args.masks_dir:
        raise RuntimeError("--masks_dir is required for initial mode to fill 'gt_mask' field")
    device, amp_enabled, autocast_device_type = _device_and_amp(args)
    model = _load_model(args.model_path, device)
    H, W = int(args.height), int(args.width)

    records: List[Dict] = []
    images = _list_images(args.images_dir)
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            images_iter = tqdm(images, desc="Initial predict")
        except Exception:
            images_iter = images
    else:
        images_iter = images

    for img_path in images_iter:
        rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=None, out_hw=(H, W))
        with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau, args.coord_label_mode)
        # scale back to native size if resized
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        # clamp to bounds
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        stem = _stem(img_path)
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
        gt_mask_path = os.path.join(args.masks_dir, f"{stem}.png")
        rec = {
            "image": os.path.abspath(img_path),
            "gt_mask": os.path.abspath(gt_mask_path),
            "sam_masks_dir": os.path.abspath(args.sam_dir) if args.sam_dir else None,
            "points": [[float(x), float(y)]],
            "labels": [int(lab)],
        }
        # keep sam_masks_dir only if provided
        if rec["sam_masks_dir"] is None:
            del rec["sam_masks_dir"]
        records.append(rec)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} records to {args.output_json}")


def run_append(args: argparse.Namespace) -> None:
    if not args.appendto_json:
        raise RuntimeError("--appendto_json required for append mode")
    if not os.path.isfile(args.appendto_json):
        raise FileNotFoundError(args.appendto_json)
    arr = json.loads(open(args.appendto_json, "r", encoding="utf-8").read().strip())
    if not isinstance(arr, list):
        raise RuntimeError("Append target must be a JSON array")
    device, amp_enabled, autocast_device_type = _device_and_amp(args)
    model = _load_model(args.model_path, device)
    H, W = int(args.height), int(args.width)

    updated = 0
    # progress bar
    rec_iter = arr
    pbar = None
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=len(arr), desc="Append predict")
        except Exception:
            pbar = None
    for rec in arr:
        if not isinstance(rec, dict):
            continue
        img_path = rec.get("image")
        sam_dir = rec.get("sam_masks_dir")
        pts = rec.get("points", [])
        labs = rec.get("labels", [])
        if not img_path:
            continue
        k = len(pts)
        # prev mask
        if k == 0:
            gray = None
        else:
            stem = _stem(img_path)
            prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
            gray = Image.open(prev_path).convert("L") if os.path.isfile(prev_path) else None
        rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=gray, out_hw=(H, W))
        with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau, args.coord_label_mode)
        # scale back and clamp
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
        rec["points"] = (pts or []) + [[x, y]]
        rec["labels"] = (labs or []) + [int(lab)]
        # sam_masks_dir should be set by external segmenter; do not inject here
        updated += 1
        if pbar is not None:
            pbar.update(1)
        elif updated % max(1, len(arr)//10 or 1) == 0:
            print(f"Processed {updated}/{len(arr)} records")

    if pbar is not None:
        pbar.close()

    with open(args.appendto_json, "w", encoding="utf-8") as f:
        json.dump(arr, f, indent=2, ensure_ascii=False)
    print(f"Appended next points to {args.appendto_json} (updated {updated} items)")


def run_debug(args: argparse.Namespace) -> None:
    if not args.debug_json:
        raise RuntimeError("--debug_json required for debug mode")
    if not args.debug_output_dir:
        raise RuntimeError("--debug_output_dir required for debug mode")
    arr = json.loads(open(args.debug_json, "r", encoding="utf-8").read().strip())
    if not isinstance(arr, list):
        raise RuntimeError("Debug JSON must be a JSON array")
    os.makedirs(args.debug_output_dir, exist_ok=True)

    num_generated = 0
    # optional progress over total steps
    total_steps = 0
    if args.progress:
        try:
            for rec in arr:
                if isinstance(rec, dict):
                    pts = rec.get("points", []) if isinstance(rec.get("points", []), list) else []
                    total_steps += len(pts)
        except Exception:
            total_steps = 0
    pbar = None
    if args.progress and total_steps > 0:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=total_steps, desc="Debug render")
        except Exception:
            pbar = None

    for rec in arr:
        if not isinstance(rec, dict):
            continue
        image_path = rec.get("image")
        sam_dir = rec.get("sam_masks_dir") or args.sam_dir
        points = rec.get("points", [])
        labels = rec.get("labels", [])
        if not (image_path and isinstance(points, list) and isinstance(labels, list)):
            continue
        if len(points) != len(labels):
            continue
        stem = _stem(image_path)
        for i, (pt, lb) in enumerate(zip(points, labels)):
            cur_path = os.path.join(sam_dir, stem, f"{i}.png")
            cur = Image.open(cur_path).convert("L") if os.path.isfile(cur_path) else None
            if cur is None:
                continue
            cur_u8 = np.array(cur, dtype=np.uint8)
            if i == 0:
                prev_u8 = np.zeros_like(cur_u8, dtype=np.uint8)
            else:
                prev_path = os.path.join(sam_dir, stem, f"{i-1}.png")
                prev_u8 = np.array(Image.open(prev_path).convert("L"), dtype=np.uint8) if os.path.isfile(prev_path) else np.zeros_like(cur_u8, dtype=np.uint8)
            try:
                subdir = os.path.join(args.debug_output_dir, stem)
                os.makedirs(subdir, exist_ok=True)
                out_path = os.path.join(subdir, f"dbg_{i}.png")
            except Exception:
                out_path = None  # type: ignore
            if _render_diff_panels_with_point is not None and out_path is not None:
                _render_diff_panels_with_point(prev_u8, cur_u8, (float(pt[0]), float(pt[1])), int(lb), out_path)
                num_generated += 1
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    print(f"Generated debug images for {num_generated} steps into {args.debug_output_dir}")


def main() -> None:
    args = parse_args()
    if args.output_json:
        run_initial(args)
        return
    if args.appendto_json:
        run_append(args)
        return
    if args.debug_json:
        run_debug(args)
        return


if __name__ == "__main__":
    main()



```

I added a CLI flag to switch strategies: --coord_label_mode with choices decoupled (default), argmax, and two_map. I updated _predict_point to branch based on this flag while keeping the existing default behavior unchanged. Now I'll fix a potential import robustness issue in the argmax branch and run a quick lint check for the modified file.

```python
#!/usr/bin/env python3

"""
Predict next SAM prompt point (position + binary label) using a trained heatmap model.

This script mirrors seg-rl/annotator/gen_point_jsonl_from_masks.py, but replaces the
EDT-based next-point computation with a model inference that takes only:
  - the RGB image, and
  - the current predicted mask (or zero mask at step 0)
and outputs the next prompt point and its label.

Modes:
  1) Initial generation (write new JSON array):
     - For each image under --images_dir, predict the first prompt (i=0) with a zero mask.
     - Write a JSON array with records like:
         {"image": "/abs/path.jpg", "sam_masks_dir": "/abs/sam_dir", "points": [[x0,y0]], "labels": [l0]}

  2) Append mode (update existing JSON array):
     - Read --appendto_json as a JSON array, for each record take current length k=len(points),
       load the latest predicted mask at {sam_dir}/{stem}/{k-1}.png (k>0; else zero), and predict
       the next point (xk, yk, lk). Append to arrays and write back in-place.

  3) Debug visualization (no file write except images):
     - Read --debug_json (JSON array) that already contains points+labels and a sam_masks_dir.
     - For each step i, load prev and cur predicted masks (i-1 and i; prev is zero for i=0) and
       call the same 3-panel visualization function used by the annotator to render panels with
       the provided point, saving to --debug_output_dir/{stem}/dbg_{i}.png.
     - Ground-truth masks are NOT used anywhere in this script (except indirectly if you supply
       them in your own JSON for reference; they are not required).

Examples:
  # Initial generation
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/heatmap_train-251001-optimized/model_epoch_185.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred_points-sft_e185-251018.jsonl

  python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred_points-e160-251019.jsonl

  # Append mode
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --appendto_json datasets/seg_r1_md/Task01_BrainTumour/pred_points-251001.jsonl

  python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --appendto_json outputs/braintumour/pred_points-e160-251019.jsonl

  # Debug visualization only
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --debug_json datasets/seg_r1_md/Task01_BrainTumour/pred_points-init.jsonl \
    --debug_output_dir outputs/braintumour/dbg_model_pred

Notes:
  - The model expects 4-channel input (RGB normalized + Gray normalized mask). This script
    normalizes RGB with ImageNet mean/std and mask with mean=0.5, std=0.5.
  - Device selection prefers CUDA, then MPS (Apple Silicon), otherwise CPU. AMP is only used on CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

try:
    # package context
    from .model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

import importlib.util as _importlib_util

def _import_render_helper() -> Optional[Callable]:
    """Dynamically import _render_diff_panels_with_point from annotator script by file path.

    This avoids package-name issues due to hyphen in 'seg-rl'.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # seg-rl
        target = os.path.join(base_dir, "annotator", "gen_point_jsonl_from_masks.py")
        if not os.path.isfile(target):
            return None
        spec = _importlib_util.spec_from_file_location("gen_point_jsonl_from_masks", target)
        if spec is None or spec.loader is None:
            return None
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        func = getattr(module, "_render_diff_panels_with_point", None)
        if callable(func):
            return func  # type: ignore[return-value]
    except Exception:
        return None
    return None

_render_diff_panels_with_point: Optional[Callable] = _import_render_helper()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict next SAM prompt point via trained heatmap model")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_json", type=str, help="Write a new JSON array with first predicted points")
    group.add_argument("--appendto_json", type=str, help="Append next predicted point to an existing JSON array")
    group.add_argument("--debug_json", type=str, help="Debug visualization only: read JSON array and render panels")

    p.add_argument("--model_path", type=str, required=False, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--images_dir", type=str, default=None, help="Directory of input images (initial mode)")
    p.add_argument("--masks_dir", type=str, default=None, help="Ground-truth masks directory (initial mode only, to fill 'gt_mask' field)")
    p.add_argument("--sam_dir", type=str, default=None, help="Directory of predicted SAM masks per step: {sam_dir}/{stem}/{k}.png (initial/debug mode)")
    # height/width: 0 means use native image size; if >0, will resize for inference
    p.add_argument("--height", type=int, default=0, help="0 = use native image height")
    p.add_argument("--width", type=int, default=0, help="0 = use native image width")
    p.add_argument("--tau", type=float, default=1.0, help="Softmax temperature for heatmap -> probability")
    # Coordinate/label selection strategy
    p.add_argument(
        "--coord_label_mode",
        type=str,
        choices=["decoupled", "argmax", "two_map"],
        default="decoupled",
        help=(
            "How to pick (x,y,label): decoupled = current default (soft-argmax coords + label head argmax); "
            "argmax = hard argmax coords + label head argmax; two_map = derive foreground/background maps from heatmap "
            "(p_fg = softmax(logits/tau), p_bg = 1 - p_fg), pick global maximum over 2xHxW to jointly decide label and pixel."
        ),
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--debug_output_dir", type=str, default=None)
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
    return p.parse_args()


def _device_and_amp(args: argparse.Namespace) -> Tuple[torch.device, bool, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    amp_enabled = args.amp and (device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    return device, amp_enabled, autocast_device_type


def _load_model(model_path: str, device: torch.device) -> PointHeatmapModel:
    cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
    model = PointHeatmapModel(cfg).to(device)
    ckpt = torch.load(model_path, map_location="cpu") if model_path and os.path.isfile(model_path) else None
    if ckpt is not None:
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _to_tensor_separate(image_path: str, gray_mask: Optional[Image.Image], out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Prepare separate RGB and grayscale tensors for the model; returns (rgb, gray, (orig_h, orig_w))."""
    im0 = Image.open(image_path).convert("RGB")
    orig_w, orig_h = im0.size
    H, W = out_hw
    # decide resize size
    if H <= 0 or W <= 0:
        H, W = orig_h, orig_w
    im = im0.resize((W, H), resample=Image.BILINEAR) if (W, H) != (orig_w, orig_h) else im0
    if gray_mask is None:
        gray0 = Image.new("L", (orig_w, orig_h), 0)
    else:
        gray0 = gray_mask.convert("L")
    gray = gray0.resize((W, H), resample=Image.NEAREST) if (W, H) != (orig_w, orig_h) else gray0
    import torchvision.transforms.functional as TF
    rgb_t = TF.to_tensor(im)
    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    g_t = TF.to_tensor(gray)
    g_t = (g_t - 0.5) / 0.5
    return rgb_t.unsqueeze(0), g_t.unsqueeze(0), (orig_h, orig_w)  # [1,3,H,W], [1,1,H,W], (orig_h,orig_w)


def _predict_point(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float, coord_label_mode: str) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        T = max(float(tau), 1e-6)
        p = F.softmax(logits.view(b, -1) / T, dim=1).view(b, 1, H, W)

        if coord_label_mode == "decoupled":
            xy = soft_argmax_from_logits(logits, temperature=tau)[0]
            lab = int(label_logits.argmax(dim=1).item())
        elif coord_label_mode == "argmax":
            # Hard argmax over pixels for coordinates
            idx = logits.view(b, -1).argmax(dim=1)
            y = (idx // W).to(torch.float32)
            x = (idx % W).to(torch.float32)
            xy = torch.stack([x, y], dim=1)[0]
            lab = int(label_logits.argmax(dim=1).item())
        elif coord_label_mode == "two_map":
            # Minimal heuristic: build two maps p_fg = p, p_bg = 1 - p; pick global maximum over both
            p_fg = p  # [B,1,H,W]
            p_bg = 1.0 - p_fg
            two_maps = torch.cat([p_fg, p_bg], dim=1)  # [B,2,H,W]
            idx = two_maps.view(b, -1).argmax(dim=1)
            # decode index -> label, y, x
            per_map = H * W
            lab_idx = (idx // per_map).to(torch.int64)
            rem = (idx % per_map)
            y = (rem // W).to(torch.float32)
            x = (rem % W).to(torch.float32)
            xy = torch.stack([x, y], dim=1)
            lab = int(lab_idx.item())
        else:
            # fallback to default
            xy = soft_argmax_from_logits(logits, temperature=tau)[0]
            lab = int(label_logits.argmax(dim=1).item())

        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int) -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"heatmap-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


def _list_images(images_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    out: List[str] = []
    for name in sorted(os.listdir(images_dir)):
        p = os.path.join(images_dir, name)
        if os.path.isdir(p):
            continue
        if os.path.splitext(name)[1].lower() in exts:
            out.append(p)
    return out


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def run_initial(args: argparse.Namespace) -> None:
    if not args.images_dir or not args.output_json:
        raise RuntimeError("--images_dir and --output_json are required for initial mode")
    if not args.masks_dir:
        raise RuntimeError("--masks_dir is required for initial mode to fill 'gt_mask' field")
    device, amp_enabled, autocast_device_type = _device_and_amp(args)
    model = _load_model(args.model_path, device)
    H, W = int(args.height), int(args.width)

    records: List[Dict] = []
    images = _list_images(args.images_dir)
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            images_iter = tqdm(images, desc="Initial predict")
        except Exception:
            images_iter = images
    else:
        images_iter = images

    for img_path in images_iter:
        rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=None, out_hw=(H, W))
        with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau, args.coord_label_mode)
        # scale back to native size if resized
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        # clamp to bounds
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        stem = _stem(img_path)
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
        gt_mask_path = os.path.join(args.masks_dir, f"{stem}.png")
        rec = {
            "image": os.path.abspath(img_path),
            "gt_mask": os.path.abspath(gt_mask_path),
            "sam_masks_dir": os.path.abspath(args.sam_dir) if args.sam_dir else None,
            "points": [[float(x), float(y)]],
            "labels": [int(lab)],
        }
        # keep sam_masks_dir only if provided
        if rec["sam_masks_dir"] is None:
            del rec["sam_masks_dir"]
        records.append(rec)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} records to {args.output_json}")


def run_append(args: argparse.Namespace) -> None:
    if not args.appendto_json:
        raise RuntimeError("--appendto_json required for append mode")
    if not os.path.isfile(args.appendto_json):
        raise FileNotFoundError(args.appendto_json)
    arr = json.loads(open(args.appendto_json, "r", encoding="utf-8").read().strip())
    if not isinstance(arr, list):
        raise RuntimeError("Append target must be a JSON array")
    device, amp_enabled, autocast_device_type = _device_and_amp(args)
    model = _load_model(args.model_path, device)
    H, W = int(args.height), int(args.width)

    updated = 0
    # progress bar
    rec_iter = arr
    pbar = None
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=len(arr), desc="Append predict")
        except Exception:
            pbar = None
    for rec in arr:
        if not isinstance(rec, dict):
            continue
        img_path = rec.get("image")
        sam_dir = rec.get("sam_masks_dir")
        pts = rec.get("points", [])
        labs = rec.get("labels", [])
        if not img_path:
            continue
        k = len(pts)
        # prev mask
        if k == 0:
            gray = None
        else:
            stem = _stem(img_path)
            prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
            gray = Image.open(prev_path).convert("L") if os.path.isfile(prev_path) else None
        rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=gray, out_hw=(H, W))
        with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau, args.coord_label_mode)
        # scale back and clamp
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
        rec["points"] = (pts or []) + [[x, y]]
        rec["labels"] = (labs or []) + [int(lab)]
        # sam_masks_dir should be set by external segmenter; do not inject here
        updated += 1
        if pbar is not None:
            pbar.update(1)
        elif updated % max(1, len(arr)//10 or 1) == 0:
            print(f"Processed {updated}/{len(arr)} records")

    if pbar is not None:
        pbar.close()

    with open(args.appendto_json, "w", encoding="utf-8") as f:
        json.dump(arr, f, indent=2, ensure_ascii=False)
    print(f"Appended next points to {args.appendto_json} (updated {updated} items)")


def run_debug(args: argparse.Namespace) -> None:
    if not args.debug_json:
        raise RuntimeError("--debug_json required for debug mode")
    if not args.debug_output_dir:
        raise RuntimeError("--debug_output_dir required for debug mode")
    arr = json.loads(open(args.debug_json, "r", encoding="utf-8").read().strip())
    if not isinstance(arr, list):
        raise RuntimeError("Debug JSON must be a JSON array")
    os.makedirs(args.debug_output_dir, exist_ok=True)

    num_generated = 0
    # optional progress over total steps
    total_steps = 0
    if args.progress:
        try:
            for rec in arr:
                if isinstance(rec, dict):
                    pts = rec.get("points", []) if isinstance(rec.get("points", []), list) else []
                    total_steps += len(pts)
        except Exception:
            total_steps = 0
    pbar = None
    if args.progress and total_steps > 0:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=total_steps, desc="Debug render")
        except Exception:
            pbar = None

    for rec in arr:
        if not isinstance(rec, dict):
            continue
        image_path = rec.get("image")
        sam_dir = rec.get("sam_masks_dir") or args.sam_dir
        points = rec.get("points", [])
        labels = rec.get("labels", [])
        if not (image_path and isinstance(points, list) and isinstance(labels, list)):
            continue
        if len(points) != len(labels):
            continue
        stem = _stem(image_path)
        for i, (pt, lb) in enumerate(zip(points, labels)):
            cur_path = os.path.join(sam_dir, stem, f"{i}.png")
            cur = Image.open(cur_path).convert("L") if os.path.isfile(cur_path) else None
            if cur is None:
                continue
            cur_u8 = np.array(cur, dtype=np.uint8)
            if i == 0:
                prev_u8 = np.zeros_like(cur_u8, dtype=np.uint8)
            else:
                prev_path = os.path.join(sam_dir, stem, f"{i-1}.png")
                prev_u8 = np.array(Image.open(prev_path).convert("L"), dtype=np.uint8) if os.path.isfile(prev_path) else np.zeros_like(cur_u8, dtype=np.uint8)
            try:
                subdir = os.path.join(args.debug_output_dir, stem)
                os.makedirs(subdir, exist_ok=True)
                out_path = os.path.join(subdir, f"dbg_{i}.png")
            except Exception:
                out_path = None  # type: ignore
            if _render_diff_panels_with_point is not None and out_path is not None:
                _render_diff_panels_with_point(prev_u8, cur_u8, (float(pt[0]), float(pt[1])), int(lb), out_path)
                num_generated += 1
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    print(f"Generated debug images for {num_generated} steps into {args.debug_output_dir}")


def main() -> None:
    args = parse_args()
    if args.output_json:
        run_initial(args)
        return
    if args.appendto_json:
        run_append(args)
        return
    if args.debug_json:
        run_debug(args)
        return


if __name__ == "__main__":
    main()



```

I'm going to quickly scan the SAM integration code to confirm whether label 1 maps to foreground and 0 to background, so the new two_map mode returns labels consistent with the rest of the pipeline. Then I'll run a linter check on the modified file.

我已实现最小改动，并提供命令行开关，默认保持原有行为不变。

- 新增参数
  - --coord_label_mode: 选择坐标/标签联合决策策略
    - decoupled（默认）: 维持现有架构，坐标用软argmax，标签用分类头argmax
    - argmax: 坐标用硬argmax（最大logit像素），标签仍用分类头argmax
    - two_map: 从单通道热力图得到 p_fg=softmax(logits/tau)，p_bg=1-p_fg，拼成两张图，直接在 2×H×W 上取全局最大，同时确定坐标与前/背景标签

- 代码变更
  - 在 predict_next_point_from_model.py 的 parse_args 增加 --coord_label_mode
  - 修改 _predict_point 支持三种模式分支，默认 decoupled 不变
  - 两个调用点 run_initial 和 run_append 传递该参数
  - 保持输出的单张概率图 prob 为 p_fg（便于与现有可视化兼容）

使用示例
- 保持原行为（默认）:
  - python -m seg-rl.heatmap.predict_next_point_from_model --model_path ... --output_json ... [其他参数]
- 硬 argmax 坐标:
  - python -m seg-rl.heatmap.predict_next_point_from_model --coord_label_mode argmax --model_path ... --appendto_json ...
- 两图联合决策:
  - python -m seg-rl.heatmap.predict_next_point_from_model --coord_label_mode two_map --model_path ... --output_json ...

注意
- two_map 模式中 label=1 表示前景（来自 p_fg 通道），label=0 表示背景（来自 p_bg 通道），与现有可视化与 SAM 集成一致。
- 该实现是“最小启用”的启发式两图方案，不改动训练与模型结构，便于快速消融。如果要进一步提升鲁棒性，可后续将两图逻辑迁移到训练目标或引入条件化头。

我已验证语法无lint问题。如需把 two_map 的概率图也输出（例如分别保存 p_fg 与 p_bg），我可以再加一个可选参数控制保存两张图。

---

**User**

是的，如果使用两张概率图时，相应地也需要分别保存。并且在seg-rl/annotator/gen_point_jsonl_from_masks.py的调试模式下，也应该根据中间结果目录（sam_masks_dir）中是否对每个样本的每个提示点序列长度有两个概率图（可用p_fg-xxx.png与p_bg-xxx.png来区分）来决定是否在可视化展示时加入第五列图，从原来的只有一个概率图变成两个

---

**Cursor**

I'll locate the rendering function in the annotator to see how to extend it to optionally show both p_fg and p_bg heatmaps, and update the predictor to save both maps when two_map mode is used. Then I'll implement the edits and run lint checks.

```python
#!/usr/bin/env python3

"""
Predict next SAM prompt point (position + binary label) using a trained heatmap model.

This script mirrors seg-rl/annotator/gen_point_jsonl_from_masks.py, but replaces the
EDT-based next-point computation with a model inference that takes only:
  - the RGB image, and
  - the current predicted mask (or zero mask at step 0)
and outputs the next prompt point and its label.

Modes:
  1) Initial generation (write new JSON array):
     - For each image under --images_dir, predict the first prompt (i=0) with a zero mask.
     - Write a JSON array with records like:
         {"image": "/abs/path.jpg", "sam_masks_dir": "/abs/sam_dir", "points": [[x0,y0]], "labels": [l0]}

  2) Append mode (update existing JSON array):
     - Read --appendto_json as a JSON array, for each record take current length k=len(points),
       load the latest predicted mask at {sam_dir}/{stem}/{k-1}.png (k>0; else zero), and predict
       the next point (xk, yk, lk). Append to arrays and write back in-place.

  3) Debug visualization (no file write except images):
     - Read --debug_json (JSON array) that already contains points+labels and a sam_masks_dir.
     - For each step i, load prev and cur predicted masks (i-1 and i; prev is zero for i=0) and
       call the same 3-panel visualization function used by the annotator to render panels with
       the provided point, saving to --debug_output_dir/{stem}/dbg_{i}.png.
     - Ground-truth masks are NOT used anywhere in this script (except indirectly if you supply
       them in your own JSON for reference; they are not required).

Examples:
  # Initial generation
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/heatmap_train-251001-optimized/model_epoch_185.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred_points-sft_e185-251018.jsonl

  python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred_points-e160-251019.jsonl

  # Append mode
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --appendto_json datasets/seg_r1_md/Task01_BrainTumour/pred_points-251001.jsonl

  python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --appendto_json outputs/braintumour/pred_points-e160-251019.jsonl

  # Debug visualization only
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --debug_json datasets/seg_r1_md/Task01_BrainTumour/pred_points-init.jsonl \
    --debug_output_dir outputs/braintumour/dbg_model_pred

Notes:
  - The model expects 4-channel input (RGB normalized + Gray normalized mask). This script
    normalizes RGB with ImageNet mean/std and mask with mean=0.5, std=0.5.
  - Device selection prefers CUDA, then MPS (Apple Silicon), otherwise CPU. AMP is only used on CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

try:
    # package context
    from .model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

import importlib.util as _importlib_util

def _import_render_helper() -> Optional[Callable]:
    """Dynamically import _render_diff_panels_with_point from annotator script by file path.

    This avoids package-name issues due to hyphen in 'seg-rl'.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # seg-rl
        target = os.path.join(base_dir, "annotator", "gen_point_jsonl_from_masks.py")
        if not os.path.isfile(target):
            return None
        spec = _importlib_util.spec_from_file_location("gen_point_jsonl_from_masks", target)
        if spec is None or spec.loader is None:
            return None
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        func = getattr(module, "_render_diff_panels_with_point", None)
        if callable(func):
            return func  # type: ignore[return-value]
    except Exception:
        return None
    return None

_render_diff_panels_with_point: Optional[Callable] = _import_render_helper()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict next SAM prompt point via trained heatmap model")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_json", type=str, help="Write a new JSON array with first predicted points")
    group.add_argument("--appendto_json", type=str, help="Append next predicted point to an existing JSON array")
    group.add_argument("--debug_json", type=str, help="Debug visualization only: read JSON array and render panels")

    p.add_argument("--model_path", type=str, required=False, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--images_dir", type=str, default=None, help="Directory of input images (initial mode)")
    p.add_argument("--masks_dir", type=str, default=None, help="Ground-truth masks directory (initial mode only, to fill 'gt_mask' field)")
    p.add_argument("--sam_dir", type=str, default=None, help="Directory of predicted SAM masks per step: {sam_dir}/{stem}/{k}.png (initial/debug mode)")
    # height/width: 0 means use native image size; if >0, will resize for inference
    p.add_argument("--height", type=int, default=0, help="0 = use native image height")
    p.add_argument("--width", type=int, default=0, help="0 = use native image width")
    p.add_argument("--tau", type=float, default=1.0, help="Softmax temperature for heatmap -> probability")
    # Coordinate/label selection strategy
    p.add_argument(
        "--coord_label_mode",
        type=str,
        choices=["decoupled", "argmax", "two_map"],
        default="decoupled",
        help=(
            "How to pick (x,y,label): decoupled = current default (soft-argmax coords + label head argmax); "
            "argmax = hard argmax coords + label head argmax; two_map = derive foreground/background maps from heatmap "
            "(p_fg = softmax(logits/tau), p_bg = 1 - p_fg), pick global maximum over 2xHxW to jointly decide label and pixel."
        ),
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--debug_output_dir", type=str, default=None)
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
    return p.parse_args()


def _device_and_amp(args: argparse.Namespace) -> Tuple[torch.device, bool, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    amp_enabled = args.amp and (device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    return device, amp_enabled, autocast_device_type


def _load_model(model_path: str, device: torch.device) -> PointHeatmapModel:
    cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
    model = PointHeatmapModel(cfg).to(device)
    ckpt = torch.load(model_path, map_location="cpu") if model_path and os.path.isfile(model_path) else None
    if ckpt is not None:
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _to_tensor_separate(image_path: str, gray_mask: Optional[Image.Image], out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Prepare separate RGB and grayscale tensors for the model; returns (rgb, gray, (orig_h, orig_w))."""
    im0 = Image.open(image_path).convert("RGB")
    orig_w, orig_h = im0.size
    H, W = out_hw
    # decide resize size
    if H <= 0 or W <= 0:
        H, W = orig_h, orig_w
    im = im0.resize((W, H), resample=Image.BILINEAR) if (W, H) != (orig_w, orig_h) else im0
    if gray_mask is None:
        gray0 = Image.new("L", (orig_w, orig_h), 0)
    else:
        gray0 = gray_mask.convert("L")
    gray = gray0.resize((W, H), resample=Image.NEAREST) if (W, H) != (orig_w, orig_h) else gray0
    import torchvision.transforms.functional as TF
    rgb_t = TF.to_tensor(im)
    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    g_t = TF.to_tensor(gray)
    g_t = (g_t - 0.5) / 0.5
    return rgb_t.unsqueeze(0), g_t.unsqueeze(0), (orig_h, orig_w)  # [1,3,H,W], [1,1,H,W], (orig_h,orig_w)


def _predict_point(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float, coord_label_mode: str) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        T = max(float(tau), 1e-6)
        p = F.softmax(logits.view(b, -1) / T, dim=1).view(b, 1, H, W)

        if coord_label_mode == "decoupled":
            xy = soft_argmax_from_logits(logits, temperature=tau)[0]
            lab = int(label_logits.argmax(dim=1).item())
        elif coord_label_mode == "argmax":
            # Hard argmax over pixels for coordinates
            idx = logits.view(b, -1).argmax(dim=1)
            y = (idx // W).to(torch.float32)
            x = (idx % W).to(torch.float32)
            xy = torch.stack([x, y], dim=1)[0]
            lab = int(label_logits.argmax(dim=1).item())
        elif coord_label_mode == "two_map":
            # Minimal heuristic: build two maps p_fg = p, p_bg = 1 - p; pick global maximum over both
            p_fg = p  # [B,1,H,W]
            p_bg = 1.0 - p_fg
            two_maps = torch.cat([p_fg, p_bg], dim=1)  # [B,2,H,W]
            idx = two_maps.view(b, -1).argmax(dim=1)
            # decode index -> label, y, x
            per_map = H * W
            lab_idx = (idx // per_map).to(torch.int64)
            rem = (idx % per_map)
            y = (rem // W).to(torch.float32)
            x = (rem % W).to(torch.float32)
            xy = torch.stack([x, y], dim=1)
            lab = int(lab_idx.item())
        else:
            # fallback to default
            xy = soft_argmax_from_logits(logits, temperature=tau)[0]
            lab = int(label_logits.argmax(dim=1).item())

        prob = p[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, prob


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int, name_prefix: str = "heatmap") -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"{name_prefix}-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


def _list_images(images_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    out: List[str] = []
    for name in sorted(os.listdir(images_dir)):
        p = os.path.join(images_dir, name)
        if os.path.isdir(p):
            continue
        if os.path.splitext(name)[1].lower() in exts:
            out.append(p)
    return out


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def run_initial(args: argparse.Namespace) -> None:
    if not args.images_dir or not args.output_json:
        raise RuntimeError("--images_dir and --output_json are required for initial mode")
    if not args.masks_dir:
        raise RuntimeError("--masks_dir is required for initial mode to fill 'gt_mask' field")
    device, amp_enabled, autocast_device_type = _device_and_amp(args)
    model = _load_model(args.model_path, device)
    H, W = int(args.height), int(args.width)

    records: List[Dict] = []
    images = _list_images(args.images_dir)
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            images_iter = tqdm(images, desc="Initial predict")
        except Exception:
            images_iter = images
    else:
        images_iter = images

    for img_path in images_iter:
        rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=None, out_hw=(H, W))
        with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau, args.coord_label_mode)
        # scale back to native size if resized
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        # clamp to bounds
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        stem = _stem(img_path)
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
        gt_mask_path = os.path.join(args.masks_dir, f"{stem}.png")
        rec = {
            "image": os.path.abspath(img_path),
            "gt_mask": os.path.abspath(gt_mask_path),
            "sam_masks_dir": os.path.abspath(args.sam_dir) if args.sam_dir else None,
            "points": [[float(x), float(y)]],
            "labels": [int(lab)],
        }
        # keep sam_masks_dir only if provided
        if rec["sam_masks_dir"] is None:
            del rec["sam_masks_dir"]
        records.append(rec)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} records to {args.output_json}")


def run_append(args: argparse.Namespace) -> None:
    if not args.appendto_json:
        raise RuntimeError("--appendto_json required for append mode")
    if not os.path.isfile(args.appendto_json):
        raise FileNotFoundError(args.appendto_json)
    arr = json.loads(open(args.appendto_json, "r", encoding="utf-8").read().strip())
    if not isinstance(arr, list):
        raise RuntimeError("Append target must be a JSON array")
    device, amp_enabled, autocast_device_type = _device_and_amp(args)
    model = _load_model(args.model_path, device)
    H, W = int(args.height), int(args.width)

    updated = 0
    # progress bar
    rec_iter = arr
    pbar = None
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=len(arr), desc="Append predict")
        except Exception:
            pbar = None
    for rec in arr:
        if not isinstance(rec, dict):
            continue
        img_path = rec.get("image")
        sam_dir = rec.get("sam_masks_dir")
        pts = rec.get("points", [])
        labs = rec.get("labels", [])
        if not img_path:
            continue
        k = len(pts)
        # prev mask
        if k == 0:
            gray = None
        else:
            stem = _stem(img_path)
            prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
            gray = Image.open(prev_path).convert("L") if os.path.isfile(prev_path) else None
        rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=gray, out_hw=(H, W))
        with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau, args.coord_label_mode)
        # scale back and clamp
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
        rec["points"] = (pts or []) + [[x, y]]
        rec["labels"] = (labs or []) + [int(lab)]
        # sam_masks_dir should be set by external segmenter; do not inject here
        updated += 1
        if pbar is not None:
            pbar.update(1)
        elif updated % max(1, len(arr)//10 or 1) == 0:
            print(f"Processed {updated}/{len(arr)} records")

    if pbar is not None:
        pbar.close()

    with open(args.appendto_json, "w", encoding="utf-8") as f:
        json.dump(arr, f, indent=2, ensure_ascii=False)
    print(f"Appended next points to {args.appendto_json} (updated {updated} items)")


def run_debug(args: argparse.Namespace) -> None:
    if not args.debug_json:
        raise RuntimeError("--debug_json required for debug mode")
    if not args.debug_output_dir:
        raise RuntimeError("--debug_output_dir required for debug mode")
    arr = json.loads(open(args.debug_json, "r", encoding="utf-8").read().strip())
    if not isinstance(arr, list):
        raise RuntimeError("Debug JSON must be a JSON array")
    os.makedirs(args.debug_output_dir, exist_ok=True)

    num_generated = 0
    # optional progress over total steps
    total_steps = 0
    if args.progress:
        try:
            for rec in arr:
                if isinstance(rec, dict):
                    pts = rec.get("points", []) if isinstance(rec.get("points", []), list) else []
                    total_steps += len(pts)
        except Exception:
            total_steps = 0
    pbar = None
    if args.progress and total_steps > 0:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=total_steps, desc="Debug render")
        except Exception:
            pbar = None

    for rec in arr:
        if not isinstance(rec, dict):
            continue
        image_path = rec.get("image")
        sam_dir = rec.get("sam_masks_dir") or args.sam_dir
        points = rec.get("points", [])
        labels = rec.get("labels", [])
        if not (image_path and isinstance(points, list) and isinstance(labels, list)):
            continue
        if len(points) != len(labels):
            continue
        stem = _stem(image_path)
        for i, (pt, lb) in enumerate(zip(points, labels)):
            cur_path = os.path.join(sam_dir, stem, f"{i}.png")
            cur = Image.open(cur_path).convert("L") if os.path.isfile(cur_path) else None
            if cur is None:
                continue
            cur_u8 = np.array(cur, dtype=np.uint8)
            if i == 0:
                prev_u8 = np.zeros_like(cur_u8, dtype=np.uint8)
            else:
                prev_path = os.path.join(sam_dir, stem, f"{i-1}.png")
                prev_u8 = np.array(Image.open(prev_path).convert("L"), dtype=np.uint8) if os.path.isfile(prev_path) else np.zeros_like(cur_u8, dtype=np.uint8)
            try:
                subdir = os.path.join(args.debug_output_dir, stem)
                os.makedirs(subdir, exist_ok=True)
                out_path = os.path.join(subdir, f"dbg_{i}.png")
            except Exception:
                out_path = None  # type: ignore
            if _render_diff_panels_with_point is not None and out_path is not None:
                _render_diff_panels_with_point(prev_u8, cur_u8, (float(pt[0]), float(pt[1])), int(lb), out_path)
                num_generated += 1
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    print(f"Generated debug images for {num_generated} steps into {args.debug_output_dir}")


def main() -> None:
    args = parse_args()
    if args.output_json:
        run_initial(args)
        return
    if args.appendto_json:
        run_append(args)
        return
    if args.debug_json:
        run_debug(args)
        return


if __name__ == "__main__":
    main()



```

```python
#!/usr/bin/env python3

"""
Predict next SAM prompt point (position + binary label) using a trained heatmap model.

This script mirrors seg-rl/annotator/gen_point_jsonl_from_masks.py, but replaces the
EDT-based next-point computation with a model inference that takes only:
  - the RGB image, and
  - the current predicted mask (or zero mask at step 0)
and outputs the next prompt point and its label.

Modes:
  1) Initial generation (write new JSON array):
     - For each image under --images_dir, predict the first prompt (i=0) with a zero mask.
     - Write a JSON array with records like:
         {"image": "/abs/path.jpg", "sam_masks_dir": "/abs/sam_dir", "points": [[x0,y0]], "labels": [l0]}

  2) Append mode (update existing JSON array):
     - Read --appendto_json as a JSON array, for each record take current length k=len(points),
       load the latest predicted mask at {sam_dir}/{stem}/{k-1}.png (k>0; else zero), and predict
       the next point (xk, yk, lk). Append to arrays and write back in-place.

  3) Debug visualization (no file write except images):
     - Read --debug_json (JSON array) that already contains points+labels and a sam_masks_dir.
     - For each step i, load prev and cur predicted masks (i-1 and i; prev is zero for i=0) and
       call the same 3-panel visualization function used by the annotator to render panels with
       the provided point, saving to --debug_output_dir/{stem}/dbg_{i}.png.
     - Ground-truth masks are NOT used anywhere in this script (except indirectly if you supply
       them in your own JSON for reference; they are not required).

Examples:
  # Initial generation
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/heatmap_train-251001-optimized/model_epoch_185.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred_points-sft_e185-251018.jsonl

  python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred_points-e160-251019.jsonl

  # Append mode
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --appendto_json datasets/seg_r1_md/Task01_BrainTumour/pred_points-251001.jsonl

  python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --appendto_json outputs/braintumour/pred_points-e160-251019.jsonl

  # Debug visualization only
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --debug_json datasets/seg_r1_md/Task01_BrainTumour/pred_points-init.jsonl \
    --debug_output_dir outputs/braintumour/dbg_model_pred

Notes:
  - The model expects 4-channel input (RGB normalized + Gray normalized mask). This script
    normalizes RGB with ImageNet mean/std and mask with mean=0.5, std=0.5.
  - Device selection prefers CUDA, then MPS (Apple Silicon), otherwise CPU. AMP is only used on CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

try:
    # package context
    from .model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

import importlib.util as _importlib_util

def _import_render_helper() -> Optional[Callable]:
    """Dynamically import _render_diff_panels_with_point from annotator script by file path.

    This avoids package-name issues due to hyphen in 'seg-rl'.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # seg-rl
        target = os.path.join(base_dir, "annotator", "gen_point_jsonl_from_masks.py")
        if not os.path.isfile(target):
            return None
        spec = _importlib_util.spec_from_file_location("gen_point_jsonl_from_masks", target)
        if spec is None or spec.loader is None:
            return None
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        func = getattr(module, "_render_diff_panels_with_point", None)
        if callable(func):
            return func  # type: ignore[return-value]
    except Exception:
        return None
    return None

_render_diff_panels_with_point: Optional[Callable] = _import_render_helper()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict next SAM prompt point via trained heatmap model")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_json", type=str, help="Write a new JSON array with first predicted points")
    group.add_argument("--appendto_json", type=str, help="Append next predicted point to an existing JSON array")
    group.add_argument("--debug_json", type=str, help="Debug visualization only: read JSON array and render panels")

    p.add_argument("--model_path", type=str, required=False, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--images_dir", type=str, default=None, help="Directory of input images (initial mode)")
    p.add_argument("--masks_dir", type=str, default=None, help="Ground-truth masks directory (initial mode only, to fill 'gt_mask' field)")
    p.add_argument("--sam_dir", type=str, default=None, help="Directory of predicted SAM masks per step: {sam_dir}/{stem}/{k}.png (initial/debug mode)")
    # height/width: 0 means use native image size; if >0, will resize for inference
    p.add_argument("--height", type=int, default=0, help="0 = use native image height")
    p.add_argument("--width", type=int, default=0, help="0 = use native image width")
    p.add_argument("--tau", type=float, default=1.0, help="Softmax temperature for heatmap -> probability")
    # Coordinate/label selection strategy
    p.add_argument(
        "--coord_label_mode",
        type=str,
        choices=["decoupled", "argmax", "two_map"],
        default="decoupled",
        help=(
            "How to pick (x,y,label): decoupled = current default (soft-argmax coords + label head argmax); "
            "argmax = hard argmax coords + label head argmax; two_map = derive foreground/background maps from heatmap "
            "(p_fg = softmax(logits/tau), p_bg = 1 - p_fg), pick global maximum over 2xHxW to jointly decide label and pixel."
        ),
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--debug_output_dir", type=str, default=None)
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
    return p.parse_args()


def _device_and_amp(args: argparse.Namespace) -> Tuple[torch.device, bool, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    amp_enabled = args.amp and (device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    return device, amp_enabled, autocast_device_type


def _load_model(model_path: str, device: torch.device) -> PointHeatmapModel:
    cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
    model = PointHeatmapModel(cfg).to(device)
    ckpt = torch.load(model_path, map_location="cpu") if model_path and os.path.isfile(model_path) else None
    if ckpt is not None:
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _to_tensor_separate(image_path: str, gray_mask: Optional[Image.Image], out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Prepare separate RGB and grayscale tensors for the model; returns (rgb, gray, (orig_h, orig_w))."""
    im0 = Image.open(image_path).convert("RGB")
    orig_w, orig_h = im0.size
    H, W = out_hw
    # decide resize size
    if H <= 0 or W <= 0:
        H, W = orig_h, orig_w
    im = im0.resize((W, H), resample=Image.BILINEAR) if (W, H) != (orig_w, orig_h) else im0
    if gray_mask is None:
        gray0 = Image.new("L", (orig_w, orig_h), 0)
    else:
        gray0 = gray_mask.convert("L")
    gray = gray0.resize((W, H), resample=Image.NEAREST) if (W, H) != (orig_w, orig_h) else gray0
    import torchvision.transforms.functional as TF
    rgb_t = TF.to_tensor(im)
    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    g_t = TF.to_tensor(gray)
    g_t = (g_t - 0.5) / 0.5
    return rgb_t.unsqueeze(0), g_t.unsqueeze(0), (orig_h, orig_w)  # [1,3,H,W], [1,1,H,W], (orig_h,orig_w)


def _predict_point(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float, coord_label_mode: str) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        T = max(float(tau), 1e-6)
        p = F.softmax(logits.view(b, -1) / T, dim=1).view(b, 1, H, W)

        if coord_label_mode == "decoupled":
            xy = soft_argmax_from_logits(logits, temperature=tau)[0]
            lab = int(label_logits.argmax(dim=1).item())
        elif coord_label_mode == "argmax":
            # Hard argmax over pixels for coordinates
            idx = logits.view(b, -1).argmax(dim=1)
            y = (idx // W).to(torch.float32)
            x = (idx % W).to(torch.float32)
            xy = torch.stack([x, y], dim=1)[0]
            lab = int(label_logits.argmax(dim=1).item())
        elif coord_label_mode == "two_map":
            # Minimal heuristic: build two maps p_fg = p, p_bg = 1 - p; pick global maximum over both
            p_fg = p  # [B,1,H,W]
            p_bg = 1.0 - p_fg
            two_maps = torch.cat([p_fg, p_bg], dim=1)  # [B,2,H,W]
            idx = two_maps.view(b, -1).argmax(dim=1)
            # decode index -> label, y, x
            per_map = H * W
            lab_idx = (idx // per_map).to(torch.int64)
            rem = (idx % per_map)
            y = (rem // W).to(torch.float32)
            x = (rem % W).to(torch.float32)
            xy = torch.stack([x, y], dim=1)
            lab = int(lab_idx.item())
        else:
            # fallback to default
            xy = soft_argmax_from_logits(logits, temperature=tau)[0]
            lab = int(label_logits.argmax(dim=1).item())

        prob = p[0, 0].detach().cpu().float().numpy()
        # Also return p_bg when two_map to allow caller to save both
        prob_bg = None
        if coord_label_mode == "two_map":
            prob_bg = (1.0 - p)[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, (prob if prob is not None else None), (prob_bg if prob_bg is not None else None)


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int, name_prefix: str = "heatmap") -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"{name_prefix}-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


def _list_images(images_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    out: List[str] = []
    for name in sorted(os.listdir(images_dir)):
        p = os.path.join(images_dir, name)
        if os.path.isdir(p):
            continue
        if os.path.splitext(name)[1].lower() in exts:
            out.append(p)
    return out


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def run_initial(args: argparse.Namespace) -> None:
    if not args.images_dir or not args.output_json:
        raise RuntimeError("--images_dir and --output_json are required for initial mode")
    if not args.masks_dir:
        raise RuntimeError("--masks_dir is required for initial mode to fill 'gt_mask' field")
    device, amp_enabled, autocast_device_type = _device_and_amp(args)
    model = _load_model(args.model_path, device)
    H, W = int(args.height), int(args.width)

    records: List[Dict] = []
    images = _list_images(args.images_dir)
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            images_iter = tqdm(images, desc="Initial predict")
        except Exception:
            images_iter = images
    else:
        images_iter = images

    for img_path in images_iter:
        rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=None, out_hw=(H, W))
        with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau, args.coord_label_mode)
        # scale back to native size if resized
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        # clamp to bounds
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        stem = _stem(img_path)
        # save heatmap for step 0
        _save_heatmap_image(prob, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h)
        gt_mask_path = os.path.join(args.masks_dir, f"{stem}.png")
        rec = {
            "image": os.path.abspath(img_path),
            "gt_mask": os.path.abspath(gt_mask_path),
            "sam_masks_dir": os.path.abspath(args.sam_dir) if args.sam_dir else None,
            "points": [[float(x), float(y)]],
            "labels": [int(lab)],
        }
        # keep sam_masks_dir only if provided
        if rec["sam_masks_dir"] is None:
            del rec["sam_masks_dir"]
        records.append(rec)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} records to {args.output_json}")


def run_append(args: argparse.Namespace) -> None:
    if not args.appendto_json:
        raise RuntimeError("--appendto_json required for append mode")
    if not os.path.isfile(args.appendto_json):
        raise FileNotFoundError(args.appendto_json)
    arr = json.loads(open(args.appendto_json, "r", encoding="utf-8").read().strip())
    if not isinstance(arr, list):
        raise RuntimeError("Append target must be a JSON array")
    device, amp_enabled, autocast_device_type = _device_and_amp(args)
    model = _load_model(args.model_path, device)
    H, W = int(args.height), int(args.width)

    updated = 0
    # progress bar
    rec_iter = arr
    pbar = None
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=len(arr), desc="Append predict")
        except Exception:
            pbar = None
    for rec in arr:
        if not isinstance(rec, dict):
            continue
        img_path = rec.get("image")
        sam_dir = rec.get("sam_masks_dir")
        pts = rec.get("points", [])
        labs = rec.get("labels", [])
        if not img_path:
            continue
        k = len(pts)
        # prev mask
        if k == 0:
            gray = None
        else:
            stem = _stem(img_path)
            prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
            gray = Image.open(prev_path).convert("L") if os.path.isfile(prev_path) else None
        rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=gray, out_hw=(H, W))
        with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            x, y, lab, prob = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau, args.coord_label_mode)
        # scale back and clamp
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h)
        rec["points"] = (pts or []) + [[x, y]]
        rec["labels"] = (labs or []) + [int(lab)]
        # sam_masks_dir should be set by external segmenter; do not inject here
        updated += 1
        if pbar is not None:
            pbar.update(1)
        elif updated % max(1, len(arr)//10 or 1) == 0:
            print(f"Processed {updated}/{len(arr)} records")

    if pbar is not None:
        pbar.close()

    with open(args.appendto_json, "w", encoding="utf-8") as f:
        json.dump(arr, f, indent=2, ensure_ascii=False)
    print(f"Appended next points to {args.appendto_json} (updated {updated} items)")


def run_debug(args: argparse.Namespace) -> None:
    if not args.debug_json:
        raise RuntimeError("--debug_json required for debug mode")
    if not args.debug_output_dir:
        raise RuntimeError("--debug_output_dir required for debug mode")
    arr = json.loads(open(args.debug_json, "r", encoding="utf-8").read().strip())
    if not isinstance(arr, list):
        raise RuntimeError("Debug JSON must be a JSON array")
    os.makedirs(args.debug_output_dir, exist_ok=True)

    num_generated = 0
    # optional progress over total steps
    total_steps = 0
    if args.progress:
        try:
            for rec in arr:
                if isinstance(rec, dict):
                    pts = rec.get("points", []) if isinstance(rec.get("points", []), list) else []
                    total_steps += len(pts)
        except Exception:
            total_steps = 0
    pbar = None
    if args.progress and total_steps > 0:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=total_steps, desc="Debug render")
        except Exception:
            pbar = None

    for rec in arr:
        if not isinstance(rec, dict):
            continue
        image_path = rec.get("image")
        sam_dir = rec.get("sam_masks_dir") or args.sam_dir
        points = rec.get("points", [])
        labels = rec.get("labels", [])
        if not (image_path and isinstance(points, list) and isinstance(labels, list)):
            continue
        if len(points) != len(labels):
            continue
        stem = _stem(image_path)
        for i, (pt, lb) in enumerate(zip(points, labels)):
            cur_path = os.path.join(sam_dir, stem, f"{i}.png")
            cur = Image.open(cur_path).convert("L") if os.path.isfile(cur_path) else None
            if cur is None:
                continue
            cur_u8 = np.array(cur, dtype=np.uint8)
            if i == 0:
                prev_u8 = np.zeros_like(cur_u8, dtype=np.uint8)
            else:
                prev_path = os.path.join(sam_dir, stem, f"{i-1}.png")
                prev_u8 = np.array(Image.open(prev_path).convert("L"), dtype=np.uint8) if os.path.isfile(prev_path) else np.zeros_like(cur_u8, dtype=np.uint8)
            try:
                subdir = os.path.join(args.debug_output_dir, stem)
                os.makedirs(subdir, exist_ok=True)
                out_path = os.path.join(subdir, f"dbg_{i}.png")
            except Exception:
                out_path = None  # type: ignore
            if _render_diff_panels_with_point is not None and out_path is not None:
                _render_diff_panels_with_point(prev_u8, cur_u8, (float(pt[0]), float(pt[1])), int(lb), out_path)
                num_generated += 1
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    print(f"Generated debug images for {num_generated} steps into {args.debug_output_dir}")


def main() -> None:
    args = parse_args()
    if args.output_json:
        run_initial(args)
        return
    if args.appendto_json:
        run_append(args)
        return
    if args.debug_json:
        run_debug(args)
        return


if __name__ == "__main__":
    main()



```

```python
#!/usr/bin/env python3

"""
Predict next SAM prompt point (position + binary label) using a trained heatmap model.

This script mirrors seg-rl/annotator/gen_point_jsonl_from_masks.py, but replaces the
EDT-based next-point computation with a model inference that takes only:
  - the RGB image, and
  - the current predicted mask (or zero mask at step 0)
and outputs the next prompt point and its label.

Modes:
  1) Initial generation (write new JSON array):
     - For each image under --images_dir, predict the first prompt (i=0) with a zero mask.
     - Write a JSON array with records like:
         {"image": "/abs/path.jpg", "sam_masks_dir": "/abs/sam_dir", "points": [[x0,y0]], "labels": [l0]}

  2) Append mode (update existing JSON array):
     - Read --appendto_json as a JSON array, for each record take current length k=len(points),
       load the latest predicted mask at {sam_dir}/{stem}/{k-1}.png (k>0; else zero), and predict
       the next point (xk, yk, lk). Append to arrays and write back in-place.

  3) Debug visualization (no file write except images):
     - Read --debug_json (JSON array) that already contains points+labels and a sam_masks_dir.
     - For each step i, load prev and cur predicted masks (i-1 and i; prev is zero for i=0) and
       call the same 3-panel visualization function used by the annotator to render panels with
       the provided point, saving to --debug_output_dir/{stem}/dbg_{i}.png.
     - Ground-truth masks are NOT used anywhere in this script (except indirectly if you supply
       them in your own JSON for reference; they are not required).

Examples:
  # Initial generation
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/heatmap_train-251001-optimized/model_epoch_185.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred_points-sft_e185-251018.jsonl

  python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred_points-e160-251019.jsonl

  # Append mode
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --appendto_json datasets/seg_r1_md/Task01_BrainTumour/pred_points-251001.jsonl

  python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --appendto_json outputs/braintumour/pred_points-e160-251019.jsonl

  # Debug visualization only
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --debug_json datasets/seg_r1_md/Task01_BrainTumour/pred_points-init.jsonl \
    --debug_output_dir outputs/braintumour/dbg_model_pred

Notes:
  - The model expects 4-channel input (RGB normalized + Gray normalized mask). This script
    normalizes RGB with ImageNet mean/std and mask with mean=0.5, std=0.5.
  - Device selection prefers CUDA, then MPS (Apple Silicon), otherwise CPU. AMP is only used on CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

try:
    # package context
    from .model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

import importlib.util as _importlib_util

def _import_render_helper() -> Optional[Callable]:
    """Dynamically import _render_diff_panels_with_point from annotator script by file path.

    This avoids package-name issues due to hyphen in 'seg-rl'.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # seg-rl
        target = os.path.join(base_dir, "annotator", "gen_point_jsonl_from_masks.py")
        if not os.path.isfile(target):
            return None
        spec = _importlib_util.spec_from_file_location("gen_point_jsonl_from_masks", target)
        if spec is None or spec.loader is None:
            return None
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        func = getattr(module, "_render_diff_panels_with_point", None)
        if callable(func):
            return func  # type: ignore[return-value]
    except Exception:
        return None
    return None

_render_diff_panels_with_point: Optional[Callable] = _import_render_helper()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict next SAM prompt point via trained heatmap model")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_json", type=str, help="Write a new JSON array with first predicted points")
    group.add_argument("--appendto_json", type=str, help="Append next predicted point to an existing JSON array")
    group.add_argument("--debug_json", type=str, help="Debug visualization only: read JSON array and render panels")

    p.add_argument("--model_path", type=str, required=False, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--images_dir", type=str, default=None, help="Directory of input images (initial mode)")
    p.add_argument("--masks_dir", type=str, default=None, help="Ground-truth masks directory (initial mode only, to fill 'gt_mask' field)")
    p.add_argument("--sam_dir", type=str, default=None, help="Directory of predicted SAM masks per step: {sam_dir}/{stem}/{k}.png (initial/debug mode)")
    # height/width: 0 means use native image size; if >0, will resize for inference
    p.add_argument("--height", type=int, default=0, help="0 = use native image height")
    p.add_argument("--width", type=int, default=0, help="0 = use native image width")
    p.add_argument("--tau", type=float, default=1.0, help="Softmax temperature for heatmap -> probability")
    # Coordinate/label selection strategy
    p.add_argument(
        "--coord_label_mode",
        type=str,
        choices=["decoupled", "argmax", "two_map"],
        default="decoupled",
        help=(
            "How to pick (x,y,label): decoupled = current default (soft-argmax coords + label head argmax); "
            "argmax = hard argmax coords + label head argmax; two_map = derive foreground/background maps from heatmap "
            "(p_fg = softmax(logits/tau), p_bg = 1 - p_fg), pick global maximum over 2xHxW to jointly decide label and pixel."
        ),
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--debug_output_dir", type=str, default=None)
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
    return p.parse_args()


def _device_and_amp(args: argparse.Namespace) -> Tuple[torch.device, bool, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    amp_enabled = args.amp and (device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    return device, amp_enabled, autocast_device_type


def _load_model(model_path: str, device: torch.device) -> PointHeatmapModel:
    cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
    model = PointHeatmapModel(cfg).to(device)
    ckpt = torch.load(model_path, map_location="cpu") if model_path and os.path.isfile(model_path) else None
    if ckpt is not None:
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _to_tensor_separate(image_path: str, gray_mask: Optional[Image.Image], out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Prepare separate RGB and grayscale tensors for the model; returns (rgb, gray, (orig_h, orig_w))."""
    im0 = Image.open(image_path).convert("RGB")
    orig_w, orig_h = im0.size
    H, W = out_hw
    # decide resize size
    if H <= 0 or W <= 0:
        H, W = orig_h, orig_w
    im = im0.resize((W, H), resample=Image.BILINEAR) if (W, H) != (orig_w, orig_h) else im0
    if gray_mask is None:
        gray0 = Image.new("L", (orig_w, orig_h), 0)
    else:
        gray0 = gray_mask.convert("L")
    gray = gray0.resize((W, H), resample=Image.NEAREST) if (W, H) != (orig_w, orig_h) else gray0
    import torchvision.transforms.functional as TF
    rgb_t = TF.to_tensor(im)
    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    g_t = TF.to_tensor(gray)
    g_t = (g_t - 0.5) / 0.5
    return rgb_t.unsqueeze(0), g_t.unsqueeze(0), (orig_h, orig_w)  # [1,3,H,W], [1,1,H,W], (orig_h,orig_w)


def _predict_point(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float, coord_label_mode: str) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        T = max(float(tau), 1e-6)
        p = F.softmax(logits.view(b, -1) / T, dim=1).view(b, 1, H, W)

        if coord_label_mode == "decoupled":
            xy = soft_argmax_from_logits(logits, temperature=tau)[0]
            lab = int(label_logits.argmax(dim=1).item())
        elif coord_label_mode == "argmax":
            # Hard argmax over pixels for coordinates
            idx = logits.view(b, -1).argmax(dim=1)
            y = (idx // W).to(torch.float32)
            x = (idx % W).to(torch.float32)
            xy = torch.stack([x, y], dim=1)[0]
            lab = int(label_logits.argmax(dim=1).item())
        elif coord_label_mode == "two_map":
            # Minimal heuristic: build two maps p_fg = p, p_bg = 1 - p; pick global maximum over both
            p_fg = p  # [B,1,H,W]
            p_bg = 1.0 - p_fg
            two_maps = torch.cat([p_fg, p_bg], dim=1)  # [B,2,H,W]
            idx = two_maps.view(b, -1).argmax(dim=1)
            # decode index -> label, y, x
            per_map = H * W
            lab_idx = (idx // per_map).to(torch.int64)
            rem = (idx % per_map)
            y = (rem // W).to(torch.float32)
            x = (rem % W).to(torch.float32)
            xy = torch.stack([x, y], dim=1)
            lab = int(lab_idx.item())
        else:
            # fallback to default
            xy = soft_argmax_from_logits(logits, temperature=tau)[0]
            lab = int(label_logits.argmax(dim=1).item())

        prob = p[0, 0].detach().cpu().float().numpy()
        # Also return p_bg when two_map to allow caller to save both
        prob_bg = None
        if coord_label_mode == "two_map":
            prob_bg = (1.0 - p)[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, (prob if prob is not None else None), (prob_bg if prob_bg is not None else None)


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int, name_prefix: str = "heatmap") -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"{name_prefix}-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


def _list_images(images_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    out: List[str] = []
    for name in sorted(os.listdir(images_dir)):
        p = os.path.join(images_dir, name)
        if os.path.isdir(p):
            continue
        if os.path.splitext(name)[1].lower() in exts:
            out.append(p)
    return out


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def run_initial(args: argparse.Namespace) -> None:
    if not args.images_dir or not args.output_json:
        raise RuntimeError("--images_dir and --output_json are required for initial mode")
    if not args.masks_dir:
        raise RuntimeError("--masks_dir is required for initial mode to fill 'gt_mask' field")
    device, amp_enabled, autocast_device_type = _device_and_amp(args)
    model = _load_model(args.model_path, device)
    H, W = int(args.height), int(args.width)

    records: List[Dict] = []
    images = _list_images(args.images_dir)
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            images_iter = tqdm(images, desc="Initial predict")
        except Exception:
            images_iter = images
    else:
        images_iter = images

    for img_path in images_iter:
        rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=None, out_hw=(H, W))
        with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            x, y, lab, prob_fg, prob_bg = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau, args.coord_label_mode)
        # scale back to native size if resized
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        # clamp to bounds
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        stem = _stem(img_path)
        # save heatmap for step 0
        _save_heatmap_image(prob_fg, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h, name_prefix="heatmap")
        if args.coord_label_mode == "two_map" and prob_bg is not None:
            _save_heatmap_image(prob_bg, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h, name_prefix="p_bg")
        gt_mask_path = os.path.join(args.masks_dir, f"{stem}.png")
        rec = {
            "image": os.path.abspath(img_path),
            "gt_mask": os.path.abspath(gt_mask_path),
            "sam_masks_dir": os.path.abspath(args.sam_dir) if args.sam_dir else None,
            "points": [[float(x), float(y)]],
            "labels": [int(lab)],
        }
        # keep sam_masks_dir only if provided
        if rec["sam_masks_dir"] is None:
            del rec["sam_masks_dir"]
        records.append(rec)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} records to {args.output_json}")


def run_append(args: argparse.Namespace) -> None:
    if not args.appendto_json:
        raise RuntimeError("--appendto_json required for append mode")
    if not os.path.isfile(args.appendto_json):
        raise FileNotFoundError(args.appendto_json)
    arr = json.loads(open(args.appendto_json, "r", encoding="utf-8").read().strip())
    if not isinstance(arr, list):
        raise RuntimeError("Append target must be a JSON array")
    device, amp_enabled, autocast_device_type = _device_and_amp(args)
    model = _load_model(args.model_path, device)
    H, W = int(args.height), int(args.width)

    updated = 0
    # progress bar
    rec_iter = arr
    pbar = None
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=len(arr), desc="Append predict")
        except Exception:
            pbar = None
    for rec in arr:
        if not isinstance(rec, dict):
            continue
        img_path = rec.get("image")
        sam_dir = rec.get("sam_masks_dir")
        pts = rec.get("points", [])
        labs = rec.get("labels", [])
        if not img_path:
            continue
        k = len(pts)
        # prev mask
        if k == 0:
            gray = None
        else:
            stem = _stem(img_path)
            prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
            gray = Image.open(prev_path).convert("L") if os.path.isfile(prev_path) else None
        rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=gray, out_hw=(H, W))
        with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            x, y, lab, prob_fg, prob_bg = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau, args.coord_label_mode)
        # scale back and clamp
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        _save_heatmap_image(prob_fg, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h, name_prefix="heatmap")
        if args.coord_label_mode == "two_map" and prob_bg is not None:
            _save_heatmap_image(prob_bg, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h, name_prefix="p_bg")
        rec["points"] = (pts or []) + [[x, y]]
        rec["labels"] = (labs or []) + [int(lab)]
        # sam_masks_dir should be set by external segmenter; do not inject here
        updated += 1
        if pbar is not None:
            pbar.update(1)
        elif updated % max(1, len(arr)//10 or 1) == 0:
            print(f"Processed {updated}/{len(arr)} records")

    if pbar is not None:
        pbar.close()

    with open(args.appendto_json, "w", encoding="utf-8") as f:
        json.dump(arr, f, indent=2, ensure_ascii=False)
    print(f"Appended next points to {args.appendto_json} (updated {updated} items)")


def run_debug(args: argparse.Namespace) -> None:
    if not args.debug_json:
        raise RuntimeError("--debug_json required for debug mode")
    if not args.debug_output_dir:
        raise RuntimeError("--debug_output_dir required for debug mode")
    arr = json.loads(open(args.debug_json, "r", encoding="utf-8").read().strip())
    if not isinstance(arr, list):
        raise RuntimeError("Debug JSON must be a JSON array")
    os.makedirs(args.debug_output_dir, exist_ok=True)

    num_generated = 0
    # optional progress over total steps
    total_steps = 0
    if args.progress:
        try:
            for rec in arr:
                if isinstance(rec, dict):
                    pts = rec.get("points", []) if isinstance(rec.get("points", []), list) else []
                    total_steps += len(pts)
        except Exception:
            total_steps = 0
    pbar = None
    if args.progress and total_steps > 0:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=total_steps, desc="Debug render")
        except Exception:
            pbar = None

    for rec in arr:
        if not isinstance(rec, dict):
            continue
        image_path = rec.get("image")
        sam_dir = rec.get("sam_masks_dir") or args.sam_dir
        points = rec.get("points", [])
        labels = rec.get("labels", [])
        if not (image_path and isinstance(points, list) and isinstance(labels, list)):
            continue
        if len(points) != len(labels):
            continue
        stem = _stem(image_path)
        for i, (pt, lb) in enumerate(zip(points, labels)):
            cur_path = os.path.join(sam_dir, stem, f"{i}.png")
            cur = Image.open(cur_path).convert("L") if os.path.isfile(cur_path) else None
            if cur is None:
                continue
            cur_u8 = np.array(cur, dtype=np.uint8)
            if i == 0:
                prev_u8 = np.zeros_like(cur_u8, dtype=np.uint8)
            else:
                prev_path = os.path.join(sam_dir, stem, f"{i-1}.png")
                prev_u8 = np.array(Image.open(prev_path).convert("L"), dtype=np.uint8) if os.path.isfile(prev_path) else np.zeros_like(cur_u8, dtype=np.uint8)
            try:
                subdir = os.path.join(args.debug_output_dir, stem)
                os.makedirs(subdir, exist_ok=True)
                out_path = os.path.join(subdir, f"dbg_{i}.png")
            except Exception:
                out_path = None  # type: ignore
            if _render_diff_panels_with_point is not None and out_path is not None:
                _render_diff_panels_with_point(prev_u8, cur_u8, (float(pt[0]), float(pt[1])), int(lb), out_path)
                num_generated += 1
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    print(f"Generated debug images for {num_generated} steps into {args.debug_output_dir}")


def main() -> None:
    args = parse_args()
    if args.output_json:
        run_initial(args)
        return
    if args.appendto_json:
        run_append(args)
        return
    if args.debug_json:
        run_debug(args)
        return


if __name__ == "__main__":
    main()



```

```python
#!/usr/bin/env python3

"""
Predict next SAM prompt point (position + binary label) using a trained heatmap model.

This script mirrors seg-rl/annotator/gen_point_jsonl_from_masks.py, but replaces the
EDT-based next-point computation with a model inference that takes only:
  - the RGB image, and
  - the current predicted mask (or zero mask at step 0)
and outputs the next prompt point and its label.

Modes:
  1) Initial generation (write new JSON array):
     - For each image under --images_dir, predict the first prompt (i=0) with a zero mask.
     - Write a JSON array with records like:
         {"image": "/abs/path.jpg", "sam_masks_dir": "/abs/sam_dir", "points": [[x0,y0]], "labels": [l0]}

  2) Append mode (update existing JSON array):
     - Read --appendto_json as a JSON array, for each record take current length k=len(points),
       load the latest predicted mask at {sam_dir}/{stem}/{k-1}.png (k>0; else zero), and predict
       the next point (xk, yk, lk). Append to arrays and write back in-place.

  3) Debug visualization (no file write except images):
     - Read --debug_json (JSON array) that already contains points+labels and a sam_masks_dir.
     - For each step i, load prev and cur predicted masks (i-1 and i; prev is zero for i=0) and
       call the same 3-panel visualization function used by the annotator to render panels with
       the provided point, saving to --debug_output_dir/{stem}/dbg_{i}.png.
     - Ground-truth masks are NOT used anywhere in this script (except indirectly if you supply
       them in your own JSON for reference; they are not required).

Examples:
  # Initial generation
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/heatmap_train-251001-optimized/model_epoch_185.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred_points-sft_e185-251018.jsonl

  python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_json outputs/braintumour/pred_points-e160-251019.jsonl

  # Append mode
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --appendto_json datasets/seg_r1_md/Task01_BrainTumour/pred_points-251001.jsonl

  python -m seg-rl.heatmap.predict_next_point_from_model \
    --model_path outputs/braintumour/points_predictor-251001-160epochs.pt \
    --appendto_json outputs/braintumour/pred_points-e160-251019.jsonl

  # Debug visualization only
  /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.predict_next_point_from_model \
    --debug_json datasets/seg_r1_md/Task01_BrainTumour/pred_points-init.jsonl \
    --debug_output_dir outputs/braintumour/dbg_model_pred

Notes:
  - The model expects 4-channel input (RGB normalized + Gray normalized mask). This script
    normalizes RGB with ImageNet mean/std and mask with mean=0.5, std=0.5.
  - Device selection prefers CUDA, then MPS (Apple Silicon), otherwise CPU. AMP is only used on CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

try:
    # package context
    from .model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from .utils import heatmap_to_pil
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits
    from heatmap.utils import heatmap_to_pil

import importlib.util as _importlib_util

def _import_render_helper() -> Optional[Callable]:
    """Dynamically import _render_diff_panels_with_point from annotator script by file path.

    This avoids package-name issues due to hyphen in 'seg-rl'.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # seg-rl
        target = os.path.join(base_dir, "annotator", "gen_point_jsonl_from_masks.py")
        if not os.path.isfile(target):
            return None
        spec = _importlib_util.spec_from_file_location("gen_point_jsonl_from_masks", target)
        if spec is None or spec.loader is None:
            return None
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        func = getattr(module, "_render_diff_panels_with_point", None)
        if callable(func):
            return func  # type: ignore[return-value]
    except Exception:
        return None
    return None

_render_diff_panels_with_point: Optional[Callable] = _import_render_helper()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict next SAM prompt point via trained heatmap model")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_json", type=str, help="Write a new JSON array with first predicted points")
    group.add_argument("--appendto_json", type=str, help="Append next predicted point to an existing JSON array")
    group.add_argument("--debug_json", type=str, help="Debug visualization only: read JSON array and render panels")

    p.add_argument("--model_path", type=str, required=False, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--images_dir", type=str, default=None, help="Directory of input images (initial mode)")
    p.add_argument("--masks_dir", type=str, default=None, help="Ground-truth masks directory (initial mode only, to fill 'gt_mask' field)")
    p.add_argument("--sam_dir", type=str, default=None, help="Directory of predicted SAM masks per step: {sam_dir}/{stem}/{k}.png (initial/debug mode)")
    # height/width: 0 means use native image size; if >0, will resize for inference
    p.add_argument("--height", type=int, default=0, help="0 = use native image height")
    p.add_argument("--width", type=int, default=0, help="0 = use native image width")
    p.add_argument("--tau", type=float, default=1.0, help="Softmax temperature for heatmap -> probability")
    # Coordinate/label selection strategy
    p.add_argument(
        "--coord_label_mode",
        type=str,
        choices=["decoupled", "argmax", "two_map"],
        default="decoupled",
        help=(
            "How to pick (x,y,label): decoupled = current default (soft-argmax coords + label head argmax); "
            "argmax = hard argmax coords + label head argmax; two_map = derive foreground/background maps from heatmap "
            "(p_fg = softmax(logits/tau), p_bg = 1 - p_fg), pick global maximum over 2xHxW to jointly decide label and pixel."
        ),
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--debug_output_dir", type=str, default=None)
    # Save probability heatmaps alongside SAM masks: {output_dir}/{stem}/heatmap-{i}.png
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save per-step heatmaps (under {stem}/heatmap-{i}.png)")
    return p.parse_args()


def _device_and_amp(args: argparse.Namespace) -> Tuple[torch.device, bool, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    amp_enabled = args.amp and (device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    return device, amp_enabled, autocast_device_type


def _load_model(model_path: str, device: torch.device) -> PointHeatmapModel:
    cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
    model = PointHeatmapModel(cfg).to(device)
    ckpt = torch.load(model_path, map_location="cpu") if model_path and os.path.isfile(model_path) else None
    if ckpt is not None:
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _to_tensor_separate(image_path: str, gray_mask: Optional[Image.Image], out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Prepare separate RGB and grayscale tensors for the model; returns (rgb, gray, (orig_h, orig_w))."""
    im0 = Image.open(image_path).convert("RGB")
    orig_w, orig_h = im0.size
    H, W = out_hw
    # decide resize size
    if H <= 0 or W <= 0:
        H, W = orig_h, orig_w
    im = im0.resize((W, H), resample=Image.BILINEAR) if (W, H) != (orig_w, orig_h) else im0
    if gray_mask is None:
        gray0 = Image.new("L", (orig_w, orig_h), 0)
    else:
        gray0 = gray_mask.convert("L")
    gray = gray0.resize((W, H), resample=Image.NEAREST) if (W, H) != (orig_w, orig_h) else gray0
    import torchvision.transforms.functional as TF
    rgb_t = TF.to_tensor(im)
    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    g_t = TF.to_tensor(gray)
    g_t = (g_t - 0.5) / 0.5
    return rgb_t.unsqueeze(0), g_t.unsqueeze(0), (orig_h, orig_w)  # [1,3,H,W], [1,1,H,W], (orig_h,orig_w)


def _predict_point(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float, coord_label_mode: str) -> Tuple[float, float, int, np.ndarray]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        # probability heatmap for saving: softmax over pixels
        b, _, H, W = logits.shape
        T = max(float(tau), 1e-6)
        p = F.softmax(logits.view(b, -1) / T, dim=1).view(b, 1, H, W)

        if coord_label_mode == "decoupled":
            xy = soft_argmax_from_logits(logits, temperature=tau)[0]
            lab = int(label_logits.argmax(dim=1).item())
        elif coord_label_mode == "argmax":
            # Hard argmax over pixels for coordinates
            idx = logits.view(b, -1).argmax(dim=1)
            y = (idx // W).to(torch.float32)
            x = (idx % W).to(torch.float32)
            xy = torch.stack([x, y], dim=1)[0]
            lab = int(label_logits.argmax(dim=1).item())
        elif coord_label_mode == "two_map":
            # Minimal heuristic: build two maps p_fg = p, p_bg = 1 - p; pick global maximum over both
            p_fg = p  # [B,1,H,W]
            p_bg = 1.0 - p_fg
            two_maps = torch.cat([p_fg, p_bg], dim=1)  # [B,2,H,W]
            idx = two_maps.view(b, -1).argmax(dim=1)
            # decode index -> label, y, x
            per_map = H * W
            lab_idx = (idx // per_map).to(torch.int64)
            rem = (idx % per_map)
            y = (rem // W).to(torch.float32)
            x = (rem % W).to(torch.float32)
            xy = torch.stack([x, y], dim=1)
            lab = int(lab_idx.item())
        else:
            # fallback to default
            xy = soft_argmax_from_logits(logits, temperature=tau)[0]
            lab = int(label_logits.argmax(dim=1).item())

        prob = p[0, 0].detach().cpu().float().numpy()
        # Also return p_bg when two_map to allow caller to save both
        prob_bg = None
        if coord_label_mode == "two_map":
            prob_bg = (1.0 - p)[0, 0].detach().cpu().float().numpy()
    return float(xy[0].item()), float(xy[1].item()), lab, (prob if prob is not None else None), (prob_bg if prob_bg is not None else None)


def _save_heatmap_image(prob: np.ndarray, out_dir: Optional[str], stem: str, step_idx: int, orig_w: int, orig_h: int, name_prefix: str = "heatmap") -> None:
    if not out_dir:
        return
    try:
        hm_img = heatmap_to_pil(prob)
        if hm_img.size != (orig_w, orig_h):
            hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        subdir = os.path.join(out_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"{name_prefix}-{step_idx}.png")
        hm_img.save(out_path)
    except Exception:
        pass


def _list_images(images_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    out: List[str] = []
    for name in sorted(os.listdir(images_dir)):
        p = os.path.join(images_dir, name)
        if os.path.isdir(p):
            continue
        if os.path.splitext(name)[1].lower() in exts:
            out.append(p)
    return out


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def run_initial(args: argparse.Namespace) -> None:
    if not args.images_dir or not args.output_json:
        raise RuntimeError("--images_dir and --output_json are required for initial mode")
    if not args.masks_dir:
        raise RuntimeError("--masks_dir is required for initial mode to fill 'gt_mask' field")
    device, amp_enabled, autocast_device_type = _device_and_amp(args)
    model = _load_model(args.model_path, device)
    H, W = int(args.height), int(args.width)

    records: List[Dict] = []
    images = _list_images(args.images_dir)
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            images_iter = tqdm(images, desc="Initial predict")
        except Exception:
            images_iter = images
    else:
        images_iter = images

    for img_path in images_iter:
        rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=None, out_hw=(H, W))
        with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            x, y, lab, prob_fg, prob_bg = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau, args.coord_label_mode)
        # scale back to native size if resized
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        # clamp to bounds
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        stem = _stem(img_path)
        # save heatmap for step 0
        # Save probability maps
        if args.coord_label_mode == "two_map" and prob_bg is not None:
            _save_heatmap_image(prob_fg, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h, name_prefix="p_fg")
            _save_heatmap_image(prob_bg, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h, name_prefix="p_bg")
        else:
            _save_heatmap_image(prob_fg, args.output_dir or args.sam_dir, stem, 0, orig_w, orig_h, name_prefix="heatmap")
        gt_mask_path = os.path.join(args.masks_dir, f"{stem}.png")
        rec = {
            "image": os.path.abspath(img_path),
            "gt_mask": os.path.abspath(gt_mask_path),
            "sam_masks_dir": os.path.abspath(args.sam_dir) if args.sam_dir else None,
            "points": [[float(x), float(y)]],
            "labels": [int(lab)],
        }
        # keep sam_masks_dir only if provided
        if rec["sam_masks_dir"] is None:
            del rec["sam_masks_dir"]
        records.append(rec)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} records to {args.output_json}")


def run_append(args: argparse.Namespace) -> None:
    if not args.appendto_json:
        raise RuntimeError("--appendto_json required for append mode")
    if not os.path.isfile(args.appendto_json):
        raise FileNotFoundError(args.appendto_json)
    arr = json.loads(open(args.appendto_json, "r", encoding="utf-8").read().strip())
    if not isinstance(arr, list):
        raise RuntimeError("Append target must be a JSON array")
    device, amp_enabled, autocast_device_type = _device_and_amp(args)
    model = _load_model(args.model_path, device)
    H, W = int(args.height), int(args.width)

    updated = 0
    # progress bar
    rec_iter = arr
    pbar = None
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=len(arr), desc="Append predict")
        except Exception:
            pbar = None
    for rec in arr:
        if not isinstance(rec, dict):
            continue
        img_path = rec.get("image")
        sam_dir = rec.get("sam_masks_dir")
        pts = rec.get("points", [])
        labs = rec.get("labels", [])
        if not img_path:
            continue
        k = len(pts)
        # prev mask
        if k == 0:
            gray = None
        else:
            stem = _stem(img_path)
            prev_path = os.path.join(sam_dir, stem, f"{k-1}.png")
            gray = Image.open(prev_path).convert("L") if os.path.isfile(prev_path) else None
        rgb_tensor, gray_tensor, (orig_h, orig_w) = _to_tensor_separate(img_path, gray_mask=gray, out_hw=(H, W))
        with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
            x, y, lab, prob_fg, prob_bg = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau, args.coord_label_mode)
        # scale back and clamp
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
        # Save heatmap for next step k = len(pts)
        stem = _stem(img_path)
        step_k = len(pts)
        if args.coord_label_mode == "two_map" and prob_bg is not None:
            _save_heatmap_image(prob_fg, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h, name_prefix="p_fg")
            _save_heatmap_image(prob_bg, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h, name_prefix="p_bg")
        else:
            _save_heatmap_image(prob_fg, args.output_dir or sam_dir, stem, step_k, orig_w, orig_h, name_prefix="heatmap")
        rec["points"] = (pts or []) + [[x, y]]
        rec["labels"] = (labs or []) + [int(lab)]
        # sam_masks_dir should be set by external segmenter; do not inject here
        updated += 1
        if pbar is not None:
            pbar.update(1)
        elif updated % max(1, len(arr)//10 or 1) == 0:
            print(f"Processed {updated}/{len(arr)} records")

    if pbar is not None:
        pbar.close()

    with open(args.appendto_json, "w", encoding="utf-8") as f:
        json.dump(arr, f, indent=2, ensure_ascii=False)
    print(f"Appended next points to {args.appendto_json} (updated {updated} items)")


def run_debug(args: argparse.Namespace) -> None:
    if not args.debug_json:
        raise RuntimeError("--debug_json required for debug mode")
    if not args.debug_output_dir:
        raise RuntimeError("--debug_output_dir required for debug mode")
    arr = json.loads(open(args.debug_json, "r", encoding="utf-8").read().strip())
    if not isinstance(arr, list):
        raise RuntimeError("Debug JSON must be a JSON array")
    os.makedirs(args.debug_output_dir, exist_ok=True)

    num_generated = 0
    # optional progress over total steps
    total_steps = 0
    if args.progress:
        try:
            for rec in arr:
                if isinstance(rec, dict):
                    pts = rec.get("points", []) if isinstance(rec.get("points", []), list) else []
                    total_steps += len(pts)
        except Exception:
            total_steps = 0
    pbar = None
    if args.progress and total_steps > 0:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=total_steps, desc="Debug render")
        except Exception:
            pbar = None

    for rec in arr:
        if not isinstance(rec, dict):
            continue
        image_path = rec.get("image")
        sam_dir = rec.get("sam_masks_dir") or args.sam_dir
        points = rec.get("points", [])
        labels = rec.get("labels", [])
        if not (image_path and isinstance(points, list) and isinstance(labels, list)):
            continue
        if len(points) != len(labels):
            continue
        stem = _stem(image_path)
        for i, (pt, lb) in enumerate(zip(points, labels)):
            cur_path = os.path.join(sam_dir, stem, f"{i}.png")
            cur = Image.open(cur_path).convert("L") if os.path.isfile(cur_path) else None
            if cur is None:
                continue
            cur_u8 = np.array(cur, dtype=np.uint8)
            if i == 0:
                prev_u8 = np.zeros_like(cur_u8, dtype=np.uint8)
            else:
                prev_path = os.path.join(sam_dir, stem, f"{i-1}.png")
                prev_u8 = np.array(Image.open(prev_path).convert("L"), dtype=np.uint8) if os.path.isfile(prev_path) else np.zeros_like(cur_u8, dtype=np.uint8)
            try:
                subdir = os.path.join(args.debug_output_dir, stem)
                os.makedirs(subdir, exist_ok=True)
                out_path = os.path.join(subdir, f"dbg_{i}.png")
            except Exception:
                out_path = None  # type: ignore
            if _render_diff_panels_with_point is not None and out_path is not None:
                _render_diff_panels_with_point(prev_u8, cur_u8, (float(pt[0]), float(pt[1])), int(lb), out_path)
                num_generated += 1
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    print(f"Generated debug images for {num_generated} steps into {args.debug_output_dir}")


def main() -> None:
    args = parse_args()
    if args.output_json:
        run_initial(args)
        return
    if args.appendto_json:
        run_append(args)
        return
    if args.debug_json:
        run_debug(args)
        return


if __name__ == "__main__":
    main()



```

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

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251002.jsonl

  # 2) 追加模式（为每条记录计算后续的第2个及以后提示点）
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl

  # 3) 调试模式（生成调试图像）
  # 若 sam2_segment_from_points.py 已生成 heatmap-{i}.png，本脚本会在调试图中追加第4幅：
  # “概率热力图 + 当前提示点”，用于核对提示点与热区的一致性。
  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --debug_json outputs/braintumour/pred_251023.jsonl \
    --debug_output_dir outputs/braintumour/dbg_gen_points

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
# from PIL import Image, ImageDraw
from PIL import Image as _PIL
from scipy.ndimage import label, distance_transform_edt
import cv2  # type: ignore

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create/Append deepest-point prompts from masks using EDT")
    # 互斥：首写输出 vs 追加到已存在的JSON数组
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_jsonl", type=str, help="Path to write JSONL lines for first prompts")
    group.add_argument("--appendto_jsonl", type=str, help="Path to existing JSON array to append next prompts")
    group.add_argument("--debug_json", type=str, help="Path to JSON array to read for debug visualization only")

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
    p.add_argument("--debug_output_dir", type=str, default=None,
                   help="If set, write per-sample debug images for diff component selection")
    return p.parse_args()


def list_files_with_exts(directory: str, allowed_exts: Tuple[str, ...]) -> List[str]:
    directory = to_abs(directory) or directory
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


def to_abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path) if not os.path.isabs(path) else path


def _find_latest_mask_path(sam_masks_dir: str, image_stem: str, current_points_len: int) -> Optional[str]:
    """返回该图像在sam_masks_dir下的最新预测mask路径。
    优先尝试 sam_masks_dir/<stem>/<current_points_len-1>.png；若不存在，
    则在该目录下寻找最大数字文件名的png。
    """
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

from scipy import ndimage as ndi

def farthest_point_from_boundary(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    # mask: 二值数组，区域为 True/1，背景为 False/0
    mask = (mask > 0)
    dist = ndi.distance_transform_edt(mask)  # 区域内每点到背景的欧氏距离
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    # r = float(dist[y, x])  # 到边界的最大最小距离
    # return (y, x), r
    return (float(x), float(y))

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

def largest_diff_component_representative(A, B, connectivity=2, min_area=0, dbg_out_path: Optional[str] = None):
    # A, B: 2D uint8 arrays, values {0,1} 或 {0,255}
    A = (A > 0).astype(np.uint8)
    B = (B > 0).astype(np.uint8)
    ### C = A.astype(np.int16) - B.astype(np.int16)  # {-1, 0, +1}
    C = A.astype(np.int8) - B.astype(np.int8)  # {-1, 0, +1}

    regions = []
    # 8 邻域结构元（connectivity=2 表示 8 邻域；=1 表示 4 邻域）
    st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    for sign, mask in ((1, C > 0), (-1, C < 0)):
        if not mask.any():
            continue
        lab, n = ndi.label(mask, structure=st)
        if n == 0:
            continue
        areas = np.bincount(lab.ravel())[1:]  # 忽略背景0
        if min_area > 0:
            keep_ids = np.where(areas >= min_area)[0] + 1
            if keep_ids.size == 0:
                continue
            mask2 = np.isin(lab, keep_ids)
            lab, n = ndi.label(mask2, structure=st)
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

    # centroid = farthest_point_from_boundary(comp)
    # # centroid = compute_centroid(comp)
    # if centroid is None:
    #     return None
    D = distance_transform_edt(comp)
    (centroidY, centroidX) = np.unravel_index(np.argmax(D), D.shape)

    # 调试可视化（改用统一的渲染函数，保持风格一致）
    if dbg_out_path:
        try:
            _render_diff_panels_with_point(
                A=A,
                B=B,
                point_xy=(float(centroidX), float(centroidY)),
                label=1 if int(sign) > 0 else 0,
                dbg_out_path=dbg_out_path,
                connectivity=connectivity,
            )
        except Exception:
            pass

    return dict(sign=sign, area=area, y=int(centroidY), x=int(centroidX), max_radius=float(D[centroidY, centroidX]))


def _render_diff_panels_with_point(
    A: np.ndarray,
    B: np.ndarray,
    point_xy: Tuple[float, float],
    label: int,
    dbg_out_path: Optional[str],
    connectivity: int = 2,
    base_image_bgr: Optional[np.ndarray] = None,
    heatmap_rgb: Optional[np.ndarray] = None,
    heatmap_fg_rgb: Optional[np.ndarray] = None,
    heatmap_bg_rgb: Optional[np.ndarray] = None,
) -> None:
    """Render 3-panel debug image reusing the visualization style in
    largest_diff_component_representative, but mark a provided point instead of
    the recomputed centroid.
    - Panel 1: A (green) vs B (red) overlay
    - Panel 2: signed diff C (positive green, negative red)
    - Panel 3: connected components colored, with marker at point_xy
    - Panel 4 (optional): probability heatmap (RGB) + current point marker
    """
    rng = np.random.default_rng(12345)
    try:
        if dbg_out_path is None:
            return
        h, w = A.shape[:2]
        # Panel 1: A/B overlay (A=green, B=red), optionally blended on base image
        overlay1 = np.zeros((h, w, 3), dtype=np.uint8)
        overlay1[..., 1] = (A > 0).astype(np.uint8) * 255  # G
        overlay1[..., 2] = (B > 0).astype(np.uint8) * 255  # R
        if base_image_bgr is not None:
            base = base_image_bgr
            if base.ndim == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            if base.shape[:2] != (h, w):
                base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
            panel1 = cv2.addWeighted(base, 1.0, overlay1, 0.6, 0.0)
        else:
            bg = np.zeros_like(overlay1)
            panel1 = cv2.addWeighted(bg, 1.0, overlay1, 0.6, 0.0)

        # Panel 2: signed diff C (positive green, negative red)
        C = A.astype(np.int16) - B.astype(np.int16)
        panel2 = np.zeros((h, w, 3), dtype=np.uint8)
        pos = (C > 0)
        neg = (C < 0)
        panel2[pos, 1] = 255
        panel2[neg, 2] = 255

        # Panel 3: connected components colored + provided point marker
        panel3 = np.zeros((h, w, 3), dtype=np.uint8)
        st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
        lab, n = ndi.label(C, structure=st)
        if n != 0:
            colors = (rng.integers(0, 256, size=(n, 3))).astype(np.uint8)
            colors = np.clip(colors + 100, 0, 255)
            for i in range(1, n + 1):
                panel3[lab == i] = colors[i - 1]

        # draw marker at provided (x,y): foreground -> caret '^', background -> 'X'
        cx = int(round(float(point_xy[0])))
        cy = int(round(float(point_xy[1])))
        def _draw_cross(img, x, y, size=6, color=(255,255,255), thickness=2):
            cv2.line(img, (x - size, y - size), (x + size, y + size), (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y + size), (x + size, y - size), (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness, lineType=cv2.LINE_AA)
        def _draw_caret(img, x, y, size=6, color=(255,255,255), thickness=2):
            p_top = (int(x), int(y - size))
            p_left = (int(x - size), int(y + size))
            p_right = (int(x + size), int(y + size))
            cv2.line(img, p_left, p_top, (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, p_right, p_top, (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, p_left, p_top, color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p_right, p_top, color, thickness, lineType=cv2.LINE_AA)
        if int(label) > 0:
            _draw_caret(panel3, cx, cy, size=6, color=(255,255,255), thickness=2)
        else:
            _draw_cross(panel3, cx, cy, size=6, color=(255,255,255), thickness=2)

        # Compute and draw metrics comparing B (pred) vs A (gt) on panel1
        def _compute_metrics(gt_u8: np.ndarray, pr_u8: np.ndarray) -> dict:
            gt = (gt_u8 > 0)
            pr = (pr_u8 > 0)
            tp = int(np.logical_and(gt, pr).sum())
            fp = int(np.logical_and(~gt, pr).sum())
            fn = int(np.logical_and(gt, ~pr).sum())
            denom_dice = 2 * tp + fp + fn
            dice = (2 * tp / denom_dice) if denom_dice > 0 else 1.0
            denom_iou = tp + fp + fn
            iou = (tp / denom_iou) if denom_iou > 0 else 1.0
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Lightweight S-measure implementation adapted for binary masks
            def _s_measure_binary(pred: np.ndarray, gt: np.ndarray) -> float:
                pred_f = pred.astype(np.float64)
                gt_f = gt.astype(np.float64)
                y_mean = gt_f.mean()
                if y_mean == 0:
                    return float(1.0 - pred_f.mean())
                if y_mean == 1:
                    return float(pred_f.mean())

                def s_object(p: np.ndarray, g: np.ndarray) -> float:
                    vals = p[g == 1]
                    if vals.size == 0:
                        return 0.0
                    mx = float(vals.mean())
                    sx = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                    return float(2 * mx / (mx * mx + 1 + sx + np.spacing(1)))

                def ssim(a: np.ndarray, b: np.ndarray) -> float:
                    h_, w_ = a.shape
                    N = h_ * w_
                    if N <= 1:
                        return 1.0
                    mx = float(a.mean()); my = float(b.mean())
                    sx = float(((a - mx) ** 2).sum() / (N - 1))
                    sy = float(((b - my) ** 2).sum() / (N - 1))
                    sxy = float(((a - mx) * (b - my)).sum() / (N - 1))
                    alpha = 4 * mx * my * sxy
                    beta = (mx * mx + my * my) * (sx + sy)
                    if alpha != 0:
                        return float(alpha / (beta + np.spacing(1)))
                    if alpha == 0 and beta == 0:
                        return 1.0
                    return 0.0

                coords = np.argwhere(gt_f == 1)
                h_, w_ = gt_f.shape
                if coords.size == 0:
                    cx, cy = int(round(w_ / 2)), int(round(h_ / 2))
                else:
                    cy, cx = coords.mean(axis=0).round().astype(int).tolist()
                cx = int(cx) + 1; cy = int(cy) + 1
                gt_LT = gt_f[0:cy, 0:cx]; gt_RT = gt_f[0:cy, cx:w_]
                gt_LB = gt_f[cy:h_, 0:cx]; gt_RB = gt_f[cy:h_, cx:w_]
                pr_LT = pred_f[0:cy, 0:cx]; pr_RT = pred_f[0:cy, cx:w_]
                pr_LB = pred_f[cy:h_, 0:cx]; pr_RB = pred_f[cy:h_, cx:w_]
                area = float(h_ * w_)
                w1 = (cx * cy) / area
                w2 = (cy * (w_ - cx)) / area
                w3 = ((h_ - cy) * cx) / area
                w4 = 1.0 - w1 - w2 - w3
                region_score = (
                    w1 * ssim(pr_LT, gt_LT) +
                    w2 * ssim(pr_RT, gt_RT) +
                    w3 * ssim(pr_LB, gt_LB) +
                    w4 * ssim(pr_RB, gt_RB)
                )
                alpha = 0.5
                return float(max(0.0, alpha * s_object(pred_f, gt_f) + (1 - alpha) * region_score))

            s_measure = _s_measure_binary(pr, gt)
            return {
                "DICE": dice,
                "IOU": iou,
                "PRECISION": precision,
                "RECALL": recall,
                "S_MEASURE": s_measure,
                "F1_SCORE": f1,
            }

        def _draw_metrics_footer(img: np.ndarray, metrics: dict, keys: List[str]) -> None:
            parts: List[str] = []
            for k in keys:
                if k in metrics:
                    parts.append(f"{k}: {metrics[k]:.3f}")
            if not parts:
                return
            text = "  ".join(parts)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            max_width = img.shape[1] - 16
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            if tw > max_width and tw > 0:
                scale = max(0.3, scale * (max_width / tw))
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = 8
            y = img.shape[0] - 6
            # stroke for readability
            cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        try:
            m = _compute_metrics(A, B)
            _draw_metrics_footer(panel1, m, ["DICE", "IOU"])
            _draw_metrics_footer(panel2, m, ["PRECISION", "RECALL"])
            _draw_metrics_footer(panel3, m, ["S_MEASURE", "F1_SCORE"])
        except Exception:
            pass

        # Helpers to build heatmap panels
        def _make_hm_panel(hm_img: np.ndarray) -> np.ndarray:
            hm = hm_img
            if hm.ndim == 2:
                hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2BGR)
            if hm.shape[:2] != (h, w):
                hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
            panel = hm.copy()
            if int(label) > 0:
                _draw_caret(panel, cx, cy, size=6, color=(255,255,255), thickness=2)
            else:
                _draw_cross(panel, cx, cy, size=6, color=(255,255,255), thickness=2)
            if base_image_bgr is not None:
                base = base_image_bgr
                if base.ndim == 2:
                    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
                if base.shape[:2] != (h, w):
                    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
                base_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                alpha = 0.5
                panel = cv2.addWeighted(base_gray_bgr, 1.0 - alpha, panel, alpha, 0.0)
            return panel

        panel4 = None
        panel5 = None
        # Prefer two maps when both provided
        if heatmap_fg_rgb is not None and heatmap_bg_rgb is not None:
            panel4 = _make_hm_panel(heatmap_fg_rgb)
            panel5 = _make_hm_panel(heatmap_bg_rgb)
        elif heatmap_rgb is not None:
            panel4 = _make_hm_panel(heatmap_rgb)

        # Titles
        def _add_title(img: np.ndarray, text: str) -> None:
            band_h = max(20, min(48, img.shape[0] // 20))
            roi = img[0:band_h, :, :]
            overlay = roi.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1] - 1, band_h - 1), (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, 0.5, roi, 0.5, 0, dst=roi)
            org = (8, band_h - 6)
            font_scale = 0.4
            font_thickness = 1
            # Truncate text to fit within image width
            max_width = img.shape[1] - 16  # Leave some padding
            (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            if text_width > max_width:
                # Binary search to find the right truncation point
                ellipsis = "..."
                for i in range(len(text), 0, -1):
                    truncated = text[:i] + ellipsis
                    (trunc_width, _), _ = cv2.getTextSize(truncated, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    if trunc_width <= max_width:
                        text = truncated
                        break
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        _add_title(panel1, "Overlay A (green) vs B (red)")
        _add_title(panel2, "Signed diff C: + (green), - (red)")
        _add_title(panel3, "Components + selected point")
        if panel4 is not None and panel5 is not None:
            _add_title(panel4, "FG probability heatmap + selected point")
            _add_title(panel5, "BG probability heatmap + selected point")
        elif panel4 is not None:
            _add_title(panel4, "Probability heatmap + selected point")

        panels = [panel1, panel2, panel3]
        if panel4 is not None and panel5 is not None:
            panels.extend([panel4, panel5])
        elif panel4 is not None:
            panels.append(panel4)
        canvas = np.concatenate(panels, axis=1)
        os.makedirs(os.path.dirname(dbg_out_path) or ".", exist_ok=True)
        _PIL.fromarray(canvas[..., ::-1]).save(dbg_out_path)
    except Exception:
        pass

def find_corresponding_image(
    images_dir: str,
    sample_stem: str,
    allowed_exts: Tuple[str, ...],
) -> Optional[str]:
    images_dir_abs = to_abs(images_dir)
    for ext in allowed_exts:
        candidate = os.path.join(images_dir_abs, sample_stem + ext)
        if os.path.isfile(candidate):
            return candidate
    # Fallback: scan directory and match by stem (handles mixed-case extensions or uncommon extensions)
    for name in os.listdir(images_dir_abs):
        if stem(name).lower() == sample_stem.lower():
            return os.path.join(images_dir_abs, name)
    return None

def calculate_next_point(gt_mask_path, sam_masks_dir, current_points_len, dbg_out_dir: Optional[str] = None):
    # 读取gt与pred
    try:
        gt_u8 = np.array(_PIL.open(to_abs(gt_mask_path)), dtype=np.uint8)
        if current_points_len > 0:
            last_mask_path = _find_latest_mask_path(sam_masks_dir, stem(gt_mask_path), current_points_len)
            if not last_mask_path or not os.path.isfile(last_mask_path):
                print(f"[WARN] Skip: last mask not found under {sam_masks_dir}")
                return None, None, None
            pred_u8 = np.array(_PIL.open(to_abs(last_mask_path)), dtype=np.uint8)
        else:
            pred_u8 = np.zeros_like(gt_u8)
    except Exception as e:
        print(f"[WARN] Skip: failed to read masks ({e})")
        return None, None, None

    # 若形状不同，将pred按最近邻缩放到gt尺寸
    h_g, w_g = gt_u8.shape[:2]
    h_p, w_p = pred_u8.shape[:2]
    if (h_g, w_g) != (h_p, w_p):
        try:
            # from PIL import Image as _PIL
            pred_u8 = np.array(_PIL.fromarray(pred_u8).resize((w_g, h_g), resample=_PIL.NEAREST), dtype=np.uint8)
        except Exception:
            pass

    ### For Debug Only:
    # if stem(gt_mask_path) == "BRATS_001_z0088":
    #     print("#Debug#")
    ### :For Debug Only

    # 计算差异的最大连通分量代表点
    dbg_path = None
    if dbg_out_dir:
        try:
            subdir = os.path.join(dbg_out_dir, stem(gt_mask_path))
            os.makedirs(subdir, exist_ok=True)
            dbg_path = os.path.join(subdir, f"dbg_{max(0, int(current_points_len))}.png")
        except Exception:
            dbg_path = None
    rep = largest_diff_component_representative(gt_u8, pred_u8, connectivity=2, min_area=0, dbg_out_path=dbg_path)
    if rep is None:
        print(f"[INFO] no diff component; skip append")
        return None, None, None
    x_next = int(rep["x"])  # 列
    y_next = int(rep["y"])  # 行
    label_next = 1 if int(rep["sign"]) > 0 else 0
    return x_next, y_next, label_next

def main() -> None:
    global _PIL
    args = parse_args()

    # Debug-only mode: read JSON, generate debug images but do not write JSON
    if args.debug_json:
        if not args.debug_output_dir:
            raise RuntimeError("--debug_output_dir is required when --debug_json is set")
        json_path = args.debug_json
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Debug JSON not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        try:
            arr = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Debug JSON must be a JSON array: {e}")
        if not isinstance(arr, list):
            raise RuntimeError("Debug JSON root must be a JSON array")

        num_generated = 0
        for rec in arr:
            if not isinstance(rec, dict):
                continue

            image_path = rec.get("image")
            gt_mask_path = rec.get("gt_mask")
            sam_masks_dir = rec.get("sam_masks_dir")
            if not (image_path and sam_masks_dir):
                continue
            sample_stem = stem(image_path)
            ### For Debug Only:
            # if not sample_stem == "BRATS_001_z0088":
            #     continue
            ### :For Debug Only
            try:
                subdir = os.path.join(args.debug_output_dir, sample_stem)
                os.makedirs(subdir, exist_ok=True)
            except Exception:
                continue

            points = rec.get("points", [])
            labels = rec.get("labels", [])
            if not (isinstance(points, list) and isinstance(labels, list)):
                continue
            if len(points) != len(labels):
                continue

            gt_mask_u8 = np.array(_PIL.open(to_abs(gt_mask_path)), dtype=np.uint8)
            # Load base image if exists
            base_img_bgr = None
            try:
                img_arr = np.array(_PIL.open(to_abs(image_path)).convert("RGB"))
                base_img_bgr = img_arr[:, :, ::-1]
            except Exception:
                base_img_bgr = None
            # heatmap will be loaded per-step inside the loop (fix: previously attempted before loop)
            # Load gt as all-zero array to preserve panel structure; we visualize differences of predicted masks only
            for i, (pt, lb) in enumerate(zip(points, labels)):
                # For step i, compare masks of ground truth and step i-1 (i==0 uses empty prev)
                if i == 0:
                    prev = np.zeros_like(gt_mask_u8, dtype=np.uint8)
                else:
                    try:
                        prev_path = os.path.join(sam_masks_dir, sample_stem, f"{i-1}.png")
                        prev = np.array(_PIL.open(to_abs(prev_path)).convert("L"), dtype=np.uint8) if os.path.isfile(prev_path) else np.zeros_like(gt_mask_u8, dtype=np.uint8)
                    except Exception:
                        prev = np.zeros_like(gt_mask_u8, dtype=np.uint8)
                # Load heatmap for this step: prefer heatmap-{i}.png; fallback to heatmap-{i-1}.png if missing
                heatmap_rgb = None
                try:
                    hm_dir = os.path.join(sam_masks_dir, sample_stem)
                    cand = [f"heatmap-{i}.png"]
                    if i - 1 >= 0:
                        cand.append(f"heatmap-{i-1}.png")
                    for name in cand:
                        hm_path = os.path.join(hm_dir, name)
                        if os.path.isfile(hm_path):
                            hm = np.array(_PIL.open(to_abs(hm_path)).convert("RGB"))
                            heatmap_rgb = hm[:, :, ::-1]  # to BGR
                            break
                except Exception:
                    heatmap_rgb = None
                # Render panels with provided point
                dbg_path = os.path.join(subdir, f"dbg_{i}.png")
                _render_diff_panels_with_point(
                    gt_mask_u8,
                    prev,
                    (float(pt[0]), float(pt[1])),
                    int(lb),
                    dbg_path,
                    base_image_bgr=base_img_bgr,
                    heatmap_rgb=heatmap_rgb,
                )
                num_generated += 1
        print(f"Generated debug images for {num_generated} steps into {args.debug_output_dir}")
        return

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

            x_next, y_next, label_next = calculate_next_point(
                gt_mask_path,
                sam_masks_dir,
                len(points_list),
                dbg_out_dir=args.debug_output_dir,
            )
            if x_next is None or y_next is None or label_next is None:
                continue

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
                mask_img = _PIL.open(to_abs(mask_path))
                mask_u8 = np.array(mask_img, dtype=np.uint8)
            except Exception as e:
                print(f"[WARN] Skipping unreadable mask: {mask_path} ({e})")
                continue
            
            x_next, y_next, label_next = calculate_next_point(
                mask_path,
                None,
                0,
                dbg_out_dir=args.debug_output_dir,
            )
            if x_next is None or y_next is None or label_next is None:
                if args.skip_empty:
                    num_skipped_empty += 1
                    continue
                else:
                    # For empty masks and skip_empty=False, place centroid at image center
                    h, w = mask_u8.shape[:2]
                    centroid = (float(w - 1) / 2.0, float(h - 1) / 2.0)
            centroid = (x_next, y_next)

            # Find corresponding image
            img_path = find_corresponding_image(images_dir, sample_stem, allowed_image_exts)
            if img_path is None:
                num_missing_images += 1
                print(f"[WARN] Missing image for mask stem '{sample_stem}' in {images_dir}")
                continue
           
            image_field = os.path.abspath(img_path) if args.abs_paths else img_path
            mask_field = os.path.abspath(mask_path) if args.abs_paths else mask_path
            record = {"image": image_field, "gt_mask": mask_field, "points": [[float(centroid[0]), float(centroid[1])]], "labels": [1]}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

            # Visualization
            if False and args.viz_dir:
                try:
                    # Load image RGB
                    with _PIL.open(to_abs(img_path)) as im_rgb:
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

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251002.jsonl

  # 2) 追加模式（为每条记录计算后续的第2个及以后提示点）
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl

  # 3) 调试模式（生成调试图像）
  # 若 sam2_segment_from_points.py 已生成 heatmap-{i}.png，本脚本会在调试图中追加第4幅：
  # “概率热力图 + 当前提示点”，用于核对提示点与热区的一致性。
  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --debug_json outputs/braintumour/pred_251023.jsonl \
    --debug_output_dir outputs/braintumour/dbg_gen_points

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
# from PIL import Image, ImageDraw
from PIL import Image as _PIL
from scipy.ndimage import label, distance_transform_edt
import cv2  # type: ignore

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create/Append deepest-point prompts from masks using EDT")
    # 互斥：首写输出 vs 追加到已存在的JSON数组
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_jsonl", type=str, help="Path to write JSONL lines for first prompts")
    group.add_argument("--appendto_jsonl", type=str, help="Path to existing JSON array to append next prompts")
    group.add_argument("--debug_json", type=str, help="Path to JSON array to read for debug visualization only")

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
    p.add_argument("--debug_output_dir", type=str, default=None,
                   help="If set, write per-sample debug images for diff component selection")
    return p.parse_args()


def list_files_with_exts(directory: str, allowed_exts: Tuple[str, ...]) -> List[str]:
    directory = to_abs(directory) or directory
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


def to_abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path) if not os.path.isabs(path) else path


def _find_latest_mask_path(sam_masks_dir: str, image_stem: str, current_points_len: int) -> Optional[str]:
    """返回该图像在sam_masks_dir下的最新预测mask路径。
    优先尝试 sam_masks_dir/<stem>/<current_points_len-1>.png；若不存在，
    则在该目录下寻找最大数字文件名的png。
    """
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

from scipy import ndimage as ndi

def farthest_point_from_boundary(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    # mask: 二值数组，区域为 True/1，背景为 False/0
    mask = (mask > 0)
    dist = ndi.distance_transform_edt(mask)  # 区域内每点到背景的欧氏距离
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    # r = float(dist[y, x])  # 到边界的最大最小距离
    # return (y, x), r
    return (float(x), float(y))

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

def largest_diff_component_representative(A, B, connectivity=2, min_area=0, dbg_out_path: Optional[str] = None):
    # A, B: 2D uint8 arrays, values {0,1} 或 {0,255}
    A = (A > 0).astype(np.uint8)
    B = (B > 0).astype(np.uint8)
    ### C = A.astype(np.int16) - B.astype(np.int16)  # {-1, 0, +1}
    C = A.astype(np.int8) - B.astype(np.int8)  # {-1, 0, +1}

    regions = []
    # 8 邻域结构元（connectivity=2 表示 8 邻域；=1 表示 4 邻域）
    st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    for sign, mask in ((1, C > 0), (-1, C < 0)):
        if not mask.any():
            continue
        lab, n = ndi.label(mask, structure=st)
        if n == 0:
            continue
        areas = np.bincount(lab.ravel())[1:]  # 忽略背景0
        if min_area > 0:
            keep_ids = np.where(areas >= min_area)[0] + 1
            if keep_ids.size == 0:
                continue
            mask2 = np.isin(lab, keep_ids)
            lab, n = ndi.label(mask2, structure=st)
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

    # centroid = farthest_point_from_boundary(comp)
    # # centroid = compute_centroid(comp)
    # if centroid is None:
    #     return None
    D = distance_transform_edt(comp)
    (centroidY, centroidX) = np.unravel_index(np.argmax(D), D.shape)

    # 调试可视化（改用统一的渲染函数，保持风格一致）
    if dbg_out_path:
        try:
            _render_diff_panels_with_point(
                A=A,
                B=B,
                point_xy=(float(centroidX), float(centroidY)),
                label=1 if int(sign) > 0 else 0,
                dbg_out_path=dbg_out_path,
                connectivity=connectivity,
            )
        except Exception:
            pass

    return dict(sign=sign, area=area, y=int(centroidY), x=int(centroidX), max_radius=float(D[centroidY, centroidX]))


def _render_diff_panels_with_point(
    A: np.ndarray,
    B: np.ndarray,
    point_xy: Tuple[float, float],
    label: int,
    dbg_out_path: Optional[str],
    connectivity: int = 2,
    base_image_bgr: Optional[np.ndarray] = None,
    heatmap_rgb: Optional[np.ndarray] = None,
    heatmap_fg_rgb: Optional[np.ndarray] = None,
    heatmap_bg_rgb: Optional[np.ndarray] = None,
) -> None:
    """Render 3-panel debug image reusing the visualization style in
    largest_diff_component_representative, but mark a provided point instead of
    the recomputed centroid.
    - Panel 1: A (green) vs B (red) overlay
    - Panel 2: signed diff C (positive green, negative red)
    - Panel 3: connected components colored, with marker at point_xy
    - Panel 4 (optional): probability heatmap (RGB) + current point marker
    """
    rng = np.random.default_rng(12345)
    try:
        if dbg_out_path is None:
            return
        h, w = A.shape[:2]
        # Panel 1: A/B overlay (A=green, B=red), optionally blended on base image
        overlay1 = np.zeros((h, w, 3), dtype=np.uint8)
        overlay1[..., 1] = (A > 0).astype(np.uint8) * 255  # G
        overlay1[..., 2] = (B > 0).astype(np.uint8) * 255  # R
        if base_image_bgr is not None:
            base = base_image_bgr
            if base.ndim == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            if base.shape[:2] != (h, w):
                base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
            panel1 = cv2.addWeighted(base, 1.0, overlay1, 0.6, 0.0)
        else:
            bg = np.zeros_like(overlay1)
            panel1 = cv2.addWeighted(bg, 1.0, overlay1, 0.6, 0.0)

        # Panel 2: signed diff C (positive green, negative red)
        C = A.astype(np.int16) - B.astype(np.int16)
        panel2 = np.zeros((h, w, 3), dtype=np.uint8)
        pos = (C > 0)
        neg = (C < 0)
        panel2[pos, 1] = 255
        panel2[neg, 2] = 255

        # Panel 3: connected components colored + provided point marker
        panel3 = np.zeros((h, w, 3), dtype=np.uint8)
        st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
        lab, n = ndi.label(C, structure=st)
        if n != 0:
            colors = (rng.integers(0, 256, size=(n, 3))).astype(np.uint8)
            colors = np.clip(colors + 100, 0, 255)
            for i in range(1, n + 1):
                panel3[lab == i] = colors[i - 1]

        # draw marker at provided (x,y): foreground -> caret '^', background -> 'X'
        cx = int(round(float(point_xy[0])))
        cy = int(round(float(point_xy[1])))
        def _draw_cross(img, x, y, size=6, color=(255,255,255), thickness=2):
            cv2.line(img, (x - size, y - size), (x + size, y + size), (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y + size), (x + size, y - size), (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness, lineType=cv2.LINE_AA)
        def _draw_caret(img, x, y, size=6, color=(255,255,255), thickness=2):
            p_top = (int(x), int(y - size))
            p_left = (int(x - size), int(y + size))
            p_right = (int(x + size), int(y + size))
            cv2.line(img, p_left, p_top, (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, p_right, p_top, (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, p_left, p_top, color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p_right, p_top, color, thickness, lineType=cv2.LINE_AA)
        if int(label) > 0:
            _draw_caret(panel3, cx, cy, size=6, color=(255,255,255), thickness=2)
        else:
            _draw_cross(panel3, cx, cy, size=6, color=(255,255,255), thickness=2)

        # Compute and draw metrics comparing B (pred) vs A (gt) on panel1
        def _compute_metrics(gt_u8: np.ndarray, pr_u8: np.ndarray) -> dict:
            gt = (gt_u8 > 0)
            pr = (pr_u8 > 0)
            tp = int(np.logical_and(gt, pr).sum())
            fp = int(np.logical_and(~gt, pr).sum())
            fn = int(np.logical_and(gt, ~pr).sum())
            denom_dice = 2 * tp + fp + fn
            dice = (2 * tp / denom_dice) if denom_dice > 0 else 1.0
            denom_iou = tp + fp + fn
            iou = (tp / denom_iou) if denom_iou > 0 else 1.0
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Lightweight S-measure implementation adapted for binary masks
            def _s_measure_binary(pred: np.ndarray, gt: np.ndarray) -> float:
                pred_f = pred.astype(np.float64)
                gt_f = gt.astype(np.float64)
                y_mean = gt_f.mean()
                if y_mean == 0:
                    return float(1.0 - pred_f.mean())
                if y_mean == 1:
                    return float(pred_f.mean())

                def s_object(p: np.ndarray, g: np.ndarray) -> float:
                    vals = p[g == 1]
                    if vals.size == 0:
                        return 0.0
                    mx = float(vals.mean())
                    sx = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                    return float(2 * mx / (mx * mx + 1 + sx + np.spacing(1)))

                def ssim(a: np.ndarray, b: np.ndarray) -> float:
                    h_, w_ = a.shape
                    N = h_ * w_
                    if N <= 1:
                        return 1.0
                    mx = float(a.mean()); my = float(b.mean())
                    sx = float(((a - mx) ** 2).sum() / (N - 1))
                    sy = float(((b - my) ** 2).sum() / (N - 1))
                    sxy = float(((a - mx) * (b - my)).sum() / (N - 1))
                    alpha = 4 * mx * my * sxy
                    beta = (mx * mx + my * my) * (sx + sy)
                    if alpha != 0:
                        return float(alpha / (beta + np.spacing(1)))
                    if alpha == 0 and beta == 0:
                        return 1.0
                    return 0.0

                coords = np.argwhere(gt_f == 1)
                h_, w_ = gt_f.shape
                if coords.size == 0:
                    cx, cy = int(round(w_ / 2)), int(round(h_ / 2))
                else:
                    cy, cx = coords.mean(axis=0).round().astype(int).tolist()
                cx = int(cx) + 1; cy = int(cy) + 1
                gt_LT = gt_f[0:cy, 0:cx]; gt_RT = gt_f[0:cy, cx:w_]
                gt_LB = gt_f[cy:h_, 0:cx]; gt_RB = gt_f[cy:h_, cx:w_]
                pr_LT = pred_f[0:cy, 0:cx]; pr_RT = pred_f[0:cy, cx:w_]
                pr_LB = pred_f[cy:h_, 0:cx]; pr_RB = pred_f[cy:h_, cx:w_]
                area = float(h_ * w_)
                w1 = (cx * cy) / area
                w2 = (cy * (w_ - cx)) / area
                w3 = ((h_ - cy) * cx) / area
                w4 = 1.0 - w1 - w2 - w3
                region_score = (
                    w1 * ssim(pr_LT, gt_LT) +
                    w2 * ssim(pr_RT, gt_RT) +
                    w3 * ssim(pr_LB, gt_LB) +
                    w4 * ssim(pr_RB, gt_RB)
                )
                alpha = 0.5
                return float(max(0.0, alpha * s_object(pred_f, gt_f) + (1 - alpha) * region_score))

            s_measure = _s_measure_binary(pr, gt)
            return {
                "DICE": dice,
                "IOU": iou,
                "PRECISION": precision,
                "RECALL": recall,
                "S_MEASURE": s_measure,
                "F1_SCORE": f1,
            }

        def _draw_metrics_footer(img: np.ndarray, metrics: dict, keys: List[str]) -> None:
            parts: List[str] = []
            for k in keys:
                if k in metrics:
                    parts.append(f"{k}: {metrics[k]:.3f}")
            if not parts:
                return
            text = "  ".join(parts)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            max_width = img.shape[1] - 16
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            if tw > max_width and tw > 0:
                scale = max(0.3, scale * (max_width / tw))
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = 8
            y = img.shape[0] - 6
            # stroke for readability
            cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        try:
            m = _compute_metrics(A, B)
            _draw_metrics_footer(panel1, m, ["DICE", "IOU"])
            _draw_metrics_footer(panel2, m, ["PRECISION", "RECALL"])
            _draw_metrics_footer(panel3, m, ["S_MEASURE", "F1_SCORE"])
        except Exception:
            pass

        # Helpers to build heatmap panels
        def _make_hm_panel(hm_img: np.ndarray) -> np.ndarray:
            hm = hm_img
            if hm.ndim == 2:
                hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2BGR)
            if hm.shape[:2] != (h, w):
                hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
            panel = hm.copy()
            if int(label) > 0:
                _draw_caret(panel, cx, cy, size=6, color=(255,255,255), thickness=2)
            else:
                _draw_cross(panel, cx, cy, size=6, color=(255,255,255), thickness=2)
            if base_image_bgr is not None:
                base = base_image_bgr
                if base.ndim == 2:
                    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
                if base.shape[:2] != (h, w):
                    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
                base_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                alpha = 0.5
                panel = cv2.addWeighted(base_gray_bgr, 1.0 - alpha, panel, alpha, 0.0)
            return panel

        panel4 = None
        panel5 = None
        # Prefer two maps when both provided
        if heatmap_fg_rgb is not None and heatmap_bg_rgb is not None:
            panel4 = _make_hm_panel(heatmap_fg_rgb)
            panel5 = _make_hm_panel(heatmap_bg_rgb)
        elif heatmap_rgb is not None:
            panel4 = _make_hm_panel(heatmap_rgb)

        # Titles
        def _add_title(img: np.ndarray, text: str) -> None:
            band_h = max(20, min(48, img.shape[0] // 20))
            roi = img[0:band_h, :, :]
            overlay = roi.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1] - 1, band_h - 1), (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, 0.5, roi, 0.5, 0, dst=roi)
            org = (8, band_h - 6)
            font_scale = 0.4
            font_thickness = 1
            # Truncate text to fit within image width
            max_width = img.shape[1] - 16  # Leave some padding
            (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            if text_width > max_width:
                # Binary search to find the right truncation point
                ellipsis = "..."
                for i in range(len(text), 0, -1):
                    truncated = text[:i] + ellipsis
                    (trunc_width, _), _ = cv2.getTextSize(truncated, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    if trunc_width <= max_width:
                        text = truncated
                        break
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        _add_title(panel1, "Overlay A (green) vs B (red)")
        _add_title(panel2, "Signed diff C: + (green), - (red)")
        _add_title(panel3, "Components + selected point")
        if panel4 is not None and panel5 is not None:
            _add_title(panel4, "FG probability heatmap + selected point")
            _add_title(panel5, "BG probability heatmap + selected point")
        elif panel4 is not None:
            _add_title(panel4, "Probability heatmap + selected point")

        panels = [panel1, panel2, panel3]
        if panel4 is not None and panel5 is not None:
            panels.extend([panel4, panel5])
        elif panel4 is not None:
            panels.append(panel4)
        canvas = np.concatenate(panels, axis=1)
        os.makedirs(os.path.dirname(dbg_out_path) or ".", exist_ok=True)
        _PIL.fromarray(canvas[..., ::-1]).save(dbg_out_path)
    except Exception:
        pass

def find_corresponding_image(
    images_dir: str,
    sample_stem: str,
    allowed_exts: Tuple[str, ...],
) -> Optional[str]:
    images_dir_abs = to_abs(images_dir)
    for ext in allowed_exts:
        candidate = os.path.join(images_dir_abs, sample_stem + ext)
        if os.path.isfile(candidate):
            return candidate
    # Fallback: scan directory and match by stem (handles mixed-case extensions or uncommon extensions)
    for name in os.listdir(images_dir_abs):
        if stem(name).lower() == sample_stem.lower():
            return os.path.join(images_dir_abs, name)
    return None

def calculate_next_point(gt_mask_path, sam_masks_dir, current_points_len, dbg_out_dir: Optional[str] = None):
    # 读取gt与pred
    try:
        gt_u8 = np.array(_PIL.open(to_abs(gt_mask_path)), dtype=np.uint8)
        if current_points_len > 0:
            last_mask_path = _find_latest_mask_path(sam_masks_dir, stem(gt_mask_path), current_points_len)
            if not last_mask_path or not os.path.isfile(last_mask_path):
                print(f"[WARN] Skip: last mask not found under {sam_masks_dir}")
                return None, None, None
            pred_u8 = np.array(_PIL.open(to_abs(last_mask_path)), dtype=np.uint8)
        else:
            pred_u8 = np.zeros_like(gt_u8)
    except Exception as e:
        print(f"[WARN] Skip: failed to read masks ({e})")
        return None, None, None

    # 若形状不同，将pred按最近邻缩放到gt尺寸
    h_g, w_g = gt_u8.shape[:2]
    h_p, w_p = pred_u8.shape[:2]
    if (h_g, w_g) != (h_p, w_p):
        try:
            # from PIL import Image as _PIL
            pred_u8 = np.array(_PIL.fromarray(pred_u8).resize((w_g, h_g), resample=_PIL.NEAREST), dtype=np.uint8)
        except Exception:
            pass

    ### For Debug Only:
    # if stem(gt_mask_path) == "BRATS_001_z0088":
    #     print("#Debug#")
    ### :For Debug Only

    # 计算差异的最大连通分量代表点
    dbg_path = None
    if dbg_out_dir:
        try:
            subdir = os.path.join(dbg_out_dir, stem(gt_mask_path))
            os.makedirs(subdir, exist_ok=True)
            dbg_path = os.path.join(subdir, f"dbg_{max(0, int(current_points_len))}.png")
        except Exception:
            dbg_path = None
    rep = largest_diff_component_representative(gt_u8, pred_u8, connectivity=2, min_area=0, dbg_out_path=dbg_path)
    if rep is None:
        print(f"[INFO] no diff component; skip append")
        return None, None, None
    x_next = int(rep["x"])  # 列
    y_next = int(rep["y"])  # 行
    label_next = 1 if int(rep["sign"]) > 0 else 0
    return x_next, y_next, label_next

def main() -> None:
    global _PIL
    args = parse_args()

    # Debug-only mode: read JSON, generate debug images but do not write JSON
    if args.debug_json:
        if not args.debug_output_dir:
            raise RuntimeError("--debug_output_dir is required when --debug_json is set")
        json_path = args.debug_json
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Debug JSON not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        try:
            arr = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Debug JSON must be a JSON array: {e}")
        if not isinstance(arr, list):
            raise RuntimeError("Debug JSON root must be a JSON array")

        num_generated = 0
        for rec in arr:
            if not isinstance(rec, dict):
                continue

            image_path = rec.get("image")
            gt_mask_path = rec.get("gt_mask")
            sam_masks_dir = rec.get("sam_masks_dir")
            if not (image_path and sam_masks_dir):
                continue
            sample_stem = stem(image_path)
            ### For Debug Only:
            # if not sample_stem == "BRATS_001_z0088":
            #     continue
            ### :For Debug Only
            try:
                subdir = os.path.join(args.debug_output_dir, sample_stem)
                os.makedirs(subdir, exist_ok=True)
            except Exception:
                continue

            points = rec.get("points", [])
            labels = rec.get("labels", [])
            if not (isinstance(points, list) and isinstance(labels, list)):
                continue
            if len(points) != len(labels):
                continue

            gt_mask_u8 = np.array(_PIL.open(to_abs(gt_mask_path)), dtype=np.uint8)
            # Load base image if exists
            base_img_bgr = None
            try:
                img_arr = np.array(_PIL.open(to_abs(image_path)).convert("RGB"))
                base_img_bgr = img_arr[:, :, ::-1]
            except Exception:
                base_img_bgr = None
            # heatmap will be loaded per-step inside the loop (fix: previously attempted before loop)
            # Load gt as all-zero array to preserve panel structure; we visualize differences of predicted masks only
            for i, (pt, lb) in enumerate(zip(points, labels)):
                # For step i, compare masks of ground truth and step i-1 (i==0 uses empty prev)
                if i == 0:
                    prev = np.zeros_like(gt_mask_u8, dtype=np.uint8)
                else:
                    try:
                        prev_path = os.path.join(sam_masks_dir, sample_stem, f"{i-1}.png")
                        prev = np.array(_PIL.open(to_abs(prev_path)).convert("L"), dtype=np.uint8) if os.path.isfile(prev_path) else np.zeros_like(gt_mask_u8, dtype=np.uint8)
                    except Exception:
                        prev = np.zeros_like(gt_mask_u8, dtype=np.uint8)
                # Load heatmap for this step:
                # Prefer two-map if present: p_fg-{i}.png and p_bg-{i}.png; else fallback to heatmap-{i}.png; else heatmap-{i-1}.png
                heatmap_rgb = None
                heatmap_fg_rgb = None
                heatmap_bg_rgb = None
                try:
                    hm_dir = os.path.join(sam_masks_dir, sample_stem)
                    # try two maps first
                    fg_path = os.path.join(hm_dir, f"p_fg-{i}.png")
                    bg_path = os.path.join(hm_dir, f"p_bg-{i}.png")
                    if os.path.isfile(fg_path) and os.path.isfile(bg_path):
                        hm_fg = np.array(_PIL.open(to_abs(fg_path)).convert("RGB"))
                        hm_bg = np.array(_PIL.open(to_abs(bg_path)).convert("RGB"))
                        heatmap_fg_rgb = hm_fg[:, :, ::-1]
                        heatmap_bg_rgb = hm_bg[:, :, ::-1]
                    else:
                        cand = [f"heatmap-{i}.png"]
                        if i - 1 >= 0:
                            cand.append(f"heatmap-{i-1}.png")
                        for name in cand:
                            hm_path = os.path.join(hm_dir, name)
                            if os.path.isfile(hm_path):
                                hm = np.array(_PIL.open(to_abs(hm_path)).convert("RGB"))
                                heatmap_rgb = hm[:, :, ::-1]  # to BGR
                                break
                except Exception:
                    heatmap_rgb = None
                    heatmap_fg_rgb = None
                    heatmap_bg_rgb = None
                # Render panels with provided point
                dbg_path = os.path.join(subdir, f"dbg_{i}.png")
                _render_diff_panels_with_point(
                    gt_mask_u8,
                    prev,
                    (float(pt[0]), float(pt[1])),
                    int(lb),
                    dbg_path,
                    base_image_bgr=base_img_bgr,
                    heatmap_rgb=heatmap_rgb,
                    heatmap_fg_rgb=heatmap_fg_rgb,
                    heatmap_bg_rgb=heatmap_bg_rgb,
                )
                num_generated += 1
        print(f"Generated debug images for {num_generated} steps into {args.debug_output_dir}")
        return

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

            x_next, y_next, label_next = calculate_next_point(
                gt_mask_path,
                sam_masks_dir,
                len(points_list),
                dbg_out_dir=args.debug_output_dir,
            )
            if x_next is None or y_next is None or label_next is None:
                continue

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
                mask_img = _PIL.open(to_abs(mask_path))
                mask_u8 = np.array(mask_img, dtype=np.uint8)
            except Exception as e:
                print(f"[WARN] Skipping unreadable mask: {mask_path} ({e})")
                continue
            
            x_next, y_next, label_next = calculate_next_point(
                mask_path,
                None,
                0,
                dbg_out_dir=args.debug_output_dir,
            )
            if x_next is None or y_next is None or label_next is None:
                if args.skip_empty:
                    num_skipped_empty += 1
                    continue
                else:
                    # For empty masks and skip_empty=False, place centroid at image center
                    h, w = mask_u8.shape[:2]
                    centroid = (float(w - 1) / 2.0, float(h - 1) / 2.0)
            centroid = (x_next, y_next)

            # Find corresponding image
            img_path = find_corresponding_image(images_dir, sample_stem, allowed_image_exts)
            if img_path is None:
                num_missing_images += 1
                print(f"[WARN] Missing image for mask stem '{sample_stem}' in {images_dir}")
                continue
           
            image_field = os.path.abspath(img_path) if args.abs_paths else img_path
            mask_field = os.path.abspath(mask_path) if args.abs_paths else mask_path
            record = {"image": image_field, "gt_mask": mask_field, "points": [[float(centroid[0]), float(centroid[1])]], "labels": [1]}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

            # Visualization
            if False and args.viz_dir:
                try:
                    # Load image RGB
                    with _PIL.open(to_abs(img_path)) as im_rgb:
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

我已实现两处改动并保持默认行为不变。

- 预测脚本
  - 在 `seg-rl/heatmap/predict_next_point_from_model.py` 中：
    - `--coord_label_mode two_map` 时，额外保存两张概率图到 `sam_masks_dir/{stem}/`：
      - `p_fg-{i}.png`: 前景概率图
      - `p_bg-{i}.png`: 背景概率图
    - 其他模式继续保存单图：
      - `heatmap-{i}.png`
    - 保存函数 `_save_heatmap_image` 支持 `name_prefix` 参数。
    - `_predict_point` 现在在 `two_map` 模式下返回 `(prob_fg, prob_bg)`，其余模式返回 `(prob_fg, None)`。

- 调试可视化（annotator）
  - 在 `seg-rl/annotator/gen_point_jsonl_from_masks.py`：
    - `_render_diff_panels_with_point` 新增可选参数 `heatmap_fg_rgb`, `heatmap_bg_rgb`。若两者齐全，将在第四、第五列分别显示“FG probability heatmap + selected point”和“BG probability heatmap + selected point”；若只有单张热力图，则维持原第四列单图显示；若没有热力图则不显示第4/5列。
    - Debug 模式读取 `sam_masks_dir/{stem}` 时，优先尝试 `p_fg-{i}.png` 和 `p_bg-{i}.png`；若存在则同时显示两张；否则回退到 `heatmap-{i}.png`（再不行则尝试 `heatmap-{i-1}.png` 以保持兼容）。
  - 可视化列布局自动从原 3/4 列扩展至 5 列（前两列为差异图，第三列为连通组件，第四/五列为热力图）。

如何使用
- 生成并保存两张概率图：
  - python -m seg-rl.heatmap.predict_next_point_from_model --coord_label_mode two_map --model_path ... --output_json ... --sam_dir ...
- 在标注器调试模式中自动显示两张热力图（若存在）：
  - python -m seg-rl.annotator.gen_point_jsonl_from_masks --debug_json ... --debug_output_dir ... --images_dir ... --masks_dir ... --sam_masks_dir ...

已通过本地静态检查，无新增 linter 报错。

- 我把预测保存、调试显示的两图逻辑都打通了。若你希望在非 two_map 模式下也能选择保存两张图（例如从分类头推导一个 p_fg），可再加一个开关；需要的话我可以继续加。

---

**User**

解释 在soft_argmax_from_logits中，为什么`b, _, h, w = logits.shape`，可以用_忽略掉c，而后面`p = ... .view(b, 1, h, w)`直接用1作为c了？为什么不存在logits的c不为1的情况？

---

**Cursor**

### 核心回答
- 在本工程里，热力图 `logits` 的通道数固定为 1（形状 `[B,1,H,W]`）。因此在 `soft_argmax_from_logits` 中用 `b, _, h, w = logits.shape` 忽略通道维，并在后续 `view(b, 1, h, w)` 写死为 1 是合理且安全的。
- 之所以“不会出现 `c != 1`”，是因为模型头部就是单通道输出（无论是 ResNet 分支还是 U-Net 分支），训练/推理的其他函数都基于这一约定。

### 代码证据
- ResNet 分支热力图头的最后一层输出 1 通道：
```37:41:/Users/donchiu/Works/Seg-R1/seg-rl/heatmap/model.py
self.head = nn.Sequential(
    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(mid_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
)
```

- U-Net 分支的最终卷积也输出 1 通道：
```105:108:/Users/donchiu/Works/Seg-R1/seg-rl/heatmap/model.py
self.final = nn.Conv2d(c1, 1, kernel_size=1)

def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...
    return self.final(d1)
```

- 模型类文档明确“输出 1 通道热力图 logits”：
```128:132:/Users/donchiu/Works/Seg-R1/seg-rl/heatmap/model.py
class PointHeatmapModel(nn.Module):
    """Backbone + 1-channel heatmap logits head.

    Output: logits of shape [B,1,H',W'] optionally upsampled to input HxW.
    """
```

- `soft_argmax_from_logits` 假设输入为 `[B,1,H,W]`，将其展平为 `[B,H*W]` 再还原为 `[B,1,H,W]`：
```207:215:/Users/donchiu/Works/Seg-R1/seg-rl/heatmap/model.py
def soft_argmax_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    b, _, h, w = logits.shape
    p = F.softmax(logits.view(b, -1) / max(temperature, 1e-6), dim=1).view(b, 1, h, w)
    ...
```

### 补充说明
- 若未来你将模型改成“每类一张热力图”（输出 2 通道），需要同时调整：
  - 软/硬 argmax 的实现（先选通道再选像素，或对每通道分别 soft-argmax），
  - 以及层级采样/对数概率等函数的通道假设。
- 当前代码的“两图模式”是通过预测后把单通道概率 `p_fg` 映射为 `p_bg = 1 - p_fg` 得到的，并未改变模型输出通道数，因此 `c` 仍为 1。

```python
self.head = nn.Sequential(
    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(mid_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
)
```

```python
self.final = nn.Conv2d(c1, 1, kernel_size=1)

def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...
    return self.final(d1)
```

```python
class PointHeatmapModel(nn.Module):
    """Backbone + 1-channel heatmap logits head.

    Output: logits of shape [B,1,H',W'] optionally upsampled to input HxW.
    """
```

```python
def soft_argmax_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    b, _, h, w = logits.shape
    p = F.softmax(logits.view(b, -1) / max(temperature, 1e-6), dim=1).view(b, 1, h, w)
    ...
```

