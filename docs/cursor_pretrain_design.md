# 机器学习模型训练任务讨论
_Exported on 9/21/2025 at 12:33:38 PDT from Cursor (1.6.42)_

---

**User**

我想做这样一种ML训练任务：训练输入数据是一组{图片, 坐标}对，图片可能是2D或3D（暂且先定为仅限2D），坐标是在图片尺度范围内的一个坐标（暂且限2D），先不用考虑坐标的实际意义，你只需要知道训练样本是这样的从图片到一个坐标的映射即可。我希望训练一个机器学习模型，使之在推理时可以根据输入图片，输出其应映射到的坐标值。我的设想是，模型最后一层应输出一个与图片包含坐标数量相同的logits，表示图片中每个坐标位置被选定为最终输出坐标的置信度，最终选取的坐标就是选择置信度最高的一个作为坐标位置的索引进行输出。请问这样一类的任务在机器学习中称作什么？然后请给出一个建议的网络模型架构设计

---

**Cursor**

### 这类任务在ML中的名称
- **关键点检测/地标定位（Keypoint/landmark localization）**：将图像映射到图像坐标系中的单个点。  
- 你的“每像素输出一个logit并取argmax”的设定，本质是**基于热力图的关键点检测（heatmap-based localization）/空间分类（dense spatial classification）**。  
- 也可视作**指向式定位（pointing / saliency-based pointing）**的单点版本。

### 建议的建模方式
- **推荐：热力图分类（Heatmap classification）**
  - Backbone产生低分辨率特征图 → 1通道logits热力图 → 上采样到输入分辨率。
  - 训练用**全图softmax + 交叉熵**（将目标像素视为唯一正类，其余为负类），或对目标周围做**高斯软标签**配合KL/MSE（容错更好）。
  - 推理用**argmax**得到坐标；若需亚像素，使用**soft-argmax**（期望坐标）。
- 可选：**坐标回归（x,y）**
  - 直接回归归一化坐标，L1/L2损失；简单但常不如热力图稳定，特别是在存在多峰/遮挡时。

### 参考网络架构（2D）
- **Backbone**（任选且与分辨率/速度需求匹配）：
  - 轻量CNN：`ResNet18/34`、`MobileNetV3`
  - 高分辨率保持：`HRNet`、`U-Net`
  - 视觉Transformer：`Swin-T`、`ViT`（需卷积/Deformable注意力做局部归纳偏置更稳）
- **Head**
  - 简单卷积头：`Conv 1x1 → (H/s, W/s, 1)` 得到单通道logits热力图；`bilinear`上采样到`H×W`。
  - 可加`Conv-BN-ReLU`堆叠或`ASPP`/`Deformable Conv`增强感受野。
- **Loss**
  - 若用一像素硬标签：将`H×W`展平，对每张图做`softmax`+`cross-entropy`。
  - 若用高斯软标签：对热力图做`log_softmax`后与归一化高斯做`KLDivLoss`；或`MSE`对比高斯热力图。
- **推理**
  - 粗定位：`argmax`。
  - 精定位：`soft-argmax`（温度τ可调）或在峰值邻域做二次插值。
- **训练要点**
  - 目标点下采样映射到`H/s × W/s`网格，或在高分辨率上训练再下采样特征。
  - 数据增强注意保持坐标一致：翻转/旋转/缩放时同步变换坐标。
  - 类别极度不平衡：优先用全图`softmax-CE`，如用BCE需`pos_weight`或`focal loss`。
  - 评价指标：像素误差、归一化距离（NME）、阈值内准确率（PCK）。

### 极简PyTorch示例（热力图分类 + argmax/soft-argmax）
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class PointLocator(nn.Module):
    def __init__(self, stride=4, pretrained=False):
        super().__init__()
        base = resnet18(weights=None if not pretrained else "DEFAULT")
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer2 = base.layer2  # stride 8 total
        self.layer3 = base.layer3  # stride 16
        self.layer4 = base.layer4  # stride 32
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )
        self.stride = 32  # resnet18默认到layer4为32倍下采样

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)                 # (B, 512, H/32, W/32)
        logits_low = self.head(x)          # (B, 1, H/32, W/32)
        logits = F.interpolate(logits_low, size=(H, W), mode="bilinear", align_corners=False)
        return logits  # 未经softmax的logits热力图

def ce_loss_over_pixels(logits, target_xy):  # logits: (B,1,H,W), target_xy: (B,2) in pixel (x,y)
    B, _, H, W = logits.shape
    # 将(H,W)展平为HW类
    flat_logits = logits.view(B, -1)  # (B, H*W)
    x = target_xy[:, 0].clamp(0, W-1).long()
    y = target_xy[:, 1].clamp(0, H-1).long()
    idx = y * W + x                    # (B,)
    return F.cross_entropy(flat_logits, idx)

def argmax_inference(logits):  # (B,1,H,W) -> (B,2)
    B, _, H, W = logits.shape
    idx = logits.view(B, -1).argmax(dim=1)
    y = (idx // W).float()
    x = (idx %  W).float()
    return torch.stack([x, y], dim=1)

def soft_argmax_inference(logits, tau=1.0):  # 温度τ越小越尖锐
    B, _, H, W = logits.shape
    p = F.softmax(logits.view(B, -1) / tau, dim=1).view(B, 1, H, W)
    xs = torch.linspace(0, W-1, W, device=logits.device).view(1,1,1,W)
    ys = torch.linspace(0, H-1, H, device=logits.device).view(1,1,H,1)
    x_exp = (p * xs).sum(dim=(2,3))
    y_exp = (p * ys).sum(dim=(2,3))
    return torch.cat([x_exp, y_exp], dim=1)
```

### 3D扩展思路
- 使用3D卷积Backbone，Head输出`D×H×W`的logits体素热力图；损失与推理同2D（展平到`DHW`）。

### 何时选哪种
- **场景模糊/多峰/遮挡**：热力图分类（配高斯软标签 + soft-argmax）。
- **坐标精度要求极高且标注精确**：soft-argmax或回归头可加权重约束与坐标正则。

- 以上设计即可直接用于你的“从图片到坐标”的训练与推理需求。

- 关键点
  - 任务名称：关键点/地标定位，基于热力图的空间分类。
  - 推荐方案：Backbone → 1通道logits热力图，上采样；全图softmax-CE；推理用argmax或soft-argmax。
  - 提供了极简PyTorch模型与损失/推理函数，可直接落地使用。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class PointLocator(nn.Module):
    def __init__(self, stride=4, pretrained=False):
        super().__init__()
        base = resnet18(weights=None if not pretrained else "DEFAULT")
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer2 = base.layer2  # stride 8 total
        self.layer3 = base.layer3  # stride 16
        self.layer4 = base.layer4  # stride 32
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )
        self.stride = 32  # resnet18默认到layer4为32倍下采样

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)                 # (B, 512, H/32, W/32)
        logits_low = self.head(x)          # (B, 1, H/32, W/32)
        logits = F.interpolate(logits_low, size=(H, W), mode="bilinear", align_corners=False)
        return logits  # 未经softmax的logits热力图

def ce_loss_over_pixels(logits, target_xy):  # logits: (B,1,H,W), target_xy: (B,2) in pixel (x,y)
    B, _, H, W = logits.shape
    # 将(H,W)展平为HW类
    flat_logits = logits.view(B, -1)  # (B, H*W)
    x = target_xy[:, 0].clamp(0, W-1).long()
    y = target_xy[:, 1].clamp(0, H-1).long()
    idx = y * W + x                    # (B,)
    return F.cross_entropy(flat_logits, idx)

def argmax_inference(logits):  # (B,1,H,W) -> (B,2)
    B, _, H, W = logits.shape
    idx = logits.view(B, -1).argmax(dim=1)
    y = (idx // W).float()
    x = (idx %  W).float()
    return torch.stack([x, y], dim=1)

def soft_argmax_inference(logits, tau=1.0):  # 温度τ越小越尖锐
    B, _, H, W = logits.shape
    p = F.softmax(logits.view(B, -1) / tau, dim=1).view(B, 1, H, W)
    xs = torch.linspace(0, W-1, W, device=logits.device).view(1,1,1,W)
    ys = torch.linspace(0, H-1, H, device=logits.device).view(1,1,H,1)
    x_exp = (p * xs).sum(dim=(2,3))
    y_exp = (p * ys).sum(dim=(2,3))
    return torch.cat([x_exp, y_exp], dim=1)
```

---

**User**

Follow up: 如果我希望把这个训练步骤作为一个使用强化学习做精细微调之前的预训练过程，你认为哪种方案更适合？

---

**Cursor**

### 更适合作为RL前置预训练的方案
- **首选：热力图分类（全图softmax + CE/KL）**
  - 理由：与后续RL的离散策略完全同构。把每个像素当成一个离散动作，`policy π(a|x) = softmax(logits(x))`。预训练的交叉熵就是对专家动作的最大似然；RL阶段可直接在同一组logits上做策略梯度（PPO/GRPO），几乎零摩擦衔接。
  - 优点：稳定、易训练、对多峰/模糊更鲁棒；RL只需接管同一头的logits，无需结构调整。
  - 做法：用硬标签（单像素）或高斯软标签（更平滑）做CE/KL；推理`argmax`，需要亚像素时用`soft-argmax`（只影响评估与可视化，不改变策略参数化）。

- 备选：连续坐标回归/分布（2D高斯）
  - 将策略参数化为连续分布（均值、方差、相关系数），预训练用NLL，RL用连续动作策略梯度。
  - 适用：强需求亚像素精度、动作空间极大且需要可控探索时。
  - 代价：实现复杂度更高，和离散像素评测/标注的天然匹配度不如热力图方案。

### 与RL无缝衔接的小建议
- **策略头复用**：将预训练的单通道logits作为策略头，RL阶段直接用作`π(a|x)`；加一个并行的`value head`做基线。
- **奖励设计**：
  - 距离型：`r = -|| (x̂,ŷ) - (x*,y*) ||` 或阈值命中奖励（近距高奖励）；可加形状化奖励（从粗到细的多阈值PCK）。
  - 若仍要亚像素监督，可在RL早期加入一个小权重的`soft-argmax`坐标MSE作为辅助损失，稳定训练。
- **温度与熵**：用温度缩放控制探索；策略熵正则防止过早收敛。
- **大分辨率效率**：
  - 分层/粗到细：先在低分辨率网格选区，再在裁剪区域细化；RL可做两步动作（选区→选点）。
  - 或保持网络下采样（stride）并上采样logits，保证可控的动作数。
- **混合训练**：RL前期保留少量监督CE（imitation loss）作为辅助项，逐步退火，能显著稳定并加速收敛。

### 简短结论
- **推荐用“热力图分类”作为预训练**：它与“像素即动作”的离散策略完全一致，预训练→RL参数无缝复用，稳定高效。  
- 如确需连续动作与亚像素强约束，再考虑“连续分布（2D高斯）”方案。

- 关键点
  - 预训练首选全图softmax-CE/KL，直接当作策略的行为克隆。
  - RL阶段沿用同一logits为策略头，加价值头与熵正则，结合距离型奖励与可选辅助监督。

---

**User**

好的！请在seg-rl目录下为我编写使用“热力图分类”方案的训练与推理示例代码，要求尽量方便修改扩展

---

**Cursor**

I'll create a modular training and inference pipeline under `seg-rl/` for heatmap-classification localization, including dataset, model, losses, utilities, CLI `train.py` and `infer.py`, plus a README. Then I'll implement each module and wire up the scripts.

I'll scaffold the `seg-rl` package and then implement dataset, model, losses/utils, and training/inference scripts.

I'll start by scaffolding the `seg-rl` package and creating core modules (`__init__.py`, `datasets.py`, `model.py`) in parallel, then continue with `losses.py`, `utils.py`, `train.py`, `infer.py`, and a concise `README.md`.

```python
"""seg_rl: Heatmap-based point localization training and inference.

Modules
-------
- datasets: Dataset and transforms for {image, x, y} supervision
- model: Backbone and heatmap head producing per-pixel logits
- losses: CE over pixels and KL/MSE to Gaussian targets
- utils: Coordinate utilities, metrics, visualization helpers
"""

__all__ = [
    "datasets",
    "model",
    "losses",
    "utils",
]


```

```python
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


```

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


@dataclass
class ModelConfig:
    backbone: Literal["resnet18"] = "resnet18"
    pretrained: bool = False
    upsample_to_input: bool = True


class HeatmapHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PointHeatmapModel(nn.Module):
    """Backbone + 1-channel heatmap logits head.

    Output: logits of shape [B,1,H',W'] optionally upsampled to input HxW.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if cfg.pretrained else None
            base = resnet18(weights=weights)
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2  # stride 8 total
            self.layer3 = base.layer3  # stride 16
            self.layer4 = base.layer4  # stride 32
            in_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {cfg.backbone}")

        self.head = HeatmapHead(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits_low = self.head(x)  # [B,1,h',w']
        if self.cfg.upsample_to_input:
            logits = F.interpolate(logits_low, size=(h, w), mode="bilinear", align_corners=False)
            return logits
        return logits_low


def argmax_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Get integer pixel coordinates from logits [B,1,H,W] -> [B,2] (x,y)."""
    b, _, h, w = logits.shape
    idx = logits.view(b, -1).argmax(dim=1)
    y = (idx // w).float()
    x = (idx % w).float()
    return torch.stack([x, y], dim=1)


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
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def ce_over_pixels(logits: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over all pixels treating each pixel as a class.

    Args:
        logits: [B,1,H,W] unnormalized scores
        target_xy: [B,2] target coordinates in pixels (x,y)
    Returns:
        scalar loss
    """
    b, _, h, w = logits.shape
    flat_logits = logits.view(b, -1)
    x = target_xy[:, 0].clamp(0, w - 1).long()
    y = target_xy[:, 1].clamp(0, h - 1).long()
    idx = y * w + x
    return F.cross_entropy(flat_logits, idx)


def gaussian_heatmap_targets(
    target_xy: torch.Tensor,
    height: int,
    width: int,
    sigma: float = 3.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Create Gaussian target heatmaps centered at target_xy.

    Args:
        target_xy: [B,2] (x,y)
        height, width: output map size
        sigma: std of Gaussian in pixels
        normalize: make per-sample heatmap sum to 1
    Returns:
        heatmaps: [B,1,H,W]
    """
    device = target_xy.device
    b = target_xy.shape[0]
    xs = torch.arange(width, device=device).view(1, 1, 1, width).float()
    ys = torch.arange(height, device=device).view(1, 1, height, 1).float()
    x0 = target_xy[:, 0].view(b, 1, 1, 1)
    y0 = target_xy[:, 1].view(b, 1, 1, 1)
    dist2 = (xs - x0) ** 2 + (ys - y0) ** 2
    heat = torch.exp(-0.5 * dist2 / (sigma ** 2))
    if normalize:
        heat = heat / (heat.sum(dim=(2, 3), keepdim=True) + 1e-8)
    return heat


def kl_to_gaussian_targets(logits: torch.Tensor, target_xy: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """KL divergence between model distribution and Gaussian soft targets.

    Treat p as soft target (Gaussian), q as model softmax over pixels, compute KL(p || q).
    """
    b, _, h, w = logits.shape
    with torch.no_grad():
        p = gaussian_heatmap_targets(target_xy, h, w, sigma=sigma, normalize=True)  # [B,1,H,W]
        p = p.clamp_min(1e-12)
    log_q = F.log_softmax(logits.view(b, -1), dim=1).view(b, 1, h, w)
    kl = (p * (p.log() - log_q)).sum(dim=(1, 2, 3))
    return kl.mean()


```

```python
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


```

```python
from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seg_rl.datasets import JsonlPointDataset, ImageSize, collate_fn
from seg_rl.model import ModelConfig, PointHeatmapModel, argmax_from_logits, soft_argmax_from_logits
from seg_rl.losses import ce_over_pixels, kl_to_gaussian_targets
from seg_rl.utils import save_checkpoint, compute_pck


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train heatmap classification point locator")
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, default=None)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", type=str, choices=["ce", "kl"], default="ce")
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma for KL loss")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="./outputs/seg_rl")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval_thresh", type=float, default=5.0, help="PCK threshold in pixels")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    image_size = ImageSize(height=args.height, width=args.width)
    train_ds = JsonlPointDataset(args.train_jsonl, image_size=image_size, training=True)
    val_ds = JsonlPointDataset(args.val_jsonl, image_size=image_size, training=False) if args.val_jsonl else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn) if val_ds else None

    cfg = ModelConfig(pretrained=args.pretrained)
    model = PointHeatmapModel(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        from seg_rl.utils import load_checkpoint
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler)
        start_epoch = ckpt.epoch
        global_step = ckpt.step

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
        print(f"Epoch {epoch+1}/{args.epochs} - train loss: {avg_loss:.4f}")

        # Validation
        if val_loader is not None:
            model.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    img = batch["image"].to(device, non_blocking=True)
                    tgt = batch["target_xy"].to(device, non_blocking=True)
                    logits = model(img)
                    pred_xy = soft_argmax_from_logits(logits)
                    d = torch.linalg.norm(pred_xy - tgt, dim=1)
                    correct += (d <= args.eval_thresh).sum().item()
                    total += d.numel()
            pck = correct / max(1, total)
            print(f"Epoch {epoch+1} - val PCK@{args.eval_thresh}: {pck:.4f}")

        # Save
        save_path = os.path.join(args.out_dir, f"model_epoch_{epoch+1}.pt")
        save_checkpoint(save_path, model, optimizer, scaler, epoch=epoch+1, step=global_step)


if __name__ == "__main__":
    main()


```

```python
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from seg_rl.model import ModelConfig, PointHeatmapModel, argmax_from_logits, soft_argmax_from_logits


def load_image(path: str, height: int, width: int):
    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), resample=Image.BILINEAR)
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return img, t.unsqueeze(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for heatmap-based point localization")
    p.add_argument("--images", type=str, nargs="+", help="One or more image paths or a directory")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--soft", action="store_true", help="Use soft-argmax instead of argmax")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--save_json", type=str, default=None)
    return p.parse_args()


def expand_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                    out.append(os.path.join(p, name))
        else:
            out.append(p)
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ModelConfig(pretrained=False)
    model = PointHeatmapModel(cfg).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    # supports both raw model state and full checkpoint
    sd = state.get("model", state)
    model.load_state_dict(sd)
    model.eval()

    paths = expand_paths(args.images)
    results = []
    for p in paths:
        img_pil, t = load_image(p, args.height, args.width)
        t = t.to(device)
        with torch.no_grad():
            logits = model(t)
            if args.soft:
                xy = soft_argmax_from_logits(logits, temperature=args.temperature)[0]
            else:
                xy = argmax_from_logits(logits)[0]
        x, y = float(xy[0].item()), float(xy[1].item())
        print(f"{p}: x={x:.1f}, y={y:.1f}")
        results.append({"image": p, "x": x, "y": y})

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


```

```markdown
*** Begin Patch
*** Add File: /Users/donchiu/Works/Seg-R1/seg-rl/README.md
+# seg-rl: Heatmap-based Point Localization (pretraining-ready)

This package provides a clean training/inference pipeline for mapping an image to a single point coordinate using heatmap classification. It is designed to be easily extended and to serve as a pretraining stage before RL fine-tuning.

## Data format

Use JSONL for supervision, one object per line:

```json
{"image": "/abs/path/img_0001.jpg", "x": 123, "y": 456}
```

- Coordinates are in pixel space of the original image. The loader resizes images to `H×W` and scales coordinates accordingly.

## Quickstart

Train:

```bash
python -m seg-rl.train --train_jsonl /abs/path/train.jsonl \
  --val_jsonl /abs/path/val.jsonl --height 512 --width 512 \
  --batch_size 16 --epochs 20 --loss ce --amp --out_dir /abs/path/outputs
```

Infer:

```bash
python -m seg-rl.infer --ckpt /abs/path/outputs/model_epoch_20.pt \
  --images /abs/path/images_dir --height 512 --width 512 --soft
```

## Modules

- `seg_rl.datasets`: JSONL dataset, resize with coordinate transforms, simple flips.
- `seg_rl.model`: ResNet18 backbone + 1-channel heatmap head; argmax/soft-argmax.
- `seg_rl.losses`: CE over pixels; KL to Gaussian soft targets.
- `seg_rl.utils`: Checkpoint I/O, PCK metric, simple visualization.

## Notes

- For high-res inputs, consider a higher-resolution backbone (HRNet/UNet) or a coarse-to-fine two-stage model.
- For RL fine-tuning, reuse the same logits as a discrete policy over pixels; add a value head.

*** End Patch
```

I'll move the training and inference scripts into the `seg_rl` package so they can be invoked as modules, add a README with usage (`python -m seg_rl.*`), and then run lint checks to fix any issues.

```python
from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import JsonlPointDataset, ImageSize, collate_fn
from .model import ModelConfig, PointHeatmapModel, argmax_from_logits, soft_argmax_from_logits
from .losses import ce_over_pixels, kl_to_gaussian_targets
from .utils import save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train heatmap classification point locator")
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, default=None)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", type=str, choices=["ce", "kl"], default="ce")
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma for KL loss")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="./outputs/seg_rl")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval_thresh", type=float, default=5.0, help="PCK threshold in pixels")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    image_size = ImageSize(height=args.height, width=args.width)
    train_ds = JsonlPointDataset(args.train_jsonl, image_size=image_size, training=True)
    val_ds = JsonlPointDataset(args.val_jsonl, image_size=image_size, training=False) if args.val_jsonl else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn) if val_ds else None

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
        print(f"Epoch {epoch+1}/{args.epochs} - train loss: {avg_loss:.4f}")

        # Validation
        if val_loader is not None:
            model.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    img = batch["image"].to(device, non_blocking=True)
                    tgt = batch["target_xy"].to(device, non_blocking=True)
                    logits = model(img)
                    pred_xy = soft_argmax_from_logits(logits)
                    d = torch.linalg.norm(pred_xy - tgt, dim=1)
                    correct += (d <= args.eval_thresh).sum().item()
                    total += d.numel()
            pck = correct / max(1, total)
            print(f"Epoch {epoch+1} - val PCK@{args.eval_thresh}: {pck:.4f}")

        # Save
        save_path = os.path.join(args.out_dir, f"model_epoch_{epoch+1}.pt")
        save_checkpoint(save_path, model, optimizer, scaler, epoch=epoch+1, step=global_step)


if __name__ == "__main__":
    main()


```

```python
from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from .model import ModelConfig, PointHeatmapModel, argmax_from_logits, soft_argmax_from_logits


def load_image(path: str, height: int, width: int):
    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), resample=Image.BILINEAR)
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return img, t.unsqueeze(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for heatmap-based point localization")
    p.add_argument("--images", type=str, nargs="+", help="One or more image paths or a directory")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--soft", action="store_true", help="Use soft-argmax instead of argmax")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--save_json", type=str, default=None)
    return p.parse_args()


def expand_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                    out.append(os.path.join(p, name))
        else:
            out.append(p)
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ModelConfig(pretrained=False)
    model = PointHeatmapModel(cfg).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    # supports both raw model state and full checkpoint
    sd = state.get("model", state)
    model.load_state_dict(sd)
    model.eval()

    paths = expand_paths(args.images)
    results = []
    for p in paths:
        _, t = load_image(p, args.height, args.width)
        t = t.to(device)
        with torch.no_grad():
            logits = model(t)
            if args.soft:
                xy = soft_argmax_from_logits(logits, temperature=args.temperature)[0]
            else:
                xy = argmax_from_logits(logits)[0]
        x, y = float(xy[0].item()), float(xy[1].item())
        print(f"{p}: x={x:.1f}, y={y:.1f}")
        results.append({"image": p, "x": x, "y": y})

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


```

```markdown
## seg-rl: Heatmap-based Point Localization (pretraining-ready)

This package provides a clean training/inference pipeline for mapping an image to a single point coordinate using heatmap classification. It is designed to be easily extended and to serve as a pretraining stage before RL fine-tuning.

### Data format

Use JSONL for supervision, one object per line:

```json
{"image": "/abs/path/img_0001.jpg", "x": 123, "y": 456}
```

- Coordinates are in pixel space of the original image. The loader resizes images to H×W and scales coordinates accordingly.

### Quickstart

Train:

```bash
python -m seg_rl.train --train_jsonl /abs/path/train.jsonl \
  --val_jsonl /abs/path/val.jsonl --height 512 --width 512 \
  --batch_size 16 --epochs 20 --loss ce --amp --out_dir /abs/path/outputs
```

Infer:

```bash
python -m seg_rl.infer --ckpt /abs/path/outputs/model_epoch_20.pt \
  --images /abs/path/images_dir --height 512 --width 512 --soft
```

### Modules

- seg_rl.datasets: JSONL dataset, resize with coordinate transforms, simple flips.
- seg_rl.model: ResNet18 backbone + 1-channel heatmap head; argmax/soft-argmax.
- seg_rl.losses: CE over pixels; KL to Gaussian soft targets.
- seg_rl.utils: Checkpoint I/O, PCK metric, simple visualization.

### Notes

- For high-res inputs, consider a higher-resolution backbone (HRNet/UNet) or a coarse-to-fine two-stage model.
- For RL fine-tuning, reuse the same logits as a discrete policy over pixels; add a value head.


```

I scaffolded a modular heatmap-classification pipeline under seg-rl, added dataset/model/losses/utils plus train and infer entrypoints, and wrote a README. You can run training/inference via python -m seg_rl.*.

- Train: python -m seg_rl.train --train_jsonl /abs/path/train.jsonl --val_jsonl /abs/path/val.jsonl --height 512 --width 512 --batch_size 16 --epochs 20 --loss ce --amp --out_dir /abs/path/outputs
- Infer: python -m seg_rl.infer --ckpt /abs/path/outputs/model_epoch_20.pt --images /abs/path/images_dir --height 512 --width 512 --soft

- Created package `seg_rl` with:
  - `datasets.py`: JSONL {image,x,y} loader with resizing and flip-safe coord transforms; `collate_fn`.
  - `model.py`: `PointHeatmapModel` (ResNet18 backbone + 1ch heatmap head), `argmax_from_logits`, `soft_argmax_from_logits`.
  - `losses.py`: `ce_over_pixels`, `kl_to_gaussian_targets` and Gaussian target generator.
  - `utils.py`: checkpoint save/load, PCK metric, simple overlay.
  - `train.py`: AMP, AdamW, CE/KL selectable, PCK@thresh eval, checkpointing.
  - `infer.py`: load ckpt or raw state dict, directory expansion, argmax/soft-argmax, optional JSON dump.
  - `README.md`: data format and usage.

Let me know if you want HRNet/UNet backbone, coarse-to-fine two-stage, or RL heads added.

---

**User**

请用中文说明你添加的各文件作用、如何使用、以及你的详细设计思路

---

**Cursor**

### 文件与作用

- `seg_rl/__init__.py`
  - **作用**: 将包导出为模块，方便 `python -m seg_rl.*` 运行与 IDE 自动补全。
- `seg_rl/datasets.py`
  - **核心类**: `JsonlPointDataset`（读取 JSONL 格式监督：`{"image": "...", "x": ..., "y": ...}`）。
  - **功能**: 
    - 统一缩放图像到 `H×W`，并按比例缩放坐标。
    - 训练期随机左右/上下翻转（默认仅左右翻），并同步变换坐标。
    - `to_tensor` + 标准化（ImageNet 均值/方差）。
    - `collate_fn` 用于 DataLoader。
  - **辅助**: `ImageSize` 用于清晰传参。
- `seg_rl/model.py`
  - **配置**: `ModelConfig`（当前支持 `resnet18`，可开关预训练，是否上采样回输入分辨率）。
  - **模型**: `PointHeatmapModel` = Backbone(ResNet18) → `HeatmapHead`（1 通道 logits 热力图）。
  - **推理工具**: `argmax_from_logits`（像素级整点），`soft_argmax_from_logits`（亚像素期望值，可调温度）。
- `seg_rl/losses.py`
  - `ce_over_pixels`：将每个像素当作一个“类别”，对全图做 `softmax + CE`。
  - `gaussian_heatmap_targets`：以目标点为中心生成高斯软标签热力图。
  - `kl_to_gaussian_targets`：对 `logits` 的 `softmax` 与高斯软标签做 `KL(p||q)`。
- `seg_rl/utils.py`
  - **训练辅助**: `save_checkpoint` / `load_checkpoint`（保存/恢复模型+优化器+AMP scaler、epoch、step）。
  - **评测**: `compute_pck`（阈值内正确率 PCK）。
  - **可视化**: `overlay_point_on_image`（在 PIL 图上画点）。
- `seg_rl/train.py`
  - **CLI 训练脚本**：加载数据、构建模型、选择损失（CE 或 KL）、AMP 训练、验证 PCK、按 epoch 保存权重、支持恢复训练。
- `seg_rl/infer.py`
  - **CLI 推理脚本**：加载 checkpoint，对单图/目录推理，支持 `argmax` 或 `soft-argmax`，可导出 JSON 预测结果。
- `seg-rl/README.md`
  - **用法文档**：数据格式、训练/推理命令、模块说明、扩展建议。

### 如何使用

- 数据准备（JSONL，一行一个样本）:
```json
{"image": "/abs/path/img_0001.jpg", "x": 123, "y": 456}
```

- 训练:
```bash
python -m seg_rl.train --train_jsonl /abs/path/train.jsonl \
  --val_jsonl /abs/path/val.jsonl --height 512 --width 512 \
  --batch_size 16 --epochs 20 --loss ce --amp --out_dir /abs/path/outputs
```
  - 常用参数:
    - `--loss ce|kl`：像素分类交叉熵或高斯软标签 KL。
    - `--sigma`：KL 中高斯 σ（像素）。
    - `--pretrained`：是否用 ResNet18 预训练权重。
    - `--resume`：恢复训练的权重路径。
    - `--eval_thresh`：验证 PCK 的像素阈值。

- 推理:
```bash
python -m seg_rl.infer --ckpt /abs/path/outputs/model_epoch_20.pt \
  --images /abs/path/images_dir --height 512 --width 512 --soft
```
  - `--soft`：使用 soft-argmax（亚像素）。
  - `--temperature`：soft-argmax 温度。
  - `--save_json`：将预测列表保存为 JSON。

### 详细设计思路

- **任务建模（热力图分类）**
  - 将输入图的每个像素视为一个离散动作/类别，输出单通道 logits 热力图，用全图 `softmax` 得到像素分布。
  - 监督形式既可用硬标签（目标像素唯一正类，`CE`），也可用高斯软标签（更平滑、容错更好，`KL`）。

- **模型架构**
  - 选用 **ResNet18** 作为通用、稳定的 CNN Backbone，兼顾速度/准确度。
  - 使用 `HeatmapHead`（`Conv-BN-ReLU-Conv1x1`）输出 1 通道 logits；默认插值回输入分辨率，方便与像素坐标直接对齐。
  - 推理侧支持：
    - `argmax`：整像素定位，速度快。
    - `soft-argmax`：计算分布期望，得到亚像素坐标，精度更高（可控温度平滑）。

- **损失函数设计**
  - `CE`：将 `H×W` 像素展开为 `HW` 类别，直接做交叉熵，优化目标与“像素分类”一致，适合与后续 RL 的“离散策略”对接。
  - `KL`：构造高斯软标签，分布对分布的匹配，能缓解标注噪声/下采样误差；σ 控制软化程度。
  - 两者都在 `logits` 上作用，不需要人为阈值。

- **数据与增强**
  - 读入原图坐标，统一 resize 到 `H×W`，同步线性缩放坐标，保证监督一致。
  - 训练期随机左右翻转（默认），可选上下翻转，确保坐标同步变换。
  - 标准化使用 ImageNet 均值/方差，便于复用预训练特征（若开启 `--pretrained`）。

- **评估指标**
  - 默认用 PCK（像素距离阈值内正确率），更贴近“是否落在附近”的业务评估。
  - 可替换/补充为平均距离、NME 等。

- **可扩展性与改造点**
  - **Backbone**：替换为 HRNet/UNet/Swin 等；只需改 `ModelConfig` 和构建 `in_channels` 即可。
  - **多点/多类**：将 `HeatmapHead` 输出改为 `C` 通道（每类/每点一通道），损失对每通道分别计算。
  - **粗到细**：先在低分辨率全图选区，再对裁剪块细化（第二阶段 head）；非常适合高分辨率场景。
  - **坐标连续分布**：可扩为回归连续高斯参数（μ, σ, ρ），适合严格亚像素需求（但与离散 RL 对接会复杂一些）。
  - **保持纵横比**：目前直接缩放；如需保纵横比，可加黑边/Pad 与对应坐标偏移映射。
  - **3D 扩展**：用 3D Conv/backbone，输出 `D×H×W` 体素热力图；损失与 2D 同理（把 `DHW` 当类）。

- **与强化学习的衔接**
  - 预训练阶段的热力图 logits 可直接作为 RL 的离散策略 `π(a|x)`；RL 仅需在同一头上接入策略梯度（另加价值头即可）。
  - 可在 RL 早期混入小权重的监督项（CE/KL）作为辅助稳定器，逐步退火。

- **工程考虑**
  - 训练脚本支持 AMP、AdamW、断点续训与按 epoch 存档，便于大规模训练。
  - 推理脚本支持目录展开、JSON 输出，便于批处理与下游对接。

### 常见改动点（快速指引）

- 换损失为 KL（高斯软标签）：
  - 训练命令加 `--loss kl --sigma 3.0`（根据图像分辨率与目标大小调 σ）。
- 提高分辨率或精度：
  - 增大 `--height/--width`，或替换高分辨率保持更好的 Backbone（如 HRNet/UNet）。
- 用预训练权重：
  - 训练命令加 `--pretrained`。
- 改增强策略（翻转概率等）：
  - 在 `JsonlPointDataset` 初始化时修改 `hflip_p` / `vflip_p`（当前脚本中为固定值，若需 CLI 控制可在 `train.py` 中添加对应参数并传入）。

- 关键点
  - 每个文件的职责清晰：数据/模型/损失/工具/训练/推理/文档。
  - 训练与推理都通过 `python -m seg_rl.*` 调用，参数化良好，易扩展。
  - 设计以“热力图分类”为核心，天然与后续离散策略的 RL 微调无缝衔接。

```json
{"image": "/abs/path/img_0001.jpg", "x": 123, "y": 456}
```

```shellscript
python -m seg_rl.train --train_jsonl /abs/path/train.jsonl \
  --val_jsonl /abs/path/val.jsonl --height 512 --width 512 \
  --batch_size 16 --epochs 20 --loss ce --amp --out_dir /abs/path/outputs
```

```shellscript
python -m seg_rl.infer --ckpt /abs/path/outputs/model_epoch_20.pt \
  --images /abs/path/images_dir --height 512 --width 512 --soft
```

---

**User**

请确保你的训练脚本seg_rl.train中包含以下功能：1. 支持按一定epochs保存断点和从断点开始继续训练；2. 绘制损失和准确率（包括val和test）曲线以供评估训练过程；3. 可视化最终的实际热力图分类效果（对比GT和predicted结果，比如对真值点以绿色十字表示，对预测点以红色三角表示），调用者可根据传入的运行参数选择，是从全体数据集中抽取少量来展示（生成一个对比图片），还是对全体数据集每个样本生成一个对比图存放在另一个专门的结果目录。

---

**Cursor**

I'll update the training pipeline to add checkpoint frequency and resume options, logging and plotting of loss/accuracy (val/test), and visualization utilities (GT vs predicted, sample or per-sample). I'll first extend `seg_rl.utils` with drawing/overlay helpers, then modify `seg_rl.train` with new CLI flags, test split, plotting, and visualization routines.

I'll add visualization utilities (cross and triangle markers, heatmap rendering, grid composer) and then update the training script to support periodic checkpoints, resume, plotting loss/PCK curves, and dataset visualizations per your options.

Adding visualization utilities and enhancing training with checkpoints, plots, and visualization as requested. Now I'll implement the helper functions in seg_rl.utils, then update seg_rl.train with new CLI flags, logging, evaluation, plotting, and visualization flows.

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


def draw_triangle(image: Image.Image, xy: Tuple[float, float], color: Tuple[int, int, int] = (255, 0, 0), size: int = 7, width: int = 2) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x, y = xy
    pts = [
        (x, y - size),
        (x - size * 0.866, y + size * 0.5),
        (x + size * 0.866, y + size * 0.5),
    ]
    draw.polygon(pts, outline=color)
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
from torch.utils.data import DataLoader

from .datasets import JsonlPointDataset, ImageSize, collate_fn
from .model import ModelConfig, PointHeatmapModel, argmax_from_logits, soft_argmax_from_logits
from .losses import ce_over_pixels, kl_to_gaussian_targets
from .utils import save_checkpoint
from .utils import draw_cross, draw_triangle, overlay_heatmap, make_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train heatmap classification point locator")
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, default=None)
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
    p.add_argument("--test_jsonl", type=str, default=None)
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
    train_ds = JsonlPointDataset(args.train_jsonl, image_size=image_size, training=True)
    val_ds = JsonlPointDataset(args.val_jsonl, image_size=image_size, training=False) if args.val_jsonl else None
    test_ds = JsonlPointDataset(args.test_jsonl, image_size=image_size, training=False) if args.test_jsonl else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn) if test_ds else None

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
                probs = torch.softmax(logits, dim=2 if logits.dim() == 4 else 1)  # safe softmax over flattened handled later
                # compute soft-argmax pred on CPU for simplicity
                pred_xy = soft_argmax_from_logits(logits).cpu()
                for i in range(img_t.size(0)):
                    if count >= max_count:
                        break
                    pil_img = TF.to_pil_image(img_t[i])
                    hm = logits[i, 0].detach().cpu().float().numpy()
                    over = overlay_heatmap(pil_img, hm, alpha=0.5)
                    # draw GT and Pred
                    over = draw_cross(over, (float(tgt[i, 0]), float(tgt[i, 1])), color=(0, 255, 0))
                    over = draw_triangle(over, (float(pred_xy[i, 0]), float(pred_xy[i, 1])), color=(255, 0, 0))
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

- 已新增功能均已在 `seg_rl.train` 和 `seg_rl.utils` 中实现，并通过静态检查。
- 我现在已完成：断点保存/恢复频率控制、训练/验证/测试曲线绘制、以及按“抽样/全量”两种可视化模式输出对比图。

使用方法简述
- 断点与恢复
  - 保存频率：加 `--save_every N`（每 N 个 epoch 保存一次）。
  - 从断点恢复：加 `--resume /abs/path/model_epoch_XX.pt`。
- 曲线绘制
  - 自动在 `--out_dir/plots`（或 `--plots_dir` 指定）下保存两张图：
    - `loss.png`：epoch 对训练 loss
    - `pck.png`：epoch 对 val/test 的 PCK@阈值曲线
- 数据集切分
  - 训练：`--train_jsonl`
  - 验证：`--val_jsonl`
  - 测试：`--test_jsonl`
- 可视化（GT vs Pred）
  - 选择模式：`--vis_mode none|sample|all`
    - sample：默认取 `--vis_count`（默认 16）个样本拼成单张网格图；文件名：`vis/val_or_train_vis.png`
    - all：对分割集内每个样本生成一张对比图，保存到 `vis/{split}/` 目录
  - 输出目录：默认 `--out_dir/vis`，可通过 `--vis_dir` 指定
  - 图中标记：
    - 绿色十字：GT 坐标
    - 红色三角：预测坐标（使用 soft-argmax）
    - 背景：输入图与预测热力图的半透明叠加

示例命令
- 训练并每 2 个 epoch 保存、绘图、抽样可视化：
```bash
python -m seg_rl.train \
  --train_jsonl /abs/train.jsonl --val_jsonl /abs/val.jsonl --test_jsonl /abs/test.jsonl \
  --height 512 --width 512 --batch_size 16 --epochs 20 --loss ce --amp \
  --eval_thresh 5.0 --save_every 2 --out_dir /abs/outputs \
  --vis_mode sample --vis_count 16
```
- 从断点恢复并全量可视化验证集：
```bash
python -m seg_rl.train \
  --train_jsonl /abs/train.jsonl --val_jsonl /abs/val.jsonl \
  --height 512 --width 512 --epochs 10 --resume /abs/outputs/model_epoch_10.pt \
  --vis_mode all --vis_dir /abs/outputs/vis_all
```

小提示
- 若缺少 `matplotlib`，请先安装以生成曲线图。
- PCK 阈值通过 `--eval_thresh` 控制；可按数据分辨率调整。
- 可视化默认使用验证集；若无验证集，则回退用训练集样本。

- 我已：
  - 在 `seg_rl.utils` 添加十字/三角标记、热力图上色与叠加、网格拼图。
  - 在 `seg_rl.train` 添加 test 集、保存频率、恢复、曲线绘制、两种可视化模式。

```shellscript
python -m seg_rl.train \
  --train_jsonl /abs/train.jsonl --val_jsonl /abs/val.jsonl --test_jsonl /abs/test.jsonl \
  --height 512 --width 512 --batch_size 16 --epochs 20 --loss ce --amp \
  --eval_thresh 5.0 --save_every 2 --out_dir /abs/outputs \
  --vis_mode sample --vis_count 16
```

```shellscript
python -m seg_rl.train \
  --train_jsonl /abs/train.jsonl --val_jsonl /abs/val.jsonl \
  --height 512 --width 512 --epochs 10 --resume /abs/outputs/model_epoch_10.pt \
  --vis_mode all --vis_dir /abs/outputs/vis_all
```

