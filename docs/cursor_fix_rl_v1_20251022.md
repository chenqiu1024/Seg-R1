# 解决强化学习训练中的ValueError错误
_Exported on 10/22/2025 at 02:01:05 PDT from Cursor (1.7.53)_

---

**User**

请阅读这两个文件，.py是强化学习训练脚本，.md是我与Cursor关于它运行时错误的一系列对话和操作的记录；另外还有我粘贴的运行时得到的命令行错误信息。你还需要知道的是，.md中记录的之前Cursor所尝试的修改方案，即把模型中的BatchNorm改为GroupNorm，是有问题的，会让模型在使用同样预训练权重的情况下推理性能变差，也无法通过强化学习来提升。所以我不希望你修改原来的模型代码本身，在这种前提下，能否仅靠修改训练脚本本身使得RL训练可以正常进行而不发生“ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])”的错误？

---

**Cursor**

I'll add a small helper that keeps only the BatchNorm layers inside the label head in eval mode during RL training, and call it after each time the model is switched to train mode so batch=1 no longer triggers the BN error, without changing the model code. Then I'll lint the updated script.

```python
#!/usr/bin/env python3

"""
GRPO fine-tuning for the point-policy model used to propose next SAM2 prompt points.

Overview
- Policy: seg-rl.heatmap.model.PointHeatmapModel that outputs pixel logits [B,1,H,W]
          and label logits [B,2]. We use its joint hierarchical policy over
          (label, cell, subpixel) with helpers already implemented in model.py.
- Environment: SAM2 predictor turns the accumulated point sequence into a mask.
- Observation: (image RGB, previous predicted mask) → model → logits, label_logits.
- Action: sample joint (label, cell, subpixel) → continuous xy in input grid.
- Reward: per-step improvement in a segmentation metric (e.g., DICE) vs previous step.
- GRPO: group-relative PPO-style objective with ratio clipping and KL to a fixed ref.

Key Features
- CLI with defaults; checkpoint save/resume; TensorBoard logging; optional step viz.
- Temperature schedule for exploration (pixel/label) that anneals across training.
- Validation: greedy rollout to report mean metrics.

Notes
- Input resolution H×W is configurable; actions are mapped back to original image size
  before calling SAM2. Masks from SAM2 are resized to original size for metric eval.
- For efficiency, we keep a single SAM2 predictor per process and reuse it across steps.

Examples (调用示例)
  1) 基本训练（使用已监督预训练权重作为初始化与参考网络）：
     /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.train_grpo_points \
       --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl \
       --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
       --out_dir outputs/braintumour/grpo-251020 \
       --init_policy pretrained/points_predictor-251001-160epochs.pt \
       --device cuda \
       --height 512 --width 512 --stride 8 --max_points 16 \
       --epochs 5 --batch_size 1 --group_size 4 --tb

  2) 在Apple Silicon上使用MPS并调整探索温度与KL权重：
     python -m seg-rl.heatmap.train_grpo_points \
       --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl \
       --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
       --out_dir outputs/braintumour/grpo-mps \
       --device mps --height 512 --width 512 --stride 8 \
       --pixel_temp_start 1.8 --pixel_temp_end 0.8 \
       --label_temp_start 1.2 --label_temp_end 0.8 \
       --beta_kl 0.02 --beta_kl_label 0.3 --beta_kl_pixel 0.7 \
       --epochs 3 --batch_size 2 --group_size 4 --tb

  3) 从断点恢复训练：
     python -m seg-rl.heatmap.train_grpo_points \
       --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl \
       --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
       --out_dir outputs/braintumour/grpo-251015 \
       --resume outputs/braintumour/grpo-251015/ckpt_step1000.pt --tb
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Local imports (robust to module/script execution)
try:
    from .model import (
        ModelConfig,
        PointHeatmapModel,
        sample_joint_label_cell_offset,
        log_prob_of_joint_action,
        soft_argmax_from_logits,
    )
except Exception:
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.abspath(__file__))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from model import (  # type: ignore
        ModelConfig,
        PointHeatmapModel,
        sample_joint_label_cell_offset,
        log_prob_of_joint_action,
        soft_argmax_from_logits,
    )


def _try_import_render_helper() -> Optional[Any]:
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # seg-rl
        target = os.path.join(base_dir, "annotator", "gen_point_jsonl_from_masks.py")
        import importlib.util as _util
        spec = _util.spec_from_file_location("gen_point_jsonl_from_masks", target)
        if not spec or not spec.loader:
            return None
        module = _util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        func = getattr(module, "_render_diff_panels_with_point", None)
        return func if callable(func) else None
    except Exception:
        return None


_render_diff_panels_with_point = _try_import_render_helper()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def _load_mask_bool(path: str) -> np.ndarray:
    arr = np.array(Image.open(path), dtype=np.uint8)
    if arr.ndim == 3:
        arr = (arr.any(axis=2)).astype(np.uint8)
    return arr > 0


def _compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
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

    # lightweight S-measure surrogate (optional)
    s_meas = float(0.0) ## FIXME: Do actual computing of S-measure
    return {
        "DICE": float(dice),
        "IOU": float(iou),
        "PRECISION": float(precision),
        "RECALL": float(recall),
        "F1": float(f1),
        "S_MEASURE": float(s_meas),
    }


def _metric_value(metrics: Dict[str, float], key: str) -> float:
    return float(metrics.get(key.upper(), 0.0))


def _resize_bool(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.array(Image.fromarray((arr.astype(np.uint8) * 255)).resize((w, h), resample=Image.NEAREST)) > 0


class JsonArrayDataset(Dataset):
    def __init__(self, json_path: str, split: str, val_ratio: float = 0.1, seed: int = 42) -> None:
        body = json.loads(open(json_path, "r", encoding="utf-8").read())
        if not isinstance(body, list):
            raise RuntimeError("input JSON must be an array of records")
        items: List[Dict[str, Any]] = []
        for rec in body:
            if not isinstance(rec, dict):
                continue
            img = rec.get("image")
            gt = rec.get("gt_mask")
            if not (img and gt and os.path.isfile(img) and os.path.isfile(gt)):
                continue
            items.append({"image": os.path.abspath(img), "gt_mask": os.path.abspath(gt)})
        # deterministic split
        rng = random.Random(seed)
        rng.shuffle(items)
        n_val = int(len(items) * val_ratio)
        self.items = items[:n_val] if split == "val" else items[n_val:]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.items[idx]


# --- SAM2 integration -------------------------------------------------------


class SAMEnv:
    def __init__(self, sam_checkpoint: str, device: Optional[str] = None) -> None:
        # defer heavy import to runtime
        import sys as _sys
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        third_party = os.path.join(base_dir, "third_party", "sam2")
        _sys.path.append(third_party)
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        if device is None:
            dev = _device()
        else:
            dev = torch.device(device)

        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model = build_sam2(model_cfg, sam_checkpoint, device=dev)
        self.predictor = SAM2ImagePredictor(sam_model)

    def predict_mask(self, image_rgb: np.ndarray, points: List[Tuple[float, float]], labels: List[int]) -> np.ndarray:
        self.predictor.set_image(image_rgb)
        import numpy as _np
        pts = _np.array(points) if points else None
        lbs = _np.array(labels) if labels else None
        mask, score, logits = self.predictor.predict(
            point_coords=pts,
            point_labels=lbs,
            box=None,
            multimask_output=False,
        )
        return mask[0]


def _load_image_rgb(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im)


def _prepare_model_inputs(image_path: str, prev_mask_u8: Optional[np.ndarray], out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    im0 = Image.open(image_path).convert("RGB")
    orig_w, orig_h = im0.size
    H, W = out_hw
    if H <= 0 or W <= 0:
        H, W = orig_h, orig_w
    im = im0.resize((W, H), resample=Image.BILINEAR) if (W, H) != (orig_w, orig_h) else im0
    if prev_mask_u8 is None:
        gray = Image.new("L", (orig_w, orig_h), 0)
    else:
        gray = Image.fromarray(prev_mask_u8.astype(np.uint8))
    gray = gray.resize((W, H), resample=Image.NEAREST) if (W, H) != (orig_w, orig_h) else gray
    import torchvision.transforms.functional as TF
    rgb_t = TF.to_tensor(im)
    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    g_t = TF.to_tensor(gray)
    g_t = (g_t - 0.5) / 0.5
    return rgb_t.unsqueeze(0), g_t.unsqueeze(0), (orig_h, orig_w)


@dataclass
class GRPOArgs:
    # data/env
    train_json: str
    sam_checkpoint: str
    out_dir: str
    height: int = 512
    width: int = 512
    stride: int = 8
    max_points: int = 16
    metric: str = "DICE"
    # training
    epochs: int = 5
    batch_size: int = 4
    group_size: int = 4  # G
    lr: float = 1e-4
    weight_decay: float = 1e-4
    clip_eps: float = 0.2
    beta_kl: float = 0.01
    beta_kl_label: float = 0.2
    beta_kl_pixel: float = 0.8
    # exploration temperatures
    pixel_temp_start: float = 1.5
    pixel_temp_end: float = 0.7
    label_temp_start: float = 1.0
    label_temp_end: float = 0.7
    # logging/checkpoint
    save_every: int = 200
    val_every: int = 200
    resume: Optional[str] = None
    tb: bool = True
    seed: int = 42
    device: Optional[str] = None
    # pretrained policy
    init_policy: Optional[str] = None  # path to supervised checkpoint (.pt)


def _interp(start: float, end: float, ratio: float) -> float:
    ratio = max(0.0, min(1.0, ratio))
    return start + (end - start) * ratio


def _calc_kl_categorical(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    # KL(P||Q) with logits [B, K]
    logp = torch.log_softmax(logits_p, dim=1)
    logq = torch.log_softmax(logits_q, dim=1)
    p = torch.softmax(logits_p, dim=1)
    return (p * (logp - logq)).sum(dim=1)


def _calc_kl_pixel_full(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    # KL over flattened pixel distribution [B,1,H,W] → [B]
    B = logits_p.size(0)
    logp = torch.log_softmax(logits_p.view(B, -1), dim=1)
    logq = torch.log_softmax(logits_q.view(B, -1), dim=1)
    p = torch.softmax(logits_p.view(B, -1), dim=1)
    return (p * (logp - logq)).sum(dim=1)


def _set_label_bn_eval(model: nn.Module) -> None:
    """Force BatchNorm layers under label head to eval mode to support B=1.

    This avoids BN training with N*H*W==1 (from AdaptiveAvgPool2d(1)) while keeping
    the rest of the network in training mode. It does not change requires_grad.
    Call this after any .train() call that might reset submodules.
    """
    for name, module in model.named_modules():
        if "label_head" in name and isinstance(module, nn.BatchNorm2d):
            module.eval()


def _save_checkpoint(path: str, model: nn.Module, optim: torch.optim.Optimizer, epoch: int, step: int, args: GRPOArgs, ref_sd: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": args.__dict__,
        "ref_model": ref_sd,
    }, path)


def _load_checkpoint(path: str, model: nn.Module, optim: torch.optim.Optimizer) -> Tuple[int, int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    if "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("step", 0)), dict(ckpt.get("ref_model", {}))


def validate(model: PointHeatmapModel, dataset: Dataset, env: SAMEnv, device: torch.device, args: GRPOArgs, max_items: int = 64) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    metrics_list: List[float] = []
    with torch.no_grad():
        for i, rec in enumerate(loader):
            if i >= max_items:
                break
            image_path = rec["image"][0]
            gt_path = rec["gt_mask"][0]
            gt_bool = _load_mask_bool(gt_path)

            points: List[Tuple[float, float]] = []
            labels: List[int] = []
            prev_mask: Optional[np.ndarray] = None
            best_val = 0.0
            for t in range(args.max_points):
                rgb_t, g_t, (orig_h, orig_w) = _prepare_model_inputs(image_path, prev_mask, (args.height, args.width))
                logits, label_logits = model(rgb_t.to(device), g_t.to(device))
                # greedy: combine heatmap soft-argmax and label argmax
                xy = soft_argmax_from_logits(logits, temperature=0.5)[0]
                x = float(xy[0].item())
                y = float(xy[1].item())
                lab = int(label_logits.argmax(dim=1).item())
                # map back to original
                inf_H, inf_W = rgb_t.shape[-2], rgb_t.shape[-1]
                scale_x = float(orig_w) / float(inf_W)
                scale_y = float(orig_h) / float(inf_H)
                x = max(0.0, min(orig_w - 1.0, x * scale_x))
                y = max(0.0, min(orig_h - 1.0, y * scale_y))
                points.append((x, y))
                labels.append(lab)
                rgb = _load_image_rgb(image_path)
                pred_mask = env.predict_mask(rgb, points, labels)
                if pred_mask.shape[:2] != gt_bool.shape[:2]:
                    pred_bool = _resize_bool(pred_mask > 0, gt_bool.shape[1], gt_bool.shape[0])
                else:
                    pred_bool = pred_mask > 0
                m = _compute_metrics(gt_bool, pred_bool)
                cur = _metric_value(m, args.metric)
                best_val = max(best_val, cur)
                prev_mask = pred_mask
            metrics_list.append(best_val)
    model.train()
    if not metrics_list:
        return {"VAL_METRIC": 0.0}
    return {"VAL_METRIC": float(np.mean(metrics_list))}


def train() -> None:
    p = argparse.ArgumentParser(description="GRPO fine-tuning for point policy (SAM2 loop)")
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--sam_checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--max_points", type=int, default=16)
    p.add_argument("--metric", type=str, default="DICE")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--beta_kl", type=float, default=0.01)
    p.add_argument("--beta_kl_label", type=float, default=0.2)
    p.add_argument("--beta_kl_pixel", type=float, default=0.8)
    p.add_argument("--pixel_temp_start", type=float, default=1.5)
    p.add_argument("--pixel_temp_end", type=float, default=0.7)
    p.add_argument("--label_temp_start", type=float, default=1.0)
    p.add_argument("--label_temp_end", type=float, default=0.7)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--tb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--init_policy", type=str, default=None, help="Supervised checkpoint (.pt) for initialization and KL reference")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--max_val_items", type=int, default=64)
    args_ns = p.parse_args()

    args = GRPOArgs(
        train_json=args_ns.train_json,
        sam_checkpoint=args_ns.sam_checkpoint,
        out_dir=args_ns.out_dir,
        height=args_ns.height,
        width=args_ns.width,
        stride=args_ns.stride,
        max_points=args_ns.max_points,
        metric=args_ns.metric,
        epochs=args_ns.epochs,
        batch_size=args_ns.batch_size,
        group_size=args_ns.group_size,
        lr=args_ns.lr,
        weight_decay=args_ns.weight_decay,
        clip_eps=args_ns.clip_eps,
        beta_kl=args_ns.beta_kl,
        beta_kl_label=args_ns.beta_kl_label,
        beta_kl_pixel=args_ns.beta_kl_pixel,
        pixel_temp_start=args_ns.pixel_temp_start,
        pixel_temp_end=args_ns.pixel_temp_end,
        label_temp_start=args_ns.label_temp_start,
        label_temp_end=args_ns.label_temp_end,
        save_every=args_ns.save_every,
        val_every=args_ns.val_every,
        resume=args_ns.resume,
        tb=args_ns.tb,
        seed=args_ns.seed,
        device=args_ns.device,
        init_policy=args_ns.init_policy,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else _device()

    # data
    train_ds = JsonArrayDataset(args.train_json, split="train", val_ratio=args_ns.val_ratio, seed=args.seed)
    val_ds = JsonArrayDataset(args.train_json, split="val", val_ratio=args_ns.val_ratio, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # env
    env = SAMEnv(args.sam_checkpoint, device=args.device)

    # policy
    cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
    policy = PointHeatmapModel(cfg).to(device)
    if args.init_policy and os.path.isfile(args.init_policy):
        sd = torch.load(args.init_policy, map_location="cpu")
        sd = sd.get("model", sd)
        policy.load_state_dict(sd, strict=False)
        print(f"[init] loaded policy from {args.init_policy}")

    # reference policy for KL (fixed)
    ref_policy = PointHeatmapModel(cfg).to(device)
    ref_policy.load_state_dict(policy.state_dict(), strict=False)
    ref_policy.eval()
    for p_ref in ref_policy.parameters():
        p_ref.requires_grad_(False)

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    global_step = 0
    ref_sd = ref_policy.state_dict()

    if args.resume and os.path.isfile(args.resume):
        try:
            start_epoch, global_step, ref_sd = _load_checkpoint(args.resume, policy, optim)
            ref_policy.load_state_dict(ref_sd, strict=False)
            print(f"[resume] loaded checkpoint {args.resume} @ epoch={start_epoch} step={global_step}")
        except Exception as e:
            print(f"[resume] failed to load: {e}")

    writer = None
    if args.tb:
        try:
            # Try torch TensorBoard first; fall back to tensorboardX if unavailable
            try:
                from torch.utils.tensorboard import SummaryWriter as _TBWriter
            except Exception:
                from tensorboardX import SummaryWriter as _TBWriter  # type: ignore
            log_dir = os.path.join(args.out_dir, "tb")
            os.makedirs(log_dir, exist_ok=True)
            print(f"[tb] logging to {log_dir}")
            writer = _TBWriter(log_dir=log_dir)
        except Exception as e:
            print(f"[tb] disabled: {e}. Install 'tensorboard' or 'tensorboardX' to enable logs.")
            writer = None

    policy.train()

    def log_scalar(name: str, val: float, step: int) -> None:
        print(f"{name}={val:.6f} @ {step}")
        if writer is not None:
            writer.add_scalar(name, val, step)

    for epoch in range(start_epoch, args.epochs):
        for batch in train_loader:
            # snapshot old policy for this update
            old_policy = PointHeatmapModel(cfg).to(device)
            old_policy.load_state_dict(policy.state_dict(), strict=False)
            old_policy.eval()
            for p_old in old_policy.parameters():
                p_old.requires_grad_(False)

            # temperature for this step (anneal over global steps)
            prog = 0.0 if args.epochs <= 1 else float(epoch) / float(max(1, args.epochs - 1))
            temp_pix = _interp(args.pixel_temp_start, args.pixel_temp_end, prog)
            temp_lbl = _interp(args.label_temp_start, args.label_temp_end, prog)

            # accumulate losses across group rollouts and steps
            total_loss = 0.0
            total_kl = 0.0
            total_adv = 0.0
            num_terms = 0

            # For each time step, we will collect group rewards and logprobs per sample
            B = args.batch_size
            G = args.group_size
            # Per-sample trajectories: list of length G, each contains per-step data
            batch_recs: List[Dict[str, Any]] = [{"image": batch["image"][i], "gt_mask": batch["gt_mask"][i]} for i in range(B)]

            # Rollout per group
            # We store for each t: rewards[B,G], logp[B,G], old_logp[B,G], kl[B]
            max_T = args.max_points
            rewards_t = [torch.zeros(B, G, device=device) for _ in range(max_T)]
            logp_t = [torch.zeros(B, G, device=device) for _ in range(max_T)]
            old_logp_t = [torch.zeros(B, G, device=device) for _ in range(max_T)]
            kl_t = [torch.zeros(B, device=device) for _ in range(max_T)]  # KL computed per state (not grouped)

            # Initialize per-(b,g) point histories and prev masks
            pts = [[[] for _ in range(G)] for _ in range(B)]  # type: ignore[var-annotated]
            lbs = [[[] for _ in range(G)] for _ in range(B)]  # type: ignore[var-annotated]
            prev_masks: List[List[Optional[np.ndarray]]] = [[None for _ in range(G)] for _ in range(B)]

            gt_cache: List[np.ndarray] = []
            rgb_cache: List[np.ndarray] = []
            for b in range(B):
                gt_bool = _load_mask_bool(batch_recs[b]["gt_mask"])
                rgb = _load_image_rgb(batch_recs[b]["image"])
                gt_cache.append(gt_bool)
                rgb_cache.append(rgb)

            for t in range(max_T):
                # collect metrics at t-1 to compute reward increment
                prev_score = torch.zeros(B, G, device=device)
                for b in range(B):
                    gt_bool = gt_cache[b]
                    for g in range(G):
                        if t == 0:
                            prev = np.zeros_like(gt_bool, dtype=np.uint8)
                        else:
                            prev = prev_masks[b][g]
                            if prev is None:
                                prev = np.zeros_like(gt_bool, dtype=np.uint8)
                        prev_bool = prev > 0
                        m_prev = _compute_metrics(gt_bool, prev_bool)
                        prev_score[b, g] = torch.tensor(_metric_value(m_prev, args.metric), device=device)

                # compute policy outputs and sample actions for each (b,g)
                for b in range(B):
                    image_path = batch_recs[b]["image"]
                    for g in range(G):
                        rgb_t, g_t, (orig_h, orig_w) = _prepare_model_inputs(image_path, prev_masks[b][g], (args.height, args.width))
                        logits, label_logits = policy(rgb_t.to(device), g_t.to(device)) ###!!!
                        with torch.no_grad():
                            logits_old, label_logits_old = old_policy(rgb_t.to(device), g_t.to(device))
                            logits_ref, label_logits_ref = ref_policy(rgb_t.to(device), g_t.to(device))

                        # sample action
                        act = sample_joint_label_cell_offset(
                            logits=logits,
                            label_logits=label_logits,
                            stride=args.stride,
                            temperature_pixel=temp_pix,
                            temperature_label=temp_lbl,
                        )
                        # recompute old log_prob for importance ratio
                        lp_old = log_prob_of_joint_action(
                            logits=logits_old,
                            label_logits=label_logits_old,
                            stride=args.stride,
                            label_idx=act["label_idx"],
                            cell_idx=act["cell_idx"],
                            sub_idx=act["sub_idx"],
                            temperature_pixel=temp_pix,
                            temperature_label=temp_lbl,
                        )
                        logp_t[t][b, g] = act["log_prob"]
                        old_logp_t[t][b, g] = lp_old

                        # KL at state (not grouped)
                        kl_label = _calc_kl_categorical(label_logits, label_logits_ref)  # [1]
                        kl_pixel = _calc_kl_pixel_full(logits, logits_ref)  # [1]
                        kl_val = args.beta_kl * (args.beta_kl_label * kl_label + args.beta_kl_pixel * kl_pixel)
                        kl_t[t][b] = kl_val.squeeze(0)

                        # map action to original image size
                        xy = act["pixel_xy"][0].to(torch.float32)
                        x = float(xy[0].item()) + 0.5
                        y = float(xy[1].item()) + 0.5
                        inf_H, inf_W = rgb_t.shape[-2], rgb_t.shape[-1]
                        scale_x = float(orig_w) / float(inf_W)
                        scale_y = float(orig_h) / float(inf_H)
                        x = max(0.0, min(orig_w - 1.0, x * scale_x))
                        y = max(0.0, min(orig_h - 1.0, y * scale_y))
                        pts[b][g].append((x, y))
                        lbs[b][g].append(int(act["label_idx"].item()))

                # step env for all (b,g)
                for b in range(B):
                    gt_bool = gt_cache[b]
                    rgb = rgb_cache[b]
                    for g in range(G):
                        pred_mask = env.predict_mask(rgb, pts[b][g], lbs[b][g])
                        prev_masks[b][g] = pred_mask
                        # compute reward increment
                        if pred_mask.shape[:2] != gt_bool.shape[:2]:
                            pred_bool = _resize_bool(pred_mask > 0, gt_bool.shape[1], gt_bool.shape[0])
                        else:
                            pred_bool = pred_mask > 0
                        m_now = _compute_metrics(gt_bool, pred_bool)
                        cur = _metric_value(m_now, args.metric)
                        rewards_t[t][b, g] = torch.tensor(cur, device=device) - prev_score[b, g]

            # Compute group-relative advantages per time-step
            losses = []
            for t in range(max_T):
                r = rewards_t[t]  # [B,G]
                mean = r.mean(dim=1, keepdim=True)
                std = r.std(dim=1, keepdim=True)
                adv = (r - mean) / (std + 1e-4)  # [B,G]
                # PPO ratio with clipping
                ratio = torch.exp(logp_t[t] - old_logp_t[t])  # [B,G]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                obj = torch.minimum(unclipped, clipped)  # [B,G]
                # subtract KL (broadcast B to [B,G])
                obj = obj - kl_t[t].unsqueeze(1)
                # negative for gradient descent
                loss_t = -obj.mean()
                losses.append(loss_t)

            loss = torch.stack(losses).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optim.step()

            global_step += 1
            log_scalar("train/loss", float(loss.item()), global_step)
            if global_step % 10 == 0:
                # log average reward and advantage stats
                avg_r = torch.stack([r.mean() for r in rewards_t]).mean().item()
                log_scalar("train/avg_reward", float(avg_r), global_step)

            if args.save_every > 0 and (global_step % args.save_every == 0):
                ckpt_path = os.path.join(args.out_dir, f"ckpt_step{global_step}.pt")
                _save_checkpoint(ckpt_path, policy, optim, epoch, global_step, args, ref_policy.state_dict())
                print(f"[ckpt] saved to {ckpt_path}")

            if args.val_every > 0 and (global_step % args.val_every == 0):
                val = validate(policy, val_ds, env, device, args, max_items=args_ns.max_val_items)
                log_scalar("val/metric", val.get("VAL_METRIC", 0.0), global_step)

        # end epoch
        ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch{epoch+1}.pt")
        _save_checkpoint(ckpt_path, policy, optim, epoch + 1, global_step, args, ref_policy.state_dict())
        print(f"[ckpt] saved epoch {epoch+1} to {ckpt_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train()



```

I'll update the training script to (1) ensure BN layers inside the label head stay in eval mode during training to avoid the batch=1 BN error, and (2) re-apply this setting after any call that flips the model back to train mode (e.g., after validation). Then I'll run a linter check on the file.

```python
#!/usr/bin/env python3

"""
GRPO fine-tuning for the point-policy model used to propose next SAM2 prompt points.

Overview
- Policy: seg-rl.heatmap.model.PointHeatmapModel that outputs pixel logits [B,1,H,W]
          and label logits [B,2]. We use its joint hierarchical policy over
          (label, cell, subpixel) with helpers already implemented in model.py.
- Environment: SAM2 predictor turns the accumulated point sequence into a mask.
- Observation: (image RGB, previous predicted mask) → model → logits, label_logits.
- Action: sample joint (label, cell, subpixel) → continuous xy in input grid.
- Reward: per-step improvement in a segmentation metric (e.g., DICE) vs previous step.
- GRPO: group-relative PPO-style objective with ratio clipping and KL to a fixed ref.

Key Features
- CLI with defaults; checkpoint save/resume; TensorBoard logging; optional step viz.
- Temperature schedule for exploration (pixel/label) that anneals across training.
- Validation: greedy rollout to report mean metrics.

Notes
- Input resolution H×W is configurable; actions are mapped back to original image size
  before calling SAM2. Masks from SAM2 are resized to original size for metric eval.
- For efficiency, we keep a single SAM2 predictor per process and reuse it across steps.

Examples (调用示例)
  1) 基本训练（使用已监督预训练权重作为初始化与参考网络）：
     /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.train_grpo_points \
       --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl \
       --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
       --out_dir outputs/braintumour/grpo-251020 \
       --init_policy pretrained/points_predictor-251001-160epochs.pt \
       --device cuda \
       --height 512 --width 512 --stride 8 --max_points 16 \
       --epochs 5 --batch_size 1 --group_size 4 --tb

  2) 在Apple Silicon上使用MPS并调整探索温度与KL权重：
     python -m seg-rl.heatmap.train_grpo_points \
       --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl \
       --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
       --out_dir outputs/braintumour/grpo-mps \
       --device mps --height 512 --width 512 --stride 8 \
       --pixel_temp_start 1.8 --pixel_temp_end 0.8 \
       --label_temp_start 1.2 --label_temp_end 0.8 \
       --beta_kl 0.02 --beta_kl_label 0.3 --beta_kl_pixel 0.7 \
       --epochs 3 --batch_size 2 --group_size 4 --tb

  3) 从断点恢复训练：
     python -m seg-rl.heatmap.train_grpo_points \
       --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl \
       --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
       --out_dir outputs/braintumour/grpo-251015 \
       --resume outputs/braintumour/grpo-251015/ckpt_step1000.pt --tb
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Local imports (robust to module/script execution)
try:
    from .model import (
        ModelConfig,
        PointHeatmapModel,
        sample_joint_label_cell_offset,
        log_prob_of_joint_action,
        soft_argmax_from_logits,
    )
except Exception:
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.abspath(__file__))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from model import (  # type: ignore
        ModelConfig,
        PointHeatmapModel,
        sample_joint_label_cell_offset,
        log_prob_of_joint_action,
        soft_argmax_from_logits,
    )


def _try_import_render_helper() -> Optional[Any]:
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # seg-rl
        target = os.path.join(base_dir, "annotator", "gen_point_jsonl_from_masks.py")
        import importlib.util as _util
        spec = _util.spec_from_file_location("gen_point_jsonl_from_masks", target)
        if not spec or not spec.loader:
            return None
        module = _util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        func = getattr(module, "_render_diff_panels_with_point", None)
        return func if callable(func) else None
    except Exception:
        return None


_render_diff_panels_with_point = _try_import_render_helper()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def _load_mask_bool(path: str) -> np.ndarray:
    arr = np.array(Image.open(path), dtype=np.uint8)
    if arr.ndim == 3:
        arr = (arr.any(axis=2)).astype(np.uint8)
    return arr > 0


def _compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
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

    # lightweight S-measure surrogate (optional)
    s_meas = float(0.0) ## FIXME: Do actual computing of S-measure
    return {
        "DICE": float(dice),
        "IOU": float(iou),
        "PRECISION": float(precision),
        "RECALL": float(recall),
        "F1": float(f1),
        "S_MEASURE": float(s_meas),
    }


def _metric_value(metrics: Dict[str, float], key: str) -> float:
    return float(metrics.get(key.upper(), 0.0))


def _resize_bool(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.array(Image.fromarray((arr.astype(np.uint8) * 255)).resize((w, h), resample=Image.NEAREST)) > 0


class JsonArrayDataset(Dataset):
    def __init__(self, json_path: str, split: str, val_ratio: float = 0.1, seed: int = 42) -> None:
        body = json.loads(open(json_path, "r", encoding="utf-8").read())
        if not isinstance(body, list):
            raise RuntimeError("input JSON must be an array of records")
        items: List[Dict[str, Any]] = []
        for rec in body:
            if not isinstance(rec, dict):
                continue
            img = rec.get("image")
            gt = rec.get("gt_mask")
            if not (img and gt and os.path.isfile(img) and os.path.isfile(gt)):
                continue
            items.append({"image": os.path.abspath(img), "gt_mask": os.path.abspath(gt)})
        # deterministic split
        rng = random.Random(seed)
        rng.shuffle(items)
        n_val = int(len(items) * val_ratio)
        self.items = items[:n_val] if split == "val" else items[n_val:]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.items[idx]


# --- SAM2 integration -------------------------------------------------------


class SAMEnv:
    def __init__(self, sam_checkpoint: str, device: Optional[str] = None) -> None:
        # defer heavy import to runtime
        import sys as _sys
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        third_party = os.path.join(base_dir, "third_party", "sam2")
        _sys.path.append(third_party)
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        if device is None:
            dev = _device()
        else:
            dev = torch.device(device)

        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model = build_sam2(model_cfg, sam_checkpoint, device=dev)
        self.predictor = SAM2ImagePredictor(sam_model)

    def predict_mask(self, image_rgb: np.ndarray, points: List[Tuple[float, float]], labels: List[int]) -> np.ndarray:
        self.predictor.set_image(image_rgb)
        import numpy as _np
        pts = _np.array(points) if points else None
        lbs = _np.array(labels) if labels else None
        mask, score, logits = self.predictor.predict(
            point_coords=pts,
            point_labels=lbs,
            box=None,
            multimask_output=False,
        )
        return mask[0]


def _load_image_rgb(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im)


def _prepare_model_inputs(image_path: str, prev_mask_u8: Optional[np.ndarray], out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    im0 = Image.open(image_path).convert("RGB")
    orig_w, orig_h = im0.size
    H, W = out_hw
    if H <= 0 or W <= 0:
        H, W = orig_h, orig_w
    im = im0.resize((W, H), resample=Image.BILINEAR) if (W, H) != (orig_w, orig_h) else im0
    if prev_mask_u8 is None:
        gray = Image.new("L", (orig_w, orig_h), 0)
    else:
        gray = Image.fromarray(prev_mask_u8.astype(np.uint8))
    gray = gray.resize((W, H), resample=Image.NEAREST) if (W, H) != (orig_w, orig_h) else gray
    import torchvision.transforms.functional as TF
    rgb_t = TF.to_tensor(im)
    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    g_t = TF.to_tensor(gray)
    g_t = (g_t - 0.5) / 0.5
    return rgb_t.unsqueeze(0), g_t.unsqueeze(0), (orig_h, orig_w)


@dataclass
class GRPOArgs:
    # data/env
    train_json: str
    sam_checkpoint: str
    out_dir: str
    height: int = 512
    width: int = 512
    stride: int = 8
    max_points: int = 16
    metric: str = "DICE"
    # training
    epochs: int = 5
    batch_size: int = 4
    group_size: int = 4  # G
    lr: float = 1e-4
    weight_decay: float = 1e-4
    clip_eps: float = 0.2
    beta_kl: float = 0.01
    beta_kl_label: float = 0.2
    beta_kl_pixel: float = 0.8
    # exploration temperatures
    pixel_temp_start: float = 1.5
    pixel_temp_end: float = 0.7
    label_temp_start: float = 1.0
    label_temp_end: float = 0.7
    # logging/checkpoint
    save_every: int = 200
    val_every: int = 200
    resume: Optional[str] = None
    tb: bool = True
    seed: int = 42
    device: Optional[str] = None
    # pretrained policy
    init_policy: Optional[str] = None  # path to supervised checkpoint (.pt)


def _interp(start: float, end: float, ratio: float) -> float:
    ratio = max(0.0, min(1.0, ratio))
    return start + (end - start) * ratio


def _calc_kl_categorical(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    # KL(P||Q) with logits [B, K]
    logp = torch.log_softmax(logits_p, dim=1)
    logq = torch.log_softmax(logits_q, dim=1)
    p = torch.softmax(logits_p, dim=1)
    return (p * (logp - logq)).sum(dim=1)


def _calc_kl_pixel_full(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    # KL over flattened pixel distribution [B,1,H,W] → [B]
    B = logits_p.size(0)
    logp = torch.log_softmax(logits_p.view(B, -1), dim=1)
    logq = torch.log_softmax(logits_q.view(B, -1), dim=1)
    p = torch.softmax(logits_p.view(B, -1), dim=1)
    return (p * (logp - logq)).sum(dim=1)


def _set_label_bn_eval(model: nn.Module) -> None:
    """Force BatchNorm layers under label head to eval mode to support B=1.

    This avoids BN training with N*H*W==1 (from AdaptiveAvgPool2d(1)) while keeping
    the rest of the network in training mode. It does not change requires_grad.
    Call this after any .train() call that might reset submodules.
    """
    for name, module in model.named_modules():
        if "label_head" in name and isinstance(module, nn.BatchNorm2d):
            module.eval()


def _save_checkpoint(path: str, model: nn.Module, optim: torch.optim.Optimizer, epoch: int, step: int, args: GRPOArgs, ref_sd: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": args.__dict__,
        "ref_model": ref_sd,
    }, path)


def _load_checkpoint(path: str, model: nn.Module, optim: torch.optim.Optimizer) -> Tuple[int, int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    if "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("step", 0)), dict(ckpt.get("ref_model", {}))


def validate(model: PointHeatmapModel, dataset: Dataset, env: SAMEnv, device: torch.device, args: GRPOArgs, max_items: int = 64) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    metrics_list: List[float] = []
    with torch.no_grad():
        for i, rec in enumerate(loader):
            if i >= max_items:
                break
            image_path = rec["image"][0]
            gt_path = rec["gt_mask"][0]
            gt_bool = _load_mask_bool(gt_path)

            points: List[Tuple[float, float]] = []
            labels: List[int] = []
            prev_mask: Optional[np.ndarray] = None
            best_val = 0.0
            for t in range(args.max_points):
                rgb_t, g_t, (orig_h, orig_w) = _prepare_model_inputs(image_path, prev_mask, (args.height, args.width))
                logits, label_logits = model(rgb_t.to(device), g_t.to(device))
                # greedy: combine heatmap soft-argmax and label argmax
                xy = soft_argmax_from_logits(logits, temperature=0.5)[0]
                x = float(xy[0].item())
                y = float(xy[1].item())
                lab = int(label_logits.argmax(dim=1).item())
                # map back to original
                inf_H, inf_W = rgb_t.shape[-2], rgb_t.shape[-1]
                scale_x = float(orig_w) / float(inf_W)
                scale_y = float(orig_h) / float(inf_H)
                x = max(0.0, min(orig_w - 1.0, x * scale_x))
                y = max(0.0, min(orig_h - 1.0, y * scale_y))
                points.append((x, y))
                labels.append(lab)
                rgb = _load_image_rgb(image_path)
                pred_mask = env.predict_mask(rgb, points, labels)
                if pred_mask.shape[:2] != gt_bool.shape[:2]:
                    pred_bool = _resize_bool(pred_mask > 0, gt_bool.shape[1], gt_bool.shape[0])
                else:
                    pred_bool = pred_mask > 0
                m = _compute_metrics(gt_bool, pred_bool)
                cur = _metric_value(m, args.metric)
                best_val = max(best_val, cur)
                prev_mask = pred_mask
            metrics_list.append(best_val)
    model.train()
    if not metrics_list:
        return {"VAL_METRIC": 0.0}
    return {"VAL_METRIC": float(np.mean(metrics_list))}


def train() -> None:
    p = argparse.ArgumentParser(description="GRPO fine-tuning for point policy (SAM2 loop)")
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--sam_checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--max_points", type=int, default=16)
    p.add_argument("--metric", type=str, default="DICE")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--beta_kl", type=float, default=0.01)
    p.add_argument("--beta_kl_label", type=float, default=0.2)
    p.add_argument("--beta_kl_pixel", type=float, default=0.8)
    p.add_argument("--pixel_temp_start", type=float, default=1.5)
    p.add_argument("--pixel_temp_end", type=float, default=0.7)
    p.add_argument("--label_temp_start", type=float, default=1.0)
    p.add_argument("--label_temp_end", type=float, default=0.7)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--tb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--init_policy", type=str, default=None, help="Supervised checkpoint (.pt) for initialization and KL reference")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--max_val_items", type=int, default=64)
    args_ns = p.parse_args()

    args = GRPOArgs(
        train_json=args_ns.train_json,
        sam_checkpoint=args_ns.sam_checkpoint,
        out_dir=args_ns.out_dir,
        height=args_ns.height,
        width=args_ns.width,
        stride=args_ns.stride,
        max_points=args_ns.max_points,
        metric=args_ns.metric,
        epochs=args_ns.epochs,
        batch_size=args_ns.batch_size,
        group_size=args_ns.group_size,
        lr=args_ns.lr,
        weight_decay=args_ns.weight_decay,
        clip_eps=args_ns.clip_eps,
        beta_kl=args_ns.beta_kl,
        beta_kl_label=args_ns.beta_kl_label,
        beta_kl_pixel=args_ns.beta_kl_pixel,
        pixel_temp_start=args_ns.pixel_temp_start,
        pixel_temp_end=args_ns.pixel_temp_end,
        label_temp_start=args_ns.label_temp_start,
        label_temp_end=args_ns.label_temp_end,
        save_every=args_ns.save_every,
        val_every=args_ns.val_every,
        resume=args_ns.resume,
        tb=args_ns.tb,
        seed=args_ns.seed,
        device=args_ns.device,
        init_policy=args_ns.init_policy,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else _device()

    # data
    train_ds = JsonArrayDataset(args.train_json, split="train", val_ratio=args_ns.val_ratio, seed=args.seed)
    val_ds = JsonArrayDataset(args.train_json, split="val", val_ratio=args_ns.val_ratio, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # env
    env = SAMEnv(args.sam_checkpoint, device=args.device)

    # policy
    cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
    policy = PointHeatmapModel(cfg).to(device)
    if args.init_policy and os.path.isfile(args.init_policy):
        sd = torch.load(args.init_policy, map_location="cpu")
        sd = sd.get("model", sd)
        policy.load_state_dict(sd, strict=False)
        print(f"[init] loaded policy from {args.init_policy}")

    # reference policy for KL (fixed)
    ref_policy = PointHeatmapModel(cfg).to(device)
    ref_policy.load_state_dict(policy.state_dict(), strict=False)
    ref_policy.eval()
    for p_ref in ref_policy.parameters():
        p_ref.requires_grad_(False)

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    global_step = 0
    ref_sd = ref_policy.state_dict()

    if args.resume and os.path.isfile(args.resume):
        try:
            start_epoch, global_step, ref_sd = _load_checkpoint(args.resume, policy, optim)
            ref_policy.load_state_dict(ref_sd, strict=False)
            print(f"[resume] loaded checkpoint {args.resume} @ epoch={start_epoch} step={global_step}")
        except Exception as e:
            print(f"[resume] failed to load: {e}")

    writer = None
    if args.tb:
        try:
            # Try torch TensorBoard first; fall back to tensorboardX if unavailable
            try:
                from torch.utils.tensorboard import SummaryWriter as _TBWriter
            except Exception:
                from tensorboardX import SummaryWriter as _TBWriter  # type: ignore
            log_dir = os.path.join(args.out_dir, "tb")
            os.makedirs(log_dir, exist_ok=True)
            print(f"[tb] logging to {log_dir}")
            writer = _TBWriter(log_dir=log_dir)
        except Exception as e:
            print(f"[tb] disabled: {e}. Install 'tensorboard' or 'tensorboardX' to enable logs.")
            writer = None

    policy.train()
    _set_label_bn_eval(policy)

    def log_scalar(name: str, val: float, step: int) -> None:
        print(f"{name}={val:.6f} @ {step}")
        if writer is not None:
            writer.add_scalar(name, val, step)

    for epoch in range(start_epoch, args.epochs):
        for batch in train_loader:
            # snapshot old policy for this update
            old_policy = PointHeatmapModel(cfg).to(device)
            old_policy.load_state_dict(policy.state_dict(), strict=False)
            old_policy.eval()
            for p_old in old_policy.parameters():
                p_old.requires_grad_(False)

            # temperature for this step (anneal over global steps)
            prog = 0.0 if args.epochs <= 1 else float(epoch) / float(max(1, args.epochs - 1))
            temp_pix = _interp(args.pixel_temp_start, args.pixel_temp_end, prog)
            temp_lbl = _interp(args.label_temp_start, args.label_temp_end, prog)

            # accumulate losses across group rollouts and steps
            total_loss = 0.0
            total_kl = 0.0
            total_adv = 0.0
            num_terms = 0

            # For each time step, we will collect group rewards and logprobs per sample
            B = args.batch_size
            G = args.group_size
            # Per-sample trajectories: list of length G, each contains per-step data
            batch_recs: List[Dict[str, Any]] = [{"image": batch["image"][i], "gt_mask": batch["gt_mask"][i]} for i in range(B)]

            # Rollout per group
            # We store for each t: rewards[B,G], logp[B,G], old_logp[B,G], kl[B]
            max_T = args.max_points
            rewards_t = [torch.zeros(B, G, device=device) for _ in range(max_T)]
            logp_t = [torch.zeros(B, G, device=device) for _ in range(max_T)]
            old_logp_t = [torch.zeros(B, G, device=device) for _ in range(max_T)]
            kl_t = [torch.zeros(B, device=device) for _ in range(max_T)]  # KL computed per state (not grouped)

            # Initialize per-(b,g) point histories and prev masks
            pts = [[[] for _ in range(G)] for _ in range(B)]  # type: ignore[var-annotated]
            lbs = [[[] for _ in range(G)] for _ in range(B)]  # type: ignore[var-annotated]
            prev_masks: List[List[Optional[np.ndarray]]] = [[None for _ in range(G)] for _ in range(B)]

            gt_cache: List[np.ndarray] = []
            rgb_cache: List[np.ndarray] = []
            for b in range(B):
                gt_bool = _load_mask_bool(batch_recs[b]["gt_mask"])
                rgb = _load_image_rgb(batch_recs[b]["image"])
                gt_cache.append(gt_bool)
                rgb_cache.append(rgb)

            for t in range(max_T):
                # collect metrics at t-1 to compute reward increment
                prev_score = torch.zeros(B, G, device=device)
                for b in range(B):
                    gt_bool = gt_cache[b]
                    for g in range(G):
                        if t == 0:
                            prev = np.zeros_like(gt_bool, dtype=np.uint8)
                        else:
                            prev = prev_masks[b][g]
                            if prev is None:
                                prev = np.zeros_like(gt_bool, dtype=np.uint8)
                        prev_bool = prev > 0
                        m_prev = _compute_metrics(gt_bool, prev_bool)
                        prev_score[b, g] = torch.tensor(_metric_value(m_prev, args.metric), device=device)

                # compute policy outputs and sample actions for each (b,g)
                for b in range(B):
                    image_path = batch_recs[b]["image"]
                    for g in range(G):
                        rgb_t, g_t, (orig_h, orig_w) = _prepare_model_inputs(image_path, prev_masks[b][g], (args.height, args.width))
                        logits, label_logits = policy(rgb_t.to(device), g_t.to(device)) ###!!!
                        with torch.no_grad():
                            logits_old, label_logits_old = old_policy(rgb_t.to(device), g_t.to(device))
                            logits_ref, label_logits_ref = ref_policy(rgb_t.to(device), g_t.to(device))

                        # sample action
                        act = sample_joint_label_cell_offset(
                            logits=logits,
                            label_logits=label_logits,
                            stride=args.stride,
                            temperature_pixel=temp_pix,
                            temperature_label=temp_lbl,
                        )
                        # recompute old log_prob for importance ratio
                        lp_old = log_prob_of_joint_action(
                            logits=logits_old,
                            label_logits=label_logits_old,
                            stride=args.stride,
                            label_idx=act["label_idx"],
                            cell_idx=act["cell_idx"],
                            sub_idx=act["sub_idx"],
                            temperature_pixel=temp_pix,
                            temperature_label=temp_lbl,
                        )
                        logp_t[t][b, g] = act["log_prob"]
                        old_logp_t[t][b, g] = lp_old

                        # KL at state (not grouped)
                        kl_label = _calc_kl_categorical(label_logits, label_logits_ref)  # [1]
                        kl_pixel = _calc_kl_pixel_full(logits, logits_ref)  # [1]
                        kl_val = args.beta_kl * (args.beta_kl_label * kl_label + args.beta_kl_pixel * kl_pixel)
                        kl_t[t][b] = kl_val.squeeze(0)

                        # map action to original image size
                        xy = act["pixel_xy"][0].to(torch.float32)
                        x = float(xy[0].item()) + 0.5
                        y = float(xy[1].item()) + 0.5
                        inf_H, inf_W = rgb_t.shape[-2], rgb_t.shape[-1]
                        scale_x = float(orig_w) / float(inf_W)
                        scale_y = float(orig_h) / float(inf_H)
                        x = max(0.0, min(orig_w - 1.0, x * scale_x))
                        y = max(0.0, min(orig_h - 1.0, y * scale_y))
                        pts[b][g].append((x, y))
                        lbs[b][g].append(int(act["label_idx"].item()))

                # step env for all (b,g)
                for b in range(B):
                    gt_bool = gt_cache[b]
                    rgb = rgb_cache[b]
                    for g in range(G):
                        pred_mask = env.predict_mask(rgb, pts[b][g], lbs[b][g])
                        prev_masks[b][g] = pred_mask
                        # compute reward increment
                        if pred_mask.shape[:2] != gt_bool.shape[:2]:
                            pred_bool = _resize_bool(pred_mask > 0, gt_bool.shape[1], gt_bool.shape[0])
                        else:
                            pred_bool = pred_mask > 0
                        m_now = _compute_metrics(gt_bool, pred_bool)
                        cur = _metric_value(m_now, args.metric)
                        rewards_t[t][b, g] = torch.tensor(cur, device=device) - prev_score[b, g]

            # Compute group-relative advantages per time-step
            losses = []
            for t in range(max_T):
                r = rewards_t[t]  # [B,G]
                mean = r.mean(dim=1, keepdim=True)
                std = r.std(dim=1, keepdim=True)
                adv = (r - mean) / (std + 1e-4)  # [B,G]
                # PPO ratio with clipping
                ratio = torch.exp(logp_t[t] - old_logp_t[t])  # [B,G]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                obj = torch.minimum(unclipped, clipped)  # [B,G]
                # subtract KL (broadcast B to [B,G])
                obj = obj - kl_t[t].unsqueeze(1)
                # negative for gradient descent
                loss_t = -obj.mean()
                losses.append(loss_t)

            loss = torch.stack(losses).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optim.step()

            global_step += 1
            log_scalar("train/loss", float(loss.item()), global_step)
            if global_step % 10 == 0:
                # log average reward and advantage stats
                avg_r = torch.stack([r.mean() for r in rewards_t]).mean().item()
                log_scalar("train/avg_reward", float(avg_r), global_step)

            if args.save_every > 0 and (global_step % args.save_every == 0):
                ckpt_path = os.path.join(args.out_dir, f"ckpt_step{global_step}.pt")
                _save_checkpoint(ckpt_path, policy, optim, epoch, global_step, args, ref_policy.state_dict())
                print(f"[ckpt] saved to {ckpt_path}")

            if args.val_every > 0 and (global_step % args.val_every == 0):
                val = validate(policy, val_ds, env, device, args, max_items=args_ns.max_val_items)
                # validate() ends with model.train(); re-freeze BN in label head to avoid B=1 BN error
                _set_label_bn_eval(policy)
                log_scalar("val/metric", val.get("VAL_METRIC", 0.0), global_step)

        # end epoch
        ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch{epoch+1}.pt")
        _save_checkpoint(ckpt_path, policy, optim, epoch + 1, global_step, args, ref_policy.state_dict())
        print(f"[ckpt] saved epoch {epoch+1} to {ckpt_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train()



```

I set BatchNorm layers under the label head to eval mode during RL training to avoid the batch=1 BN error without changing the model code, and I re-apply this setting after validation resets the mode. Now I'll run a linter check on the updated file.

- I added a helper that forces only the label-head BatchNorm layers to eval mode during RL training, avoiding the batch=1 BN failure without modifying the model. This is reapplied after validation switches the model back to train mode.

You can rerun your command; the BN error should be gone while preserving model behavior elsewhere.

---

**User**

请用中文回答上面的修改里你都做了什么？为什么这么做是有效的，并且不会影响原本预期的训练效果？

---

**Cursor**

- 我做了什么
  - 新增辅助函数 `_set_label_bn_eval(model)`：遍历 `named_modules()`，把名称包含 `label_head` 且类型为 `nn.BatchNorm2d` 的层切到 eval 模式。不会改动权重、不会改 `requires_grad`。
  - 在训练开始处 `policy.train()` 之后调用一次 `_set_label_bn_eval(policy)`。
  - 在每次调用 `validate(...)` 结束后（该函数末尾会执行 `model.train()`），再次调用 `_set_label_bn_eval(policy)`，确保 label head 的 BN 继续保持 eval。

- 为什么这样有效
  - 你的 label head 在前向里有 `AdaptiveAvgPool2d(1)`，导致进入 BN 时张量形状为 `[B, C, 1, 1]`。当 RL rollout 单样本前向（实际就是 `B=1`）时，BN 处于训练态需要每通道至少两个样本（统计量来自 `N*H*W`），于是触发报错。
  - 仅将 label head 内的 BN 切到 eval，BN 会使用已有的运行统计量而非当前 batch 的统计量，从而在 `B=1`、`1×1` 空间尺寸下也能稳定前向，错误自然消除。

- 为什么不影响原本预期训练效果
  - 仅影响 label head 的 BN“模式”，其仿射参数（gamma/beta）仍参与训练，梯度可正常回传；只是停止用当前 batch 更新其运行统计量，这在 `B=1` 的 RL 设置下本来就不可靠。
  - 其它网络部分仍保持 `train()`，包括 UNet/主干中的 BN 依旧训练并更新统计量；这些层的空间维度大（`H×W` 大），即便单样本也有足够统计量，训练行为不变。
  - 参考策略与旧策略本就处于 `eval()`，与我们对 label head BN 的处理一致，KL 计算逻辑不受影响。
  - 不改模型结构、不改权重、不换归一化类型，推理行为与预训练保持一致，避免了把 BN 换成 GN 带来的性能退化风险。

---

**User**

那这种修改是否也兼容batch_size > 1?

---

**Cursor**

是的，兼容。

- 为什么兼容
  - 我只把“label head”里的 BatchNorm 切到 eval，不改动其它层。无论 `batch_size=1` 还是 `>1`，这一处都只会使用预训练阶段累积的运行均值/方差，避免了在 `N*H*W=1`（`AdaptiveAvgPool2d(1)` 后）时训练态 BN 的数值问题。
  - 其它网络部分仍保持 `train()`（包括 UNet/主干里的 BN），当 `batch_size>1` 时它们会正常用当前 batch 的统计量更新，训练行为不变。

- 对训练效果的影响
  - label head 的 BN 只是不更新运行统计量，但其仿射参数（gamma/beta）仍参与训练、可反向传播；因此在 `batch_size>1` 时也不会阻碍优化。
  - 由于 RL rollout 的状态分布与监督训练不同，固定 label head 的 BN 统计量反而能减少不稳定性；而主干/热力图头仍可充分学习，整体期望训练效果不受损。

---

**User**

运行结果发现，即使batchSize=1, 也会报OOM:  . 有什么办法能节省显存占用，同时最大限度保持原模型的性能？

---

**Cursor**

I'll add mixed precision (AMP) and per-time-step backward to reduce peak memory, use inference mode for old/ref policies and SAM2 predictor, and adjust the optimizer step with GradScaler. Then I'll run a linter check.

I'll refactor the training loop to use mixed precision (autocast + GradScaler), switch old/ref policies to inference_mode + autocast, and perform per-time-step backward to reduce peak memory while preserving training signals. Then I'll check for linter issues.

```python
#!/usr/bin/env python3

"""
GRPO fine-tuning for the point-policy model used to propose next SAM2 prompt points.

Overview
- Policy: seg-rl.heatmap.model.PointHeatmapModel that outputs pixel logits [B,1,H,W]
          and label logits [B,2]. We use its joint hierarchical policy over
          (label, cell, subpixel) with helpers already implemented in model.py.
- Environment: SAM2 predictor turns the accumulated point sequence into a mask.
- Observation: (image RGB, previous predicted mask) → model → logits, label_logits.
- Action: sample joint (label, cell, subpixel) → continuous xy in input grid.
- Reward: per-step improvement in a segmentation metric (e.g., DICE) vs previous step.
- GRPO: group-relative PPO-style objective with ratio clipping and KL to a fixed ref.

Key Features
- CLI with defaults; checkpoint save/resume; TensorBoard logging; optional step viz.
- Temperature schedule for exploration (pixel/label) that anneals across training.
- Validation: greedy rollout to report mean metrics.

Notes
- Input resolution H×W is configurable; actions are mapped back to original image size
  before calling SAM2. Masks from SAM2 are resized to original size for metric eval.
- For efficiency, we keep a single SAM2 predictor per process and reuse it across steps.

Examples (调用示例)
  1) 基本训练（使用已监督预训练权重作为初始化与参考网络）：
     /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.train_grpo_points \
       --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl \
       --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
       --out_dir outputs/braintumour/grpo-251020 \
       --init_policy pretrained/points_predictor-251001-160epochs.pt \
       --device cuda \
       --height 512 --width 512 --stride 8 --max_points 16 \
       --epochs 5 --batch_size 1 --group_size 4 --tb

  2) 在Apple Silicon上使用MPS并调整探索温度与KL权重：
     python -m seg-rl.heatmap.train_grpo_points \
       --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl \
       --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
       --out_dir outputs/braintumour/grpo-mps \
       --device mps --height 512 --width 512 --stride 8 \
       --pixel_temp_start 1.8 --pixel_temp_end 0.8 \
       --label_temp_start 1.2 --label_temp_end 0.8 \
       --beta_kl 0.02 --beta_kl_label 0.3 --beta_kl_pixel 0.7 \
       --epochs 3 --batch_size 2 --group_size 4 --tb

  3) 从断点恢复训练：
     python -m seg-rl.heatmap.train_grpo_points \
       --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl \
       --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
       --out_dir outputs/braintumour/grpo-251015 \
       --resume outputs/braintumour/grpo-251015/ckpt_step1000.pt --tb
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Local imports (robust to module/script execution)
try:
    from .model import (
        ModelConfig,
        PointHeatmapModel,
        sample_joint_label_cell_offset,
        log_prob_of_joint_action,
        soft_argmax_from_logits,
    )
except Exception:
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.abspath(__file__))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from model import (  # type: ignore
        ModelConfig,
        PointHeatmapModel,
        sample_joint_label_cell_offset,
        log_prob_of_joint_action,
        soft_argmax_from_logits,
    )


def _try_import_render_helper() -> Optional[Any]:
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # seg-rl
        target = os.path.join(base_dir, "annotator", "gen_point_jsonl_from_masks.py")
        import importlib.util as _util
        spec = _util.spec_from_file_location("gen_point_jsonl_from_masks", target)
        if not spec or not spec.loader:
            return None
        module = _util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        func = getattr(module, "_render_diff_panels_with_point", None)
        return func if callable(func) else None
    except Exception:
        return None


_render_diff_panels_with_point = _try_import_render_helper()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def _load_mask_bool(path: str) -> np.ndarray:
    arr = np.array(Image.open(path), dtype=np.uint8)
    if arr.ndim == 3:
        arr = (arr.any(axis=2)).astype(np.uint8)
    return arr > 0


def _compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
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

    # lightweight S-measure surrogate (optional)
    s_meas = float(0.0) ## FIXME: Do actual computing of S-measure
    return {
        "DICE": float(dice),
        "IOU": float(iou),
        "PRECISION": float(precision),
        "RECALL": float(recall),
        "F1": float(f1),
        "S_MEASURE": float(s_meas),
    }


def _metric_value(metrics: Dict[str, float], key: str) -> float:
    return float(metrics.get(key.upper(), 0.0))


def _resize_bool(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.array(Image.fromarray((arr.astype(np.uint8) * 255)).resize((w, h), resample=Image.NEAREST)) > 0


class JsonArrayDataset(Dataset):
    def __init__(self, json_path: str, split: str, val_ratio: float = 0.1, seed: int = 42) -> None:
        body = json.loads(open(json_path, "r", encoding="utf-8").read())
        if not isinstance(body, list):
            raise RuntimeError("input JSON must be an array of records")
        items: List[Dict[str, Any]] = []
        for rec in body:
            if not isinstance(rec, dict):
                continue
            img = rec.get("image")
            gt = rec.get("gt_mask")
            if not (img and gt and os.path.isfile(img) and os.path.isfile(gt)):
                continue
            items.append({"image": os.path.abspath(img), "gt_mask": os.path.abspath(gt)})
        # deterministic split
        rng = random.Random(seed)
        rng.shuffle(items)
        n_val = int(len(items) * val_ratio)
        self.items = items[:n_val] if split == "val" else items[n_val:]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.items[idx]


# --- SAM2 integration -------------------------------------------------------


class SAMEnv:
    def __init__(self, sam_checkpoint: str, device: Optional[str] = None) -> None:
        # defer heavy import to runtime
        import sys as _sys
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        third_party = os.path.join(base_dir, "third_party", "sam2")
        _sys.path.append(third_party)
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        if device is None:
            dev = _device()
        else:
            dev = torch.device(device)

        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model = build_sam2(model_cfg, sam_checkpoint, device=dev)
        self.predictor = SAM2ImagePredictor(sam_model)

    def predict_mask(self, image_rgb: np.ndarray, points: List[Tuple[float, float]], labels: List[int]) -> np.ndarray:
        self.predictor.set_image(image_rgb)
        import numpy as _np
        pts = _np.array(points) if points else None
        lbs = _np.array(labels) if labels else None
        mask, score, logits = self.predictor.predict(
            point_coords=pts,
            point_labels=lbs,
            box=None,
            multimask_output=False,
        )
        return mask[0]


def _load_image_rgb(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im)


def _prepare_model_inputs(image_path: str, prev_mask_u8: Optional[np.ndarray], out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    im0 = Image.open(image_path).convert("RGB")
    orig_w, orig_h = im0.size
    H, W = out_hw
    if H <= 0 or W <= 0:
        H, W = orig_h, orig_w
    im = im0.resize((W, H), resample=Image.BILINEAR) if (W, H) != (orig_w, orig_h) else im0
    if prev_mask_u8 is None:
        gray = Image.new("L", (orig_w, orig_h), 0)
    else:
        gray = Image.fromarray(prev_mask_u8.astype(np.uint8))
    gray = gray.resize((W, H), resample=Image.NEAREST) if (W, H) != (orig_w, orig_h) else gray
    import torchvision.transforms.functional as TF
    rgb_t = TF.to_tensor(im)
    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    g_t = TF.to_tensor(gray)
    g_t = (g_t - 0.5) / 0.5
    return rgb_t.unsqueeze(0), g_t.unsqueeze(0), (orig_h, orig_w)


@dataclass
class GRPOArgs:
    # data/env
    train_json: str
    sam_checkpoint: str
    out_dir: str
    height: int = 512
    width: int = 512
    stride: int = 8
    max_points: int = 16
    metric: str = "DICE"
    # training
    epochs: int = 5
    batch_size: int = 4
    group_size: int = 4  # G
    lr: float = 1e-4
    weight_decay: float = 1e-4
    clip_eps: float = 0.2
    beta_kl: float = 0.01
    beta_kl_label: float = 0.2
    beta_kl_pixel: float = 0.8
    # exploration temperatures
    pixel_temp_start: float = 1.5
    pixel_temp_end: float = 0.7
    label_temp_start: float = 1.0
    label_temp_end: float = 0.7
    # logging/checkpoint
    save_every: int = 200
    val_every: int = 200
    resume: Optional[str] = None
    tb: bool = True
    seed: int = 42
    device: Optional[str] = None
    # pretrained policy
    init_policy: Optional[str] = None  # path to supervised checkpoint (.pt)


def _interp(start: float, end: float, ratio: float) -> float:
    ratio = max(0.0, min(1.0, ratio))
    return start + (end - start) * ratio


def _calc_kl_categorical(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    # KL(P||Q) with logits [B, K]
    logp = torch.log_softmax(logits_p, dim=1)
    logq = torch.log_softmax(logits_q, dim=1)
    p = torch.softmax(logits_p, dim=1)
    return (p * (logp - logq)).sum(dim=1)


def _calc_kl_pixel_full(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    # KL over flattened pixel distribution [B,1,H,W] → [B]
    B = logits_p.size(0)
    logp = torch.log_softmax(logits_p.view(B, -1), dim=1)
    logq = torch.log_softmax(logits_q.view(B, -1), dim=1)
    p = torch.softmax(logits_p.view(B, -1), dim=1)
    return (p * (logp - logq)).sum(dim=1)


def _set_label_bn_eval(model: nn.Module) -> None:
    """Force BatchNorm layers under label head to eval mode to support B=1.

    This avoids BN training with N*H*W==1 (from AdaptiveAvgPool2d(1)) while keeping
    the rest of the network in training mode. It does not change requires_grad.
    Call this after any .train() call that might reset submodules.
    """
    for name, module in model.named_modules():
        if "label_head" in name and isinstance(module, nn.BatchNorm2d):
            module.eval()


def _save_checkpoint(path: str, model: nn.Module, optim: torch.optim.Optimizer, epoch: int, step: int, args: GRPOArgs, ref_sd: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": args.__dict__,
        "ref_model": ref_sd,
    }, path)


def _load_checkpoint(path: str, model: nn.Module, optim: torch.optim.Optimizer) -> Tuple[int, int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    if "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("step", 0)), dict(ckpt.get("ref_model", {}))


def validate(model: PointHeatmapModel, dataset: Dataset, env: SAMEnv, device: torch.device, args: GRPOArgs, max_items: int = 64) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    metrics_list: List[float] = []
    with torch.no_grad():
        for i, rec in enumerate(loader):
            if i >= max_items:
                break
            image_path = rec["image"][0]
            gt_path = rec["gt_mask"][0]
            gt_bool = _load_mask_bool(gt_path)

            points: List[Tuple[float, float]] = []
            labels: List[int] = []
            prev_mask: Optional[np.ndarray] = None
            best_val = 0.0
            for t in range(args.max_points):
                rgb_t, g_t, (orig_h, orig_w) = _prepare_model_inputs(image_path, prev_mask, (args.height, args.width))
                logits, label_logits = model(rgb_t.to(device), g_t.to(device))
                # greedy: combine heatmap soft-argmax and label argmax
                xy = soft_argmax_from_logits(logits, temperature=0.5)[0]
                x = float(xy[0].item())
                y = float(xy[1].item())
                lab = int(label_logits.argmax(dim=1).item())
                # map back to original
                inf_H, inf_W = rgb_t.shape[-2], rgb_t.shape[-1]
                scale_x = float(orig_w) / float(inf_W)
                scale_y = float(orig_h) / float(inf_H)
                x = max(0.0, min(orig_w - 1.0, x * scale_x))
                y = max(0.0, min(orig_h - 1.0, y * scale_y))
                points.append((x, y))
                labels.append(lab)
                rgb = _load_image_rgb(image_path)
                pred_mask = env.predict_mask(rgb, points, labels)
                if pred_mask.shape[:2] != gt_bool.shape[:2]:
                    pred_bool = _resize_bool(pred_mask > 0, gt_bool.shape[1], gt_bool.shape[0])
                else:
                    pred_bool = pred_mask > 0
                m = _compute_metrics(gt_bool, pred_bool)
                cur = _metric_value(m, args.metric)
                best_val = max(best_val, cur)
                prev_mask = pred_mask
            metrics_list.append(best_val)
    model.train()
    if not metrics_list:
        return {"VAL_METRIC": 0.0}
    return {"VAL_METRIC": float(np.mean(metrics_list))}


def train() -> None:
    p = argparse.ArgumentParser(description="GRPO fine-tuning for point policy (SAM2 loop)")
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--sam_checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--max_points", type=int, default=16)
    p.add_argument("--metric", type=str, default="DICE")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--beta_kl", type=float, default=0.01)
    p.add_argument("--beta_kl_label", type=float, default=0.2)
    p.add_argument("--beta_kl_pixel", type=float, default=0.8)
    p.add_argument("--pixel_temp_start", type=float, default=1.5)
    p.add_argument("--pixel_temp_end", type=float, default=0.7)
    p.add_argument("--label_temp_start", type=float, default=1.0)
    p.add_argument("--label_temp_end", type=float, default=0.7)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--tb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--init_policy", type=str, default=None, help="Supervised checkpoint (.pt) for initialization and KL reference")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--max_val_items", type=int, default=64)
    args_ns = p.parse_args()

    args = GRPOArgs(
        train_json=args_ns.train_json,
        sam_checkpoint=args_ns.sam_checkpoint,
        out_dir=args_ns.out_dir,
        height=args_ns.height,
        width=args_ns.width,
        stride=args_ns.stride,
        max_points=args_ns.max_points,
        metric=args_ns.metric,
        epochs=args_ns.epochs,
        batch_size=args_ns.batch_size,
        group_size=args_ns.group_size,
        lr=args_ns.lr,
        weight_decay=args_ns.weight_decay,
        clip_eps=args_ns.clip_eps,
        beta_kl=args_ns.beta_kl,
        beta_kl_label=args_ns.beta_kl_label,
        beta_kl_pixel=args_ns.beta_kl_pixel,
        pixel_temp_start=args_ns.pixel_temp_start,
        pixel_temp_end=args_ns.pixel_temp_end,
        label_temp_start=args_ns.label_temp_start,
        label_temp_end=args_ns.label_temp_end,
        save_every=args_ns.save_every,
        val_every=args_ns.val_every,
        resume=args_ns.resume,
        tb=args_ns.tb,
        seed=args_ns.seed,
        device=args_ns.device,
        init_policy=args_ns.init_policy,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else _device()

    # data
    train_ds = JsonArrayDataset(args.train_json, split="train", val_ratio=args_ns.val_ratio, seed=args.seed)
    val_ds = JsonArrayDataset(args.train_json, split="val", val_ratio=args_ns.val_ratio, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # env
    env = SAMEnv(args.sam_checkpoint, device=args.device)

    # policy
    cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
    policy = PointHeatmapModel(cfg).to(device)
    if args.init_policy and os.path.isfile(args.init_policy):
        sd = torch.load(args.init_policy, map_location="cpu")
        sd = sd.get("model", sd)
        policy.load_state_dict(sd, strict=False)
        print(f"[init] loaded policy from {args.init_policy}")

    # reference policy for KL (fixed)
    ref_policy = PointHeatmapModel(cfg).to(device)
    ref_policy.load_state_dict(policy.state_dict(), strict=False)
    ref_policy.eval()
    for p_ref in ref_policy.parameters():
        p_ref.requires_grad_(False)

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    global_step = 0
    ref_sd = ref_policy.state_dict()

    if args.resume and os.path.isfile(args.resume):
        try:
            start_epoch, global_step, ref_sd = _load_checkpoint(args.resume, policy, optim)
            ref_policy.load_state_dict(ref_sd, strict=False)
            print(f"[resume] loaded checkpoint {args.resume} @ epoch={start_epoch} step={global_step}")
        except Exception as e:
            print(f"[resume] failed to load: {e}")

    writer = None
    if args.tb:
        try:
            # Try torch TensorBoard first; fall back to tensorboardX if unavailable
            try:
                from torch.utils.tensorboard import SummaryWriter as _TBWriter
            except Exception:
                from tensorboardX import SummaryWriter as _TBWriter  # type: ignore
            log_dir = os.path.join(args.out_dir, "tb")
            os.makedirs(log_dir, exist_ok=True)
            print(f"[tb] logging to {log_dir}")
            writer = _TBWriter(log_dir=log_dir)
        except Exception as e:
            print(f"[tb] disabled: {e}. Install 'tensorboard' or 'tensorboardX' to enable logs.")
            writer = None

    policy.train()
    _set_label_bn_eval(policy)
    use_cuda_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_cuda_amp)

    def log_scalar(name: str, val: float, step: int) -> None:
        print(f"{name}={val:.6f} @ {step}")
        if writer is not None:
            writer.add_scalar(name, val, step)

    for epoch in range(start_epoch, args.epochs):
        for batch in train_loader:
            # snapshot old policy for this update
            old_policy = PointHeatmapModel(cfg).to(device)
            old_policy.load_state_dict(policy.state_dict(), strict=False)
            old_policy.eval()
            for p_old in old_policy.parameters():
                p_old.requires_grad_(False)

            # temperature for this step (anneal over global steps)
            prog = 0.0 if args.epochs <= 1 else float(epoch) / float(max(1, args.epochs - 1))
            temp_pix = _interp(args.pixel_temp_start, args.pixel_temp_end, prog)
            temp_lbl = _interp(args.label_temp_start, args.label_temp_end, prog)

            # accumulate losses across group rollouts and steps
            total_loss = 0.0
            total_kl = 0.0
            total_adv = 0.0
            num_terms = 0

            # For each time step, we will collect group rewards and logprobs per sample
            B = args.batch_size
            G = args.group_size
            # Per-sample trajectories: list of length G, each contains per-step data
            batch_recs: List[Dict[str, Any]] = [{"image": batch["image"][i], "gt_mask": batch["gt_mask"][i]} for i in range(B)]

            # Rollout per group
            # We store for each t: rewards[B,G], logp[B,G], old_logp[B,G], kl[B]
            max_T = args.max_points
            rewards_t = [torch.zeros(B, G, device=device) for _ in range(max_T)]
            logp_t = [torch.zeros(B, G, device=device) for _ in range(max_T)]
            old_logp_t = [torch.zeros(B, G, device=device) for _ in range(max_T)]
            kl_t = [torch.zeros(B, device=device) for _ in range(max_T)]  # KL computed per state (not grouped)

            # Initialize per-(b,g) point histories and prev masks
            pts = [[[] for _ in range(G)] for _ in range(B)]  # type: ignore[var-annotated]
            lbs = [[[] for _ in range(G)] for _ in range(B)]  # type: ignore[var-annotated]
            prev_masks: List[List[Optional[np.ndarray]]] = [[None for _ in range(G)] for _ in range(B)]

            gt_cache: List[np.ndarray] = []
            rgb_cache: List[np.ndarray] = []
            for b in range(B):
                gt_bool = _load_mask_bool(batch_recs[b]["gt_mask"])
                rgb = _load_image_rgb(batch_recs[b]["image"])
                gt_cache.append(gt_bool)
                rgb_cache.append(rgb)

            for t in range(max_T):
                # collect metrics at t-1 to compute reward increment
                prev_score = torch.zeros(B, G, device=device)
                for b in range(B):
                    gt_bool = gt_cache[b]
                    for g in range(G):
                        if t == 0:
                            prev = np.zeros_like(gt_bool, dtype=np.uint8)
                        else:
                            prev = prev_masks[b][g]
                            if prev is None:
                                prev = np.zeros_like(gt_bool, dtype=np.uint8)
                        prev_bool = prev > 0
                        m_prev = _compute_metrics(gt_bool, prev_bool)
                        prev_score[b, g] = torch.tensor(_metric_value(m_prev, args.metric), device=device)

                # compute policy outputs and sample actions for each (b,g)
                for b in range(B):
                    image_path = batch_recs[b]["image"]
                    for g in range(G):
                        rgb_t, g_t, (orig_h, orig_w) = _prepare_model_inputs(image_path, prev_masks[b][g], (args.height, args.width))
                        with autocast(enabled=use_cuda_amp):
                            logits, label_logits = policy(rgb_t.to(device), g_t.to(device)) ###!!!
                        # old/ref in inference + autocast to save memory
                        with torch.inference_mode():
                            with autocast(enabled=use_cuda_amp):
                                logits_old, label_logits_old = old_policy(rgb_t.to(device), g_t.to(device))
                                logits_ref, label_logits_ref = ref_policy(rgb_t.to(device), g_t.to(device))

                        # sample action
                        act = sample_joint_label_cell_offset(
                            logits=logits,
                            label_logits=label_logits,
                            stride=args.stride,
                            temperature_pixel=temp_pix,
                            temperature_label=temp_lbl,
                        )
                        # recompute old log_prob for importance ratio
                        lp_old = log_prob_of_joint_action(
                            logits=logits_old,
                            label_logits=label_logits_old,
                            stride=args.stride,
                            label_idx=act["label_idx"],
                            cell_idx=act["cell_idx"],
                            sub_idx=act["sub_idx"],
                            temperature_pixel=temp_pix,
                            temperature_label=temp_lbl,
                        )
                        logp_t[t][b, g] = act["log_prob"]
                        old_logp_t[t][b, g] = lp_old

                        # KL at state (not grouped)
                        kl_label = _calc_kl_categorical(label_logits, label_logits_ref)  # [1]
                        kl_pixel = _calc_kl_pixel_full(logits, logits_ref)  # [1]
                        kl_val = args.beta_kl * (args.beta_kl_label * kl_label + args.beta_kl_pixel * kl_pixel)
                        kl_t[t][b] = kl_val.squeeze(0)

                        # map action to original image size
                        xy = act["pixel_xy"][0].to(torch.float32)
                        x = float(xy[0].item()) + 0.5
                        y = float(xy[1].item()) + 0.5
                        inf_H, inf_W = rgb_t.shape[-2], rgb_t.shape[-1]
                        scale_x = float(orig_w) / float(inf_W)
                        scale_y = float(orig_h) / float(inf_H)
                        x = max(0.0, min(orig_w - 1.0, x * scale_x))
                        y = max(0.0, min(orig_h - 1.0, y * scale_y))
                        pts[b][g].append((x, y))
                        lbs[b][g].append(int(act["label_idx"].item()))

                # step env for all (b,g)
                for b in range(B):
                    gt_bool = gt_cache[b]
                    rgb = rgb_cache[b]
                    for g in range(G):
                        pred_mask = env.predict_mask(rgb, pts[b][g], lbs[b][g])
                        prev_masks[b][g] = pred_mask
                        # compute reward increment
                        if pred_mask.shape[:2] != gt_bool.shape[:2]:
                            pred_bool = _resize_bool(pred_mask > 0, gt_bool.shape[1], gt_bool.shape[0])
                        else:
                            pred_bool = pred_mask > 0
                        m_now = _compute_metrics(gt_bool, pred_bool)
                        cur = _metric_value(m_now, args.metric)
                        rewards_t[t][b, g] = torch.tensor(cur, device=device) - prev_score[b, g]

                # per-time-step PPO objective and backward to avoid holding graphs for all T
                r = rewards_t[t]  # [B,G]
                mean = r.mean(dim=1, keepdim=True)
                std = r.std(dim=1, keepdim=True)
                adv = (r - mean) / (std + 1e-4)  # [B,G]
                ratio = torch.exp(logp_t[t] - old_logp_t[t])  # [B,G]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                obj = torch.minimum(unclipped, clipped)  # [B,G]
                obj = obj - kl_t[t].unsqueeze(1)
                loss_t = -obj.mean()

                scaler.scale(loss_t).backward()

            # Compute group-relative advantages per time-step
            losses = []
            for t in range(max_T):
                r = rewards_t[t]  # [B,G]
                mean = r.mean(dim=1, keepdim=True)
                std = r.std(dim=1, keepdim=True)
                adv = (r - mean) / (std + 1e-4)  # [B,G]
                # PPO ratio with clipping
                ratio = torch.exp(logp_t[t] - old_logp_t[t])  # [B,G]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                obj = torch.minimum(unclipped, clipped)  # [B,G]
                # subtract KL (broadcast B to [B,G])
                obj = obj - kl_t[t].unsqueeze(1)
                # negative for gradient descent
                loss_t = -obj.mean()
                losses.append(loss_t)

            # optimizer step with AMP
            optim.zero_grad(set_to_none=True)
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            global_step += 1
            # Note: per-step loss already backpropagated; for logging compute average loss proxy
            avg_r = torch.stack([r.mean() for r in rewards_t]).mean().item()
            log_scalar("train/avg_reward", float(avg_r), global_step)
            # optional: additional logs every 10 steps can be added here

            if args.save_every > 0 and (global_step % args.save_every == 0):
                ckpt_path = os.path.join(args.out_dir, f"ckpt_step{global_step}.pt")
                _save_checkpoint(ckpt_path, policy, optim, epoch, global_step, args, ref_policy.state_dict())
                print(f"[ckpt] saved to {ckpt_path}")

            if args.val_every > 0 and (global_step % args.val_every == 0):
                val = validate(policy, val_ds, env, device, args, max_items=args_ns.max_val_items)
                # validate() ends with model.train(); re-freeze BN in label head to avoid B=1 BN error
                _set_label_bn_eval(policy)
                log_scalar("val/metric", val.get("VAL_METRIC", 0.0), global_step)

        # end epoch
        ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch{epoch+1}.pt")
        _save_checkpoint(ckpt_path, policy, optim, epoch + 1, global_step, args, ref_policy.state_dict())
        print(f"[ckpt] saved epoch {epoch+1} to {ckpt_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train()



```

- 我做了这些内存优化（不改模型代码、尽量保性能）：
  - 启用混合精度：对策略前向包上 autocast，并使用 GradScaler 执行缩放反传与 step，显著降低显存占用。
  - 旧策略与参考策略推理态推断：对 old/ref policy 使用 inference_mode + autocast 做前向，彻底关闭其梯度和中间图存储，进一步省显存。
  - 按时间步反传：每个时间步单独计算 PPO 目标并立即 backward，避免把所有时间步的计算图同时留在内存里，降低峰值显存。
  - 继续保留 label head 的 BN 仅 eval 处理，避免 batch=1 时 BN 报错，且不改动模型结构与权重。

- 为什么能省显存且尽量不影响性能：
  - autocast 在 CUDA 下将大量张量转为 FP16/TF32，减半或更多中间激活内存；GradScaler 确保数值稳定，收敛不受明显影响。
  - old/ref policy 只用于对比/KL，不需要梯度；inference_mode 让它们不构建计算图，释放大量内存而不影响训练信号。
  - 按时间步反传减少并发保留的计算图段数，峰值内存显著下降；与一次性聚合 loss 的理论梯度等价（线性求和），优化目标不变。
  - 我们没有改模型层/归一化策略，不触碰预训练权重分布；训练主要在主干与热力图头进行，性能基线得以保持。

- 额外建议（可选进一步节省）：
  - 如仍 OOM，可暂时减小 `height/width` 或 `group_size`，或把 `max_points` 降一点点；这些是对峰值显存最敏感的超参。
  - 若使用 CUDA 11.4+，可设置环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片化。

---

**User**

现在报了另一个错：

---

**Cursor**

I'll fix the AMP deprecation by switching to torch.amp.autocast/GradScaler and correct the optimizer/GradScaler order: zero grads before any backward calls, then accumulate per-time-step scaled backward, unscale, clip, step, update.

```python
#!/usr/bin/env python3

"""
GRPO fine-tuning for the point-policy model used to propose next SAM2 prompt points.

Overview
- Policy: seg-rl.heatmap.model.PointHeatmapModel that outputs pixel logits [B,1,H,W]
          and label logits [B,2]. We use its joint hierarchical policy over
          (label, cell, subpixel) with helpers already implemented in model.py.
- Environment: SAM2 predictor turns the accumulated point sequence into a mask.
- Observation: (image RGB, previous predicted mask) → model → logits, label_logits.
- Action: sample joint (label, cell, subpixel) → continuous xy in input grid.
- Reward: per-step improvement in a segmentation metric (e.g., DICE) vs previous step.
- GRPO: group-relative PPO-style objective with ratio clipping and KL to a fixed ref.

Key Features
- CLI with defaults; checkpoint save/resume; TensorBoard logging; optional step viz.
- Temperature schedule for exploration (pixel/label) that anneals across training.
- Validation: greedy rollout to report mean metrics.

Notes
- Input resolution H×W is configurable; actions are mapped back to original image size
  before calling SAM2. Masks from SAM2 are resized to original size for metric eval.
- For efficiency, we keep a single SAM2 predictor per process and reuse it across steps.

Examples (调用示例)
  1) 基本训练（使用已监督预训练权重作为初始化与参考网络）：
     /opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.train_grpo_points \
       --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl \
       --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
       --out_dir outputs/braintumour/grpo-251020 \
       --init_policy pretrained/points_predictor-251001-160epochs.pt \
       --device cuda \
       --height 512 --width 512 --stride 8 --max_points 16 \
       --epochs 5 --batch_size 1 --group_size 4 --tb

  2) 在Apple Silicon上使用MPS并调整探索温度与KL权重：
     python -m seg-rl.heatmap.train_grpo_points \
       --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl \
       --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
       --out_dir outputs/braintumour/grpo-mps \
       --device mps --height 512 --width 512 --stride 8 \
       --pixel_temp_start 1.8 --pixel_temp_end 0.8 \
       --label_temp_start 1.2 --label_temp_end 0.8 \
       --beta_kl 0.02 --beta_kl_label 0.3 --beta_kl_pixel 0.7 \
       --epochs 3 --batch_size 2 --group_size 4 --tb

  3) 从断点恢复训练：
     python -m seg-rl.heatmap.train_grpo_points \
       --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl \
       --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
       --out_dir outputs/braintumour/grpo-251015 \
       --resume outputs/braintumour/grpo-251015/ckpt_step1000.pt --tb
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import amp

# Local imports (robust to module/script execution)
try:
    from .model import (
        ModelConfig,
        PointHeatmapModel,
        sample_joint_label_cell_offset,
        log_prob_of_joint_action,
        soft_argmax_from_logits,
    )
except Exception:
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.abspath(__file__))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from model import (  # type: ignore
        ModelConfig,
        PointHeatmapModel,
        sample_joint_label_cell_offset,
        log_prob_of_joint_action,
        soft_argmax_from_logits,
    )


def _try_import_render_helper() -> Optional[Any]:
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # seg-rl
        target = os.path.join(base_dir, "annotator", "gen_point_jsonl_from_masks.py")
        import importlib.util as _util
        spec = _util.spec_from_file_location("gen_point_jsonl_from_masks", target)
        if not spec or not spec.loader:
            return None
        module = _util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        func = getattr(module, "_render_diff_panels_with_point", None)
        return func if callable(func) else None
    except Exception:
        return None


_render_diff_panels_with_point = _try_import_render_helper()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stem(path: str) -> str:
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    return base


def _load_mask_bool(path: str) -> np.ndarray:
    arr = np.array(Image.open(path), dtype=np.uint8)
    if arr.ndim == 3:
        arr = (arr.any(axis=2)).astype(np.uint8)
    return arr > 0


def _compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
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

    # lightweight S-measure surrogate (optional)
    s_meas = float(0.0) ## FIXME: Do actual computing of S-measure
    return {
        "DICE": float(dice),
        "IOU": float(iou),
        "PRECISION": float(precision),
        "RECALL": float(recall),
        "F1": float(f1),
        "S_MEASURE": float(s_meas),
    }


def _metric_value(metrics: Dict[str, float], key: str) -> float:
    return float(metrics.get(key.upper(), 0.0))


def _resize_bool(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.array(Image.fromarray((arr.astype(np.uint8) * 255)).resize((w, h), resample=Image.NEAREST)) > 0


class JsonArrayDataset(Dataset):
    def __init__(self, json_path: str, split: str, val_ratio: float = 0.1, seed: int = 42) -> None:
        body = json.loads(open(json_path, "r", encoding="utf-8").read())
        if not isinstance(body, list):
            raise RuntimeError("input JSON must be an array of records")
        items: List[Dict[str, Any]] = []
        for rec in body:
            if not isinstance(rec, dict):
                continue
            img = rec.get("image")
            gt = rec.get("gt_mask")
            if not (img and gt and os.path.isfile(img) and os.path.isfile(gt)):
                continue
            items.append({"image": os.path.abspath(img), "gt_mask": os.path.abspath(gt)})
        # deterministic split
        rng = random.Random(seed)
        rng.shuffle(items)
        n_val = int(len(items) * val_ratio)
        self.items = items[:n_val] if split == "val" else items[n_val:]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.items[idx]


# --- SAM2 integration -------------------------------------------------------


class SAMEnv:
    def __init__(self, sam_checkpoint: str, device: Optional[str] = None) -> None:
        # defer heavy import to runtime
        import sys as _sys
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        third_party = os.path.join(base_dir, "third_party", "sam2")
        _sys.path.append(third_party)
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        if device is None:
            dev = _device()
        else:
            dev = torch.device(device)

        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model = build_sam2(model_cfg, sam_checkpoint, device=dev)
        self.predictor = SAM2ImagePredictor(sam_model)

    def predict_mask(self, image_rgb: np.ndarray, points: List[Tuple[float, float]], labels: List[int]) -> np.ndarray:
        self.predictor.set_image(image_rgb)
        import numpy as _np
        pts = _np.array(points) if points else None
        lbs = _np.array(labels) if labels else None
        mask, score, logits = self.predictor.predict(
            point_coords=pts,
            point_labels=lbs,
            box=None,
            multimask_output=False,
        )
        return mask[0]


def _load_image_rgb(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im)


def _prepare_model_inputs(image_path: str, prev_mask_u8: Optional[np.ndarray], out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    im0 = Image.open(image_path).convert("RGB")
    orig_w, orig_h = im0.size
    H, W = out_hw
    if H <= 0 or W <= 0:
        H, W = orig_h, orig_w
    im = im0.resize((W, H), resample=Image.BILINEAR) if (W, H) != (orig_w, orig_h) else im0
    if prev_mask_u8 is None:
        gray = Image.new("L", (orig_w, orig_h), 0)
    else:
        gray = Image.fromarray(prev_mask_u8.astype(np.uint8))
    gray = gray.resize((W, H), resample=Image.NEAREST) if (W, H) != (orig_w, orig_h) else gray
    import torchvision.transforms.functional as TF
    rgb_t = TF.to_tensor(im)
    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    g_t = TF.to_tensor(gray)
    g_t = (g_t - 0.5) / 0.5
    return rgb_t.unsqueeze(0), g_t.unsqueeze(0), (orig_h, orig_w)


@dataclass
class GRPOArgs:
    # data/env
    train_json: str
    sam_checkpoint: str
    out_dir: str
    height: int = 512
    width: int = 512
    stride: int = 8
    max_points: int = 16
    metric: str = "DICE"
    # training
    epochs: int = 5
    batch_size: int = 4
    group_size: int = 4  # G
    lr: float = 1e-4
    weight_decay: float = 1e-4
    clip_eps: float = 0.2
    beta_kl: float = 0.01
    beta_kl_label: float = 0.2
    beta_kl_pixel: float = 0.8
    # exploration temperatures
    pixel_temp_start: float = 1.5
    pixel_temp_end: float = 0.7
    label_temp_start: float = 1.0
    label_temp_end: float = 0.7
    # logging/checkpoint
    save_every: int = 200
    val_every: int = 200
    resume: Optional[str] = None
    tb: bool = True
    seed: int = 42
    device: Optional[str] = None
    # pretrained policy
    init_policy: Optional[str] = None  # path to supervised checkpoint (.pt)


def _interp(start: float, end: float, ratio: float) -> float:
    ratio = max(0.0, min(1.0, ratio))
    return start + (end - start) * ratio


def _calc_kl_categorical(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    # KL(P||Q) with logits [B, K]
    logp = torch.log_softmax(logits_p, dim=1)
    logq = torch.log_softmax(logits_q, dim=1)
    p = torch.softmax(logits_p, dim=1)
    return (p * (logp - logq)).sum(dim=1)


def _calc_kl_pixel_full(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    # KL over flattened pixel distribution [B,1,H,W] → [B]
    B = logits_p.size(0)
    logp = torch.log_softmax(logits_p.view(B, -1), dim=1)
    logq = torch.log_softmax(logits_q.view(B, -1), dim=1)
    p = torch.softmax(logits_p.view(B, -1), dim=1)
    return (p * (logp - logq)).sum(dim=1)


def _set_label_bn_eval(model: nn.Module) -> None:
    """Force BatchNorm layers under label head to eval mode to support B=1.

    This avoids BN training with N*H*W==1 (from AdaptiveAvgPool2d(1)) while keeping
    the rest of the network in training mode. It does not change requires_grad.
    Call this after any .train() call that might reset submodules.
    """
    for name, module in model.named_modules():
        if "label_head" in name and isinstance(module, nn.BatchNorm2d):
            module.eval()


def _save_checkpoint(path: str, model: nn.Module, optim: torch.optim.Optimizer, epoch: int, step: int, args: GRPOArgs, ref_sd: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": args.__dict__,
        "ref_model": ref_sd,
    }, path)


def _load_checkpoint(path: str, model: nn.Module, optim: torch.optim.Optimizer) -> Tuple[int, int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    if "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("step", 0)), dict(ckpt.get("ref_model", {}))


def validate(model: PointHeatmapModel, dataset: Dataset, env: SAMEnv, device: torch.device, args: GRPOArgs, max_items: int = 64) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    metrics_list: List[float] = []
    with torch.no_grad():
        for i, rec in enumerate(loader):
            if i >= max_items:
                break
            image_path = rec["image"][0]
            gt_path = rec["gt_mask"][0]
            gt_bool = _load_mask_bool(gt_path)

            points: List[Tuple[float, float]] = []
            labels: List[int] = []
            prev_mask: Optional[np.ndarray] = None
            best_val = 0.0
            for t in range(args.max_points):
                rgb_t, g_t, (orig_h, orig_w) = _prepare_model_inputs(image_path, prev_mask, (args.height, args.width))
                logits, label_logits = model(rgb_t.to(device), g_t.to(device))
                # greedy: combine heatmap soft-argmax and label argmax
                xy = soft_argmax_from_logits(logits, temperature=0.5)[0]
                x = float(xy[0].item())
                y = float(xy[1].item())
                lab = int(label_logits.argmax(dim=1).item())
                # map back to original
                inf_H, inf_W = rgb_t.shape[-2], rgb_t.shape[-1]
                scale_x = float(orig_w) / float(inf_W)
                scale_y = float(orig_h) / float(inf_H)
                x = max(0.0, min(orig_w - 1.0, x * scale_x))
                y = max(0.0, min(orig_h - 1.0, y * scale_y))
                points.append((x, y))
                labels.append(lab)
                rgb = _load_image_rgb(image_path)
                pred_mask = env.predict_mask(rgb, points, labels)
                if pred_mask.shape[:2] != gt_bool.shape[:2]:
                    pred_bool = _resize_bool(pred_mask > 0, gt_bool.shape[1], gt_bool.shape[0])
                else:
                    pred_bool = pred_mask > 0
                m = _compute_metrics(gt_bool, pred_bool)
                cur = _metric_value(m, args.metric)
                best_val = max(best_val, cur)
                prev_mask = pred_mask
            metrics_list.append(best_val)
    model.train()
    if not metrics_list:
        return {"VAL_METRIC": 0.0}
    return {"VAL_METRIC": float(np.mean(metrics_list))}


def train() -> None:
    p = argparse.ArgumentParser(description="GRPO fine-tuning for point policy (SAM2 loop)")
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--sam_checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--max_points", type=int, default=16)
    p.add_argument("--metric", type=str, default="DICE")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--beta_kl", type=float, default=0.01)
    p.add_argument("--beta_kl_label", type=float, default=0.2)
    p.add_argument("--beta_kl_pixel", type=float, default=0.8)
    p.add_argument("--pixel_temp_start", type=float, default=1.5)
    p.add_argument("--pixel_temp_end", type=float, default=0.7)
    p.add_argument("--label_temp_start", type=float, default=1.0)
    p.add_argument("--label_temp_end", type=float, default=0.7)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--tb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--init_policy", type=str, default=None, help="Supervised checkpoint (.pt) for initialization and KL reference")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--max_val_items", type=int, default=64)
    args_ns = p.parse_args()

    args = GRPOArgs(
        train_json=args_ns.train_json,
        sam_checkpoint=args_ns.sam_checkpoint,
        out_dir=args_ns.out_dir,
        height=args_ns.height,
        width=args_ns.width,
        stride=args_ns.stride,
        max_points=args_ns.max_points,
        metric=args_ns.metric,
        epochs=args_ns.epochs,
        batch_size=args_ns.batch_size,
        group_size=args_ns.group_size,
        lr=args_ns.lr,
        weight_decay=args_ns.weight_decay,
        clip_eps=args_ns.clip_eps,
        beta_kl=args_ns.beta_kl,
        beta_kl_label=args_ns.beta_kl_label,
        beta_kl_pixel=args_ns.beta_kl_pixel,
        pixel_temp_start=args_ns.pixel_temp_start,
        pixel_temp_end=args_ns.pixel_temp_end,
        label_temp_start=args_ns.label_temp_start,
        label_temp_end=args_ns.label_temp_end,
        save_every=args_ns.save_every,
        val_every=args_ns.val_every,
        resume=args_ns.resume,
        tb=args_ns.tb,
        seed=args_ns.seed,
        device=args_ns.device,
        init_policy=args_ns.init_policy,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else _device()

    # data
    train_ds = JsonArrayDataset(args.train_json, split="train", val_ratio=args_ns.val_ratio, seed=args.seed)
    val_ds = JsonArrayDataset(args.train_json, split="val", val_ratio=args_ns.val_ratio, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # env
    env = SAMEnv(args.sam_checkpoint, device=args.device)

    # policy
    cfg = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
    policy = PointHeatmapModel(cfg).to(device)
    if args.init_policy and os.path.isfile(args.init_policy):
        sd = torch.load(args.init_policy, map_location="cpu")
        sd = sd.get("model", sd)
        policy.load_state_dict(sd, strict=False)
        print(f"[init] loaded policy from {args.init_policy}")

    # reference policy for KL (fixed)
    ref_policy = PointHeatmapModel(cfg).to(device)
    ref_policy.load_state_dict(policy.state_dict(), strict=False)
    ref_policy.eval()
    for p_ref in ref_policy.parameters():
        p_ref.requires_grad_(False)

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    global_step = 0
    ref_sd = ref_policy.state_dict()

    if args.resume and os.path.isfile(args.resume):
        try:
            start_epoch, global_step, ref_sd = _load_checkpoint(args.resume, policy, optim)
            ref_policy.load_state_dict(ref_sd, strict=False)
            print(f"[resume] loaded checkpoint {args.resume} @ epoch={start_epoch} step={global_step}")
        except Exception as e:
            print(f"[resume] failed to load: {e}")

    writer = None
    if args.tb:
        try:
            # Try torch TensorBoard first; fall back to tensorboardX if unavailable
            try:
                from torch.utils.tensorboard import SummaryWriter as _TBWriter
            except Exception:
                from tensorboardX import SummaryWriter as _TBWriter  # type: ignore
            log_dir = os.path.join(args.out_dir, "tb")
            os.makedirs(log_dir, exist_ok=True)
            print(f"[tb] logging to {log_dir}")
            writer = _TBWriter(log_dir=log_dir)
        except Exception as e:
            print(f"[tb] disabled: {e}. Install 'tensorboard' or 'tensorboardX' to enable logs.")
            writer = None

    policy.train()
    _set_label_bn_eval(policy)
    use_cuda_amp = (device.type == "cuda")
    scaler = amp.GradScaler("cuda" if use_cuda_amp else "cpu", enabled=use_cuda_amp)

    def log_scalar(name: str, val: float, step: int) -> None:
        print(f"{name}={val:.6f} @ {step}")
        if writer is not None:
            writer.add_scalar(name, val, step)

    for epoch in range(start_epoch, args.epochs):
        for batch in train_loader:
            # snapshot old policy for this update
            old_policy = PointHeatmapModel(cfg).to(device)
            old_policy.load_state_dict(policy.state_dict(), strict=False)
            old_policy.eval()
            for p_old in old_policy.parameters():
                p_old.requires_grad_(False)

            # temperature for this step (anneal over global steps)
            prog = 0.0 if args.epochs <= 1 else float(epoch) / float(max(1, args.epochs - 1))
            temp_pix = _interp(args.pixel_temp_start, args.pixel_temp_end, prog)
            temp_lbl = _interp(args.label_temp_start, args.label_temp_end, prog)

            # accumulate losses across group rollouts and steps
            total_loss = 0.0
            total_kl = 0.0
            total_adv = 0.0
            num_terms = 0

            # For each time step, we will collect group rewards and logprobs per sample
            B = args.batch_size
            G = args.group_size
            # Per-sample trajectories: list of length G, each contains per-step data
            batch_recs: List[Dict[str, Any]] = [{"image": batch["image"][i], "gt_mask": batch["gt_mask"][i]} for i in range(B)]

            # Rollout per group
            # We store for each t: rewards[B,G], logp[B,G], old_logp[B,G], kl[B]
            max_T = args.max_points
            rewards_t = [torch.zeros(B, G, device=device) for _ in range(max_T)]
            logp_t = [torch.zeros(B, G, device=device) for _ in range(max_T)]
            old_logp_t = [torch.zeros(B, G, device=device) for _ in range(max_T)]
            kl_t = [torch.zeros(B, device=device) for _ in range(max_T)]  # KL computed per state (not grouped)

            # Initialize per-(b,g) point histories and prev masks
            pts = [[[] for _ in range(G)] for _ in range(B)]  # type: ignore[var-annotated]
            lbs = [[[] for _ in range(G)] for _ in range(B)]  # type: ignore[var-annotated]
            prev_masks: List[List[Optional[np.ndarray]]] = [[None for _ in range(G)] for _ in range(B)]

            gt_cache: List[np.ndarray] = []
            rgb_cache: List[np.ndarray] = []
            for b in range(B):
                gt_bool = _load_mask_bool(batch_recs[b]["gt_mask"])
                rgb = _load_image_rgb(batch_recs[b]["image"])
                gt_cache.append(gt_bool)
                rgb_cache.append(rgb)

            # clear grads before accumulating per-time-step backward
            optim.zero_grad(set_to_none=True)

            for t in range(max_T):
                # collect metrics at t-1 to compute reward increment
                prev_score = torch.zeros(B, G, device=device)
                for b in range(B):
                    gt_bool = gt_cache[b]
                    for g in range(G):
                        if t == 0:
                            prev = np.zeros_like(gt_bool, dtype=np.uint8)
                        else:
                            prev = prev_masks[b][g]
                            if prev is None:
                                prev = np.zeros_like(gt_bool, dtype=np.uint8)
                        prev_bool = prev > 0
                        m_prev = _compute_metrics(gt_bool, prev_bool)
                        prev_score[b, g] = torch.tensor(_metric_value(m_prev, args.metric), device=device)

                # compute policy outputs and sample actions for each (b,g)
                for b in range(B):
                    image_path = batch_recs[b]["image"]
                    for g in range(G):
                        rgb_t, g_t, (orig_h, orig_w) = _prepare_model_inputs(image_path, prev_masks[b][g], (args.height, args.width))
                        with amp.autocast("cuda", enabled=use_cuda_amp):
                            logits, label_logits = policy(rgb_t.to(device), g_t.to(device)) ###!!!
                        # old/ref in inference + autocast to save memory
                        with torch.inference_mode():
                            with amp.autocast("cuda", enabled=use_cuda_amp):
                                logits_old, label_logits_old = old_policy(rgb_t.to(device), g_t.to(device))
                                logits_ref, label_logits_ref = ref_policy(rgb_t.to(device), g_t.to(device))

                        # sample action
                        act = sample_joint_label_cell_offset(
                            logits=logits,
                            label_logits=label_logits,
                            stride=args.stride,
                            temperature_pixel=temp_pix,
                            temperature_label=temp_lbl,
                        )
                        # recompute old log_prob for importance ratio
                        lp_old = log_prob_of_joint_action(
                            logits=logits_old,
                            label_logits=label_logits_old,
                            stride=args.stride,
                            label_idx=act["label_idx"],
                            cell_idx=act["cell_idx"],
                            sub_idx=act["sub_idx"],
                            temperature_pixel=temp_pix,
                            temperature_label=temp_lbl,
                        )
                        logp_t[t][b, g] = act["log_prob"]
                        old_logp_t[t][b, g] = lp_old

                        # KL at state (not grouped)
                        kl_label = _calc_kl_categorical(label_logits, label_logits_ref)  # [1]
                        kl_pixel = _calc_kl_pixel_full(logits, logits_ref)  # [1]
                        kl_val = args.beta_kl * (args.beta_kl_label * kl_label + args.beta_kl_pixel * kl_pixel)
                        kl_t[t][b] = kl_val.squeeze(0)

                        # map action to original image size
                        xy = act["pixel_xy"][0].to(torch.float32)
                        x = float(xy[0].item()) + 0.5
                        y = float(xy[1].item()) + 0.5
                        inf_H, inf_W = rgb_t.shape[-2], rgb_t.shape[-1]
                        scale_x = float(orig_w) / float(inf_W)
                        scale_y = float(orig_h) / float(inf_H)
                        x = max(0.0, min(orig_w - 1.0, x * scale_x))
                        y = max(0.0, min(orig_h - 1.0, y * scale_y))
                        pts[b][g].append((x, y))
                        lbs[b][g].append(int(act["label_idx"].item()))

                # step env for all (b,g)
                for b in range(B):
                    gt_bool = gt_cache[b]
                    rgb = rgb_cache[b]
                    for g in range(G):
                        pred_mask = env.predict_mask(rgb, pts[b][g], lbs[b][g])
                        prev_masks[b][g] = pred_mask
                        # compute reward increment
                        if pred_mask.shape[:2] != gt_bool.shape[:2]:
                            pred_bool = _resize_bool(pred_mask > 0, gt_bool.shape[1], gt_bool.shape[0])
                        else:
                            pred_bool = pred_mask > 0
                        m_now = _compute_metrics(gt_bool, pred_bool)
                        cur = _metric_value(m_now, args.metric)
                        rewards_t[t][b, g] = torch.tensor(cur, device=device) - prev_score[b, g]

                # per-time-step PPO objective and backward to avoid holding graphs for all T
                r = rewards_t[t]  # [B,G]
                mean = r.mean(dim=1, keepdim=True)
                std = r.std(dim=1, keepdim=True)
                adv = (r - mean) / (std + 1e-4)  # [B,G]
                ratio = torch.exp(logp_t[t] - old_logp_t[t])  # [B,G]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                obj = torch.minimum(unclipped, clipped)  # [B,G]
                obj = obj - kl_t[t].unsqueeze(1)
                loss_t = -obj.mean()

                scaler.scale(loss_t).backward()

            # Compute group-relative advantages per time-step
            losses = []
            for t in range(max_T):
                r = rewards_t[t]  # [B,G]
                mean = r.mean(dim=1, keepdim=True)
                std = r.std(dim=1, keepdim=True)
                adv = (r - mean) / (std + 1e-4)  # [B,G]
                # PPO ratio with clipping
                ratio = torch.exp(logp_t[t] - old_logp_t[t])  # [B,G]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                obj = torch.minimum(unclipped, clipped)  # [B,G]
                # subtract KL (broadcast B to [B,G])
                obj = obj - kl_t[t].unsqueeze(1)
                # negative for gradient descent
                loss_t = -obj.mean()
                losses.append(loss_t)

            # optimizer step with AMP
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            global_step += 1
            # Note: per-step loss already backpropagated; for logging compute average loss proxy
            avg_r = torch.stack([r.mean() for r in rewards_t]).mean().item()
            log_scalar("train/avg_reward", float(avg_r), global_step)
            # optional: additional logs every 10 steps can be added here

            if args.save_every > 0 and (global_step % args.save_every == 0):
                ckpt_path = os.path.join(args.out_dir, f"ckpt_step{global_step}.pt")
                _save_checkpoint(ckpt_path, policy, optim, epoch, global_step, args, ref_policy.state_dict())
                print(f"[ckpt] saved to {ckpt_path}")

            if args.val_every > 0 and (global_step % args.val_every == 0):
                val = validate(policy, val_ds, env, device, args, max_items=args_ns.max_val_items)
                # validate() ends with model.train(); re-freeze BN in label head to avoid B=1 BN error
                _set_label_bn_eval(policy)
                log_scalar("val/metric", val.get("VAL_METRIC", 0.0), global_step)

        # end epoch
        ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch{epoch+1}.pt")
        _save_checkpoint(ckpt_path, policy, optim, epoch + 1, global_step, args, ref_policy.state_dict())
        print(f"[ckpt] saved epoch {epoch+1} to {ckpt_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train()



```

- 我已修复两个问题：
  - 将已弃用的 `torch.cuda.amp.autocast/GradScaler` 切换为 `torch.amp.autocast/GradScaler`（按设备传入 `'cuda'`）。
  - 修正 GradScaler 使用顺序：在时间步循环之前 `optimizer.zero_grad()`，每个时间步 `scaler.scale(loss_t).backward()` 累积梯度；循环后 `scaler.unscale_(optimizer)` → 梯度裁剪 → `scaler.step(optimizer)` → `scaler.update()`。这避免了 “No inf checks were recorded for this optimizer” 的断言。

- 你现在可以重试运行；若仍有显存压力，可把 `group_size` 或 `max_points` 稍微下调一点点以进一步降低峰值内存。

