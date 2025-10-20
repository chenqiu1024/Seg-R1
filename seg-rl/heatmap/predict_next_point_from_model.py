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
except Exception:
    # script context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, soft_argmax_from_logits

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
    p.add_argument("--amp", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--debug_output_dir", type=str, default=None)
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


def _predict_point(model: PointHeatmapModel, rgb_tensor: torch.Tensor, gray_tensor: torch.Tensor, device: torch.device, tau: float) -> Tuple[float, float, int]:
    with torch.no_grad():
        logits, label_logits = model(rgb_tensor.to(device), gray_tensor.to(device))
        xy = soft_argmax_from_logits(logits, temperature=tau)[0]
        lab = int(label_logits.argmax(dim=1).item())
    return float(xy[0].item()), float(xy[1].item()), lab


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
            x, y, lab = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau)
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
            x, y, lab = _predict_point(model, rgb_tensor, gray_tensor, device, args.tau)
        # scale back and clamp
        inf_H, inf_W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
        if (inf_W, inf_H) != (orig_w, orig_h):
            scale_x = float(orig_w) / float(inf_W)
            scale_y = float(orig_h) / float(inf_H)
            x = x * scale_x
            y = y * scale_y
        x = float(max(0.0, min(orig_w - 1.0, x)))
        y = float(max(0.0, min(orig_h - 1.0, y)))
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


