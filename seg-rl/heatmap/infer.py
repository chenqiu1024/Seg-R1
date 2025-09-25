from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from .model import ModelConfig, PointHeatmapModel, argmax_from_logits, soft_argmax_from_logits

"""
热力图点定位模型推理

基础用法（与训练时保持架构一致）:
python -m seg-rl.heatmap.infer \
  --images /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
  --ckpt /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/heatmap_train-0/model_epoch_40.pt \
  --height 512 --width 512 \
  --arch unet_s \
  --soft --temperature 1.0 \
  --save_json /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/heatmap_train-0/pred_salient_points.jsonl

单张图片推理:
python -m seg_rl.heatmap.infer \
  --images /path/to/image.jpg \
  --ckpt /path/to/checkpoint.pt \
  --arch unet_s --soft

批量处理多个路径:
python -m seg_rl.heatmap.infer \
  --images /path/to/dir1 /path/to/dir2 /path/to/image.png \
  --ckpt /path/to/checkpoint.pt \
  --arch unet_s --soft --save_json all_results.json
"""


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
    p.add_argument("--arch", type=str, choices=["unet_s", "resnet18"], default="unet_s")
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

    cfg = ModelConfig(backbone=args.arch, pretrained=False)
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
        results.append({"image": p, "points": [[x, y]], "labels": [1]})

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()


