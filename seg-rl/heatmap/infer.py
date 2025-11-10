from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch
from PIL import Image
import torchvision.transforms.functional as TF

try:
    from .model import ModelConfig, PointHeatmapModel, PointHeatmapModelWithSAM, argmax_from_logits, soft_argmax_from_logits
    from .utils import load_checkpoint
except ImportError:  # allow running as a script without package context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.model import ModelConfig, PointHeatmapModel, PointHeatmapModelWithSAM, argmax_from_logits, soft_argmax_from_logits
    from heatmap.utils import load_checkpoint

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


def _print_progress(note: str) -> None:
    print(f"\r{note}", end="", flush=True)


def load_image(path: str, height: int, width: int):
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((width, height), resample=Image.BILINEAR)
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # return original image size for coordinate scaling back to original space
    return (orig_w, orig_h), t.unsqueeze(0)


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
    
    # SAM Late LoRA 相关参数（用于加载 SAM-based checkpoint）
    p.add_argument("--sam_checkpoint", type=str, default=None, 
                   help="Path to SAM checkpoint (auto-detected from model checkpoint if not provided)")
    
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

    # 加载 checkpoint 元数据以检测模型类型
    state = torch.load(args.ckpt, map_location="cpu")
    metadata = state.get("metadata", {})
    use_sam_encoder = metadata.get("use_sam_encoder", False)
    
    # 检测 PEFT 方法（向后兼容）
    peft_method = metadata.get("sam_peft_method")
    if peft_method is None and metadata.get("sam_lora_enabled", False):
        peft_method = "late_lora"
    
    print(f"[Inference] Loading checkpoint from {args.ckpt}")
    print(f"[Inference] Model type: {'SAM encoder' if use_sam_encoder else 'Standard'}")
    if use_sam_encoder:
        print(f"[Inference] PEFT method: {peft_method or 'None (frozen)'}")
    
    # 根据 checkpoint 类型创建模型
    if use_sam_encoder:
        # 需要 SAM checkpoint 路径
        sam_checkpoint = args.sam_checkpoint
        if sam_checkpoint is None:
            # 尝试从元数据中获取
            sam_checkpoint = metadata.get("sam_checkpoint")
        
        if sam_checkpoint is None:
            print("[Error] SAM checkpoint path required for SAM-based model")
            print("[Error] Please provide --sam_checkpoint argument")
            return
        
        print(f"[Inference] SAM checkpoint: {sam_checkpoint}")
        
        # 解析 Conv-LoRA blocks（如果有）
        conv_lora_blocks = metadata.get("sam_conv_lora_blocks")
        
        cfg = ModelConfig(
            backbone=args.arch,
            pretrained=False,
            main_in_channels=3,
            cond_in_channels=1,
            use_sam_encoder=True,
            sam_checkpoint=sam_checkpoint,
            sam_peft_method=peft_method,
            sam_freeze_encoder=True,  # 推理时总是冻结
            # Late LoRA 参数
            sam_lora_enabled=(peft_method == "late_lora"),
            sam_lora_rank=metadata.get("sam_lora_rank", 8),
            sam_lora_alpha=metadata.get("sam_lora_alpha", 16.0),
            sam_lora_dropout=metadata.get("sam_lora_dropout", 0.0),
            # Conv-LoRA 参数
            sam_conv_lora_rank=metadata.get("sam_conv_lora_rank", 8),
            sam_conv_lora_alpha=metadata.get("sam_conv_lora_alpha", 16.0),
            sam_conv_lora_kernel_size=metadata.get("sam_conv_lora_kernel_size", 3),
            sam_conv_lora_dropout=metadata.get("sam_conv_lora_dropout", 0.0),
            sam_conv_lora_blocks=conv_lora_blocks,
        )
        model = PointHeatmapModelWithSAM(cfg).to(device)
    else:
        # 标准模型
        cfg = ModelConfig(backbone=args.arch, pretrained=False)
        model = PointHeatmapModel(cfg).to(device)
    
    # 加载模型参数
    load_checkpoint(args.ckpt, model, strict=False)
    model.eval()
    print(f"[Inference] Model loaded successfully\n")

    paths = expand_paths(args.images)
    results = []
    total = len(paths)
    for idx, p in enumerate(paths, 1):
        (orig_w, orig_h), t = load_image(p, args.height, args.width)
        t = t.to(device)
        with torch.no_grad():
            # SAM-based 模型需要分离输入
            if use_sam_encoder:
                # 对于单张图片推理，条件图像为空（第一步）
                cond = torch.zeros(1, 1, args.height, args.width, device=device)
                logits, _ = model(t, cond)
            else:
                logits, _ = model(t)
        
            if args.soft:
                xy = soft_argmax_from_logits(logits, temperature=args.temperature)[0]
            else:
                xy = argmax_from_logits(logits)[0]
        # coordinates are in resized (width x height) space; scale back to original image size
        x_resized, y_resized = float(xy[0].item()), float(xy[1].item())
        scale_x = float(orig_w) / float(args.width)
        scale_y = float(orig_h) / float(args.height)
        x_orig = x_resized * scale_x
        y_orig = y_resized * scale_y
        _print_progress(f"[{idx}/{total}] {os.path.basename(p)} x={x_orig:.1f}, y={y_orig:.1f} orig={orig_w}x{orig_h}")
        results.append({"image": p, "points": [[x_orig, y_orig]], "labels": [1]})

    # finalize progress line
    print()

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()


