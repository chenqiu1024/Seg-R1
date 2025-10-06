from __future__ import annotations

import argparse
import os
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from PIL import Image

try:
    from .datasets import JsonlPointDataset, SamSequencePointDataset, ImageSize, collate_fn
    from .model import ModelConfig, PointHeatmapModel, argmax_from_logits, soft_argmax_from_logits
    from .losses import ce_over_pixels, kl_to_gaussian_targets, mse_to_gaussian_targets
    from .utils import save_checkpoint
    from .utils import draw_cross, draw_triangle, draw_diagonal_cross, overlay_heatmap, make_grid
except ImportError:  # allow running as a script without package context
    import sys as _sys
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from heatmap.datasets import JsonlPointDataset, SamSequencePointDataset, ImageSize, collate_fn
    from heatmap.model import ModelConfig, PointHeatmapModel, argmax_from_logits, soft_argmax_from_logits
    from heatmap.losses import ce_over_pixels, kl_to_gaussian_targets, mse_to_gaussian_targets
    from heatmap.utils import save_checkpoint
    from heatmap.utils import draw_cross, draw_triangle, draw_diagonal_cross, overlay_heatmap, make_grid

"""
训练热力图分类点定位模型，支持软高斯目标分布

# 数据准备 - 从mask生成训练数据:
# python seg-rl/annotator/gen_point_jsonl_from_masks.py \\
#   --images_dir /path/to/images \\
#   --masks_dir /path/to/masks \\
#   --output_jsonl /path/to/training_data.jsonl

# 支持的JSONL格式:
#   新格式: {"image": "/path/img.jpg", "points": [[x,y]], "labels": [1]}
#   旧格式: {"image": "/path/img.jpg", "x": x, "y": y}

# 推荐用法（UNet + KL软目标，更适合生成平滑的距离衰减热力图）:
# python -m seg-rl.heatmap.train \
#   --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/mask_salient_points-0.jsonl \
#   --height 512 --width 512 \
#   --arch unet_s \
#   --loss kl --sigma 6.0 --tau 1.0 \
#   --batch_size 16 --epochs 40 --amp \
#   --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
#   --eval_thresh 5.0 --save_every 5 \
#   --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/heatmap_train-0 \
#   --vis_mode sample --vis_count 16

/opt/anaconda3/envs/seg-r1/bin/python -m seg-rl.heatmap.train \
  --jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl \
  --sam_dir datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks-251001 \
  --height 512 --width 512 --arch unet_s \
  --loss kl --sigma 6.0 --tau 1.0 \
  --batch_size 16 --epochs 50 --amp \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
  --save_every 1 --save_steps 500 --progress --auto_resume \
  --out_dir outputs/braintumour/heatmap_train-251001

高分辨率场景（增大sigma获得更软的分布）:
python -m seg-rl.heatmap.train \
  --data_jsonl /root/autodl-tmp/works/Seg-R0/datasets/seg_r1_md/Task01_BrainTumour/mask_salient_points-0.jsonl \
  --height 1024 --width 1024 \
  --arch unet_s \
  --loss kl --sigma 10.0 --tau 1.2 \
  --batch_size 8 --epochs 50 --amp \
  --lr 1e-4 --weight_decay 1e-4 \
  --out_dir /root/autodl-tmp/works/Seg-R0/outputs/seg_r1_md/Task01_BrainTumour/heatmap_train-0 \
  --vis_mode sample --vis_count 16

MSE损失选项（更稳定的形状匹配）:
python -m seg-rl.heatmap.train \
  --data_jsonl /path/to/your/data.jsonl \
  --arch unet_s --loss mse --sigma 8.0 \
  --epochs 30 --amp
"""
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train heatmap model with self-supervised sequence dataset (RGB + Gray)")
    p.add_argument("--jsonl", type=str, required=True, help="Path to JSONL (points+labels per image)")
    p.add_argument("--sam_dir", type=str, required=True, help="Directory of SAM masks per step: {sam_dir}/{stem}/{k}.png")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--arch", type=str, choices=["unet_s", "resnet18"], default="unet_s")
    p.add_argument("--loss", type=str, choices=["ce", "kl", "mse"], default="kl")
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma for KL/MSE targets")
    p.add_argument("--tau", type=float, default=1.0, help="Temperature for KL/model softmax")
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
    p.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every N steps (0 to disable)")
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bar during training")
    p.add_argument("--auto_resume", action="store_true", help="If set and --resume not provided, try <out_dir>/last.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Prefer CUDA, then Apple Metal (MPS), else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # AMP only on CUDA for stability
    amp_enabled = args.amp and (device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = args.plots_dir or os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    vis_dir = args.vis_dir or os.path.join(args.out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    image_size = ImageSize(height=args.height, width=args.width)
    # Load once to get length for splitting
    full_ds_for_len = SamSequencePointDataset(args.jsonl, sam_dir=args.sam_dir, image_size=image_size, training=True)
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
    train_base = SamSequencePointDataset(args.jsonl, sam_dir=args.sam_dir, image_size=image_size, training=True)
    val_base = SamSequencePointDataset(args.jsonl, sam_dir=args.sam_dir, image_size=image_size, training=False) if n_val > 0 else None
    test_base = SamSequencePointDataset(args.jsonl, sam_dir=args.sam_dir, image_size=image_size, training=False) if n_test > 0 else None

    train_ds = Subset(train_base, idx_train)
    val_ds = Subset(val_base, idx_val) if val_base is not None else None
    test_ds = Subset(test_base, idx_test) if test_base is not None else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn) if test_ds is not None else None

    cfg = ModelConfig(backbone=args.arch, pretrained=args.pretrained, in_channels=4)
    model = PointHeatmapModel(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    global_step = 0
    tried_auto = False
    if args.resume and os.path.isfile(args.resume):
        from .utils import load_checkpoint
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler)
        start_epoch = ckpt.epoch
        global_step = ckpt.step
        print(f"[Resume] Loaded checkpoint from {args.resume} (epoch={start_epoch}, step={global_step})")
    elif args.auto_resume:
        auto_path = os.path.join(args.out_dir, "last.pt")
        if os.path.isfile(auto_path):
            tried_auto = True
            from .utils import load_checkpoint
            ckpt = load_checkpoint(auto_path, model, optimizer, scaler)
            start_epoch = ckpt.epoch
            global_step = ckpt.step
            print(f"[Auto-Resume] Loaded checkpoint from {auto_path} (epoch={start_epoch}, step={global_step})")

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
                logits, label_logits = model(img)
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
                img_t = batch["image_rgb"]  # [B,3,H,W] use RGB for visualization
                tgt = batch["target_xy"]  # [B,2]
                # Build 4ch input for the model: RGB + Gray
                img4 = torch.cat([img_t.to(device), batch["image_gray"].to(device)], dim=1)
                logits, label_logits = model(img4)
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
        # Progress bar setup
        _iter = train_loader
        _tqdm = None
        if args.progress:
            try:
                from tqdm import tqdm  # type: ignore
                _tqdm = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
            except Exception:
                _tqdm = None
        for bi, batch in enumerate(train_loader, 1):
            img = batch["image"].to(device, non_blocking=True)
            tgt = batch["target_xy"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                logits, label_logits = model(img)
                if args.loss == "ce":
                    loss_hm = ce_over_pixels(logits, tgt)
                elif args.loss == "kl":
                    loss_hm = kl_to_gaussian_targets(logits, tgt, sigma=args.sigma, tau=args.tau)
                else:
                    loss_hm = mse_to_gaussian_targets(logits, tgt, sigma=args.sigma)
                loss_label = nn.CrossEntropyLoss()(label_logits, batch["target_label"].to(device))
                loss = loss_hm + 0.2 * loss_label
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

            # Periodic checkpoint
            if args.save_steps > 0 and (global_step % args.save_steps == 0):
                save_path = os.path.join(args.out_dir, f"step_{global_step}.pt")
                save_checkpoint(save_path, model, optimizer, scaler, epoch=epoch + 1, step=global_step)
                # also update last.pt symlink-like copy
                last_path = os.path.join(args.out_dir, "last.pt")
                save_checkpoint(last_path, model, optimizer, scaler, epoch=epoch + 1, step=global_step)

            # Progress output
            if _tqdm is not None:
                _tqdm.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
                _tqdm.update(1)
            elif (bi % max(1, len(train_loader)//10)) == 0:
                print(f"Epoch {epoch+1}/{args.epochs} [{bi}/{len(train_loader)}] loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if _tqdm is not None:
            _tqdm.close()

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

        # Save epoch checkpoint and update last.pt
        if ((epoch + 1) % max(1, args.save_every)) == 0:
            save_path = os.path.join(args.out_dir, f"model_epoch_{epoch+1}.pt")
            save_checkpoint(save_path, model, optimizer, scaler, epoch=epoch+1, step=global_step)
        last_path = os.path.join(args.out_dir, "last.pt")
        save_checkpoint(last_path, model, optimizer, scaler, epoch=epoch+1, step=global_step)


if __name__ == "__main__":
    main()


