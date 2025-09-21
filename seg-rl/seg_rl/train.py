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


