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


