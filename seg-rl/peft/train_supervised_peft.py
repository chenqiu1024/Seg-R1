#!/usr/bin/env python3
"""
PEFT监督预训练脚本

使用启发式生成的点序列数据训练SAM2 LoRA + 点预测网络

训练流程:
1. 加载SAM2并注入Late LoRA
2. 初始化点预测网络
3. 双优化器：SAM LoRA (小学习率) + 点网络 (正常学习率)
4. 监督loss: KL/MSE热力图 + CE标签
5. 定期验证并保存checkpoint

调用示例:
    # 基础训练
    python -m seg-rl.peft.train_supervised_peft \\
      --jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --lora_rank 16 --lora_alpha 32 \\
      --feature_scale 8 --fusion_mode film \\
      --image_size 512 512 \\
      --loss kl --sigma 8.0 --label_loss_weight 0.1 \\
      --batch_size 8 --epochs 40 --amp \\
      --lr_sam 1e-5 --lr_point 1e-4 \\
      --out_dir outputs/braintumour/peft_supervised \\
      --save_every 5 --auto_resume
    
    # 冻结SAM，只训练点网络（调试）
    python -m seg-rl.peft.train_supervised_peft \\
      --jsonl ... \\
      --freeze_sam \\
      --lr_point 1e-4 \\
      --out_dir outputs/braintumour/peft_frozen_sam
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.datasets_peft import PEFTPointDataset, collate_fn_peft
from peft.utils_peft import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    create_warmup_cosine_scheduler, compute_batch_pck,
    set_seed, count_parameters, AverageMeter
)

# 复用现有loss函数
try:
    from heatmap.losses import kl_to_gaussian_targets, mse_to_gaussian_targets
except ImportError:
    print("Warning: Could not import loss functions from heatmap module")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PEFT Supervised Pretraining")
    
    # 数据
    p.add_argument("--jsonl", type=str, required=True, help="Path to JSONL training data")
    p.add_argument("--image_size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"),
                   help="Image size (height width)")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--max_points_per_sample", type=int, default=None,
                   help="Max points to use per sample (None=all)")
    
    # SAM2 + LoRA
    p.add_argument("--sam_checkpoint", type=str, required=True, help="SAM2 checkpoint path")
    p.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--freeze_sam", action="store_true",
                   help="Freeze SAM (only train point predictor)")
    
    # 点预测网络
    p.add_argument("--feature_scale", type=int, default=8, choices=[4, 8, 16],
                   help="SAM feature scale (8=H/8, recommended)")
    p.add_argument("--fusion_mode", type=str, default="film", choices=["film", "concat"],
                   help="Feature fusion mode")
    
    # 损失函数
    p.add_argument("--loss", type=str, default="kl", choices=["kl", "mse"],
                   help="Heatmap loss type")
    p.add_argument("--sigma", type=float, default=8.0, help="Gaussian sigma for target heatmap")
    p.add_argument("--tau", type=float, default=1.0, help="Temperature for KL loss")
    p.add_argument("--label_loss_weight", type=float, default=0.1, help="Label loss weight")
    
    # 训练
    p.add_argument("--batch_size", type=int, default=8, help="Batch size")
    p.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    p.add_argument("--lr_sam", type=float, default=1e-5, help="Learning rate for SAM LoRA")
    p.add_argument("--lr_point", type=float, default=1e-4, help="Learning rate for point predictor")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    p.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    
    # 学习率调度
    p.add_argument("--lr_scheduler", type=str, default="warmup_cosine",
                   choices=["none", "warmup_cosine", "cosine", "step"],
                   help="Learning rate scheduler")
    p.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    
    # Checkpoint
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    p.add_argument("--auto_resume", action="store_true", help="Auto resume from latest checkpoint")
    p.add_argument("--resume", type=str, default=None, help="Resume from specific checkpoint")
    
    # 验证与评估
    p.add_argument("--eval_thresholds", type=str, default="8,12,16,20",
                   help="PCK thresholds (comma-separated)")
    
    # 其他
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu/mps)")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")
    
    return p.parse_args()


def compute_loss(
    heatmap_logits: torch.Tensor,
    target_points: torch.Tensor,
    label_logits: torch.Tensor,
    target_labels: torch.Tensor,
    loss_type: str,
    sigma: float,
    tau: float,
    label_weight: float,
) -> tuple:
    """
    计算总损失
    
    Returns:
        (total_loss, heatmap_loss, label_loss)
    """
    # 热力图loss
    if loss_type == "kl":
        heatmap_loss = kl_to_gaussian_targets(heatmap_logits, target_points, sigma=sigma, tau=tau)
    elif loss_type == "mse":
        heatmap_loss = mse_to_gaussian_targets(heatmap_logits, target_points, sigma=sigma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # 标签loss
    label_loss = F.cross_entropy(label_logits, target_labels)
    
    # 总loss
    total_loss = heatmap_loss + label_weight * label_loss
    
    return total_loss, heatmap_loss, label_loss


def train_one_epoch(
    epoch: int,
    sam2_lora: LoRASAM2Wrapper,
    point_predictor: PointPredictorFromSAMFeatures,
    train_loader: DataLoader,
    optimizer_sam: Optional[torch.optim.Optimizer],
    optimizer_point: torch.optim.Optimizer,
    args,
    scaler: Optional[torch.cuda.amp.GradScaler],
    writer: Optional[SummaryWriter],
    global_step: int,
) -> tuple:
    """
    训练一个epoch
    
    Returns:
        (avg_loss, global_step)
    """
    point_predictor.train()
    
    loss_meter = AverageMeter()
    heatmap_loss_meter = AverageMeter()
    label_loss_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(args.device)
        prev_masks = batch['prev_mask'].to(args.device)
        target_points = batch['target_point'].to(args.device)
        target_labels = batch['target_label'].to(args.device)
        
        # 前向传播
        with torch.cuda.amp.autocast(enabled=args.amp):
            # 提取SAM特征
            sam_features = sam2_lora.get_image_features(images, feature_scale=args.feature_scale)
            
            # 点预测网络
            heatmap_logits, label_logits = point_predictor(sam_features, prev_masks)
            
            # 计算loss
            total_loss, heatmap_loss, label_loss = compute_loss(
                heatmap_logits, target_points, label_logits, target_labels,
                args.loss, args.sigma, args.tau, args.label_loss_weight
            )
        
        # 反向传播
        if optimizer_sam is not None:
            optimizer_sam.zero_grad()
        optimizer_point.zero_grad()
        
        if scaler is not None:
            scaler.scale(total_loss).backward()
            
            # 梯度裁剪
            if args.grad_clip > 0:
                if optimizer_sam is not None:
                    scaler.unscale_(optimizer_sam)
                    torch.nn.utils.clip_grad_norm_(sam2_lora.get_lora_parameters(), args.grad_clip)
                scaler.unscale_(optimizer_point)
                torch.nn.utils.clip_grad_norm_(point_predictor.parameters(), args.grad_clip)
            
            if optimizer_sam is not None:
                scaler.step(optimizer_sam)
            scaler.step(optimizer_point)
            scaler.update()
        else:
            total_loss.backward()
            
            # 梯度裁剪
            if args.grad_clip > 0:
                if optimizer_sam is not None:
                    torch.nn.utils.clip_grad_norm_(sam2_lora.get_lora_parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(point_predictor.parameters(), args.grad_clip)
            
            if optimizer_sam is not None:
                optimizer_sam.step()
            optimizer_point.step()
        
        # 更新统计
        batch_size = images.size(0)
        loss_meter.update(total_loss.item(), batch_size)
        heatmap_loss_meter.update(heatmap_loss.item(), batch_size)
        label_loss_meter.update(label_loss.item(), batch_size)
        
        # TensorBoard
        if writer is not None:
            writer.add_scalar('train/total_loss', total_loss.item(), global_step)
            writer.add_scalar('train/heatmap_loss', heatmap_loss.item(), global_step)
            writer.add_scalar('train/label_loss', label_loss.item(), global_step)
        
        global_step += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'hm': f'{heatmap_loss_meter.avg:.4f}',
            'lb': f'{label_loss_meter.avg:.4f}',
        })
    
    return loss_meter.avg, global_step


@torch.no_grad()
def validate(
    sam2_lora: LoRASAM2Wrapper,
    point_predictor: PointPredictorFromSAMFeatures,
    val_loader: DataLoader,
    args,
    writer: Optional[SummaryWriter],
    epoch: int,
) -> dict:
    """
    验证
    
    Returns:
        metrics dict
    """
    point_predictor.eval()
    
    loss_meter = AverageMeter()
    pck_meters = {thresh: AverageMeter() for thresh in args.eval_thresholds}
    
    for batch in tqdm(val_loader, desc="Validating"):
        images = batch['image'].to(args.device)
        prev_masks = batch['prev_mask'].to(args.device)
        target_points = batch['target_point'].to(args.device)
        target_labels = batch['target_label'].to(args.device)
        
        # 前向传播
        sam_features = sam2_lora.get_image_features(images, feature_scale=args.feature_scale)
        heatmap_logits, label_logits = point_predictor(sam_features, prev_masks)
        
        # 计算loss
        total_loss, _, _ = compute_loss(
            heatmap_logits, target_points, label_logits, target_labels,
            args.loss, args.sigma, args.tau, args.label_loss_weight
        )
        
        # 预测点（使用argmax）
        B, _, H, W = heatmap_logits.shape
        heatmap_flat = heatmap_logits.view(B, -1)
        pred_idx = heatmap_flat.argmax(dim=1)
        pred_y = (pred_idx // W).float()
        pred_x = (pred_idx % W).float()
        pred_points = torch.stack([pred_x, pred_y], dim=1)
        
        # 计算PCK
        for thresh in args.eval_thresholds:
            pck, _ = compute_batch_pck(pred_points, target_points, thresh)
            pck_meters[thresh].update(pck, images.size(0))
        
        loss_meter.update(total_loss.item(), images.size(0))
    
    # 汇总metrics
    metrics = {
        'val_loss': loss_meter.avg,
    }
    for thresh in args.eval_thresholds:
        metrics[f'val_pck@{thresh}'] = pck_meters[thresh].avg
    
    # TensorBoard
    if writer is not None:
        for key, val in metrics.items():
            writer.add_scalar(f'val/{key}', val, epoch)
    
    # 打印
    print(f"\nValidation Results:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")
    
    return metrics


def main():
    args = parse_args()
    
    # 解析eval_thresholds
    args.eval_thresholds = [float(t) for t in args.eval_thresholds.split(',')]
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # TensorBoard
    writer = None
    if args.tb:
        writer = SummaryWriter(os.path.join(args.out_dir, 'tensorboard'))
    
    # 设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 1. 加载SAM2 + LoRA
    print("\n" + "="*80)
    print("Initializing SAM2 with LoRA...")
    print("="*80)
    sam2_lora = LoRASAM2Wrapper(
        sam_checkpoint=args.sam_checkpoint,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=str(device)
    )
    
    if args.freeze_sam:
        print("Freezing SAM (only training point predictor)")
        sam2_lora.freeze_sam_base()
    
    # 2. 初始化点预测网络
    print("\n" + "="*80)
    print("Initializing Point Predictor...")
    print("="*80)
    point_predictor = PointPredictorFromSAMFeatures(
        sam_feature_dim=256,
        output_size=tuple(args.image_size),
        fusion_mode=args.fusion_mode,
        feature_scale=args.feature_scale,
    ).to(device)
    
    total_params, trainable_params = count_parameters(point_predictor)
    print(f"Point Predictor: {trainable_params:,} / {total_params:,} trainable params")
    
    # 3. 数据加载
    print("\n" + "="*80)
    print("Loading Dataset...")
    print("="*80)
    full_dataset = PEFTPointDataset(
        jsonl_path=args.jsonl,
        image_size=tuple(args.image_size),
        augmentation=True,
        max_points_per_sample=args.max_points_per_sample,
    )
    
    # 划分训练/验证集
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_peft,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_peft,
        pin_memory=True,
    )
    
    # 4. 优化器
    print("\n" + "="*80)
    print("Initializing Optimizers...")
    print("="*80)
    if args.freeze_sam:
        optimizer_sam = None
        print("SAM optimizer: None (frozen)")
    else:
        optimizer_sam = torch.optim.AdamW(
            sam2_lora.get_lora_parameters(),
            lr=args.lr_sam,
            weight_decay=args.weight_decay
        )
        print(f"SAM optimizer: AdamW(lr={args.lr_sam})")
    
    optimizer_point = torch.optim.AdamW(
        point_predictor.parameters(),
        lr=args.lr_point,
        weight_decay=args.weight_decay
    )
    print(f"Point optimizer: AdamW(lr={args.lr_point})")
    
    # 5. 学习率调度器
    scheduler_sam = None
    scheduler_point = None
    if args.lr_scheduler == "warmup_cosine":
        if optimizer_sam is not None:
            scheduler_sam = create_warmup_cosine_scheduler(
                optimizer_sam, args.warmup_epochs, args.epochs
            )
        scheduler_point = create_warmup_cosine_scheduler(
            optimizer_point, args.warmup_epochs, args.epochs
        )
    elif args.lr_scheduler == "cosine":
        if optimizer_sam is not None:
            scheduler_sam = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_sam, T_max=args.epochs
            )
        scheduler_point = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_point, T_max=args.epochs
        )
    
    # 6. AMP scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    # 7. 自动续传或指定checkpoint
    start_epoch = 0
    global_step = 0
    best_val_pck = 0.0
    
    if args.resume:
        resume_path = args.resume
    elif args.auto_resume:
        resume_path = find_latest_checkpoint(args.out_dir)
    else:
        resume_path = None
    
    if resume_path and os.path.exists(resume_path):
        print(f"\nResuming from {resume_path}")
        start_epoch, global_step, config, metrics = load_checkpoint(
            resume_path, point_predictor, sam2_lora,
            optimizer_sam, optimizer_point, None, str(device)
        )
        start_epoch += 1
        best_val_pck = metrics.get('val_pck@8', 0.0)  # 使用第一个阈值作为主指标
    
    # 8. 训练循环
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs-1}")
        print(f"{'='*80}")
        
        # 训练
        avg_loss, global_step = train_one_epoch(
            epoch, sam2_lora, point_predictor, train_loader,
            optimizer_sam, optimizer_point, args, scaler, writer, global_step
        )
        
        # 学习率调度
        if scheduler_sam is not None:
            scheduler_sam.step()
        if scheduler_point is not None:
            scheduler_point.step()
        
        # 验证
        val_metrics = validate(
            sam2_lora, point_predictor, val_loader, args, writer, epoch
        )
        
        # 保存checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f'checkpoint_epoch{epoch:03d}.pt')
            save_checkpoint(
                ckpt_path, epoch, global_step,
                point_predictor, sam2_lora,
                optimizer_sam, optimizer_point, None,
                config=vars(args), metrics=val_metrics
            )
        
        # 保存最佳模型
        current_pck = val_metrics.get(f'val_pck@{args.eval_thresholds[0]}', 0.0)
        if current_pck > best_val_pck:
            best_val_pck = current_pck
            best_path = os.path.join(args.out_dir, 'checkpoint_best.pt')
            save_checkpoint(
                best_path, epoch, global_step,
                point_predictor, sam2_lora,
                optimizer_sam, optimizer_point, None,
                config=vars(args), metrics=val_metrics
            )
            print(f"New best model saved! PCK@{args.eval_thresholds[0]}: {best_val_pck:.4f}")
    
    if writer is not None:
        writer.close()
    
    print("\n" + "="*80)
    print("Training Completed!")
    print("="*80)
    print(f"Best Val PCK@{args.eval_thresholds[0]}: {best_val_pck:.4f}")
    print(f"Checkpoints saved in: {args.out_dir}")


if __name__ == "__main__":
    main()

