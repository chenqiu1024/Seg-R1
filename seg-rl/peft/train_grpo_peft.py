#!/usr/bin/env python3
"""
GRPO强化学习训练脚本（PEFT版本）

使用Group Relative Policy Optimization微调SAM2 LoRA + 点预测网络

训练流程:
1. 加载监督预训练的checkpoint作为初始策略
2. 创建参考策略（固定）
3. Rollout: 迭代预测点 → SAM2生成掩模 → 计算奖励
4. GRPO更新: 组内相对优势 + PPO裁剪 + KL惩罚
5. 定期验证并保存checkpoint

调用示例:
    python -m seg-rl.peft.train_grpo_peft \\
      --train_json datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251001.jsonl \\
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \\
      --init_policy outputs/braintumour/peft_supervised/checkpoint_best.pt \\
      --out_dir outputs/braintumour/peft_grpo \\
      --height 512 --width 512 --stride 8 \\
      --max_points 16 --group_size 4 \\
      --epochs 5 --batch_size 8 \\
      --lr_sam 5e-6 --lr_point 5e-5 \\
      --beta_kl 0.02 --beta_entropy 0.01 \\
      --clip_epsilon 0.2 \\
      --tb --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加父目录到路径
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from peft.lora_sam2 import LoRASAM2Wrapper
from peft.point_predictor_peft import PointPredictorFromSAMFeatures
from peft.utils_peft import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    compute_dice, set_seed, count_parameters, AverageMeter
)

# 复用层级采样函数
try:
    from heatmap.model import sample_joint_label_cell_offset, log_prob_of_joint_action
except ImportError:
    print("Warning: Could not import sampling functions from heatmap module")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO Training for PEFT")
    
    # 数据
    p.add_argument("--train_json", type=str, required=True, help="Training JSONL")
    p.add_argument("--height", type=int, default=512, help="Image height")
    p.add_argument("--width", type=int, default=512, help="Image width")
    
    # SAM2 + LoRA + 点预测网络
    p.add_argument("--sam_checkpoint", type=str, required=True, help="SAM2 checkpoint")
    p.add_argument("--init_policy", type=str, required=True,
                   help="Initial policy checkpoint (from supervised training)")
    
    # Rollout
    p.add_argument("--max_points", type=int, default=16, help="Max points per rollout")
    p.add_argument("--stride", type=int, default=8, help="Feature stride for sampling")
    p.add_argument("--dice_stop_threshold", type=float, default=0.95,
                   help="Stop rollout if Dice > threshold")
    p.add_argument("--improvement_stop_threshold", type=float, default=0.001,
                   help="Stop if improvement < threshold")
    
    # GRPO参数
    p.add_argument("--group_size", type=int, default=4,
                   help="Group size for relative advantage")
    p.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    p.add_argument("--beta_kl", type=float, default=0.02, help="KL penalty coefficient")
    p.add_argument("--beta_entropy", type=float, default=0.01, help="Entropy bonus coefficient")
    
    # 温度调度
    p.add_argument("--pixel_temp_start", type=float, default=1.5,
                   help="Initial pixel temperature")
    p.add_argument("--pixel_temp_end", type=float, default=0.8,
                   help="Final pixel temperature")
    p.add_argument("--label_temp_start", type=float, default=1.2,
                   help="Initial label temperature")
    p.add_argument("--label_temp_end", type=float, default=0.8,
                   help="Final label temperature")
    
    # 训练
    p.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size (number of images per update)")
    p.add_argument("--lr_sam", type=float, default=5e-6, help="Learning rate for SAM LoRA")
    p.add_argument("--lr_point", type=float, default=5e-5,
                   help="Learning rate for point predictor")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    
    # Checkpoint
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument("--save_every", type=int, default=100, help="Save every N steps")
    p.add_argument("--auto_resume", action="store_true", help="Auto resume")
    
    # 其他
    p.add_argument("--device", type=str, default="cuda", help="Device")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--tb", action="store_true", help="Enable TensorBoard")
    
    return p.parse_args()


class PolicyNetwork:
    """
    策略网络：SAM2 LoRA + 点预测器
    """
    
    def __init__(
        self,
        sam2_lora: LoRASAM2Wrapper,
        point_predictor: PointPredictorFromSAMFeatures,
        stride: int = 8,
    ):
        self.sam2 = sam2_lora
        self.point_predictor = point_predictor
        self.stride = stride
    
    def predict_next_point(
        self,
        image: torch.Tensor,
        prev_mask: torch.Tensor,
        pixel_temperature: float = 1.0,
        label_temperature: float = 1.0,
    ) -> Tuple[Dict, torch.Tensor]:
        """
        预测并采样下一个点
        
        Returns:
            (action_dict, log_prob)
        """
        # 提取SAM特征
        sam_features = self.sam2.get_image_features(image, feature_scale=self.stride)
        
        # 点预测网络
        heatmap_logits, label_logits = self.point_predictor(sam_features, prev_mask)
        
        # 层级采样
        action = sample_joint_label_cell_offset(
            heatmap_logits, label_logits,
            stride=self.stride,
            pixel_temperature=pixel_temperature,
            label_temperature=label_temperature
        )
        
        # 计算log_prob
        log_prob = log_prob_of_joint_action(
            heatmap_logits, label_logits, action,
            stride=self.stride,
            pixel_temperature=pixel_temperature,
            label_temperature=label_temperature
        )
        
        return action, log_prob


def rollout_single_image(
    image: np.ndarray,
    gt_mask: np.ndarray,
    policy: PolicyNetwork,
    args,
    pixel_temperature: float,
    label_temperature: float,
) -> List[Dict]:
    """
    对单张图像执行rollout
    
    Args:
        image: numpy array [H, W, 3]
        gt_mask: numpy array [H, W], 二值
        policy: 策略网络
        args: 参数
        pixel_temperature: 像素采样温度
        label_temperature: 标签采样温度
        
    Returns:
        trajectory: list of step dicts
    """
    device = args.device
    
    # 转换为tensor
    from PIL import Image as PILImage
    import torchvision.transforms.functional as TF
    
    image_pil = PILImage.fromarray(image)
    gt_mask_pil = PILImage.fromarray(gt_mask)
    
    image_resized = image_pil.resize((args.width, args.height), PILImage.BILINEAR)
    gt_mask_resized = gt_mask_pil.resize((args.width, args.height), PILImage.NEAREST)
    
    image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)  # [1, 3, H, W]
    gt_mask_tensor = TF.to_tensor(gt_mask_resized).to(device)  # [1, H, W]
    gt_mask_binary = (gt_mask_tensor > 0.5).float()
    
    # Rollout
    trajectory = []
    points_seq = []
    labels_seq = []
    prev_dice = 0.0
    
    for step in range(args.max_points):
        # 1. 生成当前掩模
        if step == 0:
            prev_mask = torch.zeros_like(gt_mask_binary)
        else:
            # 用SAM2生成掩模（no_grad）
            with torch.no_grad():
                # 将点坐标缩放回原始尺寸
                orig_h, orig_w = image.shape[:2]
                points_orig = [(px * orig_w / args.width, py * orig_h / args.height)
                              for px, py in points_seq]
                prev_mask_np = policy.sam2.predict_mask(image, points_orig, labels_seq)
                prev_mask = torch.from_numpy(prev_mask_np).unsqueeze(0).to(device)
                # Resize到训练尺寸
                prev_mask = F.interpolate(
                    prev_mask.unsqueeze(0), size=(args.height, args.width), mode='nearest'
                )[0]
        
        # 2. 策略预测（保持梯度）
        action, log_prob = policy.predict_next_point(
            image_tensor, prev_mask.unsqueeze(0),
            pixel_temperature, label_temperature
        )
        
        # 提取点坐标和标签
        point_xy = action['continuous_xy'][0]  # [2]
        label = action['label'][0]  # scalar
        
        points_seq.append((point_xy[0].item(), point_xy[1].item()))
        labels_seq.append(label.item())
        
        # 3. 环境反馈：生成新掩模
        with torch.no_grad():
            orig_h, orig_w = image.shape[:2]
            points_orig = [(px * orig_w / args.width, py * orig_h / args.height)
                          for px, py in points_seq]
            new_mask_np = policy.sam2.predict_mask(image, points_orig, labels_seq)
            new_mask = torch.from_numpy(new_mask_np).unsqueeze(0).to(device)
            new_mask = F.interpolate(
                new_mask.unsqueeze(0), size=(args.height, args.width), mode='nearest'
            )[0]
        
        # 4. 计算奖励
        dice_new = compute_dice(new_mask[0], gt_mask_binary[0])
        reward = dice_new - prev_dice  # 增量奖励
        
        # 5. 记录
        trajectory.append({
            'image': image_tensor.detach(),
            'prev_mask': prev_mask.detach(),
            'action': action,
            'log_prob': log_prob.detach(),
            'reward': reward,
            'dice': dice_new,
        })
        
        # 停止条件
        if dice_new > args.dice_stop_threshold:
            break
        if step > 0 and (dice_new - prev_dice) < args.improvement_stop_threshold:
            break
        
        prev_dice = dice_new
    
    return trajectory


def grpo_update(
    batch_trajectories: List[List[Dict]],
    policy: PolicyNetwork,
    ref_policy: PolicyNetwork,
    optimizer_sam: torch.optim.Optimizer,
    optimizer_point: torch.optim.Optimizer,
    args,
    pixel_temperature: float,
    label_temperature: float,
) -> Dict:
    """
    GRPO更新
    
    Args:
        batch_trajectories: list of trajectories (one per image)
        policy: 当前策略
        ref_policy: 参考策略（固定）
        optimizer_sam: SAM优化器
        optimizer_point: 点网络优化器
        args: 参数
        pixel_temperature: 像素温度
        label_temperature: 标签温度
        
    Returns:
        metrics dict
    """
    # 将batch分组
    num_images = len(batch_trajectories)
    num_groups = num_images // args.group_size
    
    if num_groups == 0:
        # batch太小，无法分组
        return {}
    
    total_policy_loss = 0.0
    total_kl_loss = 0.0
    total_entropy_loss = 0.0
    num_steps = 0
    
    for group_idx in range(num_groups):
        group_start = group_idx * args.group_size
        group_end = group_start + args.group_size
        group_trajs = batch_trajectories[group_start:group_end]
        
        # 计算组内总奖励
        group_rewards = []
        for traj in group_trajs:
            total_reward = sum([step['reward'] for step in traj])
            group_rewards.append(total_reward)
        
        # 组内平均奖励（基线）
        baseline = np.mean(group_rewards)
        
        # 计算每个trajectory的优势
        advantages = [r - baseline for r in group_rewards]
        
        # 对每个trajectory更新策略
        for traj, advantage in zip(group_trajs, advantages):
            for step_data in traj:
                # 重新前向传播（保持梯度）
                sam_features = policy.sam2.get_image_features(
                    step_data['image'], feature_scale=args.stride
                )
                heatmap_logits, label_logits = policy.point_predictor(
                    sam_features, step_data['prev_mask'].unsqueeze(0)
                )
                
                # 当前策略的log_prob
                log_prob_new = log_prob_of_joint_action(
                    heatmap_logits, label_logits, step_data['action'],
                    stride=args.stride,
                    pixel_temperature=pixel_temperature,
                    label_temperature=label_temperature
                )
                
                # 参考策略的log_prob（no_grad）
                with torch.no_grad():
                    ref_sam_features = ref_policy.sam2.get_image_features(
                        step_data['image'], feature_scale=args.stride
                    )
                    ref_heatmap, ref_label = ref_policy.point_predictor(
                        ref_sam_features, step_data['prev_mask'].unsqueeze(0)
                    )
                    log_prob_ref = log_prob_of_joint_action(
                        ref_heatmap, ref_label, step_data['action'],
                        stride=args.stride,
                        pixel_temperature=pixel_temperature,
                        label_temperature=label_temperature
                    )
                
                # 重要性比率
                log_prob_old = step_data['log_prob']
                ratio = torch.exp(log_prob_new - log_prob_old)
                
                # PPO裁剪
                clipped_ratio = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage
                )
                
                # KL惩罚（相对参考策略）
                kl_penalty = args.beta_kl * (log_prob_new - log_prob_ref)
                
                # 熵正则（鼓励探索）
                # 简化：使用heatmap logits的熵
                heatmap_probs = F.softmax(heatmap_logits.flatten(1) / pixel_temperature, dim=1)
                entropy = -(heatmap_probs * torch.log(heatmap_probs + 1e-10)).sum(dim=1).mean()
                
                # 总loss
                step_loss = policy_loss + kl_penalty - args.beta_entropy * entropy
                
                total_policy_loss += policy_loss.item()
                total_kl_loss += kl_penalty.item()
                total_entropy_loss += entropy.item()
                num_steps += 1
                
                # 累积梯度
                step_loss.backward()
    
    # 更新参数
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(policy.sam2.get_lora_parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(policy.point_predictor.parameters(), args.grad_clip)
    
    optimizer_sam.step()
    optimizer_point.step()
    optimizer_sam.zero_grad()
    optimizer_point.zero_grad()
    
    # 返回metrics
    metrics = {
        'policy_loss': total_policy_loss / max(num_steps, 1),
        'kl_loss': total_kl_loss / max(num_steps, 1),
        'entropy': total_entropy_loss / max(num_steps, 1),
        'num_steps': num_steps,
    }
    
    return metrics


def main():
    args = parse_args()
    
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
    
    # 1. 加载初始策略checkpoint
    print(f"\nLoading initial policy from {args.init_policy}")
    init_ckpt = torch.load(args.init_policy, map_location='cpu')
    init_config = init_ckpt.get('config', {})
    
    lora_rank = init_config.get('lora_rank', 16)
    lora_alpha = init_config.get('lora_alpha', 32)
    fusion_mode = init_config.get('fusion_mode', 'film')
    feature_scale = init_config.get('feature_scale', 8)
    
    # 2. 初始化策略SAM2 + 点预测器
    print("\nInitializing Policy Network...")
    sam2_policy = LoRASAM2Wrapper(
        sam_checkpoint=args.sam_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=str(device)
    )
    
    point_predictor_policy = PointPredictorFromSAMFeatures(
        sam_feature_dim=256,
        output_size=(args.height, args.width),
        fusion_mode=fusion_mode,
        feature_scale=feature_scale,
    ).to(device)
    
    # 加载权重
    load_checkpoint(
        args.init_policy, point_predictor_policy, sam2_policy,
        None, None, None, str(device)
    )
    
    policy = PolicyNetwork(sam2_policy, point_predictor_policy, stride=args.stride)
    
    # 3. 创建参考策略（深拷贝并冻结）
    print("Creating Reference Policy...")
    sam2_ref = LoRASAM2Wrapper(
        sam_checkpoint=args.sam_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=str(device)
    )
    point_predictor_ref = PointPredictorFromSAMFeatures(
        sam_feature_dim=256,
        output_size=(args.height, args.width),
        fusion_mode=fusion_mode,
        feature_scale=feature_scale,
    ).to(device)
    
    load_checkpoint(
        args.init_policy, point_predictor_ref, sam2_ref,
        None, None, None, str(device)
    )
    
    # 冻结参考策略
    for param in point_predictor_ref.parameters():
        param.requires_grad = False
    sam2_ref.freeze_sam_base()
    
    ref_policy = PolicyNetwork(sam2_ref, point_predictor_ref, stride=args.stride)
    
    # 4. 优化器
    print("\nInitializing Optimizers...")
    optimizer_sam = torch.optim.AdamW(
        sam2_policy.get_lora_parameters(),
        lr=args.lr_sam
    )
    optimizer_point = torch.optim.AdamW(
        point_predictor_policy.parameters(),
        lr=args.lr_point
    )
    
    # 5. 加载训练数据
    print(f"\nLoading training data from {args.train_json}")
    with open(args.train_json, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            train_samples = json.loads(content)
        else:
            train_samples = [json.loads(line) for line in content.splitlines() if line.strip()]
    
    print(f"Loaded {len(train_samples)} training samples")
    
    # 6. 训练循环
    print("\n" + "="*80)
    print("Starting GRPO Training...")
    print("="*80)
    
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs-1}")
        
        # 温度退火
        progress = epoch / max(args.epochs - 1, 1)
        pixel_temp = args.pixel_temp_start + progress * (args.pixel_temp_end - args.pixel_temp_start)
        label_temp = args.label_temp_start + progress * (args.label_temp_end - args.label_temp_start)
        
        print(f"Temperature: pixel={pixel_temp:.3f}, label={label_temp:.3f}")
        
        # 打乱数据
        import random
        random.shuffle(train_samples)
        
        # 批处理
        num_batches = len(train_samples) // args.batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch}"):
            batch_start = batch_idx * args.batch_size
            batch_end = batch_start + args.batch_size
            batch_samples = train_samples[batch_start:batch_end]
            
            # Rollout batch
            batch_trajectories = []
            batch_rewards = []
            
            for sample in batch_samples:
                image_path = sample['image']
                gt_mask_path = sample['gt_mask']
                
                # 加载图像和掩模
                from PIL import Image as PILImage
                image = np.array(PILImage.open(image_path).convert('RGB'))
                gt_mask = np.array(PILImage.open(gt_mask_path).convert('L'))
                gt_mask = (gt_mask > 127).astype(np.uint8) * 255
                
                # Rollout
                trajectory = rollout_single_image(
                    image, gt_mask, policy, args, pixel_temp, label_temp
                )
                
                batch_trajectories.append(trajectory)
                total_reward = sum([step['reward'] for step in trajectory])
                batch_rewards.append(total_reward)
            
            # GRPO更新
            metrics = grpo_update(
                batch_trajectories, policy, ref_policy,
                optimizer_sam, optimizer_point, args,
                pixel_temp, label_temp
            )
            
            # 记录
            if writer is not None:
                writer.add_scalar('train/mean_reward', np.mean(batch_rewards), global_step)
                writer.add_scalar('train/policy_loss', metrics.get('policy_loss', 0), global_step)
                writer.add_scalar('train/kl_loss', metrics.get('kl_loss', 0), global_step)
                writer.add_scalar('train/entropy', metrics.get('entropy', 0), global_step)
                writer.add_scalar('train/pixel_temp', pixel_temp, global_step)
            
            global_step += 1
            
            # 保存checkpoint
            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.out_dir, f'checkpoint_step{global_step:06d}.pt')
                save_checkpoint(
                    ckpt_path, epoch, global_step,
                    point_predictor_policy, sam2_policy,
                    optimizer_sam, optimizer_point, None,
                    config=vars(args),
                    metrics={'mean_reward': np.mean(batch_rewards)}
                )
        
        # Epoch结束，保存
        ckpt_path = os.path.join(args.out_dir, f'checkpoint_epoch{epoch:03d}.pt')
        save_checkpoint(
            ckpt_path, epoch, global_step,
            point_predictor_policy, sam2_policy,
            optimizer_sam, optimizer_point, None,
            config=vars(args),
            metrics={}
        )
    
    if writer is not None:
        writer.close()
    
    print("\n" + "="*80)
    print("GRPO Training Completed!")
    print("="*80)
    print(f"Checkpoints saved in: {args.out_dir}")


if __name__ == "__main__":
    main()

